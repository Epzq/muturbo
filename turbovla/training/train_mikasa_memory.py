#!/usr/bin/env python3
"""muTurboVLA: MIKASA training with muVLA-style recurrent memory.

Same recipe and flags as train_mikasa, plus:
  --memory_tokens M        recurrent memory tokens (default 32)
  --memory_update U        "tbptt" (default) or "ema"
  --tbptt_length K         consecutive steps per training window (default 2). With tbptt, gradients
                           flow through the memory chain within the window; memory is detached at the
                           window boundary. With ema, K is just the window length.
  --memory_ema_alpha A     ema only: M_in(t+1) = A * M_out(t).detach() + (1 - A) * M_in(t).detach()

--batch_size is the number of parallel episode streams per GPU; one optimizer update processes
grad_accum_steps windows of tbptt_length steps each (batch_size * tbptt_length samples per window).
See docs/MU_TURBOVLA.md.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from . import train_mikasa, trainer
from ..data.mikasa_episodic import MikasaEpisodicDataset


def parse_args_with_memory():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--memory_tokens", type=int, default=32)
    parser.add_argument("--memory_update", type=str, default="tbptt", choices=("tbptt", "ema"))
    parser.add_argument("--tbptt_length", type=int, default=2)
    parser.add_argument("--memory_ema_alpha", type=float, default=0.1)
    known, remaining = parser.parse_known_args()
    if known.memory_tokens < 1:
        raise ValueError("--memory_tokens must be >= 1 for recurrent memory training")
    if known.tbptt_length < 1:
        raise ValueError("--tbptt_length must be >= 1")

    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = train_mikasa.parse_args_with_history()
    finally:
        sys.argv = original_argv
    for name, value in vars(known).items():
        setattr(args, name, value)
    print(
        f"mu_memory: memory_tokens={args.memory_tokens}, memory_update={args.memory_update}, "
        f"tbptt_length={args.tbptt_length}, memory_ema_alpha={args.memory_ema_alpha}, "
        f"streams_per_gpu={args.batch_size}",
        flush=True,
    )
    return args


_BUILD_WITHOUT_MEMORY = trainer.build_turbovla


def build_turbovla_with_memory(model_args):
    model_args.memory_tokens = trainer._ACTIVE_TRAIN_ARGS.memory_tokens
    return _BUILD_WITHOUT_MEMORY(model_args)


def make_episodic_dataloader(args, rank, world_size, device):
    step_dataset = trainer.LiberoRLDSDataset(
        dataset_dir=args.dataset_dir,
        LOCAL_DINOV3_PATH=args.dinov3_path,
        rank=rank,
        world_size=world_size,
        chunk_size=args.chunk_size,
        split=args.dataset_split,
        shuffle_buffer=args.shuffle_buffer,
        shuffle_steps_within_episode=False,
        step_mix_buffer_size=0,
        seed=args.seed,
        local_files_only=not args.allow_hf_download,
        expected_image_size=args.expected_image_size,
    )
    dataset = MikasaEpisodicDataset(step_dataset, num_streams=args.batch_size, unroll=args.tbptt_length, seed=args.seed)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )


# Memory carried between windows, per DataLoader worker (each worker owns its own streams).
_CARRIED_MEMORY: dict[int, torch.Tensor] = {}


def run_recurrent_update(model, data_stream, args, device):
    loss_accum = 0.0
    is_ddp = isinstance(model, DDP)
    for accum_index in range(args.grad_accum_steps):
        item = next(data_stream)
        worker_id = int(item["worker_id"])
        unroll, targets = [], []
        for step in item["steps"]:
            samples, instructions, states, gt_actions, action_chunk_masks = step["batch"]
            unroll.append(
                {
                    "instructions": instructions,
                    "samples": trainer.move_samples_to_device(samples, device),
                    "state": states.to(device, non_blocking=True),
                    "reset_mask": step["is_first"].to(device, non_blocking=True),
                }
            )
            targets.append(
                (gt_actions.to(device, non_blocking=True), action_chunk_masks.to(device, non_blocking=True))
            )
        carried = _CARRIED_MEMORY.get(worker_id)  # detached at the window boundary (TBPTT truncation)

        last = accum_index == args.grad_accum_steps - 1
        sync_context = model.no_sync() if (is_ddp and not last) else nullcontext()
        with sync_context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16_amp"):
                predictions, final_memory = model(
                    None,
                    None,
                    None,
                    recurrent_memory=carried,
                    unroll=unroll,
                    memory_update=args.memory_update,
                    memory_ema_alpha=args.memory_ema_alpha,
                )
                losses = []
                for pred_actions, (gt_actions, masks) in zip(predictions, targets):
                    if pred_actions.shape != gt_actions.shape:
                        raise ValueError(f"pred {tuple(pred_actions.shape)} vs gt {tuple(gt_actions.shape)} mismatch")
                    losses.append(trainer.masked_l1_loss(pred_actions, gt_actions, masks))
                loss = torch.stack(losses).mean()
            (loss / args.grad_accum_steps).backward()
        _CARRIED_MEMORY[worker_id] = final_memory.detach()
        loss_accum += loss.detach().item()
    return loss_accum / args.grad_accum_steps


trainer.parse_args = parse_args_with_memory
trainer.build_turbovla = build_turbovla_with_memory
trainer.make_train_dataloader = make_episodic_dataloader
trainer.run_update = run_recurrent_update


def main():
    trainer.train_model()


if __name__ == "__main__":
    main()
