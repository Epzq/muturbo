#!/usr/bin/env python3
"""Mixed-task MIKASA-Robo-VLA trainer; train_mixed recipe plus observation/state history flags.

History flags (see turbovla/data/mikasa_rlds.py for the sample layout):
  --history_size K            past window slots (default 0 = current step only)
  --history_stride S          steps between window slots (default 1)
  --obs_history               append each slot's camera views as extra model views
  --state_history             append each slot's normalized state to the state vector
  --history_first_frame       add episode step 0 as an extra slot
  --history_obs_views         cameras per past slot: both | base (default both)
  --history_valid_bits        ablation: append per-slot valid bits to the state vector
  --history_dropout P         fraction of samples whose history is replaced by the current step
  --history_time_embedding    put image views in time order (oldest -> current) and use a shared
                              camera embedding + learned per-time-position embedding (needs --obs_history)
  --history_tokens            keep the original TurboVLA path for the 2 current views; history views go through
                              DINOv3 without gradients, are average-pooled to --history_pool^2 tokens each,
                              detached, mapped by a separate learned memory projector, and appended to the
                              action decoder memory after the state tokens
  --history_pool N            pooled grid per history view (1 = one token per view; 2 = 2x2 tokens)

Image flag:
  --resize_images_to N        resize camera images (128px in MIKASA RLDS) to N x N with Lanczos before
                              DINOv3, e.g. 256 for a 16x16 patch grid; also sets --expected_image_size N.
                              Evaluation reads it from the checkpoint and applies the same resize.

--state_dim and the model's view count are derived from these flags and the stats file.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import train_mixed, trainer
from ..data.mikasa_rlds import HISTORY_OBS_VIEWS, MikasaHistoryConfig, MikasaHistoryMixin, mikasa_collate_fn
from ..data.suite_stats import _select_stats


def parse_args_with_history():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--history_size", type=int, default=0)
    parser.add_argument("--history_stride", type=int, default=1)
    parser.add_argument("--obs_history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--state_history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history_first_frame", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history_obs_views", type=str, default="both", choices=HISTORY_OBS_VIEWS)
    parser.add_argument("--history_valid_bits", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history_dropout", type=float, default=0.0)
    parser.add_argument("--history_time_embedding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history_tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history_pool", type=int, default=1)
    parser.add_argument("--resize_images_to", type=int, default=0)
    known, remaining = parser.parse_known_args()

    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = train_mixed.parse_args_and_record()
    finally:
        sys.argv = original_argv

    history = MikasaHistoryConfig.from_namespace(known)
    for name, value in history.to_dict().items():
        setattr(args, name, value)
    args.resize_images_to = known.resize_images_to
    if args.resize_images_to:
        args.expected_image_size = args.resize_images_to

    with open(args.stats_path, encoding="utf-8") as f:
        stats = _select_stats(json.load(f), args.stats_key)
    base_state_dim = len(stats["proprio" if "proprio" in stats else "state"]["mean"])
    args.state_dim = history.state_dim(base_state_dim)
    args.num_views = history.num_views()
    print(
        f"mikasa_history={history.to_dict()}, base_state_dim={base_state_dim}, "
        f"state_dim={args.state_dim}, num_views={args.num_views}, "
        f"resize_images_to={args.resize_images_to or 'off'}, image_size={args.expected_image_size}",
        flush=True,
    )
    return args


class MixedTaskStatsMikasaRLDSDataset(MikasaHistoryMixin, train_mixed.MixedSuiteStatsLiberoRLDSDataset):
    def __init__(self, *args, **kwargs):
        active = trainer._ACTIVE_TRAIN_ARGS
        history = MikasaHistoryConfig.from_namespace(active)
        super().__init__(*args, history=history, resize_images_to=active.resize_images_to, **kwargs)


_ORIGINAL_BUILD_TURBOVLA = trainer.build_turbovla


def build_turbovla_with_history_views(model_args):
    active = trainer._ACTIVE_TRAIN_ARGS
    model_args.num_views = active.num_views
    history = MikasaHistoryConfig.from_namespace(active)
    if history.history_time_embedding:
        model_args.view_camera_ids = history.view_camera_ids()
        model_args.view_time_ids = history.view_time_ids()
    if history.history_tokens:
        # Original TurboVLA vision path on the 2 current views; history views become decoder tokens.
        model_args.num_views = 2
        model_args.history_views = history.num_history_views
        model_args.history_pool = history.history_pool
    return _ORIGINAL_BUILD_TURBOVLA(model_args)


trainer.parse_args = parse_args_with_history
trainer.LiberoRLDSDataset = MixedTaskStatsMikasaRLDSDataset
trainer.vla_collate_fn = mikasa_collate_fn
trainer.build_turbovla = build_turbovla_with_history_views


def main():
    trainer.train_model()


if __name__ == "__main__":
    main()
