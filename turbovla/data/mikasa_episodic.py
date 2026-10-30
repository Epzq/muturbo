"""Episodic (stream) data loading for muTurboVLA recurrent memory training.

The standard MIKASA loader shuffles steps across episodes, which breaks recurrence. Here each
DataLoader worker owns ``num_streams`` independent streams (one per batch slot). Every stream walks a
single episode step by step and, when it ends, starts a new randomly drawn episode (from any task).
This mirrors muVLA's episodic dataloader ("B independent streams ... with a per-slot is_first flag").

Each yielded item is a *window* of ``unroll`` consecutive batches for the same streams, so truncated
backprop-through-time (TBPTT) always unrolls over contiguous steps of the same episodes:

    {"worker_id": int,
     "steps": [ {"batch": (samples, instructions, states, actions, masks),   # mikasa_collate_fn output
                 "is_first": BoolTensor[B],          # slot started a new episode at this step
                 "episode_step": LongTensor[B]},     # step index within the episode (debug/tests)
                ... unroll times ... ]}

Per-step samples are built by the regular MIKASA step dataset (``_build_step_sample``), so images,
state normalization, action chunking and padding are identical to non-recurrent training.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .mikasa_rlds import mikasa_collate_fn

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


class MikasaEpisodicDataset(IterableDataset):
    def __init__(self, step_dataset, num_streams: int, unroll: int, seed: int = 42):
        """``step_dataset``: a MIKASA step dataset (provides dataset_dirs, split, rank/world_size,
        shuffle_buffer and ``_build_step_sample``)."""
        if num_streams < 1 or unroll < 1:
            raise ValueError("num_streams and unroll must be >= 1")
        history = getattr(step_dataset, "history", None)
        if history is not None and history.enabled:
            raise ValueError("recurrent memory training does not support history slots (obs/state history)")
        self.step_dataset = step_dataset
        self.num_streams = int(num_streams)
        self.unroll = int(unroll)
        self.seed = int(seed)

    def _episode_iterator(self, worker_id: int, num_workers: int):
        ds = self.step_dataset
        dataset = None
        for path in ds.dataset_dirs:
            current = tfds.builder_from_directory(builder_dir=path).as_dataset(split=ds.split)
            if ds.world_size > 1:
                current = current.shard(num_shards=ds.world_size, index=ds.rank)
            if num_workers > 1:
                current = current.shard(num_shards=num_workers, index=worker_id)
            dataset = current if dataset is None else dataset.concatenate(current)
        seed = self.seed + 1009 * ds.rank + 9176 * worker_id
        dataset = dataset.shuffle(buffer_size=ds.shuffle_buffer, seed=seed, reshuffle_each_iteration=True).repeat()
        for episode in tfds.as_numpy(dataset):
            yield list(episode["steps"])

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        episodes = self._episode_iterator(worker_id, num_workers)
        streams = [{"steps": next(episodes), "t": 0} for _ in range(self.num_streams)]
        while True:
            window = []
            for _ in range(self.unroll):
                samples, is_first, episode_step = [], [], []
                for stream in streams:
                    if stream["t"] >= len(stream["steps"]):
                        stream["steps"], stream["t"] = next(episodes), 0
                    steps, t = stream["steps"], stream["t"]
                    samples.append(self.step_dataset._build_step_sample(steps, t, len(steps)))
                    is_first.append(t == 0)
                    episode_step.append(t)
                    stream["t"] += 1
                window.append(
                    {
                        "batch": mikasa_collate_fn(samples),
                        "is_first": torch.tensor(is_first, dtype=torch.bool),
                        "episode_step": torch.tensor(episode_step, dtype=torch.long),
                    }
                )
            yield {"worker_id": worker_id, "steps": window}


__all__ = ["MikasaEpisodicDataset"]
