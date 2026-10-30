#!/usr/bin/env python3
"""Compute MIKASA-Robo-VLA proprio/action statistics across multiple TFDS tasks.

Writes the same payload layout as scripts/libero/compute_mixed_stats.py, so the
output works with --stats_path/--stats_key. MIKASA stores proprioception as
observation/proprio; images are left encoded since only proprio/action are needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libero"))
from compute_mixed_stats import _finalize, _update_accum  # noqa: E402

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", action="append", required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--stats_key", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output} (use --overwrite to replace it)")

    state_accum = None
    action_accum = None
    task_summaries = []
    total_episodes = 0

    for dataset_dir in args.dataset_dir:
        builder = tfds.builder_from_directory(builder_dir=dataset_dir)
        decoders = {
            "steps": {
                "observation": {
                    "image": tfds.decode.SkipDecoding(),
                    "wrist_image": tfds.decode.SkipDecoding(),
                }
            }
        }
        dataset = builder.as_dataset(split=args.split, decoders=decoders)
        episode_lengths: list[int] = []

        for episode in tfds.as_numpy(dataset):
            states = []
            actions = []
            for step in episode["steps"]:
                states.append(np.asarray(step["observation"]["proprio"], dtype=np.float64).reshape(-1))
                actions.append(np.asarray(step["action"], dtype=np.float64).reshape(-1))
            if not actions:
                continue
            state_arr = np.stack(states, axis=0)
            action_arr = np.stack(actions, axis=0)
            state_accum = _update_accum(state_accum, state_arr)
            action_accum = _update_accum(action_accum, action_arr)
            episode_lengths.append(int(action_arr.shape[0]))
            total_episodes += 1
            if args.log_every > 0 and total_episodes % args.log_every == 0:
                print(f"processed total_episodes={total_episodes} actions={int(action_accum['count'])}", flush=True)

        if not episode_lengths:
            raise RuntimeError(f"no episodes found in dataset={dataset_dir} split={args.split}")
        task_summaries.append(
            {
                "dataset_dir": os.path.abspath(dataset_dir),
                "num_episodes": int(len(episode_lengths)),
                "num_actions": int(sum(episode_lengths)),
                "episode_length_min": int(min(episode_lengths)),
                "episode_length_max": int(max(episode_lengths)),
                "episode_length_mean": float(np.mean(episode_lengths)),
            }
        )

    if state_accum is None or action_accum is None:
        raise RuntimeError("no samples found")

    payload = {
        args.stats_key: {
            "state": _finalize(state_accum),
            "proprio": _finalize(state_accum),
            "action": _finalize(action_accum),
        },
        "metadata": {
            "dataset_dirs": [os.path.abspath(path) for path in args.dataset_dir],
            "split": args.split,
            "stats_key": args.stats_key,
            "num_actions": int(action_accum["count"]),
            "num_states": int(state_accum["count"]),
            "num_episodes": int(total_episodes),
            "task_summaries": task_summaries,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, output)
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
