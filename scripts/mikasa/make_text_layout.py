#!/usr/bin/env python3
"""Build a MIKASA online text layout: each instruction's exact BERT token length.

Instructions come from MIKASA-Robo's task manifest (mikasa_robo_vla_envs.csv). With
--dataset_root, every task's RLDS language_instruction is checked against the manifest,
because layout entries only apply to exact string matches. Templated instructions
(e.g. Rotate*: "Rotate the peg by {angle_deg} degrees ...") vary per episode, so they are
left out of the layout and fall back to output_padding_length; the dataset check matches
them by pattern.

Train with:
  --text_layout_path experiments/mikasa/configs/online_text_layout.json \
  --text_padding_length <output_padding_length>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default="MIKASA-Robo/mikasa_robo_vla_envs.csv")
    parser.add_argument("--bert_path", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--output", type=str, default="experiments/mikasa/configs/online_text_layout.json")
    parser.add_argument("--dataset_root", type=str, default=None, help="RLDS root to cross-check instructions.")
    parser.add_argument("--output_padding_length", type=int, default=None, help="Default: longest instruction.")
    return parser.parse_args()


def _is_template(instruction: str) -> bool:
    return "{" in instruction


def _template_pattern(instruction: str) -> re.Pattern:
    parts = re.split(r"\{[^}]*\}", instruction)
    return re.compile("^" + "(.+?)".join(re.escape(part) for part in parts) + "$")


def _check_dataset(rows: list[dict], dataset_root: str) -> None:
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    # Each RLDS task directory records its env ID in metadata.json.
    task_dirs = {
        json.loads(path.read_text(encoding="utf-8"))["env_id"]: path.parent
        for path in Path(dataset_root).glob("*/1.0.0/metadata.json")
    }
    skip = {"image": tfds.decode.SkipDecoding(), "wrist_image": tfds.decode.SkipDecoding()}
    mismatches, missing = [], []
    for row in rows:
        task_dir = task_dirs.get(row["Name"])
        if task_dir is None:
            missing.append(row["Name"])
            continue
        builder = tfds.builder_from_directory(str(task_dir))
        dataset = builder.as_dataset(split="train", decoders={"steps": {"observation": skip}}).take(1)
        episode = next(iter(tfds.as_numpy(dataset)))
        instruction = next(iter(episode["steps"]))["language_instruction"].decode("utf-8")
        expected = row["language_instruction"]
        matches = (
            _template_pattern(expected).match(instruction) is not None
            if _is_template(expected)
            else instruction == expected
        )
        if not matches:
            mismatches.append((row["Name"], instruction, row["language_instruction"]))
    print(f"dataset check: {len(rows) - len(missing)} tasks checked, {len(mismatches)} mismatches, {len(missing)} missing")
    for name, got, expected in mismatches:
        print(f"  MISMATCH {name}: dataset={got!r} manifest={expected!r}")
    if mismatches:
        raise SystemExit("dataset instructions differ from the manifest; layout entries would not apply")


def main() -> None:
    args = parse_args()
    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    if args.dataset_root:
        _check_dataset(rows, args.dataset_root)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    lengths = {}
    templates = sorted({row["language_instruction"] for row in rows if _is_template(row["language_instruction"])})
    for row in rows:
        instruction = row["language_instruction"]
        if not _is_template(instruction):
            lengths[instruction] = len(tokenizer(instruction)["input_ids"])
    longest = max(lengths.values())
    output_length = args.output_padding_length or longest
    if output_length < longest:
        raise SystemExit(f"output_padding_length={output_length} is shorter than the longest instruction ({longest})")

    layout = {
        "output_padding_length": output_length,
        "padding_length_by_instruction": dict(sorted(lengths.items())),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(layout, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}: {len(lengths)} instructions from {len(rows)} tasks, "
          f"lengths {min(lengths.values())}-{longest}, output_padding_length={output_length}")
    for template in templates:
        # A filled-in template gets output_padding_length; make sure a long fill still fits.
        worst = len(tokenizer(re.sub(r"\{[^}]*\}", "-179", template))["input_ids"])
        print(f"  template (not in layout, ~{worst} tokens when filled): {template}")
        if worst > output_length:
            raise SystemExit(f"filled template would exceed output_padding_length={output_length}")


if __name__ == "__main__":
    main()
