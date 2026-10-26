#!/usr/bin/env python3
"""Convert ADD 2026 Track 2 annotations to MOSS-Audio SFT JSONL."""

import argparse
import json
import re
from pathlib import Path


SYSTEM = (
    "You are a senior audio forensics scientist specializing in speech "
    "deepfake detection."
)
USER = (
    "Analyze this audio for authenticity using observable acoustic evidence "
    "from the audio, and produce a structured XML forensic report covering "
    "spectral, prosodic, pronunciation, noise, and other acoustic anomalies, "
    "with a final decision of Real or Fake."
)
PROMPT = f"{SYSTEM}\n\n{USER}"


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def convert_split(data_root, split, limit=None):
    label_path = data_root / "labels" / f"{split}_labels.jsonl"
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    annotations = read_jsonl(label_path)
    if limit is not None:
        annotations = annotations[:limit]

    file_ids = [row.get("file_id") for row in annotations]
    if len(file_ids) != len(set(file_ids)):
        raise ValueError(f"{split}: duplicate file_id values")

    records = []
    label_counts = {"real": 0, "fake": 0}
    for row in annotations:
        file_id = row.get("file_id")
        label = row.get("label")
        answer = row.get("explanation")
        audio_path = (data_root / split / f"{file_id}.wav").resolve()

        if not file_id or not isinstance(answer, str):
            raise ValueError(f"{split}: invalid annotation row: {row}")
        if label not in label_counts:
            raise ValueError(f"{split}: invalid label for {file_id}: {label}")
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        decisions = re.findall(
            r"<Decision>\s*(real|fake)\s*</Decision>", answer, re.I
        )
        evidence = re.findall(
            r"<Evidence>(.*?)</Evidence>", answer, re.I | re.S
        )
        if (
            len(decisions) != 1
            or decisions[0].lower() != label
            or len(evidence) != 1
            or not evidence[0].strip()
        ):
            raise ValueError(f"{split}: invalid XML reference for {file_id}")

        records.append(
            {
                "id": file_id,
                "label": label,
                "conversation": [
                    {
                        "role": "user",
                        "message_type": "audio",
                        "content": audio_path.as_posix(),
                    },
                    {
                        "role": "user",
                        "message_type": "text",
                        "content": PROMPT,
                    },
                    {
                        "role": "assistant",
                        "message_type": "text",
                        "content": answer,
                    },
                ],
            }
        )
        label_counts[label] += 1

    return records, label_counts


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def balanced_overfit_records(records, per_class):
    by_label = {
        label: sorted(
            (record for record in records if record["label"] == label),
            key=lambda record: record["id"],
        )[:per_class]
        for label in ("real", "fake")
    }
    for label, rows in by_label.items():
        if len(rows) != per_class:
            raise ValueError(
                f"Need {per_class} {label} samples, found {len(rows)}"
            )

    selected = []
    for index in range(per_class):
        selected.extend((by_label["real"][index], by_label["fake"][index]))
    return selected


def evaluation_subset(records, split):
    subset = []
    for record in records:
        answer = record["conversation"][-1]["content"]
        subset.append(
            {
                "id": record["id"],
                "label": record["label"],
                "audios": [f"data/{split}/{record['id']}.wav"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/moss"))
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--overfit-per-class", type=int, default=4)
    args = parser.parse_args()

    if args.train_limit is not None and args.train_limit <= 0:
        parser.error("--train-limit must be positive")
    if args.dev_limit is not None and args.dev_limit <= 0:
        parser.error("--dev-limit must be positive")
    if args.overfit_per_class <= 0:
        parser.error("--overfit-per-class must be positive")

    data_root = args.data_root.resolve()
    split_records = {}
    for split, limit in (
        ("train", args.train_limit),
        ("dev", args.dev_limit),
    ):
        records, counts = convert_split(data_root, split, limit)
        split_records[split] = records
        output_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(output_path, records)
        print(
            f"[saved] {output_path}: {len(records)} samples "
            f"(real={counts['real']}, fake={counts['fake']})"
        )

    overfit = balanced_overfit_records(
        split_records["train"], args.overfit_per_class
    )
    total = len(overfit)
    overfit_path = args.output_dir / f"overfit_{total}.jsonl"
    subset_path = args.output_dir / f"overfit_{total}_subset.jsonl"
    write_jsonl(overfit_path, overfit)
    write_jsonl(subset_path, evaluation_subset(overfit, "train"))
    print(
        f"[saved] {overfit_path}: {total} fixed samples "
        f"(real={args.overfit_per_class}, fake={args.overfit_per_class})"
    )
    print(f"[saved] {subset_path}: {total} evaluation references")


if __name__ == "__main__":
    main()
