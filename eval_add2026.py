#!/usr/bin/env python3
"""Evaluate real/fake predictions with a class-balanced joint score.

For sample i, the explanation score e_i is BERTScore-F1 between the
predicted and reference <Evidence> text. A wrong or missing decision receives
zero joint credit:

    J_c = (1 / N_c) * sum_{i:y_i=c} 1[pred_i == y_i] * e_i
    J   = (J_real + J_fake) / 2
    Score = 100 * J

The prediction and reference JSONL files are aligned by audios[0].
"""

import argparse
import json
import os
import re
import sys


DEFAULT_SUBSET_JSONL = "0924_swift_add_test_en.jsonl"

BERT_LANG = "en"
BERT_MODEL_TYPE = None
BERT_RESCALE = False


def audio_key(audios):
    """Extract audios[0] as the sample key."""
    if isinstance(audios, list) and audios:
        return str(audios[0])
    if isinstance(audios, str):
        return audios
    return None


def normalize_label(label):
    """Normalize the ground-truth label."""
    if not isinstance(label, str):
        return ""

    label = label.strip().lower()
    return label if label in {"real", "fake"} else ""


def extract_decision(response):
    """Extract Real/Fake from <Decision>."""
    if not isinstance(response, str):
        return "unknown"

    match = re.search(
        r"<Decision>\s*(real|fake)\s*</Decision>",
        response,
        flags=re.IGNORECASE,
    )

    return match.group(1).lower() if match else "unknown"


def extract_evidence(response):
    """Extract plain text inside <Evidence> and remove XML tags."""
    if not isinstance(response, str):
        return ""

    match = re.search(
        r"<Evidence>(.*?)</Evidence>",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return ""

    text = re.sub(r"<[^>]+>", " ", match.group(1))
    return re.sub(r"\s+", " ", text).strip()


def get_assistant_content(record):
    """Get the final assistant message from a reference record."""
    messages = record.get("messages")

    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
        ):
            content = message.get("content")
            return content if isinstance(content, str) else ""

    return ""


def load_subset(path):
    """Load reference data as audio_path -> label/reference."""
    subset = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = audio_key(record.get("audios"))
            label = normalize_label(record.get("label"))

            if key and label:
                subset[key] = {
                    "label": label,
                    "reference": extract_evidence(
                        get_assistant_content(record)
                    ),
                }

    return subset


def load_predictions(path):
    """Load predictions as audio_path -> response."""
    predictions = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = audio_key(record.get("audios"))
            response = record.get("response")

            if (
                key
                and key not in predictions
                and isinstance(response, str)
            ):
                predictions[key] = response

    return predictions


def count_nonempty_lines(path):
    """Count non-empty lines in a JSONL file."""
    with open(path, "r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def bertscore_per_sample(
    candidates,
    references,
    batch_size,
    device,
):
    """Calculate BERTScore P/R/F1 for every explanation pair."""
    from bert_score import score

    kwargs = {
        "cands": candidates,
        "refs": references,
        "lang": BERT_LANG,
        "batch_size": batch_size,
        "device": device,
        "rescale_with_baseline": BERT_RESCALE,
        "verbose": True,
    }

    if BERT_MODEL_TYPE:
        kwargs["model_type"] = BERT_MODEL_TYPE

    precision, recall, f1 = score(**kwargs)

    return (
        precision.cpu().tolist(),
        recall.cpu().tolist(),
        f1.cpu().tolist(),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate class-balanced accuracy and joint explanation score."
        )
    )

    parser.add_argument(
        "ckpt",
        help="Checkpoint prediction JSONL file.",
    )

    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET_JSONL,
        help="Ground-truth subset JSONL file.",
    )

    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON; default: <checkpoint>.joint_eval.json",
    )

    parser.add_argument(
        "--bert-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--bert-device",
        default="cuda:0",
        help="BERTScore device (default: cuda:0; use cpu for CPU)",
    )

    parser.add_argument(
        "--no-bert",
        action="store_true",
        help="Only calculate classification accuracy.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.ckpt):
        sys.exit(f"[error] checkpoint not found: {args.ckpt}")

    if not os.path.isfile(args.subset):
        sys.exit(f"[error] subset not found: {args.subset}")

    subset = load_subset(args.subset)
    predictions = load_predictions(args.ckpt)

    subset_file_rows = count_nonempty_lines(args.subset)
    prediction_file_rows = count_nonempty_lines(args.ckpt)

    print(
        f"[data] subset rows={subset_file_rows}, "
        f"unique valid audios={len(subset)}; "
        f"prediction rows={prediction_file_rows}, "
        f"unique audios={len(predictions)}"
    )

    rows = []
    bert_indices = []
    candidates = []
    references = []

    missing_prediction = 0
    empty_prediction_evidence = 0
    empty_reference_evidence = 0

    # Iterate over every reference sample.
    # Missing predictions receive zero joint credit.
    for key, item in subset.items():
        response = predictions.get(key, "")
        evidence = extract_evidence(response)
        reference = item["reference"]

        row = {
            "audio": key,
            "label": item["label"],
            "pred": extract_decision(response),
            "bert_p": 0.0,
            "bert_r": 0.0,
            "bert_f1": 0.0,
        }

        if key not in predictions:
            missing_prediction += 1
        elif not evidence:
            empty_prediction_evidence += 1

        if not reference:
            empty_reference_evidence += 1

        if evidence and reference:
            bert_indices.append(len(rows))
            candidates.append(evidence)
            references.append(reference)

        rows.append(row)

    if not rows:
        sys.exit("[error] no valid samples in subset")

    if not args.no_bert and candidates:
        print(
            f"[bert] scoring {len(candidates)}/{len(rows)} "
            f"evidence pairs "
            f"(batch_size={args.bert_batch_size}, "
            f"device={args.bert_device})"
        )

        if len(candidates) != len(rows):
            print(
                "[bert] skipped: "
                f"missing_prediction={missing_prediction}, "
                f"empty_prediction_evidence="
                f"{empty_prediction_evidence}, "
                f"empty_reference_evidence="
                f"{empty_reference_evidence}"
            )

        p_values, r_values, f1_values = bertscore_per_sample(
            candidates=candidates,
            references=references,
            batch_size=args.bert_batch_size,
            device=args.bert_device,
        )

        for index, p, r, f1 in zip(
            bert_indices,
            p_values,
            r_values,
            f1_values,
        ):
            rows[index]["bert_p"] = max(
                0.0,
                min(1.0, float(p)),
            )
            rows[index]["bert_r"] = max(
                0.0,
                min(1.0, float(r)),
            )
            rows[index]["bert_f1"] = max(
                0.0,
                min(1.0, float(f1)),
            )

    result = {
        "checkpoint": args.ckpt,
        "subset": args.subset,
        "subset_file_rows": subset_file_rows,
        "subset_total": len(rows),
        "prediction_file_rows": prediction_file_rows,
        "unique_predictions": len(predictions),
        "matched_predictions": sum(
            1 for key in subset if key in predictions
        ),
        "missing_predictions": missing_prediction,
        "empty_prediction_evidence": empty_prediction_evidence,
        "empty_reference_evidence": empty_reference_evidence,
        "bertscore_pairs": len(candidates),
    }

    class_results = {}

    for label in ("real", "fake"):
        class_rows = [
            row for row in rows
            if row["label"] == label
        ]

        total = len(class_rows)

        correct_rows = [
            row for row in class_rows
            if row["pred"] == label
        ]

        correct = len(correct_rows)

        recall = (
            correct / total
            if total
            else 0.0
        )

        mean_explanation_all = (
            sum(row["bert_f1"] for row in class_rows) / total
            if total
            else 0.0
        )

        mean_explanation_correct = (
            sum(row["bert_f1"] for row in correct_rows) / correct
            if correct
            else 0.0
        )

        joint = (
            sum(row["bert_f1"] for row in correct_rows) / total
            if total
            else 0.0
        )

        class_results[label] = {
            "total": total,
            "correct": correct,
            "recall": recall,
            "mean_bertscore_f1_all": mean_explanation_all,
            "mean_bertscore_f1_correct": (
                mean_explanation_correct
            ),
            "joint": joint,
        }

    real = class_results["real"]
    fake = class_results["fake"]

    overall_correct = (
        real["correct"] + fake["correct"]
    )

    balanced_accuracy = (
        real["recall"] + fake["recall"]
    ) / 2

    joint_score = (
        real["joint"] + fake["joint"]
    ) / 2

    mean_bertscore_f1 = (
        (
            real["mean_bertscore_f1_all"] * real["total"]
            + fake["mean_bertscore_f1_all"] * fake["total"]
        )
        / len(rows)
        if not args.no_bert
        else None
    )

    result.update(
        {
            "real": real,
            "fake": fake,
            "overall_accuracy": (
                overall_correct / len(rows)
            ),
            "balanced_accuracy": balanced_accuracy,
            "mean_bertscore_f1": mean_bertscore_f1,
            "joint_score": (
                None if args.no_bert else joint_score
            ),
            "final_score_100": (
                None
                if args.no_bert
                else 100 * joint_score
            ),
            "formula": (
                "0.5 * "
                "(mean_real[correct * bert_f1] + "
                "mean_fake[correct * bert_f1])"
            ),
        }
    )

    print(
        f"[acc] real={real['recall']:.4f} "
        f"({real['correct']}/{real['total']}), "
        f"fake={fake['recall']:.4f} "
        f"({fake['correct']}/{fake['total']}), "
        f"balanced={balanced_accuracy:.4f}"
    )

    if args.no_bert:
        print("[bert] skipped (--no-bert)")
        print("[joint] skipped (--no-bert)")
    else:
        print(
            f"[bert] real={real['mean_bertscore_f1_all']:.6f}, "
            f"fake={fake['mean_bertscore_f1_all']:.6f}, "
            f"mean={mean_bertscore_f1:.6f}, "
            f"mean_100={100 * mean_bertscore_f1:.4f}"
        )
        print(
            f"[joint] real={real['joint']:.4f}, "
            f"fake={fake['joint']:.4f}, "
            f"J={joint_score:.4f}, "
            f"Score={100 * joint_score:.2f}"
        )

    out_json = (
        args.out_json
        if args.out_json
        else args.ckpt + ".joint_eval.json"
    )

    with open(out_json, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
