#!/usr/bin/env python3
"""Create deterministic RawBoost Algo-5 WAVs and a doubled MOSS manifest.

The signal-processing operations are adapted from the MIT-licensed official
RawBoost implementation:
https://github.com/TakHemlata/RawBoost-antispoofing

Algo 5 applies linear/non-linear convolutive noise followed by impulsive
signal-dependent additive noise. No external audio data is used.
"""

import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


RAWBOOST_PARAMETERS = {
    "algorithm": 5,
    "n_bands": 5,
    "min_frequency_hz": 20,
    "max_frequency_hz": 8000,
    "min_bandwidth_hz": 100,
    "max_bandwidth_hz": 1000,
    "min_filter_coefficients": 10,
    "max_filter_coefficients": 100,
    "min_gain_db": 0,
    "max_gain_db": 0,
    "min_nonlinear_bias_db": 5,
    "max_nonlinear_bias_db": 20,
    "nonlinearity_order": 5,
    "max_impulsive_samples_percent": 10,
    "impulsive_gain": 2,
}


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_rng(seed, file_id):
    digest = hashlib.sha256(f"{seed}:{file_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def normalize_waveform(waveform, always=False):
    peak = float(np.max(np.abs(waveform)))
    if peak == 0:
        return waveform
    if always or peak > 1:
        return waveform / peak
    return waveform


def random_range(rng, lower, upper, integer=False):
    # The official implementation uses legacy np.random.uniform, which also
    # accepts the reversed gain interval produced after applying nonlinear
    # bias. Generator.uniform rejects that interval, so sample the same affine
    # distribution explicitly.
    value = lower + (upper - lower) * rng.random()
    return int(value) if integer else float(value)


def notch_coefficients(rng, sample_rate, min_gain, max_gain):
    parameters = RAWBOOST_PARAMETERS
    coefficients = np.asarray([1.0], dtype=np.float64)
    nyquist = sample_rate / 2
    for _ in range(parameters["n_bands"]):
        center = random_range(
            rng,
            parameters["min_frequency_hz"],
            min(parameters["max_frequency_hz"], nyquist),
        )
        bandwidth = random_range(
            rng,
            parameters["min_bandwidth_hz"],
            parameters["max_bandwidth_hz"],
        )
        count = random_range(
            rng,
            parameters["min_filter_coefficients"],
            parameters["max_filter_coefficients"],
            integer=True,
        )
        if count % 2 == 0:
            count += 1
        lower = max(center - bandwidth / 2, 1e-3)
        upper = min(center + bandwidth / 2, nyquist - 1e-3)
        if lower >= upper:
            lower = max(1e-3, min(center, nyquist - 2e-3))
            upper = min(nyquist - 1e-3, lower + 1e-3)
        band_stop = signal.firwin(
            count,
            [lower, upper],
            window="hamming",
            fs=sample_rate,
        )
        coefficients = np.convolve(coefficients, band_stop)
    gain = random_range(rng, min_gain, max_gain)
    _, response = signal.freqz(coefficients, 1, fs=sample_rate)
    peak = float(np.max(np.abs(response)))
    if peak == 0:
        raise RuntimeError("RawBoost generated a zero-response FIR filter")
    return (10 ** (gain / 20)) * coefficients / peak


def filter_fir(waveform, coefficients):
    padding = coefficients.shape[0] + 1
    padded = np.pad(waveform, (0, padding), mode="constant")
    filtered = signal.lfilter(coefficients, 1, padded)
    half = padding // 2
    return filtered[half : filtered.shape[0] - half]


def convolutive_noise(waveform, sample_rate, rng):
    parameters = RAWBOOST_PARAMETERS
    output = np.zeros_like(waveform, dtype=np.float64)
    min_gain = parameters["min_gain_db"]
    max_gain = parameters["max_gain_db"]
    for order in range(parameters["nonlinearity_order"]):
        if order == 1:
            min_gain -= parameters["min_nonlinear_bias_db"]
            max_gain -= parameters["max_nonlinear_bias_db"]
        coefficients = notch_coefficients(
            rng, sample_rate, min_gain, max_gain
        )
        output += filter_fir(np.power(waveform, order + 1), coefficients)
    output -= np.mean(output)
    return normalize_waveform(output)


def impulsive_noise(waveform, rng):
    parameters = RAWBOOST_PARAMETERS
    percentage = random_range(
        rng, 0, parameters["max_impulsive_samples_percent"]
    )
    count = int(waveform.shape[0] * percentage / 100)
    if count == 0:
        return waveform.copy()
    positions = rng.permutation(waveform.shape[0])[:count]
    random_gain = (
        (2 * rng.random(positions.shape[0]) - 1)
        * (2 * rng.random(positions.shape[0]) - 1)
    )
    output = waveform.copy()
    output[positions] += (
        parameters["impulsive_gain"]
        * waveform[positions]
        * random_gain
    )
    return normalize_waveform(output)


def rawboost_algo5(waveform, sample_rate, rng):
    output = convolutive_noise(waveform, sample_rate, rng)
    output = impulsive_noise(output, rng)
    if output.shape != waveform.shape or not np.isfinite(output).all():
        raise RuntimeError("RawBoost produced an invalid waveform")
    return output


def audio_path(record):
    matches = [
        message
        for message in record.get("conversation", [])
        if message.get("role") == "user"
        and message.get("message_type") == "audio"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one user audio message for {record.get('id')}"
        )
    return Path(matches[0]["content"]), matches[0]


def valid_cached_audio(path, source_info):
    if not path.is_file():
        return False
    try:
        cached = sf.info(path)
    except RuntimeError:
        return False
    return (
        cached.samplerate == source_info.samplerate
        and cached.channels == 1
        and cached.frames == source_info.frames
        and cached.subtype == "PCM_16"
    )


def format_eta(seconds):
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-audio-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    source_rows = read_jsonl(args.input_manifest)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise ValueError("Input manifest is empty")

    source_ids = [row.get("id") for row in source_rows]
    if None in source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("Input manifest contains missing or duplicate IDs")
    label_counts = Counter(row.get("label") for row in source_rows)
    if set(label_counts) - {"real", "fake"}:
        raise ValueError(f"Invalid labels: {dict(label_counts)}")

    args.output_audio_dir.mkdir(parents=True, exist_ok=True)
    augmented_rows = []
    generated = 0
    reused = 0
    started = time.monotonic()
    last_report = started

    for index, record in enumerate(source_rows, start=1):
        source_path, _ = audio_path(record)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_info = sf.info(source_path)
        if (
            source_info.samplerate != 16000
            or source_info.channels != 1
            or source_info.subtype != "PCM_16"
        ):
            raise ValueError(
                f"Expected 16 kHz mono PCM-16 input: {source_path}"
            )

        output_path = (
            args.output_audio_dir / f"{record['id']}__rawboost5.wav"
        ).resolve()
        if valid_cached_audio(output_path, source_info):
            reused += 1
        else:
            waveform, sample_rate = sf.read(
                source_path, dtype="float64", always_2d=False
            )
            augmented = rawboost_algo5(
                waveform,
                sample_rate,
                stable_rng(args.seed, record["id"]),
            )
            if augmented.shape[0] != source_info.frames:
                raise RuntimeError(
                    f"Length changed for {record['id']}: "
                    f"{source_info.frames}->{augmented.shape[0]}"
                )
            sf.write(
                output_path,
                augmented,
                sample_rate,
                subtype="PCM_16",
                format="WAV",
            )
            if not valid_cached_audio(output_path, source_info):
                raise RuntimeError(f"Invalid generated WAV: {output_path}")
            generated += 1

        augmented_record = copy.deepcopy(record)
        augmented_record["id"] = f"{record['id']}__rawboost5"
        _, augmented_audio_message = audio_path(augmented_record)
        augmented_audio_message["content"] = output_path.as_posix()
        augmented_record["augmentation"] = {
            "name": "rawboost",
            "algorithm": 5,
            "seed": args.seed,
            "source_id": record["id"],
        }
        augmented_rows.append(augmented_record)

        now = time.monotonic()
        if now - last_report >= 5 or index == len(source_rows):
            elapsed = now - started
            rate = index / elapsed if elapsed > 0 else 0
            eta = (len(source_rows) - index) / rate if rate > 0 else math.inf
            print(
                f"[rawboost] {index}/{len(source_rows)} "
                f"({100 * index / len(source_rows):.1f}%) "
                f"generated={generated} reused={reused} "
                f"rate={rate:.2f} files/s ETA={format_eta(eta)}",
                flush=True,
            )
            last_report = now

    combined = source_rows + augmented_rows
    combined_ids = [row["id"] for row in combined]
    if len(combined_ids) != len(set(combined_ids)):
        raise RuntimeError("Output manifest IDs are not unique")
    output_labels = Counter(row["label"] for row in combined)
    expected_labels = Counter(
        {label: count * 2 for label, count in label_counts.items()}
    )
    if output_labels != expected_labels:
        raise RuntimeError(
            f"Label counts changed: {output_labels} != {expected_labels}"
        )

    write_jsonl(args.output_manifest, combined)
    summary_path = args.summary_json or args.output_manifest.with_suffix(
        ".summary.json"
    )
    summary = {
        "input_manifest": args.input_manifest.resolve().as_posix(),
        "output_manifest": args.output_manifest.resolve().as_posix(),
        "output_audio_dir": args.output_audio_dir.resolve().as_posix(),
        "seed": args.seed,
        "parameters": RAWBOOST_PARAMETERS,
        "source_samples": len(source_rows),
        "augmented_samples": len(augmented_rows),
        "combined_samples": len(combined),
        "source_label_counts": dict(label_counts),
        "combined_label_counts": dict(output_labels),
        "generated_files": generated,
        "reused_files": reused,
        "elapsed_seconds": time.monotonic() - started,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {args.output_manifest}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
