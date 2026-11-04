#!/usr/bin/env python3
"""Add all 88 openSMILE eGeMAPSv02 functionals to MOSS prompts."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import opensmile


FEATURE_LABELS = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": "Average Fundamental Frequency in Semitones from 27.5 Hz",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "Normalized Standard Deviation of Fundamental Frequency",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0": "20th Percentile of Fundamental Frequency in Semitones from 27.5 Hz",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0": "Median Fundamental Frequency in Semitones from 27.5 Hz",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0": "80th Percentile of Fundamental Frequency in Semitones from 27.5 Hz",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": "20th-to-80th Percentile Range of Fundamental Frequency",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope": "Average Rising Slope of Fundamental Frequency",
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope": "Standard Deviation of Rising Slopes of Fundamental Frequency",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope": "Average Falling Slope of Fundamental Frequency",
    "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope": "Standard Deviation of Falling Slopes of Fundamental Frequency",
    "loudness_sma3_amean": "Average Loudness",
    "loudness_sma3_stddevNorm": "Normalized Standard Deviation of Loudness",
    "loudness_sma3_percentile20.0": "20th Percentile of Loudness",
    "loudness_sma3_percentile50.0": "Median Loudness",
    "loudness_sma3_percentile80.0": "80th Percentile of Loudness",
    "loudness_sma3_pctlrange0-2": "20th-to-80th Percentile Range of Loudness",
    "loudness_sma3_meanRisingSlope": "Average Rising Slope of Loudness",
    "loudness_sma3_stddevRisingSlope": "Standard Deviation of Rising Slopes of Loudness",
    "loudness_sma3_meanFallingSlope": "Average Falling Slope of Loudness",
    "loudness_sma3_stddevFallingSlope": "Standard Deviation of Falling Slopes of Loudness",
    "spectralFlux_sma3_amean": "Average Spectral Flux",
    "spectralFlux_sma3_stddevNorm": "Normalized Standard Deviation of Spectral Flux",
    "mfcc1_sma3_amean": "Average First Mel-Frequency Cepstral Coefficient",
    "mfcc1_sma3_stddevNorm": "Normalized Standard Deviation of the First Mel-Frequency Cepstral Coefficient",
    "mfcc2_sma3_amean": "Average Second Mel-Frequency Cepstral Coefficient",
    "mfcc2_sma3_stddevNorm": "Normalized Standard Deviation of the Second Mel-Frequency Cepstral Coefficient",
    "mfcc3_sma3_amean": "Average Third Mel-Frequency Cepstral Coefficient",
    "mfcc3_sma3_stddevNorm": "Normalized Standard Deviation of the Third Mel-Frequency Cepstral Coefficient",
    "mfcc4_sma3_amean": "Average Fourth Mel-Frequency Cepstral Coefficient",
    "mfcc4_sma3_stddevNorm": "Normalized Standard Deviation of the Fourth Mel-Frequency Cepstral Coefficient",
    "jitterLocal_sma3nz_amean": "Average Local Jitter",
    "jitterLocal_sma3nz_stddevNorm": "Normalized Standard Deviation of Local Jitter",
    "shimmerLocaldB_sma3nz_amean": "Average Local Shimmer (dB)",
    "shimmerLocaldB_sma3nz_stddevNorm": "Normalized Standard Deviation of Local Shimmer",
    "HNRdBACF_sma3nz_amean": "Average Harmonics-to-Noise Ratio (dB)",
    "HNRdBACF_sma3nz_stddevNorm": "Normalized Standard Deviation of the Harmonics-to-Noise Ratio",
    "logRelF0-H1-H2_sma3nz_amean": "Average Logarithmic Relative F0-to-H1-H2 Measure",
    "logRelF0-H1-H2_sma3nz_stddevNorm": "Normalized Standard Deviation of the Logarithmic Relative F0-to-H1-H2 Measure",
    "logRelF0-H1-A3_sma3nz_amean": "Average Logarithmic Relative F0-to-H1-A3 Measure",
    "logRelF0-H1-A3_sma3nz_stddevNorm": "Normalized Standard Deviation of the Logarithmic Relative F0-to-H1-A3 Measure",
    "F1frequency_sma3nz_amean": "Average First Formant Frequency (Hz)",
    "F1frequency_sma3nz_stddevNorm": "Normalized Standard Deviation of the First Formant Frequency",
    "F1bandwidth_sma3nz_amean": "Average First Formant Bandwidth (Hz)",
    "F1bandwidth_sma3nz_stddevNorm": "Normalized Standard Deviation of the First Formant Bandwidth",
    "F1amplitudeLogRelF0_sma3nz_amean": "Average First Formant Amplitude Relative to F0 (dB)",
    "F1amplitudeLogRelF0_sma3nz_stddevNorm": "Normalized Standard Deviation of the First Formant Amplitude Relative to F0",
    "F2frequency_sma3nz_amean": "Average Second Formant Frequency (Hz)",
    "F2frequency_sma3nz_stddevNorm": "Normalized Standard Deviation of the Second Formant Frequency",
    "F2bandwidth_sma3nz_amean": "Average Second Formant Bandwidth (Hz)",
    "F2bandwidth_sma3nz_stddevNorm": "Normalized Standard Deviation of the Second Formant Bandwidth",
    "F2amplitudeLogRelF0_sma3nz_amean": "Average Second Formant Amplitude Relative to F0 (dB)",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm": "Normalized Standard Deviation of the Second Formant Amplitude Relative to F0",
    "F3frequency_sma3nz_amean": "Average Third Formant Frequency (Hz)",
    "F3frequency_sma3nz_stddevNorm": "Normalized Standard Deviation of the Third Formant Frequency",
    "F3bandwidth_sma3nz_amean": "Average Third Formant Bandwidth (Hz)",
    "F3bandwidth_sma3nz_stddevNorm": "Normalized Standard Deviation of the Third Formant Bandwidth",
    "F3amplitudeLogRelF0_sma3nz_amean": "Average Third Formant Amplitude Relative to F0 (dB)",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": "Normalized Standard Deviation of the Third Formant Amplitude Relative to F0",
    "alphaRatioV_sma3nz_amean": "Average Alpha Ratio During Voiced Segments (dB)",
    "alphaRatioV_sma3nz_stddevNorm": "Normalized Standard Deviation of the Alpha Ratio During Voiced Segments",
    "hammarbergIndexV_sma3nz_amean": "Average Hammarberg Index During Voiced Segments (dB)",
    "hammarbergIndexV_sma3nz_stddevNorm": "Normalized Standard Deviation of the Hammarberg Index During Voiced Segments",
    "slopeV0-500_sma3nz_amean": "Average Spectral Slope from 0 to 500 Hz During Voiced Segments",
    "slopeV0-500_sma3nz_stddevNorm": "Normalized Standard Deviation of the 0-to-500 Hz Spectral Slope During Voiced Segments",
    "slopeV500-1500_sma3nz_amean": "Average Spectral Slope from 500 to 1500 Hz During Voiced Segments",
    "slopeV500-1500_sma3nz_stddevNorm": "Normalized Standard Deviation of the 500-to-1500 Hz Spectral Slope During Voiced Segments",
    "spectralFluxV_sma3nz_amean": "Average Spectral Flux During Voiced Segments",
    "spectralFluxV_sma3nz_stddevNorm": "Normalized Standard Deviation of Spectral Flux During Voiced Segments",
    "mfcc1V_sma3nz_amean": "Average First Mel-Frequency Cepstral Coefficient During Voiced Segments",
    "mfcc1V_sma3nz_stddevNorm": "Normalized Standard Deviation of the First Mel-Frequency Cepstral Coefficient During Voiced Segments",
    "mfcc2V_sma3nz_amean": "Average Second Mel-Frequency Cepstral Coefficient During Voiced Segments",
    "mfcc2V_sma3nz_stddevNorm": "Normalized Standard Deviation of the Second Mel-Frequency Cepstral Coefficient During Voiced Segments",
    "mfcc3V_sma3nz_amean": "Average Third Mel-Frequency Cepstral Coefficient During Voiced Segments",
    "mfcc3V_sma3nz_stddevNorm": "Normalized Standard Deviation of the Third Mel-Frequency Cepstral Coefficient During Voiced Segments",
    "mfcc4V_sma3nz_amean": "Average Fourth Mel-Frequency Cepstral Coefficient During Voiced Segments",
    "mfcc4V_sma3nz_stddevNorm": "Normalized Standard Deviation of the Fourth Mel-Frequency Cepstral Coefficient During Voiced Segments",
    "alphaRatioUV_sma3nz_amean": "Average Alpha Ratio During Unvoiced Segments (dB)",
    "hammarbergIndexUV_sma3nz_amean": "Average Hammarberg Index During Unvoiced Segments (dB)",
    "slopeUV0-500_sma3nz_amean": "Average Spectral Slope from 0 to 500 Hz During Unvoiced Segments",
    "slopeUV500-1500_sma3nz_amean": "Average Spectral Slope from 500 to 1500 Hz During Unvoiced Segments",
    "spectralFluxUV_sma3nz_amean": "Average Spectral Flux During Unvoiced Segments",
    "loudnessPeaksPerSec": "Loudness Peaks per Second",
    "VoicedSegmentsPerSec": "Voiced Segments per Second",
    "MeanVoicedSegmentLengthSec": "Average Voiced Segment Length (seconds)",
    "StddevVoicedSegmentLengthSec": "Standard Deviation of Voiced Segment Length (seconds)",
    "MeanUnvoicedSegmentLength": "Average Unvoiced Segment Length (seconds)",
    "StddevUnvoicedSegmentLength": "Standard Deviation of Unvoiced Segment Length (seconds)",
    "equivalentSoundLevel_dBp": "Equivalent Sound Level (dBp)",
}

PROMPT_HEADER = (
    "Here are 88 acoustic features extracted from the current audio sample.\n"
    "Use these measurements as supporting evidence when producing the "
    "forensic report.\n"
    "The features are:"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def audio_message(record):
    matches = [
        message
        for message in record.get("conversation", [])
        if message.get("role") == "user"
        and message.get("message_type") == "audio"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one user audio message for {record.get('id')}")
    return matches[0]


def text_message(record):
    matches = [
        message
        for message in record.get("conversation", [])
        if message.get("role") == "user"
        and message.get("message_type") == "text"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one user text message for {record.get('id')}")
    return matches[0]


def validate_feature_names(smile):
    expected = list(FEATURE_LABELS)
    actual = list(smile.feature_names)
    if len(expected) != 88 or len(set(FEATURE_LABELS.values())) != 88:
        raise RuntimeError("Natural-language feature mapping must contain 88 unique labels")
    if actual != expected:
        missing = sorted(set(actual) - set(expected))
        extra = sorted(set(expected) - set(actual))
        raise RuntimeError(
            f"openSMILE feature mapping mismatch: missing={missing}, extra={extra}"
        )


def load_cache(path):
    cached = {}
    if not path.is_file():
        return cached
    for row in read_jsonl(path):
        file_id = row.get("id")
        if not file_id or file_id in cached:
            raise ValueError(f"Missing or duplicate cache ID in {path}: {file_id}")
        if list(row.get("features", {})) != list(FEATURE_LABELS):
            raise ValueError(f"Invalid cached feature set for {file_id}")
        cached[file_id] = row
    return cached


def cache_matches(row, audio_path):
    stat = audio_path.stat()
    return (
        row.get("audio_path") == audio_path.as_posix()
        and row.get("source_size") == stat.st_size
        and row.get("source_mtime_ns") == stat.st_mtime_ns
        and row.get("opensmile_version") == opensmile.__version__
    )


def extract_chunk(smile, records):
    paths = [record["audio_path"].as_posix() for record in records]
    frame = smile.process_files(paths)
    if frame.shape != (len(records), 88):
        raise RuntimeError(
            f"Unexpected openSMILE output shape: {frame.shape} for {len(records)} files"
        )
    by_path = {}
    for index, values in frame.iterrows():
        path = Path(index[0]).resolve().as_posix()
        if path in by_path:
            raise RuntimeError(f"Duplicate openSMILE output path: {path}")
        by_path[path] = values

    rows = []
    for record in records:
        audio_path = record["audio_path"]
        values = by_path.get(audio_path.as_posix())
        if values is None:
            raise RuntimeError(f"Missing openSMILE output for {audio_path}")
        features = {}
        for name in FEATURE_LABELS:
            value = float(values[name])
            features[name] = value if math.isfinite(value) else None
        stat = audio_path.stat()
        rows.append(
            {
                "id": record["id"],
                "audio_path": audio_path.as_posix(),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "opensmile_version": opensmile.__version__,
                "feature_set": "eGeMAPSv02",
                "feature_level": "Functionals",
                "features": features,
            }
        )
    return rows


def format_feature_prompt(features):
    lines = [PROMPT_HEADER, ""]
    for name, label in FEATURE_LABELS.items():
        value = features[name]
        rendered = "Unavailable" if value is None else f"{value:.3f}"
        lines.append(f"{label}: {rendered}")
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.workers <= 0 or args.chunk_size <= 0:
        raise ValueError("--workers and --chunk-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    source_rows = read_jsonl(args.input_manifest)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise ValueError("Input manifest is empty")
    ids = [row.get("id") for row in source_rows]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("Input manifest contains missing or duplicate IDs")

    records = []
    for row in source_rows:
        audio_path = Path(audio_message(row)["content"]).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        records.append({"id": row["id"], "audio_path": audio_path})

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=args.workers,
    )
    validate_feature_names(smile)

    cached = load_cache(args.feature_cache)
    unexpected = set(cached) - set(ids)
    if unexpected:
        raise ValueError(f"Feature cache contains unexpected IDs: {sorted(unexpected)[:5]}")
    for record in records:
        existing = cached.get(record["id"])
        if existing is not None and not cache_matches(existing, record["audio_path"]):
            raise ValueError(
                f"Stale feature cache for {record['id']}; remove {args.feature_cache} and retry"
            )

    pending = [record for record in records if record["id"] not in cached]
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    processed = len(records) - len(pending)
    mode = "a" if args.feature_cache.is_file() else "w"
    with args.feature_cache.open(mode, encoding="utf-8", buffering=1) as handle:
        for start in range(0, len(pending), args.chunk_size):
            chunk = pending[start : start + args.chunk_size]
            extracted = extract_chunk(smile, chunk)
            for row in extracted:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                cached[row["id"]] = row
            handle.flush()
            processed += len(chunk)
            elapsed = time.monotonic() - started
            newly_processed = processed - (len(records) - len(pending))
            rate = newly_processed / elapsed if elapsed > 0 else 0
            eta = (len(records) - processed) / rate if rate > 0 else math.inf
            eta_text = "unknown" if not math.isfinite(eta) else time.strftime("%H:%M:%S", time.gmtime(eta))
            print(
                f"[opensmile] {processed}/{len(records)} "
                f"({100 * processed / len(records):.1f}%) "
                f"rate={rate:.2f} files/s ETA={eta_text}",
                flush=True,
            )

    output_rows = []
    unavailable_counts = {name: 0 for name in FEATURE_LABELS}
    for source in source_rows:
        feature_row = cached[source["id"]]
        features = feature_row["features"]
        for name, value in features.items():
            if value is None:
                unavailable_counts[name] += 1
        output = copy.deepcopy(source)
        message = text_message(output)
        message["content"] = (
            message["content"].rstrip()
            + "\n\n"
            + format_feature_prompt(features)
        )
        output_rows.append(output)

    write_jsonl(args.output_manifest, output_rows)
    summary_path = args.summary_json or args.output_manifest.with_suffix(".summary.json")
    summary = {
        "input_manifest": args.input_manifest.resolve().as_posix(),
        "output_manifest": args.output_manifest.resolve().as_posix(),
        "feature_cache": args.feature_cache.resolve().as_posix(),
        "opensmile_version": opensmile.__version__,
        "feature_set": "eGeMAPSv02",
        "feature_level": "Functionals",
        "feature_count": len(FEATURE_LABELS),
        "samples": len(output_rows),
        "unavailable_counts": unavailable_counts,
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
