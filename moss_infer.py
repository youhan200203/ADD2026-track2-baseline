#!/usr/bin/env python3
"""Run deterministic MOSS-Audio inference and save ADD-compatible JSONL."""

import argparse
import json
import re
import sys
import time
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moss-repo", type=Path, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-key-prefix", default="data/dev")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validate-batch-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-lora", action="store_true")
    return parser.parse_args()


def prepare_batch(processor, audio_paths, device):
    import torch
    from src.audio_io import load_audio

    features = []
    for audio_path in audio_paths:
        raw_audio = load_audio(
            audio_path.as_posix(),
            sample_rate=processor.config.mel_sr,
        )
        item = processor(
            text=PROMPT,
            audios=[raw_audio],
            return_tensors="pt",
        )
        item["audio_input_mask"] = (
            item["input_ids"] == processor.audio_token_id
        )
        features.append(item)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")

    batch_size = len(features)
    max_text_length = max(int(item["input_ids"].shape[1]) for item in features)
    max_audio_frames = max(int(item["audio_data"].shape[-1]) for item in features)
    mel_bins = int(features[0]["audio_data"].shape[1])

    input_ids = torch.full(
        (batch_size, max_text_length),
        int(pad_token_id),
        dtype=features[0]["input_ids"].dtype,
    )
    attention_mask = torch.zeros(
        (batch_size, max_text_length),
        dtype=features[0]["attention_mask"].dtype,
    )
    audio_input_mask = torch.zeros(
        (batch_size, max_text_length),
        dtype=torch.bool,
    )
    audio_data = torch.zeros(
        (batch_size, mel_bins, max_audio_frames),
        dtype=features[0]["audio_data"].dtype,
    )
    audio_data_seqlens = torch.empty(batch_size, dtype=torch.long)

    for index, item in enumerate(features):
        text_length = int(item["input_ids"].shape[1])
        audio_frames = int(item["audio_data_seqlens"].reshape(-1)[0].item())
        input_ids[index, -text_length:] = item["input_ids"][0]
        attention_mask[index, -text_length:] = item["attention_mask"][0]
        audio_input_mask[index, -text_length:] = item["audio_input_mask"][0]
        audio_data[index, :, :audio_frames] = item["audio_data"][0, :, :audio_frames]
        audio_data_seqlens[index] = audio_frames

    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "audio_input_mask": audio_input_mask.to(device),
        "audio_data": audio_data.to(device=device, dtype=torch.bfloat16),
        "audio_data_seqlens": audio_data_seqlens.to(device),
    }
    return batch, int(pad_token_id)


def generate_batch(model, processor, audio_paths, args):
    import torch

    inputs, pad_token_id = prepare_batch(processor, audio_paths, args.device)
    input_width = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
    generated = output_ids[:, input_width:]
    responses = processor.batch_decode(generated, skip_special_tokens=True)
    token_count = sum(
        len(processor.tokenizer.encode(response, add_special_tokens=False))
        for response in responses
    )
    return responses, token_count


def extract_decision(response):
    matches = re.findall(
        r"<Decision>\s*(real|fake)\s*</Decision>", response, re.I
    )
    return matches[0].lower() if len(matches) == 1 else None


def extract_evidence(response):
    match = re.search(r"<Evidence>(.*?)</Evidence>", response, re.I | re.S)
    if match is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", match.group(1))
    return re.sub(r"\s+", " ", text).strip()


def valid_xml_response(response):
    decisions = re.findall(
        r"<Decision>\s*(real|fake)\s*</Decision>", response, re.I
    )
    evidence = re.findall(r"<Evidence>(.*?)</Evidence>", response, re.I | re.S)
    return len(decisions) == 1 and len(evidence) == 1 and bool(evidence[0].strip())


def validate_batch_generation(model, processor, audio_paths, args):
    from tqdm.auto import tqdm

    single_responses = []
    single_progress = tqdm(
        audio_paths,
        desc="batch-check single",
        unit="sample",
        dynamic_ncols=True,
    )
    for audio_path in single_progress:
        responses, _ = generate_batch(model, processor, [audio_path], args)
        single_responses.extend(responses)

    batched_responses = []
    batch_progress = tqdm(
        total=len(audio_paths),
        desc=f"batch-check batch={args.batch_size}",
        unit="sample",
        dynamic_ncols=True,
    )
    for start in range(0, len(audio_paths), args.batch_size):
        batch_paths = audio_paths[start : start + args.batch_size]
        responses, _ = generate_batch(
            model,
            processor,
            batch_paths,
            args,
        )
        batched_responses.extend(responses)
        batch_progress.update(len(batch_paths))
    batch_progress.close()

    rows = []
    for audio_path, single, batched in zip(
        audio_paths, single_responses, batched_responses
    ):
        rows.append(
            {
                "id": audio_path.stem,
                "exact_match": single == batched,
                "single_xml_valid": valid_xml_response(single),
                "batch_xml_valid": valid_xml_response(batched),
                "single_decision": extract_decision(single),
                "batch_decision": extract_decision(batched),
                "single_evidence": extract_evidence(single),
                "batch_evidence": extract_evidence(batched),
            }
        )

    summary = {
        "samples": len(rows),
        "batch_size": args.batch_size,
        "exact_matches": sum(row["exact_match"] for row in rows),
        "decision_matches": sum(
            row["single_decision"] is not None
            and row["single_decision"] == row["batch_decision"]
            for row in rows
        ),
        "single_xml_valid": sum(row["single_xml_valid"] for row in rows),
        "batch_xml_valid": sum(row["batch_xml_valid"] for row in rows),
        "safe_for_exact_reproducibility": all(
            row["exact_match"] for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    printable = dict(summary)
    printable["rows"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"single_evidence", "batch_evidence"}
        }
        for row in rows
    ]
    print(
        f"[batch-validation] "
        f"{json.dumps(printable, ensure_ascii=False, indent=2)}"
    )
    print(f"[batch-validation] saved evidence pairs to {args.output}")
    if summary["batch_xml_valid"] != len(rows):
        raise RuntimeError("Batch validation failed: batch output contains invalid XML")
    if summary["decision_matches"] != len(rows):
        print("[batch-validation] WARNING: at least one Decision changed")
    if not summary["safe_for_exact_reproducibility"]:
        print(
            "[batch-validation] WARNING: some generated texts differ from "
            "batch size 1; compare reference BERTScore before using batching"
        )


def completed_ids(path):
    completed = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error
            file_id = row.get("id")
            if not file_id or file_id in completed:
                raise ValueError(
                    f"Missing or duplicate id at {path}:{line_number}"
                )
            completed.add(file_id)
    return completed


def main():
    args = parse_args()
    import torch
    from tqdm.auto import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for MOSS-Audio inference")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.validate_batch_samples < 0:
        raise ValueError("--validate-batch-samples cannot be negative")
    if args.validate_batch_samples and args.batch_size <= 1:
        raise ValueError("Batch validation requires --batch-size greater than 1")

    moss_repo = args.moss_repo.resolve()
    sys.path.insert(0, moss_repo.as_posix())
    from peft import PeftModel
    from src.configuration_moss_audio import MossAudioConfig
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    audio_files = sorted(args.audio_dir.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {args.audio_dir}")
    if not args.adapter_dir.is_dir():
        raise FileNotFoundError(args.adapter_dir)

    if args.validate_batch_samples:
        done = set()
        pending = audio_files[: args.validate_batch_samples]
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and not args.resume:
            raise FileExistsError(
                f"Output already exists: {args.output}; pass --resume to continue"
            )
        done = completed_ids(args.output) if args.resume else set()
        valid_ids = {path.stem for path in audio_files}
        unexpected = done - valid_ids
        if unexpected:
            raise ValueError(
                f"Output contains unexpected ids: {sorted(unexpected)[:5]}"
            )
        pending = [path for path in audio_files if path.stem not in done]
    print(
        f"[data] total={len(audio_files)} completed={len(done)} "
        f"pending={len(pending)}"
    )
    if not pending:
        print(f"[done] already complete: {args.output}")
        return

    config = MossAudioConfig.from_pretrained(args.model_dir)
    for item in (config, getattr(config, "language_config", None)):
        if item is not None:
            item._attn_implementation = "sdpa"

    print(f"[load] base model: {args.model_dir}")
    model = MossAudioModel.from_pretrained(
        args.model_dir,
        config=config,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    print(f"[load] LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir.as_posix())
    if args.merge_lora:
        print("[load] merging LoRA adapter into the base model")
        model = model.merge_and_unload(safe_merge=True)
        print("[load] LoRA merge complete")
    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        enable_time_marker=True,
    )

    if args.validate_batch_samples:
        validate_batch_generation(model, processor, pending, args)
        return

    start = time.monotonic()
    generated_tokens = 0
    prefix = args.audio_key_prefix.rstrip("/")
    with args.output.open("a", encoding="utf-8", buffering=1) as handle:
        progress = tqdm(
            total=len(audio_files),
            initial=len(done),
            unit="sample",
            dynamic_ncols=True,
        )
        for batch_start in range(0, len(pending), args.batch_size):
            batch_paths = pending[batch_start : batch_start + args.batch_size]
            responses, batch_tokens = generate_batch(
                model, processor, batch_paths, args
            )
            generated_tokens += batch_tokens
            for audio_path, response in zip(batch_paths, responses):
                row = {
                    "id": audio_path.stem,
                    "audios": [f"{prefix}/{audio_path.name}"],
                    "response": response,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            progress.update(len(batch_paths))
            progress.set_postfix(tokens=generated_tokens)

    elapsed = time.monotonic() - start
    print(
        f"[done] {len(pending)} new samples in {elapsed:.1f}s "
        f"({len(pending) / elapsed:.4f} samples/s); output={args.output}"
    )


if __name__ == "__main__":
    main()
