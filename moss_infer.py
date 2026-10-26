#!/usr/bin/env python3
"""Run deterministic MOSS-Audio inference and save ADD-compatible JSONL."""

import argparse
import json
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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

    moss_repo = args.moss_repo.resolve()
    sys.path.insert(0, moss_repo.as_posix())
    from peft import PeftModel
    from src.audio_io import load_audio
    from src.configuration_moss_audio import MossAudioConfig
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    audio_files = sorted(args.audio_dir.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {args.audio_dir}")
    if not args.adapter_dir.is_dir():
        raise FileNotFoundError(args.adapter_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {args.output}; pass --resume to continue"
        )
    done = completed_ids(args.output) if args.resume else set()
    valid_ids = {path.stem for path in audio_files}
    unexpected = done - valid_ids
    if unexpected:
        raise ValueError(f"Output contains unexpected ids: {sorted(unexpected)[:5]}")

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
    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        enable_time_marker=True,
    )

    start = time.monotonic()
    generated_tokens = 0
    prefix = args.audio_key_prefix.rstrip("/")
    with args.output.open("a", encoding="utf-8", buffering=1) as handle:
        progress = tqdm(
            pending,
            total=len(audio_files),
            initial=len(done),
            unit="sample",
            dynamic_ncols=True,
        )
        for audio_path in progress:
            raw_audio = load_audio(
                audio_path.as_posix(),
                sample_rate=processor.config.mel_sr,
            )
            inputs = processor(
                text=PROMPT,
                audios=[raw_audio],
                return_tensors="pt",
            )
            inputs = inputs.to(args.device)
            if inputs.get("audio_data") is not None:
                inputs["audio_data"] = inputs["audio_data"].to(torch.bfloat16)
            inputs["audio_input_mask"] = (
                inputs["input_ids"] == processor.audio_token_id
            )
            input_length = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                )

            new_ids = output_ids[0, input_length:]
            response = processor.decode(new_ids, skip_special_tokens=True)
            generated_tokens += int(new_ids.numel())
            row = {
                "id": audio_path.stem,
                "audios": [f"{prefix}/{audio_path.name}"],
                "response": response,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            progress.set_postfix(tokens=generated_tokens)

    elapsed = time.monotonic() - start
    print(
        f"[done] {len(pending)} new samples in {elapsed:.1f}s "
        f"({len(pending) / elapsed:.4f} samples/s); output={args.output}"
    )


if __name__ == "__main__":
    main()
