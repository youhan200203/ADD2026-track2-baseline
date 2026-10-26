#!/usr/bin/env python3
"""Add resume, LoRA-gradient checks, and profiling to the official trainer."""

import argparse
import atexit
import importlib.util
import json
import math
import sys
import time
from pathlib import Path


def parse_wrapper_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--moss-repo", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--require-lora-gradient", action="store_true")
    wrapper, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return wrapper, remaining


def argument_value(arguments, name, default=None):
    try:
        return arguments[arguments.index(name) + 1]
    except (ValueError, IndexError):
        return default


def latest_checkpoint(output_dir):
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return max(checkpoints, default=(None, None))[1]


def directory_size(path, recursive=True):
    if not path.is_dir():
        return 0
    files = path.rglob("*") if recursive else path.glob("*")
    return sum(item.stat().st_size for item in files if item.is_file())


def main():
    wrapper, trainer_arguments = parse_wrapper_args()
    moss_repo = wrapper.moss_repo.resolve()
    official_script = moss_repo / "finetune/finetune.py"
    if not official_script.is_file():
        raise FileNotFoundError(official_script)
    sys.path.insert(0, moss_repo.as_posix())

    import torch
    from transformers import Trainer, TrainerCallback

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for MOSS-Audio training")

    output_dir = Path(
        argument_value(trainer_arguments, "--output_dir", "output")
    )
    requested_resume = wrapper.resume_from_checkpoint
    resume_path = None
    if requested_resume == "auto":
        resume_path = latest_checkpoint(output_dir)
        if resume_path is None:
            print(f"[resume] no checkpoint found under {output_dir}; starting fresh")
        else:
            print(f"[resume] using {resume_path}")
    elif requested_resume:
        resume_path = Path(requested_resume)
        if not resume_path.is_dir():
            raise FileNotFoundError(resume_path)
        print(f"[resume] using {resume_path}")

    captured = {
        "gradient_checks": 0,
        "max_lora_gradient_norm": 0.0,
        "nonzero_lora_gradient_parameters": 0,
        "loss_history": [],
    }

    class ProfileCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            squared_norm = 0.0
            nonzero = 0
            for name, parameter in model.named_parameters():
                if "lora_" not in name or parameter.grad is None:
                    continue
                norm = float(parameter.grad.detach().float().norm().item())
                squared_norm += norm * norm
                if norm > 0:
                    nonzero += 1
            total_norm = math.sqrt(squared_norm)
            captured["gradient_checks"] += 1
            captured["max_lora_gradient_norm"] = max(
                captured["max_lora_gradient_norm"], total_norm
            )
            captured["nonzero_lora_gradient_parameters"] = max(
                captured["nonzero_lora_gradient_parameters"], nonzero
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                captured["loss_history"].append(
                    {"step": int(state.global_step), "loss": float(logs["loss"])}
                )

    original_init = Trainer.__init__
    original_train = Trainer.train

    def patched_init(self, *args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", None) or [])
        callbacks.append(ProfileCallback())
        original_init(self, *args, callbacks=callbacks, **kwargs)
        captured["trainer"] = self

    def patched_train(self, *args, **kwargs):
        if resume_path is not None:
            kwargs["resume_from_checkpoint"] = resume_path.as_posix()
        torch.cuda.reset_peak_memory_stats()
        captured["started_at"] = time.perf_counter()
        result = original_train(self, *args, **kwargs)
        captured["elapsed_seconds"] = time.perf_counter() - captured["started_at"]
        captured["train_metrics"] = dict(result.metrics)
        return result

    Trainer.__init__ = patched_init
    Trainer.train = patched_train

    def restore_trainer():
        Trainer.__init__ = original_init
        Trainer.train = original_train

    atexit.register(restore_trainer)

    spec = importlib.util.spec_from_file_location(
        "moss_official_finetune", official_script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.train()
    restore_trainer()

    if wrapper.require_lora_gradient and (
        captured["gradient_checks"] == 0
        or captured["max_lora_gradient_norm"] <= 0
        or captured["nonzero_lora_gradient_parameters"] == 0
    ):
        raise RuntimeError(
            "LoRA gradient check failed: no non-zero LoRA gradient was observed"
        )

    trainer = captured["trainer"]
    metrics = captured.get("train_metrics", {})
    runtime = float(metrics.get("train_runtime", captured["elapsed_seconds"]))
    steps_per_second = float(metrics.get("train_steps_per_second", 0.0))
    samples_per_second = float(metrics.get("train_samples_per_second", 0.0))
    train_samples = len(trainer.train_dataset)
    epoch_seconds = (
        train_samples / samples_per_second if samples_per_second > 0 else None
    )
    final_checkpoint = latest_checkpoint(output_dir)
    profile = {
        "output_dir": output_dir.as_posix(),
        "resumed_from": resume_path.as_posix() if resume_path else None,
        "global_step": int(trainer.state.global_step),
        "train_runtime_seconds": runtime,
        "step_time_seconds": (
            1.0 / steps_per_second if steps_per_second > 0 else None
        ),
        "train_samples_per_second": samples_per_second,
        "train_loss": metrics.get("train_loss"),
        "estimated_one_epoch_seconds": epoch_seconds,
        "estimated_three_epoch_seconds": (
            3 * epoch_seconds if epoch_seconds is not None else None
        ),
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gradient_checks": captured["gradient_checks"],
        "max_lora_gradient_norm": captured["max_lora_gradient_norm"],
        "nonzero_lora_gradient_parameters": captured[
            "nonzero_lora_gradient_parameters"
        ],
        "loss_history": captured["loss_history"],
        "adapter_size_bytes": directory_size(output_dir, recursive=False),
        "latest_checkpoint": (
            final_checkpoint.as_posix() if final_checkpoint else None
        ),
        "latest_checkpoint_size_bytes": (
            directory_size(final_checkpoint) if final_checkpoint else None
        ),
    }
    wrapper.profile_json.parent.mkdir(parents=True, exist_ok=True)
    wrapper.profile_json.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print(f"[saved] {wrapper.profile_json}")


if __name__ == "__main__":
    main()
