#!/usr/bin/env python3
"""Add resume, LoRA-gradient checks, and profiling to the official trainer."""

import argparse
import atexit
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")


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


def resolve_moss_model(model):
    get_base_model = getattr(model, "get_base_model", None)
    moss_model = get_base_model() if callable(get_base_model) else model
    if not hasattr(moss_model, "language_model"):
        raise RuntimeError(
            "Could not resolve MossAudioModel from the training model"
        )
    return moss_model


def disable_training_cache(moss_model):
    configs = (
        ("model.config", getattr(moss_model, "config", None)),
        (
            "model.config.language_config",
            getattr(getattr(moss_model, "config", None), "language_config", None),
        ),
        (
            "model.language_model.config",
            getattr(getattr(moss_model, "language_model", None), "config", None),
        ),
    )
    disabled = []
    for name, config in configs:
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
            disabled.append(name)
    if not disabled:
        raise RuntimeError("No MOSS/Qwen cache configuration was found")
    if any(
        getattr(config, "use_cache", None) is not False
        for _, config in configs
        if config is not None and hasattr(config, "use_cache")
    ):
        raise RuntimeError("Failed to disable use_cache for MOSS training")
    return disabled


def install_safe_gradient_checkpointing(model, captured):
    original_enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(original_enable):
        raise RuntimeError("Training model does not support gradient checkpointing")

    moss_model = resolve_moss_model(model)
    language_model = moss_model.language_model
    layers = list(getattr(language_model, "layers", ()))
    deepstack_mergers = list(
        getattr(moss_model, "deepstack_audio_merger_list", ())
    )
    hooked_layer_count = len(deepstack_mergers)
    if not layers:
        raise RuntimeError("MOSS language model has no decoder layers")
    if hooked_layer_count > len(layers):
        raise RuntimeError(
            "DeepStack hook count exceeds the number of language-model layers"
        )

    def safe_enable(gradient_checkpointing_kwargs=None):
        checkpointing_kwargs = dict(gradient_checkpointing_kwargs or {})
        checkpointing_kwargs["use_reentrant"] = False
        disabled_cache_configs = disable_training_cache(moss_model)
        original_enable(
            gradient_checkpointing_kwargs=checkpointing_kwargs
        )

        enable_input_grads = getattr(model, "enable_input_require_grads", None)
        if not callable(enable_input_grads):
            raise RuntimeError(
                "LoRA gradient checkpointing requires input gradients"
            )
        enable_input_grads()

        for layer_index, layer in enumerate(layers):
            if not hasattr(layer, "gradient_checkpointing"):
                raise RuntimeError(
                    f"Language-model layer {layer_index} has no "
                    "gradient_checkpointing flag"
                )
            if layer_index < hooked_layer_count:
                layer.gradient_checkpointing = False

        skipped_layers = [
            index
            for index, layer in enumerate(layers)
            if not layer.gradient_checkpointing
        ]
        checkpointed_layers = [
            index
            for index, layer in enumerate(layers)
            if layer.gradient_checkpointing
        ]
        expected_skipped = list(range(hooked_layer_count))
        expected_checkpointed = list(range(hooked_layer_count, len(layers)))
        if skipped_layers != expected_skipped:
            raise RuntimeError(
                "Unexpected non-checkpointed language-model layers: "
                f"{skipped_layers}; expected {expected_skipped}"
            )
        if checkpointed_layers != expected_checkpointed:
            raise RuntimeError(
                "Unexpected checkpointed language-model layers: "
                f"{checkpointed_layers}; expected {expected_checkpointed}"
            )

        captured["gradient_checkpointing"] = True
        captured["gradient_checkpointing_kwargs"] = checkpointing_kwargs
        captured["input_require_grads_enabled"] = True
        captured["deepstack_non_checkpointed_layers"] = skipped_layers
        captured["checkpointed_language_layers"] = checkpointed_layers
        captured["cache_disabled_configs"] = disabled_cache_configs
        print(
            "[gradient-checkpointing] "
            f"DeepStack layers {skipped_layers} excluded; "
            f"checkpointed layers={len(checkpointed_layers)}/{len(layers)}; "
            f"kwargs={checkpointing_kwargs}; use_cache=False"
        )

    model.gradient_checkpointing_enable = safe_enable


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
        "gradient_checkpointing": False,
        "gradient_checkpointing_kwargs": None,
        "input_require_grads_enabled": False,
        "deepstack_non_checkpointed_layers": [],
        "checkpointed_language_layers": [],
        "cache_disabled_configs": [],
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
        if self.args.gradient_checkpointing:
            checkpointing_kwargs = dict(
                self.args.gradient_checkpointing_kwargs or {}
            )
            checkpointing_kwargs["use_reentrant"] = False
            self.args.gradient_checkpointing_kwargs = checkpointing_kwargs
            install_safe_gradient_checkpointing(self.model, captured)
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
        "gradient_checkpointing": captured["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": captured[
            "gradient_checkpointing_kwargs"
        ],
        "input_require_grads_enabled": captured[
            "input_require_grads_enabled"
        ],
        "deepstack_non_checkpointed_layers": captured[
            "deepstack_non_checkpointed_layers"
        ],
        "checkpointed_language_layers": captured[
            "checkpointed_language_layers"
        ],
        "cache_disabled_configs": captured["cache_disabled_configs"],
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
