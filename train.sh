#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12
export MAX_PIXELS=1003520
export ENABLE_AUDIO_OUTPUT=0
export WANDB_MODE=disabled
python prepare_data.py
swift sft \
    --model "${MODEL:-Qwen/Qwen2.5-Omni-7B}" \
    --dataset data/swift/train.jsonl \
    --val_dataset data/swift/dev.jsonl \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --save_only_model true \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --freeze_vit false \
    --freeze_aligner false \
    --max_length 1536 \
    --dataset_shuffle true \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end false \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --report_to none \
    --output_dir runs/track2 \
    --create_checkpoint_symlink true \
    --max_grad_norm 1.0 \
    --seed 42
