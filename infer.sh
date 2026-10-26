#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
CHECKPOINT=${1:?Usage: bash infer.sh CHECKPOINT AUDIO_DIRECTORY [OUTPUT_DIRECTORY]}
AUDIO_DIR=${2:?Audio directory required}
OUTPUT=${3:-outputs}
mkdir -p "$OUTPUT"
test ! -e "$OUTPUT/raw_predictions.jsonl" || { echo 'Output already exists; choose a new directory' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ENABLE_AUDIO_OUTPUT=0
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
python prepare_data.py --audio-dir "$AUDIO_DIR" --output "$OUTPUT/input.jsonl"
swift infer \
    --model "$CHECKPOINT" \
    --val_dataset "$OUTPUT/input.jsonl" \
    --result_path "$OUTPUT/raw_predictions.jsonl" \
    --infer_backend transformers \
    --template qwen2_5_omni \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --max_length 4096 \
    --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
    --temperature 0 \
    --max_batch_size "${MAX_BATCH_SIZE:-16}" \
    --write_batch_size 32 \
    --load_args false
python export_submission.py --input "$OUTPUT/input.jsonl" \
    --raw "$OUTPUT/raw_predictions.jsonl" --output "$OUTPUT/prediction.jsonl"
