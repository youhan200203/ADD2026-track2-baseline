# ADD 2026 Track 2 Baseline

Qwen2.5-Omni-7B with ms-swift. Requires Linux, Python 3.11 and CUDA 12.4.

Best weights: [Google Drive](https://drive.google.com/file/d/1_Tr_Bpm5mmTyIQi2Px7ATyXDPENqXIxp/view?usp=sharing) (primary) | [Baidu Drive](https://pan.baidu.com/s/1DyiOrfeuHue3y5kp9dhLFw) (backup; extraction code: `w4nj`).
Extract `track2_baseline_weights.zip` into `track2_baseline/`:

```text
track2_baseline/
├── infer.sh
└── track2_baseline_weights/
    ├── config.json
    └── ...
```

Run from `track2_baseline/`:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash infer.sh track2_baseline_weights /path/to/test_wavs outputs/test
```

This writes `outputs/test/prediction.jsonl`. Each line contains `file_id`,
`label` (`real`/`fake`) and `explanation`. `file_id` is the WAV filename
without `.wav`. Submit a ZIP whose root contains only `prediction.jsonl`.

To train, place the released dataset's `train/`, `dev/` and `labels/`
folders under `data/`. Training requires four 80 GB GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train.sh
```

The best model is selected by development loss and saved under
`runs/track2/<run>/best`.
