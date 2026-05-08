# ReliFormer

PyTorch implementation of the current **ReliFormer** design for SPAD-guided color reconstruction.

## Dataset

Default dataset root:

`vfi_dataset/i2-2kfps_v1_flat`

Expected split layout:

- `vfi_dataset/i2-2kfps_v1_flat/train`
- `vfi_dataset/i2-2kfps_v1_flat/test`

Loader supports `.npz`, `.pt`, `.pth` sample files with keys compatible with:

- `spad` (shape `(T,H,W)`)
- `cmos` (shape `(4,H/2,W/2)`)
- `target_rgb` (shape `(3,H,W)`)
- `target_lab_L` (shape `(H,W)` or `(1,H,W)`, optional)

For raw frame directories (like `i2-2kfps_v1_flat`), training data is synthesized on-the-fly:

- SPAD burst from clean RGB burst using a Binomial SPAD model with `--ppp` and `--bins`.
- Long-exposure packed RGGB CMOS from burst integration with Poisson-Gaussian noise (`--cmos-sigma`).
- Target is the center clean RGB frame.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Sanity check

```bash
python3 scripts/sanity_check.py
```

## Train

```bash
wandb login
python3 scripts/train.py --data-root vfi_dataset/i2-2kfps_v1_flat
```

## Evaluate

```bash
python3 scripts/eval.py --data-root vfi_dataset/i2-2kfps_v1_flat --ckpt checkpoints/reliformer_epoch_050.pt
```

## W&B defaults in training script

- `--epochs 50`
- `--steps-per-epoch 2000`
- `--eval-samples 20`
- Evaluation visual panels: predicted RGB, target RGB, luma, luma proxy, and edge maps at all 3 scales.

Useful options:

```bash
python3 scripts/train.py \
  --wandb-project reliformer \
  --wandb-run-name reliformer-i2-2kfps \
  --wandb-mode online
```

## Design defaults implemented

- SPAD reliability: `4p(1-p)` (SPAD only)
- CMOS reliability in ACMF: `1 - edge_map_from_spad` (default)
- Decoder color injection: `ColorCrossAttention + StructureGate`
- Loss: `L_total = L_rgb + 0.05 * L_edge`
- No default `L_luma`, no default `L_smooth`
