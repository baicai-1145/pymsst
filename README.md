# pymsst

`pymsst` is the training package for `pymss-core` music source separation
models. It is separate from the `pymss` inference package.

The first milestone targets:

- `bs_roformer`
- `mel_band_roformer`
- MUSDB18-HQ wav training
- PyTorch Lightning single-GPU and DDP training
- checkpoints that can be exported for `pymss-core` inference

## Local Setup

This project uses `uv`.

```bash
uv sync --dev
uv run pymsst --version
```

The local MUSDB18-HQ dataset is expected at:

```text
/root/pymsst/musdb18hq
```

Check the dataset layout:

```bash
uv run pymsst check-musdb /root/pymsst/musdb18hq
```

## Training

`pymsst train` builds the model through `pymss-core`, reads MUSDB18-HQ wav
chunks directly, trains with PyTorch Lightning, writes Lightning checkpoints,
and can export a `pymss-core` compatible checkpoint.

Short BSR smoke test:

```bash
CONFIG_ROOT=/root/Music-Source-Separation-Training/configs
uv run pymsst train \
  --model-type bs_roformer \
  --config-path "$CONFIG_ROOT/config_musdb18_bs_roformer.yaml" \
  --data-root /root/pymsst/musdb18hq \
  --results-path /root/pymsst/runs_smoke/bsr \
  --batch-size 1 \
  --steps-per-epoch 1 \
  --max-steps 1 \
  --val-tracks 1 \
  --accelerator gpu \
  --devices 1 \
  --precision 16-mixed \
  --export-core-checkpoint /root/pymsst/runs_smoke/bs_roformer_core.ckpt
```

Short MBR smoke test:

```bash
CONFIG_ROOT=/root/Music-Source-Separation-Training/configs
uv run pymsst train \
  --model-type mel_band_roformer \
  --config-path "$CONFIG_ROOT/config_musdb18_mel_band_roformer.yaml" \
  --data-root /root/pymsst/musdb18hq \
  --results-path /root/pymsst/runs_smoke/mbr \
  --batch-size 1 \
  --steps-per-epoch 1 \
  --max-steps 1 \
  --val-tracks 1 \
  --accelerator gpu \
  --devices 1 \
  --precision 16-mixed \
  --export-core-checkpoint /root/pymsst/runs_smoke/mel_band_roformer_core.ckpt
```

Use `--resume-checkpoint <last.ckpt>` to resume a Lightning run. Use
`--start-checkpoint <model.ckpt>` to initialize model weights before a new
training run. `run_metadata.json` records the effective precision, gradient
accumulation, gradient clipping, seed, data path, and config used by the run.

Validation tracks are split from the MUSDB `train/` split. The official
`test/` split is only checked for layout and kept out of training, early
stopping, and comparison-plan validation subsets.

DDP uses Lightning's standard device and strategy controls:

```bash
uv run pymsst train \
  --model-type bs_roformer \
  --config-path "$CONFIG_ROOT/config_musdb18_bs_roformer.yaml" \
  --data-root /root/pymsst/musdb18hq \
  --results-path /root/pymsst/runs/bsr_ddp \
  --accelerator gpu \
  --devices 0,1 \
  --strategy ddp
```

Only rank zero writes `run_metadata.json`, loss curves, and exported
`pymss-core` checkpoints. The current local machine has one 12GB GPU, so local
validation covers the DDP-safe code paths and device parsing, not a real
multi-GPU run.

Run the optional CPU DDP smoke test with:

```bash
PYMSST_RUN_DDP_SMOKE=1 uv run pytest -q tests/test_ddp_smoke.py
```

This launches two local Gloo ranks on a tiny BSR config and verifies `last.ckpt`,
loss CSV, and exported `pymss-core` checkpoint writes.

## Original Training Comparison

`compare-direct` runs a lightweight pymsst or original-repo training loop and
writes only `loss_curve.csv` plus metadata. It does not save model checkpoints,
so it is suitable for longer loss-curve checks on limited disk space. The direct
comparison path pins original metadata scanning to one process so track order is
deterministic across pymsst and the original repo.

Example BSR pretrained reduced-chunk check:

```bash
CONFIG_ROOT=runs_smoke/compare_plan_pretrained_1step_b2_chunk32768/configs
CKPT=/root/pymsst/pretrained/model_bs_roformer_ep_317_sdr_12.9755.ckpt
uv run pymsst compare-direct \
  --implementation pymsst \
  --model-type bs_roformer \
  --config-path "$CONFIG_ROOT/bs_roformer_pretrained_seed0.yaml" \
  --data-root /root/pymsst/musdb18hq \
  --results-path runs_smoke/direct_compare_bsr_pretrained_2step/pymsst \
  --steps 2 \
  --seed 0 \
  --start-checkpoint "$CKPT"
```

Run the same command with `--implementation original`, then compare:

```bash
RUN_ROOT=runs_smoke/direct_compare_bsr_pretrained_2step
uv run pymsst compare-loss \
  --pymsst-loss-curve "$RUN_ROOT/pymsst/loss_curve.csv" \
  --original-checkpoint "$RUN_ROOT/original/loss_curve.csv"
```

Use `compare-summary` to aggregate completed comparison JSON files and enforce
minimum point count plus loss-difference thresholds:

```bash
P=runs_smoke
uv run pymsst compare-summary \
  --glob "$P/direct_compare_*_detmetadata/comparison.json" \
  "$P/direct_compare_bsr_pretrained_1000step_seed0/comparison.json" \
  "$P/direct_compare_mbr_pretrained_1000step_seed0/comparison.json" \
  --min-points 1000 \
  --max-mean-abs-diff 3.5e-5 \
  --max-max-abs-diff 1.7e-3 \
  --output "$P/reduced_1000step_summary.json"
```

The historical no-checkpoint 1000-step reduced matrix was produced while using
an editable local `pymss-core` checkout with RoFormer alignment fixes:

- BSR scratch: seeds 0, 1, 2; batch 2, chunk 32768.
  Mean abs diff <= `8.91e-7`; max abs diff <= `1.07e-4`.
- MBR scratch: seeds 0, 1, 2; batch 2, chunk 32768.
  Mean abs diff <= `1.12e-8`; max abs diff <= `2.33e-7`.
- BSR pretrained: seeds 0, 1, 2; batch 2, chunk 32768.
  Mean abs diff <= `5.73e-6`; max abs diff <= `7.46e-4`.
- MBR pretrained: seeds 0, 1, 2; batch 2, chunk 32634.
  Mean abs diff <= `3.41e-5`; max abs diff <= `1.66e-3`.

Those reduced comparison outputs live under `runs_smoke/direct_compare_*` and
include the exact command metadata plus `comparison.json`. They were produced
before `pymss-core==0.1.2` was released, so they are retained as historical
evidence rather than the current package baseline.

With PyPI `pymss-core==0.1.2`, fresh 100-step reduced probes run against the
original repo are much closer:

- BSR scratch, `batch=2`, `chunk=32768`: mean abs diff `8.12e-6`,
  max abs diff `4.28e-5`.
- MBR scratch, `batch=2`, `chunk=32768`: mean abs diff `4.02e-6`,
  max abs diff `4.35e-5`.

The package trains and strict-loads the local pretrained weights. Longer
1000-step PyPI `pymss-core==0.1.2` comparison runs should replace the historical
matrix when disk and run time allow.

Full pretrained viperx chunks exceed the local 12GB GPU memory, and
checkpoint-saving comparison runs exceed the current system disk headroom; use
`compare-direct` for no-checkpoint curves.

The current machine also fails a no-checkpoint full-chunk BSR pretrained
single-step probe with `batch_size=1` and the original `chunk_size=352800`:

```bash
CONFIG_ROOT=/root/Music-Source-Separation-Training/configs/viperx
CKPT=/root/pymsst/pretrained/model_bs_roformer_ep_317_sdr_12.9755.ckpt
uv run pymsst compare-direct \
  --implementation pymsst \
  --model-type bs_roformer \
  --config-path "$CONFIG_ROOT/model_bs_roformer_ep_317_sdr_12.9755.yaml" \
  --data-root /root/pymsst/musdb18hq \
  --results-path runs_smoke/full_chunk_probe_bsr_pymsst \
  --steps 1 \
  --seed 0 \
  --batch-size 1 \
  --start-checkpoint "$CKPT"
```

On the local RTX 3080 Ti 12GB this reaches `11.61 GiB` in use and fails while
allocating another `98 MiB`, so the remaining full-chunk evidence requires a GPU
with more memory or a memory-saving model/runtime change outside this training
package.

## Local Pretrained Weights

These local files are used for first-stage fine-tuning comparisons:

```text
/root/pymsst/pretrained/model_bs_roformer_ep_317_sdr_12.9755.ckpt
/root/pymsst/pretrained/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt
```

They are intentionally ignored by git.

Strict-load both local pretrained checkpoints through the training module:

```bash
PYMSST_RUN_PRETRAINED_COMPAT=1 uv run pytest -q tests/test_pretrained_compat.py
```

## Package Boundary

`pymss-core` owns model definitions, config loading, and checkpoint
compatibility helpers.

`pymsst` owns datasets, audio chunk loading, augmentations, losses, metrics,
Lightning modules, training loops, validation, and checkpoint export.

Default MUSDB18-HQ wav training should not require PyAV, Librosa, SoundFile, or
Torchaudio. Non-wav format support should live behind optional extras.
