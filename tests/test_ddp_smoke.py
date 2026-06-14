from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import yaml


def _write_wav(path: Path, *, frames: int = 512, channels: int = 2, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.linspace(-0.4, 0.4, num=frames * channels, dtype=np.float32).reshape(frames, channels)
    pcm = np.clip(values * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _write_musdb_track(root: Path) -> None:
    for stem in ("mixture", "vocals", "bass", "drums", "other"):
        _write_wav(root / "train" / "Song A" / f"{stem}.wav")


def _tiny_bs_roformer_config() -> dict:
    return {
        "audio": {
            "chunk_size": 256,
            "dim_f": 32,
            "dim_t": 16,
            "hop_length": 16,
            "n_fft": 64,
            "num_channels": 2,
            "sample_rate": 44100,
            "min_mean_abs": 0.0,
        },
        "model": {
            "dim": 8,
            "depth": 1,
            "stereo": True,
            "num_stems": 1,
            "time_transformer_depth": 1,
            "freq_transformer_depth": 1,
            "linear_transformer_depth": 0,
            "dim_head": 8,
            "heads": 1,
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "flash_attn": False,
            "stft_n_fft": 64,
            "stft_hop_length": 16,
            "stft_win_length": 64,
            "stft_normalized": False,
            "mask_estimator_depth": 1,
            "mlp_expansion_factor": 2,
            "freqs_per_bands": [8, 8, 8, 9],
        },
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "grad_clip": 0,
            "instruments": ["vocals", "bass", "drums", "other"],
            "lr": 1.0e-4,
            "target_instrument": "vocals",
            "num_epochs": 1,
            "num_steps": 2,
            "q": 0.95,
            "coarse_loss_clip": True,
            "optimizer": "adam",
            "use_amp": False,
        },
        "inference": {"batch_size": 1, "dim_t": 16, "num_overlap": 2},
    }


@pytest.mark.integration
def test_train_cli_cpu_ddp_smoke(tmp_path):
    if os.environ.get("PYMSST_RUN_DDP_SMOKE") != "1":
        pytest.skip("set PYMSST_RUN_DDP_SMOKE=1 to run the CPU DDP smoke test")

    _write_musdb_track(tmp_path / "musdb")
    config_path = tmp_path / "bs_roformer.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_bs_roformer_config()), encoding="utf-8")
    results_path = tmp_path / "runs"
    export_path = tmp_path / "core.ckpt"
    command = [
        "uv",
        "run",
        "pymsst",
        "train",
        "--model-type",
        "bs_roformer",
        "--config-path",
        str(config_path),
        "--data-root",
        str(tmp_path / "musdb"),
        "--results-path",
        str(results_path),
        "--batch-size",
        "1",
        "--steps-per-epoch",
        "2",
        "--max-epochs",
        "1",
        "--max-steps",
        "1",
        "--val-tracks",
        "0",
        "--accelerator",
        "cpu",
        "--devices",
        "2",
        "--strategy",
        "ddp",
        "--precision",
        "32-true",
        "--export-core-checkpoint",
        str(export_path),
    ]

    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, timeout=60)

    run_dir = results_path / "bs_roformer"
    assert (run_dir / "checkpoints" / "last.ckpt").is_file()
    assert (run_dir / "loss_curve.csv").is_file()
    assert export_path.is_file()
