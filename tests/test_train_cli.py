from __future__ import annotations

import csv
import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pymss_core import get_model_from_config, load_model_weights

from pymsst.cli import main


def _write_wav(path: Path, *, frames: int = 512, channels: int = 2, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.linspace(-0.4, 0.4, num=frames * channels, dtype=np.float32).reshape(frames, channels)
    pcm = np.clip(values * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _write_musdb_track(root: Path, name: str = "Song A") -> None:
    for stem in ("mixture", "vocals", "bass", "drums", "other"):
        _write_wav(root / "train" / name / f"{stem}.wav")


def _tiny_config(model_type: str) -> dict:
    model = {
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
    }
    if model_type == "bs_roformer":
        model["freqs_per_bands"] = [8, 8, 8, 9]
    else:
        model["num_bands"] = 8
        model["sample_rate"] = 44100

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
        "model": model,
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "grad_clip": 0,
            "instruments": ["vocals", "bass", "drums", "other"],
            "lr": 1.0e-4,
            "target_instrument": "vocals",
            "num_epochs": 1,
            "num_steps": 1,
            "q": 0.95,
            "coarse_loss_clip": True,
            "optimizer": "adam",
            "use_amp": False,
        },
        "inference": {"batch_size": 1, "dim_t": 16, "num_overlap": 2},
    }


@pytest.mark.parametrize("model_type", ("bs_roformer", "mel_band_roformer"))
def test_train_cli_saves_lightning_checkpoint_loss_curve_and_core_export(tmp_path, model_type):
    _write_musdb_track(tmp_path / "musdb")
    config_path = tmp_path / f"{model_type}.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_config(model_type)), encoding="utf-8")
    results_path = tmp_path / "runs"
    loss_history_path = tmp_path / f"{model_type}_loss.csv"
    export_path = tmp_path / f"{model_type}_core.ckpt"

    status = main(
        [
            "train",
            "--model-type",
            model_type,
            "--config-path",
            str(config_path),
            "--data-root",
            str(tmp_path / "musdb"),
            "--results-path",
            str(results_path),
            "--batch-size",
            "1",
            "--steps-per-epoch",
            "1",
            "--max-epochs",
            "1",
            "--max-steps",
            "1",
            "--val-tracks",
            "0",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--precision",
            "32-true",
            "--accumulate-grad-batches",
            "1",
            "--gradient-clip-val",
            "0.25",
            "--loss-history-path",
            str(loss_history_path),
            "--export-core-checkpoint",
            str(export_path),
        ]
    )

    assert status == 0
    assert (results_path / model_type / "checkpoints" / "last.ckpt").is_file()
    metadata = json.loads((results_path / model_type / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["accumulate_grad_batches"] == 1
    assert metadata["gradient_clip_val"] == 0.25
    assert metadata["precision"] == "32-true"
    assert metadata["cuda_graph_mode"] == "off"
    assert metadata["cuda_graph_active"] == "off"
    assert metadata["cuda_graph_reason"] is None
    assert loss_history_path.is_file()
    with loss_history_path.open(newline="", encoding="utf-8") as handle:
        train_rows = [row for row in csv.DictReader(handle) if row["event"] == "train"]
    assert len(train_rows) == 1
    assert float(train_rows[0]["loss"]) >= 0

    package = torch.load(export_path, map_location="cpu", weights_only=False)
    exported_config_path = tmp_path / f"{model_type}_exported_config.yaml"
    exported_config_path.write_text(yaml.safe_dump(package["config"]), encoding="utf-8")
    fresh_model, _ = get_model_from_config(model_type, exported_config_path)
    load_result = load_model_weights(fresh_model, export_path, model_type=model_type, strict=True)
    assert export_path.is_file()
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys


def test_train_cli_resumes_from_lightning_checkpoint(tmp_path):
    model_type = "bs_roformer"
    _write_musdb_track(tmp_path / "musdb")
    config_path = tmp_path / f"{model_type}.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_config(model_type)), encoding="utf-8")
    results_path = tmp_path / "runs"
    first_loss_path = tmp_path / "first_loss.csv"

    status = main(
        [
            "train",
            "--model-type",
            model_type,
            "--config-path",
            str(config_path),
            "--data-root",
            str(tmp_path / "musdb"),
            "--results-path",
            str(results_path),
            "--batch-size",
            "1",
            "--steps-per-epoch",
            "1",
            "--max-epochs",
            "1",
            "--max-steps",
            "1",
            "--val-tracks",
            "0",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--precision",
            "32-true",
            "--loss-history-path",
            str(first_loss_path),
        ]
    )
    assert status == 0
    last_checkpoint = results_path / model_type / "checkpoints" / "last.ckpt"
    assert int(torch.load(last_checkpoint, map_location="cpu", weights_only=False)["global_step"]) == 1

    resumed_loss_path = tmp_path / "resumed_loss.csv"
    status = main(
        [
            "train",
            "--model-type",
            model_type,
            "--config-path",
            str(config_path),
            "--data-root",
            str(tmp_path / "musdb"),
            "--results-path",
            str(results_path),
            "--resume-checkpoint",
            str(last_checkpoint),
            "--batch-size",
            "1",
            "--steps-per-epoch",
            "1",
            "--max-epochs",
            "2",
            "--max-steps",
            "2",
            "--val-tracks",
            "0",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--precision",
            "32-true",
            "--loss-history-path",
            str(resumed_loss_path),
        ]
    )

    assert status == 0
    assert int(torch.load(last_checkpoint, map_location="cpu", weights_only=False)["global_step"]) == 2
    with resumed_loss_path.open(newline="", encoding="utf-8") as handle:
        train_rows = [row for row in csv.DictReader(handle) if row["event"] == "train"]
    assert len(train_rows) == 1
    assert int(train_rows[0]["global_step"]) == 2


def test_train_cli_initializes_from_core_start_checkpoint(tmp_path):
    model_type = "mel_band_roformer"
    _write_musdb_track(tmp_path / "musdb")
    config_path = tmp_path / f"{model_type}.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_config(model_type)), encoding="utf-8")
    source_results_path = tmp_path / "source_runs"
    start_checkpoint = tmp_path / "start_core.ckpt"

    status = main(
        [
            "train",
            "--model-type",
            model_type,
            "--config-path",
            str(config_path),
            "--data-root",
            str(tmp_path / "musdb"),
            "--results-path",
            str(source_results_path),
            "--batch-size",
            "1",
            "--steps-per-epoch",
            "1",
            "--max-epochs",
            "1",
            "--max-steps",
            "1",
            "--val-tracks",
            "0",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--precision",
            "32-true",
            "--export-core-checkpoint",
            str(start_checkpoint),
        ]
    )
    assert status == 0
    assert start_checkpoint.is_file()

    finetune_loss_path = tmp_path / "finetune_loss.csv"
    finetune_results_path = tmp_path / "finetune_runs"
    status = main(
        [
            "train",
            "--model-type",
            model_type,
            "--config-path",
            str(config_path),
            "--data-root",
            str(tmp_path / "musdb"),
            "--results-path",
            str(finetune_results_path),
            "--start-checkpoint",
            str(start_checkpoint),
            "--batch-size",
            "1",
            "--steps-per-epoch",
            "1",
            "--max-epochs",
            "1",
            "--max-steps",
            "1",
            "--val-tracks",
            "0",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--precision",
            "32-true",
            "--loss-history-path",
            str(finetune_loss_path),
        ]
    )

    assert status == 0
    assert (finetune_results_path / model_type / "checkpoints" / "last.ckpt").is_file()
    with finetune_loss_path.open(newline="", encoding="utf-8") as handle:
        train_rows = [row for row in csv.DictReader(handle) if row["event"] == "train"]
    assert len(train_rows) == 1
    assert float(train_rows[0]["loss"]) >= 0


@pytest.mark.parametrize("model_type", ("bs_roformer", "mel_band_roformer"))
def test_train_cli_with_validation_logs_metrics_and_monitored_checkpoint(tmp_path, model_type):
    _write_musdb_track(tmp_path / "musdb", "Train A")
    _write_musdb_track(tmp_path / "musdb", "Train B")
    config_path = tmp_path / f"{model_type}.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_config(model_type)), encoding="utf-8")
    results_path = tmp_path / "runs"
    loss_history_path = tmp_path / f"{model_type}_loss.csv"

    status = main(
        [
            "train",
            "--model-type",
            model_type,
            "--config-path",
            str(config_path),
            "--data-root",
            str(tmp_path / "musdb"),
            "--results-path",
            str(results_path),
            "--batch-size",
            "1",
            "--steps-per-epoch",
            "1",
            "--max-epochs",
            "1",
            "--max-steps",
            "1",
            "--val-tracks",
            "1",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--precision",
            "32-true",
            "--loss-history-path",
            str(loss_history_path),
        ]
    )

    assert status == 0
    with loss_history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train_rows = [row for row in rows if row["event"] == "train"]
    validation_rows = [row for row in rows if row["event"] == "validation"]
    checkpoint_dir = results_path / model_type / "checkpoints"
    monitored_checkpoints = [
        path
        for path in checkpoint_dir.glob("*.ckpt")
        if path.name != "last.ckpt"
    ]

    assert len(train_rows) == 1
    assert len(validation_rows) == 1
    assert validation_rows[0]["val_loss"]
    assert validation_rows[0]["val_sdr"]
    assert math.isfinite(float(validation_rows[0]["val_loss"]))
    assert math.isfinite(float(validation_rows[0]["val_sdr"]))
    assert (checkpoint_dir / "last.ckpt").is_file()
    assert len(monitored_checkpoints) == 1
