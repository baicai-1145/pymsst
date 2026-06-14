from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from pytest import approx

from pymsst.validation.direct_compare import _apply_overrides
from pymsst.validation.comparison import compare_loss_curves, read_pymsst_loss_curve, summarize_comparisons, write_comparison_plan


def _touch_musdb_track(root: Path, split: str, name: str) -> None:
    track = root / split / name
    track.mkdir(parents=True)
    for stem in ("mixture", "vocals", "drums", "bass", "other"):
        (track / f"{stem}.wav").write_bytes(b"")


def _write_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "audio": {
                    "chunk_size": 256,
                    "num_channels": 2,
                    "sample_rate": 44100,
                    "min_mean_abs": 0.0,
                },
                "model": {
                    "dim": 8,
                    "depth": 1,
                    "stereo": True,
                    "num_stems": 1,
                    "freqs_per_bands": [8, 8, 8, 9],
                    "stft_n_fft": 64,
                    "stft_hop_length": 16,
                    "stft_win_length": 64,
                },
                "training": {
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "instruments": ["vocals", "bass", "drums", "other"],
                    "lr": 1.0e-4,
                    "target_instrument": "vocals",
                    "num_epochs": 10,
                    "num_steps": 10,
                    "use_amp": True,
                },
                "inference": {"batch_size": 1, "num_overlap": 2},
            }
        ),
        encoding="utf-8",
    )


def test_write_comparison_plan_creates_config_and_commands(tmp_path, monkeypatch):
    source_config = tmp_path / "source.yaml"
    _write_config(source_config)
    monkeypatch.setattr(
        "pymsst.validation.comparison.DEFAULT_SCRATCH_CONFIGS",
        {"bs_roformer": source_config},
    )

    plan = write_comparison_plan(
        tmp_path / "plan",
        pymsst_root=tmp_path / "pymsst",
        original_root=tmp_path / "original",
        data_root=tmp_path / "musdb",
        steps=3,
        repeats=1,
        include_pretrained=False,
        batch_size=1,
        chunk_size=128,
        use_amp=False,
    )

    assert len(plan["runs"]) == 2
    assert (tmp_path / "plan" / "comparison_plan.json").is_file()
    assert (tmp_path / "plan" / "run_all.sh").is_file()
    generated = yaml.safe_load((tmp_path / "plan" / "configs" / "bs_roformer_scratch_seed0.yaml").read_text())
    assert generated["audio"]["chunk_size"] == 128
    assert generated["training"]["num_epochs"] == 1
    assert generated["training"]["num_steps"] == 3
    assert generated["training"]["batch_size"] == 1
    assert generated["training"]["use_amp"] is False
    assert generated["training"]["read_metadata_procs"] == 1
    assert plan["runs"][0]["implementation"] == "pymsst"
    assert "--original-loss-compat" in plan["runs"][0]["command"]
    assert plan["runs"][1]["implementation"] == "original"


def test_write_comparison_plan_validation_subset_uses_train_split_only(tmp_path, monkeypatch):
    source_config = tmp_path / "source.yaml"
    data_root = tmp_path / "musdb"
    _write_config(source_config)
    _touch_musdb_track(data_root, "train", "Train A")
    _touch_musdb_track(data_root, "train", "Train B")
    _touch_musdb_track(data_root, "test", "Official Test")
    monkeypatch.setattr(
        "pymsst.validation.comparison.DEFAULT_SCRATCH_CONFIGS",
        {"bs_roformer": source_config},
    )

    plan = write_comparison_plan(
        tmp_path / "plan",
        pymsst_root=tmp_path / "pymsst",
        original_root=tmp_path / "original",
        data_root=data_root,
        steps=1,
        repeats=1,
        include_pretrained=False,
        val_tracks=1,
    )

    subset_tracks = sorted(item.name for item in (tmp_path / "plan" / "validation" / "train_subset").iterdir())

    assert subset_tracks == ["Train A"]
    assert "Official Test" not in subset_tracks
    assert str(tmp_path / "plan" / "validation" / "train_subset") in plan["runs"][1]["command"]


def test_compare_loss_curves_reads_pymsst_csv_and_original_checkpoint(tmp_path):
    pymsst_csv = tmp_path / "loss_curve.csv"
    with pymsst_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event", "global_step", "epoch", "batch_idx", "loss", "val_loss", "val_sdr"))
        writer.writeheader()
        writer.writerow({"event": "train", "global_step": 1, "epoch": 0, "batch_idx": 0, "loss": 0.5})
        writer.writerow({"event": "validation", "global_step": 1, "epoch": 0, "batch_idx": "", "loss": "", "val_loss": 0.4})
        writer.writerow({"event": "train", "global_step": 2, "epoch": 0, "batch_idx": 1, "loss": 0.25})

    original_ckpt = tmp_path / "original.ckpt"
    torch.save({"all_losses": {"epoch_0": [0.4, 0.2]}}, original_ckpt)

    result = compare_loss_curves(pymsst_csv, original_ckpt)

    assert read_pymsst_loss_curve(pymsst_csv) == [0.5, 0.25]
    assert json.loads(json.dumps(result.__dict__))["compared_points"] == 2
    assert result.mean_abs_diff == approx(0.075)


def test_compare_loss_curves_accepts_direct_original_csv(tmp_path):
    pymsst_csv = tmp_path / "pymsst.csv"
    original_csv = tmp_path / "original.csv"
    fieldnames = ("event", "global_step", "epoch", "batch_idx", "loss", "val_loss", "val_sdr")
    for path, losses in ((pymsst_csv, (1.0, 2.0)), (original_csv, (1.25, 1.75))):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, loss in enumerate(losses):
                writer.writerow({"event": "train", "global_step": index + 1, "epoch": 0, "batch_idx": index, "loss": loss})

    result = compare_loss_curves(pymsst_csv, original_csv)

    assert result.compared_points == 2
    assert result.mean_abs_diff == approx(0.25)


def test_direct_compare_overrides_make_original_metadata_deterministic():
    config = SimpleNamespace(
        audio=SimpleNamespace(chunk_size=256),
        training=SimpleNamespace(
            batch_size=4,
            num_epochs=10,
            num_steps=10,
            read_metadata_procs=8,
        ),
    )

    _apply_overrides(config, steps=3, batch_size=2, chunk_size=128)

    assert config.training.num_epochs == 1
    assert config.training.num_steps == 3
    assert config.training.batch_size == 2
    assert config.training.read_metadata_procs == 1
    assert config.audio.chunk_size == 128


def test_summarize_comparisons_aggregates_thresholds(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for path, mean_diff, max_diff in ((first, 0.1, 0.2), (second, 0.05, 0.15)):
        path.write_text(
            json.dumps(
                {
                    "pymsst_points": 1000,
                    "original_points": 1000,
                    "compared_points": 1000,
                    "mean_abs_diff": mean_diff,
                    "max_abs_diff": max_diff,
                    "first_pymsst_loss": 1.0,
                    "first_original_loss": 1.0,
                    "last_pymsst_loss": 0.5,
                    "last_original_loss": 0.5,
                }
            ),
            encoding="utf-8",
        )

    passing = summarize_comparisons(
        [first, second],
        min_points=1000,
        max_mean_abs_diff=0.1,
        max_max_abs_diff=0.2,
    )
    failing = summarize_comparisons(
        [first, second],
        min_points=1001,
        max_mean_abs_diff=0.09,
        max_max_abs_diff=0.19,
    )

    assert passing["aggregate"]["passed"] is True
    assert passing["aggregate"]["count"] == 2
    assert passing["aggregate"]["max_mean_abs_diff"] == approx(0.1)
    assert passing["aggregate"]["max_max_abs_diff"] == approx(0.2)
    assert failing["aggregate"]["passed"] is False
    assert len(failing["failures"]) == 2
