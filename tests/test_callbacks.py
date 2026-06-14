from __future__ import annotations

import csv
from types import SimpleNamespace

import torch

from pymsst.training.callbacks import LossCurveRecorder


def test_loss_curve_recorder_prefers_logged_train_loss(tmp_path):
    path = tmp_path / "loss_curve.csv"
    recorder = LossCurveRecorder(path)
    trainer = SimpleNamespace(
        is_global_zero=True,
        callback_metrics={"train/loss": torch.tensor(8.0)},
        global_step=1,
        current_epoch=0,
    )

    recorder.on_fit_start(trainer, None)
    recorder.on_train_batch_end(trainer, None, torch.tensor(1.0), None, batch_idx=0)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["loss"] == "8.0"


def test_loss_curve_recorder_respects_interval(tmp_path):
    path = tmp_path / "loss_curve.csv"
    recorder = LossCurveRecorder(path, interval=2)
    trainer = SimpleNamespace(
        is_global_zero=True,
        callback_metrics={"train/loss": torch.tensor(8.0)},
        global_step=1,
        current_epoch=0,
    )

    recorder.on_fit_start(trainer, None)
    recorder.on_train_batch_end(trainer, None, torch.tensor(1.0), None, batch_idx=0)
    recorder.on_train_batch_end(trainer, None, torch.tensor(1.0), None, batch_idx=1)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["batch_idx"] == "1"
