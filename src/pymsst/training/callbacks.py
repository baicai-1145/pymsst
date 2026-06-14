from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class LossCurveRecorder(Callback):
    fieldnames = ("event", "global_step", "epoch", "batch_idx", "loss", "val_loss", "val_sdr")

    def __init__(self, path: str | Path, interval: int = 1) -> None:
        self.path = Path(path)
        self.interval = int(interval)
        if self.interval < 0:
            raise ValueError("loss curve recording interval must be >= 0")

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        if not self.interval or (batch_idx + 1) % self.interval != 0:
            return
        loss = _to_float(trainer.callback_metrics.get("train/loss"))
        if loss is None:
            loss = _extract_loss(outputs)
        if loss is None:
            return
        self._append(
            {
                "event": "train",
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "batch_idx": int(batch_idx),
                "loss": loss,
                "val_loss": "",
                "val_sdr": "",
            }
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        val_loss = _to_float(trainer.callback_metrics.get("val/loss"))
        val_sdr = _to_float(trainer.callback_metrics.get("val_sdr"))
        if val_loss is None and val_sdr is None:
            return
        self._append(
            {
                "event": "validation",
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "batch_idx": "",
                "loss": "",
                "val_loss": "" if val_loss is None else val_loss,
                "val_sdr": "" if val_sdr is None else val_sdr,
            }
        )

    def _append(self, row: dict[str, object]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(row)


def _extract_loss(outputs: Any) -> float | None:
    if isinstance(outputs, torch.Tensor):
        return _to_float(outputs)
    if isinstance(outputs, dict):
        return _to_float(outputs.get("loss"))
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
