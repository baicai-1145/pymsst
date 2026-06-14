from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pymss_core import get_model_from_config, load_config, load_model_weights

from .losses import compute_loss, signal_to_distortion_ratio


class SeparationLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model_type: str,
        config_path: str | Path,
        learning_rate: float | None = None,
        loss_name: str = "masked_loss",
        q: float | None = None,
        coarse_loss_clip: bool | None = None,
        original_loss_compat: bool = False,
        start_checkpoint: str | Path | None = None,
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if model_type not in {"bs_roformer", "mel_band_roformer"}:
            raise ValueError("model_type must be 'bs_roformer' or 'mel_band_roformer'")

        self.model_type = model_type
        self.config_path = str(config_path)
        model_kwargs_override = _model_kwargs_override(model_type, config_path)
        self.model, self.config = get_model_from_config(model_type, config_path, model_kwargs_override=model_kwargs_override)
        for key, value in model_kwargs_override.items():
            self.config.model[key] = value
        self.learning_rate = float(learning_rate if learning_rate is not None else self.config.training.lr)
        self.loss_name = loss_name
        self.q = float(q if q is not None else self.config.training.get("q", 0.95))
        self.coarse_loss_clip = bool(
            coarse_loss_clip if coarse_loss_clip is not None else self.config.training.get("coarse_loss_clip", True)
        )
        self.original_loss_compat = bool(original_loss_compat)

        if start_checkpoint:
            load_model_weights(self.model, start_checkpoint, model_type=model_type, strict=strict_checkpoint)

        self.save_hyperparameters(
            {
                "model_type": self.model_type,
                "config_path": self.config_path,
                "learning_rate": self.learning_rate,
                "loss_name": self.loss_name,
                "q": self.q,
                "coarse_loss_clip": self.coarse_loss_clip,
                "original_loss_compat": self.original_loss_compat,
                "start_checkpoint": str(start_checkpoint) if start_checkpoint else None,
                "strict_checkpoint": strict_checkpoint,
            }
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return self.model(mixture)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        target, mixture = batch
        prediction = self(mixture)
        loss_prediction = self._prediction_for_loss(prediction, target)
        loss = compute_loss(
            loss_prediction,
            target,
            loss_name=self.loss_name,
            q=self.q,
            coarse=self.coarse_loss_clip,
            original_shape_compat=self.original_loss_compat,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        target, mixture = batch
        prediction = self(mixture)
        loss_prediction = self._prediction_for_loss(prediction, target)
        loss = compute_loss(
            loss_prediction,
            target,
            loss_name=self.loss_name,
            q=self.q,
            coarse=self.coarse_loss_clip,
            original_shape_compat=self.original_loss_compat,
        )
        sdr = signal_to_distortion_ratio(prediction, target).mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/sdr", sdr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_sdr", sdr, on_epoch=True, sync_dist=True)
        return {"val_loss": loss.detach(), "val_sdr": sdr.detach()}

    def _prediction_for_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if (
            self.original_loss_compat
            and self.model_type == "bs_roformer"
            and prediction.ndim == 3
            and target.ndim == 4
            and target.shape[1] == 1
        ):
            return prediction[:, None]
        return prediction

    def configure_optimizers(self) -> Any:
        optimizer_name = str(self.config.training.get("optimizer", "adam")).lower()
        if optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        if optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        raise ValueError(f"unsupported optimizer for first-stage pymsst training: {optimizer_name}")


def _model_kwargs_override(model_type: str, config_path: str | Path) -> dict[str, object]:
    if model_type != "mel_band_roformer":
        return {}
    config = load_config(config_path)
    if "mlp_hidden_layers" in config.model:
        return {}
    return {"mlp_hidden_layers": int(config.model.get("mask_estimator_depth", 1))}
