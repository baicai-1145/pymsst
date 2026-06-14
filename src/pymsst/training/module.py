from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pymss_core import get_model_from_config, load_config, load_model_weights

from .cuda_graph import CudaGraphRuntimeStatus, RoformerMaskCoreCudaGraph, normalize_cuda_graph_mode
from .fused_roformer import (
    apply_roformer_fused_training_backend,
    normalize_fused_kernel_backend,
    normalize_fused_rms_norm_mode,
)
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
        optimizer_backend: str | None = None,
        cuda_attention_backend: str | None = None,
        fused_kernel_backend: str | None = None,
        fused_rms_norm_mode: str | None = None,
        cuda_graph_mode: str | None = None,
        train_log_interval: int = 1,
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
        self.optimizer_backend = _normalize_optimizer_backend(
            optimizer_backend if optimizer_backend is not None else self.config.training.get("optimizer_backend", "default")
        )
        self.cuda_attention_backend = _optional_string(
            cuda_attention_backend if cuda_attention_backend is not None else self.config.training.get("cuda_attention_backend")
        )
        self.config.training.optimizer_backend = self.optimizer_backend
        if self.cuda_attention_backend is not None:
            _set_cuda_attention_backend(self.model, self.cuda_attention_backend)
            self.config.training.cuda_attention_backend = self.cuda_attention_backend
        self.fused_kernel_backend = normalize_fused_kernel_backend(
            fused_kernel_backend if fused_kernel_backend is not None else self.config.training.get("fused_kernel_backend")
        )
        self.fused_rms_norm_mode = normalize_fused_rms_norm_mode(
            fused_rms_norm_mode if fused_rms_norm_mode is not None else self.config.training.get("fused_rms_norm_mode")
        )
        self.fused_training_status = apply_roformer_fused_training_backend(
            self.model,
            self.fused_kernel_backend,
            rms_norm_mode=self.fused_rms_norm_mode,
        )
        self.config.training.fused_kernel_backend = self.fused_kernel_backend
        self.config.training.fused_rms_norm_mode = self.fused_rms_norm_mode or "off"
        self.config.training.fused_kernel_active = self.fused_training_status.active
        self.config.training.fused_kernel_patched_band_splits = self.fused_training_status.patched_band_splits
        self.config.training.fused_kernel_patched_rotary = self.fused_training_status.patched_rotary
        self.config.training.fused_kernel_patched_rms_norm = self.fused_training_status.patched_rms_norm
        self.config.training.fused_kernel_rms_norm_mode = self.fused_training_status.rms_norm_mode
        self.config.training.fused_kernel_patched_single_stem_masks = self.fused_training_status.patched_single_stem_masks
        self.config.training.fused_kernel_patched_attention_gates = self.fused_training_status.patched_attention_gates
        self.config.training.fused_kernel_attention_gate_mode = self.fused_training_status.attention_gate_mode
        if self.fused_training_status.reason:
            self.config.training.fused_kernel_reason = self.fused_training_status.reason
        self.cuda_graph_mode = normalize_cuda_graph_mode(
            cuda_graph_mode if cuda_graph_mode is not None else self.config.training.get("cuda_graph_mode")
        )
        self.cuda_graph_status = CudaGraphRuntimeStatus(requested=self.cuda_graph_mode, active="off")
        self._cuda_graph_runner: RoformerMaskCoreCudaGraph | None = None
        self.config.training.cuda_graph_mode = self.cuda_graph_mode
        self.config.training.cuda_graph_active = self.cuda_graph_status.active
        self.train_log_interval = int(train_log_interval)
        if self.train_log_interval < 0:
            raise ValueError("train_log_interval must be >= 0")
        self.config.training.train_log_interval = self.train_log_interval

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
                "optimizer_backend": self.optimizer_backend,
                "cuda_attention_backend": self.cuda_attention_backend,
                "fused_kernel_backend": self.fused_kernel_backend,
                "fused_rms_norm_mode": self.fused_rms_norm_mode,
                "fused_kernel_active": self.fused_training_status.active,
                "fused_kernel_patched_band_splits": self.fused_training_status.patched_band_splits,
                "fused_kernel_patched_rotary": self.fused_training_status.patched_rotary,
                "fused_kernel_patched_rms_norm": self.fused_training_status.patched_rms_norm,
                "fused_kernel_rms_norm_mode": self.fused_training_status.rms_norm_mode,
                "fused_kernel_patched_single_stem_masks": self.fused_training_status.patched_single_stem_masks,
                "fused_kernel_patched_attention_gates": self.fused_training_status.patched_attention_gates,
                "fused_kernel_attention_gate_mode": self.fused_training_status.attention_gate_mode,
                "cuda_graph_mode": self.cuda_graph_mode,
                "cuda_graph_active": self.cuda_graph_status.active,
                "train_log_interval": self.train_log_interval,
            }
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if self.training and self.cuda_graph_mode != "off" and mixture.device.type == "cuda":
            return self._cuda_graph_forward(mixture)
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
        if self.train_log_interval and (batch_idx + 1) % self.train_log_interval == 0:
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
        optimizer_kwargs = _optimizer_backend_kwargs(self.optimizer_backend)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, **optimizer_kwargs)
        if optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, **optimizer_kwargs)
        raise ValueError(f"unsupported optimizer for first-stage pymsst training: {optimizer_name}")

    def _cuda_graph_forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if self._cuda_graph_runner is None:
            self._cuda_graph_runner = RoformerMaskCoreCudaGraph(self.model, self.cuda_graph_mode)

        runner = self._cuda_graph_runner
        use_amp = torch.is_autocast_enabled("cuda")
        if runner.status.active == "off" and runner.status.reason is None:
            self.cuda_graph_status = runner.capture(mixture, use_amp=use_amp)
            self._record_cuda_graph_status()

        prediction = runner.forward(mixture, use_amp=use_amp)
        self.cuda_graph_status = runner.status
        self._record_cuda_graph_status()
        return prediction

    def _record_cuda_graph_status(self) -> None:
        self.config.training.cuda_graph_active = self.cuda_graph_status.active
        if self.cuda_graph_status.reason:
            self.config.training.cuda_graph_reason = self.cuda_graph_status.reason


def _model_kwargs_override(model_type: str, config_path: str | Path) -> dict[str, object]:
    if model_type != "mel_band_roformer":
        return {}
    config = load_config(config_path)
    if "mlp_hidden_layers" in config.model:
        return {}
    return {"mlp_hidden_layers": int(config.model.get("mask_estimator_depth", 1))}


def _normalize_optimizer_backend(value: str) -> str:
    backend = str(value or "default").lower().replace("-", "_")
    if backend in {"default", "torch"}:
        return "default"
    if backend in {"foreach", "for_each"}:
        return "foreach"
    if backend == "fused":
        return "fused"
    if backend in {"fused_capturable", "capturable_fused"}:
        return "fused_capturable"
    raise ValueError("optimizer_backend must be one of: default, foreach, fused, fused_capturable")


def _optimizer_backend_kwargs(backend: str) -> dict[str, bool]:
    if backend == "default":
        return {}
    if backend == "foreach":
        return {"foreach": True}
    if backend == "fused":
        return {"fused": True}
    if backend == "fused_capturable":
        return {"fused": True, "capturable": True}
    raise ValueError("optimizer_backend must be one of: default, foreach, fused, fused_capturable")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _set_cuda_attention_backend(model: torch.nn.Module, backend: str) -> None:
    matched = False
    for module in model.modules():
        setter = getattr(module, "set_cuda_attention_backend", None)
        if setter is None:
            continue
        setter(backend)
        matched = True
    if not matched:
        raise ValueError("model does not expose set_cuda_attention_backend")
