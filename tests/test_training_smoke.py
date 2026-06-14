from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from pymss_core import get_model_from_config, load_model_weights

from pymsst.export import export_core_checkpoint
from pymsst.training.fused_extension import FusedKernelUnavailable
from pymsst.training.fused_roformer import apply_roformer_fused_training_backend
from pymsst.training import SeparationLightningModule, compute_loss, masked_loss


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


def _write_config(tmp_path: Path, model_type: str) -> Path:
    path = tmp_path / f"{model_type}.yaml"
    path.write_text(yaml.safe_dump(_tiny_config(model_type)), encoding="utf-8")
    return path


def test_bs_and_mel_band_roformer_forward_backward(tmp_path):
    mixture = torch.randn(1, 2, 256)
    target = torch.randn(1, 2, 256)

    for model_type in ("bs_roformer", "mel_band_roformer"):
        module = SeparationLightningModule(model_type=model_type, config_path=_write_config(tmp_path, model_type))
        prediction = module(mixture)
        loss = compute_loss(prediction, target, loss_name="masked_loss")
        loss.backward()

        assert prediction.shape == target.shape
        assert torch.isfinite(loss)


def test_exported_checkpoint_loads_with_pymss_core(tmp_path):
    config_path = _write_config(tmp_path, "bs_roformer")
    module = SeparationLightningModule(model_type="bs_roformer", config_path=config_path)
    export_path = tmp_path / "core.ckpt"

    export_core_checkpoint(module.model, export_path, model_type="bs_roformer", config=module.config)
    fresh_model, _ = get_model_from_config("bs_roformer", config_path)
    result = load_model_weights(fresh_model, export_path, model_type="bs_roformer", strict=True)

    assert export_path.is_file()
    assert not result.missing_keys
    assert not result.unexpected_keys


def test_mel_band_roformer_legacy_mlp_depth_compatibility(tmp_path):
    config = _tiny_config("mel_band_roformer")
    config["model"]["mask_estimator_depth"] = 2
    legacy_path = tmp_path / "legacy_mel_band_roformer.yaml"
    legacy_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    legacy_module = SeparationLightningModule(model_type="mel_band_roformer", config_path=legacy_path)
    legacy_mlp = legacy_module.model.mask_estimators[0].to_freqs[0][0]
    legacy_linear_layers = [layer for layer in legacy_mlp if isinstance(layer, torch.nn.Linear)]

    assert len(legacy_linear_layers) == 3
    assert legacy_module.config.model.mlp_hidden_layers == 2

    config["model"]["mlp_hidden_layers"] = 1
    explicit_path = tmp_path / "explicit_mel_band_roformer.yaml"
    explicit_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    explicit_module = SeparationLightningModule(model_type="mel_band_roformer", config_path=explicit_path)
    explicit_mlp = explicit_module.model.mask_estimators[0].to_freqs[0][0]
    explicit_linear_layers = [layer for layer in explicit_mlp if isinstance(layer, torch.nn.Linear)]

    assert len(explicit_linear_layers) == 2


@pytest.mark.parametrize(
    ("optimizer_backend", "expected_default"),
    (("foreach", "foreach"), ("fused", "fused"), ("fused_capturable", "capturable")),
)
def test_optimizer_backend_configures_torch_adam(tmp_path, optimizer_backend, expected_default):
    module = SeparationLightningModule(
        model_type="bs_roformer",
        config_path=_write_config(tmp_path, "bs_roformer"),
        optimizer_backend=optimizer_backend,
    )

    optimizer = module.configure_optimizers()

    assert optimizer.defaults[expected_default] is True
    assert module.config.training.optimizer_backend == optimizer_backend
    if optimizer_backend == "fused_capturable":
        assert optimizer.defaults["fused"] is True


def test_optimizer_backend_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="optimizer_backend"):
        SeparationLightningModule(
            model_type="bs_roformer",
            config_path=_write_config(tmp_path, "bs_roformer"),
            optimizer_backend="unknown",
        )


def test_cuda_attention_backend_configures_core_attention_modules(tmp_path):
    module = SeparationLightningModule(
        model_type="mel_band_roformer",
        config_path=_write_config(tmp_path, "mel_band_roformer"),
        cuda_attention_backend="efficient",
    )

    configured = [
        item.cuda_attention_backend for item in module.model.modules() if hasattr(item, "cuda_attention_backend")
    ]

    assert configured
    assert set(configured) == {"efficient"}
    assert module.config.training.cuda_attention_backend == "efficient"


def test_cuda_attention_backend_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="cuda_attention_backend"):
        SeparationLightningModule(
            model_type="mel_band_roformer",
            config_path=_write_config(tmp_path, "mel_band_roformer"),
            cuda_attention_backend="unknown",
        )


def test_fused_kernel_backend_defaults_to_off(tmp_path):
    module = SeparationLightningModule(model_type="bs_roformer", config_path=_write_config(tmp_path, "bs_roformer"))

    assert module.fused_kernel_backend == "off"
    assert module.fused_rms_norm_mode is None
    assert module.fused_training_status.active == "off"
    assert module.config.training.fused_kernel_backend == "off"
    assert module.config.training.fused_rms_norm_mode == "off"
    assert module.config.training.fused_kernel_active == "off"
    assert module.cuda_graph_mode == "off"
    assert module.cuda_graph_status.active == "off"
    assert module.config.training.cuda_graph_mode == "off"
    assert module.config.training.cuda_graph_active == "off"


def test_fused_kernel_backend_falls_back_when_extension_is_unavailable(tmp_path, monkeypatch):
    def fail_load():
        raise FusedKernelUnavailable("no extension in this test")

    monkeypatch.setattr("pymsst.training.fused_roformer.load_fused_band_split_extension", fail_load)

    with pytest.warns(RuntimeWarning, match="no extension"):
        module = SeparationLightningModule(
            model_type="bs_roformer",
            config_path=_write_config(tmp_path, "bs_roformer"),
            fused_kernel_backend="cuda_ext",
        )

    assert module.fused_kernel_backend == "cuda_ext"
    assert module.fused_training_status.active == "off"
    assert "no extension" in module.fused_training_status.reason
    assert module.config.training.fused_kernel_active == "off"


def test_fused_kernel_backend_can_patch_band_split_without_building_extension(tmp_path):
    module = SeparationLightningModule(model_type="bs_roformer", config_path=_write_config(tmp_path, "bs_roformer"))

    status = apply_roformer_fused_training_backend(module.model, "cuda_ext", extension=object())

    assert status.active == "cuda_ext"
    assert status.patched_band_splits == 1
    assert module.model.band_split._pymsst_fused_band_split is True


def test_fused_kernel_backend_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="fused_kernel_backend"):
        SeparationLightningModule(
            model_type="bs_roformer",
            config_path=_write_config(tmp_path, "bs_roformer"),
            fused_kernel_backend="unknown",
        )


def test_fused_rms_norm_mode_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="fused_rms_norm_mode"):
        SeparationLightningModule(
            model_type="bs_roformer",
            config_path=_write_config(tmp_path, "bs_roformer"),
            fused_rms_norm_mode="unknown",
        )


def test_cuda_graph_mode_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="cuda_graph_mode"):
        SeparationLightningModule(
            model_type="bs_roformer",
            config_path=_write_config(tmp_path, "bs_roformer"),
            cuda_graph_mode="unknown",
        )


def test_cuda_graph_mode_accepts_stft_mask_core(tmp_path):
    module = SeparationLightningModule(
        model_type="bs_roformer",
        config_path=_write_config(tmp_path, "bs_roformer"),
        cuda_graph_mode="stft-mask-core",
    )

    assert module.cuda_graph_mode == "stft_mask_core"
    assert module.cuda_graph_status.active == "off"
    assert module.config.training.cuda_graph_mode == "stft_mask_core"


def test_masked_loss_original_shape_compatibility_matches_broadcasting():
    prediction = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    target = prediction[:, None].clone()

    fixed_loss = masked_loss(prediction, target)
    compat_loss = masked_loss(prediction, target, original_shape_compat=True)

    assert fixed_loss == 0
    assert compat_loss > 0


def test_bs_roformer_original_loss_compat_restores_stem_dim(tmp_path):
    module = SeparationLightningModule(
        model_type="bs_roformer",
        config_path=_write_config(tmp_path, "bs_roformer"),
        original_loss_compat=True,
    )
    prediction = torch.zeros(2, 2, 16)
    target = torch.zeros(2, 1, 2, 16)

    loss_prediction = module._prediction_for_loss(prediction, target)

    assert loss_prediction.shape == target.shape
