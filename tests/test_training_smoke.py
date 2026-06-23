from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from pymss_core import get_model_from_config, load_model_weights

from pymsst.export import export_core_checkpoint
from pymsst.training.fused_extension import FusedKernelUnavailable
from pymsst.training.fused_roformer import (
    _MaskFinalGroupedLinearGLU,
    _band_split_group_param_refs,
    _copy_band_group,
    _contiguous_index_range,
    _mask_layer_param_refs,
    _packed_mask_layer_param_refs,
    _select_band_group,
    _stack_converted_tensors,
    apply_roformer_fused_training_backend,
    normalize_fused_rms_norm_mode,
)
from pymsst.training.callbacks import ExponentialMovingAverage
from pymsst.training import SeparationLightningModule, compute_loss, masked_loss


def _reference_grouped_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    group_count, _, in_features = weight.shape
    leading_shape = x.shape[:-2]
    x = x.reshape(-1, group_count, in_features).transpose(0, 1)
    weight = weight.transpose(1, 2)
    if bias is None:
        out = torch.bmm(x, weight)
    else:
        out = torch.baddbmm(bias.unsqueeze(1).expand(-1, x.shape[1], -1), x, weight)
    return out.transpose(0, 1).reshape(*leading_shape, group_count, -1)


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

    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    assert configured["lr_scheduler"]["monitor"] == "val_sdr"

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


def test_ema_callback_tracks_module_model(tmp_path):
    module = SeparationLightningModule(
        model_type="bs_roformer",
        config_path=_write_config(tmp_path, "bs_roformer"),
    )
    callback = ExponentialMovingAverage(0.9)
    trainer = type("Trainer", (), {})()
    callback.on_fit_start(trainer, module)
    assert callback.ema_model is not None
    assert module.ema_model is callback.ema_model
    assert module.ema_decay == 0.9


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


def test_cuda_attention_backend_defaults_to_torch_default_for_training(tmp_path):
    module = SeparationLightningModule(model_type="mel_band_roformer", config_path=_write_config(tmp_path, "mel_band_roformer"))

    configured = [
        item.cuda_attention_backend for item in module.model.modules() if hasattr(item, "cuda_attention_backend")
    ]

    assert configured
    assert module.cuda_attention_backend == "default"
    assert set(configured) == {"default"}
    assert module.config.training.cuda_attention_backend == "default"


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


def test_band_split_bwd_v3_env_takes_precedence_over_v2(tmp_path, monkeypatch):
    class FakeExtension:
        def band_split_backward_v2(self):
            raise AssertionError("not called")

        def band_split_backward_v3(self):
            raise AssertionError("not called")

    monkeypatch.setenv("PYMSST_EXPERIMENTAL_BANDSPLIT_BWD_V2", "1")
    monkeypatch.setenv("PYMSST_EXPERIMENTAL_BANDSPLIT_BWD_V3", "1")
    module = SeparationLightningModule(model_type="bs_roformer", config_path=_write_config(tmp_path, "bs_roformer"))

    status = apply_roformer_fused_training_backend(module.model, "cuda_ext", extension=FakeExtension())

    assert status.active == "cuda_ext"
    assert status.patched_band_splits == 1
    assert status.patched_band_split_bwd_v2 is False
    assert status.patched_band_split_bwd_v3 is True
    assert module.model.band_split._pymsst_fused_band_split_bwd_mode == "v3"


def test_fused_kernel_backend_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="fused_kernel_backend"):
        SeparationLightningModule(
            model_type="bs_roformer",
            config_path=_write_config(tmp_path, "bs_roformer"),
            fused_kernel_backend="unknown",
        )


def test_layout_reduce_copy_helpers_slice_contiguous_band_indices():
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    index_calls = 0

    def index_tensor(indices, device):
        nonlocal index_calls
        index_calls += 1
        return torch.as_tensor(indices, device=device)

    assert _contiguous_index_range((1, 2)) == (1, 3)

    selected = _select_band_group(x, (1, 2), x.device, index_tensor)

    assert index_calls == 0
    assert torch.equal(selected, x[..., 1:3, :])

    destination = torch.zeros_like(x)
    _copy_band_group(destination, (1, 2), selected, x.device, index_tensor)

    assert index_calls == 0
    assert torch.equal(destination[..., 1:3, :], selected)


def test_layout_reduce_copy_helpers_keep_index_select_for_noncontiguous_band_indices():
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    index_calls = 0

    def index_tensor(indices, device):
        nonlocal index_calls
        index_calls += 1
        return torch.as_tensor(indices, device=device)

    assert _contiguous_index_range((0, 2)) is None

    selected = _select_band_group(x, (0, 2), x.device, index_tensor)

    assert index_calls == 1
    assert torch.equal(selected, x.index_select(-2, torch.tensor((0, 2))))


def test_layout_reduce_copy_deep_mask_grouping_matches_original(tmp_path, monkeypatch):
    import pymss_core.modules.bs_roformer.bands as bands

    config_path = _write_config(tmp_path, "mel_band_roformer")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["mask_estimator_depth"] = 2
    config["model"]["mlp_hidden_layers"] = 2
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setenv("PYMSS_CORE_EXPERIMENTAL_DEEP_MASK_GROUPING", "1")
    monkeypatch.setenv("PYMSST_EXPERIMENTAL_LAYOUT_REDUCE_COPY", "1")
    original_module = SeparationLightningModule(model_type="mel_band_roformer", config_path=config_path)
    original_forward = getattr(
        bands.MaskEstimator._forward_layer_grouped_mlp,
        "_pymsst_original_forward_layer_grouped_mlp",
        bands.MaskEstimator._forward_layer_grouped_mlp,
    )
    patched_module = SeparationLightningModule(model_type="mel_band_roformer", config_path=config_path)
    patched_module.model.load_state_dict(original_module.model.state_dict())

    status = apply_roformer_fused_training_backend(patched_module.model, "cuda_ext", extension=object())

    estimator = patched_module.model.mask_estimators[0]
    x = torch.randn(2, len(estimator.dim_inputs), int(patched_module.config.model.dim))
    reference = original_forward(original_module.model.mask_estimators[0], x)
    candidate = estimator._forward_layer_grouped_mlp(x)

    assert status.patched_layout_reduce_copy is True
    assert reference is not None
    assert candidate is not None
    assert torch.equal(candidate, reference)


def test_train_param_pack_autograd_matches_stack_to(monkeypatch):
    tensors = [torch.randn(2, 3, dtype=torch.float32, requires_grad=True) for _ in range(3)]
    reference_tensors = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    grad = torch.randn(3, 2, 3, dtype=torch.float16)

    monkeypatch.setenv("PYMSST_EXPERIMENTAL_TRAIN_PARAM_PACK_AUTOGRAD", "1")

    packed = _stack_converted_tensors(tensors, torch.device("cpu"), torch.float16)
    reference = torch.stack([tensor.to(dtype=torch.float16) for tensor in reference_tensors], dim=0)
    packed.backward(grad)
    reference.backward(grad)

    assert torch.equal(packed, reference)
    for tensor, reference_tensor in zip(tensors, reference_tensors):
        assert torch.equal(tensor.grad, reference_tensor.grad)


def test_mask_final_autograd_status_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr("pymsst.training.fused_roformer.load_fused_band_split_extension", lambda: object())
    monkeypatch.setenv("PYMSST_EXPERIMENTAL_LAYOUT_REDUCE_COPY", "1")
    monkeypatch.setenv("PYMSST_EXPERIMENTAL_MASK_FINAL_AUTOGRAD", "1")
    module = SeparationLightningModule(
        model_type="mel_band_roformer",
        config_path=_write_config(tmp_path, "mel_band_roformer"),
        fused_kernel_backend="cuda_ext",
    )

    assert module.fused_training_status.patched_layout_reduce_copy is True
    assert module.fused_training_status.patched_mask_final_autograd is True
    assert module.config.training.fused_kernel_patched_mask_final_autograd is True
    assert module.hparams["fused_kernel_patched_mask_final_autograd"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capability check requires a CUDA device")
def test_mask_estimator_final_grouped_is_disabled_on_sm86(monkeypatch):
    from pymsst.training.fused_roformer import _experimental_mask_estimator_final_mode, _patch_core_mask_estimator_final

    class DummyExtension:
        mask_estimator_final_forward = object()
        mask_estimator_final_backward = object()
        mask_estimator_final_direct_forward = object()
        mask_estimator_final_direct_backward = object()

    monkeypatch.setenv("PYMSST_EXPERIMENTAL_MASK_ESTIMATOR_FINAL_FUSED", "1")
    mode = _experimental_mask_estimator_final_mode(DummyExtension())
    assert mode == "grouped"
    if torch.cuda.get_device_capability()[0] < 9:
        assert _patch_core_mask_estimator_final(DummyExtension(), mode) is False


def test_mask_final_grouped_linear_glu_matches_reference():
    x = torch.randn(2, 5, 3, 4, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(3, 6, 4, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(3, 6, dtype=torch.float32, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    ref_bias = bias.detach().clone().requires_grad_()
    grad = torch.randn(2, 5, 3, 3)

    out = _MaskFinalGroupedLinearGLU.apply(x, weight, bias)
    ref = torch.nn.functional.glu(_reference_grouped_linear(ref_x, ref_weight, ref_bias), dim=-1)
    out.backward(grad)
    ref.backward(grad)

    assert torch.allclose(out, ref)
    assert torch.allclose(x.grad, ref_x.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(bias.grad, ref_bias.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast-style dtype check requires CUDA")
def test_mask_final_grouped_linear_glu_accepts_float_grad_for_half_inputs():
    x = torch.randn(2, 5, 3, 4, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(3, 6, 4, device="cuda", dtype=torch.float16, requires_grad=True)
    bias = torch.randn(3, 6, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(2, 5, 3, 3, device="cuda", dtype=torch.float32)

    out = _MaskFinalGroupedLinearGLU.apply(x, weight, bias)
    out.backward(grad)

    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
    assert x.grad.dtype == torch.float16
    assert weight.grad.dtype == torch.float16
    assert bias.grad.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast-style dtype check requires CUDA")
def test_mask_final_grouped_linear_glu_matches_reference_under_half_inputs():
    x = torch.randn(2, 5, 3, 4, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(3, 6, 4, device="cuda", dtype=torch.float16, requires_grad=True)
    bias = torch.randn(3, 6, device="cuda", dtype=torch.float16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    ref_bias = bias.detach().clone().requires_grad_()
    grad = torch.randn(2, 5, 3, 3, device="cuda", dtype=torch.float32)

    out = _MaskFinalGroupedLinearGLU.apply(x, weight, bias)
    ref = torch.nn.functional.glu(_reference_grouped_linear(ref_x, ref_weight, ref_bias), dim=-1)
    out.backward(grad)
    ref.backward(grad)

    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(x.grad, ref_x.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(bias.grad, ref_bias.grad, atol=1e-2, rtol=1e-2)


def test_layout_reduce_copy_parameter_reference_caches_are_stable(tmp_path):
    module = SeparationLightningModule(model_type="mel_band_roformer", config_path=_write_config(tmp_path, "mel_band_roformer"))
    estimator = module.model.mask_estimators[0]
    apply_roformer_fused_training_backend(module.model, "cuda_ext", extension=object())

    band_refs_1 = _band_split_group_param_refs(module.model.band_split, 0, 2)
    band_refs_2 = _band_split_group_param_refs(module.model.band_split, 0, 2)
    mask_refs_1 = _mask_layer_param_refs(estimator, 0, (0, 1))
    mask_refs_2 = _mask_layer_param_refs(estimator, 0, (0, 1))
    packed_refs_1 = _packed_mask_layer_param_refs(estimator, (estimator,), 0, (0, 1))
    packed_refs_2 = _packed_mask_layer_param_refs(estimator, (estimator,), 0, (0, 1))

    assert band_refs_1 is band_refs_2
    assert mask_refs_1 is mask_refs_2
    assert packed_refs_1 is packed_refs_2
    assert len(band_refs_1[0]) == 2
    assert len(mask_refs_1[0]) == 2
    assert len(packed_refs_1[0]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension requires a CUDA device")
def test_band_split_ragged_backward_matches_per_band_extension(monkeypatch):
    from pymsst.training.fused_extension import load_fused_band_split_extension

    monkeypatch.setenv("PYMSST_EXPERIMENTAL_BANDSPLIT_RAGGED_FORWARD", "1")
    monkeypatch.setenv("PYMSST_EXPERIMENTAL_BANDSPLIT_RAGGED_BACKWARD", "1")

    ext = load_fused_band_split_extension()
    if not hasattr(ext, "band_split_ragged_backward"):
        pytest.skip("extension does not expose ragged backward")

    torch.manual_seed(7)
    m_count = 5
    dim_inputs = (3, 5, 2, 7)
    out_features = 4
    x = torch.randn(m_count, sum(dim_inputs), device="cuda", dtype=torch.float16)
    gammas = [torch.randn(dim_in, device="cuda", dtype=torch.float16) for dim_in in dim_inputs]
    weights = [torch.randn(out_features, dim_in, device="cuda", dtype=torch.float16) for dim_in in dim_inputs]
    biases = [torch.randn(out_features, device="cuda", dtype=torch.float16) for _ in dim_inputs]
    offsets = torch.tensor((0, 3, 8, 10, 17), device="cuda", dtype=torch.long)

    out, normed, inv_norm = ext.band_split_ragged_forward(x, gammas, weights, biases, list(dim_inputs), offsets)
    grad_out = torch.randn_like(out)
    ragged = ext.band_split_ragged_backward(grad_out, normed, inv_norm, gammas, weights, list(dim_inputs), offsets)

    grad_x = torch.zeros_like(x)
    ref_gammas = []
    ref_weights = []
    ref_biases = []
    for band_index, dim_in in enumerate(dim_inputs):
        start = int(offsets[band_index].item())
        end = int(offsets[band_index + 1].item())
        grad_band = grad_out[:, band_index : band_index + 1, :].contiguous()
        normed_band = normed[:, start:end].reshape(m_count, 1, dim_in).contiguous()
        inv_band = inv_norm[:, band_index : band_index + 1].contiguous()
        gamma = gammas[band_index].reshape(1, dim_in).contiguous()
        weight = weights[band_index].reshape(1, out_features, dim_in).contiguous()
        grad_band_x, grad_gamma, grad_weight, grad_bias = ext.band_split_backward_v3(
            grad_band,
            normed_band,
            inv_band,
            gamma,
            weight,
            True,
        )
        grad_x[:, start:end] = grad_band_x.reshape(m_count, dim_in)
        ref_gammas.append(grad_gamma[0])
        ref_weights.append(grad_weight[0])
        ref_biases.append(grad_bias[0])

    assert torch.equal(ragged[0], grad_x)
    base = 1
    for band_index in range(len(dim_inputs)):
        assert torch.equal(ragged[base + band_index], ref_gammas[band_index])
    base += len(dim_inputs)
    for band_index in range(len(dim_inputs)):
        assert torch.equal(ragged[base + band_index], ref_weights[band_index])
    base += len(dim_inputs)
    for band_index in range(len(dim_inputs)):
        assert torch.equal(ragged[base + band_index], ref_biases[band_index])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension requires a CUDA device")
def test_mask_estimator_final_direct_matches_reference():
    from pymsst.training.fused_extension import load_fused_band_split_extension

    ext = load_fused_band_split_extension()
    if not hasattr(ext, "mask_estimator_final_direct_forward"):
        pytest.skip("extension does not expose direct mask final kernels")

    torch.manual_seed(11)
    m_count = 4
    band_count = 5
    hidden = 16
    dim_inputs = (3, 4, 2, 5, 6)
    x = torch.randn(m_count, band_count, hidden, device="cuda", dtype=torch.float16)
    weights = [torch.randn(dim_in * 2, hidden, device="cuda", dtype=torch.float16) for dim_in in dim_inputs]
    bias = torch.randn(sum(dim_inputs) * 2, device="cuda", dtype=torch.float16)
    offsets = torch.tensor((0, 3, 7, 9, 14, 20), device="cuda", dtype=torch.long)

    out, pre, x_saved = ext.mask_estimator_final_direct_forward(x, weights, bias, list(dim_inputs), offsets)
    grad_out = torch.randn_like(out)
    result = ext.mask_estimator_final_direct_backward(grad_out, x_saved, pre, weights, list(dim_inputs), offsets)

    ref_x = x.detach().clone().requires_grad_()
    ref_weights = [weight.detach().clone().requires_grad_() for weight in weights]
    ref_bias = bias.detach().clone().requires_grad_()
    outs = []
    for band_index, dim_in in enumerate(dim_inputs):
        start = int(offsets[band_index].item())
        pre_ref = torch.nn.functional.linear(
            ref_x[:, band_index, :],
            ref_weights[band_index],
            ref_bias[start * 2 : start * 2 + dim_in * 2],
        )
        outs.append(torch.nn.functional.glu(pre_ref, dim=-1))
    ref = torch.cat(outs, dim=-1)
    ref.backward(grad_out)

    assert torch.allclose(out, ref, atol=8e-3, rtol=1e-3)
    assert torch.allclose(result[0], ref_x.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(result[1], ref_bias.grad, atol=2e-3, rtol=1e-3)
    for band_index, ref_weight in enumerate(ref_weights):
        assert torch.allclose(result[2 + band_index], ref_weight.grad, atol=2e-3, rtol=1e-3)


def test_fused_rms_norm_mode_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="fused_rms_norm_mode"):
        SeparationLightningModule(
            model_type="bs_roformer",
            config_path=_write_config(tmp_path, "bs_roformer"),
            fused_rms_norm_mode="unknown",
        )


def test_rms_norm_bwd_v2_env_enables_fast_rms_norm_mode(monkeypatch):
    monkeypatch.delenv("PYMSST_EXPERIMENTAL_FUSED_RMSNORM", raising=False)
    monkeypatch.delenv("PYMSST_EXPERIMENTAL_RMSNORM_BWD_V2", raising=False)

    assert normalize_fused_rms_norm_mode(None) is None

    monkeypatch.setenv("PYMSST_EXPERIMENTAL_RMSNORM_BWD_V2", "1")

    assert normalize_fused_rms_norm_mode(None) == "fast"
    assert normalize_fused_rms_norm_mode("off") is None


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
