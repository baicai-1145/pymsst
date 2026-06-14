from __future__ import annotations

import os
import types
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .fused_extension import FusedKernelUnavailable, load_fused_band_split_extension


CORE_GROUPED_TRAIN_ENV = "PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS"
EXPERIMENTAL_GROUPED_LINEAR_ENV = "PYMSST_EXPERIMENTAL_FUSED_GROUPED_LINEAR"
EXPERIMENTAL_ROTARY_ENV = "PYMSST_EXPERIMENTAL_FUSED_ROTARY"
EXPERIMENTAL_RMS_NORM_ENV = "PYMSST_EXPERIMENTAL_FUSED_RMSNORM"
EXPERIMENTAL_ATTENTION_GATES_ENV = "PYMSST_EXPERIMENTAL_FUSED_ATTENTION_GATES"


@dataclass(frozen=True)
class FusedTrainingStatus:
    requested: str
    active: str
    reason: str | None = None
    patched_band_splits: int = 0
    patched_rotary: bool = False
    patched_rms_norm: bool = False
    rms_norm_mode: str | None = None
    patched_single_stem_masks: bool = False
    patched_attention_gates: bool = False
    attention_gate_mode: str | None = None


def apply_roformer_fused_training_backend(
    model: torch.nn.Module,
    backend: str | None,
    *,
    extension: Any | None = None,
    rms_norm_mode: str | None = None,
) -> FusedTrainingStatus:
    normalized = normalize_fused_kernel_backend(backend)
    if normalized == "off":
        return FusedTrainingStatus(requested="off", active="off")

    if normalized != "cuda_ext":
        raise ValueError("fused_kernel_backend must be one of: off, cuda_ext")

    try:
        ext = extension if extension is not None else load_fused_band_split_extension()
    except FusedKernelUnavailable as exc:
        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        return FusedTrainingStatus(requested=normalized, active="off", reason=str(exc))

    patched = _patch_band_split_modules(model, ext)
    patched_single_stem_masks = _patch_single_stem_mask_estimation()
    patched_rotary = _patch_core_rotary(ext) if _experimental_rotary_enabled() else False
    normalized_rms_norm_mode = normalize_fused_rms_norm_mode(rms_norm_mode)
    patched_rms_norm = _patch_core_rms_norm(ext, normalized_rms_norm_mode) if normalized_rms_norm_mode else False
    attention_gate_mode = _experimental_attention_gates_mode()
    patched_attention_gates = _patch_core_attention_gates(ext, attention_gate_mode) if attention_gate_mode else False
    patched_grouped_linear = _patch_core_grouped_linear(ext) if _experimental_grouped_linear_enabled() else False
    if (
        not patched
        and not patched_single_stem_masks
        and not patched_rotary
        and not patched_rms_norm
        and not patched_attention_gates
        and not patched_grouped_linear
    ):
        reason = "no compatible pymss-core BandSplit modules were found"
        warnings.warn(reason, RuntimeWarning, stacklevel=2)
        return FusedTrainingStatus(requested=normalized, active="off", reason=reason)

    os.environ.setdefault(CORE_GROUPED_TRAIN_ENV, "1")
    return FusedTrainingStatus(
        requested=normalized,
        active=normalized,
        patched_band_splits=patched,
        patched_rotary=patched_rotary,
        patched_rms_norm=patched_rms_norm,
        rms_norm_mode=normalized_rms_norm_mode if patched_rms_norm else None,
        patched_single_stem_masks=patched_single_stem_masks,
        patched_attention_gates=patched_attention_gates,
        attention_gate_mode=attention_gate_mode if patched_attention_gates else None,
    )


def normalize_fused_kernel_backend(value: str | None) -> str:
    backend = str(value or "off").strip().lower().replace("-", "_")
    if backend in {"", "none", "false", "0", "off", "default"}:
        return "off"
    if backend in {"cuda", "cuda_ext", "cuda_extension", "cpp_cuda", "cxx_cuda"}:
        return "cuda_ext"
    raise ValueError("fused_kernel_backend must be one of: off, cuda_ext")


def normalize_fused_rms_norm_mode(value: str | None) -> str | None:
    if value is None:
        return _experimental_rms_norm_mode()
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"", "none", "false", "0", "off", "default"}:
        return None
    if mode in {"mixed", "fp32", "safe"}:
        return "mixed"
    if mode in {"1", "true", "yes", "on", "fast", "half", "fp16"}:
        return "fast"
    raise ValueError("fused_rms_norm_mode must be one of: off, fast, mixed")


def _experimental_grouped_linear_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_GROUPED_LINEAR_ENV, "").lower() in {"1", "true", "yes", "on"}


def _experimental_rotary_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_ROTARY_ENV, "").lower() in {"1", "true", "yes", "on"}


def _experimental_rms_norm_mode() -> str | None:
    value = os.environ.get(EXPERIMENTAL_RMS_NORM_ENV, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return None
    if value in {"mixed", "fp32", "safe"}:
        return "mixed"
    if value in {"1", "true", "yes", "on", "fast", "half"}:
        return "fast"
    return None


def _experimental_attention_gates_mode() -> str | None:
    value = os.environ.get(EXPERIMENTAL_ATTENTION_GATES_ENV, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return None
    if value in {"1", "true", "yes", "on", "full", "gate"}:
        return "full"
    return None


def _patch_band_split_modules(model: torch.nn.Module, extension: Any) -> int:
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "BandSplit":
            continue
        if not hasattr(module, "_get_group_params") or not hasattr(module, "_dim_offsets"):
            continue
        if getattr(module, "_pymsst_fused_band_split", False):
            continue
        module._pymsst_original_forward_grouped = module._forward_grouped
        module._pymsst_fused_extension = extension
        module._forward_grouped = types.MethodType(_forward_grouped_band_split_cuda_ext, module)
        module._pymsst_fused_band_split = True
        patched += 1
    return patched


def _patch_core_grouped_linear(extension: Any) -> bool:
    if not hasattr(extension, "grouped_linear_forward"):
        return False
    try:
        import pymss_core.modules.bs_roformer.bands as bands
    except Exception:
        return False
    if getattr(bands.grouped_linear, "_pymsst_fused_grouped_linear", False):
        return True

    original_grouped_linear = bands.grouped_linear

    def grouped_linear_cuda_ext(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        if x.device.type != "cuda" or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return original_grouped_linear(x, weight, bias)
        if weight.device != x.device or weight.dtype != x.dtype or (bias is not None and (bias.device != x.device or bias.dtype != x.dtype)):
            return original_grouped_linear(x, weight, bias)
        return _FusedGroupedLinear.apply(x, weight, bias, extension)

    grouped_linear_cuda_ext._pymsst_fused_grouped_linear = True
    grouped_linear_cuda_ext._pymsst_original_grouped_linear = original_grouped_linear
    bands.grouped_linear = grouped_linear_cuda_ext
    return True


def _patch_single_stem_mask_estimation() -> bool:
    try:
        import pymss_core.modules.bs_roformer.common as common
    except Exception:
        return False
    if getattr(common.RoformerRuntimeMixin._estimate_masks, "_pymsst_single_stem_shortcut", False):
        return True

    original_estimate_masks = common.RoformerRuntimeMixin._estimate_masks

    def estimate_masks_single_stem_shortcut(self, x: torch.Tensor):
        estimators = self._active_mask_estimators()
        if len(estimators) == 1:
            return estimators[0](x).unsqueeze(1)
        return original_estimate_masks(self, x)

    estimate_masks_single_stem_shortcut._pymsst_single_stem_shortcut = True
    estimate_masks_single_stem_shortcut._pymsst_original_estimate_masks = original_estimate_masks
    common.RoformerRuntimeMixin._estimate_masks = estimate_masks_single_stem_shortcut
    return True


def _patch_core_rotary(extension: Any) -> bool:
    if not hasattr(extension, "rotary_forward") or not hasattr(extension, "rotary_backward"):
        return False
    try:
        import pymss_core.modules.bs_roformer.transformer as transformer
    except Exception:
        return False
    if getattr(transformer.apply_rotary_emb_fast, "_pymsst_fused_rotary", False):
        return True

    original_apply_rotary = transformer.apply_rotary_emb_fast

    def apply_rotary_cuda_ext(cos: torch.Tensor, sin: torch.Tensor, t: torch.Tensor):
        if t.device.type != "cuda" or t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return original_apply_rotary(cos, sin, t)
        if t.ndim != 4 or cos.device != t.device or sin.device != t.device or cos.dtype != t.dtype or sin.dtype != t.dtype:
            return original_apply_rotary(cos, sin, t)
        return _FusedRotary.apply(t, cos[..., ::2], sin[..., ::2], extension)

    apply_rotary_cuda_ext._pymsst_fused_rotary = True
    apply_rotary_cuda_ext._pymsst_original_apply_rotary = original_apply_rotary
    transformer.apply_rotary_emb_fast = apply_rotary_cuda_ext
    return True


def _patch_core_rms_norm(extension: Any, mode: str) -> bool:
    if mode == "mixed":
        required = ("rms_norm_forward_mixed", "rms_norm_backward_mixed")
    elif mode == "fast":
        required = ("rms_norm_forward", "rms_norm_backward")
    else:
        return False

    if any(not hasattr(extension, name) for name in required):
        return False
    try:
        import pymss_core.modules.bs_roformer.transformer as transformer
    except Exception:
        return False
    if getattr(transformer.RMSNorm.forward, "_pymsst_fused_rms_norm", False):
        return True

    original_forward = transformer.RMSNorm.forward

    def rms_norm_cuda_ext(self, x: torch.Tensor):
        if not self.training or x.device.type != "cuda" or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return original_forward(self, x)
        if mode == "mixed" and self.gamma.dtype == torch.float32:
            return _FusedRMSNormMixed.apply(x, self.gamma, extension)
        gamma = self.gamma.to(device=x.device, dtype=x.dtype)
        return _FusedRMSNorm.apply(x, gamma, extension)

    rms_norm_cuda_ext._pymsst_fused_rms_norm = True
    rms_norm_cuda_ext._pymsst_fused_rms_norm_mode = mode
    rms_norm_cuda_ext._pymsst_original_forward = original_forward
    transformer.RMSNorm.forward = rms_norm_cuda_ext
    return True


def _patch_core_attention_gates(extension: Any, mode: str) -> bool:
    if mode == "full":
        required = ("attention_gate_forward", "attention_gate_backward")
    else:
        return False

    if any(not hasattr(extension, name) for name in required):
        return False
    try:
        import pymss_core.modules.bs_roformer.transformer as transformer
    except Exception:
        return False
    if getattr(transformer.Attention.forward, "_pymsst_fused_attention_gates", False):
        return True

    original_forward = transformer.Attention.forward

    def attention_forward_cuda_ext(self, x: torch.Tensor):
        if (
            not self.training
            or x.device.type != "cuda"
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or self.to_gates.bias is None
        ):
            return original_forward(self, x)

        x = self.norm(x)
        q, k, v = transformer.qkv_to_bnhd(self.to_qkv(x), self.heads)

        if self.rotary_embed is not None:
            q, k = transformer.rotate_qk_fast_bnhd(self.rotary_embed, q, k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = self._attention(q, k, v).transpose(1, 2)
        gate_dtype = out.dtype
        gated = _FusedAttentionGate.apply(
            out,
            x.to(dtype=gate_dtype),
            self.to_gates.weight.to(dtype=gate_dtype),
            self.to_gates.bias.to(dtype=gate_dtype),
            extension,
        )
        return self.to_out(gated)

    attention_forward_cuda_ext._pymsst_fused_attention_gates = True
    attention_forward_cuda_ext._pymsst_fused_attention_gate_mode = mode
    attention_forward_cuda_ext._pymsst_original_forward = original_forward
    transformer.Attention.forward = attention_forward_cuda_ext
    return True


def _forward_grouped_band_split_cuda_ext(self, x: torch.Tensor) -> torch.Tensor:
    if not self.training or x.device.type != "cuda":
        return self._pymsst_original_forward_grouped(x)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return self._pymsst_original_forward_grouped(x)

    ext = self._pymsst_fused_extension
    leading_shape = x.shape[:-1]
    outs = []
    for start, end, dim_in in self._dim_groups:
        offset_start = self._dim_offsets[start]
        offset_end = self._dim_offsets[end]
        group_x = x[..., offset_start:offset_end].reshape(-1, end - start, dim_in)
        gamma, weight, bias = self._get_group_params(start, end, x.device, x.dtype)
        outs.append(_FusedBandSplitGroup.apply(group_x, gamma, weight, bias, ext).reshape(*leading_shape, end - start, -1))
    return torch.cat(outs, dim=-2)


class _FusedBandSplitGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, extension):
        out, normed, inv_norm = extension.band_split_forward(x, gamma, weight, bias)
        ctx.save_for_backward(normed, inv_norm, gamma, weight)
        ctx.has_bias = bias is not None
        ctx.extension = extension
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        normed, inv_norm, gamma, weight = ctx.saved_tensors
        grad_out = grad_out.reshape(normed.shape[0], normed.shape[1], weight.shape[1]).contiguous()
        grad_x, grad_gamma, grad_weight, grad_bias = ctx.extension.band_split_backward(
            grad_out,
            normed,
            inv_norm,
            gamma,
            weight,
            ctx.has_bias,
        )
        if not ctx.has_bias:
            grad_bias = None
        return grad_x, grad_gamma, grad_weight, grad_bias, None


class _FusedGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, extension):
        group_count = x.shape[-2]
        in_features = x.shape[-1]
        x_3d = x.reshape(-1, group_count, in_features)
        out = extension.grouped_linear_forward(x_3d, weight, bias)
        ctx.input_shape = tuple(x.shape)
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x_3d, weight)
        return out.reshape(*x.shape[:-2], group_count, weight.shape[1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_3d, weight = ctx.saved_tensors
        grad_out_3d = grad_out.reshape(x_3d.shape[0], x_3d.shape[1], weight.shape[1])
        grad_x = torch.einsum("mgo,goi->mgi", grad_out_3d, weight).reshape(ctx.input_shape)
        grad_weight = torch.einsum("mgo,mgi->goi", grad_out_3d, x_3d)
        grad_bias = grad_out_3d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_weight, grad_bias, None


class _FusedRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, extension):
        out = extension.rotary_forward(t, cos, sin)
        ctx.save_for_backward(cos.contiguous(), sin.contiguous())
        ctx.extension = extension
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        cos, sin = ctx.saved_tensors
        grad_t = ctx.extension.rotary_backward(grad_out, cos, sin)
        return grad_t, None, None, None


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, extension):
        out, normed, inv_norm = extension.rms_norm_forward(x, gamma)
        ctx.input_shape = tuple(x.shape)
        ctx.save_for_backward(normed, inv_norm, gamma)
        ctx.extension = extension
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        normed, inv_norm, gamma = ctx.saved_tensors
        grad_x, grad_gamma = ctx.extension.rms_norm_backward(grad_out.reshape(ctx.input_shape), normed, inv_norm, gamma)
        return grad_x.reshape(ctx.input_shape), grad_gamma, None


class _FusedRMSNormMixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, extension):
        out, normed, inv_norm = extension.rms_norm_forward_mixed(x, gamma)
        ctx.input_shape = tuple(x.shape)
        ctx.save_for_backward(normed, inv_norm, gamma, x.new_empty(0))
        ctx.extension = extension
        return out.reshape(*x.shape[:-1], x.shape[-1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        normed, inv_norm, gamma, x_meta = ctx.saved_tensors
        grad_x, grad_gamma = ctx.extension.rms_norm_backward_mixed(
            grad_out.reshape(ctx.input_shape),
            normed,
            inv_norm,
            gamma,
            x_meta,
        )
        return grad_x.reshape(ctx.input_shape), grad_gamma, None


class _FusedAttentionGate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        attn: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        extension,
    ):
        attn_shape = tuple(attn.shape)
        x_shape = tuple(x.shape)
        heads = attn.shape[-2]
        dim_head = attn.shape[-1]
        attn_3d = attn.reshape(-1, heads, dim_head)
        x_2d = x.reshape(-1, x.shape[-1])
        out, gate, saved_attn, saved_x = extension.attention_gate_forward(attn_3d, x_2d, weight, bias)
        ctx.attn_shape = attn_shape
        ctx.x_shape = x_shape
        ctx.save_for_backward(saved_attn, saved_x, weight, gate)
        ctx.extension = extension
        return out.reshape(*x.shape[:-1], heads * dim_head)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        saved_attn, saved_x, weight, gate = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(saved_attn.shape[0], saved_attn.shape[1] * saved_attn.shape[2])
        grad_attn, grad_x, grad_weight, grad_bias = ctx.extension.attention_gate_backward(
            grad_out_2d,
            saved_attn,
            saved_x,
            weight,
            gate,
        )
        return grad_attn.reshape(ctx.attn_shape), grad_x.reshape(ctx.x_shape), grad_weight, grad_bias, None


def reference_band_split_group(x: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    dim_in = x.shape[-1]
    normalized = F.normalize(x, dim=-1) * (dim_in**0.5) * gamma.unsqueeze(0)
    out = torch.einsum("mgi,goi->mgo", normalized, weight)
    if bias is not None:
        out = out + bias.unsqueeze(0)
    return out
