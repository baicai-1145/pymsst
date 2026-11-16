"""Permutation-matched reconstruction and stop-token supervision for variable speaker counts."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def pairwise_si_sdr(estimate, target, eps=1e-8):
    estimate, target = estimate.float().flatten(1), target.float().flatten(1)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = estimate @ target.T
    energy = target.square().sum(dim=-1).clamp_min(eps)
    projected = dot.square() / energy[None]
    error = (estimate.square().sum(dim=-1, keepdim=True) - projected).clamp_min(eps)
    return 10 * torch.log10((projected + eps) / (error + eps))


def spectral_loss(estimate, target, fft_sizes=(256, 512, 1024)):
    estimate, target = estimate.float().flatten(0, 1), target.float().flatten(0, 1)
    total = estimate.new_zeros(())
    for n_fft in fft_sizes:
        window = torch.hann_window(n_fft, device=estimate.device)
        args = {"n_fft": n_fft, "hop_length": n_fft // 4, "window": window, "return_complex": True, "pad_mode": "constant"}
        a, b = torch.stft(estimate, **args), torch.stft(target, **args)
        total = total + (a - b).abs().mean() / b.abs().mean().clamp_min(1e-4)
    return total / len(fft_sizes)


def speaker_loss(output, target, counts, *, mixture=None, fft_sizes=(256, 512, 1024),
                 spectral_weight=0.5, existence_weight=1., silence_weight=0.1):
    from scipy.optimize import linear_sum_assignment

    prediction, logits = output.audio.float(), output.existence_logits.float()
    if prediction.shape != target.shape or logits.shape != (target.shape[0], target.shape[1] + 1):
        raise ValueError("training requires padded [B,S,C,T] audio and [B,S+1] existence logits")
    if counts.shape != (target.shape[0],) or (counts < 0).any() or (counts > target.shape[1]).any():
        raise ValueError("counts must be [batch] within the target speaker range")
    labels = torch.arange(logits.shape[1], device=logits.device)[None] < counts[:, None]
    existence = F.binary_cross_entropy_with_logits(logits, labels.float())
    reconstruction, empty, scores, improvements = [], [], [], []
    for b, count in enumerate(counts.tolist()):
        if count:
            p, t = prediction[b, :count], target[b, :count].float()
            si_sdr = pairwise_si_sdr(p, t)
            l1 = (p[:, None] - t[None]).abs().mean(dim=(-1, -2)) / t.abs().mean(dim=(-1, -2)).clamp_min(1e-4)[None]
            cost = -si_sdr.detach() / 10 + l1.detach()
            rows, columns = linear_sum_assignment(cost.cpu().numpy())
            aligned, truth = p[rows], t[columns]
            score = si_sdr[rows, columns].mean()
            loss = -score / 10 + l1[rows, columns].mean()
            if spectral_weight:
                loss = loss + spectral_weight * spectral_loss(aligned, truth, fft_sizes)
            reconstruction.append(loss)
            scores.append(score)
            if mixture is not None and count > 1:
                baseline = pairwise_si_sdr(mixture[b:b + 1], truth).mean()
                improvements.append(score - baseline)
        if count < prediction.shape[1]:
            # Absolute energy has a defined optimum for silence; SI-SDR does not.
            empty.append(prediction[b, count:].square().mean())
    zero = prediction.sum() * 0
    reconstruction = torch.stack(reconstruction).mean() if reconstruction else zero
    empty = torch.stack(empty).mean() if empty else zero
    loss = reconstruction + existence_weight * existence + silence_weight * empty
    predicted_counts = output.counts()
    return {
        "loss": loss, "reconstruction": reconstruction.detach(), "existence": existence.detach(),
        "silence": empty.detach(), "oracle_si_sdr": torch.stack(scores).mean().detach() if scores else zero.detach(),
        "oracle_si_sdr_i": torch.stack(improvements).mean().detach() if improvements else zero.detach(),
        "count_accuracy": (predicted_counts == counts).float().mean().detach(),
        "count_mae": (predicted_counts - counts).abs().float().mean().detach(),
    }
