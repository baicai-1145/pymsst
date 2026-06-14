from __future__ import annotations

import os
from pathlib import Path

import pytest

from pymsst.training import SeparationLightningModule


PRETRAINED_CASES = (
    (
        "bs_roformer",
        Path("/root/Music-Source-Separation-Training/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml"),
        Path("/root/pymsst/pretrained/model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
        None,
    ),
    (
        "mel_band_roformer",
        Path("/root/Music-Source-Separation-Training/configs/viperx/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"),
        Path("/root/pymsst/pretrained/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"),
        2,
    ),
)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PYMSST_RUN_PRETRAINED_COMPAT") != "1",
    reason="set PYMSST_RUN_PRETRAINED_COMPAT=1 to strict-load local pretrained checkpoints",
)
@pytest.mark.parametrize(("model_type", "config_path", "checkpoint_path", "expected_mlp_hidden_layers"), PRETRAINED_CASES)
def test_local_pretrained_checkpoint_strict_loads_with_training_module(
    model_type: str,
    config_path: Path,
    checkpoint_path: Path,
    expected_mlp_hidden_layers: int | None,
) -> None:
    if not config_path.is_file() or not checkpoint_path.is_file():
        pytest.skip("local original config or pretrained checkpoint is unavailable")

    module = SeparationLightningModule(
        model_type=model_type,
        config_path=config_path,
        start_checkpoint=checkpoint_path,
        strict_checkpoint=True,
    )

    assert sum(parameter.numel() for parameter in module.model.parameters()) > 0
    assert module.config.model.get("mlp_hidden_layers") == expected_mlp_hidden_layers
