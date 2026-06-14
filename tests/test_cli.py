import random
import json
from pathlib import Path
from types import SimpleNamespace

from pymsst.cli import _consume_original_roformer_python_rng, _is_rank_zero_process, _parse_devices, _resolve_precision, main


def _touch_track(root: Path, split: str, name: str) -> None:
    track = root / split / name
    track.mkdir(parents=True)
    for stem in ("mixture", "vocals", "drums", "bass", "other"):
        (track / f"{stem}.wav").write_bytes(b"")


def test_check_musdb_command(tmp_path, capsys):
    _touch_track(tmp_path, "train", "Song A")

    status = main(["check-musdb", str(tmp_path), "--split", "train"])

    assert status == 0
    assert "train: 1 complete tracks" in capsys.readouterr().out


def test_roformer_legacy_rng_compat_consumes_three_python_random_words():
    random.seed(0)
    random.getrandbits(96)
    expected = random.random()

    random.seed(0)
    _consume_original_roformer_python_rng("bs_roformer")
    observed = random.random()

    assert observed == expected

    random.seed(0)
    no_op_expected = random.random()
    random.seed(0)
    _consume_original_roformer_python_rng("mdx23c")

    assert random.random() == no_op_expected


def test_parse_devices_accepts_auto_int_and_comma_list():
    assert _parse_devices("auto") == "auto"
    assert _parse_devices("2") == 2
    assert _parse_devices("0,1") == [0, 1]
    assert _parse_devices("cuda") == "cuda"


def test_resolve_precision_uses_amp_config_only_when_cuda_available():
    config = SimpleNamespace(training={"use_amp": True})

    assert _resolve_precision("32-true", config, cuda_available=True) == "32-true"
    assert _resolve_precision("auto", config, cuda_available=True) == "16-mixed"
    assert _resolve_precision("auto", config, cuda_available=False) == "32-true"


def test_is_rank_zero_process_uses_ddp_environment(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert _is_rank_zero_process()

    monkeypatch.setenv("LOCAL_RANK", "1")
    assert not _is_rank_zero_process()

    monkeypatch.setenv("RANK", "0")
    assert _is_rank_zero_process()

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert not _is_rank_zero_process()


def test_compare_direct_command_invokes_runner(tmp_path, monkeypatch, capsys):
    calls = {}

    def fake_run_direct_loss_curve(**kwargs):
        calls.update(kwargs)
        return {"loss_curve_path": str(tmp_path / "loss_curve.csv")}

    monkeypatch.setattr("pymsst.validation.run_direct_loss_curve", fake_run_direct_loss_curve)

    status = main(
        [
            "compare-direct",
            "--implementation",
            "pymsst",
            "--model-type",
            "bs_roformer",
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--results-path",
            str(tmp_path / "results"),
            "--steps",
            "3",
            "--optimizer-backend",
            "fused_capturable",
            "--cuda-attention-backend",
            "efficient",
            "--fused-kernel-backend",
            "cuda_ext",
            "--fused-rms-norm-mode",
            "fast",
            "--cuda-graph-mode",
            "mask_core",
            "--num-workers",
            "4",
            "--pin-memory",
            "--persistent-workers",
            "--loss-record-interval",
            "100",
        ]
    )

    assert status == 0
    assert calls["implementation"] == "pymsst"
    assert calls["model_type"] == "bs_roformer"
    assert calls["steps"] == 3
    assert calls["optimizer_backend"] == "fused_capturable"
    assert calls["cuda_attention_backend"] == "efficient"
    assert calls["fused_kernel_backend"] == "cuda_ext"
    assert calls["fused_rms_norm_mode"] == "fast"
    assert calls["cuda_graph_mode"] == "mask_core"
    assert calls["num_workers"] == 4
    assert calls["pin_memory"] is True
    assert calls["persistent_workers"] is True
    assert calls["loss_record_interval"] == 100
    assert "loss_curve.csv" in capsys.readouterr().out


def test_compare_summary_command_writes_output_and_returns_failure(tmp_path, capsys):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "pymsst_points": 10,
                "original_points": 10,
                "compared_points": 10,
                "mean_abs_diff": 0.2,
                "max_abs_diff": 0.3,
                "first_pymsst_loss": 1.0,
                "first_original_loss": 1.0,
                "last_pymsst_loss": 0.5,
                "last_original_loss": 0.5,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"

    status = main(
        [
            "compare-summary",
            "--glob",
            str(tmp_path / "comparison*.json"),
            "--min-points",
            "10",
            "--max-mean-abs-diff",
            "0.1",
            "--output",
            str(output),
        ]
    )

    assert status == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["aggregate"]["passed"] is False
    assert payload["failures"]
    assert "summary.json" in capsys.readouterr().out
