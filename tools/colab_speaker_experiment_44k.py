"""44.1 kHz stereo run: VCTK + VocalSet identities, on-the-fly stereo mixtures.

Uploaded and executed remotely; never reads credentials. SPEAKER_STEPS is the absolute
update target. The run starts with a 20-step smoke test, then resumes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

WORKSPACE = Path("/content/adaptive-speakers")
RUN = Path("/content/runs/adaptive-44k-stereo")
DATA = Path("/content/data")
CONFIG = WORKSPACE / "pymsst/configs/adaptive_bs_roformer_44k_stereo.yaml"


def run(command, log=None, label="step"):
    log = Path(log) if log else Path(f"/content/adaptive-step-{label}.log")
    print(json.dumps({"event": "command", "args": [str(part) for part in command[:4]], "log": str(log)}), flush=True)
    with log.open("w") as handle:
        result = subprocess.run([str(part) for part in command], stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = [line for line in log.read_text(errors="replace").splitlines() if line.strip()][-12:]
        raise RuntimeError(f"{' '.join(command[:3])} failed ({result.returncode}): " + " | ".join(tail))


def main():
    revisions = json.loads((WORKSPACE / "source-revisions.json").read_text())
    steps = int(os.environ.get("SPEAKER_STEPS", "2000"))
    if steps < 20:
        raise ValueError("SPEAKER_STEPS must be at least 20")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    if "L4" not in gpu:
        raise RuntimeError(f"expected the requested L4 runtime, got {gpu}")
    print(json.dumps({"event": "gpu", "gpu": gpu}), flush=True)

    install_log = Path("/content/adaptive-install.log")
    for extra in (WORKSPACE / "pymss-core", WORKSPACE / "pymss", WORKSPACE / "pymsst/src"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    needed = ("torch", "pytorch_lightning", "soundfile", "librosa", "scipy", "pyarrow", "yaml")
    missing = []
    for module in needed:
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    print(json.dumps({"event": "deps", "missing": missing}), flush=True)
    if missing:
        run([sys.executable, "-m", "pip", "uninstall", "-y", "pymsst", "pymss-core", "pymss"], install_log, "uninstall")
        run([sys.executable, "-m", "pip", "install", "-e", WORKSPACE / "pymss-core",
             "-e", str(WORKSPACE / "pymsst") + "[speakers]", "-e", WORKSPACE / "pymss"], install_log, "install")

    # Import only after installation: a fresh runtime has no pymsst yet.
    from pymsst.data.sources import HF_CORPORA

    print(json.dumps({"event": "source", "revisions": {k: v[:7] for k, v in revisions.items()},
                      "hf_corpora": sorted(HF_CORPORA)}), flush=True)
    if "vctk-hf" not in HF_CORPORA:
        raise RuntimeError("stale source tree: pymsst.data.sources has no vctk-hf entry")

    cli = [sys.executable, "-m", "pymsst"]
    corpora = DATA / "corpora"
    for name, marker in (("vctk-hf", corpora / "vctk"), ("vocalset", corpora / "VocalSet")):
        if marker.is_dir() and any(marker.iterdir()):
            print(json.dumps({"event": "corpus_cached", "name": name}), flush=True)
            continue
        run([*cli, "fetch-corpus", "--name", name, "--root", corpora], label=f"fetch-{name}")
        print(json.dumps({"event": "corpus", "name": name, "present": marker.is_dir()}), flush=True)
    linked = DATA / "speakers"
    if linked.exists():
        shutil.rmtree(linked)
    # mic2 duplicates mic1; keeping both doubles near-identical audio without adding identity.
    run([*cli, "link-speakers", "--root", corpora, "--output", linked, "--drop", "mic2"], label="link")
    manifests = DATA / "manifests"
    if not (manifests / "train.jsonl").is_file():
        run([*cli, "prepare-speakers", "--root", linked, "--output", manifests, "--layout", "speaker-dirs"], label="prepare")

    train = [*cli, "train-speakers", "--config", CONFIG,
             "--train-manifest", manifests / "train.jsonl",
             "--validation-manifest", manifests / "validation.jsonl",
             "--run-dir", RUN, "--device", "cuda", "--num-workers", "4"]
    if not (RUN / "last.pt").exists():
        run([*train, "--steps", "20"])
    if steps > 20:
        run([*train, "--steps", str(steps), "--resume", RUN / "last.pt"])

    import soundfile as sf
    import torch
    from pymss_core import get_model_from_config

    from pymsst.data.speakers import SpeakerMixtureDataset, validate_disjoint_manifests
    from pymsst.training.speaker_run import evaluate_speakers

    for other in ("train", "validation"):
        validate_disjoint_manifests(manifests / f"{other}.jsonl", manifests / "test.jsonl")
    model, config = get_model_from_config("auto", RUN / "config.yaml")
    package = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(package["state_dict"], strict=True)
    model.cuda().eval()
    test = SpeakerMixtureDataset(manifests / "test.jsonl", sample_rate=config.audio.sample_rate,
                                 duration=config.audio.duration, max_speakers=model.max_speakers,
                                 samples_per_epoch=48, validation=True, seed=20261145,
                                 augmentation=dict(config.augmentation), channels=model.audio_channels)
    metrics = evaluate_speakers(model, test, torch.device("cuda"), dict(config.loss))
    (RUN / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    examples = RUN / "examples"
    examples.mkdir(exist_ok=True)
    summaries = []
    for count in (2, 3, 4, 5):
        mixture, target, _ = test[count]
        with torch.inference_mode():
            output = model(mixture[None].cuda(), stop_early=True)
        predicted = int(output.counts()[0])
        directory = examples / f"{count}_speakers"
        directory.mkdir(exist_ok=True)
        sf.write(directory / "mixture.wav", mixture.T.numpy(), test.sample_rate, subtype="FLOAT")
        for i in range(min(count, 2)):
            sf.write(directory / f"reference_{i + 1}.wav", target[i].T.numpy(), test.sample_rate, subtype="FLOAT")
        for i in range(min(predicted, 5)):
            sf.write(directory / f"prediction_{i + 1}.wav", output.audio[0, i].T.cpu().numpy(), test.sample_rate, subtype="FLOAT")
        summaries.append({"reference_speakers": count, "predicted_speakers": predicted,
                          "limit_reached": bool(output.at_limit()[0])})
    (RUN / "example_counts.json").write_text(json.dumps(summaries, indent=2) + "\n")
    artifact = Path("/content/adaptive-speakers-artifacts.tar.gz")
    with tarfile.open(artifact, "w:gz") as handle:
        for path in sorted(RUN.rglob("*")):
            if path.is_file() and ".tmp" not in path.relative_to(RUN).parts:
                handle.add(path, arcname=str(Path("adaptive-44k-stereo") / path.relative_to(RUN)))
    print(json.dumps({"event": "complete", "steps": steps, "test_metrics": metrics,
                      "examples": summaries, "artifact": str(artifact),
                      "artifact_bytes": artifact.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
