"""Run via Colab CLI exec --file after uploading the three source repos to /content/adaptive-speakers.

No credentials are read here. The local CLI authenticates separately. SPEAKER_STEPS controls
an absolute update target (default 2000); the first 20 updates are a resumable GPU smoke test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def main():
    workspace = Path("/content/adaptive-speakers")
    run = Path("/content/runs/adaptive-bs-l4")
    steps = int(os.environ.get("SPEAKER_STEPS", "2000"))
    if steps < 20:
        raise ValueError("SPEAKER_STEPS must be at least 20")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    if "L4" not in gpu:
        raise RuntimeError(f"expected the requested L4 runtime, got {gpu}")
    print(json.dumps({"event": "gpu", "gpu": gpu}), flush=True)
    with Path("/content/adaptive-install.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(workspace / "pymss-core"),
                        "-e", str(workspace / "pymsst") + "[speakers]", "-e", str(workspace / "pymss")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [sys.executable, "-m", "pymsst"]
    subprocess.run([*command, "download-speakers", "--root", "/content/data"], check=True)
    subprocess.run([*command, "prepare-speakers", "--root", "/content/data/LibriSpeech",
                    "--output", "/content/data/manifests"], check=True)
    train = [*command, "train-speakers", "--config", str(workspace / "pymsst/configs/adaptive_bs_roformer_l4.yaml"),
             "--train-manifest", "/content/data/manifests/train.jsonl",
             "--validation-manifest", "/content/data/manifests/validation.jsonl",
             "--run-dir", str(run), "--device", "cuda", "--num-workers", "2"]
    if not (run / "last.pt").exists():
        subprocess.run([*train, "--steps", "20"], check=True)
    if steps > 20:
        subprocess.run([*train, "--steps", str(steps), "--resume", str(run / "last.pt")], check=True)

    import soundfile as sf
    import torch
    from pymss_core import get_model_from_config

    from pymsst.data.speakers import SpeakerMixtureDataset, validate_disjoint_manifests
    from pymsst.training.speaker_run import evaluate_speakers

    for other in ("train", "validation"):
        validate_disjoint_manifests(f"/content/data/manifests/{other}.jsonl", "/content/data/manifests/test.jsonl")
    model, config = get_model_from_config("auto", run / "config.yaml")
    package = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(package["state_dict"], strict=True)
    model.cuda().eval()
    test = SpeakerMixtureDataset("/content/data/manifests/test.jsonl", sample_rate=config.audio.sample_rate,
                                 duration=config.audio.duration, max_speakers=model.max_speakers,
                                 samples_per_epoch=48, validation=True, seed=20261145,
                                 augmentation=dict(config.augmentation))
    metrics = evaluate_speakers(model, test, torch.device("cuda"), dict(config.loss))
    (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    examples = run / "examples"
    examples.mkdir(exist_ok=True)
    summaries = []
    for count in (2, 3, 4, 5):
        mixture, target, _ = test[count]
        with torch.inference_mode():
            output = model(mixture[None].cuda(), stop_early=True)
        predicted = int(output.counts()[0])
        directory = examples / f"{count}_speakers"
        directory.mkdir(exist_ok=True)
        sf.write(directory / "mixture.wav", mixture[0].numpy(), test.sample_rate, subtype="FLOAT")
        for i in range(count):
            sf.write(directory / f"reference_{i + 1}.wav", target[i, 0].numpy(), test.sample_rate, subtype="FLOAT")
        for i in range(predicted):
            sf.write(directory / f"prediction_{i + 1}.wav", output.audio[0, i, 0].cpu().numpy(), test.sample_rate, subtype="FLOAT")
        summaries.append({"reference_speakers": count, "predicted_speakers": predicted, "limit_reached": bool(output.at_limit()[0])})
    (run / "example_counts.json").write_text(json.dumps(summaries, indent=2) + "\n")
    artifact = Path("/content/adaptive-speakers-artifacts.tar.gz")
    with tarfile.open(artifact, "w:gz") as handle:
        for path in sorted(run.rglob("*")):
            if path.is_file() and ".tmp" not in path.relative_to(run).parts:
                handle.add(path, arcname=str(Path("adaptive-bs-l4") / path.relative_to(run)))
    print(json.dumps({"event": "complete", "steps": steps, "test_metrics": metrics,
                          "examples": summaries, "artifact": str(artifact), "artifact_bytes": artifact.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
