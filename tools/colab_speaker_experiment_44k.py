"""44.1 kHz stereo run: VCTK + VocalSet identities, on-the-fly stereo mixtures.

Uploaded and executed remotely; never reads credentials. SPEAKER_STEPS is the absolute
update target. The run starts with a 20-step smoke test, then resumes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import tarfile
from pathlib import Path

import torch

WORKSPACE = Path("/content/adaptive-speakers")
RUN = Path("/content/runs/adaptive-44k-stereo")
DATA = Path("/content/data")
CONFIG = Path(os.environ.get("SPEAKER_CONFIG",
                             str(WORKSPACE / "pymsst/configs/adaptive_bs_roformer_44k_stereo.yaml")))


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
    if not any(name in gpu for name in ("L4", "A100", "H100", "V100", "G4")):
        raise RuntimeError(f"no supported GPU in this runtime: {gpu}")
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
    shard_cap = int(os.environ.get("SPEAKER_MAX_SHARDS", "0")) or None
    wanted = [name.strip() for name in os.environ.get("SPEAKER_CORPORA", "vctk-hf").split(",") if name.strip()]
    complete = {"vctk-hf": corpora / "vctk" / ".fetch_complete",
                "vocalset": corpora / ".archives" / "vocalset.extracted",
                "emilia-yodas-en": corpora / "emilia-yodas-en" / ".fetch_complete"}
    markers = {"vctk-hf": corpora / "vctk", "vocalset": corpora / "VocalSet",
               "emilia-yodas-en": corpora / "emilia-yodas-en"}
    for name in wanted:
        marker = markers[name]
        if complete[name].exists():
            print(json.dumps({"event": "corpus_cached", "name": name}), flush=True)
            continue
        # Adopt a tree that is evidently whole, even if an older run never wrote the marker.
        if marker.is_dir():
            if name == "vctk-hf":
                whole = len([d for d in marker.iterdir() if d.is_dir() and d.name.startswith(("p", "s"))]) >= 100
            elif name == "emilia-yodas-en":
                whole = (marker / ".shards").is_dir() and len(list((marker / ".shards").iterdir())) >= (shard_cap or 200)
            else:
                whole = len(list((marker / "FULL").glob("*"))) >= 20 if (marker / "FULL").is_dir() else False
            if whole:
                complete[name].parent.mkdir(parents=True, exist_ok=True)
                complete[name].write_text("adopted\n")
                print(json.dumps({"event": "corpus_adopted", "name": name}), flush=True)
                continue
            # Resumable only when shard markers prove what is already decoded.
            if name in ("vctk-hf", "emilia-yodas-en") and (marker / ".shards").is_dir():
                print(json.dumps({"event": "corpus_resuming", "name": name,
                                  "shards_done": len(list((marker / ".shards").iterdir()))}), flush=True)
                continue
            shutil.rmtree(marker, ignore_errors=True)
            print(json.dumps({"event": "corpus_discarded", "name": name}), flush=True)
        extra = ["--max-shards", str(shard_cap)] if shard_cap and name == "emilia-yodas-en" else []
        run([*cli, "fetch-corpus", "--name", name, "--root", corpora, *extra], label=f"fetch-{name}")
        print(json.dumps({"event": "corpus", "name": name, "present": marker.is_dir()}), flush=True)
    linked = DATA / "speakers"
    if linked.exists():
        shutil.rmtree(linked)
    # mic2 duplicates mic1; keeping both doubles near-identical audio without adding identity.
    run([*cli, "link-speakers", "--root", corpora, "--output", linked, "--drop", "mic2"], label="link")
    manifests = DATA / "manifests"
    # Always rebuild: a manifest left over from a partial corpus silently skips identities.
    if manifests.exists():
        shutil.rmtree(manifests)
    run([*cli, "prepare-speakers", "--root", linked, "--output", manifests, "--layout", "speaker-dirs"], label="prepare")
    summary = json.loads((manifests / "summary.json").read_text())
    print(json.dumps({"event": "manifests", **summary}), flush=True)
    if any(part["speakers"] < 1 for part in summary.values()):
        raise RuntimeError(f"a split has no identities: {summary}")

    train = [*cli, "train-speakers", "--config", CONFIG,
             "--train-manifest", manifests / "train.jsonl",
             "--validation-manifest", manifests / "validation.jsonl",
             "--run-dir", RUN, "--device", "cuda", "--num-workers", "8"]
    # A reclaimed runtime loses RUN; the orchestrator re-uploads weights to warm start from.
    seed = Path("/content/adaptive-seed-checkpoint.pt")
    if os.environ.get("SPEAKER_WAIT_SEED", "0") == "1" and not seed.is_file():
        # Uploads run beside the corpus download, so wait here instead of idling the GPU.
        deadline = time.time() + float(os.environ.get("SPEAKER_SEED_WAIT_SECONDS", "1200"))
        while time.time() < deadline and not seed.is_file():
            time.sleep(15)
        print(json.dumps({"event": "seed_wait_done", "present": seed.is_file()}), flush=True)
    if seed.is_file():
        print(json.dumps({"event": "seed_available",
                          "step": torch.load(seed, map_location="cpu", weights_only=True).get("step")}), flush=True)
    # Fill the card: start big and step down only if the allocator refuses.
    wanted = int(os.environ.get("SPEAKER_BATCH", "8"))
    candidates = [wanted] + [size for size in (8, 4, 2, 1) if size < wanted]
    batch = wanted if (RUN / "last.pt").exists() else None  # a resumed run keeps the proven batch
    if not (RUN / "last.pt").exists():
        for size in candidates:
            try:
                run([*train, "--steps", "20", "--batch-size", str(size)], label="smoke")
                batch = size
                break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                print(json.dumps({"event": "oom", "batch": size}), flush=True)
                shutil.rmtree(RUN, ignore_errors=True)
        if batch is None:
            raise RuntimeError("every candidate batch size ran out of memory")
        resumed = torch.load(RUN / "last.pt", map_location="cpu", weights_only=True)["step"]
        print(json.dumps({"event": "batch", "selected": batch, "smoke_step": resumed}), flush=True)
    if steps > 20:
        if (RUN / "last.pt").exists():
            carry = ["--resume", RUN / "last.pt"]
        elif seed.is_file():
            # Same weights, new corpus: --resume would reject the changed manifests by design.
            carry = ["--warm-start", seed]
        else:
            carry = []
        run([*train, "--steps", str(steps), *carry, "--batch-size", str(batch)], label="train")

    import soundfile as sf
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
    # Separation ability and robustness to pitch/EQ/reverb/noise are different claims; report both.
    clean = SpeakerMixtureDataset(manifests / "test.jsonl", sample_rate=config.audio.sample_rate,
                                  duration=config.audio.duration, max_speakers=model.max_speakers,
                                  samples_per_epoch=48, validation=True, seed=20261145, augment=False,
                                  augmentation=dict(config.augmentation), channels=model.audio_channels)
    clean_metrics = evaluate_speakers(model, clean, torch.device("cuda"), dict(config.loss))
    (RUN / "test_metrics_clean.json").write_text(json.dumps(clean_metrics, indent=2) + "\n")
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
