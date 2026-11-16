"""Resumable single-device training and per-count validation for adaptive speaker separation."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from pymss_core import get_model_from_config
from torch.utils.data import DataLoader

from pymsst.data.speakers import SpeakerMixtureDataset, validate_disjoint_manifests
from pymsst.export import export_core_checkpoint

from .speakers import speaker_loss


def evaluate_speakers(model, dataset, device, loss_options=None):
    model.eval()
    groups = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for mixture, target, counts in DataLoader(dataset, batch_size=1):
            mixture, target, counts = mixture.to(device), target.to(device), counts.to(device)
            output = model(mixture)
            values = speaker_loss(output, target, counts, mixture=mixture, **(loss_options or {}))
            count = int(counts[0])
            for key, value in values.items():
                if (count == 0 and key.startswith("oracle_si_sdr")) or (count < 2 and key == "oracle_si_sdr_i"):
                    continue
                groups[str(count)][key].append(float(value))
    per_count = {n: {key: sum(values) / len(values) for key, values in group.items()} for n, group in groups.items()}
    overall = defaultdict(list)
    for group in groups.values():
        for key, values in group.items():
            overall[key].extend(values)
    return {"overall": {key: sum(values) / len(values) for key, values in overall.items()}, "per_count": per_count}


def _atomic_save(package, path):
    path = Path(path)
    scratch = path.parent / ".tmp"
    scratch.mkdir(exist_ok=True)
    temporary = scratch / path.name
    torch.save(package, temporary)
    temporary.replace(path)


def train_speakers(args):
    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("model_type") not in {"adaptive_bs_roformer", "adaptive_mel_band_roformer"}:
        raise ValueError("train-speakers requires an explicitly labelled adaptive model config")
    settings, audio = config.get("training", {}), config["audio"]
    seed = int(settings.get("seed", 0))
    torch.manual_seed(seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, _ = get_model_from_config("auto", args.config)
    if int(audio.get("channels", model.audio_channels)) != model.audio_channels:
        raise ValueError("audio.channels must match the model stereo setting")
    model.to(device)
    maximum = model.max_speakers
    batch_size = int(args.batch_size if getattr(args, "batch_size", None) else settings.get("batch_size", 2))
    steps_per_epoch = int(settings.get("steps_per_epoch", 500))
    max_steps = args.steps if args.steps is not None else int(settings.get("max_steps", 20000))
    validate_every = int(settings.get("validate_every", 250))
    save_every = int(settings.get("save_every", 250))
    val_samples = int(settings.get("validation_samples", (maximum + 1) * 4))
    if min(batch_size, steps_per_epoch, max_steps, validate_every, save_every, int(settings.get("log_every", 10))) < 1 or val_samples < maximum + 1:
        raise ValueError("positive training intervals and validation coverage of every speaker count are required")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    validate_disjoint_manifests(args.train_manifest, args.validation_manifest)
    common = {"sample_rate": audio["sample_rate"], "duration": audio["duration"], "max_speakers": maximum,
                  "augmentation": config.get("augmentation", {}), "noise_root": args.noise_root,
                  "channels": model.audio_channels}
    train = SpeakerMixtureDataset(args.train_manifest, samples_per_epoch=steps_per_epoch * batch_size, seed=seed, **common)
    validation = SpeakerMixtureDataset(args.validation_manifest, samples_per_epoch=val_samples,
                                       seed=seed + 1000000, validation=True, **common)
    fingerprint = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for name, path in (("train", args.train_manifest), ("validation", args.validation_manifest))}
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume and (run / "last.pt").exists():
        raise ValueError("run directory already has a checkpoint; use --resume or a new directory")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("lr", 2e-4)),
                                  weight_decay=float(settings.get("weight_decay", 1e-4)))
    use_amp = device.type == "cuda" and settings.get("amp", True)
    if use_amp and not torch.cuda.is_bf16_supported():
        raise ValueError("this training recipe uses bf16; disable amp on GPUs without bf16 support")
    step, best = 0, float("inf")
    if args.resume:
        package = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous = package["config"]
        if previous["model"] != config["model"] or previous["audio"] != audio or package["manifest_sha256"] != fingerprint:
            raise ValueError("resume requires the same model, audio settings and dataset manifests")
        model.load_state_dict(package["state_dict"], strict=True)
        optimizer.load_state_dict(package["optimizer"])
        step, best = int(package["step"]), float(package["best_validation_loss"])
        torch.set_rng_state(package["rng_state"])
        if device.type == "cuda" and package.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(package["cuda_rng_state"])
    config_path = run / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "parameters": sum(p.numel() for p in model.parameters()), "seed": seed, "manifest_sha256": fingerprint,
                    "train_speakers": len(train.speakers), "validation_speakers": len(validation.speakers),
                    "training_max_speakers": maximum, "validation_seed": validation.seed}
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(event="start", step=step, **metadata)), flush=True)
    loss_options = config.get("loss", {})
    with (run / "history.jsonl").open("a", buffering=1) as history:
        def record(payload):
            text = json.dumps(dict(step=step, **payload), allow_nan=False)
            history.write(text + "\n")
            print(text, flush=True)

        def save():
            package = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                           "config": config, "model_type": config["model_type"], "best_validation_loss": best,
                           "manifest_sha256": fingerprint, "rng_state": torch.get_rng_state(),
                           "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
            _atomic_save(package, run / "last.pt")

        if step == 0:
            record(dict(event="initial_validation", **evaluate_speakers(model, validation, device, loss_options)))
        while step < max_steps:
            epoch, offset = divmod(step, steps_per_epoch)
            train.epoch = epoch
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=device.type == "cuda", generator=generator,
                                prefetch_factor=4 if args.num_workers else None,
                                persistent_workers=bool(args.num_workers))
            for batch_index, (mixture, target, counts) in enumerate(loader):
                if batch_index < offset:
                    continue
                model.train()
                mixture, target, counts = mixture.to(device), target.to(device), counts.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    output = model(mixture)
                    values = speaker_loss(output, target, counts, mixture=mixture, **loss_options)
                if not torch.isfinite(values["loss"]):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                values["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip", 5.)), error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step == 1 or step % int(settings.get("log_every", 10)) == 0:
                    record(dict(event="train", **{key: float(value.detach()) for key, value in values.items()}))
                if step % validate_every == 0 or step == max_steps:
                    metrics = evaluate_speakers(model, validation, device, loss_options)
                    record(dict(event="validation", peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, **metrics))
                    if metrics["overall"]["loss"] < best:
                        best = metrics["overall"]["loss"]
                        export_core_checkpoint(model, run / "best.pt", model_type=config["model_type"], config=config,
                                               metadata=dict(step=step, validation=metrics, **metadata))
                if step % save_every == 0 or step == max_steps:
                    save()
                if step >= max_steps:
                    break
    export_core_checkpoint(model, run / "model.pt", model_type=config["model_type"], config=config,
                           metadata=dict(step=step, **metadata))
    return run
