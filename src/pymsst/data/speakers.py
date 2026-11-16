"""Speaker-disjoint manifests and reproducible on-the-fly mixtures of 0..N identities."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def prepare_manifests(root, output, *, layout="librispeech"):
    """LibriSpeech uses its official splits; custom speaker directories use an identity hash."""
    import soundfile as sf

    root, output = Path(root).resolve(), Path(output)
    if not root.is_dir() or layout not in {"librispeech", "speaker-dirs"}:
        raise ValueError("a dataset directory and a supported layout are required")
    rows = {name: [] for name in ("train", "validation", "test")}
    seen_speakers = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".wav", ".flac"}:
            continue
        parts = path.relative_to(root).parts
        if layout == "librispeech":
            split_index = next((i for i, p in enumerate(parts) if p.startswith(("train-", "dev-", "test-"))), None)
            if split_index is None or len(parts) < split_index + 4:
                raise ValueError(f"not an official LibriSpeech path: {path}")
            speaker = "librispeech:" + parts[split_index + 1]
            split = {"train": "train", "dev": "validation", "test": "test"}[parts[split_index].split("-")[0]]
        else:
            if len(parts) < 2:
                raise ValueError(f"expected root/speaker_id/audio under {root}")
            speaker = parts[0]
            bucket = int(hashlib.sha256(speaker.encode()).hexdigest(), 16) % 20
            split = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
        if speaker in seen_speakers and seen_speakers[speaker] != split:
            raise ValueError(f"speaker {speaker} appears in different splits")
        seen_speakers[speaker] = split
        info = sf.info(path)
        if info.frames < info.samplerate / 4:
            continue
        rows[split].append({"path": str(path), "speaker": speaker, "sample_rate": info.samplerate, "frames": info.frames})
    if not any(rows.values()):
        raise ValueError(f"no usable audio found in {root}")
    output.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in items))
    summary = {split: {"utterances": len(items), "speakers": len({r['speaker'] for r in items})} for split, items in rows.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_speaker_manifest(path):
    path = Path(path)
    groups = defaultdict(list)
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("speaker"), str) or not row["speaker"]:
            raise ValueError(f"{path}:{number}: speaker must be a nonempty identity string")
        audio = Path(row["path"])
        audio = audio if audio.is_absolute() else path.parent / audio
        if not audio.is_file():
            raise ValueError(f"missing audio: {audio}")
        groups[row["speaker"]].append(str(audio.resolve()))
    if not groups:
        raise ValueError(f"empty manifest: {path}")
    return dict(groups)


def validate_disjoint_manifests(train, validation):
    a, b = load_speaker_manifest(train), load_speaker_manifest(validation)
    if set(a) & set(b):
        raise ValueError("training and validation must have disjoint speaker identities")
    if {p for paths in a.values() for p in paths} & {p for paths in b.values() for p in paths}:
        raise ValueError("training and validation must have disjoint audio files")


def _read_segment(path, samples, sample_rate, rng, channels=1):
    import soundfile as sf
    from scipy.signal import resample_poly

    with sf.SoundFile(path) as handle:
        frames = min(len(handle), math.ceil(samples * handle.samplerate / sample_rate))
        start = int(rng.integers(max(1, len(handle) - frames + 1)))
        handle.seek(start)
        audio = handle.read(frames, dtype="float32", always_2d=True)
        source_rate = handle.samplerate
    audio = audio.mean(axis=1, keepdims=True) if channels == 1 else (audio[:, :1].repeat(2, axis=1) if audio.shape[1] == 1 else audio[:, :2])
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, source_rate // divisor, axis=0)
    audio = audio[:samples].astype(np.float32)
    return audio[:, 0] if channels == 1 else np.ascontiguousarray(audio.T)


def _spatialize(audio, sample_rate, rng, options):
    """Place one voice in a stereo field: constant-power pan, inter-ear delay, shared room."""
    theta = rng.uniform(0, np.pi / 2)
    gains = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) * np.sqrt(2)
    near = 0 if gains[0] >= gains[1] else 1
    delay = min(int(round(abs(np.cos(theta)) * options.get("max_itd_ms", 0.7) * sample_rate / 1000)), len(audio) - 1)
    channels = np.stack([audio, audio]) * gains[:, None]
    if delay > 0:
        channels[near] = np.concatenate([np.zeros(delay, np.float32), audio[:len(audio) - delay]]) * gains[near]
    if rng.random() < options.get("reverb_probability", 0.4):
        rt60 = rng.uniform(0.1, options.get("max_rt60", 0.6))
        size = max(2, int(sample_rate * rt60))
        decay = np.exp(-6.91 * np.arange(size) / size)
        direct = min(int(round(abs(np.cos(theta)) * 0.003 * sample_rate)), size)
        fft_size = 1 << (len(audio) + size + direct - 2).bit_length()
        spectrum = np.fft.rfft(channels, fft_size, axis=1)
        for channel in range(2):
            # Decorrelated tails keep the image wide; the direct path stays aligned for both ears.
            impulse = rng.normal(size=size) * decay
            impulse *= rng.uniform(0.1, 0.5) / max(np.linalg.norm(impulse), 1e-8)
            response = np.zeros(size + direct)
            response[direct] = 1.0
            response[direct:] += impulse
            channels[channel] = np.fft.irfft(spectrum[channel] * np.fft.rfft(response, fft_size), fft_size)[:len(audio)]
    return channels.astype(np.float32)


def _augment_voice(audio, sample_rate, rng, options, reverb=True):
    if rng.random() < options.get("pitch_probability", 0.2):
        import librosa
        semitones = rng.uniform(-options.get("pitch_semitones", 2.), options.get("pitch_semitones", 2.))
        n_fft = min(2048, 1 << (max(2, len(audio)).bit_length() - 1))
        audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones, n_fft=n_fft)
    if rng.random() < options.get("eq_probability", 0.6):
        spectrum = np.fft.rfft(audio)
        anchors = rng.uniform(-options.get("eq_db", 6.), options.get("eq_db", 6.), size=8)
        response = np.interp(np.linspace(0, 1, len(spectrum)), np.linspace(0, 1, len(anchors)), anchors)
        audio = np.fft.irfft(spectrum * 10 ** (response / 20), n=len(audio)).astype(np.float32)
    if reverb and rng.random() < options.get("reverb_probability", 0.4):
        rt60 = rng.uniform(0.1, options.get("max_rt60", 0.6))
        size = max(2, int(sample_rate * rt60))
        impulse = rng.normal(size=size) * np.exp(-6.91 * np.arange(size) / size)
        impulse[0] = 0
        impulse *= rng.uniform(0.1, 0.5) / max(np.linalg.norm(impulse), 1e-8)
        impulse[0] = 1
        fft_size = 1 << (len(audio) + size - 2).bit_length()
        audio = np.fft.irfft(np.fft.rfft(audio, fft_size) * np.fft.rfft(impulse, fft_size), fft_size)[:len(audio)]
    return np.asarray(audio, dtype=np.float32)


class SpeakerMixtureDataset(Dataset):
    def __init__(self, manifest, *, sample_rate=16000, duration=4., max_speakers=5,
                 samples_per_epoch=2000, seed=0, validation=False, augment=True,
                 augmentation=None, noise_root=None, channels=1):
        self.groups = load_speaker_manifest(manifest)
        self.speakers = sorted(self.groups)
        self.sample_rate, self.length = int(sample_rate), int(sample_rate * duration)
        self.max_speakers, self.size = int(max_speakers), int(samples_per_epoch)
        self.channels = int(channels)
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 (mono) or 2 (stereo)")
        if self.max_speakers < 1 or len(self.speakers) < self.max_speakers:
            raise ValueError("manifest needs at least max_speakers distinct identities")
        if self.sample_rate < 1 or self.length < 16 or self.size < 1:
            raise ValueError("positive rate, duration and epoch size are required")
        self.seed, self.validation, self.augment, self.epoch = seed, validation, augment, 0
        self.options = dict(augmentation or {})
        self.noises = sorted(str(p) for p in Path(noise_root).rglob("*") if p.suffix.lower() in {".wav", ".flac"}) if noise_root else []
        if noise_root and not self.noises:
            raise ValueError("noise_root contains no WAV/FLAC files")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index + (0 if self.validation else self.epoch * self.size))
        if self.validation:
            count = index % (self.max_speakers + 1)
        else:
            weights = np.ones(self.max_speakers + 1)
            weights[0] = self.options.get("empty_weight", 0.15)
            count = int(rng.choice(self.max_speakers + 1, p=weights / weights.sum()))
        chosen = rng.choice(self.speakers, size=count, replace=False)
        targets = np.zeros((self.max_speakers, self.channels, self.length), dtype=np.float32)
        for slot, speaker in enumerate(chosen):
            for _ in range(10):
                wanted = int(rng.uniform(self.options.get("min_active_fraction", 0.5), 1.) * self.length)
                voice = _read_segment(rng.choice(self.groups[speaker]), wanted, self.sample_rate, rng)
                if len(voice) and np.sqrt(np.mean(voice ** 2)) > 1e-5:
                    break
            else:
                raise ValueError(f"no audible segment found for speaker {speaker}")
            start = int(rng.integers(self.length - len(voice) + 1))
            track = np.zeros(self.length, dtype=np.float32)
            track[start:start + len(voice)] = voice
            if self.augment:
                track = _augment_voice(track, self.sample_rate, rng, self.options, reverb=self.channels == 1)
            track = _spatialize(track, self.sample_rate, rng, self.options) if self.channels == 2 else track[None]
            rms = max(float(np.sqrt(np.mean(track ** 2))), 1e-6)
            targets[slot] = track * (10 ** (rng.uniform(-6, 3) / 20) * 0.08 / rms)
        mixture = targets.sum(axis=0)
        # Noise belongs only to the mixture. Targets include each voice's EQ, pitch and room response.
        if self.augment and rng.random() < self.options.get("noise_probability", 0.5):
            if self.noises:
                noise = _read_segment(rng.choice(self.noises), self.length, self.sample_rate, rng, channels=self.channels)
            else:
                noise = rng.normal(size=(self.channels, self.length)).astype(np.float32)
            noise = np.pad(noise, ((0, 0), (0, self.length - noise.shape[-1])))
            noise_rms = max(float(np.sqrt(np.mean(noise ** 2))), 1e-6)
            signal_rms = max(float(np.sqrt(np.mean(mixture ** 2))), 0.02)
            mixture = mixture + noise * (signal_rms * 10 ** (-rng.uniform(10, 30) / 20) / noise_rms)
        peak = max(float(np.abs(mixture).max()), float(np.abs(targets).max()), 0.9)
        mixture, targets = mixture * (0.9 / peak), targets * (0.9 / peak)
        return torch.from_numpy(mixture), torch.from_numpy(targets), torch.tensor(count, dtype=torch.long)
