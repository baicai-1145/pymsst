from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from pymsst.data import MusdbChunkDataset, MusdbDataModule, MusdbTrackInfo, read_wav_chunk, read_wav_info, scan_musdb_split


def _write_wav(path: Path, frames: int = 128, channels: int = 2, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.linspace(-0.5, 0.5, num=frames * channels, dtype=np.float32).reshape(frames, channels)
    pcm = np.clip(values * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _write_track(root: Path, split: str, name: str) -> None:
    for stem in ("mixture", "vocals", "drums", "bass", "other"):
        _write_wav(root / split / name / f"{stem}.wav")


def test_read_wav_chunk_returns_channels_first_and_pads(tmp_path):
    path = tmp_path / "audio.wav"
    _write_wav(path, frames=8, channels=1)

    info = read_wav_info(path)
    chunk = read_wav_chunk(path, start=4, frames=8, target_channels=2)

    assert info.sample_rate == 44100
    assert info.channels == 1
    assert chunk.shape == (2, 8)
    assert np.allclose(chunk[0], chunk[1])
    assert np.all(chunk[:, 4:] == 0.0)


def test_musdb_chunk_dataset_loads_target_and_mixture(tmp_path):
    _write_track(tmp_path, "train", "Song A")
    scan = scan_musdb_split(tmp_path, "train")
    info = MusdbTrackInfo(track=scan.tracks[0], frames=read_wav_info(scan.tracks[0].files["mixture"]).frames)
    dataset = MusdbChunkDataset(
        [info],
        instruments=("vocals", "drums", "bass", "other"),
        target_instrument="vocals",
        chunk_size=64,
        sample_rate=44100,
        channels=2,
        examples=1,
        random_chunks=False,
    )

    target, mixture = dataset[0]

    assert target.shape == (2, 64)
    assert mixture.shape == (2, 64)
    assert target.dtype == mixture.dtype


def test_musdb_chunk_dataset_stem_mix_does_not_advance_torch_rng(tmp_path):
    _write_track(tmp_path, "train", "Song A")
    scan = scan_musdb_split(tmp_path, "train")
    info = MusdbTrackInfo(track=scan.tracks[0], frames=read_wav_info(scan.tracks[0].files["mixture"]).frames)
    dataset = MusdbChunkDataset(
        [info],
        instruments=("vocals", "drums", "bass", "other"),
        target_instrument="vocals",
        chunk_size=64,
        sample_rate=44100,
        channels=2,
        examples=1,
        random_chunks=True,
        mix_from_stems=True,
    )
    torch.manual_seed(123)
    state = torch.random.get_rng_state()

    target, mixture = dataset[0]

    assert target.shape == (2, 64)
    assert mixture.shape == (2, 64)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_musdb_datamodule_val_tracks_zero_keeps_all_tracks_for_training(tmp_path):
    _write_track(tmp_path, "train", "Song A")
    _write_track(tmp_path, "train", "Song B")
    data = MusdbDataModule(
        tmp_path,
        instruments=("vocals", "drums", "bass", "other"),
        target_instrument="vocals",
        chunk_size=64,
        sample_rate=44100,
        channels=2,
        batch_size=1,
        steps_per_epoch=1,
        val_tracks=0,
    )

    data.setup()

    assert data.train_dataset is not None
    assert len(data.train_dataset.tracks) == 2
    assert data.val_dataset is None
    assert data.val_dataloader() == []


def test_musdb_datamodule_single_track_val_tracks_zero_disables_validation(tmp_path):
    _write_track(tmp_path, "train", "Only Song")
    data = MusdbDataModule(
        tmp_path,
        instruments=("vocals", "drums", "bass", "other"),
        target_instrument="vocals",
        chunk_size=64,
        sample_rate=44100,
        channels=2,
        batch_size=1,
        steps_per_epoch=1,
        val_tracks=0,
    )

    data.setup()

    assert data.train_dataset is not None
    assert len(data.train_dataset.tracks) == 1
    assert data.val_dataset is None


@pytest.mark.parametrize("val_tracks", (None, 1))
def test_musdb_datamodule_single_track_validation_requires_disjoint_track(tmp_path, val_tracks):
    _write_track(tmp_path, "train", "Only Song")
    data = MusdbDataModule(
        tmp_path,
        instruments=("vocals", "drums", "bass", "other"),
        target_instrument="vocals",
        chunk_size=64,
        sample_rate=44100,
        channels=2,
        batch_size=1,
        steps_per_epoch=1,
        val_tracks=val_tracks,
    )

    with pytest.raises(ValueError, match="at least two complete tracks"):
        data.setup()


def test_musdb_datamodule_validation_split_uses_train_split_only(tmp_path):
    _write_track(tmp_path, "train", "Train A")
    _write_track(tmp_path, "train", "Train B")
    _write_track(tmp_path, "train", "Train C")
    _write_track(tmp_path, "test", "Official Test")
    data = MusdbDataModule(
        tmp_path,
        instruments=("vocals", "drums", "bass", "other"),
        target_instrument="vocals",
        chunk_size=64,
        sample_rate=44100,
        channels=2,
        batch_size=1,
        steps_per_epoch=1,
        val_tracks=1,
        seed=0,
    )

    data.setup()

    assert data.train_dataset is not None
    assert data.val_dataset is not None
    train_names = {track.track.name for track in data.train_dataset.tracks}
    val_names = {track.track.name for track in data.val_dataset.tracks}

    assert train_names | val_names == {"Train A", "Train B", "Train C"}
    assert train_names.isdisjoint(val_names)
    assert "Official Test" not in train_names
    assert "Official Test" not in val_names
