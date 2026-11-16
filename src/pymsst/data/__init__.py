"""Dataset utilities for pymsst."""

from .audio import WavInfo, read_wav_chunk, read_wav_info
from .musdb import DEFAULT_STEMS, IncompleteTrack, MusdbSplitScan, MusdbTrack, scan_musdb_root, scan_musdb_split


def __getattr__(name):
    if name not in {"MusdbChunkDataset", "MusdbDataModule", "MusdbTrackInfo"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(".datamodule", __name__), name)
    globals()[name] = value
    return value


__all__ = (
    "DEFAULT_STEMS",
    "IncompleteTrack",
    "MusdbChunkDataset",
    "MusdbDataModule",
    "MusdbSplitScan",
    "MusdbTrack",
    "MusdbTrackInfo",
    "WavInfo",
    "read_wav_chunk",
    "read_wav_info",
    "scan_musdb_root",
    "scan_musdb_split",
)
