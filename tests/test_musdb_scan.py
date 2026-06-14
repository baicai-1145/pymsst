from pathlib import Path

from pymsst.data import scan_musdb_root, scan_musdb_split


def _touch_track(root: Path, split: str, name: str, stems: tuple[str, ...]) -> None:
    track = root / split / name
    track.mkdir(parents=True)
    for stem in stems:
        (track / f"{stem}.wav").write_bytes(b"")


def test_scan_musdb_split_finds_complete_tracks(tmp_path):
    stems = ("mixture", "vocals", "drums", "bass", "other")
    _touch_track(tmp_path, "train", "Song A", stems)

    scan = scan_musdb_split(tmp_path, "train")

    assert scan.ok
    assert len(scan.tracks) == 1
    assert scan.tracks[0].name == "Song A"
    assert sorted(scan.tracks[0].files) == sorted(stems)


def test_scan_musdb_split_reports_missing_stems(tmp_path):
    _touch_track(tmp_path, "train", "Song A", ("mixture", "vocals"))

    scan = scan_musdb_split(tmp_path, "train")

    assert not scan.ok
    assert not scan.tracks
    assert scan.incomplete[0].name == "Song A"
    assert "drums(.wav|.flac)" in scan.incomplete[0].missing_files


def test_scan_musdb_root_accepts_multiple_splits(tmp_path):
    stems = ("mixture", "vocals", "drums", "bass", "other")
    _touch_track(tmp_path, "train", "Song A", stems)
    _touch_track(tmp_path, "test", "Song B", stems)

    scans = scan_musdb_root(tmp_path)

    assert set(scans) == {"train", "test"}
    assert len(scans["train"].tracks) == 1
    assert len(scans["test"].tracks) == 1
