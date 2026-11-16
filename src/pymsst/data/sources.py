"""Fetch and normalize public single-speaker corpora into identity-per-directory trees."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

CORPORA = {
    # name: (url, archive name, speaker directory pattern, prefix)
    "vctk": ("https://datashare.ed.ac.uk/download/DS_10283_3443.zip", "VCTK-Corpus-0.92.zip", r"^p\d{3}$", "vctk"),
    "vocalset": ("https://zenodo.org/api/records/10200775/files/VocalSet.zip/content", "VocalSet.zip", r"^(?:male|female)\d{1,2}$", "vocalset"),
}
AUDIO_SUFFIXES = {".wav", ".flac"}


def _expected_md5(url):
    """Zenodo publishes per-file md5; datashare serves no sidecar, so crc checks cover it."""
    record = re.search(r"zenodo\.org/api/records/(\d+)/", url)
    if not record:
        return None
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record.group(1)}", timeout=60) as response:
        payload = json.load(response)
    for item in payload.get("files", []):
        if item["links"]["self"] in (url, url.replace("/content", "")):
            return item.get("checksum", "").removeprefix("md5:")
    return None


def download_archive(name, root, *, timeout=60):
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; known: {', '.join(sorted(CORPORA))}")
    url, filename, _, _ = CORPORA[name]
    archives = Path(root) / ".archives"
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / filename
    marker = archives / (filename + ".md5")
    expected = _expected_md5(url)
    if archive.is_file() and expected and marker.is_file() and marker.read_text().strip() == expected:
        print(f"{name}: archive already verified", flush=True)
        return archive
    partial = archives / (filename + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("ab" if offset else "wb") as handle:
        shutil.copyfileobj(response, handle, length=4 * 1024 * 1024)
    digest = hashlib.md5()
    with partial.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if expected and digest.hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {partial}; delete it before retrying")
    partial.replace(archive)
    marker.write_text((expected or digest.hexdigest()) + "\n")
    print(f"{name}: downloaded {archive.stat().st_size} bytes", flush=True)
    return archive


def extract_archive(archive, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError(f"unsafe archive member: {member.filename}")
        bad = handle.testzip()
        if bad:
            raise ValueError(f"corrupt archive member: {bad}")
        handle.extractall(root)
    return root


def link_speaker_tree(roots, output, *, patterns=None, drop_patterns=None):
    """Link every audio file into output/<corpus>:<speaker>/<name>, refusing silent identity loss."""
    output = Path(output).resolve()
    patterns = patterns or {name: spec[2] for name, spec in CORPORA.items()}
    dropped = [re.compile(p) for p in (drop_patterns or [])]
    counts, seen = {}, {}
    for root in roots:
        root = Path(root).resolve()
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in AUDIO_SUFFIXES or any(p.search(path.name) for p in dropped):
                continue
            parts = path.relative_to(root).parts[:-1]
            for depth in range(len(parts) - 1, -1, -1):
                for name, pattern in patterns.items():
                    if re.match(pattern, parts[depth]):
                        speaker = f"{name}:{parts[depth]}"
                        break
                else:
                    continue
                break
            else:
                unlabelled = seen.setdefault("unlabelled", 0)
                seen["unlabelled"] = unlabelled + 1
                continue
            target = output / speaker
            target.mkdir(parents=True, exist_ok=True)
            link = target / (path.stem + path.suffix.lower())
            if not link.exists():
                link.symlink_to(path)
                counts[speaker] = counts.get(speaker, 0) + 1
    if seen.get("unlabelled"):
        raise ValueError(f"{seen['unlabelled']} files had no speaker directory matching {sorted(patterns)}")
    if not counts:
        raise ValueError("no speaker audio found under the given roots")
    return {"speakers": len(counts), "files": sum(counts.values())}
