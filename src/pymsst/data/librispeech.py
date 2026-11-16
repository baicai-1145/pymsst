"""Download official LibriSpeech archives with checksum and safe extraction."""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://www.openslr.org/resources/12/"
SUBSETS = {"train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other"}


def download_librispeech(root, subsets):
    root = Path(root).resolve()
    subsets = list(subsets)
    if not subsets or set(subsets) - SUBSETS:
        raise ValueError("choose official LibriSpeech subset names")
    archives = root / ".archives"
    archives.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE_URL + "md5sum.txt", timeout=60) as response:
        checksums = {line.split()[1].lstrip("*"): line.split()[0] for line in response.read(65536).decode().splitlines() if len(line.split()) == 2}
    for subset in subsets:
        name = subset + ".tar.gz"
        marker = archives / (subset + ".complete")
        expected = checksums[name]
        if marker.is_file() and marker.read_text().strip() == expected and (root / "LibriSpeech" / subset).is_dir():
            print(f"{subset}: already extracted", flush=True)
            continue
        archive = archives / name
        if not archive.is_file():
            partial = archives / (name + ".part")
            with urllib.request.urlopen(BASE_URL + name, timeout=60) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            partial.replace(archive)
        digest = hashlib.md5()
        with archive.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"checksum mismatch for {archive}; remove the damaged archive before retrying")
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                destination = (root / member.name).resolve()
                if not destination.is_relative_to(root) or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError("unsafe archive member")
            handle.extractall(root)
        marker.write_text(expected + "\n")
        print(f"{subset}: verified and extracted", flush=True)
    return root / "LibriSpeech"
