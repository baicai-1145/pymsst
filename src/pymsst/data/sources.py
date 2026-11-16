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
    "vctk": ("https://datashare.ed.ac.uk/download/DS_10283_3443.zip", "VCTK-Corpus-0.92.zip", r"^(?:p\d{3}|s\d+)$", "vctk"),
    "vocalset": ("https://zenodo.org/api/records/10200775/files/VocalSet.zip/content", "VocalSet.zip", r"^(?:male|female)\d{1,2}$", "vocalset"),
}
# datashare.ed.ac.uk answers 403 to Colab egress; this HF mirror is the same 48 kHz corpus.
HF_CORPORA = {"vctk-hf": ("sanchit-gandhi/vctk", "vctk")}
AUDIO_SUFFIXES = {".wav", ".flac"}
# macOS zips carry resource forks (__MACOSX/._*); they are audio-suffixed but not audio.
ARCHIVE_JUNK = {"__MACOSX", ".DS_Store"}


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


def extract_parquet_audio(source, output, *, speaker_key="speaker_id", prefix=""):
    """Decode the audio bytes of one HF audio shard into <output>/<prefix><speaker>/<name>.flac."""
    import io

    import pyarrow.parquet as pq
    import soundfile as sf

    output = Path(output)
    shard = pq.ParquetFile(source)
    written, rates = 0, set()
    for batch in shard.iter_batches(batch_size=256, columns=[speaker_key, "audio"]):
        speakers = batch.column(0).to_pylist()
        encoded = batch.column(1).to_pylist()
        for speaker, item in zip(speakers, encoded):
            payload = item.get("bytes") if isinstance(item, dict) else item
            if not payload:
                continue
            samples, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
            rates.add(rate)
            directory = output / f"{prefix}{speaker}"
            directory.mkdir(parents=True, exist_ok=True)
            sf.write(directory / f"{speaker}_{written:06d}.flac", samples, rate, format="FLAC")
            written += 1
    return {"written": written, "sample_rates": sorted(rates)}


def fetch_hf_corpus(name, root, *, timeout=120):
    """Download and decode an HF-hosted audio corpus one shard at a time, deleting each shard after use."""
    if name not in HF_CORPORA:
        raise ValueError(f"unknown HF corpus {name!r}; known: {', '.join(sorted(HF_CORPORA))}")
    import urllib.request

    repo, prefix = HF_CORPORA[name]
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true", timeout=timeout) as response:
        listing = json.load(response)
    shards = sorted(item["path"] for item in listing if item.get("type") == "file" and item["path"].endswith(".parquet"))
    if not shards:
        raise ValueError(f"{repo} has no parquet shards")
    root = Path(root).resolve()
    scratch = root / ".archives"
    scratch.mkdir(parents=True, exist_ok=True)
    totals, rates = 0, set()
    for number, shard in enumerate(shards, 1):
        local = scratch / f"{name}-{number:03d}.parquet"
        if not local.is_file():
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{shard}"
            with urllib.request.urlopen(url, timeout=timeout) as response, local.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        summary = extract_parquet_audio(local, root, prefix=prefix + "/")
        totals += summary["written"]
        rates.update(summary["sample_rates"])
        local.unlink()
        print(f"{name}: shard {number}/{len(shards)} decoded, {totals} files total", flush=True)
    return {"written": totals, "sample_rates": sorted(rates)}


def link_speaker_tree(roots, output, *, patterns=None, drop_patterns=None):
    """Link every audio file into output/<corpus>:<speaker>/<name>, refusing silent identity loss."""
    output = Path(output).resolve()
    patterns = patterns or {name: spec[2] for name, spec in CORPORA.items()}
    dropped = [re.compile(p) for p in (drop_patterns or [])]
    counts, seen = {}, {}
    for root in roots:
        root = Path(root).resolve()
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in AUDIO_SUFFIXES or path.name.startswith("._"):
                continue
            if ARCHIVE_JUNK & set(path.relative_to(root).parts) or any(p.search(path.name) for p in dropped):
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
