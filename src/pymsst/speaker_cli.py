"""Dedicated variable-speaker training commands; imported by the main CLI without ML imports."""
from __future__ import annotations

import json
from pathlib import Path


def add_speaker_commands(subparsers):
    download = subparsers.add_parser("download-speakers", help="Download checksum-verified official LibriSpeech subsets.")
    download.add_argument("--root", type=Path, required=True)
    download.add_argument("--subsets", nargs="+", default=["train-clean-100", "dev-clean", "test-clean"])
    download.set_defaults(func=_download)

    prepare = subparsers.add_parser("prepare-speakers", help="Build identity-disjoint train/validation/test manifests.")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--layout", choices=("librispeech", "speaker-dirs"), default="librispeech")
    prepare.set_defaults(func=_prepare)

    train = subparsers.add_parser("train-speakers", help="Train adaptive BS/Mel-band RoFormer with dynamic mixtures.")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--train-manifest", type=Path, required=True)
    train.add_argument("--validation-manifest", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    train.add_argument("--steps", type=int)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--noise-root", type=Path)
    train.set_defaults(func=_train)


def _download(args):
    from .data.librispeech import download_librispeech
    download_librispeech(args.root, args.subsets)
    return 0


def _prepare(args):
    from .data.speakers import prepare_manifests
    print(json.dumps(prepare_manifests(args.root, args.output, layout=args.layout), indent=2))
    return 0


def _train(args):
    from .training.speaker_run import train_speakers
    train_speakers(args)
    return 0
