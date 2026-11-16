# Unknown-count speaker separation on L4

Use the `feat/adaptive-speaker-roformer` branches of baicai-1145/pymss-core, pymsst and pymss in sibling directories. Install the source core together with the training package; released core packages do not contain these experimental architectures.

```bash
python -m pip install -e ../pymss-core -e ".[speakers]"
python -m pymsst download-speakers --root /content/data \
  --subsets train-clean-100 dev-clean test-clean
python -m pymsst prepare-speakers --root /content/data/LibriSpeech \
  --output /content/data/manifests
python -m pymsst train-speakers \
  --config configs/adaptive_bs_roformer_l4.yaml \
  --train-manifest /content/data/manifests/train.jsonl \
  --validation-manifest /content/data/manifests/validation.jsonl \
  --run-dir /content/runs/adaptive-bs --device cuda --num-workers 2
```

The downloader uses official [OpenSLR LibriSpeech](https://www.openslr.org/12/) archives and published MD5 checksums, rejects unsafe archive entries, and reuses completed downloads. LibriSpeech is an English read-speech starting point, not a replacement for conversational or singing data. The default subsets include 100 hours of training speech plus official dev/test sets. Download artifacts remain on the runtime. No Google credentials belong in a notebook, repository or checkpoint.

For other single-speaker corpora, organize files as `root/speaker_id/**/*.wav` or `.flac` and use `prepare-speakers --layout speaker-dirs`. Speaker IDs must refer to real identities, not individual files or augmented copies. Custom directory splits use a stable identity hash (90% train, 5% validation, 5% test). Merge/customize manifests if a small corpus leaves too few validation speakers.

Each JSONL row contains `path` and `speaker`; relative paths resolve from the manifest directory. Training checks that identities and paths do not overlap across train/validation. Do not build training manifests from the test set. The test manifest is held out and not read by training.

Each mixture draws distinct identities and samples 0..5 people, random gains and offsets/partial overlap. EQ, pitch shifts and per-source synthetic reverberation are applied to each source and its target together. Noise is added only to the mixture; it is not a speaker. The default noise is synthetic; `--noise-root /path/to/noise` uses WAV/FLAC environmental recordings instead. Pitch variants of one speaker remain the same identity. Exact alignment is preserved after augmentation and common peak scaling.

Validation mixes are deterministic, augmented, speaker-disjoint, and balanced by speaker count. Logs report count accuracy/MAE and **oracle-count** matched SI-SDR/SI-SDR improvement separately. Oracle-count metrics evaluate the first N outputs given true N; they must not be presented as automatic-count end-to-end quality. Zero-speaker items have no SI-SDR metric. SI-SDR improvement is reported only for two or more speakers; a clean single-speaker mixture already equals its target and makes the improvement baseline degenerate. No improvement over the original model is claimed without comparative experiments.

The BS recipe uses dim 128, depth 4, 32 bands, 512-point STFT/128 hop, 4-second mono 16 kHz chunks and bf16 on L4. The Mel recipe substitutes 32 overlapping mel bands. Batch size 2 is a starting configuration; actual memory/throughput must be measured. Full training is configured for 20,000 updates; `--steps 20` is a smoke run and `--steps 2000` is an initial experiment, neither a quality guarantee.

Outputs in the run directory:

- `config.yaml`, `metadata.json`: reproducible configuration, source manifest hashes and hardware.
- `history.jsonl`: training and per-count validation records.
- `last.pt`: atomic training checkpoint with optimizer, RNG and completed step.
- `best.pt`: core-compatible weights selected by validation loss.
- `model.pt`: core-compatible weights at the final requested step.

Resume with the same command plus `--resume /content/runs/adaptive-bs/last.pt`. `--steps` is the absolute target update count, not additional updates. Model/audio settings and manifest hashes are checked. Keep the sampling/training settings unchanged for reproducible continuation. An existing run with checkpoints requires explicit resume. Save/copy checkpoints outside the ephemeral runtime before it is released.

Inference is provided by `python -m pymss.speakers` or `pymss separate-speakers`; see the companion pymss branch. Long-recording chunk identity stitching and singing-domain training remain separate work.
