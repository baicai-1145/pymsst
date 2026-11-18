# Model card: pymss adaptive speaker separation (44.1 kHz stereo)

## Summary

`pymsst` separates an arbitrary number of speakers from a single stereo mixture without being told
how many people are talking. The model predicts one stem per active speaker plus a voice-activity
head that decides how many stems to trust, so the output count adapts per clip.

- Parameters: 40,282,606
- Sample rate: 44,100 Hz, stereo in / stereo out
- Training targets: per-speaker complex ratio masks, silence head, speaker-existence head
- Best validation SI-SDRi observed during development: **12.19 dB** (16,500 steps, 4,710 training identities)

## Training data

The released recipe trains on the following corpora. **All are research corpora and several forbid
commercial use; the resulting weights inherit the strictest of those restrictions** — see Licensing.

| Corpus | Used for | Licence |
| --- | --- | --- |
| VCTK (`sanchit-gandhi/vctk` mirror of CSTR VCTK) | multi-mic single-speaker speech, 109 identities | CC BY 4.0 |
| VocalSet | singing voice, 20 identities | CC BY 4.0 (research use) |
| Emilia YODAS-EN | conversational speech, ~4,500 identities (30 parquet shards) | CC BY-NC 4.0 |
| MUSDB18-HQ | mixture realism / musical accompaniment (earlier experiments) | **non-commercial** |

Final mixture manifests: 107,265 training utterances across 4,710 identities, 8,447 validation
utterances across 262 identities, 7,642 test utterances across 262 identities.

## Licensing

- **Code**: MIT (see `LICENSE`).
- **Weights**: **non-commercial use only.** Emilia YODAS-EN is CC BY-NC 4.0 and MUSDB18-HQ is
  non-commercial; a model trained on them is a derivative work under those terms. Publishing the
  code under MIT does not grant any right to use these weights commercially or to redistribute them
  under a permissive licence.
- If you need commercially usable weights, retrain on permissively licensed corpora (for example
  VCTK + VocalSet only) — the recipe and code are unchanged, only the corpus list differs.

## Evaluation

Metrics are produced by `python -m pymsst train-speakers` at the end of a run and written to
`test_metrics.json` / `test_metrics_clean.json` in the run directory.

- `oracle_si_sdr_i`: SI-SDR improvement with oracle speaker assignment
- `count_accuracy` / `count_mae`: how often the predicted speaker count matches the truth
- `per_count`: the same metrics split by the true number of speakers

Reference points from the released checkpoint (`seeds/slim-00500.pt` weights, evaluated on the
held-out test split of 7,642 utterances / 262 identities):

| Split | SI-SDRi | count accuracy | count MAE |
| --- | --- | --- | --- |
| test | 10.55 | 0.85 | 0.17 |
| test (clean subset) | 10.54 | 0.83 | 0.17 |

Per speaker count (test): 2 speakers 12.60 dB, 3 speakers 10.38 dB, 4 speakers 10.59 dB,
5 speakers 8.62 dB.

Reference points from the development runs (validation split, 262 unseen identities):

| Step | SI-SDRi | count accuracy | count MAE |
| --- | --- | --- | --- |
| 20 | -0.34 | 0.17 | 1.50 |
| 1,500 | 9.69 | 0.75 | 0.25 |
| 4,500 | 10.55 | 0.75 | 0.25 |
| 7,500 | 11.13 | 0.79 | 0.25 |
| 10,500 | 11.49 | 0.83 | 0.17 |
| 13,500 | 11.77 | 0.92 | 0.08 |
| 16,500 | 12.19 | 0.92 | 0.08 |
| 18,000 | 12.06 | 0.88 | 0.12 |

## Limitations

- Trained on studio and read speech plus sung voice; heavily reverberant or far-field recordings are
  out of distribution.
- The speaker-count head was trained on at most six simultaneous speakers.
- Musical mixtures from MUSDB18-HQ were used only in early experiments; the released run focuses on
  speech.
- Identity labels come from corpus directory names; label noise in Emilia is inherited.

## Reproducing

```bash
python -m pymsst fetch-corpus --name vctk-hf     --root data/corpora
python -m pymsst fetch-corpus --name vocalset    --root data/corpora
python -m pymsst fetch-corpus --name emilia-yodas-en --root data/corpora --max-shards 30
python -m pymsst link-speakers --root data/corpora --output data/linked --drop mic2
python -m pymsst prepare-speakers --root data/linked --output data/manifests --layout speaker-dirs
python -m pymsst train-speakers \
  --config configs/adaptive_bs_roformer_44k_stereo_large.yaml \
  --train-manifest data/manifests/train.jsonl \
  --validation-manifest data/manifests/validation.jsonl \
  --run-dir runs/adaptive-44k-stereo --device cuda --num-workers 8 --steps 18000
```
