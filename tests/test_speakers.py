from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from pymss_core.modules.bs_roformer import AdaptiveBSRoformer
from pymss_core.modules.bs_roformer.adaptive import AdaptiveSeparationOutput

from pymsst.data.speakers import SpeakerMixtureDataset, prepare_manifests, validate_disjoint_manifests
from pymsst.training.speaker_run import train_speakers
from pymsst.training.speakers import speaker_loss


def manifest(tmp_path, prefix, speakers=5):
    rows = []
    for i in range(speakers):
        path = tmp_path / f'{prefix}_{i}.wav'
        rng = np.random.default_rng(i)
        wave = rng.normal(0, 0.1, 800).astype(np.float32)
        sf.write(path, wave, 8000)
        rows.append({'path': str(path), 'speaker': f'{prefix}:{i}'})
    path = tmp_path / f'{prefix}.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def test_mixtures_cover_zero_to_five_and_are_reproducible(tmp_path):
    source = manifest(tmp_path, 'train')
    dataset = SpeakerMixtureDataset(source, sample_rate=8000, duration=.1, max_speakers=5,
                                    samples_per_epoch=12, validation=True, augment=False)
    for index in range(6):
        mixture, target, count = dataset[index]
        assert int(count) == index
        assert mixture.shape == (1, 800) and target.shape == (5, 1, 800)
        assert torch.allclose(mixture, target.sum(0), atol=1e-7)
        assert torch.count_nonzero(target[index:]) == 0
        assert all(torch.equal(a, b) for a, b in zip(dataset[index], dataset[index]))
    train = SpeakerMixtureDataset(source, sample_rate=8000, duration=.1, samples_per_epoch=12, augment=False)
    first = train[0][0]
    train.epoch = 1
    assert not torch.equal(first, train[0][0])


def test_augmentations_preserve_target_mix_alignment(tmp_path):
    source = manifest(tmp_path, 'train')
    augmentation = {'pitch_probability': 1., 'eq_probability': 1., 'reverb_probability': 1., 'noise_probability': 0.}
    dataset = SpeakerMixtureDataset(source, sample_rate=8000, duration=.1, samples_per_epoch=6,
                                    validation=True, augmentation=augmentation)
    mixture, target, count = dataset[3]
    assert int(count) == 3
    assert torch.isfinite(target).all()
    assert torch.allclose(mixture, target.sum(0), atol=1e-6)
    assert target.abs().max() <= .901
    dataset.options['noise_probability'] = 1.
    mixture, target, _ = dataset[3]
    assert (mixture - target.sum(0)).square().mean() > 0


def test_unaugmented_mixture_is_exactly_the_sum_of_targets(tmp_path):
    source = manifest(tmp_path, 'train')
    dataset = SpeakerMixtureDataset(source, sample_rate=8000, duration=.1, max_speakers=5, samples_per_epoch=6,
                                    validation=True, augment=False, noise_root=None,
                                    augmentation={'noise_probability': 1.})
    for index in range(6):
        mixture, target, count = dataset[index]
        assert torch.allclose(mixture, target.sum(0), atol=1e-7), 'no noise may leak into a clean mixture'
        assert torch.count_nonzero(target[int(count):]) == 0


def test_stereo_mixtures_place_speakers_in_a_stereo_field(tmp_path):
    source = manifest(tmp_path, 'train')
    dataset = SpeakerMixtureDataset(source, sample_rate=8000, duration=.1, max_speakers=5, samples_per_epoch=6,
                                    validation=True, channels=2,
                                    augmentation={'reverb_probability': 1., 'noise_probability': 0., 'eq_probability': 0., 'pitch_probability': 0.})
    for index in range(6):
        mixture, target, count = dataset[index]
        assert mixture.shape == (2, 800) and target.shape == (5, 2, 800)
        assert torch.allclose(mixture, target.sum(0), atol=1e-6)
        if int(count):
            active = target[:int(count)]
            assert active.std(dim=-1).min() > 0, 'room response must differ between channels'
            assert not torch.allclose(active[:, 0], active[:, 1])
        assert torch.count_nonzero(target[int(count):]) == 0


def test_shipped_44k_stereo_configs_build_and_forward():
    from pymss_core import get_model_from_config

    root = Path(__file__).resolve().parents[1] / 'configs'
    for name in ('adaptive_bs_roformer_44k_stereo.yaml', 'adaptive_bs_roformer_44k_stereo_large.yaml',
                 'adaptive_mel_band_roformer_44k_stereo.yaml'):
        model, config = get_model_from_config('auto', root / name)
        assert model.audio_channels == 2 and int(config.audio.sample_rate) == 44100
        model.eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 2, 11025))
        assert output.audio.shape == (1, 5, 2, 11025)
        assert output.existence_logits.shape == (1, 6)
        assert torch.isfinite(output.audio).all()


def test_extract_parquet_audio_reads_struct_speakers_and_mp3(tmp_path):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf

    from pymsst.data.sources import HF_SHARD_FIELDS, extract_parquet_audio

    def encoded(frequency):
        buffer = io.BytesIO()
        sf.write(buffer, np.sin(2 * np.pi * frequency * np.arange(8000) / 16000).astype(np.float32), 16000, format='FLAC')
        return buffer.getvalue()

    shard = tmp_path / 'shard.parquet'
    table = pa.table({
        'speaker': pa.array(['EN_abc_SPEAKER_00', 'EN_abc_SPEAKER_01']),
        'mp3': pa.array([{'bytes': encoded(220)}, {'bytes': encoded(330)}],
                        type=pa.struct([('bytes', pa.binary())])),
    })
    pq.write_table(table, shard)

    fields = HF_SHARD_FIELDS['emilia-yodas-en']
    summary = extract_parquet_audio(shard, tmp_path / 'out', speaker_key=fields['speaker'],
                                    audio_key=fields['audio'], prefix='emilia-yodas-en/', raw_suffix='mp3')
    assert summary['written'] == 2 and summary['sample_rates'] == []
    written = sorted((tmp_path / 'out').rglob('*.mp3'))
    # compressed sources are stored as-is so the corpus fits on a Colab disk
    assert [path.parent.name for path in written] == ['EN_abc_SPEAKER_00', 'EN_abc_SPEAKER_01']
    assert written[0].read_bytes() == encoded(220)


def test_link_speaker_tree_maps_corpus_layouts_and_rejects_unknown(tmp_path):
    from pymsst.data.sources import link_speaker_tree

    vctk = tmp_path / 'vctk' / 'wav48_silence_trimmed' / 'p225'
    vctk.mkdir(parents=True)
    sf.write(vctk / 'p225_001_mic1.flac', np.ones(400, dtype=np.float32) * .1, 8000)
    sf.write(vctk / 'p225_001_mic2.flac', np.ones(400, dtype=np.float32) * .1, 8000)
    vocal = tmp_path / 'vocalset' / 'FULL' / 'female1' / 'vibrato'
    vocal.mkdir(parents=True)
    sf.write(vocal / 'f1_a_vibrato.wav', np.ones(400, dtype=np.float32) * .1, 8000)
    summary = link_speaker_tree([tmp_path / 'vctk', tmp_path / 'vocalset'], tmp_path / 'linked', drop_patterns=['mic2'])
    assert summary == {'speakers': 2, 'files': 2}
    assert (tmp_path / 'linked' / 'vctk:p225' / 'p225_001_mic1.flac').is_symlink()
    assert (tmp_path / 'linked' / 'vocalset:female1' / 'f1_a_vibrato.wav').is_symlink()
    extra = tmp_path / 'vctk' / 's5'
    extra.mkdir(parents=True)
    sf.write(extra / 's5_001_mic1.flac', np.ones(400, dtype=np.float32) * .1, 8000)
    assert link_speaker_tree([tmp_path / 'vctk'], tmp_path / 's5-ok', drop_patterns=['mic2']) == {'speakers': 2, 'files': 2}
    junk = tmp_path / 'vocalset' / '__MACOSX' / 'FULL' / 'female1'
    junk.mkdir(parents=True)
    sf.write(junk / '._f1_a_vibrato.wav', np.zeros(400, dtype=np.float32), 8000)
    assert link_speaker_tree([tmp_path / 'vocalset'], tmp_path / 'junk-ok') == {'speakers': 1, 'files': 1}
    stray = tmp_path / 'vctk' / 'wav48_silence_trimmed' / 'misc'
    stray.mkdir(parents=True)
    sf.write(stray / 'noise.wav', np.zeros(400, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match='no speaker directory'):
        link_speaker_tree([tmp_path / 'vctk'], tmp_path / 'again', drop_patterns=['mic2'])


def test_sources_module_exposes_known_corpora(tmp_path):
    """A name used but never defined only fails at runtime, so assert the registry itself."""
    from pymsst.data.sources import CORPORA, HF_CORPORA, fetch_hf_corpus

    assert HF_CORPORA["vctk-hf"][0] == "sanchit-gandhi/vctk"
    assert {"vctk", "vocalset"} <= set(CORPORA)
    with pytest.raises(ValueError, match="unknown HF corpus"):
        fetch_hf_corpus("nope", tmp_path)


def test_hf_parquet_shard_decodes_into_per_speaker_flac(tmp_path):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    from pymsst.data.sources import extract_parquet_audio, link_speaker_tree

    rows = []
    for speaker, value in (("p225", .1), ("p226", .2)):
        buffer = io.BytesIO()
        sf.write(buffer, np.full(2400, value, dtype=np.float32), 48000, format="FLAC")
        rows.append({"speaker_id": speaker, "audio": {"bytes": buffer.getvalue(), "path": None}})
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), shard)
    summary = extract_parquet_audio(shard, tmp_path / "out", prefix="vctk/")
    assert summary == {"written": 2, "sample_rates": [48000]}
    files = sorted((tmp_path / "out").rglob("*.flac"))
    assert [f.parent.name for f in files] == ["p225", "p226"]
    assert sf.info(files[0]).samplerate == 48000
    assert link_speaker_tree([tmp_path / "out"], tmp_path / "linked") == {"speakers": 2, "files": 2}


def test_prepare_manifests_keeps_every_split_non_empty(tmp_path):
    from pymsst.data.speakers import prepare_manifests

    root = tmp_path / 'speakers'
    for index in range(40):
        directory = root / f'spk{index:02d}'
        directory.mkdir(parents=True)
        sf.write(directory / 'a.wav', np.ones(8000, dtype=np.float32) * .1, 8000)
    summary = prepare_manifests(root, tmp_path / 'manifests', layout='speaker-dirs')
    assert all(part['speakers'] >= 1 for part in summary.values())
    assert sum(part['speakers'] for part in summary.values()) == 40
    again = prepare_manifests(root, tmp_path / 'manifests2', layout='speaker-dirs')
    assert again == summary, 'the identity split must be reproducible'
    single = tmp_path / 'one' / 'only'
    single.mkdir(parents=True)
    sf.write(single / 'a.wav', np.ones(8000, dtype=np.float32) * .1, 8000)
    with pytest.raises(ValueError, match='at least three identities'):
        prepare_manifests(tmp_path / 'one', tmp_path / 'one-manifests', layout='speaker-dirs')


def test_identity_disjointness_and_insufficient_speakers(tmp_path):
    train, validation = manifest(tmp_path, 'train'), manifest(tmp_path, 'validation')
    validate_disjoint_manifests(train, validation)
    with pytest.raises(ValueError, match='disjoint'):
        validate_disjoint_manifests(train, train)
    with pytest.raises(ValueError, match='distinct'):
        SpeakerMixtureDataset(train, max_speakers=6)


def test_official_manifest_splits_and_leak_detection(tmp_path):
    for split, speaker in [('train-clean-100', '1'), ('dev-clean', '2'), ('test-clean', '3')]:
        directory = tmp_path / 'corpus' / split / speaker / 'chapter'
        directory.mkdir(parents=True)
        sf.write(directory / 'voice.wav', np.ones(4000) * .1, 8000)
    summary = prepare_manifests(tmp_path / 'corpus', tmp_path / 'manifests')
    assert all(item['speakers'] == 1 for item in summary.values())
    leak = tmp_path / 'corpus' / 'dev-clean' / '1' / 'chapter'
    leak.mkdir(parents=True)
    sf.write(leak / 'voice.wav', np.ones(4000) * .1, 8000)
    with pytest.raises(ValueError, match='different splits'):
        prepare_manifests(tmp_path / 'corpus', tmp_path / 'other')


def test_loss_is_permutation_invariant_and_handles_empty_samples():
    torch.manual_seed(2)
    target = torch.randn(2, 3, 1, 128)
    target[1] = 0
    prediction = (target + .1 * torch.randn_like(target)).requires_grad_()
    logits = torch.randn(2, 4, requires_grad=True)
    counts = torch.tensor([3, 0])
    options = {'fft_sizes': (32, 64)}
    a = speaker_loss(AdaptiveSeparationOutput(prediction, logits), target, counts, **options)
    b = speaker_loss(AdaptiveSeparationOutput(prediction[:, [2, 0, 1]], logits), target, counts, **options)
    assert torch.allclose(a['loss'], b['loss'], atol=1e-5)
    a['loss'].backward()
    assert torch.isfinite(prediction.grad).all() and torch.isfinite(logits.grad).all()
    empty = speaker_loss(AdaptiveSeparationOutput(prediction, logits), torch.zeros_like(target), torch.zeros(2, dtype=torch.long), **options)
    assert torch.isfinite(empty['loss'])


def test_training_checkpoint_resume_and_validation(tmp_path):
    train, validation = manifest(tmp_path, 'train'), manifest(tmp_path, 'val')
    config = {'model_type': 'adaptive_bs_roformer', 'model': {
        'dim': 8, 'depth': 1, 'max_speakers': 5, 'decoder_heads': 1, 'memory_frames': 8,
        'heads': 1, 'dim_head': 8, 'freqs_per_bands': [4, 4, 4, 5],
        'stft_n_fft': 32, 'stft_win_length': 32, 'stft_hop_length': 8,
        'time_transformer_depth': 1, 'freq_transformer_depth': 1},
        'audio': {'sample_rate': 8000, 'duration': .02},
        'training': {'batch_size': 1, 'steps_per_epoch': 2, 'max_steps': 2, 'validate_every': 2,
                      'save_every': 1, 'validation_samples': 6, 'log_every': 1, 'amp': False},
        'augmentation': {'pitch_probability': 0., 'eq_probability': 0., 'reverb_probability': 0., 'noise_probability': 0.},
        'loss': {'fft_sizes': [32, 64]}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    args = Namespace(config=path, train_manifest=train, validation_manifest=validation,
                     run_dir=tmp_path / 'run', resume=None, warm_start=None, device='cpu',
                     steps=2, num_workers=0, noise_root=None)
    train_speakers(args)
    checkpoint = torch.load(args.run_dir / 'last.pt', weights_only=True)
    assert checkpoint['step'] == 2
    assert (args.run_dir / 'model.pt').is_file()
    events = [json.loads(line) for line in (args.run_dir / 'history.jsonl').read_text().splitlines()]
    val = next(event for event in events if event['event'] == 'validation')
    assert set(val['per_count']) == {'0', '1', '2', '3', '4', '5'}
    assert 'oracle_si_sdr' not in val['per_count']['0']
    # A different corpus invalidates --resume on purpose; --warm-start reuses the weights.
    slim = tmp_path / 'slim.pt'
    torch.save({'state_dict': checkpoint['state_dict'], 'step': checkpoint['step']}, slim)
    warm = Namespace(config=path, train_manifest=train, validation_manifest=validation,
                     run_dir=tmp_path / 'warm', resume=None, warm_start=slim, device='cpu',
                     steps=2, num_workers=0, noise_root=None)
    train_speakers(warm)
    warm_events = [json.loads(line) for line in (warm.run_dir / 'history.jsonl').read_text().splitlines()]
    assert any(event.get('event') == 'warm_start' for event in warm_events)
    assert json.loads((warm.run_dir / 'metadata.json').read_text())['warm_start_step'] == 2
    args.resume, args.steps = args.run_dir / 'last.pt', 3
    train_speakers(args)
    assert torch.load(args.run_dir / 'last.pt', weights_only=True)['step'] == 3
    loaded = AdaptiveBSRoformer(**config['model'])
    loaded.load_state_dict(torch.load(args.run_dir / 'model.pt', weights_only=True)['state_dict'], strict=True)
