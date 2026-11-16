from __future__ import annotations

import json
from argparse import Namespace

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
                     run_dir=tmp_path / 'run', resume=None, device='cpu', steps=2, num_workers=0, noise_root=None)
    train_speakers(args)
    checkpoint = torch.load(args.run_dir / 'last.pt', weights_only=True)
    assert checkpoint['step'] == 2
    assert (args.run_dir / 'model.pt').is_file()
    events = [json.loads(line) for line in (args.run_dir / 'history.jsonl').read_text().splitlines()]
    val = next(event for event in events if event['event'] == 'validation')
    assert set(val['per_count']) == {'0', '1', '2', '3', '4', '5'}
    assert 'oracle_si_sdr' not in val['per_count']['0']
    args.resume, args.steps = args.run_dir / 'last.pt', 3
    train_speakers(args)
    assert torch.load(args.run_dir / 'last.pt', weights_only=True)['step'] == 3
    loaded = AdaptiveBSRoformer(**config['model'])
    loaded.load_state_dict(torch.load(args.run_dir / 'model.pt', weights_only=True)['state_dict'], strict=True)
