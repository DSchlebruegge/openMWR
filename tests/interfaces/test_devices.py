"""Device selection and explicit overrides without requiring GPU hardware."""

import logging

import pytest
import torch

from openMWR.devices import device_name, resolve_device


@pytest.fixture
def hardware(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: f'Test CUDA GPU {device.index}')
    monkeypatch.setattr(torch.backends.mps, 'get_name', lambda: 'Test Apple GPU', raising=False)
    # Replace only the hardware allocation boundary. The probe still calculates
    # and synchronizes its result, using a real CPU tensor in these offline tests.
    ones = torch.ones
    monkeypatch.setattr(torch, 'ones', lambda size, **kwargs: ones(size, dtype=kwargs.get('dtype')))


@pytest.mark.parametrize(('requested', 'cuda', 'mps', 'expected', 'name'), [
    ('auto', True, True, 'cuda:1', 'Test CUDA GPU 1'),
    ('auto', True, False, 'cuda:1', 'Test CUDA GPU 1'),
    ('auto', False, True, 'mps', 'Test Apple GPU'),
    ('auto', False, False, 'cpu', 'CPU'),
    ('cpu', True, True, 'cpu', 'CPU'),
    ('cuda:0', True, True, 'cuda:0', 'Test CUDA GPU 0'),
    ('cuda', True, True, 'cuda:1', 'Test CUDA GPU 1'),
    ('mps:0', True, True, 'mps', 'Test Apple GPU'),
])
def test_selection_priority_overrides_and_hardware_log(requested, cuda, mps, expected, name,
                                                       hardware, monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)
    with caplog.at_level(logging.INFO):
        selected = resolve_device(requested)
    assert selected == expected
    assert device_name(selected) == name
    assert f'Using device {expected} ({name}), requested {requested}.' in caplog.text


@pytest.mark.parametrize(('requested', 'cuda', 'mps'), [
    ('cuda', False, True),
    ('cuda:2', True, True),
    ('mps', True, False),
    ('mps:1', True, True),
])
def test_unavailable_explicit_device_is_not_replaced(requested, cuda, mps, hardware, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)
    with pytest.raises(ValueError, match='Requested device .* is unavailable'):
        resolve_device(requested)


@pytest.mark.parametrize(('unusable', 'mps', 'expected'), [
    ({0, 1}, False, 'cpu'),
    ({0, 1}, True, 'mps'),
    ({1}, False, 'cuda:0'),
])
def test_auto_skips_gpus_that_cannot_execute_the_installed_build(unusable, mps, expected,
                                                               hardware, monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)
    ones = torch.ones
    checked = []

    def allocate(size, *, device, dtype):
        index = torch.device(device).index
        checked.append(index)
        if index in unusable:
            raise RuntimeError('CUDA error: no kernel image is available for execution on the device')
        return ones(size, dtype=dtype)

    monkeypatch.setattr(torch, 'ones', allocate)
    assert resolve_device() == expected
    assert checked == [1, 0]
    assert 'Skipping device cuda:1: CUDA error: no kernel image' in caplog.text


@pytest.mark.parametrize('stage', ['allocation', 'synchronization'])
def test_explicit_cuda_failure_is_reported_without_fallback(stage, hardware, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)

    def unsupported(*args, **kwargs):
        raise RuntimeError('CUDA error: no kernel image is available for execution on the device')

    if stage == 'allocation':
        monkeypatch.setattr(torch, 'ones', unsupported)
    else:
        monkeypatch.setattr(torch.Tensor, 'item', unsupported)
    with pytest.raises(ValueError, match='cannot run PyTorch CUDA calculations:.*Use --device cpu'):
        resolve_device('cuda:0')
