"""Every test uses temporary outputs, a headless plot backend, and no network."""

import os
import socket
from functools import lru_cache

os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/openmwr-test-matplotlib')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    """Keep numerical workflows on CPU. Selection tests override availability."""
    # Preserve the cached-function interface inspected by PyTorch's compiler.
    @lru_cache
    def mps_unavailable():
        return False

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', mps_unavailable)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    connect = socket.socket.connect

    def local_only(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            pytest.fail(f'Tests must not contact the network: {address}')
        return connect(self, address)

    monkeypatch.setattr(socket.socket, 'connect', local_only)

    def no_dns(*args, **kwargs):
        pytest.fail('Tests must not perform network name resolution.')

    monkeypatch.setattr(socket, 'getaddrinfo', no_dns)


@pytest.fixture(autouse=True)
def release_figures():
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    yield
    plt.close('all')
