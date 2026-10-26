"""Shared PyTorch device selection for commands and Python workflows."""

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(device='auto', *, log=True):
    """Choose CUDA, MPS or CPU once, validating explicit accelerator choices.

    Return a concrete device string. CUDA without an index uses the current
    CUDA device. Execution errors propagate without retrying on another device.
    """
    requested = device
    if device == 'auto':
        device = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else 'cpu')
    selected = torch.device(device)
    if selected.type == 'cuda':
        if not torch.cuda.is_available() or (selected.index is not None and selected.index >= torch.cuda.device_count()):
            raise ValueError(f'Requested device {requested!r} is unavailable.')
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        selected = torch.device('cuda', index)
    elif selected.type == 'mps':
        if not torch.backends.mps.is_available() or selected.index not in (None, 0):
            raise ValueError(f'Requested device {requested!r} is unavailable.')
        selected = torch.device('mps')
    resolved = str(selected)
    if log:
        logger.info('Using device %s (%s), requested %s.', resolved, device_name(resolved), requested)
    return resolved


def device_name(device):
    """Describe resolved hardware for startup logs and saved run metadata."""
    selected = torch.device(device)
    if selected.type == 'cuda':
        return torch.cuda.get_device_name(selected)
    if selected.type == 'mps':
        get_name = getattr(torch.backends.mps, 'get_name', None)
        return get_name() if get_name is not None else 'Apple GPU'
    return selected.type.upper()
