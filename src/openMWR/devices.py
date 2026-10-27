"""Shared PyTorch device selection for commands and Python workflows."""

import logging

import torch

logger = logging.getLogger(__name__)


def _cuda_error(index):
    """Check actual kernel and matrix-multiply support, including synchronization."""
    try:
        value = torch.ones((1, 1), device=f'cuda:{index}', dtype=torch.float32)
        (value @ value + 1).item()
    except RuntimeError as exc:
        return str(exc).partition('\n')[0] or type(exc).__name__
    return None


def resolve_device(device='auto', *, log=True):
    """Choose CUDA, MPS or CPU once, validating explicit accelerator choices.

    Return a concrete device string. CUDA without an index uses the current
    CUDA device. Auto skips GPUs that cannot run a small CUDA calculation,
    preferring the current device before other visible GPUs. Errors during the
    actual workflow propagate without retrying on another device.
    """
    requested = device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            candidates = [current, *(index for index in range(torch.cuda.device_count()) if index != current)]
            for index in candidates:
                error = _cuda_error(index)
                if error is None:
                    device = f'cuda:{index}'
                    break
                if log:
                    logger.warning('Skipping device cuda:%s: %s', index, error)
    selected = torch.device(device)
    if selected.type == 'cuda':
        if not torch.cuda.is_available() or (selected.index is not None and selected.index >= torch.cuda.device_count()):
            raise ValueError(f'Requested device {requested!r} is unavailable.')
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        if requested != 'auto':
            error = _cuda_error(index)
            if error is not None:
                raise ValueError(f'Requested device {requested!r} cannot run PyTorch CUDA calculations: {error}. '
                                 'Use --device cpu or install a PyTorch build compatible with this GPU.')
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
