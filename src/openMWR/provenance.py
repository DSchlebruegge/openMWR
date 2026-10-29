"""Run records and logs shared by Python and command workflows."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import version, PackageNotFoundError
import logging
from pathlib import Path
import platform
import subprocess

import numpy as np

from openMWR.paths import retrieval_dir
from openMWR.background import task_log_handler


def plain_value(value):
    """Make settings stable JSON values, including existing Optuna suggestions."""
    if isinstance(value, dict):
        return {str(k): plain_value(v) for k, v in value.items()}
    if hasattr(value, "_asdict"):
        return {"kind": type(value).__name__, **plain_value(value._asdict())}
    if isinstance(value, (list, tuple)):
        return [plain_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def git_identity(path):
    """Return available checkout identity; installed packages need no Git."""
    path = Path(path)
    if not (path / ".git").exists():
        return None
    try:
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        modified = bool(subprocess.check_output(["git", "-C", str(path), "status", "--short", "--untracked-files=normal"], text=True, stderr=subprocess.DEVNULL).strip())
        return {"revision": revision, "modified": modified}
    except (OSError, subprocess.CalledProcessError):
        return None


def software_identity():
    packages = {}
    for name in ("openMWR", "torch", "numpy", "xarray", "optuna"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    return {"python": platform.python_version(), "packages": packages,
            "openMWR_checkout": git_identity(Path(__file__).resolve().parents[2])}


def checkpoint_id(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_id(data):
    """Hash in-memory values and coordinates, independent of files and attributes.

    Callers select the data used by their experiment before hashing. This also
    records custom Python preprocessing without hashing unrelated source caches.
    """
    digest = hashlib.sha256()

    def record(value):
        encoded = json.dumps(value, ensure_ascii=False, default=str).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)

    for name in sorted(data.variables):
        variable = data[name]
        values = np.asarray(variable.values)
        # NetCDF string encodings can load as either object or Unicode arrays.
        if values.dtype.kind in 'OUS':
            record([name, variable.dims, values.shape, 'text'])
            record(values.astype(str).tolist())
        else:
            dtype = values.dtype.newbyteorder('<')
            record([name, variable.dims, values.shape, dtype.str])
            values = np.asarray(values, dtype=dtype)
            if values.dtype.kind in 'fc' and np.isnan(values).any():
                values = np.where(np.isnan(values), np.nan, values)
            digest.update(np.ascontiguousarray(values))
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def task_log(path):
    """Attach a run log temporarily, preserving the caller's logging handlers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = task_log_handler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(previous_level)


@contextmanager
def retrieval_log(project_dir, site, name, action):
    """Record one training or optimization run next to its retrieval."""
    filename = {'train': 'training', 'optimize': 'optimization'}[action]
    with task_log(retrieval_dir(project_dir, site, name) / f'{filename}.log'):
        logger = logging.getLogger(__name__)
        try:
            logger.info('Starting %s for site %s, retrieval %s', action, site, name)
            yield
            logger.info('Finished %s for site %s, retrieval %s', action, site, name)
        except BaseException:
            logger.exception('%s failed for site %s, retrieval %s', action, site, name)
            raise
