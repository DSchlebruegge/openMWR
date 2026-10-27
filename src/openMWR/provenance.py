"""Run records and logs shared by Python and command workflows."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
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
    for name in ("openMWR", "torch", "numpy", "xarray", "scikit-learn", "optuna"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    return {"python": platform.python_version(), "packages": packages,
            "openMWR_checkout": git_identity(Path(__file__).resolve().parents[2])}


def checkpoint_id(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
