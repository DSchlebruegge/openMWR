"""Detached commands finish after their launcher exits and retain usable logs."""

import os
import re
import signal
import subprocess
import sys
import time

import pandas as pd
import pytest

from openMWR.paths import dataset_file
from tests.fixtures.datasets import dataset_project, saved_simulation


pytestmark = pytest.mark.skipif(os.name != 'posix', reason='Detached commands use POSIX sessions.')


def launch(project, *args):
    controller = subprocess.Popen(
        [sys.executable, '-m', 'openMWR', *args, '-p', str(project)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    stdout, stderr = controller.communicate(timeout=30)
    assert controller.returncode == 0, stderr
    pid = int(re.search(r'PID (\d+)', stdout)[1])
    # Simulate the hangup that would affect a job left in its launcher's group.
    try:
        os.killpg(controller.pid, signal.SIGHUP)
    except ProcessLookupError:
        pass
    return pid


def wait_for_log(path, message, pid):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        text = path.read_text()
        if message in text:
            return text
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    pytest.fail(f'Background command did not report {message!r}:\n{path.read_text()}')


def test_detached_rebuild_finishes_preserves_log_and_records_messages_once(dataset_project):
    saved_simulation(dataset_project, pd.date_range('2020-01-01', periods=40).values)
    directory = dataset_project / 'data/sites/test/training'
    directory.mkdir()
    log = directory / 'build.log'
    log.write_text('Earlier build log\n')
    (directory / 'obsolete.nc').write_bytes(b'old output')
    pid = launch(dataset_project, 'data', 'test', 'training', '-br')
    text = wait_for_log(log, 'Finished data command.', pid)
    assert text.startswith('Earlier build log\n')
    assert not (directory / 'obsolete.nc').exists()
    for split in ('training', 'validation', 'test'):
        path = dataset_file(dataset_project, 'test', 'training', '03715', split=split)
        assert path.is_file()
        assert text.count(f'Saved {path}:') == 1


def test_detached_startup_failure_reaches_dataset_log(dataset_project):
    (dataset_project / 'sites/test/datasets/broken.toml').write_text('kind = "unknown"\n')
    pid = launch(dataset_project, 'data', 'test', 'broken', '--background')
    log = dataset_project / 'data/sites/test/broken/build.log'
    text = wait_for_log(log, 'Data failed:', pid)
    assert 'broken: kind must be' in text
    assert 'Finished data command.' not in text
