"""Only parser, dispatch and error presentation live here; algorithms have other owners."""

from datetime import date
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.fixtures.cli import run_cli
from tests.fixtures.datasets import dataset_project
from tests.fixtures.retrievals import retrieval_project


def test_installed_console_and_module_entry_points():
    console = Path(sys.executable).with_name('openMWR')
    for invocation in ([str(console)], [sys.executable, '-m', 'openMWR']):
        result = subprocess.run([*invocation, '-h'], capture_output=True, text=True, check=True)
        assert all(name in result.stdout for name in ('create-site', 'migrate', 'data', 'train', 'evaluate', 'predict', 'omb'))


def test_data_flags_and_multiple_targets_reach_workflow(tmp_path, monkeypatch):
    factory = Mock()
    monkeypatch.setattr('openMWR.datasets.DatasetWorkflow', factory)
    result = run_cli('data', 'site', 'training', 'simulation', '-p', str(tmp_path), '-w', '-r', '-n')
    assert result.returncode == 0, result.stderr
    factory.assert_called_once_with(tmp_path, 'site')
    factory.return_value.build.assert_called_once_with(
        ['training', 'simulation'], with_dependencies=True, rebuild=True, dry_run=True)


@pytest.mark.parametrize('optimize', [False, True], ids=['fixed', 'search'])
def test_train_selects_action_and_passes_overrides(tmp_path, monkeypatch, optimize):
    workflow = Mock(requires_optimization=optimize)
    factory = Mock(return_value=workflow)
    monkeypatch.setattr('openMWR.train.TrainingWorkflow.from_recipe', factory)
    args = ['train', 'site', 'model', '-p', str(tmp_path)]
    if optimize:
        args += ['-D', 'cpu', '-c', '-t', '7', '--data-source', 'era5', '--dataset', 'analysis',
                 '-s', '2024-01-01', '-e', '2024-03-31', '--days-of-month', '4', '14']
    else:
        args += ['-f']
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    factory.assert_called_once_with(tmp_path, 'site', 'model', 'cpu' if optimize else 'auto')
    if optimize:
        workflow.train.assert_not_called()
        workflow.optimize.assert_called_once_with(n_trials=7, data_source='era5', dataset='analysis',
            run_mode='continue', start_date='2024-01-01', end_date='2024-03-31',
            days_of_month={'first': 4, 'last': 14})
    else:
        workflow.optimize.assert_not_called()
        workflow.train.assert_called_once_with(run_mode='fresh')
        rejected = run_cli(*args, '-t', '1')
        assert rejected.returncode == 1 and 'require input-noise optimization or search ranges' in rejected.stderr
        assert workflow.train.call_count == 1


def test_evaluation_selects_recipe_and_protects_existing_results(tmp_path, monkeypatch):
    evaluate, save, load = Mock(), Mock(return_value=tmp_path / 'result.html'), Mock()
    monkeypatch.setattr('openMWR.evaluation.evaluate', evaluate)
    monkeypatch.setattr('openMWR.evaluation.save_evaluation', save)
    monkeypatch.setattr('openMWR.evaluation.EvaluationRecipe.load', load)
    args = ['evaluate', 'site', 'comparison', '-p', str(tmp_path),
            '-r', 'first', 'second', '-o', str(tmp_path / 'result.html')]
    result = run_cli(*args)
    assert result.returncode == 0 and result.stdout.strip() == str(tmp_path / 'result.html')
    load.assert_called_once_with(tmp_path, 'site', 'comparison')
    evaluate.assert_called_once_with(load.return_value, retrievals=['first', 'second'], device='auto')
    save.assert_called_once_with(evaluate.return_value, tmp_path / 'result.html')
    (tmp_path / 'result.html').write_text('Existing report')
    assert run_cli(*args).returncode == 1
    assert evaluate.call_count == save.call_count == 1
    assert (tmp_path / 'result.html').read_text() == 'Existing report'


def test_prediction_range_reuses_workflow_and_reports_partial_success(tmp_path, monkeypatch):
    workflow = Mock()
    workflow.predict.side_effect = [SimpleNamespace(errors={}), SimpleNamespace(errors={'bls': 'missing'})]
    workflow.save.side_effect = [[tmp_path / 'day1.nc'], [tmp_path / 'day2.nc']]
    factory = Mock(return_value=workflow)
    monkeypatch.setattr('openMWR.predict.PredictionWorkflow', factory)
    result = run_cli('predict', 'site', 'first', 'second', '-p', str(tmp_path),
                     '-s', '2024-01-01', '-e', '2024-01-02', '-o', 'output')
    assert result.returncode == 1 and 'Prediction incomplete' in result.stderr
    assert result.stdout.splitlines() == [str(tmp_path / 'day1.nc'), str(tmp_path / 'day2.nc')]
    factory.assert_called_once_with(tmp_path, 'site', ['first', 'second'], 'auto', output_dir=Path('output'))
    assert [call.args[0] for call in workflow.predict.call_args_list] == [date(2024, 1, 1), date(2024, 1, 2)]


@pytest.mark.parametrize('args', [
    [], ['--date', 'invalid'], ['--start-date', '2024-01-01'],
    ['--date', '2024-01-01', '--end-date', '2024-01-02'],
    ['--start-date', '2024-01-02', '--end-date', '2024-01-01'],
])
def test_prediction_invalid_dates_create_nothing(tmp_path, args):
    result = run_cli('predict', 'site', *args, '-p', str(tmp_path))
    assert result.returncode == 2 and not list(tmp_path.iterdir())


def test_data_dry_run_and_diagnostic_are_read_only(dataset_project):
    before = {path: path.read_bytes() for path in dataset_project.rglob('*') if path.is_file()}
    dry = run_cli('data', 'test', 'training', 'r20', '-p', str(dataset_project), '-w', '-n')
    assert dry.returncode == 0, dry.stderr
    assert 'source -> profiles -> r17 -> training -> r20' in dry.stderr
    assert before == {path: path.read_bytes() for path in dataset_project.rglob('*') if path.is_file()}
    missing = run_cli('train', 'test', 'missing', '-p', str(dataset_project))
    assert missing.returncode == 1 and 'Train failed:' in missing.stderr
    assert 'Traceback' not in missing.stderr


@pytest.mark.skipif(os.name != 'posix', reason='Detached commands use POSIX sessions.')
@pytest.mark.parametrize(('command', 'search', 'extra', 'log'), [
    ('train', False, [], 'sites/test/retrieval/example/training.log'),
    ('train', True, [], 'sites/test/retrieval/example/optimization.log'),
    ('evaluate', False, [], 'sites/test/evaluations/example.log'),
    ('predict', False, ['-d', '2024-01-01', '-o', 'output with spaces'], 'output with spaces/prediction.log'),
    ('omb', False, [], 'data/sites/test/example/omb.log'),
])
def test_background_launch_uses_current_environment_and_existing_task_logs(tmp_path, monkeypatch, command, search, extra, log):
    from openMWR.background import LOG_ENV

    retrieval_project(tmp_path, search=search)
    launch = Mock(return_value=SimpleNamespace(pid=12345))
    monkeypatch.setattr('openMWR.background.subprocess.Popen', launch)
    args = [command, 'test', 'example', '-p', str(tmp_path), '-b', *extra]
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    assert 'PID 12345' in result.stdout and str(tmp_path / log) in result.stdout
    positional, options = launch.call_args
    assert positional[0] == [sys.executable, '-u', '-m', 'openMWR', *args]
    assert options['start_new_session'] is True
    assert options['stdin'] == subprocess.DEVNULL and options['stderr'] == subprocess.STDOUT
    assert options['env'][LOG_ENV] == str(tmp_path / log)
    assert not (tmp_path / 'sites/test/retrieval/example/model.pth').exists()


def test_background_dry_run_is_rejected_without_writes(tmp_path):
    result = run_cli('data', 'test', 'training', '-p', str(tmp_path), '-bn')
    assert result.returncode == 2 and '--background cannot be combined with --dry-run' in result.stderr
    assert not list(tmp_path.iterdir())


def test_background_console_and_library_messages_are_recorded_once(tmp_path, monkeypatch, caplog):
    from openMWR import background

    path = tmp_path / 'build.log'
    root = logging.getLogger()
    client = logging.getLogger('background_test.external_client')
    monkeypatch.setattr(background, '_console_path', None)
    with path.open('a') as output, caplog.at_level(logging.INFO):
        monkeypatch.setattr(sys, 'stderr', output)
        monkeypatch.setattr(root, 'handlers', [logging.StreamHandler(output)])
        # Like the CDS client, this library also propagates to the root logger.
        monkeypatch.setattr(client, 'handlers', [logging.StreamHandler(output)])
        monkeypatch.setattr(client, 'propagate', True)
        background.capture_console(path)
        handler = background.task_log_handler(path)
        root.addHandler(handler)
        try:
            root.info('Task progress')
            client.info('Download progress')
            print('Startup error', file=sys.stderr)
        finally:
            root.removeHandler(handler)
            handler.close()
    text = path.read_text()
    for message in ('Task progress', 'Download progress', 'Startup error'):
        assert text.count(message) == 1
