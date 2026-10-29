"""One-time layout conversion, with all-site preflight and no scientific guesses."""

import json
from pathlib import Path
import tomllib

import pytest
import torch
import tomli_w

from openMWR.models import load_model
from openMWR.site import get_config_parameter
from tests.fixtures.cli import run_cli


def snapshot(project):
    return {path.relative_to(project): path.read_bytes() if path.is_file() else None
            for path in project.rglob('*')}


def legacy_site(project, name):
    directory = project / 'data/sites' / name
    (directory / 'retrieval').mkdir(parents=True)
    settings = dict(gen='G5', mwr_height=550, latitude=48.148, longitude=11.573,
                    hatpro_data_dir='raw measurements', plot_dir=None,
                    retrieval_output_dir=str(project / 'external output'),
                    frequency_shift=[0., -0.01], cut_off_at_mwr_height={'03715': False})
    (directory / 'config.json').write_text(json.dumps(settings))
    return directory, settings


@pytest.fixture
def old_checkpoint():
    reference = torch.load(Path(__file__).parents[1] / 'data/checkpoint_01.pt',
                           map_location='cpu', weights_only=False)
    return next(case['checkpoint'] for case in reference['cases'] if case['name'] == 'mlr')


@pytest.mark.parametrize('custom_grids', [False, True], ids=['new-grids', 'keep-user-grids'])
def test_migrate_legacy_project_preview_application_and_repeat(tmp_path, custom_grids, old_checkpoint):
    project = tmp_path / 'old project'
    first, settings = legacy_site(project, 'first')
    second, _ = legacy_site(project, 'second')
    (first / 'retrieval/MLR.pth').write_bytes(old_checkpoint)
    (first / 'retrieval/MLR.pth').chmod(0o640)
    original_stat = (first / 'retrieval/MLR.pth').stat()
    (second / 'retrieval/another.pth').write_bytes(old_checkpoint)
    for path in ('data/sites/first/hatpro/hatpro_data.nc', 'data/study.sqlite3', 'bin/main.py'):
        target = project / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'untouched historical file')
    if custom_grids:
        (project / 'grids.toml').write_text('[heights]\ncustom = [0, 1000]\n')
    # An identical config from a partial earlier conversion is safe to keep.
    existing = project / 'sites/second/site.toml'
    existing.parent.mkdir(parents=True)
    converted = {key: value for key, value in settings.items() if value is not None}
    converted['hatpro_format'] = 'netcdf'
    existing.write_text('# User comment\n' + tomli_w.dumps(converted))
    before = snapshot(project)

    args = ('migrate', '-p', str(project))
    preview = run_cli(*args, '-n')
    assert preview.returncode == 0, preview.stderr
    assert '2 checkpoints to convert' in preview.stderr
    assert 'site.toml' in preview.stderr and 'MLR.pth' in preview.stderr
    assert snapshot(project) == before

    applied = run_cli(*args)
    assert applied.returncode == 0, applied.stderr
    assert '1 site configurations created, 2 checkpoints converted' in applied.stderr
    assert 'openMWR predict SITE RETRIEVAL' in applied.stdout
    for path, content in before.items():
        if content is not None and path.suffix != '.pth':
            assert (project / path).read_bytes() == content
    assert tomllib.loads((project / 'sites/first/site.toml').read_text()) == converted
    assert get_config_parameter('first', 'hatpro_data_dir', project) == str(project / 'raw measurements')
    assert (project / 'sites/first/retrieval/MLR/model.pth.bak').read_bytes() == old_checkpoint
    for filename in ('model.pth', 'model.pth.bak'):
        saved = (project / 'sites/first/retrieval/MLR' / filename).stat()
        assert saved.st_mode == original_stat.st_mode
        assert saved.st_mtime_ns == original_stat.st_mtime_ns
    assert load_model('MLR', 'first', project).is_MLR
    assert (project / 'sites/second/retrieval/another/model.pth.bak').read_bytes() == old_checkpoint
    assert load_model('another', 'second', project).is_MLR
    assert not list((project / 'data').rglob('*.pth'))
    assert not list((project / 'sites').rglob('retrieval.toml'))
    assert not list((project / 'sites/first/datasets').iterdir())
    assert (project / 'grids.toml').is_file()

    migrated = snapshot(project)
    repeated = run_cli(*args)
    assert repeated.returncode == 0, repeated.stderr
    assert '0 site configurations created, 0 checkpoints converted' in repeated.stderr
    assert snapshot(project) == migrated


@pytest.mark.parametrize('conflict', ['site-settings', 'checkpoint', 'backup', 'blocked-parent', 'invalid-json', 'nested-null', 'invalid-name', 'invalid-checkpoint'])
def test_preflight_failure_at_later_site_preserves_entire_project(tmp_path, conflict, old_checkpoint):
    first, _ = legacy_site(tmp_path, 'first')
    second, _ = legacy_site(tmp_path, 'second')
    (first / 'retrieval/MLR.pth').write_bytes(old_checkpoint)
    (second / 'retrieval/NN.pth').write_bytes(old_checkpoint)
    target = tmp_path / 'sites/second'
    if conflict in ('site-settings', 'checkpoint', 'backup', 'blocked-parent'):
        target.mkdir(parents=True)
        if conflict == 'site-settings':
            (target / 'site.toml').write_text('gen = "G4"\n')
        elif conflict in ('checkpoint', 'backup'):
            (target / 'retrieval/NN').mkdir(parents=True)
            filename = 'model.pth' if conflict == 'checkpoint' else 'model.pth.bak'
            (target / 'retrieval/NN' / filename).write_bytes(b'existing model')
        else:
            (target / 'datasets').write_bytes(b'user file')
    elif conflict == 'invalid-json':
        (second / 'config.json').write_text('{')
    elif conflict == 'nested-null':
        (second / 'config.json').write_text('{"custom": {"unsupported": null}}')
    elif conflict == 'invalid-name':
        (second / 'retrieval/bad\\name.pth').write_bytes(b'invalid name')
    else:
        (second / 'retrieval/NN.pth').write_bytes(b'invalid checkpoint')
    before = snapshot(tmp_path)
    for flags in [('-n',), ()]:
        result = run_cli('migrate', '-p', str(tmp_path), *flags)
        assert result.returncode == 1 and 'Migrate failed:' in result.stderr
        assert 'Traceback' not in result.stderr
        assert 'second' in result.stderr
        assert snapshot(tmp_path) == before


def test_missing_or_current_project_is_not_populated(tmp_path):
    result = run_cli('migrate', '-p', str(tmp_path))
    assert result.returncode == 1 and 'No site directory' in result.stderr
    assert not list(tmp_path.iterdir())
    (tmp_path / 'data/sites').mkdir(parents=True)
    before = snapshot(tmp_path)
    result = run_cli('migrate', '-p', str(tmp_path))
    assert result.returncode == 0 and 'Nothing to migrate' in result.stderr
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('operation', ['copy', 'save'])
@pytest.mark.parametrize('layout', ['legacy', 'current'])
def test_failed_conversion_preserves_original_and_can_be_retried(tmp_path, monkeypatch, old_checkpoint, operation, layout):
    directory, _ = legacy_site(tmp_path, 'test')
    source = directory / 'retrieval/NN.pth'
    target = tmp_path / 'sites/test/retrieval/NN/model.pth'
    if layout == 'current':
        target.parent.mkdir(parents=True)
        source = target
    source.write_bytes(old_checkpoint)

    def interrupted(original, migrated):
        migrated.write(original.read(3))
        raise OSError('copy interrupted')

    with monkeypatch.context() as patch:
        if operation == 'copy':
            patch.setattr('openMWR.migrate.shutil.copyfileobj', interrupted)
        else:
            def interrupted_save(checkpoint, stream):
                stream.write(b'partial archive')
                raise OSError('save interrupted')
            patch.setattr('torch.save', interrupted_save)
        result = run_cli('migrate', '-p', str(tmp_path))
    assert result.returncode == 1 and f'{operation} interrupted' in result.stderr
    assert source.read_bytes() == old_checkpoint
    if layout == 'legacy':
        assert not target.exists()
    assert not target.with_name('model.pth.bak').exists()
    assert not list(target.parent.glob('.model-*.tmp'))
    repeated = run_cli('migrate', '-p', str(tmp_path))
    assert repeated.returncode == 0, repeated.stderr
    assert target.with_name('model.pth.bak').read_bytes() == old_checkpoint
    assert load_model('NN', 'test', tmp_path).is_MLR
    if layout == 'legacy':
        assert not source.exists()
