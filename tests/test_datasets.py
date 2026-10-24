"""Offline data workflows with real NetCDF, preparation, RT and split outputs."""

import json
from datetime import date
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd
import pytest
import tomli_w
import xarray as xr

from openMWR.datasets import DatasetRecipe, DatasetWorkflow
from openMWR.era5 import ERA5Processor
from openMWR.paths import dataset_file, era5_cache_file, radiosonde_year_file
from openMWR.site import create_site


def recipe(project, name, **values):
    # Explicit fixture choices; production loaders do not fill these values.
    base = {
        'radiosonde': {'start_year': 1990, 'end_year': 'current_year'},
        'era5': {'start_date': 'site', 'end_date': 'today', 'hour_interval': 6, 'delay_days': 5},
        'hatpro': {'start_date': 'site', 'end_date': 'today'},
        'profiles': {'preparation': {'heights': 'levels_208_30km'}},
        'simulation': {'radiative_transfer': {'backend': 'torchMWRT', 'absorption_model': 'R24',
                                              'frequencies': 'hatpro', 'angles': 'hatpro_bls',
                                              'infrared': {'droplet_number_density': 200e6, 'min_effective_radius': 2.5}}},
        'analysis': {'analysis': {'max_time_difference_minutes': 10, 'exclude_rain': True, 'height_range': [0, 10000]}},
        'split': {'split': {'test_days_of_month': {'first': 1, 'last': 3}, 'validation_days_of_month': {'first': 15, 'last': 17}}},
    }.get(values['kind'], {})
    if values['kind'] == 'profiles':
        source = project / 'sites/test/datasets' / (values.get('inputs', {}).get('source', '') + '.toml')
        if source.is_file() and tomllib.loads(source.read_text())['kind'] == 'radiosonde':
            base['preparation'].update(incomplete_profiles='monthly_mean', cloud={'rh_threshold': 0.96, 'min_temperature': 253.15, 'adiabatic_fraction': 'power_law', 'a': 0.7, 'b': 2.3},
                                       quality={'min_top_height': 15000, 'min_levels': 500})
    for key, value in base.items():
        values[key] = {**value, **values.get(key, {})} if isinstance(value, dict) else values.get(key, value)
    path = project / 'sites/test/datasets' / f'{name}.toml'
    path.write_text(tomli_w.dumps(values))
    return path


@pytest.fixture
def project(tmp_path):
    create_site('test', tmp_path, gen='G5', mwr_height=550,
                latitude=48.148, longitude=11.573, mwr_measurement_start_date='2020-01-01',
                rpg_retrieval_exists=False, hatpro_data_dir='observations')
    recipe(tmp_path, 'source', kind='radiosonde', stations=['03715'], start_year=2020, end_year=2020)
    recipe(tmp_path, 'profiles', kind='profiles', inputs={'source': 'source'}, preparation={
        'heights': [0, 500, 1000, 2000, 5000, 10000, 15000, 20000, 30000],
        'scale_rh_clear_sky': True, 'scale_rh_clear_sky_fraction': 1.0,
        'scale_rh_clear_sky_factor': 0.2, 'scale_rh_clear_sky_seed': 42})
    for model in ('R17', 'R20'):
        recipe(tmp_path, model.lower(), kind='simulation', inputs={'profiles': 'profiles'},
               radiative_transfer={'absorption_model': model, 'frequencies': [22.24], 'angles': [90.0]},
               execution={'processes': 1})
    recipe(tmp_path, 'training', kind='split', inputs={'simulation': 'r17'})
    return tmp_path


def raw_profiles(times):
    heights = np.linspace(484, 35484, 610)
    temperature = 285 - np.minimum(heights - 484, 11000) * 0.006
    pressure = 950 * np.exp(-(heights - 484) / 7500)
    shape = (len(times), len(heights))
    return xr.Dataset({
        'height': (('synop', 'meastime'), np.broadcast_to(heights, shape).copy()),
        'T': (('synop', 'meastime'), np.broadcast_to(temperature, shape).copy()),
        'p': (('synop', 'meastime'), np.broadcast_to(pressure, shape).copy()),
        'rh': (('synop', 'meastime'), np.full(shape, 40.0)),
        'starttime': ('synop', times - np.timedelta64(75, 'm')),
    }, coords={'synop': times, 'meastime': np.arange(len(heights)) * 2},
       attrs={'station_id': '03715', 'station_height': 484})


def install_raw(project, times):
    path = radiosonde_year_file(project, '03715', 2020)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_profiles(times).to_netcdf(path)


def test_real_preparation_rt_shared_dependencies_and_incremental_updates(project, monkeypatch):
    import openMWR.run_RT as rt

    # Only the external infrared solver is replaced; microwave RT is real.
    calls = []
    def infrared(*args, **kwargs):
        calls.append(1)
        return 200.0
    monkeypatch.setattr(rt, 'calculate_IR_band_TB', infrared)
    times = pd.date_range('2020-01-01', periods=3, freq='12h').values
    install_raw(project, times)
    workflow = DatasetWorkflow(project, 'test')
    order = workflow.build(['r17', 'r20'], with_dependencies=True)
    assert order == ['source', 'profiles', 'r17', 'r20']
    assert len(calls) == 6
    profile_path = dataset_file(project, 'test', 'profiles', '03715')
    old_profiles = xr.load_dataset(profile_path)
    np.testing.assert_allclose(old_profiles.rh.sel(height=0), 8.0)
    old_rt = xr.load_dataset(dataset_file(project, 'test', 'r17', '03715'))
    assert np.isfinite(old_rt.TB).all()
    assert not np.array_equal(old_rt.TB.values, xr.load_dataset(dataset_file(project, 'test', 'r20', '03715')).TB.values)

    workflow.build('r17', with_dependencies=True)
    assert len(calls) == 6  # Repeating the command performs no microwave/IR work.
    new_times = pd.date_range('2020-01-01', periods=4, freq='12h').values
    install_raw(project, new_times)
    workflow.build('r17', with_dependencies=True)
    assert len(calls) == 7
    updated_profiles = xr.load_dataset(profile_path)
    xr.testing.assert_identical(updated_profiles.sel(synop=times), old_profiles)
    updated_rt = xr.load_dataset(dataset_file(project, 'test', 'r17', '03715'))
    xr.testing.assert_identical(updated_rt.sel(time=times), old_rt)
    assert xr.load_dataset(dataset_file(project, 'test', 'r20', '03715')).time.size == 3


def test_cli_dry_run_and_configuration_errors_write_nothing(project):
    before = sorted(str(p) for p in project.rglob('*'))
    result = subprocess.run([sys.executable, '-m', 'openMWR', 'data', 'test', 'training', 'r20',
        '-p', str(project), '-w', '-n'],
        capture_output=True, text=True, cwd=project.parent)
    assert result.returncode == 0, result.stderr
    assert 'source -> profiles -> r17 -> training -> r20' in result.stderr
    assert before == sorted(str(p) for p in project.rglob('*'))
    with pytest.raises(FileNotFoundError, match='Missing input dataset'):
        DatasetWorkflow(project, 'test').build('training')
    recipe(project, 'profiles', kind='profiles', inputs={'source': 'training'})
    with pytest.raises(ValueError, match='Circular dataset dependency'):
        DatasetWorkflow(project, 'test').build('training', with_dependencies=True)
    assert before == sorted(str(p) for p in project.rglob('*'))


@pytest.mark.parametrize('values, message', [
    ({'kind': 'simulation', 'inputs': {'profiles': 'source'}}, 'incompatible kind'),
    ({'kind': 'simulation', 'inputs': {'profiles': 'missing'}}, 'Missing dataset recipe'),
    ({'kind': 'simulation', 'inputs': {'profiles': 'profiles'}, 'radiative_transfer': {'absorpton_model': 'R17'}}, 'Unknown'),
    ({'kind': 'profiles', 'inputs': {'source': 'source'}, 'stations': ['02290']}, 'subset'),
    ({'kind': 'profiles', 'inputs': {'source': 'source'}, 'preparation': {'heights': [100, 0]}}, 'must be increasing'),
    ({'kind': 'profiles', 'inputs': {'source': 'source'}, 'execution': {'processes': 2}}, 'preparation is serial'),
    ({'kind': 'split', 'inputs': {'simulation': 'r17'}, 'split': {'test_days_of_month': {'first': 15, 'last': 18}}}, 'must not overlap'),
    ({'kind': 'split', 'inputs': {'simulation': 'r17'}, 'split': {'test_days_of_month': {'first': 0, 'last': 3}}}, 'at least 1'),
    ({'kind': 'split', 'inputs': {'simulation': 'r17'}, 'split': {'test_days_of_month': {'first': 1, 'last': 14}, 'validation_days_of_month': {'first': 15, 'last': 31}}}, 'leave days for training'),
])
def test_invalid_graphs_fail_before_output_creation(project, values, message):
    recipe(project, 'bad', **values)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        DatasetWorkflow(project, 'test').build('bad', with_dependencies=True)
    assert list((project / 'data/sites/test').iterdir()) == []


@pytest.mark.parametrize(('name', 'section', 'key'), [
    ('source', None, 'end_year'),
    ('profiles', 'preparation', 'heights'),
    ('profiles', 'preparation', 'cloud'),
    ('profiles', 'preparation', 'quality'),
    ('profiles', 'preparation', 'incomplete_profiles'),
    ('r17', 'radiative_transfer', 'absorption_model'),
    ('r17', 'radiative_transfer', 'infrared'),
    ('training', 'split', 'test_days_of_month'),
])
def test_missing_scientific_choices_fail_before_processing(project, name, section, key):
    path = project / 'sites/test/datasets' / f'{name}.toml'
    values = tomllib.loads(path.read_text())
    del (values[section] if section else values)[key]
    path.write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match=f'Missing .* settings: {key}'):
        DatasetWorkflow(project, 'test').build('training', with_dependencies=True)
    assert list((project / 'data/sites/test').iterdir()) == []


def test_named_grids_and_optional_scaling_are_explicit(project):
    from openMWR.config import load_grid

    original = load_grid(project, 'heights', 'levels_208_30km')

    recipe(project, 'profiles', kind='profiles', inputs={'source': 'source'},
           preparation={'heights': 'levels_208_30km'})
    workflow = DatasetWorkflow(project, 'test')
    np.testing.assert_array_equal(workflow.coordinates('training')['heights'], original)
    assert not workflow.settings('profiles')['scale_rh_clear_sky']
    # Resolution preserves editable presets and observes Python edits next time.
    preparation = workflow.recipe('profiles').values['preparation']
    assert preparation['heights'] == 'levels_208_30km'
    preparation['heights'] = [0., 1000.]
    np.testing.assert_array_equal(workflow.coordinates('training')['heights'], [0., 1000.])
    preparation['heights'] = [1000., 0.]
    with pytest.raises(ValueError, match='must be increasing'):
        workflow.coordinates('training')
    preparation['heights'] = [0., 500., 1000.]
    np.testing.assert_array_equal(workflow.coordinates('training')['heights'], [0., 500., 1000.])
    path = project / 'sites/test/datasets/profiles.toml'
    values = tomllib.loads(path.read_text())
    values['preparation']['scale_rh_clear_sky'] = True
    path.write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match='Missing .*scale_rh_clear_sky_factor'):
        DatasetWorkflow(project, 'test').build('profiles', dry_run=True)
    values['preparation'].update(scale_rh_clear_sky_factor=0.2, scale_rh_clear_sky_fraction=0.1,
                                 scale_rh_clear_sky_seed=42)
    path.write_text(tomli_w.dumps(values))
    DatasetWorkflow(project, 'test').build('profiles', with_dependencies=True, dry_run=True)


def test_project_grid_edits_are_shared_by_python_and_dataset_recipes(project, monkeypatch):
    from openMWR.config import load_grid

    path = project / 'grids.toml'
    grids = {'heights': {'custom': [0., 500., 1000.]},
             'frequencies': {'custom': [22.24, 31.4]},
             'angles': {'custom': [90., 30.]}}
    path.write_text(tomli_w.dumps(grids))
    recipe(project, 'profiles', kind='profiles', inputs={'source': 'source'},
           preparation={'heights': 'custom'})
    recipe(project, 'r17', kind='simulation', inputs={'profiles': 'profiles'},
           radiative_transfer={'frequencies': 'custom', 'angles': 'custom'})
    workflow = DatasetWorkflow(project, 'test')
    monkeypatch.chdir(project.parent)
    for kind, values in workflow.coordinates('training').items():
        assert values == load_grid(project, kind, 'custom') == grids[kind]['custom']
    # Edits are visible on later calls using the same workflow.
    grids['heights']['custom'] = [0., 1000.]
    path.write_text(tomli_w.dumps(grids))
    assert workflow.coordinates('training')['heights'] == [0., 1000.]
    with pytest.raises(ValueError, match='Unknown .*levels_208_30km'):
        load_grid(project, 'heights', 'levels_208_30km')
    path.unlink()
    assert load_grid(project, 'heights', [0., 1000.]) == [0., 1000.]
    with pytest.raises(FileNotFoundError, match='grids.toml'):
        load_grid(project, 'heights', 'custom')


@pytest.mark.parametrize(('kind', 'values', 'message'), [
    ('heights', [0., 1000., 1000.], 'duplicates'),
    ('heights', [0., 1000., 500.], 'increasing'),
    ('frequencies', [22.24, float('nan')], 'finite'),
    ('angles', [90., 0.], 'between 0 and 90'),
])
def test_invalid_named_grids_are_rejected(project, kind, values, message):
    from openMWR.config import load_grid

    (project / 'grids.toml').write_text(tomli_w.dumps({kind: {'invalid': values}}))
    with pytest.raises(ValueError, match=message):
        load_grid(project, kind, 'invalid')


def synthetic_simulation(times):
    return xr.Dataset({'T': (('time', 'height'), np.arange(len(times))[:, None] + [[280., 279.]]),
                       'TB': (('time', 'frq'), np.arange(len(times))[:, None] + 100.)},
                      coords={'time': times, 'height': [0., 100.], 'frq': [22.24]}).expand_dims(station=['03715'])


def saved_simulation(project, times):
    path = dataset_file(project, 'test', 'r17', '03715')
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_simulation(times).to_netcdf(path)
    workflow = DatasetWorkflow(project, 'test')
    workflow.record('r17')
    return workflow


def rebuild_hint(error):
    return shlex.split(next(line.strip() for line in str(error).splitlines()
                            if line.strip().startswith('openMWR data ')))


def test_cli_split_creation_update_and_target_only_rebuild(project):
    times = pd.date_range('2020-01-01', periods=40).values
    workflow = saved_simulation(project, times)
    original = dataset_file(project, 'test', 'r17', '03715').read_bytes()
    command = [sys.executable, '-m', 'openMWR', 'data', 'test', 'training',
               '--project-dir', str(project)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    paths = {split: dataset_file(project, 'test', 'training', '03715', split=split)
             for split in ('training', 'validation', 'test')}
    before = {split: xr.load_dataset(path) for split, path in paths.items()}
    assert [before[key].time.size for key in ('training','validation','test')] == [31, 3, 6]
    np.testing.assert_array_equal(before['test'].time.values, pd.to_datetime([
        '2020-01-01', '2020-01-02', '2020-01-03', '2020-02-01', '2020-02-02', '2020-02-03']).values)
    np.testing.assert_array_equal(before['validation'].time.values,
                                  pd.to_datetime(['2020-01-15', '2020-01-16', '2020-01-17']).values)
    hashes = {key: path.read_bytes() for key, path in paths.items()}
    workflow.build('training')
    assert hashes == {key: path.read_bytes() for key, path in paths.items()}
    all_times = pd.date_range('2020-01-01', periods=60).values
    synthetic_simulation(all_times).to_netcdf(dataset_file(project, 'test', 'r17', '03715'))
    workflow.build('training')
    after = {split: xr.load_dataset(path) for split, path in paths.items()}
    for split, data in before.items():
        xr.testing.assert_identical(after[split].sel(time=data.time), data)
    combined = np.concatenate([data.time.values for data in after.values()])
    assert np.unique(combined).size == len(combined) == 60
    # Rebuilding this selected split leaves its upstream simulation untouched.
    source_bytes = dataset_file(project, 'test', 'r17', '03715').read_bytes()
    from openMWR.cli import main
    main(['data', 'test', 'training', '-p', str(project), '-r'])
    assert dataset_file(project, 'test', 'r17', '03715').read_bytes() == source_bytes
    assert source_bytes != original
    # Rebuilding gives exactly the same memberships and profile values.
    for split, data in after.items():
        xr.testing.assert_identical(xr.load_dataset(paths[split]), data)


def test_calendar_split_groups_whole_days_across_sources_and_months(project):
    recipe(project, 'source', kind='radiosonde', stations=['03715', '02290'], start_year=2020, end_year=2020)
    dates = pd.to_datetime(['2020-01-01', '2020-01-03', '2020-01-04', '2020-01-15', '2020-01-17',
                            '2020-02-02', '2020-02-16', '2020-02-29', '2020-03-01'])
    times = np.sort(np.concatenate([dates.values, (dates + pd.Timedelta(hours=12)).values]))
    workflow = saved_simulation(project, times)
    other = synthetic_simulation(times).assign_coords(station=['02290'])
    other.to_netcdf(dataset_file(project, 'test', 'r17', '02290'))
    workflow.record('r17')
    workflow.build('training')
    expected = {'test': ['2020-01-01', '2020-01-03', '2020-02-02', '2020-03-01'],
                'validation': ['2020-01-15', '2020-01-17', '2020-02-16'],
                'training': ['2020-01-04', '2020-02-29']}
    for source in ('03715', '02290'):
        original = synthetic_simulation(times).assign_coords(station=[source])
        for split, dates in expected.items():
            dates = pd.to_datetime(dates)
            expected_times = np.sort(np.concatenate([dates.values, (dates + pd.Timedelta(hours=12)).values]))
            part = xr.load_dataset(dataset_file(project, 'test', 'training', source, split=split))
            xr.testing.assert_identical(part, original.sel(time=expected_times))


def test_short_calendar_period_reports_empty_partition_before_writing(project):
    workflow = saved_simulation(project, pd.date_range('2020-01-01', periods=3).values)
    with pytest.raises(ValueError, match='No training profiles.*adjust the period or split day ranges'):
        workflow.build('training')
    assert not list((project / 'data/sites/test/training').glob('*.nc'))


def test_split_period_selects_all_three_sets_before_assignment(project):
    times = pd.date_range('2019-12-31', '2020-02-02', freq='12h').values
    workflow = saved_simulation(project, times)
    recipe(project, 'training', kind='split', inputs={'simulation': 'r17'},
           split={'start_date': '2020-01-02', 'end_date': '2020-01-31'})
    workflow.build('training')
    paths = [dataset_file(project, 'test', 'training', '03715', split=split)
             for split in ('training', 'validation', 'test')]
    before = [xr.load_dataset(path) for path in paths]
    combined = np.sort(np.concatenate([part.time.values for part in before]))
    np.testing.assert_array_equal(combined, pd.date_range('2020-01-02', '2020-01-31T12:00', freq='12h').values)
    # Source updates outside the selected period do not add samples to any split.
    extra = pd.date_range('2019-12-31', '2020-02-15', freq='12h').values
    synthetic_simulation(extra).to_netcdf(dataset_file(project, 'test', 'r17', '03715'))
    workflow.build('training')
    for original, path in zip(before, paths):
        xr.testing.assert_identical(original, xr.load_dataset(path))
    recipe(project, 'training', kind='split', inputs={'simulation': 'r17'},
           split={'start_date': '2020-01-02', 'end_date': '2020-01-30'})
    with pytest.raises(ValueError, match='end_date changed'):
        DatasetWorkflow(project, 'test').build('training')
    for original, path in zip(before, paths):
        xr.testing.assert_identical(original, xr.load_dataset(path))


def test_changed_settings_require_rebuild_before_any_deletion(project):
    times = pd.date_range('2020-01-01', periods=20).values
    saved_simulation(project, times).build('training')
    path = dataset_file(project, 'test', 'training', '03715', split='training')
    original = path.read_bytes()
    recipe(project, 'training', kind='split', inputs={'simulation': 'r17'}, split={'test_days_of_month': {'first': 1, 'last': 6}})
    with pytest.raises(ValueError, match='settings or input definitions changed') as error:
        DatasetWorkflow(project, 'test').build('training')
    assert 'training: test_days_of_month.last changed from 3 to 6' in str(error.value)
    hint = rebuild_hint(error.value)
    assert hint[3:hint.index('--project-dir')] == ['training']
    assert path.read_bytes() == original
    from openMWR.cli import main
    main(hint[1:])
    assert path.read_bytes() != original
    # Rebuilt input identity also requires downstream regeneration.
    workflow = DatasetWorkflow(project, 'test')
    workflow.record('r17', rebuilt=True)
    with pytest.raises(ValueError, match='input r17 was rebuilt'):
        workflow.build('training')


@pytest.mark.parametrize('change', ['setting', 'input_name', 'corrupt_input', 'selected_rebuild'])
def test_rebuild_hint_repairs_only_the_affected_chain(project, monkeypatch, change):
    from openMWR.cli import main

    monkeypatch.setattr('openMWR.run_RT.calculate_IR_band_TB', lambda *args, **kwargs: 200.0)
    def no_download(*args, **kwargs):
        pytest.fail('A rebuild hint must reuse the healthy raw cache.')
    monkeypatch.setattr('openMWR.dwd_opendata.update_radiosonde_data', no_download)
    install_raw(project, pd.to_datetime(['2020-01-01', '2020-01-04', '2020-01-15']).values)
    DatasetWorkflow(project, 'test').build(['training', 'r20'], with_dependencies=True)
    raw_path = radiosonde_year_file(project, '03715', 2020)
    raw_bytes = raw_path.read_bytes()
    r20_path = dataset_file(project, 'test', 'r20', '03715')
    r20_bytes = r20_path.read_bytes()
    targets = ['training', 'r20']
    options = {}
    expected = ['r17', 'training']
    if change == 'setting':
        path = project / 'sites/test/datasets/r17.toml'
        values = tomllib.loads(path.read_text())
        values['radiative_transfer']['absorption_model'] = 'R20'
        path.write_text(tomli_w.dumps(values))
    elif change == 'input_name':
        path = project / 'sites/test/datasets/training.toml'
        values = tomllib.loads(path.read_text())
        values['inputs']['simulation'] = 'r20'
        path.write_text(tomli_w.dumps(values))
        expected = ['training']
    elif change == 'corrupt_input':
        (project / 'data/sites/test/r17/state.json').write_text('{')
    else:
        targets = ['profiles', 'training']
        options = {'with_dependencies': True, 'rebuild': True}
        expected = ['profiles', 'r17', 'training']
    before = {path: path.read_bytes() for path in (project / 'data').rglob('*') if path.is_file()}
    with pytest.raises(ValueError) as error:
        DatasetWorkflow(project, 'test').build(targets, **options)
    assert before == {path: path.read_bytes() for path in (project / 'data').rglob('*') if path.is_file()}
    message = str(error.value)
    if change == 'setting':
        assert 'r17: absorption_model changed from "R17" to "R20"' in message
    elif change == 'input_name':
        assert 'training: inputs.simulation changed from "r17" to "r20"' in message
    elif change == 'corrupt_input':
        assert str(project / 'data/sites/test/r17/state.json') in message
        assert 'processing history cannot be verified' in message
    hint = rebuild_hint(error.value)
    assert hint[3:hint.index('--project-dir')] == expected
    assert hint[hint.index('--project-dir') + 1] == str(project)
    assert '--with-dependencies' not in hint
    main(hint[1:])
    assert raw_path.read_bytes() == raw_bytes
    assert r20_path.read_bytes() == r20_bytes
    DatasetWorkflow(project, 'test').require('training')


@pytest.mark.parametrize('damage', ['missing', 'invalid_json', 'invalid_structure'])
def test_missing_or_damaged_state_has_a_working_recovery_hint(project, damage):
    from openMWR.cli import main

    workflow = saved_simulation(project, pd.date_range('2020-01-01', periods=20).values)
    workflow.build('training')
    path = project / 'data/sites/test/training/state.json'
    if damage == 'missing':
        path.unlink()
    elif damage == 'invalid_json':
        path.write_text('{')
    else:
        state = json.loads(path.read_text())
        state['definition']['inputs']['simulation'] = []
        path.write_text(json.dumps(state))
    files = {p: p.read_bytes() for p in (project / 'data').rglob('*') if p.is_file()}
    for action in (lambda: workflow.build('training'), lambda: workflow.require('training')):
        with pytest.raises(ValueError) as error:
            action()
        assert str(path) in str(error.value)
        assert 'processing history cannot be verified' in str(error.value)
        hint = rebuild_hint(error.value)
        assert hint[3:hint.index('--project-dir')] == ['training']
    assert files == {p: p.read_bytes() for p in (project / 'data').rglob('*') if p.is_file()}
    simulation_path = dataset_file(project, 'test', 'r17', '03715')
    simulation_bytes = simulation_path.read_bytes()
    main(hint[1:])
    assert simulation_path.read_bytes() == simulation_bytes
    DatasetWorkflow(project, 'test').require('training')


def test_readiness_checks_selected_sources_and_splits(project):
    workflow = DatasetWorkflow(project, 'test')
    workflow.recipe('source').values['stations'].append('02290')
    for split in ('training', 'validation'):
        path = dataset_file(project, 'test', 'training', '03715', split=split)
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_simulation(pd.date_range('2020-01-01', periods=2)).to_netcdf(path)
    workflow.record('training')
    workflow.require('training', sources=['03715'], splits=('training', 'validation'))
    with pytest.raises(FileNotFoundError, match='02290.nc'):
        workflow.require('training', splits=('training', 'validation'))
    with pytest.raises(ValueError, match='subset'):
        workflow.require('training', sources=['missing'])
    with pytest.raises(ValueError, match='split dataset'):
        workflow.require('r17', splits=('training',))
    with pytest.raises(ValueError, match='splits must select'):
        workflow.require('training', splits=('typo',))


@pytest.mark.parametrize('change', ['definition', 'input_generation', 'unknown_history'])
def test_readiness_rejects_datasets_that_need_rebuilding(project, change):
    workflow = saved_simulation(project, pd.date_range('2020-01-01', periods=20).values)
    workflow.build('training')
    if change == 'definition':
        workflow.recipe('training').values['split']['test_days_of_month']['last'] = 4
    elif change == 'input_generation':
        workflow.record('r17', rebuilt=True)
    else:
        workflow.record('training', requires_rebuild=True)
    with pytest.raises(ValueError, match='Rebuild this dataset explicitly'):
        workflow.require('training', sources=['03715'], splits=('training', 'validation'))


def test_build_keeps_one_date_snapshot_and_refreshes_next_time(project, monkeypatch):
    class Clock(date):
        current = date(2020, 1, 6)

        @classmethod
        def today(cls):
            return cls.current

    monkeypatch.setattr('openMWR.datasets.date', Clock)
    recipe(project, 'era5', kind='era5', start_date='2020-01-01')
    recipe(project, 'era5_profiles', kind='profiles', inputs={'source': 'era5'},
           preparation={'heights': [0., 1000., 2000.]})
    requested = []

    def download(self, start, end, h_freq=6):
        requested.append((start, end))
        Clock.current = date(2020, 1, 7)  # The download crosses midnight.
        times = pd.date_range(start, end + pd.Timedelta(days=1), freq=f'{h_freq}h', inclusive='left')
        return xr.Dataset({name: (('time', 'p'), np.broadcast_to(values, (len(times), 3)).copy())
            for name, values in {'height': [550., 1550., 2550.], 'T': [280., 274., 268.],
                                 'rh': [40., 50., 60.], 'lwc': [0., 0., 0.]}.items()},
            coords={'time': times, 'p': [950., 850., 750.]})

    monkeypatch.setattr(ERA5Processor, 'download_era5', download)
    workflow = DatasetWorkflow(project, 'test')
    workflow.build('era5_profiles', with_dependencies=True)
    state_dir = project / 'data/sites/test'
    source = json.loads((state_dir / 'era5/state.json').read_text())
    profiles = json.loads((state_dir / 'era5_profiles/state.json').read_text())
    assert source['effective']['end_date'] == '2020-01-01'
    assert profiles['coverage']['inputs']['source']['range']['end_date'] == '2020-01-01'
    output = dataset_file(project, 'test', 'era5_profiles', 'era5')
    assert xr.load_dataset(output).time.size == 4
    assert workflow.settings('era5')['end_date'] == '2020-01-02'
    workflow.build('era5_profiles', with_dependencies=True)
    assert xr.load_dataset(output).time.size == 8
    assert requested == [(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01')),
                         (pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-02'))]
    before = output.read_bytes()
    workflow.build('era5_profiles', with_dependencies=True)
    assert output.read_bytes() == before
    assert len(requested) == 2

    def failed_download(*args, **kwargs):
        Clock.current = date(2020, 1, 9)
        raise OSError('Download interrupted')

    Clock.current = date(2020, 1, 8)
    monkeypatch.setattr(ERA5Processor, 'download_era5', failed_download)
    with pytest.raises(OSError, match='Download interrupted'):
        workflow.build('era5_profiles', with_dependencies=True)
    assert output.read_bytes() == before
    assert workflow.settings('era5')['end_date'] == '2020-01-04'


def test_shared_era5_cache_appends_after_last_day_and_preserves_other_coverage(project, monkeypatch):
    requested = []
    def download(self, start, end, h_freq=6):
        requested.append((start, end, h_freq))
        times = pd.date_range(start, end + pd.Timedelta(days=1), freq=f'{h_freq}h', inclusive='left')
        return xr.Dataset({'T': ('time', np.full(len(times), 280.))}, coords={'time': times})
    monkeypatch.setattr(ERA5Processor, 'download_era5', download)
    cache = era5_cache_file(project, 48.148, 11.573, 2)
    processor = ERA5Processor('test', project, start_date='2020-01-01', end_date='2020-01-03', hour_interval=2)
    created = processor.create_original_dataset()
    assert requested == [(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-03'), 2)]
    assert created.time.size == 36
    requested.clear()
    # Historical six-hour samples and an incomplete last day must not trigger
    # backfilling when the recipe requests two-hour data for routine updates.
    old_times = pd.date_range('2020-01-02', periods=4, freq='6h').append(pd.DatetimeIndex(['2020-01-05']))
    old = xr.Dataset({'T': ('time', np.full(len(old_times), 270.))}, coords={'time': old_times})
    old.to_netcdf(cache)
    processor.end_date = pd.Timestamp('2020-01-07')
    updated = processor.create_original_dataset()
    assert requested == [(pd.Timestamp('2020-01-06'), pd.Timestamp('2020-01-07'), 2)]
    assert updated.time.size == 29
    xr.testing.assert_identical(updated.sel(time=old_times), old)
    original = cache.read_bytes()
    processor.create_original_dataset()
    assert len(requested) == 1
    assert cache.read_bytes() == original
    processor.end_date = pd.Timestamp('2020-01-03')
    processor.create_original_dataset()
    assert len(requested) == 1  # An older requested window also leaves gaps alone.
    assert cache.read_bytes() == original
    processor.update_only = False
    rebuilt = processor.create_original_dataset()
    assert requested[-1] == (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-03'), 2)
    assert rebuilt.time.size == 61
    assert rebuilt.T.sel(time='2020-01-02T00:00').item() == 280
    assert rebuilt.T.sel(time='2020-01-05').item() == 270
    assert era5_cache_file(project, 48.148, 11.573, 6) != cache


def test_era5_profiles_are_reusable_and_incremental(project):
    recipe(project, 'era5', kind='era5', start_date='2020-01-01', end_date='2020-01-01', hour_interval=6)
    recipe(project, 'era5_profiles', kind='profiles', inputs={'source': 'era5'},
           preparation={'heights': [0., 1000., 2000.]}, execution={'processes': 1})
    raw_file = era5_cache_file(project, 48.148, 11.573, 6)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range('2020-01-01', periods=2, freq='6h')
    shape = (2, 3)
    raw = xr.Dataset({name: (('time','p'), np.broadcast_to(values, shape).copy())
        for name, values in {'height':[550.,1550.,2550.], 'T':[280.,274.,268.], 'rh':[40.,50.,60.], 'lwc':[0.,0.,0.]}.items()},
        coords={'time':times, 'p':[950.,850.,750.]})
    raw.to_netcdf(raw_file)
    workflow = DatasetWorkflow(project, 'test')
    workflow.build('era5_profiles')
    output = dataset_file(project, 'test', 'era5_profiles', 'era5')
    old = xr.load_dataset(output)
    np.testing.assert_allclose(old.surface_T, 280.)
    assert {'ah','TD','iwv','lwp','doy_sin'} <= set(old.data_vars)
    assert 'TB' not in old
    original = output.read_bytes()
    workflow.build('era5_profiles')
    assert output.read_bytes() == original
    raw = xr.concat([raw, raw.isel(time=[0]).assign_coords(time=[np.datetime64('2020-01-01T12:00')])], dim='time')
    raw.to_netcdf(raw_file)
    workflow.build('era5_profiles')
    updated = xr.load_dataset(output)
    assert updated.time.size == 3
    xr.testing.assert_identical(updated.sel(time=old.time), old)
    # Expanding requested coverage must also work when the shared cache already
    # contains that extra day and its file timestamp has not changed.
    raw = xr.concat([raw, raw.isel(time=[0]).assign_coords(time=[np.datetime64('2020-01-02T00:00')])], dim='time')
    raw.to_netcdf(raw_file)
    workflow.build('era5_profiles')
    assert xr.load_dataset(output).time.size == 3
    recipe(project, 'era5', kind='era5', start_date='2020-01-01', end_date='2020-01-02', hour_interval=6)
    DatasetWorkflow(project, 'test').build('era5_profiles')
    assert xr.load_dataset(output).time.size == 4
    recipe(project, 'era5', kind='era5', start_date='2020-01-01', end_date='2020-01-01', hour_interval=6)
    with pytest.raises(ValueError, match='date coverage narrowed') as error:
        DatasetWorkflow(project, 'test').build('era5_profiles')
    assert 'era5: end_date changed from "2020-01-02" to "2020-01-01"' in str(error.value)
    raw_bytes = raw_file.read_bytes()
    DatasetWorkflow(project, 'test').build('era5_profiles', rebuild=True)
    assert xr.load_dataset(output).time.size == 3
    assert raw_file.read_bytes() == raw_bytes


def test_source_creation_remembers_unavailable_historical_years(project, monkeypatch):
    import openMWR.dwd_opendata as dwd

    calls = []
    monkeypatch.setattr(dwd, 'get_df_stations', lambda: None)
    def download(station, year, stations):
        calls.append(year)
        if year == 2020:
            raise FileNotFoundError('No historical soundings')
        return raw_profiles(np.array([f'{year}-01-01'], dtype='datetime64[ns]'))
    monkeypatch.setattr(dwd, '_download_radiosondes_year', download)
    recipe(project, 'source', kind='radiosonde', stations=['03715'], start_year=2020, end_year=2021)
    workflow = DatasetWorkflow(project, 'test')
    workflow.build('source')
    assert calls == [2020, 2021]
    workflow.build('source')
    assert calls == [2020, 2021]
    workflow.build('source', rebuild=True)
    assert calls == [2020, 2021, 2020, 2021]


def test_named_analysis_uses_its_inputs_and_refreshes_only_when_changed(project, monkeypatch):
    import openMWR.run_RT as rt

    monkeypatch.setattr(rt, 'calculate_IR_band_TB', lambda *args, **kwargs: 200.0)
    times = pd.date_range('2020-01-01', periods=3, freq='12h').values
    install_raw(project, times)
    workflow = DatasetWorkflow(project, 'test')
    workflow.build('r17', with_dependencies=True)
    recipe(project, 'observations', kind='hatpro')
    recipe(project, 'comparison', kind='analysis', inputs={'observations': 'observations', 'reference': 'r17'})
    measured_times = times - np.timedelta64(75, 'm')
    measured = xr.Dataset({'TB': (('time', 'frq'), np.full((3, 1), 100.)), 'rain_flag': ('time', [0., 0., 0.])},
                          coords={'time': measured_times, 'frq': [22.24]})
    for bls in (False, True):
        path = dataset_file(project, 'test', 'observations', bls=bls)
        path.parent.mkdir(parents=True, exist_ok=True)
        (measured.expand_dims(ang=[90.0]) if bls else measured).to_netcdf(path)
    workflow.record('observations')
    workflow.build('comparison')
    output = dataset_file(project, 'test', 'comparison')
    pairs = xr.load_dataset(output)
    assert pairs.time.size == 3
    assert set(pairs.data_source.values) == {'radiosonde', 'hatpro'}
    np.testing.assert_allclose(pairs.TB.sel(data_source='hatpro'), 100.)
    original = output.read_bytes()
    workflow.build('comparison')
    assert output.read_bytes() == original
    measured.assign(TB=measured.TB + 2).to_netcdf(dataset_file(project, 'test', 'observations'))
    workflow.build('comparison')
    np.testing.assert_allclose(xr.load_dataset(output).TB.sel(data_source='hatpro'), 102.)

    measured = measured.assign_coords(time=measured_times + np.array([0, 6, 12], dtype='timedelta64[m]'))
    measured['rain_flag'] = ('time', [0., 1., 0.])
    for bls in (False, True):
        (measured.expand_dims(ang=[90.0]) if bls else measured).to_netcdf(
            dataset_file(project, 'test', 'observations', bls=bls))
    workflow.recipe('comparison').values['analysis'].update(
        max_time_difference_minutes=10, exclude_rain=False, height_range=[0, 1000])
    workflow.build('comparison', rebuild=True)
    pairs = xr.load_dataset(output)
    np.testing.assert_array_equal(pairs.time, measured_times[:2])
    assert pairs.height.max() == 1000
    workflow.recipe('comparison').values['analysis'].update(
        max_time_difference_minutes=15, exclude_rain=True, height_range=[0, 2000])
    workflow.build('comparison', rebuild=True)
    pairs = xr.load_dataset(output)
    np.testing.assert_array_equal(pairs.time, measured_times[[0, 2]])
    assert pairs.height.max() == 2000


def test_cloud_threshold_and_adiabatic_parameters_change_liquid_water():
    from openMWR.cloud import CloudColumn, CloudModelConfig
    from openMWR.radiosonde import add_cloud_to_ds

    heights = np.arange(0., 2100., 100.)
    profile = xr.Dataset({'T': ('height', 285 - 0.006 * heights),
                          'p': ('height', 1000 * np.exp(-heights / 7500)),
                          'rh': ('height', np.where((heights >= 500) & (heights <= 1500), 97., 50.))},
                         coords={'height': heights})
    standard = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15, adiabatic_fraction="power_law", a=0.7, b=2.3).lwc
    assert standard.max() > 0
    clear = add_cloud_to_ds(profile.copy(), rh_threshold=0.98, min_temperature=253.15, adiabatic_fraction="power_law", a=0.7, b=2.3).lwc
    assert clear.max() == 0
    halved = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15, adiabatic_fraction="power_law", a=0.35, b=2.3).lwc
    np.testing.assert_allclose(halved, standard / 2)
    different_shape = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15, adiabatic_fraction="power_law", a=0.7, b=1.0).lwc
    assert not np.allclose(different_shape, standard)
    karstens = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15, adiabatic_fraction='karstens').lwc
    adiabatic = CloudColumn(heights, profile.p.values * 100, profile.T.values,
                           profile.rh.values / 100, CloudModelConfig(rh_thres=0.96, use_f_ad=False)).calculate_lwc()
    h = np.maximum(heights - 500., 0.)
    np.testing.assert_allclose(karstens, adiabatic * np.maximum(0., 1.239 - 0.145 * np.log(h + 1)))
    assert not np.allclose(karstens, standard)
    fully_adiabatic = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15,
                                    adiabatic_fraction='none').lwc
    np.testing.assert_allclose(fully_adiabatic, adiabatic)
    warmer_threshold = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=283.,
                                      adiabatic_fraction='none').lwc
    assert warmer_threshold.max() == 0


@pytest.mark.parametrize('method', ['power_law', 'karstens'])
@pytest.mark.parametrize('completion', ['monthly_mean', 'filter'])
def test_profile_quality_limits_select_source_profiles_before_interpolation(project, method, completion):
    path = project / 'sites/test/datasets/profiles.toml'
    values = tomllib.loads(path.read_text())
    values['preparation']['cloud']['adiabatic_fraction'] = method
    values['preparation']['incomplete_profiles'] = completion
    if method == 'karstens':
        for key in ('a', 'b'):
            del values['preparation']['cloud'][key]
    path.write_text(tomli_w.dumps(values))
    times = pd.date_range('2020-01-01', periods=3, freq='12h').values
    raw = raw_profiles(times)
    # One shallow ascent, one ascent with too few valid levels, one valid ascent.
    raw['height'][0] = np.linspace(484, 12000, raw.sizes['meastime'])
    raw['T'][1, 450:] = np.nan
    path = radiosonde_year_file(project, '03715', 2020)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_netcdf(path)
    workflow = DatasetWorkflow(project, 'test')
    workflow.build('profiles')
    selected = xr.load_dataset(dataset_file(project, 'test', 'profiles', '03715'))
    np.testing.assert_array_equal(selected.synop, times[2:])
    values = workflow.recipe('profiles').values
    values['preparation']['quality'] = {'min_top_height': 10000, 'min_levels': 400}
    (project / 'sites/test/datasets/profiles.toml').write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match='scientific settings'):
        DatasetWorkflow(project, 'test').build('profiles')
    DatasetWorkflow(project, 'test').build('profiles', rebuild=True)
    selected = xr.load_dataset(dataset_file(project, 'test', 'profiles', '03715'))
    np.testing.assert_array_equal(selected.synop, times if completion == 'monthly_mean' else times[2:])
    assert np.isfinite(selected[['T', 'p', 'rh', 'lwc']].to_array()).all()


@pytest.mark.parametrize(('cloud', 'message'), [
    ({'rh_threshold': 0.96, 'min_temperature': 253.15, 'a': 0.7, 'b': 2.3}, 'Missing .*adiabatic_fraction'),
    ({'rh_threshold': 0.96, 'min_temperature': 253.15, 'adiabatic_fraction': 'unknown'}, "must be 'power_law', 'karstens' or 'none'"),
    ({'rh_threshold': 0.96, 'min_temperature': 253.15, 'adiabatic_fraction': 'power_law', 'a': 0.7}, 'Missing .*b'),
    ({'rh_threshold': 0.96, 'min_temperature': 253.15, 'adiabatic_fraction': 'karstens', 'a': 0.7}, 'remove them for karstens'),
])
def test_cloud_method_choices_fail_before_processing(project, cloud, message):
    path = project / 'sites/test/datasets/profiles.toml'
    values = tomllib.loads(path.read_text())
    values['preparation']['cloud'] = cloud
    path.write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match=message):
        DatasetWorkflow(project, 'test').build('profiles')
    assert list((project / 'data/sites/test').iterdir()) == []


@pytest.mark.parametrize('interface', ['recipe', 'python'])
@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('cloud', 'rh_threshold', 2.0),
    ('quality', 'min_top_height', -1),
    ('quality', 'min_levels', 0),
])
def test_invalid_profile_settings_fail_before_io(project, interface, section, key, value):
    from openMWR.radiosonde import create_radiosonde_dataset_for_RT

    path = project / 'sites/test/datasets/profiles.toml'
    values = tomllib.loads(path.read_text())
    preparation = values['preparation']
    preparation[section][key] = value
    with pytest.raises(ValueError, match=rf'preparation\.{section}\.{key}'):
        if interface == 'recipe':
            path.write_text(tomli_w.dumps(values))
            DatasetRecipe.load(project, 'test', 'profiles')
        else:
            create_radiosonde_dataset_for_RT('test', '03715', project, np.array(preparation['heights']),
                cloud=preparation['cloud'], quality=preparation['quality'],
                incomplete_profiles=preparation['incomplete_profiles'])
    assert list((project / 'data/sites/test').iterdir()) == []


def test_monthly_completion_rejects_profiles_without_upper_level_reference(project):
    times = pd.date_range('2020-01-01', periods=2, freq='12h').values
    raw = raw_profiles(times).isel(meastime=slice(0, 520))
    path = radiosonde_year_file(project, '03715', 2020)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_netcdf(path)
    # Both ascents pass the source-quality checks but neither reaches the top grid level.
    with pytest.raises(ValueError, match='No complete profiles remain for RT'):
        DatasetWorkflow(project, 'test').build('profiles')
    assert not dataset_file(project, 'test', 'profiles', '03715').is_file()


@pytest.mark.parametrize('radius', [2.49, 60.0])
def test_unsupported_infrared_radius_fails_before_processing(project, radius):
    path = project / 'sites/test/datasets/r17.toml'
    values = tomllib.loads(path.read_text())
    values['radiative_transfer']['infrared']['min_effective_radius'] = radius
    path.write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match='min_effective_radius must be at least 2.5 and below 60.*libRadtran'):
        DatasetWorkflow(project, 'test').build('r17', with_dependencies=True)
    assert list((project / 'data/sites/test').iterdir()) == []


@pytest.mark.parametrize('radius', [2.5, 59.99])
def test_supported_infrared_radius_allows_build_planning(project, radius):
    path = project / 'sites/test/datasets/r17.toml'
    values = tomllib.loads(path.read_text())
    values['radiative_transfer']['infrared']['min_effective_radius'] = radius
    path.write_text(tomli_w.dumps(values))
    assert DatasetWorkflow(project, 'test').build('r17', with_dependencies=True, dry_run=True) == ['source', 'profiles', 'r17']


@pytest.mark.parametrize('backend', ['torchMWRT', 'pyrtlib'])
def test_infrared_microphysics_reaches_rt_and_incomplete_inputs_are_rejected(project, monkeypatch, backend):
    import openMWR.run_RT as rt

    monkeypatch.setattr(rt, 'calculate_IR_band_TB', lambda *args: float(np.mean(args[-1])))
    profiles = xr.Dataset({
        'T': (('time', 'height'), [[285., 282., 279.], [285., 282., 279.]]),
        'p': (('time', 'height'), [[1000., 950., 900.], [1000., 950., 900.]]),
        'rh': (('time', 'height'), [[80., 80., 80.], [80., 80., 80.]]),
        'lwc': (('time', 'height'), [[0., 0.5, 0.], [0., 0.5, 0.]]),
    }, coords={'time': pd.date_range('2020-01-01', periods=2), 'height': [0., 500., 1000.]})
    options = dict(site='test', project_dir=project, freqs=np.array([22.24]),
                   angles=np.array([90.]), num_of_processes=1, absmdl='R20', RT_model=backend)
    first = rt.run_RT(profiles, **options, infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
    second = rt.run_RT(profiles, **options, infrared={'droplet_number_density': 1600e6, 'min_effective_radius': 3.0})
    cloudy_radius = (0.5 / (200e6 * 4 / 3 * np.pi * 1e6)) ** (1 / 3) * 1e6
    np.testing.assert_allclose(first.TB_IR, (2 * 2.5 + cloudy_radius) / 3)
    np.testing.assert_allclose(second.TB_IR, (2 * 3.0 + cloudy_radius / 2) / 3)
    xr.testing.assert_equal(first.TB, second.TB)
    invalid = profiles.copy(deep=True)
    invalid['T'][0, -1] = np.nan
    with pytest.raises(ValueError, match='RT requires complete profiles'):
        rt.run_RT(invalid, **options, infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
    invalid = profiles.copy(deep=True)
    invalid['lwc'][0, 1] = 1.0  # At 1 cm^-3, this gives about 62 micrometres.
    with pytest.raises(ValueError, match='Calculated droplet effective radii must be below 60.*libRadtran'):
        rt.run_RT(invalid, **options, infrared={'droplet_number_density': 1e6, 'min_effective_radius': 2.5})
