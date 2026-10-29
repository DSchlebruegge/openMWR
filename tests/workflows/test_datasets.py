"""Dataset creation, incremental updates, and recovery through the public workflow."""

import json
from datetime import date
import tomllib

import numpy as np
import pandas as pd
import pytest
import tomli_w
import xarray as xr

from openMWR.datasets import DatasetWorkflow
from openMWR.era5 import ERA5Processor
from openMWR.paths import dataset_file, era5_cache_file, radiosonde_year_file


from tests.fixtures.datasets import dataset_project, recipe, install_raw, synthetic_simulation, saved_simulation, rebuild_hint

def test_reserved_dataset_name_cannot_rebuild_prediction_outputs(dataset_project):
    directory = dataset_project / 'data/sites/test/predictions'
    directory.mkdir()
    saved = directory / 'daily.nc'
    saved.write_bytes(b'Saved prediction product')
    # Case variants also collide on case-insensitive filesystems.
    for name in ('predictions', 'Predictions'):
        settings = recipe(dataset_project, name, kind='radiosonde', stations=['03715'])
        before = settings.read_bytes()
        with pytest.raises(ValueError, match='reserved for prediction outputs'):
            DatasetWorkflow(dataset_project, 'test').build(name, rebuild=True)
        assert saved.read_bytes() == b'Saved prediction product'
        assert settings.read_bytes() == before


def test_real_preparation_rt_shared_dependencies_and_incremental_updates(dataset_project, monkeypatch):
    import openMWR.run_RT as rt

    # Only the external infrared solver is replaced; microwave RT is real.
    calls = []
    def infrared(*args, **kwargs):
        calls.append(1)
        return 200.0
    monkeypatch.setattr(rt, 'calculate_IR_band_TB', infrared)
    times = pd.date_range('2020-01-01', periods=3, freq='12h').values
    install_raw(dataset_project, times)
    workflow = DatasetWorkflow(dataset_project, 'test')
    order = workflow.build(['r17', 'r20'], with_dependencies=True)
    assert order == ['source', 'profiles', 'r17', 'r20']
    assert len(calls) == 6
    profile_path = dataset_file(dataset_project, 'test', 'profiles', '03715')
    old_profiles = xr.load_dataset(profile_path)
    np.testing.assert_allclose(old_profiles.rh.sel(height=0), 8.0)
    old_rt = xr.load_dataset(dataset_file(dataset_project, 'test', 'r17', '03715'))
    assert np.isfinite(old_rt.TB).all()
    assert not np.array_equal(old_rt.TB.values, xr.load_dataset(dataset_file(dataset_project, 'test', 'r20', '03715')).TB.values)

    workflow.build('r17', with_dependencies=True)
    assert len(calls) == 6  # Repeating the command performs no microwave/IR work.
    new_times = pd.date_range('2020-01-01', periods=4, freq='12h').values
    install_raw(dataset_project, new_times)
    workflow.build('r17', with_dependencies=True)
    assert len(calls) == 7
    updated_profiles = xr.load_dataset(profile_path)
    xr.testing.assert_identical(updated_profiles.sel(synop=times), old_profiles)
    updated_rt = xr.load_dataset(dataset_file(dataset_project, 'test', 'r17', '03715'))
    xr.testing.assert_identical(updated_rt.sel(time=times), old_rt)
    assert xr.load_dataset(dataset_file(dataset_project, 'test', 'r20', '03715')).time.size == 3


def test_split_creation_update_and_target_only_rebuild(dataset_project):
    times = pd.date_range('2020-01-01', periods=40).values
    workflow = saved_simulation(dataset_project, times)
    original = dataset_file(dataset_project, 'test', 'r17', '03715').read_bytes()
    workflow.build('training')
    paths = {split: dataset_file(dataset_project, 'test', 'training', '03715', split=split)
             for split in ('training', 'validation', 'test')}
    before = {split: xr.load_dataset(path) for split, path in paths.items()}
    assert [before[key].time.size for key in ('training','validation','test')] == [31, 3, 6]
    hashes = {key: path.read_bytes() for key, path in paths.items()}
    workflow.build('training')
    assert hashes == {key: path.read_bytes() for key, path in paths.items()}
    all_times = pd.date_range('2020-01-01', periods=60).values
    synthetic_simulation(all_times).to_netcdf(dataset_file(dataset_project, 'test', 'r17', '03715'))
    workflow.build('training')
    after = {split: xr.load_dataset(path) for split, path in paths.items()}
    for split, data in before.items():
        xr.testing.assert_identical(after[split].sel(time=data.time), data)
    combined = np.concatenate([data.time.values for data in after.values()])
    assert np.unique(combined).size == len(combined) == 60
    # Rebuilding this selected split leaves its upstream simulation untouched.
    source_bytes = dataset_file(dataset_project, 'test', 'r17', '03715').read_bytes()
    workflow.build('training', rebuild=True)
    assert dataset_file(dataset_project, 'test', 'r17', '03715').read_bytes() == source_bytes
    assert source_bytes != original
    # Rebuilding gives exactly the same memberships and profile values.
    for split, data in after.items():
        xr.testing.assert_identical(xr.load_dataset(paths[split]), data)


def test_calendar_split_groups_whole_days_across_sources_and_months(dataset_project):
    recipe(dataset_project, 'source', kind='radiosonde', stations=['03715', '02290'], start_year=2020, end_year=2020)
    dates = pd.to_datetime(['2020-01-01', '2020-01-03', '2020-01-04', '2020-01-15', '2020-01-17',
                            '2020-02-02', '2020-02-16', '2020-02-29', '2020-03-01'])
    times = np.sort(np.concatenate([dates.values, (dates + pd.Timedelta(hours=12)).values]))
    workflow = saved_simulation(dataset_project, times)
    other = synthetic_simulation(times).assign_coords(station=['02290'])
    other.to_netcdf(dataset_file(dataset_project, 'test', 'r17', '02290'))
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
            part = xr.load_dataset(dataset_file(dataset_project, 'test', 'training', source, split=split))
            xr.testing.assert_identical(part, original.sel(time=expected_times))


def test_short_calendar_period_reports_empty_partition_before_writing(dataset_project):
    workflow = saved_simulation(dataset_project, pd.date_range('2020-01-01', periods=3).values)
    with pytest.raises(ValueError, match='No training profiles.*adjust the period or split day ranges'):
        workflow.build('training')
    assert not list((dataset_project / 'data/sites/test/training').glob('*.nc'))


def test_split_period_selects_all_three_sets_before_assignment(dataset_project):
    times = pd.date_range('2019-12-31', '2020-02-02', freq='12h').values
    workflow = saved_simulation(dataset_project, times)
    recipe(dataset_project, 'training', kind='split', inputs={'simulation': 'r17'},
           split={'start_date': '2020-01-02', 'end_date': '2020-01-31'})
    workflow.build('training')
    paths = [dataset_file(dataset_project, 'test', 'training', '03715', split=split)
             for split in ('training', 'validation', 'test')]
    before = [xr.load_dataset(path) for path in paths]
    combined = np.sort(np.concatenate([part.time.values for part in before]))
    np.testing.assert_array_equal(combined, pd.date_range('2020-01-02', '2020-01-31T12:00', freq='12h').values)
    # Source updates outside the selected period do not add samples to any split.
    extra = pd.date_range('2019-12-31', '2020-02-15', freq='12h').values
    synthetic_simulation(extra).to_netcdf(dataset_file(dataset_project, 'test', 'r17', '03715'))
    workflow.build('training')
    for original, path in zip(before, paths):
        xr.testing.assert_identical(original, xr.load_dataset(path))
    recipe(dataset_project, 'training', kind='split', inputs={'simulation': 'r17'},
           split={'start_date': '2020-01-02', 'end_date': '2020-01-30'})
    with pytest.raises(ValueError, match='end_date changed'):
        DatasetWorkflow(dataset_project, 'test').build('training')
    for original, path in zip(before, paths):
        xr.testing.assert_identical(original, xr.load_dataset(path))


def test_changed_settings_require_rebuild_before_any_deletion(dataset_project):
    times = pd.date_range('2020-01-01', periods=20).values
    saved_simulation(dataset_project, times).build('training')
    path = dataset_file(dataset_project, 'test', 'training', '03715', split='training')
    original = path.read_bytes()
    recipe(dataset_project, 'training', kind='split', inputs={'simulation': 'r17'}, split={'test_days_of_month': {'first': 1, 'last': 6}})
    with pytest.raises(ValueError, match='settings or input definitions changed') as error:
        DatasetWorkflow(dataset_project, 'test').build('training')
    assert 'training: test_days_of_month.last changed from 3 to 6' in str(error.value)
    with pytest.raises(ValueError):
        DatasetWorkflow(dataset_project, 'test').require('training')
    hint = rebuild_hint(error.value)
    assert hint[3:hint.index('--project-dir')] == ['training']
    assert path.read_bytes() == original
    from openMWR.cli import main
    main(hint[1:])
    assert path.read_bytes() != original
    # Rebuilt input identity also requires downstream regeneration.
    workflow = DatasetWorkflow(dataset_project, 'test')
    workflow.record('r17', rebuilt=True)
    with pytest.raises(ValueError, match='input r17 was rebuilt'):
        workflow.build('training')


@pytest.mark.parametrize('change', ['setting', 'input_name', 'corrupt_input', 'selected_rebuild'])
def test_rebuild_hint_repairs_only_the_affected_chain(dataset_project, monkeypatch, change):
    from openMWR.cli import main

    monkeypatch.setattr('openMWR.run_RT.calculate_IR_band_TB', lambda *args, **kwargs: 200.0)
    def no_download(*args, **kwargs):
        pytest.fail('A rebuild hint must reuse the healthy raw cache.')
    monkeypatch.setattr('openMWR.dwd_opendata.update_radiosonde_data', no_download)
    install_raw(dataset_project, pd.to_datetime(['2020-01-01', '2020-01-04', '2020-01-15']).values)
    DatasetWorkflow(dataset_project, 'test').build(['training', 'r20'], with_dependencies=True)
    raw_path = radiosonde_year_file(dataset_project, '03715', 2020)
    raw_bytes = raw_path.read_bytes()
    r20_path = dataset_file(dataset_project, 'test', 'r20', '03715')
    r20_bytes = r20_path.read_bytes()
    targets = ['training', 'r20']
    options = {}
    expected = ['r17', 'training']
    if change == 'setting':
        path = dataset_project / 'sites/test/datasets/r17.toml'
        values = tomllib.loads(path.read_text())
        values['radiative_transfer']['absorption_model'] = 'R20'
        path.write_text(tomli_w.dumps(values))
    elif change == 'input_name':
        path = dataset_project / 'sites/test/datasets/training.toml'
        values = tomllib.loads(path.read_text())
        values['inputs']['simulation'] = 'r20'
        path.write_text(tomli_w.dumps(values))
        expected = ['training']
    elif change == 'corrupt_input':
        (dataset_project / 'data/sites/test/r17/state.json').write_text('{')
    else:
        targets = ['profiles', 'training']
        options = {'with_dependencies': True, 'rebuild': True}
        expected = ['profiles', 'r17', 'training']
    before = {path: path.read_bytes() for path in (dataset_project / 'data').rglob('*') if path.is_file()}
    with pytest.raises(ValueError) as error:
        DatasetWorkflow(dataset_project, 'test').build(targets, **options)
    assert before == {path: path.read_bytes() for path in (dataset_project / 'data').rglob('*') if path.is_file()}
    message = str(error.value)
    if change == 'setting':
        assert 'r17: absorption_model changed from "R17" to "R20"' in message
    elif change == 'input_name':
        assert 'training: inputs.simulation changed from "r17" to "r20"' in message
    elif change == 'corrupt_input':
        assert str(dataset_project / 'data/sites/test/r17/state.json') in message
        assert 'processing history cannot be verified' in message
    hint = rebuild_hint(error.value)
    assert hint[3:hint.index('--project-dir')] == expected
    assert hint[hint.index('--project-dir') + 1] == str(dataset_project)
    assert '--with-dependencies' not in hint
    main(hint[1:])
    assert raw_path.read_bytes() == raw_bytes
    assert r20_path.read_bytes() == r20_bytes
    DatasetWorkflow(dataset_project, 'test').require('training')


@pytest.mark.parametrize('damage', ['missing', 'invalid_json', 'invalid_structure', 'unknown_settings'])
def test_missing_or_damaged_state_has_a_working_recovery_hint(dataset_project, damage):
    from openMWR.cli import main

    workflow = saved_simulation(dataset_project, pd.date_range('2020-01-01', periods=20).values)
    workflow.build('training')
    path = dataset_project / 'data/sites/test/training/state.json'
    if damage == 'unknown_settings':
        workflow.record('training', requires_rebuild=True, provenance='Historical settings are unknown')
    elif damage == 'missing':
        path.unlink()
    elif damage == 'invalid_json':
        path.write_text('{')
    else:
        state = json.loads(path.read_text())
        state['definition']['inputs']['simulation'] = []
        path.write_text(json.dumps(state))
    files = {p: p.read_bytes() for p in (dataset_project / 'data').rglob('*') if p.is_file()}
    for action in (lambda: workflow.build('training'), lambda: workflow.require('training')):
        with pytest.raises(ValueError) as error:
            action()
        if damage == 'unknown_settings':
            assert 'Historical settings are unknown' in str(error.value)
        else:
            assert str(path) in str(error.value)
            assert 'processing history cannot be verified' in str(error.value)
        hint = rebuild_hint(error.value)
        assert hint[3:hint.index('--project-dir')] == ['training']
    assert files == {p: p.read_bytes() for p in (dataset_project / 'data').rglob('*') if p.is_file()}
    simulation_path = dataset_file(dataset_project, 'test', 'r17', '03715')
    simulation_bytes = simulation_path.read_bytes()
    main(hint[1:])
    assert simulation_path.read_bytes() == simulation_bytes
    DatasetWorkflow(dataset_project, 'test').require('training')


def test_readiness_checks_selected_sources_and_splits(dataset_project):
    workflow = DatasetWorkflow(dataset_project, 'test')
    workflow.recipe('source').values['stations'].append('02290')
    for split in ('training', 'validation'):
        path = dataset_file(dataset_project, 'test', 'training', '03715', split=split)
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


def test_build_keeps_one_date_snapshot_and_refreshes_next_time(dataset_project, monkeypatch):
    class Clock(date):
        current = date(2020, 1, 6)

        @classmethod
        def today(cls):
            return cls.current

    monkeypatch.setattr('openMWR.datasets.date', Clock)
    recipe(dataset_project, 'era5', kind='era5', start_date='2020-01-01')
    recipe(dataset_project, 'era5_profiles', kind='profiles', inputs={'source': 'era5'},
           preparation={'heights': [0., 1000., 2000.]})
    requested = []

    def download(self, start, end, h_freq=6):
        requested.append((start, end))
        Clock.current = date(2020, 1, 7)  # The download crosses midnight.
        config = dataset_project / 'sites/test/site.toml'
        settings = tomllib.loads(config.read_text())
        settings['mwr_height'] = 1550
        config.write_text(tomli_w.dumps(settings))
        times = pd.date_range(start, end + pd.Timedelta(days=1), freq=f'{h_freq}h', inclusive='left')
        return xr.Dataset({name: (('time', 'p'), np.broadcast_to(values, (len(times), 3)).copy())
            for name, values in {'height': [550., 1550., 2550.], 'T': [280., 274., 268.],
                                 'rh': [40., 50., 60.], 'lwc': [0., 0., 0.]}.items()},
            coords={'time': times, 'p': [950., 850., 750.]})

    monkeypatch.setattr(ERA5Processor, 'download_era5', download)
    workflow = DatasetWorkflow(dataset_project, 'test')
    workflow.build('era5_profiles', with_dependencies=True)
    state_dir = dataset_project / 'data/sites/test'
    source = json.loads((state_dir / 'era5/state.json').read_text())
    profiles = json.loads((state_dir / 'era5_profiles/state.json').read_text())
    assert source['effective']['end_date'] == '2020-01-01'
    assert profiles['coverage']['inputs']['source']['range']['end_date'] == '2020-01-01'
    output = dataset_file(dataset_project, 'test', 'era5_profiles', 'era5')
    assert xr.load_dataset(output).time.size == 4
    # Source metadata and preparation use the original 550 m instrument height.
    np.testing.assert_allclose(xr.load_dataset(output).T.sel(height=0), 280.)
    assert DatasetWorkflow(dataset_project, 'test').site_settings.require('mwr_height') == 1550
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


def test_era5_profiles_are_reusable_and_incremental(dataset_project):
    recipe(dataset_project, 'era5', kind='era5', start_date='2020-01-01', end_date='2020-01-01', hour_interval=6)
    recipe(dataset_project, 'era5_profiles', kind='profiles', inputs={'source': 'era5'},
           preparation={'heights': [0., 1000., 2000.]}, execution={'processes': 1})
    raw_file = era5_cache_file(dataset_project, 48.148, 11.573, 6)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range('2020-01-01', periods=2, freq='6h')
    shape = (2, 3)
    raw = xr.Dataset({name: (('time','p'), np.broadcast_to(values, shape).copy())
        for name, values in {'height':[550.,1550.,2550.], 'T':[280.,274.,268.], 'rh':[40.,50.,60.], 'lwc':[0.,0.,0.]}.items()},
        coords={'time':times, 'p':[950.,850.,750.]})
    raw.to_netcdf(raw_file)
    workflow = DatasetWorkflow(dataset_project, 'test')
    workflow.build('era5_profiles')
    output = dataset_file(dataset_project, 'test', 'era5_profiles', 'era5')
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
    recipe(dataset_project, 'era5', kind='era5', start_date='2020-01-01', end_date='2020-01-02', hour_interval=6)
    DatasetWorkflow(dataset_project, 'test').build('era5_profiles')
    assert xr.load_dataset(output).time.size == 4
    recipe(dataset_project, 'era5', kind='era5', start_date='2020-01-01', end_date='2020-01-01', hour_interval=6)
    with pytest.raises(ValueError, match='date coverage narrowed') as error:
        DatasetWorkflow(dataset_project, 'test').build('era5_profiles')
    assert 'era5: end_date changed from "2020-01-02" to "2020-01-01"' in str(error.value)
    raw_bytes = raw_file.read_bytes()
    DatasetWorkflow(dataset_project, 'test').build('era5_profiles', rebuild=True)
    assert xr.load_dataset(output).time.size == 3
    assert raw_file.read_bytes() == raw_bytes


def test_named_analysis_uses_its_inputs_and_refreshes_only_when_changed(dataset_project, monkeypatch):
    import openMWR.run_RT as rt

    monkeypatch.setattr(rt, 'calculate_IR_band_TB', lambda *args, **kwargs: 200.0)
    times = pd.date_range('2020-01-01', periods=3, freq='12h').values
    install_raw(dataset_project, times)
    workflow = DatasetWorkflow(dataset_project, 'test')
    workflow.build('r17', with_dependencies=True)
    recipe(dataset_project, 'observations', kind='hatpro')
    recipe(dataset_project, 'comparison', kind='analysis', inputs={'observations': 'observations', 'reference': 'r17'})
    measured_times = times - np.timedelta64(75, 'm')
    measured = xr.Dataset({'TB': (('time', 'frq'), np.full((3, 1), 100.)),
                          'surface_p': ('time', [740.] * 3), 'rain_flag': ('time', [0., 0., 0.])},
                          coords={'time': measured_times, 'frq': [22.24]})
    for bls in (False, True):
        path = dataset_file(dataset_project, 'test', 'observations', bls=bls)
        path.parent.mkdir(parents=True, exist_ok=True)
        (measured.expand_dims(ang=[90.0]) if bls else measured).to_netcdf(path)
    workflow.record('observations')
    workflow.build('comparison')
    output = dataset_file(dataset_project, 'test', 'comparison')
    pairs = xr.load_dataset(output)
    assert pairs.time.size == 3
    assert set(pairs.data_source.values) == {'radiosonde', 'hatpro'}
    np.testing.assert_allclose(pairs.TB.sel(data_source='hatpro'), 100.)
    original = output.read_bytes()
    workflow.build('comparison')
    assert output.read_bytes() == original
    measured.assign(TB=measured.TB + 2).to_netcdf(dataset_file(dataset_project, 'test', 'observations'))
    workflow.build('comparison')
    np.testing.assert_allclose(xr.load_dataset(output).TB.sel(data_source='hatpro'), 102.)

    measured = measured.assign_coords(time=measured_times + np.array([0, 6, 12], dtype='timedelta64[m]'))
    measured['rain_flag'] = ('time', [0., 1., 0.])
    for bls in (False, True):
        (measured.expand_dims(ang=[90.0]) if bls else measured).to_netcdf(
            dataset_file(dataset_project, 'test', 'observations', bls=bls))
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

    # Existing paired files must also respect newly declared measurement issues.
    from openMWR.observations import load_analysis_dataset

    site = dataset_project / 'sites/test/site.toml'
    site.write_text(site.read_text() + '''
[[invalid_measurements]]
variables = ["surface_p"]
periods = [{ start_date = "2020-01-01", end_date = "2020-01-01" }]
''')
    recipe(dataset_project, 'comparison', kind='analysis', inputs={'observations': 'observations', 'reference': 'r17'},
           analysis={'required_observations': ['TB', 'surface_p']})
    before = output.read_bytes()
    filtered = load_analysis_dataset(dataset_project, 'test', 'comparison')
    np.testing.assert_array_equal(filtered.time, measured_times[:1])
    xr.testing.assert_identical(filtered.sel(data_source='radiosonde').TB,
                                pairs.sel(data_source='radiosonde').TB.isel(time=[0]))
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match='invalid_measurements'):
        DatasetWorkflow(dataset_project, 'test').build('comparison')
