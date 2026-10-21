"""Offline data workflows with real NetCDF, preparation, RT and split outputs."""

import json
from datetime import date
from pathlib import Path
import subprocess
import sys

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
    recipe(tmp_path, 'training', kind='split', inputs={'simulation': 'r17'}, split={'seed': 12})
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
        '--project-dir', str(project), '--with-dependencies', '--dry-run'],
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
    ({'kind': 'profiles', 'inputs': {'source': 'source'}, 'preparation': {'heights': [100, 0]}}, 'increase from zero'),
    ({'kind': 'profiles', 'inputs': {'source': 'source'}, 'execution': {'processes': 2}}, 'preparation is serial'),
])
def test_invalid_graphs_fail_before_output_creation(project, values, message):
    recipe(project, 'bad', **values)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        DatasetWorkflow(project, 'test').build('bad', with_dependencies=True)
    assert list((project / 'data/sites/test').iterdir()) == []


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
    assert [before[key].time.size for key in ('training','validation','test')] == [33, 3, 4]
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
    workflow.build('training', rebuild=True)
    assert dataset_file(project, 'test', 'r17', '03715').read_bytes() == source_bytes
    assert source_bytes != original


def test_changed_settings_require_rebuild_before_any_deletion(project):
    times = pd.date_range('2020-01-01', periods=20).values
    saved_simulation(project, times).build('training')
    path = dataset_file(project, 'test', 'training', '03715', split='training')
    original = path.read_bytes()
    recipe(project, 'training', kind='split', inputs={'simulation': 'r17'}, split={'test_fraction': 0.2})
    with pytest.raises(ValueError, match='settings or input definitions changed'):
        DatasetWorkflow(project, 'test').build('training')
    assert path.read_bytes() == original
    DatasetWorkflow(project, 'test').build('training', rebuild=True)
    assert path.read_bytes() != original
    # Rebuilt input identity also requires downstream regeneration.
    workflow = DatasetWorkflow(project, 'test')
    workflow.record('r17', rebuilt=True)
    with pytest.raises(ValueError, match='input r17 was rebuilt'):
        workflow.build('training')


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
        workflow.recipe('training').values['split']['seed'] = 13
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
    with pytest.raises(ValueError, match='date coverage narrowed'):
        DatasetWorkflow(project, 'test').build('era5_profiles')
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
