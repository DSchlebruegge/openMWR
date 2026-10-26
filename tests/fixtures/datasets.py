"""Small explicit recipes and NetCDF inputs; no downloads or shared mutable data."""

import shlex
import tomllib

import numpy as np
import pytest
import tomli_w
import xarray as xr

from openMWR.datasets import DatasetWorkflow
from openMWR.paths import dataset_file, radiosonde_year_file
from openMWR.site import create_site


def recipe(dataset_project, name, **values):
    # Explicit fixture choices; production loaders do not fill these values.
    base = {
        'radiosonde': {'start_year': 1990, 'end_year': 'current_year'},
        'era5': {'start_date': 'site', 'end_date': 'today', 'hour_interval': 6, 'delay_days': 5},
        'hatpro': {'start_date': 'site', 'end_date': 'today'},
        'profiles': {'preparation': {'heights': 'levels_208_30km'}},
        'simulation': {'radiative_transfer': {'backend': 'torchMWRT', 'absorption_model': 'R24',
                                              'frequencies': 'hatpro', 'angles': 'hatpro_bls',
                                              'infrared': {'droplet_number_density': 200e6, 'min_effective_radius': 2.5}}},
        'analysis': {'analysis': {'max_time_difference_minutes': 10, 'exclude_rain': True, 'height_range': [0, 10000], 'required_observations': ['TB']}},
        'split': {'split': {'test_days_of_month': {'first': 1, 'last': 3}, 'validation_days_of_month': {'first': 15, 'last': 17}}},
    }.get(values['kind'], {})
    if values['kind'] == 'profiles':
        source = dataset_project / 'sites/test/datasets' / (values.get('inputs', {}).get('source', '') + '.toml')
        if source.is_file() and tomllib.loads(source.read_text())['kind'] == 'radiosonde':
            base['preparation'].update(incomplete_profiles='monthly_mean', cloud={'rh_threshold': 0.96, 'min_temperature': 253.15, 'adiabatic_fraction': 'power_law', 'a': 0.7, 'b': 2.3},
                                       quality={'min_top_height': 15000, 'min_levels': 500})
    for key, value in base.items():
        values[key] = {**value, **values.get(key, {})} if isinstance(value, dict) else values.get(key, value)
    path = dataset_project / 'sites/test/datasets' / f'{name}.toml'
    path.write_text(tomli_w.dumps(values))
    return path


@pytest.fixture
def dataset_project(tmp_path):
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


def install_raw(dataset_project, times):
    path = radiosonde_year_file(dataset_project, '03715', 2020)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_profiles(times).to_netcdf(path)


def synthetic_simulation(times):
    return xr.Dataset({'T': (('time', 'height'), np.arange(len(times))[:, None] + [[280., 279.]]),
                       'TB': (('time', 'frq'), np.arange(len(times))[:, None] + 100.)},
                      coords={'time': times, 'height': [0., 100.], 'frq': [22.24]}).expand_dims(station=['03715'])


def saved_simulation(dataset_project, times):
    path = dataset_file(dataset_project, 'test', 'r17', '03715')
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_simulation(times).to_netcdf(path)
    workflow = DatasetWorkflow(dataset_project, 'test')
    workflow.record('r17')
    return workflow


def rebuild_hint(error):
    return shlex.split(next(line.strip() for line in str(error).splitlines()
                            if line.strip().startswith('openMWR data ')))
