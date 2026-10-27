"""Tiny RPG daily files and checkpoints with deliberately known coefficients."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from openMWR.models import Model, SpectralConsistencyModel
from openMWR.site import create_site

FREQUENCIES = np.array([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.4,
                        51.26, 52.28, 53.86, 54.94, 56.66, 57.3, 58.0])
ANGLES = np.array([90.0, 30.0, 19.2, 14.4, 11.4, 8.4, 6.6, 5.4, 4.8, 4.2, 0.0])


def raw_day(prediction_project, date, *, bls=True):
    directory = prediction_project / 'raw' / date.strftime('Y%Y/M%m/D%d')
    directory.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(date, periods=12, freq='30s')
    sample = np.arange(len(times), dtype=float)
    metadata = {'radiometer_software_version': 'test'}
    brt = xr.Dataset({
        'TBs': (('time', 'number_frequencies'), 100 + sample[:, None] + np.zeros((12, 14))),
        'frequencies': ('number_frequencies', FREQUENCIES),
        'rain_flag': ('time', sample % 3 == 0),
        'azimuth_angle': ('time', sample * 0),
        'elevation_angle': ('time', sample * 0 + 90),
    }, coords={'time': times}, attrs=metadata)
    met = xr.Dataset({
        'env_pressure': ('time', 950 + sample / 10),
        'env_temperature': ('time', 290 + sample / 10),
        'env_relative_humidity': ('time', 50 + sample / 10),
    }, coords={'time': times})
    irt = xr.Dataset({
        'IRR_data': (('time', 'number_wavelength'), -20 + sample[:, None]),
        'frequencies': ('number_wavelength', [10.5]),
    }, coords={'time': times})
    files = {'BRT': brt, 'MET': met, 'IRT': irt}
    if bls:
        scan_times = times[::2]
        files['BLB'] = xr.Dataset({
            'TBs': (('time', 'number_scan_angles', 'number_frequencies'),
                    100 + np.arange(6)[:, None, None] + np.zeros((6, 11, 14))),
            'elevation_scan_angles': ('number_scan_angles', ANGLES),
            'frequencies': ('number_frequencies', FREQUENCIES),
            'rain_flag': ('time', np.zeros(6)),
            'azimuth_angle': ('time', np.zeros(6)),
        }, coords={'time': scan_times}, attrs=metadata)
    for suffix, data in files.items():
        data.to_netcdf(directory / f'{date:%y%m%d}.{suffix}.NC')


def checkpoint(prediction_project, name, *, bls=False, spectral=False, weather=True):
    torch.manual_seed(2)
    if spectral:
        model = SpectralConsistencyModel(name, FREQUENCIES[:2], hidden_layer_size=3)
    else:
        model = Model(name, np.array([0., 1000., 2000.]), FREQUENCIES[:2],
            is_bls_model=bls, is_MLR=True, angles=[90., 30.] if bls else None,
            additional_input_vars=(['surface_T', 'surface_rh', 'surface_p', 'doy_cos', 'doy_sin']
                                   if weather else ['TB_IR', 'doy_cos', 'doy_sin']))
        with torch.no_grad():
            model.stack.weight.zero_()
            model.stack.weight[0, 0] = 0.1
            model.stack.bias.copy_(torch.tensor([290., 283., 276., 50., 50., 50., .01, .01, .01, 20., 10.]))
    specification = xr.merge([model.input, model.output])
    model.nor = xr.Dataset({name: xr.ones_like(values).expand_dims(stat=['mean', 'std', 'max'])
                            for name, values in specification.items()})
    model.save('test', prediction_project, {})


@pytest.fixture
def prediction_project(tmp_path):
    create_site('test', tmp_path, gen='G5', hatpro_data_dir='raw',
                retrieval_output_dir='published', mwr_height=550,
                prediction={'retrievals': ['primary', 'scan']})
    checkpoint(tmp_path, 'primary')
    checkpoint(tmp_path, 'scan', bls=True)
    raw_day(tmp_path, datetime(2024, 1, 1))
    return tmp_path
