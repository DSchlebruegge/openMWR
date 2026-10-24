"""The reference inputs use metres, kelvin, hPa, RH fraction and g/m³ cloud water."""

from pathlib import Path

import numpy as np
import xarray as xr

from torchMWRT import AtmProfile, RTModel

DATA = Path(__file__).resolve().parents[1] / 'data'
FREQUENCIES = [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.4,
               51.26, 52.28, 53.86, 54.94, 56.66, 57.3, 58.0]


def profile_inputs(case):
    data = xr.load_dataset(DATA / case['source'])
    if 'time_idx' in case:
        data = data.isel(time=case['time_idx'])
    if 'temperature' in data:
        inputs = dict(temperature=data.temperature, height=data.height,
                      pressure=data.pressure / 100, rh=data.rh / 100,
                      lwc=data.LWC, iwc=data.IWC)
    else:
        inputs = dict(temperature=data.T, height=data.height,
                      pressure=data.p, rh=data.rh / 100,
                      lwc=data.lwc)
    if case.get('clear', False):
        inputs.pop('lwc', None)
        inputs.pop('iwc', None)
    return inputs


def torch_brightness(case):
    model = RTModel(freqs=np.array(FREQUENCIES), angles=np.array(case['angles']),
                    absmdl=case['absorption_model'],
                    ray_tracing=case.get('ray_tracing', False),
                    from_sat=case.get('from_sat', False))
    return model.execute(AtmProfile(**profile_inputs(case)), return_ds=True).tbtotal
