"""Independent brightness temperatures, batch coordinates and humidity input contracts."""

from hashlib import sha256

import json

import numpy as np
import pytest
import xarray as xr

from torchMWRT import AtmProfile
from tests.fixtures.radiative_transfer import DATA, FREQUENCIES, profile_inputs, torch_brightness


REFERENCE = json.loads((DATA / 'rt_reference.json').read_text())

def case_name(case):
    flags = '-'.join(name for name in ('clear', 'ray_tracing', 'from_sat') if case.get(name))
    return f'{case["source"]}-{case.get("time_idx", "")}-{case["absorption_model"]}-{case["angles"]}-{flags}'


def test_reference_inputs_are_unchanged():
    assert REFERENCE['frequencies_GHz'] == FREQUENCIES
    for name, digest in REFERENCE['inputs_sha256'].items():
        assert sha256((DATA / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=case_name)
def test_brightness_temperature_matches_independent_reference(case):
    result = torch_brightness(case)
    assert result.dims == ('frq', 'ang')
    np.testing.assert_array_equal(result.frq, FREQUENCIES)
    np.testing.assert_array_equal(result.ang, case['angles'])
    # Retain the original tolerances: the gridded fixture allows small backend
    # differences; radiosonde R17 cases agree within 1e-8 K.
    rtol, atol = (5e-5, 1e-6) if 'time_idx' in case else (0, 1e-8)
    np.testing.assert_allclose(result, case['tb_K'], rtol=rtol, atol=atol)


@pytest.mark.parametrize('ray_tracing', [False, True], ids=['straight', 'curved'])
def test_time_batch_preserves_coordinates_and_reference_values(ray_tracing):
    cases = [next(case for case in REFERENCE['cases']
                  if case.get('time_idx') == i and case['angles'] == [90., 50.]
                  and case['absorption_model'] == 'R17'
                  and case.get('ray_tracing', False) == ray_tracing)
             for i in (11, 12, 24)]
    from torchMWRT import RTModel

    inputs = [profile_inputs(case) for case in cases]
    batch = {name: xr.concat([item[name] for item in inputs], dim='time')
             if name != 'height' else inputs[0][name] for name in inputs[0]}
    result = RTModel(freqs=np.array(FREQUENCIES), angles=np.array([90., 50.]),
                     absmdl='R17', ray_tracing=ray_tracing).execute(
                         AtmProfile(**batch), return_ds=True).tbtotal
    assert result.dims == ('time', 'frq', 'ang')
    np.testing.assert_array_equal(result.time, batch['temperature'].time)
    np.testing.assert_allclose(result, [case['tb_K'] for case in cases], rtol=5e-5, atol=1e-6)


def test_profile_orientation_shapes_and_explicit_humidity():
    inputs = profile_inputs(REFERENCE['cases'][-1])
    ascending = AtmProfile(**inputs)
    descending = AtmProfile(**{name: value.isel(height=slice(None, None, -1)).copy(deep=True)
                               for name, value in inputs.items()})
    np.testing.assert_array_equal(descending.z, ascending.z)
    np.testing.assert_allclose(descending.e, ascending.e)
    np.testing.assert_array_equal(descending.tk, ascending.tk)
    with pytest.raises(ValueError, match='exactly one'):
        AtmProfile(**inputs, q=np.zeros_like(inputs['rh']))
    bad = dict(inputs, pressure=np.array([1000., 900.]))
    with pytest.raises(ValueError, match='vertical length|height profile'):
        AtmProfile(**bad)


@pytest.mark.reference
@pytest.mark.parametrize('index', [0, -1], ids=['gridded', 'radiosonde'])
def test_reference_agrees_with_installed_pyrtlib(index):
    """Optional dependency-upgrade check; default numerical tests use fixed data."""
    from tests.tools.generate_rt_reference import pyrtlib_brightness

    case = REFERENCE['cases'][index]
    np.testing.assert_allclose(pyrtlib_brightness(case), case['tb_K'], rtol=5e-5, atol=1e-6)
