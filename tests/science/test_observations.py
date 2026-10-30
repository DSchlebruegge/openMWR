"""Variable-specific validity, inclusive UTC periods and paired sample selection."""

import numpy as np
import pytest
import xarray as xr

from openMWR.config import check_invalid_measurements
from openMWR.observations import mask_invalid_measurements, valid_observation_times, reference_source
from openMWR.xr_utils import match_analysis_data, mean_n_min


def test_reference_selection_infers_only_an_unambiguous_background():
    paired = xr.Dataset(coords={'data_source': ['hatpro', 'era5', 'radiosonde']})
    assert reference_source(paired.sel(data_source=['hatpro', 'era5'])) == 'era5'
    assert reference_source(paired, 'radiosonde') == 'radiosonde'
    with pytest.raises(ValueError, match='multiple references'):
        reference_source(paired)
    with pytest.raises(ValueError, match='absent'):
        reference_source(paired.sel(data_source=['hatpro', 'era5']), 'radiosonde')
    with pytest.raises(ValueError, match='distinct reference sources'):
        reference_source(paired.sel(data_source=['hatpro']))
    with pytest.raises(ValueError, match='distinct reference sources'):
        reference_source(paired.sel(data_source=['hatpro', 'era5', 'era5']))


def test_invalid_period_preserves_other_variables_and_cannot_be_bridged_by_matching():
    times = np.array(['2024-01-01T23:59', '2024-01-02T00:01', '2024-01-02T12:00',
                      '2024-01-03T00:00'], dtype='datetime64[ns]')
    observations = xr.Dataset({'TB': ('time', [100., 101., 102., 103.]),
                               'surface_p': ('time', [740.] * 4),
                               'unrelated_sensor': ('time', [np.nan] * 4),
                               'rain_flag': ('time', [0.] * 4)}, coords={'time': times})
    rules = [{'variables': ['surface_p'],
              'periods': [{'start_date': '2024-01-02', 'end_date': '2024-01-02'}]}]
    masked = mask_invalid_measurements(observations, rules)
    xr.testing.assert_identical(masked.TB, observations.TB)
    np.testing.assert_array_equal(valid_observation_times(masked, ['TB', 'surface_p']), [True, False, False, True])
    assert np.isfinite(observations.surface_p).all()
    reference = xr.Dataset({'T': (('time', 'height'), np.full((4, 2), 280.))},
        coords={'time': times + np.array([0, -1, 0, 0], dtype='timedelta64[m]'), 'height': [0., 1000.]})
    settings = dict(max_time_difference_minutes=10, exclude_rain=True, height_range=[0, 1000],
                    invalid_measurements=rules)
    measured, paired = match_analysis_data(observations, reference,
        required_observations=['TB', 'surface_p'], **settings)
    np.testing.assert_array_equal(measured.time, reference.time.values[[0, 3]])
    xr.testing.assert_identical(paired, reference.isel(time=[0, 3]))
    measured, _ = match_analysis_data(observations, reference, required_observations=['TB'], **settings)
    assert measured.time.size == 4

    # Complete inputs must belong to the same station/time pair.
    stations = xr.Dataset({'TB': (('station', 'time'), [[100.], [np.nan]]),
                           'surface_p': (('station', 'time'), [[np.nan], [740.]])},
                          coords={'station': ['a', 'b'], 'time': times[:1]})
    valid = valid_observation_times(stations, ['TB', 'surface_p'])
    assert set(valid.dims) == {'time', 'station'} and not bool(valid.any())


def test_exact_utc_intervals_preserve_good_samples_between_faults_before_averaging():
    times = np.array(['2024-01-02T12:00:00', '2024-01-02T12:00:10',
                      '2024-01-02T12:00:10.000001', '2024-01-02T12:00:10.000002',
                      '2024-01-02T12:00:30', '2024-01-02T12:00:40',
                      '2024-01-02T12:01:00'], dtype='datetime64[ns]')
    data = xr.Dataset({'TB_IR': ('time', [200., 124., 124., 220., 124., 240., 260.]),
                       'surface_p': ('time', [740.] * 7)}, coords={'time': times})
    rules = [dict(variables=['TB_IR'], reason='Two short outages', periods=[
        dict(start_time='2024-01-02T12:00:10Z', end_time='2024-01-02T12:00:10.000002Z'),
        dict(start_time='2024-01-02T12:00:30Z', end_time='2024-01-02T12:00:40Z')])]
    masked = mask_invalid_measurements(data, rules)
    np.testing.assert_array_equal(masked.TB_IR, [200., np.nan, np.nan, 220., np.nan, 240., 260.])
    xr.testing.assert_identical(masked.surface_p, data.surface_p)
    np.testing.assert_array_equal(data.TB_IR, [200., 124., 124., 220., 124., 240., 260.])
    np.testing.assert_allclose(mean_n_min(masked, 1).TB_IR, [220., 260.])

    ongoing = mask_invalid_measurements(data, [dict(variables=['surface_p'],
        periods=[dict(start_time='2024-01-02T12:00:40Z')])])
    np.testing.assert_array_equal(ongoing.surface_p, [740.] * 5 + [np.nan, np.nan])
    xr.testing.assert_identical(ongoing.TB_IR, data.TB_IR)


@pytest.mark.parametrize('rules', [None, {}, [{}],
    [{'variables': [], 'periods': [{'start_date': '2024-01-01'}]}],
    [{'variables': ['TB'], 'start_date': '2024-01-01'}],
    [{'variables': ['TB'], 'periods': []}],
    [{'variables': ['TB'], 'periods': {}}],
] + [[{'variables': ['TB_IR'], 'periods': [period]}] for period in [
    {}, None,
    {'start_date': '2024-02-01', 'end_date': '2024-01-01'},
    {'start_date': 'not-a-date'},
    {'end_time': '2024-01-02T12:00:00Z'},
    {'start_time': '2024-01-02T12:00:00'},
    {'start_time': '2024-01-02T12:00:00+01:00'},
    {'start_time': '2024-02-30T12:00:00Z'},
    {'start_time': '2024-01-02T12:00:00Z', 'end_time': None},
    {'start_time': '2024-01-02T12:00:00Z', 'end_time': '2024-01-02T12:00:00Z'},
    {'start_time': '2024-01-02T12:00:00Z', 'end_time': '2024-01-02T11:00:00Z'},
    {'start_time': '2024-01-02T12:00:00Z', 'end_date': '2024-01-03'},
    {'start_time': '2024-01-02T12:00:00Z', 'start_date': '2024-01-02'},
]])
def test_invalid_measurement_declarations_fail_before_loading(rules):
    with pytest.raises(ValueError):
        check_invalid_measurements(rules)
