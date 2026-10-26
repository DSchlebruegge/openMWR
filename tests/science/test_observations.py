"""Variable-specific validity, inclusive UTC periods and paired sample selection."""

import numpy as np
import pytest
import xarray as xr

from openMWR.config import check_invalid_measurements
from openMWR.observations import mask_invalid_measurements, valid_observation_times
from openMWR.xr_utils import match_analysis_data


def test_invalid_period_preserves_other_variables_and_cannot_be_bridged_by_matching():
    times = np.array(['2024-01-01T23:59', '2024-01-02T00:01', '2024-01-02T12:00',
                      '2024-01-03T00:00'], dtype='datetime64[ns]')
    observations = xr.Dataset({'TB': ('time', [100., 101., 102., 103.]),
                               'surface_p': ('time', [740.] * 4),
                               'unrelated_sensor': ('time', [np.nan] * 4),
                               'rain_flag': ('time', [0.] * 4)}, coords={'time': times})
    periods = [{'variables': ['surface_p'], 'start_date': '2024-01-02', 'end_date': '2024-01-02'}]
    masked = mask_invalid_measurements(observations, periods)
    xr.testing.assert_identical(masked.TB, observations.TB)
    np.testing.assert_array_equal(valid_observation_times(masked, ['TB', 'surface_p']), [True, False, False, True])
    assert np.isfinite(observations.surface_p).all()
    reference = xr.Dataset({'T': (('time', 'height'), np.full((4, 2), 280.))},
        coords={'time': times + np.array([0, -1, 0, 0], dtype='timedelta64[m]'), 'height': [0., 1000.]})
    settings = dict(max_time_difference_minutes=10, exclude_rain=True, height_range=[0, 1000],
                    invalid_measurements=periods)
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


@pytest.mark.parametrize('periods', [None, {}, [{}], [{'variables': [], 'start_date': '2024-01-01'}],
    [{'variables': ['TB'], 'start_date': '2024-02-01', 'end_date': '2024-01-01'}],
    [{'variables': ['TB'], 'start_date': 'not-a-date'}]])
def test_invalid_measurement_declarations_fail_before_loading(periods):
    with pytest.raises(ValueError):
        check_invalid_measurements(periods)
