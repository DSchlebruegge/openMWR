"""Shared calendar selections retain complete profiles and inclusive end days."""

from datetime import date

import numpy as np
import pytest
import xarray as xr

from openMWR.config import check_time_selection, select_time


def test_date_and_monthly_filters_intersect_without_changing_values():
    # Deliberately unsorted; the last nanosecond of the end date is included.
    times = np.array(['2025-02-21', '2024-02-22', '2024-02-21T23:59:59.999999999',
                      '2024-02-20', '2024-01-21', '2024-02-21T00:00:00'], dtype='datetime64[ns]')
    data = xr.Dataset({'value': (('time', 'height'), np.arange(12).reshape(6, 2))},
                      coords={'time': times, 'height': [0, 1000]})
    selected = select_time(data, start_date='2024-02-01', end_date='2024-02-21',
                           days_of_month={'first': 21, 'last': 31})
    xr.testing.assert_identical(selected, data.isel(time=[2, 5]))
    xr.testing.assert_identical(select_time(data), data)
    assert not select_time(data, start_date='2026-01-01').time.size


@pytest.mark.parametrize('selection', [
    {'end_date': '2024-02-30'}, {'start_date': date(2024, 1, 1)},
    {'start_date': '2024-01-02', 'end_date': '2024-01-01'},
    {'days_of_month': {'first': 21, 'last': 20}},
])
def test_invalid_time_selections_fail(selection):
    with pytest.raises(ValueError, match='selection'):
        check_time_selection(selection)
