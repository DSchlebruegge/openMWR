"""Physical OMB moments and invariants of experimental channel regression."""

import numpy as np
import pytest
from scipy.linalg import hadamard
import xarray as xr

from openMWR.omb import omb_statistics, unexplained_omb


def test_moments_covariance_and_missing_samples_preserve_station_and_angle():
    values = xr.DataArray([[1., -1., 5.], [2., -2., 5.], [3., -3., 5.], [4., -4., np.inf]],
                          dims=('time', 'frq'), coords={'frq': [22., 23., 51.]})
    values = values.expand_dims(station=['a', 'b'], ang=[90., 30.])
    result = omb_statistics(values)
    assert result.sizes == {'station': 2, 'ang': 2, 'frq': 3, 'frq2': 3}
    selected = result.sel(station='b', ang=30)
    np.testing.assert_array_equal(selected['count'], [4, 4, 3])
    np.testing.assert_allclose(selected['mean'], [2.5, -2.5, 5.])
    np.testing.assert_allclose(selected['std'], [np.sqrt(1.25), np.sqrt(1.25), 0.])
    np.testing.assert_allclose(selected.rmse ** 2, selected['mean'] ** 2 + selected['std'] ** 2)
    assert selected.covariance_count == 3
    np.testing.assert_allclose(selected.covariance, [[2/3, -2/3, 0], [-2/3, 2/3, 0], [0, 0, 0]])
    assert selected.correlation.sel(frq=22, frq2=23) == pytest.approx(-1.)
    assert selected.correlation.sel(frq=51).isnull().all()
    assert selected['std'].attrs['units'] == 'K'
    assert selected.covariance.attrs['units'] == 'K2'
    unavailable = omb_statistics(values.isel(time=[0]).where(values.frq != 51))
    assert unavailable['std'].isnull().all() and unavailable.covariance.isnull().all()
    assert unavailable['count'].sel(frq=51).max() == 0


def test_regression_has_known_physical_scatter_and_is_offset_scale_and_order_invariant():
    # Orthogonal, zero-mean signals give cov(y,x)=2, var(y)=4.25,
    # var(x)=2, hence residual variance 4.25 - 4/2 = 2.25 K².
    h = hadamard(8)
    values = np.column_stack([2*h[:, 1] + .5*h[:, 2], h[:, 1] + h[:, 3], np.full(8, 5.)])
    difference = xr.DataArray(values, dims=('time', 'frq'), coords={'frq': [22., 23., 51.]})
    bands = [[20., 30.], [50., 60.]]
    residual, count = unexplained_omb(difference, frequency_bands=bands)
    assert residual.std('time').sel(frq=22) == pytest.approx(1.5)
    assert residual.std('time').sel(frq=51) == 0
    np.testing.assert_array_equal(count, [8, 8, 8])
    shifted = difference + xr.DataArray([20., -7., 3.], dims='frq')
    offset, _ = unexplained_omb(shifted, frequency_bands=bands)
    scaled, _ = unexplained_omb(10*difference, frequency_bands=bands)
    reordered, _ = unexplained_omb(difference.isel(frq=[2, 0, 1]), frequency_bands=bands)
    np.testing.assert_allclose(offset, residual, atol=1e-12)
    np.testing.assert_allclose(scaled, 10*residual, atol=1e-12)
    xr.testing.assert_allclose(reordered.sel(frq=difference.frq), residual)
    assert residual.attrs['units'] == 'K'

    # A distant same-band predictor could explain channel 23 exactly, but its
    # immediate neighbors cannot. Identical signals across the gap must also
    # remain unexplained rather than becoming predictors for one another.
    channels = xr.DataArray(np.column_stack([h[:, 1], h[:, 2], h[:, 3],
                                             h[:, 1] + h[:, 2], h[:, 1] + h[:, 2], h[:, 3]]),
                            dims=('time', 'frq'), coords={'frq': [22., 23., 24., 25., 51., 52.]})
    neighbors_only, _ = unexplained_omb(channels, frequency_bands=bands)
    np.testing.assert_allclose(neighbors_only.std('time').sel(frq=[23., 25., 51.]), [1., np.sqrt(2), np.sqrt(2)])


def test_regression_handles_missing_constant_and_insufficient_samples_without_fabricating_noise():
    difference = xr.DataArray(np.column_stack([np.arange(6.), np.zeros(6), np.ones(6)]),
                              dims=('time', 'frq'), coords={'frq': [22., 23., 24.]})
    difference[0, 0] = np.nan
    difference[1, 1] = np.inf
    residual, count = unexplained_omb(difference.expand_dims(ang=[90., 30.]), frequency_bands=[[20., 30.]])
    assert residual.sel(ang=30, frq=22).count() == 4
    assert count.sel(ang=30, frq=22) == 4
    np.testing.assert_allclose(residual.sel(frq=24).fillna(0), 0)
    assert np.isfinite(residual.sel(frq=22).isel(time=slice(2, None))).all()
    short = difference.isel(time=[2, 3], frq=[0, 1])
    insufficient, counts = unexplained_omb(short, frequency_bands=[[20., 30.]])
    # A constant predictor leaves one degree of freedom for the varying target.
    assert counts.sel(frq=22) == 2
    linear = short.copy()
    linear.loc[{'frq': 23}] = short.sel(frq=22)
    insufficient, counts = unexplained_omb(linear, frequency_bands=[[20., 30.]])
    assert insufficient.isnull().all() and (counts == 0).all()
    with pytest.raises(ValueError, match='cover every'):
        unexplained_omb(difference, frequency_bands=[[22., 23.]])
