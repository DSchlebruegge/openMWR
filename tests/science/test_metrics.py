"""Numerical expectations are small enough to calculate by hand."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from openMWR.calc_errors import calculate_errors
from openMWR.predict import denormalize, normalize
from openMWR.train import MAEPlusBiasLoss, MSEPlusBiasLoss


def test_errors_reduce_samples_keep_profiles_and_skip_missing_values():
    # Residuals at the two heights are [1, -1, 3] and [2, -2, missing].
    truth = xr.Dataset({'T': (('time', 'height'), [[10., 10.]] * 3),
                        'TB': ('time', [10., 10., 10.])},
                       coords={'time': [0, 1, 2], 'height': [0., 1000.]})
    prediction = truth.copy(deep=True)
    prediction['T'][:] = [[11., 12.], [9., 8.], [13., np.nan]]
    prediction['TB'][:] = [11., 9., 13.]
    kinds = ['ME', 'MAE', 'MSE', 'RMSE', 'MRE', 'MARE']
    overall, profiles = calculate_errors(prediction.expand_dims(station=['one']), truth, kinds)
    assert set(overall) == set(profiles) == {'T'}
    assert profiles.T.dims == ('error', 'height')
    np.testing.assert_array_equal(profiles.height, [0., 1000.])
    np.testing.assert_allclose(profiles.T, [[1., 0.], [5 / 3, 2.], [11 / 3, 4.],
                                          [np.sqrt(11 / 3), 2.], [.1, 0.], [1 / 6, .2]])
    np.testing.assert_allclose(overall.T, [.6, 1.8, 3.8, np.sqrt(3.8), .06, .18])
    spectral, _ = calculate_errors(prediction, truth, variables=['TB'])
    assert spectral.TB.sel(error='ME').item() == pytest.approx(1.)
    with pytest.raises(ValueError, match='Unknown error type'):
        calculate_errors(prediction, truth, ['invalid'])


def test_normalization_has_known_values_inverses_and_constant_channel_behavior():
    data = xr.Dataset({name: ('time', values) for name, values in {
        'T': [280., 284.], 'surface_p': [900., 1000.], 'lwp': [2., 4.],
        'lwc': [0., np.e - 1], 'TB': [100., 100.], 'rh': [-5., 120.],
    }.items()}, coords={'time': [0, 1]})
    original = data.copy(deep=True)
    names = list(data)
    stats = {'T': [280., 2., 284.], 'surface_p': [0., 100., 1000.],
             'lwp': [0., 1., 4.], 'lwc': [0., 1., 1.],
             'TB': [100., 0., 100.], 'rh': [0., 1., 100.]}
    model = SimpleNamespace(input_vars=['TB', 'surface_p'], output_vars=['T', 'rh', 'lwp', 'lwc'],
        norm_functions=dict(T='z_norm', surface_p='std_norm', lwp='max_norm',
                            lwc='log_norm', TB='z_norm', rh='z_norm'),
        nor=xr.Dataset({name: ('stat', stats[name]) for name in names},
                       coords={'stat': ['mean', 'std', 'max']}))
    scaled = normalize(data, model)
    for name, expected in dict(T=[0., 2.], surface_p=[9., 10.], lwp=[.5, 1.],
                               lwc=[0., 1.], TB=[0., 0.]).items():
        np.testing.assert_allclose(scaled[name], expected)
    restored = denormalize(scaled, model)
    for name in names:
        np.testing.assert_allclose(restored[name], [1., 100.] if name == 'rh' else data[name], atol=1e-12)
    xr.testing.assert_identical(data, original)


@pytest.mark.parametrize('subset', [False, True])
def test_normalization_aligns_reordered_and_subset_feature_statistics(subset):
    coordinates = {'frq': [22.24, 31.4], 'ang': [30., 90.], 'height': [0., 1000.]}
    data = xr.Dataset({'TB': (('frq', 'ang'), np.full((2, 2), 104.)),
                       'T': ('height', [284., 288.])}, coords=coordinates)
    model = SimpleNamespace(input_vars=['TB'], output_vars=['T'],
                            norm_functions={'TB': 'z_norm', 'T': 'z_norm'},
                            nor=xr.Dataset({'TB': (('stat', 'frq', 'ang'),
                                                  np.array([np.full((2, 2), 100.), np.full((2, 2), 2.)])),
                                            'T': (('stat', 'height'), [[280., 284.], [4., 4.]])},
                                           coords={**coordinates, 'stat': ['mean', 'std']}))
    selection = {name: values[-1:] if subset else values[::-1] for name, values in coordinates.items()}
    selected = data.sel(selection)
    normalized = normalize(selected, model)
    np.testing.assert_allclose(normalized.TB, 2.)
    np.testing.assert_allclose(normalized.T, 1.)
    for name, values in selection.items():
        np.testing.assert_array_equal(normalized[name], values)
    xr.testing.assert_allclose(denormalize(normalized, model), selected)
    with pytest.raises(ValueError, match='Normalization statistics do not contain the requested feature coordinates'):
        normalize(data.assign_coords(frq=[22.24, 30.]), model)


def test_bias_penalty_uses_mean_error_and_explicit_coefficient():
    prediction, truth = torch.tensor([[2.], [4.]]), torch.zeros((2, 1))
    assert MAEPlusBiasLoss(.2)(prediction, truth).item() == pytest.approx(3.6)
    assert MSEPlusBiasLoss(.2)(prediction, truth).item() == pytest.approx(10.6)
