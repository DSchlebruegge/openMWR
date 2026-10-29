from types import MappingProxyType

import numpy as np
import pytest
import torch
import xarray as xr

from openMWR.models import Model
from openMWR.train import TrainingWorkflow


@pytest.mark.parametrize(('noise', 'duplicate'), [
    ([0., 0.], False), ([0.2, 0.7], False),
    ([0., 0.], True), ([0., 0.7], True),
])
def test_mlr_expected_noise_solution_and_unpenalized_intercept(tmp_path, noise, duplicate):
    model = Model('linear', heights=[0.], freqs=[22.24, 31.4],
                  is_MLR=True, additional_input_vars=[], output_vars=["T", "rh", "lwc", "lwp", "iwv"]).double()
    workflow = TrainingWorkflow(model, 'test', 'cpu', tmp_path)
    # Physical noise is converted to normalized units by the real noise helper.
    scales = np.array([2., 5.])
    model.nor = xr.Dataset({'TB': ('frq', scales)}, coords={'frq': model.freqs}).expand_dims(stat=['std'])
    parameters = {f'input_noise_TB_{frequency}': sigma * scale
                  for frequency, sigma, scale in zip(model.freqs, noise, scales)}
    x = np.array([[-2., -1.], [-1., -1.], [0., 1.], [1., 0.], [2., 1.], [3., -2.]])
    if duplicate:
        x[:, 1] = x[:, 0]
    x += [20., -5.]
    slopes = np.array([[1., -2., .5, 3., 0.], [2., 1., -.5, 0., 4.]])
    y = x @ slopes + [30., 40., 50., 60., 70.]
    original_x, original_y = x.copy(), y.copy()
    fitted = workflow._calc_linear_regression(model, torch.from_numpy(x), torch.from_numpy(y), parameters)
    weights, intercept = fitted.stack.weight.detach().numpy().T, fitted.stack.bias.detach().numpy()
    centered_x, centered_y = x - x.mean(0), y - y.mean(0)
    if duplicate and not any(noise):
        # Identical columns share the minimum-norm solution equally.
        expected = np.repeat(slopes.mean(0, keepdims=True), 2, axis=0)
    else:
        # Independent check using the penalized covariance equations.
        expected = np.linalg.solve(centered_x.T @ centered_x + len(x) * np.diag(np.square(noise)),
                                   centered_x.T @ centered_y)
    np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(intercept, y.mean(0) - x.mean(0) @ expected, atol=1e-12)
    # Repeating a dataset does not strengthen the per-sample noise penalty.
    workflow._calc_linear_regression(model, torch.from_numpy(np.tile(x, (3, 1))),
                                    torch.from_numpy(np.tile(y, (3, 1))), parameters)
    np.testing.assert_allclose(model.stack.weight.detach().numpy().T, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize(('scheme', 'bls', 'expected'), [
    ({**{name: {'upper_bounds': [10000], 'values': [1., .5]} for name in ('T', 'rh', 'lwc')},
      'lwp': 1., 'iwv': 1.}, False, [1., 1., 1., 1., .5]),
    ({**{name: {'upper_bounds': [2000, 10000], 'values': [1., .5, .25]} for name in ('T', 'rh', 'lwc')},
      'lwp': 1., 'iwv': 1.}, True, [1., 1., .5, .5, .25]),
    ({name: 1. for name in ('T', 'rh', 'lwc', 'lwp', 'iwv')}, False, [1., 1., 1., 1., 1.]),
    ({'T': [1., 2., 3., 4., 5.], 'rh': 2., 'lwc': .1, 'lwp': 1., 'iwv': .5},
     False, [1., 2., 3., 4., 5.]),
    (MappingProxyType({'T': 2., 'rh': 1., 'lwc': 1., 'lwp': 1., 'iwv': 1.}),
     False, [2., 2., 2., 2., 2.]),
])
def test_loss_weight_choices_and_exact_band_boundaries(scheme, bls, expected):
    model = Model('weighted', heights=[0., 2000., 2001., 10000., 10001.],
                  freqs=[22.24], is_bls_model=bls, hidden_layers=[4],
                  additional_input_vars=[], loss_weights=scheme, angles=[90.] if bls else None, output_vars=["T", "rh", "lwc", "lwp", "iwv"])
    np.testing.assert_array_equal(model.loss_weights.T, expected)


@pytest.mark.parametrize('weights', [
    None,
    'unknown',
    {'T': 1.},
    {'T': [1.], 'rh': 1., 'lwc': 1., 'lwp': 1., 'iwv': 1.},
    {'T': -1., 'rh': 1., 'lwc': 1., 'lwp': 1., 'iwv': 1.},
    {'T': 0., 'rh': 0., 'lwc': 0., 'lwp': 0., 'iwv': 0.},
    {'T': {'upper_bounds': [10000], 'values': [1.]}, 'rh': 1., 'lwc': 1., 'lwp': 1., 'iwv': 1.},
    {'T': {'upper_bounds': [10000, 2000], 'values': [1., .5, .25]}, 'rh': 1., 'lwc': 1., 'lwp': 1., 'iwv': 1.},
    {'T': {'upper_bounds': [10000, 10000], 'values': [1., .5, .25]}, 'rh': 1., 'lwc': 1., 'lwp': 1., 'iwv': 1.},
    {'T': {'upper_bounds': [10000], 'values': [1., -1.]}, 'rh': 1., 'lwc': 1., 'lwp': 1., 'iwv': 1.},
    {'T': 1., 'rh': 1., 'lwc': 1., 'lwp': {'upper_bounds': [10000], 'values': [1., .5]}, 'iwv': 1.},
])
def test_invalid_loss_weights_are_rejected(weights):
    with pytest.raises(ValueError, match='loss_weights'):
        Model('invalid', heights=np.array([0., 1000.]), freqs=np.array([22.24]),
              hidden_layers=[4], additional_input_vars=[], loss_weights=weights, output_vars=["T", "rh", "lwc", "lwp", "iwv"])
