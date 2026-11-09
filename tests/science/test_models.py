from types import MappingProxyType

import numpy as np
import pytest
import torch
import xarray as xr

from openMWR.models import Model, SpectralConsistencyModel
from openMWR.normalization import denormalize, get_input_noise, normalize, normalized_input_noise
from openMWR.train import TrainingWorkflow
from openMWR.predict import run_model


def test_spectral_normalization_applies_once_and_matches_physical_noise():
    model = SpectralConsistencyModel('spectral', [22.24, 31.4])
    model.nor = xr.Dataset({'TB': (('stat', 'frq'), [[100., 200.], [2., 4.]])},
                           coords={'stat': ['mean', 'std'], 'frq': model.freqs})
    data = xr.Dataset({'TB': ('frq', [104., 208.])}, coords={'frq': model.freqs})
    normalized = normalize(data, model)
    np.testing.assert_allclose(normalized.TB, [2., 2.])
    xr.testing.assert_allclose(denormalize(normalized, model), data)
    parameters = {'input_noise_TB_22.24': 2., 'input_noise_TB_31.4': 4.}
    np.testing.assert_allclose(normalized_input_noise(model, parameters), [1., 1.])
    np.testing.assert_allclose((normalize(data + get_input_noise(model, parameters), model) - normalized).TB,
                               [1., 1.])


@pytest.mark.parametrize('variable', ['lwc', 'lwp'])
def test_z_normalized_cloud_water_is_bounded_in_physical_units(variable):
    model = Model('water', [0.] if variable == 'lwc' else None, [22.24], is_MLR=True,
                  additional_input_vars=[], output_vars=[variable],
                  norm_functions={'TB': 'z_norm', variable: 'z_norm'})
    water = (('stat', 'height'), [[10.], [20.]]) if variable == 'lwc' else ('stat', [10., 20.])
    model.nor = xr.Dataset({'TB': (('stat', 'frq'), [[100.], [2.]]), variable: water},
                           coords={'stat': ['mean', 'std'], 'frq': [22.24],
                                   **({'height': [0.]} if variable == 'lwc' else {})})
    with torch.no_grad():
        model.stack.weight.fill_(1.)
        model.stack.bias.zero_()
    data = xr.Dataset({'TB': (('time', 'frq'), [[98.], [100.], [102.]])},
                      coords={'time': np.arange('2024-01-01', '2024-01-04', dtype='datetime64[D]'),
                              'frq': [22.24]})
    prediction = run_model(model, data, 'cpu', include_input_vars=False)
    np.testing.assert_allclose(prediction[variable].values.ravel(), [0., 10., 30.])


def test_input_noise_uses_the_selected_linear_scale_and_rejects_undefined_conversions():
    model = Model('noise', heights=None, freqs=[22.24, 31.4], is_MLR=True,
                  additional_input_vars=['surface_T'], output_vars=['iwv'])
    model.nor = xr.Dataset({'TB': (('stat', 'frq'), [[200., 100.], [2., 0.], [300., 0.]]),
                            'surface_T': ('stat', [280., 2., 282.])},
                           coords={'stat': ['mean', 'std', 'max'], 'frq': model.freqs})
    model.norm_functions['surface_T'] = 'std_norm'
    parameters = {'input_noise_TB_22.24': 3., 'input_noise_TB_31.4': 0., 'input_noise_surface_T': 1.}
    for method, expected in [('z_norm', 1.5), ('std_norm', 1.5), ('max_norm', .01)]:
        model.norm_functions['TB'] = method
        np.testing.assert_allclose(normalized_input_noise(model, parameters),
                                   [expected, 0., .5])
        with pytest.raises(ValueError, match='TB: nonzero input noise requires a finite, nonzero normalization'):
            normalized_input_noise(model, {**parameters, 'input_noise_TB_31.4': 1.})
    model.norm_functions['TB'] = 'log_norm'
    with pytest.raises(ValueError, match='TB: additive input noise is not supported with log_norm'):
        normalized_input_noise(model, parameters)
    np.testing.assert_allclose(normalized_input_noise(
        model, {**parameters, 'input_noise_TB_22.24': 0.}), [0., 0., .5])


def test_scan_noise_matches_normalized_perturbations_in_model_feature_order():
    model = Model('scan', heights=None, freqs=[31.4, 22.24], angles=[30., 90.],
                  is_bls_model=True, is_MLR=True, additional_input_vars=['surface_T'],
                  output_vars=['iwv'], vary_input_noise_with_angle_linear=True)
    # Different scales and deliberately reversed coordinates expose ordering errors.
    model.nor = xr.Dataset({'TB': (('stat', 'ang', 'frq'),
                                   [[[200., 250.], [230., 280.]],
                                    [[2., 4.], [5., 10.]]]),
                            'surface_T': ('stat', [280., 2.])},
                           coords={'stat': ['mean', 'std'], 'frq': [22.24, 31.4], 'ang': [90., 30.]})
    parameters = {'input_noise_TB_22.24': 2., 'input_noise_TB_31.4': 4.,
                  'input_noise_TB_angle_factor': 3., 'input_noise_surface_T': 1.}
    data = xr.Dataset({'TB': (('ang', 'frq'), [[200., 250.], [240., 280.]]), 'surface_T': 280.},
                      coords={'ang': [90., 30.], 'frq': [22.24, 31.4]})
    noise = get_input_noise(model, parameters).sel(ang=data.ang, frq=data.frq)
    difference = normalize(data + noise, model) - normalize(data, model)
    expected = [1.2, 1.2, 1., 1., .5]
    np.testing.assert_allclose(normalized_input_noise(model, parameters), expected)
    actual = np.concatenate([difference[name].sel(**spec.coords).transpose(*spec.dims).values.ravel()
                             for name, spec in model.input.items()])
    np.testing.assert_allclose(actual, expected)
    # The same saved scan statistics also support a zenith-only dataset.
    zenith = normalize(data.sel(ang=90, drop=True), model)
    xr.testing.assert_allclose(zenith, normalize(data, model).sel(ang=90, drop=True))


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
