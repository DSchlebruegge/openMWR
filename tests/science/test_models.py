from types import MappingProxyType

import numpy as np
import pytest

from openMWR.models import Model


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
                  additional_input_vars=[], loss_weights=scheme, angles=[90.] if bls else None)
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
              hidden_layers=[4], additional_input_vars=[], loss_weights=weights)
