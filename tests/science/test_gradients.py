"""Autograd must agree with finite differences, including batched curved paths."""

import numpy as np
import pytest
import torch

from torchMWRT import AtmProfile, RTModel


def profile():
    return {name: torch.tensor(values, dtype=torch.float64) for name, values in {
        'height': [0., 500., 1000., 2000., 3000.],  # m
        'temperature': [290., 286., 282., 276., 270.],  # K
        'pressure': [1010., 950., 900., 800., 700.],  # hPa
        'rh': [.55, .60, .65, .70, .75],  # fraction
        'lwc': [0., .02, .01, 0., 0.],  # g/m³
        'iwc': [0., 0., .005, .002, 0.],
    }.items()}


@pytest.mark.parametrize(('variable', 'index', 'epsilon'), [
    ('temperature', 2, 1e-3), ('rh', 2, 1e-4), ('lwc', 1, 1e-5), ('iwc', 2, 1e-5),
])
def test_profile_gradients_match_finite_differences(variable, index, epsilon):
    inputs = profile()
    if variable in ('lwc', 'iwc'):
        inputs.pop('iwc' if variable == 'lwc' else 'lwc')
    inputs[variable].requires_grad_()
    model = RTModel(freqs=np.array([22.24, 31.4]), angles=np.array([50., 90.]), absmdl='R17')

    def loss(values):
        return model.execute(AtmProfile(**values), return_intermediate=True)['tbtotal'].mean()

    loss(inputs).backward()
    gradient = inputs[variable].grad[index].item()
    plus = {name: value.detach().clone() for name, value in inputs.items()}
    minus = {name: value.detach().clone() for name, value in inputs.items()}
    plus[variable][index] += epsilon
    minus[variable][index] -= epsilon
    difference = ((loss(plus) - loss(minus)) / (2 * epsilon)).item()
    assert np.isfinite(gradient) and abs(gradient) > 1e-10
    assert gradient == pytest.approx(difference, rel=1e-3, abs=1e-6)


@pytest.mark.parametrize('from_sat', [False, True], ids=['ground', 'satellite'])
def test_curved_time_batch_gradients_match_finite_differences(from_sat):
    inputs = profile()
    inputs.pop('lwc')
    inputs.pop('iwc')
    inputs = {name: torch.stack([value, value + .5]) if name == 'temperature'
              else value for name, value in inputs.items()}
    inputs['temperature'].requires_grad_()
    model = RTModel(freqs=np.array([51.26, 52.28]), angles=np.array([55.]),
                    absmdl='R17', ray_tracing=True, from_sat=from_sat)

    def loss(temperature):
        return model.execute(AtmProfile(**dict(inputs, temperature=temperature)))['tbtotal'].mean()

    loss(inputs['temperature']).backward()
    gradient = inputs['temperature'].grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    plus, minus = inputs['temperature'].detach().clone(), inputs['temperature'].detach().clone()
    plus[1, 2] += 1e-3
    minus[1, 2] -= 1e-3
    expected = ((loss(plus) - loss(minus)) / .002).item()
    assert gradient[1, 2].item() == pytest.approx(expected, rel=1e-3, abs=1e-6)
