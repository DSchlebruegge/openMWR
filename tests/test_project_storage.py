"""Site settings and saved retrievals share an explicit working-project root."""

import tomllib
from types import MappingProxyType

import numpy as np
import optuna
import pytest
import torch
import xarray as xr

from openMWR.models import Model
from openMWR.site import create_site, get_config_parameter
from openMWR.train import TrainingWorkflow


def test_site_toml_preserves_settings_and_resolves_paths_from_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    external = str(tmp_path / "external output")
    settings = dict(
        gen="G5", mwr_height=550, latitude=48.148,
        rpg_retrieval_exists=False, mwr_measurement_start_date="2024-11-01",
        frequency_shift=[0.0, -0.01], hatpro_data_dir="observations",
        retrieval_output_dir=external, plot_dir=None,
    )
    create_site("test", project, **settings)
    with (project / "sites/test/site.toml").open("rb") as stream:
        stored = tomllib.load(stream)
    assert stored == {key: value for key, value in settings.items() if value is not None}

    monkeypatch.chdir(tmp_path)
    assert get_config_parameter("test", "hatpro_data_dir", project) == str(project / "observations")
    assert get_config_parameter("test", "retrieval_output_dir", project) == external
    assert get_config_parameter("test", ["gen", "plot_dir", "frequency_shift"], project) == ("G5", None, [0.0, -0.01])


def test_named_retrieval_roundtrip_and_replacement(tmp_path):
    model = Model("MLR", heights=np.array([0.0, 100.0]), freqs=np.array([22.24]), is_MLR=True)
    model.nor = xr.Dataset({"scale": ("height", [1.0, 2.0])}, coords={"height": [0.0, 100.0]})
    inputs = torch.zeros(2, model.n_input)
    model.save("test", tmp_path, {"epochs": 1})
    assert (tmp_path / "sites/test/retrieval/MLR/model.pth").is_file()
    loaded = Model.load("MLR", "test", tmp_path)
    torch.testing.assert_close(loaded(inputs), model(inputs))
    xr.testing.assert_identical(loaded.nor, model.nor)
    assert Model.load_hyper_params(site="test", name="MLR", project_dir=tmp_path) == {"epochs": 1}

    with torch.no_grad():
        model.stack.bias.add_(1)
    model.save("test", tmp_path, {"epochs": 2})
    replacement = Model.load("MLR", "test", tmp_path)
    torch.testing.assert_close(replacement(inputs), model(inputs))
    assert replacement.hyper_params == {"epochs": 2}


def test_workflow_loads_retrieval_optimization_history(tmp_path):
    create_site("test", tmp_path, gen="G5")
    directory = tmp_path / 'sites/reference/retrieval/existing_model'
    directory.mkdir(parents=True)
    study = optuna.create_study(
        study_name="existing_model",
        storage=f"sqlite:///{directory / 'optimization.db'}",
    )
    study.add_trial(optuna.trial.create_trial(
        params={"learning_rate": 0.01},
        distributions={"learning_rate": optuna.distributions.FloatDistribution(0.001, 0.1)},
        value=1.0,
    ))
    model = Model("MLR", heights=np.array([0.0, 100.0]), freqs=np.array([22.24]), is_MLR=True)
    workflow = TrainingWorkflow(model, "test", "cpu", tmp_path)
    workflow.get_hyper_params_from_existing_study("reference", "existing_model")
    assert workflow.hyper_params["learning_rate"] == 0.01


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
def test_loss_weight_choices_and_saved_numeric_weights(tmp_path, scheme, bls, expected):
    model = Model('weighted', heights=[0., 2000., 2001., 10000., 10001.],
                  freqs=[22.24], is_bls_model=bls, hidden_layers=[4],
                  additional_input_vars=[], loss_weights=scheme, angles=[90.] if bls else None)
    np.testing.assert_array_equal(model.loss_weights.T, expected)
    model.nor = xr.Dataset(coords={'ang': model.angles}) if bls else xr.Dataset()
    model.eval()
    inputs = torch.zeros(2, model.n_input)
    before = model(inputs)
    model.save('test', tmp_path, {})
    restored = Model.load('weighted', 'test', tmp_path)
    restored.eval()
    xr.testing.assert_identical(restored.loss_weights, model.loss_weights)
    torch.testing.assert_close(restored(inputs), before)
    assert isinstance(restored.checkpoint_structure()['loss_weights'], dict)


@pytest.mark.parametrize(('bls', 'bounds', 'values'), [
    (False, [10000], [1., .5]),
    (True, [2000, 10000], [1., .5, .25]),
], ids=['zenith', 'bls'])
def test_historical_checkpoint_recovers_weights_and_angles(tmp_path, bls, bounds, values):
    weights = {**{name: {'upper_bounds': bounds, 'values': values} for name in ('T', 'rh', 'lwc')},
               'lwp': 1., 'iwv': 1.}
    model = Model('historical', heights=[0., 2000., 2001., 10000., 10001.],
                  freqs=[22.24], is_bls_model=bls, hidden_layers=[4],
                  additional_input_vars=[], loss_weights=weights, angles=[90.] if bls else None)
    model.nor = xr.Dataset(coords={'ang': model.angles}) if bls else xr.Dataset()
    model.eval()
    inputs = torch.zeros(2, model.n_input)
    before = model(inputs)
    model.save('test', tmp_path, {})

    path = tmp_path / 'sites/test/retrieval/historical/model.pth'
    checkpoint = torch.load(path, weights_only=False)
    checkpoint['structure'].pop('loss_weights')
    checkpoint['structure']['angles'] = None
    torch.save(checkpoint, path)

    restored = Model.load('historical', 'test', tmp_path)
    restored.eval()
    xr.testing.assert_identical(restored.loss_weights, model.loss_weights)
    if bls:
        assert restored.angles == model.angles
    torch.testing.assert_close(restored(inputs), before)


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
