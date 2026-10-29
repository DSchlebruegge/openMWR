import tomllib
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch
import xarray as xr

from openMWR.models import Model, load_model, CHECKPOINT_VERSION
from openMWR.migrate import migrate_project
from openMWR.predict import run_model
from openMWR.site import create_site, get_config_parameter, load_site_config
from openMWR.train import TrainingWorkflow


def test_site_toml_preserves_settings_and_resolves_paths_from_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    external = str(tmp_path / "external output")
    settings = dict(
        gen="G5", mwr_height=550, latitude=48.148,
        rpg_retrieval_exists=False, mwr_measurement_start_date="2024-11-01",
        custom={'frequency_shift': [0.0, -0.01]}, hatpro_data_dir="observations",
        retrieval_output_dir=external, plot_dir=None,
    )
    create_site("test", project, **settings)
    with (project / "sites/test/site.toml").open("rb") as stream:
        stored = tomllib.load(stream)
    assert stored == {key: value for key, value in settings.items() if value is not None}

    monkeypatch.chdir(tmp_path)
    assert get_config_parameter("test", "hatpro_data_dir", project) == str(project / "observations")
    assert get_config_parameter("test", "retrieval_output_dir", project) == external
    assert get_config_parameter("test", ["gen", "plot_dir", "custom"], project) == ("G5", None, {"frequency_shift": [0.0, -0.01]})
    config = load_site_config('test', project)
    assert config.get('plot_dir') is None
    with pytest.raises(ValueError, match='Unknown site setting'):
        config.get('mwr_heigth')
    with pytest.raises(ValueError, match='requested site and project'):
        load_site_config('other', project, config=config)


@pytest.mark.parametrize('setting', [
    'mwr_heigth = 500', 'mwr_height = "500"', 'gen = "G6"',
    'latitude = 91', 'longitude = -181', 'latitude = nan',
    'hatpro_data_dir = ""', 'hatpro_format = "automatic"',
    'hatpro_time_offset_hours = 25', 'rpg_retrieval_exists = "false"',
    'mwr_measurement_start_date = "2024-02-30"',
    '[prediction]\nretrievals = ["../outside"]',
    '[prediction]\nretrievals = ["model", "model"]',
    'custom = "settings"',
])
def test_invalid_site_settings_report_the_file_before_processing(tmp_path, setting):
    directory = create_site('test', tmp_path)
    path = directory / 'site.toml'
    path.write_text(setting)
    with pytest.raises(ValueError, match=r'sites/test/site.toml:'):
        load_site_config('test', tmp_path)
    assert not list((tmp_path / 'data').rglob('*.*'))


def test_named_retrieval_roundtrip_and_replacement(tmp_path):
    model = Model("MLR", heights=np.array([0.0, 100.0]), freqs=np.array([22.24]), is_MLR=True, output_vars=["T", "rh", "lwc", "lwp", "iwv"])
    model.nor = xr.Dataset({"scale": ("height", [1.0, 2.0])}, coords={"height": [0.0, 100.0]})
    inputs = torch.zeros(2, model.n_input)
    model.save("test", tmp_path, {"epochs": np.int64(1)}, training_metadata={'scores': np.array([1., 2.]), 'path': tmp_path})
    assert (tmp_path / "sites/test/retrieval/MLR/model.pth").is_file()
    payload = torch.load(tmp_path / 'sites/test/retrieval/MLR/model.pth', weights_only=True)
    assert payload['format_version'] == CHECKPOINT_VERSION
    assert payload['structure']['heights'] == [0., 100.]
    assert payload['training_metadata'] == {'scores': [1., 2.], 'path': str(tmp_path)}
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
    before = (tmp_path / 'sites/test/retrieval/MLR/model.pth').read_bytes()
    with pytest.raises(ValueError, match='unsupported metadata type'):
        model.save('test', tmp_path, {'unsupported': object()})
    assert (tmp_path / 'sites/test/retrieval/MLR/model.pth').read_bytes() == before


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
    model = Model("MLR", heights=np.array([0.0, 100.0]), freqs=np.array([22.24]), is_MLR=True, output_vars=["T", "rh", "lwc", "lwp", "iwv"])
    workflow = TrainingWorkflow(model, "test", "cpu", tmp_path)
    workflow.get_hyper_params_from_existing_study("reference", "existing_model")
    assert workflow.hyper_params["learning_rate"] == 0.01


@pytest.mark.parametrize('name', ['zenith', 'bls', 'mlr', 'spectral'])
def test_version_01_checkpoint_migration_preserves_predictions(tmp_path, name):
    reference = torch.load(Path(__file__).parents[1] / 'data/checkpoint_01.pt',
                           map_location='cpu', weights_only=False)
    case = next(case for case in reference['cases'] if case['name'] == name)
    create_site('test', tmp_path, gen='G5')
    current = tmp_path / 'sites/test/retrieval' / name / 'model.pth'
    current.parent.mkdir(parents=True)
    current.write_bytes(case['checkpoint'])
    with pytest.raises(ValueError, match='openMWR migrate'):
        load_model(name, 'test', tmp_path)
    migrate_project(tmp_path)

    model = load_model(name, 'test', tmp_path)
    inputs = xr.Dataset.from_dict(case['inputs'])
    expected = xr.Dataset.from_dict(case['prediction'])
    actual = run_model(model, inputs, 'cpu', include_input_vars=False)
    xr.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert current.with_name('model.pth.bak').read_bytes() == case['checkpoint']
    converted = torch.load(current, map_location='cpu', weights_only=True)
    original = torch.load(current.with_name('model.pth.bak'), map_location='cpu', weights_only=False)
    for key, value in original['model_state_dict'].items():
        assert torch.equal(value, converted['model_state_dict'][key])
    xr.testing.assert_identical(xr.Dataset.from_dict(converted['normalization']),
                               xr.Dataset.from_dict(original['normalization']))
    assert 'file_extension' not in converted['hyper_params']
    assert not current.with_name('retrieval.toml').exists()
    assert model.training_dataset == 'training_r17'
    if case['loss_weights'] is not None:
        xr.testing.assert_identical(model.loss_weights, xr.Dataset.from_dict(case['loss_weights']))
    if name == 'bls':
        np.testing.assert_array_equal(model.angles, model.nor.ang)
