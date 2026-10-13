"""Site settings and saved retrievals share an explicit working-project root."""

import tomllib

import numpy as np
import optuna
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


def test_workflow_reuses_site_optimization_history(tmp_path):
    create_site("test", tmp_path, gen="G5")
    study = optuna.create_study(
        study_name="existing_model",
        storage=f"sqlite:///{tmp_path / 'sites/test/optimization_studies.db'}",
    )
    study.add_trial(optuna.trial.create_trial(
        params={"learning_rate": 0.01},
        distributions={"learning_rate": optuna.distributions.FloatDistribution(0.001, 0.1)},
        value=1.0,
    ))
    model = Model("MLR", heights=np.array([0.0, 100.0]), freqs=np.array([22.24]), is_MLR=True)
    workflow = TrainingWorkflow(model, "test", "cpu", tmp_path)
    workflow.get_hyper_params_from_existing_study("test", "existing_model")
    assert workflow.hyper_params["learning_rate"] == 0.01
