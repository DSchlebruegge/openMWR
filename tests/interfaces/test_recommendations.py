"""Python recommendations come from the shipped recipes, without a working project."""

from importlib.resources import files
from shutil import copytree
import tomllib

import numpy as np
import pytest
import tomli_w

from openMWR.models import Model, SpectralConsistencyModel
from openMWR.train import TrainingWorkflow


def workflow(project, family, **overrides):
    frequencies = np.array([22.24, 31.4])
    if family == 'SPC':
        model = SpectralConsistencyModel('example', frequencies, hidden_layer_size=4)
    else:
        model = Model('example', heights=[0., 11000.], freqs=frequencies,
                      is_MLR=family == 'MLR', is_bls_model=family == 'NN_BLS',
                      angles=[90., 30.] if family == 'NN_BLS' else None,
                      hidden_layers=[4], additional_input_vars=overrides.pop('inputs', []),
                      loss_weights=None if family == 'MLR' else
                      {name: 1. for name in ('T', 'rh', 'lwc', 'lwp', 'iwv')}, **overrides)
    return TrainingWorkflow(model, 'test', 'cpu', project)


@pytest.mark.parametrize('family', ['NN', 'MLR', 'NN_BLS', 'SPC'])
def test_python_helpers_observe_template_values_and_overrides(tmp_path, monkeypatch, family):
    package = tmp_path / 'package'
    copytree(files('openMWR').joinpath('templates'), package / 'templates')
    monkeypatch.setattr('openMWR.recipes.files', lambda name: package)
    key = 'seed' if family == 'MLR' else 'epochs'
    for suffix in ('', '_opt'):
        path = package / f'templates/site/retrieval/{family}{suffix}_example/retrieval.toml'
        settings = tomllib.loads(path.read_text())
        if suffix:
            settings['input_noise']['TB'] = {'low': 0.2, 'high': 0.8}
        else:
            settings['training'][key] = 17
            settings['input_noise']['TB'] = 0.25
            settings['input_noise']['TB_22.24'] = 0.15
        path.write_text(tomli_w.dumps(settings))

    project = tmp_path / 'project'
    training = workflow(project, family)
    training.add_standard_training_hyper_params()
    assert training.hyper_params[key] == 17
    assert ('weight_decay' in training.hyper_params) == (family != 'MLR')
    training.add_standard_training_hyper_params(**{key: 3})
    assert training.hyper_params[key] == 3
    training.add_input_noise_params()
    assert training.hyper_params['input_noise_TB_22.24'] == 0.15
    assert training.hyper_params['input_noise_TB_31.4'] == 0.25
    training.add_input_noise_params(TB=0.4, **{'TB_22.24': 0.1})
    assert training.hyper_params['input_noise_TB_22.24'] == 0.1
    assert training.hyper_params['input_noise_TB_31.4'] == 0.4
    spaces = training.default_search_spaces()
    assert spaces['input_noise_TB_22.24'] == training.suggest_float(0.2, 0.8)
    assert ('weight_decay' in spaces) == (family != 'MLR')
    training.add_input_noise_params(optimization=True)
    assert training.hyper_params['input_noise_TB_31.4'] == spaces['input_noise_TB_31.4']
    assert training.requires_optimization
    assert not project.exists()


@pytest.mark.parametrize('optimization', [False, True])
def test_unlisted_inputs_require_explicit_values_or_ranges(tmp_path, optimization):
    training = workflow(tmp_path, 'NN', inputs=['years_since_1970'])
    with pytest.raises(ValueError, match='Missing input_noise settings: years_since_1970'):
        training.add_input_noise_params(optimization=optimization)
    assert training.hyper_params == {}
    value = {'low': 0.5, 'high': 3.0} if optimization else 1.5
    training.add_input_noise_params(optimization=optimization, years_since_1970=value)
    expected = training.suggest_float(0.5, 3.0) if optimization else value
    assert training.hyper_params['input_noise_years_since_1970'] == expected


def test_angle_noise_without_a_recommendation_requires_explicit_values(tmp_path):
    training = workflow(tmp_path, 'NN_BLS', vary_input_noise_with_angle_linear=True)
    with pytest.raises(ValueError, match='Missing input_noise settings: TB_angle_factor'):
        training.add_input_noise_params()
    training.add_input_noise_params(TB_angle_factor=3.0)
    assert training.hyper_params['input_noise_TB_angle_factor'] == 3.0
