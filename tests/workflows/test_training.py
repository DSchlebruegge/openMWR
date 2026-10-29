"""Fixed training, Python/command parity and checkpoint-owned settings."""


import numpy as np
import pytest
import torch
import tomllib
import tomli_w
import xarray as xr

from openMWR.evaluation import EvaluationRecipe, evaluate
from openMWR.models import load_model
from openMWR.train import TrainingWorkflow
from openMWR.predict import PredictionWorkflow, run_model


from tests.fixtures.retrievals import retrieval_project
from tests.fixtures.cli import run_cli


@pytest.mark.parametrize(('kind', 'outputs'), [
    ('nn', ['T']), ('nn', ['rh', 'T']), ('nn', ['lwc']),
    ('mlr', ['lwp']), ('mlr', ['iwv', 'T']),
])
def test_selected_outputs_control_training_prediction_and_evaluation(tmp_path, kind, outputs):
    recipe, data = retrieval_project(tmp_path, kind=kind, search=False)
    settings = tomllib.loads(recipe.read_text())
    settings['model']['output_vars'] = outputs
    profiles = bool(set(outputs) & {'T', 'rh', 'lwc'})
    if not profiles:
        settings['model'].pop('heights')
    if kind == 'nn':
        settings['model']['loss_weights'] = {name: 1. for name in outputs}
    settings['optimization']['score_weights'] = {outputs[0]: 1.}
    recipe.write_text(tomli_w.dumps(settings))
    # Missing unselected targets must not exclude training samples or be required.
    for path in (tmp_path / 'data/sites/test/training_variant').glob('*.nc'):
        xr.load_dataset(path).drop_vars([name for name in ('T', 'rh', 'lwc', 'lwp', 'iwv')
                                       if name not in outputs]).to_netcdf(path)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert workflow.model.output_vars == outputs
    model = workflow.train()
    restored = load_model('example', 'test', tmp_path)
    assert restored.output_vars == outputs
    assert set(restored.nor.data_vars) == set(outputs) | {'TB'}
    prediction = run_model(restored, data, device='cpu', include_input_vars=False)
    assert list(prediction.data_vars) == outputs
    assert np.isfinite(prediction.to_array()).all()
    if not profiles:
        assert 'height' not in prediction.dims
    daily = data.sel(time=slice('2024-01-10', '2024-01-10')).sel(ang=90., drop=True)
    result = PredictionWorkflow(tmp_path, 'test', ['example'], smooth=False).predict(
        '2024-01-10', observations=daily)
    expected = set(outputs)
    if 'T' in outputs:
        expected.add('T_grad')
    if {'T', 'rh'} <= set(outputs):
        expected.update(('ah', 'TD', 'iwv_from_ah'))
    if 'lwc' in outputs:
        expected.add('lwp_from_lwc')
    assert set(result.zenith.data_vars) == expected | {'TB'}
    evaluation = EvaluationRecipe.load(tmp_path, 'test', 'example')
    for section in ('observations', 'test'):
        evaluation.values[section]['variables'] = outputs
        if not profiles:
            evaluation.values[section].pop('height_range')
    metrics = evaluate(evaluation)
    assert all(set(case.metrics['errors'].data_vars) == set(outputs) for case in metrics.evaluations)
    # Scoring unavailable variables must fail with the model's name.
    evaluation.values['test']['variables'] = ['iwv'] if 'iwv' not in outputs else ['lwp']
    evaluation.values['test'].pop('height_range', None)
    with pytest.raises(ValueError, match='example does not predict requested variables'):
        evaluate(evaluation)
    if outputs == ['rh', 'T']:
        settings['training']['lr_height'] = {'low': 0.001, 'high': 0.002}
        recipe.write_text(tomli_w.dumps(settings))
        study = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(run_mode='fresh')
        assert len(study.trials) == 1
        assert load_model('example', 'test', tmp_path).output_vars == outputs


def command(retrieval_project, action, *extra, check=True):
    result = run_cli(action, "test", "example", "-p", str(retrieval_project), *extra)
    if check:
        result.check_returncode()
    return result


def test_auto_device_is_saved_with_fixed_model_without_fitting(tmp_path, monkeypatch, caplog):
    recipe, _ = retrieval_project(tmp_path, search=False)
    with caplog.at_level('INFO'):
        workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert workflow.device == 'cpu'
    assert 'Using device cpu (CPU), requested auto.' in caplog.text
    assert TrainingWorkflow(workflow.model, 'test', project_dir=tmp_path).device == 'cpu'
    # Exercise the run/checkpoint metadata with a fixed model, without fitting.
    monkeypatch.setattr(workflow, '_calc_linear_regression', lambda model, *args: model)
    workflow.train()
    metadata = load_model('example', 'test', tmp_path).training_metadata
    assert metadata['device'] == 'cpu'
    assert metadata['device_name'] == 'CPU'
    assert 'Training device: cpu (CPU).' in (recipe.parent / 'training.log').read_text()
    analysis = evaluate(EvaluationRecipe.load(tmp_path, 'test', 'example'))
    assert analysis.metadata['device'] == 'cpu'
    assert analysis.metadata['device_name'] == 'CPU'


def test_python_and_command_training_agree_fit_heldout_data_and_save_settings(tmp_path):
    recipe, data = retrieval_project(tmp_path, search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    workflow.train()
    python_model = load_model('example', 'test', tmp_path)
    repeated = command(tmp_path, 'train', check=False)
    assert repeated.returncode == 1
    assert 'Choose --fresh' in repeated.stderr
    continued = command(tmp_path, 'train', '--continue', check=False)
    assert continued.returncode == 1
    assert 'Continuing is available only for optimization' in continued.stderr
    result = command(tmp_path, 'train', '-f')
    cli_model = load_model('example', 'test', tmp_path)
    for key, values in python_model.state_dict().items():
        torch.testing.assert_close(values, cli_model.state_dict()[key])
    assert cli_model.hyper_params['training_dataset'] == 'training_variant'
    assert 'lr_low' not in cli_model.hyper_params
    assert 'lr_low' not in cli_model.training_metadata['effective']['hyper_params']
    assert cli_model.training_metadata['software']['packages']['openMWR']
    assert (recipe.parent / 'training.log').stat().st_size > 0
    assert not (tmp_path / 'sites/test/retrieval/example/optimization.db').exists()
    evaluation = evaluate(EvaluationRecipe.load(tmp_path, 'test', 'example'))
    errors = evaluation.evaluations[1].metrics['errors']
    assert errors.T.sel(error='RMSE').item() < 1e-4
    assert errors.rh.sel(error='RMSE').item() < 1e-4
    recipe.write_text(recipe.read_text().replace('kind = "mlr"', 'kind = "nn"'))
    loaded = load_model('example', 'test', tmp_path)
    assert loaded.is_MLR
    torch.testing.assert_close(loaded.stack.weight, cli_model.stack.weight)


@pytest.mark.parametrize(('kind', 'mode'), [('nn', 'zenith'), ('nn', 'bls'), ('spectral', 'zenith')])
def test_one_epoch_models_train_and_evaluate(kind, mode, tmp_path):
    recipe, data = retrieval_project(tmp_path, kind, mode, search=False)
    if kind == 'nn':
        import tomllib
        import tomli_w

        settings = tomllib.loads(recipe.read_text())
        settings['model']['loss_weights'] = {'T': [1., .3], 'rh': .7, 'lwc': .2, 'lwp': 1., 'iwv': 1.}
        recipe.write_text(tomli_w.dumps(settings))
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    initial = {key: value.clone() for key, value in workflow.model.state_dict().items()}
    model = workflow.train()
    assert any(not torch.equal(initial[key], value) for key, value in model.state_dict().items())
    assert torch.isfinite(next(model.parameters())).all()
    assert (recipe.parent / 'training_progress.pdf').is_file()
    if kind == 'nn':
        restored = load_model('example', 'test', tmp_path)
        np.testing.assert_array_equal(restored.loss_weights.T, [1., .3])
    analysis = evaluate(EvaluationRecipe.load(tmp_path, 'test', 'example'))
    assert len(analysis.evaluations) == 2
    assert analysis.evaluations[0].samples['time_coordinates'] == 3
    assert analysis.evaluations[1].samples['time_coordinates'] == 6


def test_python_overrides_are_saved_and_prediction_needs_no_recipe(tmp_path):
    recipe, data = retrieval_project(tmp_path, suffix='', search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    workflow.change_hyper_params(seed=123)
    workflow.train()
    path = recipe.parent / 'model.pth'
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint['training_metadata']['effective']['hyper_params']['seed'] == 123
    assert checkpoint['training_metadata']['setup']['fitting'] == 'least squares with expected input-noise penalty'
    checkpoint.pop('training_metadata')
    torch.save(checkpoint, path)
    recipe.unlink()
    assert load_model('example', 'test', tmp_path).training_metadata is None
    analysis = evaluate(EvaluationRecipe.load(tmp_path, 'test', 'example'))
    assert len(analysis.evaluations) == 2


def test_manual_python_workflow_still_uses_the_same_training(tmp_path):
    recipe, data = retrieval_project(tmp_path, search=False)
    configured = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    configured.train()
    before = load_model('example', 'test', tmp_path)
    model = type(configured.model)(**configured.model.structure)
    workflow = TrainingWorkflow(model, 'test', 'cpu', tmp_path)
    workflow.load_training_data(['03715'], dataset='training_variant')
    workflow.add_standard_training_hyper_params()
    workflow.add_input_noise_params(**{f'TB_{f}': 0 for f in model.freqs})
    checkpoint = (recipe.parent / 'model.pth').read_bytes()
    with pytest.raises(ValueError, match='Results already exist'):
        workflow.train()
    assert (recipe.parent / 'model.pth').read_bytes() == checkpoint
    workflow.train(run_mode='fresh')
    for key, values in before.state_dict().items():
        torch.testing.assert_close(values, workflow.model.state_dict()[key])
    assert 'recipe' not in load_model('example', 'test', tmp_path).training_metadata


def test_training_seed_controls_initialization_and_training_noise(tmp_path):
    recipe, _ = retrieval_project(tmp_path, kind='nn', search=False)
    recipe.write_text(recipe.read_text().replace('TB = 0.0', 'TB = 0.15'))
    first = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').train()
    weights = {key: value.clone() for key, value in first.state_dict().items()}
    repeated = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').train(run_mode='fresh')
    for key, value in weights.items():
        torch.testing.assert_close(value, repeated.state_dict()[key], rtol=0, atol=0)
    recipe.write_text(recipe.read_text().replace('seed = 42', 'seed = 43'))
    changed = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').train(run_mode='fresh')
    assert any(not torch.equal(value, changed.state_dict()[key]) for key, value in weights.items())
