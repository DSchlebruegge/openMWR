"""Optimization searches, safe continuation, reproducibility and result isolation."""


import numpy as np
import optuna
import pytest
import xarray as xr

from openMWR.models import load_model
from openMWR.recipes import RetrievalRecipe
from openMWR.train import TrainingWorkflow


from tests.fixtures.retrievals import retrieval_project, training_workflow


def test_optimization_continues_only_matching_experiments(tmp_path):
    recipe, data = retrieval_project(tmp_path)
    workflow = training_workflow(tmp_path)
    with pytest.raises(ValueError, match='No optimization study exists'):
        workflow.optimize(n_trials=1, run_mode='continue')
    study = workflow.optimize(n_trials=1)
    assert len(study.trials) == 1
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'optimize'
    before = (recipe.parent / 'model.pth').read_bytes()
    with pytest.raises(ValueError, match='Results already exist'):
        training_workflow(tmp_path).optimize(n_trials=1)
    assert (recipe.parent / 'model.pth').read_bytes() == before
    recipe.write_text(recipe.read_text().replace('n_trials = 1', 'n_trials = 2'))
    # Matching settings can resume through either setup path.
    continued = training_workflow(tmp_path, manual=True).optimize(n_trials=1, run_mode='continue')
    assert len(continued.trials) == 2
    changed_bias = training_workflow(tmp_path)
    changed_bias.change_hyper_params(mean_bias=[0.1, 0.1])
    with pytest.raises(ValueError, match='different or unrecorded experiment'):
        changed_bias.optimize(n_trials=1, run_mode='continue')
    recipe.write_text(recipe.read_text().replace('high = 0.002', 'high = 0.003'))
    with pytest.raises(ValueError, match='different or unrecorded experiment'):
        training_workflow(tmp_path).optimize(n_trials=1, run_mode='continue')
    recipe.write_text(recipe.read_text().replace('high = 0.003', 'high = 0.002'))
    continued.set_user_attr('openMWR_experiment', None)
    with pytest.raises(ValueError, match='different or unrecorded experiment'):
        training_workflow(tmp_path).optimize(n_trials=1, run_mode='continue')
    assert len(optuna.load_study(study_name='example', storage=f'sqlite:///{tmp_path}/sites/test/retrieval/example/optimization.db').trials) == 2


@pytest.mark.parametrize('replacement', ['deleted', 'replaced'])
def test_resuming_requires_the_best_study_checkpoint(replacement, tmp_path):
    recipe, data = retrieval_project(tmp_path)
    training_workflow(tmp_path).optimize(n_trials=1)
    if replacement == 'deleted':
        (recipe.parent / 'model.pth').unlink()
    else:
        model = load_model('example', 'test', tmp_path)
        model.save('test', tmp_path, model.hyper_params)
    with pytest.raises(ValueError, match='best study checkpoint is missing or has been replaced'):
        training_workflow(tmp_path).optimize(n_trials=1, run_mode='continue')


@pytest.mark.parametrize(('kind', 'mode'), [('nn', 'bls'), ('spectral', 'zenith')])
def test_optimization_runs_for_scan_and_spectral_models(kind, mode, tmp_path, monkeypatch):
    retrieval_project(tmp_path, kind, mode)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    filename = 'bls.nc' if mode == 'bls' else 'zenith.nc'
    assert workflow.validate_optimization() is workflow
    # A separate validation reads current inputs rather than retaining old data.
    analysis = tmp_path / f'data/sites/test/analysis_radiosonde/{filename}'
    temporary = analysis.with_suffix('.disabled')
    analysis.rename(temporary)
    with pytest.raises(FileNotFoundError, match='Missing optimization analysis dataset'):
        workflow.validate_optimization()
    temporary.rename(analysis)
    study = workflow.optimize()
    assert np.isfinite(study.best_value)
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'optimize'


def test_noise_search_uses_fixed_channels_and_explicit_ranges(tmp_path):
    recipe, data = retrieval_project(tmp_path, 'nn', 'bls', search=False)
    tb = '{ low = 0.1, high = 0.2 }'
    settings = recipe.read_text().replace('TB_angle_factor = 3.0', '').replace('TB = 0.0', f'''TB = {tb}
"TB_22.24" = 0.35
TB_angle_factor = {{ low = 1.5, high = 3.0 }}''')
    recipe.write_text(settings)
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize()
    study = optuna.load_study(study_name='example', storage=f'sqlite:///{tmp_path}/sites/test/retrieval/example/optimization.db')
    assert 'input_noise_TB_22.24' not in study.best_params
    assert load_model('example', 'test', tmp_path).hyper_params['input_noise_TB_22.24'] == 0.35
    low, high = 0.1, 0.2
    assert low <= study.best_params['input_noise_TB_31.4'] <= high
    assert 1.5 <= study.best_params['input_noise_TB_angle_factor'] <= 3.0
    assert np.isfinite(study.best_value)
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'optimize'
    # Python callers can still explicitly select one run and override its noise.
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    with pytest.raises(ValueError, match='needs fixed parameter values'):
        workflow.train()
    workflow.add_input_noise_params(TB=0.0, TB_angle_factor=3.0)
    workflow.train(run_mode='fresh')
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'train'


def test_training_search_supports_integer_choices_log_ranges_and_overrides(tmp_path):
    recipe, data = retrieval_project(tmp_path, 'nn')
    settings = recipe.read_text().replace('batch_size = 4', 'batch_size = { choices = [2, 4] }')
    settings = settings.replace('epoch_max_lr = 50', 'epoch_max_lr = { low = 1, high = 3, step = 1 }')
    settings = settings.replace('weight_decay = 1e-5', 'weight_decay = { low = 1e-8, high = 1e-4, log = true }')
    recipe.write_text(settings)
    custom = tmp_path / 'data/sites/test/custom_analysis/zenith.nc'
    custom.parent.mkdir()
    (tmp_path / 'data/sites/test/analysis_radiosonde/zenith.nc').rename(custom)
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(
        n_trials=2, data_source='radiosonde', dataset='custom_analysis',
        start_date='2024-01-11', end_date='2024-01-12')
    study = optuna.load_study(study_name='example', storage=f'sqlite:///{tmp_path}/sites/test/retrieval/example/optimization.db')
    assert len(study.trials) == 2
    assert study.user_attrs['openMWR_experiment']['optimization']['start_date'] == '2024-01-11'
    assert study.user_attrs['openMWR_experiment']['optimization']['end_date'] == '2024-01-12'
    assert 'lr_height' in study.best_params
    assert study.best_params['batch_size'] in [2, 4]
    assert type(study.best_params['epoch_max_lr']) is int
    assert 1e-8 <= study.best_params['weight_decay'] <= 1e-4
    assert study.trials[0].distributions['weight_decay'].log
    assert study.trials[0].distributions['weight_decay'].low == 1e-8
    assert study.trials[0].distributions['weight_decay'].high == 1e-4
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'optimize'


def test_fresh_and_continue_affect_only_the_selected_retrieval(tmp_path):
    recipe, data = retrieval_project(tmp_path)
    storage = f'sqlite:///{tmp_path}/sites/test/retrieval/example/optimization.db'
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize()
    other_path = recipe.parent.parent / 'other/model.pth'
    other_path.parent.mkdir()
    other_storage = f'sqlite:///{other_path.with_name("optimization.db")}'
    other = optuna.create_study(study_name='other', storage=other_storage)
    other.add_trial(optuna.trial.create_trial(value=12.0))
    other_database = other_path.with_name('optimization.db').read_bytes()
    other_path.write_bytes(b'keep the other retrieval')
    with pytest.raises(ValueError, match='Results already exist'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize()
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(run_mode='continue')
    assert len(optuna.load_study(study_name='example', storage=storage).trials) == 2
    before = (recipe.parent / 'model.pth').read_bytes()
    analysis = tmp_path / 'data/sites/test/analysis_radiosonde/zenith.nc'
    temporary = analysis.with_suffix('.disabled')
    analysis.rename(temporary)
    try:
        with pytest.raises(FileNotFoundError, match='Missing optimization analysis dataset'):
            TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(run_mode='fresh')
        assert (recipe.parent / 'model.pth').read_bytes() == before
        assert len(optuna.load_study(study_name='example', storage=storage).trials) == 2
    finally:
        temporary.rename(analysis)
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(run_mode='fresh')
    assert len(optuna.load_study(study_name='example', storage=storage).trials) == 1
    assert other_path.with_name('optimization.db').read_bytes() == other_database
    assert optuna.load_study(study_name='other', storage=other_storage).best_value == 12.0
    assert other_path.read_bytes() == b'keep the other retrieval'


def test_seeded_optimization_continuation_matches_uninterrupted_trials(tmp_path):
    recipe, _ = retrieval_project(tmp_path)
    recipe.write_text(recipe.read_text().replace('first = 1, last = 20', 'first = 21, last = 22')
                      .replace('score_weights = { T = 1.0, rh = 0.1 }', 'score_weights = { T = 2.5, rh = 0.0 }'))
    # Cross TPE's ten-trial startup phase as well as its initial random sampling.
    uninterrupted = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(n_trials=12)
    history = [(trial.params, trial.value) for trial in uninterrupted.trials]
    from openMWR.predict import run_model_list
    model = load_model('example', 'test', tmp_path)
    paired = xr.load_dataset(tmp_path / 'data/sites/test/analysis_radiosonde/zenith.nc').sel(
        time=slice('2024-01-21', '2024-01-22'))
    prediction = run_model_list([model], paired.sel(data_source='hatpro', drop=True), 'cpu')
    residual = prediction.T - paired.sel(data_source='radiosonde', drop=True).T
    expected = 2.5 * np.sqrt((residual ** 2).mean(('time', 'station'))).mean().item()
    assert uninterrupted.best_value == pytest.approx(expected)
    for override in ({'seed': 43}, {'days_of_month': {'first': 1, 'last': 20}}, {'score_weights': {'rh': 1.0}}):
        with pytest.raises(ValueError, match='different or unrecorded experiment'):
            TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(n_trials=1, run_mode='continue', **override)

    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(n_trials=1, run_mode='fresh')
    continued = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(n_trials=11, run_mode='continue')
    assert [trial.params for trial in continued.trials] == [params for params, _ in history]
    assert continued.trials[0].params != continued.trials[1].params
    np.testing.assert_allclose([trial.value for trial in continued.trials],
                               [value for _, value in history], rtol=0, atol=0)


def test_bias_loss_requires_and_uses_explicit_coefficient(tmp_path, monkeypatch):
    recipe, _ = retrieval_project(tmp_path, kind='nn', search=False)
    recipe.write_text(recipe.read_text().replace('loss_fn = "MAE"', 'loss_fn = "MAEPlusBiasLoss"'))
    with pytest.raises(ValueError, match='Missing training settings: bias_weight'):
        RetrievalRecipe.load(tmp_path, 'test', 'example')
    recipe.write_text(recipe.read_text().replace('[training]', '[training]\nbias_weight = { low = 0.1, high = 0.3 }'))
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert workflow.requires_optimization
    study = workflow.optimize(n_trials=1)
    assert 0.1 <= study.best_params['bias_weight'] <= 0.3
