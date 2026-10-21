"""Small real workflows through recipes, Python, module commands and checkpoints."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import optuna
import pytest
import torch
import xarray as xr

from openMWR.evaluation import evaluate_retrieval, validate_evaluation, write_report
from openMWR.models import load_model
from openMWR.site import create_site
from openMWR.train import TrainingWorkflow


def project(tmp_path, kind='mlr', mode='zenith', suffix='variant', search=True):
    create_site('test', tmp_path, gen='G5', mwr_height=550)
    directory = tmp_path / 'sites/test/retrieval/example'
    directory.mkdir()
    settings = f'''schema_version = 1
[model]
kind = "{kind}"
mode = "{mode}"
freqs = [22.24, 31.4]
'''
    if kind != 'spectral':
        settings += 'heights = [0.0, 1000.0]\nadditional_input_vars = []\n'
        if kind == 'nn':
            settings += 'hidden_layers = [4]\ndropout_rate = 0.0\n'
    else:
        settings += 'hidden_layer_size = 4\n'
    if mode == 'bls':
        settings += 'angles = [90.0, 30.0]\nvary_input_noise_with_angle_linear = true\n'
    training = 'repeats_for_linear_regression = 1\n' if kind == 'mlr' else 'epochs = 1\nbatch_size = 4\n'
    variant = f'dataset = "training_{suffix}"\n' if suffix else 'dataset = "training"\n'
    settings += f'''
[data]
sources = ["03715"]
{variant}

[training]
{training}

[input_noise]
method = "standard"
TB = 0.0

[optimization]
n_trials = 1
data_source = "radiosonde"
'''
    if search:
        if kind == 'mlr':
            settings = settings.replace('TB = 0.0', 'TB = { low = 0.001, high = 0.002 }')
        else:
            settings = settings.replace('[training]\n', '[training]\nlr_height = { low = 0.001, high = 0.002 }\n')
    settings += '''
[evaluation]
days_gt_20 = true
station_id = "03715"
'''
    recipe = directory / 'retrieval.toml'
    recipe.write_text(settings)
    import tomli_w
    dataset_name = f'training_{suffix}' if suffix else 'training'
    for name, definition in {
        'radiosondes': {'kind': 'radiosonde', 'stations': ['03715']},
        'profiles_standard': {'kind': 'profiles', 'inputs': {'source': 'radiosondes'}},
        'rt_r20': {'kind': 'simulation', 'inputs': {'profiles': 'profiles_standard'}},
        dataset_name: {'kind': 'split', 'inputs': {'simulation': 'rt_r20'}},
    }.items():
        (tmp_path / f'sites/test/datasets/{name}.toml').write_text(tomli_w.dumps(definition))
    for name in (dataset_name, 'analysis_radiosonde', 'analysis_era5'):
        (tmp_path / f'data/sites/test/{name}').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)
    times = np.array(['2024-01-10', '2024-01-11', '2024-01-12', '2024-01-21', '2024-01-22', '2024-01-23'], dtype='datetime64[ns]')
    data = xr.Dataset(
        {**{name: (('time', 'height'), base + rng.normal(size=(6, 2)))
            for name, base in (('T', 280), ('rh', 50), ('lwc', 1))},
         'lwp': ('time', 20 + rng.normal(size=6)),
         'iwv': ('time', 10 + rng.normal(size=6)),
         'TB': (('time', 'ang', 'frq'), 100 + rng.normal(size=(6, 2, 2)))},
        coords={'time': times, 'height': [0.0, 1000.0], 'ang': [90.0, 30.0], 'frq': [22.24, 31.4]},
    )
    # Common training files can contain scans; zenith workflows select 90 degrees.
    for split in ('training_data', 'validation_data', 'test_data'):
        name = f'{split.removesuffix("_data")}_03715'
        data.to_netcdf(tmp_path / f'data/sites/test/{dataset_name}/{name}.nc')
    observations = xr.concat([data, data.assign(T=data.T + 0.5)], dim='data_source').assign_coords(data_source=['radiosonde', 'hatpro']).expand_dims(station=['03715'])
    observations.sel(ang=90, drop=True).to_netcdf(tmp_path / 'data/sites/test/analysis_radiosonde/zenith.nc')
    observations.to_netcdf(tmp_path / 'data/sites/test/analysis_radiosonde/bls.nc')
    from openMWR.datasets import DatasetWorkflow
    DatasetWorkflow(tmp_path, 'test').record(dataset_name)
    return recipe, data


def command(tmp_path, action, *extra, check=True):
    return subprocess.run([sys.executable, '-m', 'openMWR', action, '--project-dir', str(tmp_path), '--site', 'test', '--retrieval', 'example', *extra], cwd=tmp_path.parent, capture_output=True, text=True, check=check)


def test_python_and_module_training_agree_and_checkpoint_owns_settings(tmp_path):
    recipe, data = project(tmp_path, search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    workflow.train()
    python_model = load_model('example', 'test', tmp_path)
    repeated = command(tmp_path, 'train', check=False)
    assert repeated.returncode == 1
    assert 'Choose --fresh' in repeated.stderr
    continued = command(tmp_path, 'train', '--continue', check=False)
    assert continued.returncode == 1
    assert 'Continuing is available only for optimization' in continued.stderr
    result = command(tmp_path, 'train', '--fresh')
    assert 'Training one model: example' in result.stderr
    cli_model = load_model('example', 'test', tmp_path)
    for key, values in python_model.state_dict().items():
        torch.testing.assert_close(values, cli_model.state_dict()[key])
    assert cli_model.hyper_params['training_dataset'] == 'training_variant'
    assert cli_model.hyper_params['lr_low'] == 1e-7
    assert cli_model.training_metadata['effective']['hyper_params']['lr_low'] == 1e-7
    assert cli_model.training_metadata['software']['packages']['openMWR']
    assert (recipe.parent / 'training.log').read_text().count('Finished train') == 2
    assert not (tmp_path / 'sites/test/retrieval/example/optimization.db').exists()
    invalid = command(tmp_path, 'train', '--n-trials', '2', check=False)
    assert invalid.returncode == 1
    assert 'require input-noise optimization or search ranges' in invalid.stderr
    # Omitting parameters keeps their fixed defaults rather than starting a search.
    recipe.write_text(recipe.read_text().replace('TB = 0.0', ''))
    omitted = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert not omitted.requires_optimization
    assert omitted.hyper_params['input_noise_TB_22.24'] == 0.5
    assert omitted.hyper_params['weight_decay'] == 1e-5
    recipe.write_text(recipe.read_text().replace('kind = "mlr"', 'kind = "nn"'))
    loaded = load_model('example', 'test', tmp_path)
    assert loaded.is_MLR
    torch.testing.assert_close(loaded.stack.weight, cli_model.stack.weight)


@pytest.mark.parametrize(('kind', 'mode'), [('nn', 'zenith'), ('nn', 'bls'), ('spectral', 'zenith')])
def test_one_epoch_models_train_and_evaluate(kind, mode, tmp_path):
    recipe, data = project(tmp_path, kind, mode, search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    model = workflow.train()
    assert torch.isfinite(next(model.parameters())).all()
    assert (recipe.parent / 'training_progress.pdf').is_file()
    analysis = evaluate_retrieval(tmp_path, 'test', 'example')
    assert len(analysis.evaluations) == 2
    assert analysis.evaluations[0].samples['time_coordinates'] == 3
    assert analysis.evaluations[1].samples['time_coordinates'] == 6
    report = write_report(analysis)
    html = report.read_text()
    assert 'data:image/' in html
    assert report.parent == recipe.parent / 'evaluation'
    with pytest.raises(FileExistsError):
        write_report(analysis, report)


def test_python_overrides_are_saved_and_old_checkpoint_needs_no_recipe(tmp_path):
    recipe, data = project(tmp_path, suffix='', search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    workflow.change_hyper_params(repeats_for_linear_regression=2)
    workflow.train()
    path = recipe.parent / 'model.pth'
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint['training_metadata']['effective']['hyper_params']['repeats_for_linear_regression'] == 2
    for key in ('training_metadata', 'model_class'):
        checkpoint.pop(key)
    torch.save(checkpoint, path)
    recipe.unlink()
    assert load_model('example', 'test', tmp_path).training_metadata is None
    analysis = evaluate_retrieval(tmp_path, 'test', 'example', days_gt_20=True)
    assert len(analysis.evaluations) == 2
    with pytest.raises(ValueError, match='days_gt_20 explicitly'):
        validate_evaluation(tmp_path, 'test', 'example')


def test_optimization_continues_only_matching_experiments(tmp_path):
    recipe, data = project(tmp_path)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    study = workflow.optimize()
    assert len(study.trials) == 1
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'optimize'
    recipe.write_text(recipe.read_text().replace('n_trials = 1', 'n_trials = 2'))
    continued = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(n_trials=1, run_mode='continue')
    assert len(continued.trials) == 2
    recipe.write_text(recipe.read_text().replace('high = 0.002', 'high = 0.003'))
    with pytest.raises(ValueError, match='different or unrecorded experiment'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(run_mode='continue')
    assert len(optuna.load_study(study_name='example', storage=f'sqlite:///{tmp_path}/sites/test/retrieval/example/optimization.db').trials) == 2


def test_manual_python_workflow_still_uses_the_same_training(tmp_path):
    recipe, data = project(tmp_path, search=False)
    configured = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    configured.train()
    before = load_model('example', 'test', tmp_path)
    model = type(configured.model)(**configured.model.structure)
    workflow = TrainingWorkflow(model, 'test', 'cpu', tmp_path)
    workflow.load_training_data(['03715'], dataset='training_variant')
    workflow.add_standard_training_hyper_params(epochs=1, batch_size=4, repeats_for_linear_regression=1)
    workflow.add_input_noise_params(**{f'TB_{f}': 0 for f in model.freqs})
    workflow.train()
    for key, values in before.state_dict().items():
        torch.testing.assert_close(values, workflow.model.state_dict()[key])
    assert 'recipe' not in load_model('example', 'test', tmp_path).training_metadata


@pytest.mark.parametrize('replacement', ['deleted', 'trained'])
def test_resuming_requires_the_best_study_checkpoint(replacement, tmp_path):
    recipe, data = project(tmp_path)
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize()
    if replacement == 'deleted':
        (recipe.parent / 'model.pth').unlink()
    else:
        workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
        workflow.add_input_noise_params(**{f'TB_{f}': 0.0 for f in workflow.model.freqs})
        workflow.recipe = None  # A manual script can still replace a checkpoint.
        workflow.train()
    with pytest.raises(ValueError, match='best study checkpoint is missing or has been replaced'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(run_mode='continue')


@pytest.mark.parametrize(('kind', 'mode'), [('nn', 'bls'), ('spectral', 'zenith')])
def test_optimization_runs_for_scan_and_spectral_models(kind, mode, tmp_path, monkeypatch):
    project(tmp_path, kind, mode)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    loaded = []
    load = xr.load_dataset

    def read(path, *args, **kwargs):
        loaded.append(Path(path))
        return load(path, *args, **kwargs)

    monkeypatch.setattr(xr, 'load_dataset', read)
    filename = 'bls.nc' if mode == 'bls' else 'zenith.nc'
    assert workflow.validate_optimization() is workflow
    assert loaded == [tmp_path / f'data/sites/test/analysis_radiosonde/{filename}']
    # A separate validation reads current inputs rather than retaining old data.
    analysis = loaded[0]
    temporary = analysis.with_suffix('.disabled')
    analysis.rename(temporary)
    with pytest.raises(FileNotFoundError, match='Missing optimization analysis dataset'):
        workflow.validate_optimization()
    temporary.rename(analysis)
    loaded.clear()
    study = workflow.optimize()
    assert loaded == [analysis]
    assert np.isfinite(study.best_value)
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'optimize'


def test_recipe_training_requires_only_selected_training_and_validation_files(tmp_path):
    project(tmp_path, search=False)
    (tmp_path / 'data/sites/test/training_variant/test_03715.nc').unlink()
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert workflow.training_data.time.size == 6
    (tmp_path / 'data/sites/test/training_variant/validation_03715.nc').unlink()
    with pytest.raises(FileNotFoundError, match='validation_03715.nc'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')


@pytest.mark.parametrize('space', ['default', 'custom'])
def test_train_command_optimizes_noise_search(space, tmp_path):
    recipe, data = project(tmp_path, 'nn', 'bls', search=False)
    tb = '{ optimize = true }' if space == 'default' else '{ low = 0.1, high = 0.2 }'
    settings = recipe.read_text().replace('TB = 0.0', f'''TB = {tb}
"TB_22.24" = 0.35
TB_angle_factor = {{ low = 1.5, high = 3.0 }}''')
    recipe.write_text(settings)
    result = command(tmp_path, 'train')
    assert 'Optimizing example (trial budget: 1' in result.stderr
    study = optuna.load_study(study_name='example', storage=f'sqlite:///{tmp_path}/sites/test/retrieval/example/optimization.db')
    assert 'input_noise_TB_22.24' not in study.best_params
    assert load_model('example', 'test', tmp_path).hyper_params['input_noise_TB_22.24'] == 0.35
    low, high = (0.0, 1.5) if space == 'default' else (0.1, 0.2)
    assert low <= study.best_params['input_noise_TB_31.4'] <= high
    search_line = next(line for line in result.stderr.splitlines() if 'Search input_noise.TB_31.4:' in line)
    assert f'low={low}' in search_line and f'high={high}' in search_line
    assert 1.5 <= study.best_params['input_noise_TB_angle_factor'] <= 3.0
    assert np.isfinite(study.best_value)
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'optimize'
    # Python callers can still explicitly select one run and override its noise.
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    with pytest.raises(ValueError, match='needs fixed parameter values'):
        workflow.train()
    workflow.add_input_noise_params(**{f'TB_{f}': 0.0 for f in workflow.model.freqs})
    workflow.train(run_mode='fresh')
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'train'


def test_train_command_optimizes_training_search_and_honors_overrides(tmp_path):
    recipe, data = project(tmp_path, 'nn')
    settings = recipe.read_text().replace('batch_size = 4', 'batch_size = { choices = [2, 4] }')
    settings = settings.replace('[training]\n', '''[training]
epoch_max_lr = { low = 1, high = 3, step = 1 }
weight_decay = { optimize = true }
''')
    recipe.write_text(settings)
    result = command(tmp_path, 'train', '--n-trials', '2', '--data-source', 'radiosonde')
    assert 'Optimizing example (trial budget: 2, reference: radiosonde)' in result.stderr
    study = optuna.load_study(study_name='example', storage=f'sqlite:///{tmp_path}/sites/test/retrieval/example/optimization.db')
    assert len(study.trials) == 2
    assert 'lr_height' in study.best_params
    assert study.best_params['batch_size'] in [2, 4]
    assert type(study.best_params['epoch_max_lr']) is int
    assert 1e-8 <= study.best_params['weight_decay'] <= 1e-4
    assert study.trials[0].distributions['weight_decay'].log
    assert study.trials[0].distributions['weight_decay'].low == 1e-8
    assert study.trials[0].distributions['weight_decay'].high == 1e-4
    search_line = next(line for line in result.stderr.splitlines() if 'Search training.weight_decay:' in line)
    assert 'low=1e-08' in search_line and 'high=0.0001' in search_line and 'log=True' in search_line
    assert 'Training sources:' in result.stderr and 'Fitting: MAE' in result.stderr
    assert 'days of month: 1-20' in result.stderr
    assert 'Optimization score (separate from fitting loss)' in result.stderr
    assert load_model('example', 'test', tmp_path).training_metadata['action'] == 'optimize'


def test_evaluate_command_writes_report(tmp_path):
    project(tmp_path, search=False)
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').train()
    result = command(tmp_path, 'evaluate')
    assert Path(result.stdout.strip()).is_file()
    assert 'outside the standard optimization' in Path(result.stdout.strip()).read_text()
    all_days = evaluate_retrieval(tmp_path, 'test', 'example', days_gt_20=False)
    assert all_days.evaluations[0].samples['time_coordinates'] == 6
    assert 'All-days diagnostic' in all_days.evaluations[0].note


def test_fresh_and_continue_affect_only_the_selected_retrieval(tmp_path):
    recipe, data = project(tmp_path)
    storage = f'sqlite:///{tmp_path}/sites/test/retrieval/example/optimization.db'
    command(tmp_path, 'train')
    other_path = recipe.parent.parent / 'other/model.pth'
    other_path.parent.mkdir()
    other_storage = f'sqlite:///{other_path.with_name("optimization.db")}'
    other = optuna.create_study(study_name='other', storage=other_storage)
    other.add_trial(optuna.trial.create_trial(value=12.0))
    other_database = other_path.with_name('optimization.db').read_bytes()
    other_path.write_bytes(b'keep the other retrieval')
    assert command(tmp_path, 'train', check=False).returncode == 1
    command(tmp_path, 'train', '--continue')
    assert len(optuna.load_study(study_name='example', storage=storage).trials) == 2
    before = (recipe.parent / 'model.pth').read_bytes()
    analysis = tmp_path / 'data/sites/test/analysis_radiosonde/zenith.nc'
    temporary = analysis.with_suffix('.disabled')
    analysis.rename(temporary)
    try:
        rejected = command(tmp_path, 'train', '--fresh', check=False)
        assert rejected.returncode == 1
        assert 'Missing optimization analysis dataset' in rejected.stderr
        assert (recipe.parent / 'model.pth').read_bytes() == before
        assert len(optuna.load_study(study_name='example', storage=storage).trials) == 2
    finally:
        temporary.rename(analysis)
    command(tmp_path, 'train', '--fresh')
    assert len(optuna.load_study(study_name='example', storage=storage).trials) == 1
    assert other_path.with_name('optimization.db').read_bytes() == other_database
    assert optuna.load_study(study_name='other', storage=other_storage).best_value == 12.0
    assert other_path.read_bytes() == b'keep the other retrieval'


def test_station_id_is_independent_of_file_order_and_requires_a_choice(tmp_path):
    recipe, data = project(tmp_path, search=False)
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').train()
    expected = evaluate_retrieval(tmp_path, 'test', 'example')
    path = tmp_path / 'data/sites/test/analysis_radiosonde/zenith.nc'
    original = xr.load_dataset(path)
    other = original.assign_coords(station=['02290'])
    other = other.assign(T=other.T + 50)
    xr.concat([other, original], dim='station').to_netcdf(path)
    selected = evaluate_retrieval(tmp_path, 'test', 'example')
    xr.testing.assert_equal(selected.evaluations[0].metrics['errors'], expected.evaluations[0].metrics['errors'])
    assert selected.metadata['filters']['radiosonde_station_id'] == '03715'
    assert selected.evaluations[0].samples['stations'] == ['03715']
    alternate = evaluate_retrieval(tmp_path, 'test', 'example', station_id='02290')
    assert alternate.evaluations[0].metrics['errors'].T.sel(error='RMSE').item() > 40
    invalid = command(tmp_path, 'evaluate', '--station-id', 'missing', check=False)
    assert invalid.returncode == 1
    assert 'available stations' in invalid.stderr
    recipe.write_text(recipe.read_text().replace('station_id = "03715"', ''))
    with pytest.raises(ValueError, match='Select evaluation.station_id'):
        evaluate_retrieval(tmp_path, 'test', 'example')


@pytest.mark.parametrize(('kind', 'section', 'setting'), [
    ('mlr', 'training', 'epochs = 10'),
    ('mlr', 'training', 'weight_decay = { optimize = true }'),
    ('mlr', 'model', 'hidden_layers = [4]'),
    ('mlr', 'model', 'dropout_rate = 0.1'),
    ('nn', 'training', 'repeats_for_linear_regression = 2'),
])
def test_unused_model_settings_are_rejected(kind, section, setting, tmp_path):
    recipe, data = project(tmp_path, kind, search=False)
    recipe.write_text(recipe.read_text().replace(f'[{section}]\n', f'[{section}]\n{setting}\n'))
    with pytest.raises(ValueError, match='is not used by'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert not (recipe.parent / 'model.pth').exists()


def test_manual_search_cannot_optimize_an_unused_mlr_parameter(tmp_path):
    project(tmp_path, search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    workflow.hyper_params['weight_decay'] = workflow.suggest_float(1e-8, 1e-4, log=True)
    with pytest.raises(ValueError, match='weight_decay is not used by mlr'):
        workflow.optimize()


@pytest.mark.parametrize(('before', 'after', 'message'), [
    ('batch_size = 4', 'batch_szie = 4', 'Did you mean'),
    ('batch_size = 4', 'batch_size = 0', 'at least 1'),
    ('TB = 0.0', 'BT = 0.0', 'Unknown input_noise'),
    ('low = 0.001', 'kind = "float", low = 0.001', 'Unknown training.lr_height'),
    ('high = 0.002', 'high = 0.0001', 'must be at least low'),
    ('method = "standard"', '', 'input_noise.method must be'),
    ('method = "standard"', 'method = "typo"', 'input_noise.method must be'),
    ('batch_size = 4', 'batch_size = { low = 1.5, high = 4 }', 'must be a finite integer'),
    ('batch_size = 4', 'batch_size = { choices = [2, 0] }', 'at least 1'),
    ('low = 0.001', 'low = 0.0, log = true', 'requires positive bounds'),
    ('low = 0.001', 'low = 0.001, step = 0.001, log = true', 'invalid step'),
    ('batch_size = 4', 'batch_size = { choices = [] }', 'nonempty list'),
    ('batch_size = 4', 'batch_size = { optimize = true }', 'No default search space for training.batch_size'),
    ('TB = 0.0', 'TB = { optimize = false }', 'optimize must be true'),
    ('TB = 0.0', 'TB = { optimize = true, low = 0.0, high = 1.0 }', 'Unknown input_noise.TB'),
    ('dropout_rate = 0.0', 'dropout_rate = { low = 0.0, high = 0.5 }', 'model.dropout_rate does not support a search'),
    ('[optimization]', '[optimization]\n[optimization.search.weight_decay]\nlow = 1e-8\nhigh = 1e-4', 'optimization.search does not support a search'),
    ('TB = 0.0', 'TB = 0.0\n[input_noise.search.TB]\nlow = 0.1\nhigh = 0.5', 'Unknown input_noise setting'),
])
def test_bad_recipes_fail_before_training(before, after, message, tmp_path):
    recipe, data = project(tmp_path, 'nn')
    recipe.write_text(recipe.read_text().replace(before, after))
    with pytest.raises(ValueError, match=message):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert not (recipe.parent / 'model.pth').exists()


def test_missing_data_and_coordinate_mismatch_are_clear(tmp_path):
    recipe, data = project(tmp_path)
    file = tmp_path / 'data/sites/test/training_variant/validation_03715.nc'
    file.unlink()
    with pytest.raises(FileNotFoundError, match='Missing input dataset'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    data.assign_coords(frq=[23.0, 31.4]).to_netcdf(file)
    with pytest.raises(ValueError, match='coordinates do not match'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    result = command(tmp_path, 'train', check=False)
    assert result.returncode == 1
    assert 'Train failed:' in result.stderr
    assert 'Traceback' not in result.stderr


def test_era5_only_recipe_reuses_bias_correction_without_station_ids(tmp_path):
    recipe, data = project(tmp_path)
    recipe.write_text(recipe.read_text().replace('sources = ["03715"]', 'sources = ["era5"]')
                      + '\n[bias_correction]\ndata_source = "era5"\nfactor = 0.5\n')
    (tmp_path / 'sites/test/datasets/radiosondes.toml').write_text('kind = "era5"\nstart_date = "2024-01-01"\nend_date = "2024-01-31"\n')
    site_file = tmp_path / 'sites/test/site.toml'
    site_file.write_text(site_file.read_text() + '\nlatitude = 48.0\nlongitude = 11.0\n')
    for split in ('training_data', 'validation_data'):
        data.to_netcdf(tmp_path / f'data/sites/test/training_variant/{split.removesuffix("_data")}_era5.nc')
    zenith = data.sel(ang=90, drop=True)
    observations = xr.concat([zenith, zenith.assign(TB=zenith.TB + 3)], dim='data_source').assign_coords(data_source=['era5', 'hatpro'])
    observations.to_netcdf(tmp_path / 'data/sites/test/analysis_era5/zenith.nc')
    from openMWR.datasets import DatasetWorkflow
    DatasetWorkflow(tmp_path, 'test').record('training_variant')
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    np.testing.assert_allclose(workflow.hyper_params['mean_bias'], [1.5, 1.5])
    np.testing.assert_allclose(workflow.training_data.TB.isel(station=0), zenith.TB - 1.5)


def test_recipe_recomputes_omb_noise_and_its_factor(tmp_path):
    recipe, data = project(tmp_path, search=False)
    # Use the normal 14-channel grid so the existing neighbor-based OMB method
    # has channels on both sides of its K/V-band split.
    from openMWR.consts import std_freqs
    import tomli_w
    import tomllib

    rng = np.random.default_rng(17)
    data = data.drop_vars('TB').drop_dims('frq')
    data['TB'] = (('time', 'ang', 'frq'), 100 + rng.normal(size=(6, 2, len(std_freqs))))
    data = data.assign_coords(frq=std_freqs)
    for split in ('training_data', 'validation_data'):
        data.to_netcdf(tmp_path / f'data/sites/test/training_variant/{split.removesuffix("_data")}_03715.nc')
    background = data.sel(ang=90, drop=True)
    measured = background.assign(TB=background.TB + rng.normal(size=(6, len(std_freqs))))
    observations = xr.concat([background, measured], dim='data_source').assign_coords(data_source=['era5', 'hatpro'])
    observations.to_netcdf(tmp_path / 'data/sites/test/analysis_era5/zenith.nc')
    settings = tomllib.loads(recipe.read_text())
    settings['model']['freqs'] = std_freqs.tolist()
    settings.pop('optimization')
    settings['input_noise'] = {'method': 'omb', 'data_source': 'era5', 'factor': 1.0}
    recipe.write_text(tomli_w.dumps(settings))
    first = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    settings['input_noise']['factor'] = 2.0
    recipe.write_text(tomli_w.dumps(settings))
    second = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    for frequency in std_freqs:
        key = f'input_noise_TB_{frequency}'
        assert np.isfinite(first.hyper_params[key])
        assert second.hyper_params[key] == pytest.approx(2 * first.hyper_params[key])
    settings['input_noise'].update({'TB_22.24': 0.25, 'TB_31.4': {'low': 0.1, 'high': 0.4}})
    recipe.write_text(tomli_w.dumps(settings))
    overridden = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert overridden.hyper_params['input_noise_TB_22.24'] == 0.25
    assert overridden.hyper_params['input_noise_TB_31.4'] == overridden.suggest_float(0.1, 0.4)
    assert overridden.hyper_params['input_noise_TB_23.04'] == second.hyper_params['input_noise_TB_23.04']
    assert overridden.requires_optimization
    assert not (recipe.parent / 'model.pth').exists()
