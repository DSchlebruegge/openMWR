"""Recipe parsing and resolution reject invalid or ignored settings."""


import numpy as np
import pytest

from openMWR.train import TrainingWorkflow


from tests.fixtures.retrievals import retrieval_project




def test_recipe_training_requires_only_selected_training_and_validation_files(tmp_path):
    retrieval_project(tmp_path, search=False)
    (tmp_path / 'data/sites/test/training_variant/test_03715.nc').unlink()
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert workflow.training_data.time.size == 6
    (tmp_path / 'data/sites/test/training_variant/validation_03715.nc').unlink()
    with pytest.raises(FileNotFoundError, match='validation_03715.nc'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')


@pytest.mark.parametrize(('kind', 'section', 'setting'), [
    ('mlr', 'training', 'epochs = 10'),
    ('mlr', 'training', 'weight_decay = { optimize = true }'),
    ('mlr', 'model', 'hidden_layers = [4]'),
    ('mlr', 'model', 'dropout_rate = 0.1'),
    ('mlr', 'model', 'loss_weights = { T = 1.0, rh = 1.0, lwc = 1.0, lwp = 1.0, iwv = 1.0 }'),
    ('mlr', 'training', 'repeats_for_linear_regression = 2'),
])
def test_unused_model_settings_are_rejected(kind, section, setting, tmp_path):
    recipe, data = retrieval_project(tmp_path, kind, search=False)
    recipe.write_text(recipe.read_text().replace(f'[{section}]\n', f'[{section}]\n{setting}\n'))
    with pytest.raises(ValueError, match='is not used by|Unknown training setting'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert not (recipe.parent / 'model.pth').exists()


def test_manual_search_cannot_optimize_an_unused_mlr_parameter(tmp_path):
    retrieval_project(tmp_path, search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    workflow.hyper_params['weight_decay'] = workflow.suggest_float(1e-8, 1e-4, log=True)
    with pytest.raises(ValueError, match='weight_decay is not used by mlr'):
        workflow.optimize()


@pytest.mark.parametrize(('before', 'after', 'message'), [
    ('batch_size = 4', 'batch_szie = 4', 'Did you mean'),
    ('batch_size = 4', 'batch_size = 0', 'at least 1'),
    ('learning_rate_min = 1e-7', 'learning_rate_min = 0.003', 'learning_rate_min must not exceed'),
    ('learning_rate_min = 1e-7', 'learning_rate_min = { low = 1e-7, high = 0.0015 }', 'learning_rate_min must not exceed'),
    ('learning_rate_min = 1e-7', 'learning_rate_min = { choices = [1e-7, 0.003] }', 'learning_rate_min must not exceed'),
    ('TB = 0.0', 'BT = 0.0', 'Unknown input_noise'),
    ('TB = 0.0', '"TB_31.4" = 0.0', 'input_noise.channel_noise'),
    ('TB = 0.0', 'TB = 0.0\nchannel_noise = {}', 'nonempty table'),
    ('TB = 0.0', 'TB = 0.0\nchannel_noise = { bad = 0.1 }', 'numeric coordinates'),
    ('TB = 0.0', 'TB = 0.0\nchannel_noise = { "31.4" = 0.1, "31.40" = 0.2 }', 'Duplicate'),
    ('TB = 0.0', 'TB = 0.0\nchannel_noise = { "nan" = 0.1 }', 'finite number'),
    ('TB = 0.0', 'TB = 0.0\nchannel_noise = { "50.0" = 0.1 }', 'Unknown input_noise'),
    ('TB = 0.0', '', 'Missing input_noise settings'),
    ('low = 0.001', 'kind = "float", low = 0.001', 'Unknown training.learning_rate_max'),
    ('high = 0.002', 'high = 0.0001', 'must be at least low'),
    ('method = "specified"', '', 'input_noise.method must be'),
    ('method = "specified"', 'method = "typo"', 'input_noise.method must be'),
    ('batch_size = 4', 'batch_size = { low = 1.5, high = 4 }', 'must be a finite integer'),
    ('batch_size = 4', 'batch_size = { choices = [2, 0] }', 'at least 1'),
    ('low = 0.001', 'low = 0.0, log = true', 'requires positive bounds'),
    ('low = 0.001', 'low = 0.001, step = 0.001, log = true', 'invalid step'),
    ('batch_size = 4', 'batch_size = { choices = [] }', 'nonempty list'),
    ('batch_size = 4', 'batch_size = { optimize = true }', 'specify low/high or choices explicitly'),
    ('TB = 0.0', 'TB = { optimize = false }', 'specify low/high or choices explicitly'),
    ('TB = 0.0', 'TB = { optimize = true, low = 0.0, high = 1.0 }', 'specify low/high or choices explicitly'),
    ('weight_decay = 1e-5', '', 'Missing training settings: weight_decay'),
    ('[training]\nseed = 42', '[training]', 'Missing training settings: seed'),
    ('[optimization]\nseed = 42', '[optimization]', 'Missing optimization settings: seed'),
    ('seed = 42', 'seed = -1', 'at least 0'),
    ('first = 1, last = 20', 'first = 21, last = 20', 'first <= last <= 31'),
    ('score_weights = { T = 1.0, rh = 0.1 }', 'score_weights = { T = 0.0 }', 'at least one positive weight'),
    ('hidden_layers = [4]', '', 'Missing model settings: hidden_layers'),
    ('output_vars = ["T", "rh", "lwc", "lwp", "iwv"]', '', 'Missing model settings: output_vars'),
    ('output_vars = ["T", "rh", "lwc", "lwp", "iwv"]', 'output_vars = []', 'output_vars must select'),
    ('output_vars = ["T", "rh", "lwc", "lwp", "iwv"]', 'output_vars = ["T", "T"]', 'must not contain duplicates'),
    ('output_vars = ["T", "rh", "lwc", "lwp", "iwv"]', 'output_vars = ["pressure"]', 'output_vars must select'),
    ('loss_weights = {', '# loss_weights = {', 'Missing model settings: loss_weights'),
    ('normalization = {', '# normalization = {', 'Missing model settings: normalization'),
    ('normalization = {', 'normalization = { TB_IR = "z_norm",', 'Unknown model.normalization'),
    ('TB = "z_norm", T', 'T', 'Missing model.normalization settings: TB'),
    ('frequencies = [22.24, 31.4]', 'frequencies = [22.24, 22.24]', 'must not contain duplicates'),
    ('heights = [0.0, 1000.0]', 'heights = [1000.0, 0.0]', 'must be increasing'),
    ('dropout_rate = 0.0', 'dropout_rate = { low = 0.0, high = 0.5 }', 'model.dropout_rate does not support a search'),
    ('[optimization]', '[optimization]\n[optimization.search.weight_decay]\nlow = 1e-8\nhigh = 1e-4', 'optimization.search does not support a search'),
    ('TB = 0.0', 'TB = 0.0\n[input_noise.search.TB]\nlow = 0.1\nhigh = 0.5', 'Unknown input_noise setting'),
])
def test_bad_recipes_fail_before_training(before, after, message, tmp_path):
    recipe, data = retrieval_project(tmp_path, 'nn')
    recipe.write_text(recipe.read_text().replace(before, after))
    with pytest.raises(ValueError, match=message):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert not (recipe.parent / 'model.pth').exists()


def test_missing_data_and_coordinate_mismatch_are_clear(tmp_path):
    recipe, data = retrieval_project(tmp_path)
    file = tmp_path / 'data/sites/test/training_variant/validation_03715.nc'
    file.unlink()
    with pytest.raises(FileNotFoundError, match='Missing input dataset'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    data.assign_coords(frq=[23.0, 31.4]).to_netcdf(file)
    with pytest.raises(ValueError, match='coordinates do not match'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')



def test_model_can_explicitly_inherit_dataset_grids_and_omit_dropout(tmp_path):
    recipe, data = retrieval_project(tmp_path, 'nn', 'bls', search=False)
    recipe.write_text(recipe.read_text().replace('frequencies = [22.24, 31.4]', 'frequencies = "dataset"')
                      .replace('heights = [0.0, 1000.0]', 'heights = "dataset"')
                      .replace('angles = [90.0, 30.0]', 'angles = "dataset"')
                      .replace('dropout_rate = 0.0', ''))
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    np.testing.assert_array_equal(workflow.model.freqs, data.frq)
    np.testing.assert_array_equal(workflow.model.heights, data.height)
    np.testing.assert_array_equal(workflow.model.angles, data.ang)
    assert workflow.model.dropout_rate == 0.0


def test_python_learning_rate_searches_validate_before_replacing_results(tmp_path):
    path, _ = retrieval_project(tmp_path, 'nn', search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example', 'cpu')
    checkpoint = path.parent / 'model.pth'
    checkpoint.write_bytes(b'Existing result')
    workflow.change_hyper_params(lr_low=workflow.suggest_float(1e-7, 0.003))
    with pytest.raises(ValueError, match='learning_rate_min must not exceed'):
        workflow.optimize(run_mode='fresh')
    assert checkpoint.read_bytes() == b'Existing result'


def test_per_angle_noise_has_distinct_parameters_from_equal_channel_frequencies(tmp_path):
    import tomllib
    import tomli_w
    from openMWR.recipes import RetrievalRecipe
    from openMWR.omb import get_input_noise

    path, _ = retrieval_project(tmp_path, 'nn', 'bls', search=False)
    values = tomllib.loads(path.read_text())
    values['model']['frequencies'] = [30.0, 31.4]
    values['input_noise'] = {'method': 'specified', 'TB': 2.0,
                             'channel_noise': {'31.40': 0.5},
                             'angle_dependence': 'per_angle', 'angle_factors': {'90.0': 1.0, '30.0': 3.0}}
    path.write_text(tomli_w.dumps(values))
    recipe = RetrievalRecipe.load(tmp_path, 'test', 'example')
    workflow = TrainingWorkflow(recipe.create_model(), 'test', 'cpu', tmp_path)
    assert not recipe.requires_optimization
    parameters = recipe.input_noise_params(workflow)
    assert parameters['TB_30.0'] == 2.0
    assert parameters['TB_angle_30.0'] == 3.0
    physical = get_input_noise(workflow.model, {f'input_noise_{key}': value for key, value in parameters.items()})
    np.testing.assert_array_equal(physical.TB.sel(ang=90), [2., 0.5])
    np.testing.assert_array_equal(physical.TB.sel(ang=30), [6., 1.5])
    # A fully specified channel table needs no common TB value.
    values['input_noise'].pop('TB')
    values['input_noise']['channel_noise']['30'] = 2.0
    values['input_noise']['channel_noise']['31.40'] = {'low': 0.1, 'high': 0.5}
    path.write_text(tomli_w.dumps(values))
    recipe = RetrievalRecipe.load(tmp_path, 'test', 'example')
    assert recipe.requires_optimization
    assert recipe.input_noise_params(workflow)['TB_31.4'] == workflow.suggest_float(0.1, 0.5)
    values['input_noise']['angle_factors']['30.0'] = {'low': 1.5, 'high': 3.0}
    path.write_text(tomli_w.dumps(values))
    recipe = RetrievalRecipe.load(tmp_path, 'test', 'example')
    assert recipe.requires_optimization
    assert recipe.input_noise_params(workflow)['TB_angle_30.0'] == workflow.suggest_float(1.5, 3.0)


@pytest.mark.parametrize(('section', 'setting', 'value', 'message'), [
    (None, 'retrievals', ['example', 'example'], 'duplicates'),
    (None, 'schema_version', 2, 'schema_version'),
    (None, 'bls_baseline', 'missing', 'must name a retrieval in retrievals'),
    (None, 'bls_baseline', '../example', 'without directory separators'),
    ('observations', 'data_source', 'other', 'radiosonde or era5'),
    ('observations', 'height_range', None, 'Missing observations settings'),
    ('observations', 'variables', ['TB', 'T'], 'TB separately'),
    ('observations', 'start_date', 'yesterday', 'quoted ISO date'),
    ('test', 'sources', [], 'at least one test source'),
    ('test', 'dataset', '../training', 'without directory separators'),
])
def test_named_evaluation_rejects_ambiguous_or_invalid_choices(tmp_path, section, setting, value, message):
    from openMWR.evaluation import EvaluationRecipe
    retrieval_project(tmp_path, search=False)
    recipe = EvaluationRecipe.load(tmp_path, 'test', 'example')
    settings = recipe.values if section is None else recipe.values[section]
    if value is None:
        settings.pop(setting)
    else:
        settings[setting] = value
    with pytest.raises(ValueError, match=message):
        recipe.validate()
