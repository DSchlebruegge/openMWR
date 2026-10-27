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
    ('nn', 'training', 'repeats_for_linear_regression = 2'),
])
def test_unused_model_settings_are_rejected(kind, section, setting, tmp_path):
    recipe, data = retrieval_project(tmp_path, kind, search=False)
    recipe.write_text(recipe.read_text().replace(f'[{section}]\n', f'[{section}]\n{setting}\n'))
    with pytest.raises(ValueError, match='is not used by'):
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
    ('TB = 0.0', 'BT = 0.0', 'Unknown input_noise'),
    ('TB = 0.0', '', 'Missing input_noise settings'),
    ('low = 0.001', 'kind = "float", low = 0.001', 'Unknown training.lr_height'),
    ('high = 0.002', 'high = 0.0001', 'must be at least low'),
    ('method = "standard"', '', 'input_noise.method must be'),
    ('method = "standard"', 'method = "typo"', 'input_noise.method must be'),
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
    ('loss_weights = {', '# loss_weights = {', 'Missing model settings: loss_weights'),
    ('norm_functions = {', '# norm_functions = {', 'Missing model settings: norm_functions'),
    ('freqs = [22.24, 31.4]', 'freqs = [22.24, 22.24]', 'must not contain duplicates'),
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
    recipe.write_text(recipe.read_text().replace('freqs = [22.24, 31.4]', 'freqs = "dataset"')
                      .replace('heights = [0.0, 1000.0]', 'heights = "dataset"')
                      .replace('angles = [90.0, 30.0]', 'angles = "dataset"')
                      .replace('dropout_rate = 0.0', ''))
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    np.testing.assert_array_equal(workflow.model.freqs, data.frq)
    np.testing.assert_array_equal(workflow.model.heights, data.height)
    np.testing.assert_array_equal(workflow.model.angles, data.ang)
    assert workflow.model.dropout_rate == 0.0


@pytest.mark.parametrize(('section', 'setting', 'value', 'message'), [
    (None, 'retrievals', ['example', 'example'], 'duplicates'),
    (None, 'schema_version', 2, 'schema_version'),
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
