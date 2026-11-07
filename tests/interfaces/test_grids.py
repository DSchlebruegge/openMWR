"""Editable named grids are shared by dataset recipes and Python calls."""

import tomllib

import numpy as np
import pytest
import tomli_w

from openMWR.datasets import DatasetWorkflow


from tests.fixtures.datasets import dataset_project, recipe

def test_named_grids_and_optional_scaling_are_explicit(dataset_project):
    from openMWR.config import load_grid

    original = load_grid(dataset_project, 'heights', 'levels_208_30km')

    recipe(dataset_project, 'profiles', kind='profiles', inputs={'source': 'source'},
           preparation={'heights': 'levels_208_30km'})
    workflow = DatasetWorkflow(dataset_project, 'test')
    np.testing.assert_array_equal(workflow.coordinates('training')['heights'], original)
    assert not workflow.settings('profiles')['scale_rh_clear_sky']
    # Resolution preserves editable presets and observes Python edits next time.
    preparation = workflow.recipe('profiles').values['preparation']
    assert preparation['heights'] == 'levels_208_30km'
    preparation['heights'] = [0., 1000.]
    np.testing.assert_array_equal(workflow.coordinates('training')['heights'], [0., 1000.])
    preparation['heights'] = [1000., 0.]
    with pytest.raises(ValueError, match='must be increasing'):
        workflow.coordinates('training')
    preparation['heights'] = [0., 500., 1000.]
    np.testing.assert_array_equal(workflow.coordinates('training')['heights'], [0., 500., 1000.])
    path = dataset_project / 'sites/test/datasets/profiles.toml'
    values = tomllib.loads(path.read_text())
    values['preparation']['scale_rh_clear_sky'] = True
    path.write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match='Missing .*scale_rh_clear_sky_factor'):
        DatasetWorkflow(dataset_project, 'test').build('profiles', dry_run=True)
    values['preparation'].update(scale_rh_clear_sky_factor=0.2, scale_rh_clear_sky_fraction=0.1,
                                 scale_rh_clear_sky_seed=42)
    path.write_text(tomli_w.dumps(values))
    DatasetWorkflow(dataset_project, 'test').build('profiles', with_dependencies=True, dry_run=True)


def test_project_grid_edits_are_shared_by_python_and_dataset_recipes(dataset_project, monkeypatch):
    from openMWR.config import load_grid

    path = dataset_project / 'grids.toml'
    grids = {'heights': {'custom': [0., 500., 1000.]},
             'frequencies': {'custom': [22.24, 31.4]},
             'angles': {'custom': [90., 30.]}}
    path.write_text(tomli_w.dumps(grids))
    recipe(dataset_project, 'profiles', kind='profiles', inputs={'source': 'source'},
           preparation={'heights': 'custom'})
    recipe(dataset_project, 'r24', kind='simulation', inputs={'profiles': 'profiles'},
           radiative_transfer={'frequencies': 'custom', 'angles': 'custom'})
    workflow = DatasetWorkflow(dataset_project, 'test')
    monkeypatch.chdir(dataset_project.parent)
    for kind, values in workflow.coordinates('training').items():
        assert values == load_grid(dataset_project, kind, 'custom') == grids[kind]['custom']
    # Edits are visible on later calls using the same workflow.
    grids['heights']['custom'] = [0., 1000.]
    path.write_text(tomli_w.dumps(grids))
    assert workflow.coordinates('training')['heights'] == [0., 1000.]
    with pytest.raises(ValueError, match='Unknown .*levels_208_30km'):
        load_grid(dataset_project, 'heights', 'levels_208_30km')
    path.unlink()
    assert load_grid(dataset_project, 'heights', [0., 1000.]) == [0., 1000.]
    with pytest.raises(FileNotFoundError, match='grids.toml'):
        load_grid(dataset_project, 'heights', 'custom')


@pytest.mark.parametrize(('kind', 'values', 'message'), [
    ('heights', [0., 1000., 1000.], 'duplicates'),
    ('heights', [0., 1000., 500.], 'increasing'),
    ('frequencies', [22.24, float('nan')], 'finite'),
    ('angles', [90., 0.], 'between 0 and 90'),
])
def test_invalid_named_grids_are_rejected(dataset_project, kind, values, message):
    from openMWR.config import load_grid

    (dataset_project / 'grids.toml').write_text(tomli_w.dumps({kind: {'invalid': values}}))
    with pytest.raises(ValueError, match=message):
        load_grid(dataset_project, kind, 'invalid')
