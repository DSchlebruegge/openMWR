"""Invalid scientific choices and dependency graphs fail before I/O."""

import tomllib

import pytest
import tomli_w

from openMWR.datasets import DatasetRecipe, DatasetWorkflow


from tests.fixtures.datasets import dataset_project, recipe

@pytest.mark.parametrize('values, message', [
    ({'kind': 'simulation', 'inputs': {'profiles': 'source'}}, 'incompatible kind'),
    ({'kind': 'simulation', 'inputs': {'profiles': 'missing'}}, 'Missing dataset recipe'),
    ({'kind': 'simulation', 'inputs': {'profiles': 'profiles'}, 'radiative_transfer': {'absorpton_model': 'R24'}}, 'Unknown'),
    ({'kind': 'simulation', 'inputs': {'profiles': 'profiles'}, 'radiative_transfer': {'ray_tracing': 'true'}}, 'ray_tracing must be true or false'),
    ({'kind': 'simulation', 'inputs': {'profiles': 'profiles'}, 'radiative_transfer': {'ray_tracing': 1}}, 'ray_tracing must be true or false'),
    ({'kind': 'profiles', 'inputs': {'source': 'source'}, 'sources': ['02290']}, 'subset'),
    ({'kind': 'profiles', 'inputs': {'source': 'source'}, 'preparation': {'heights': [100, 0]}}, 'must be increasing'),
    ({'kind': 'profiles', 'inputs': {'source': 'source'}, 'execution': {'processes': 2}}, 'preparation is serial'),
    ({'kind': 'split', 'inputs': {'simulation': 'r24'}, 'split': {'test_days_of_month': {'first': 15, 'last': 18}}}, 'must not overlap'),
    ({'kind': 'split', 'inputs': {'simulation': 'r24'}, 'split': {'test_days_of_month': {'first': 0, 'last': 3}}}, 'at least 1'),
    ({'kind': 'split', 'inputs': {'simulation': 'r24'}, 'split': {'test_days_of_month': {'first': 1, 'last': 14}, 'validation_days_of_month': {'first': 15, 'last': 31}}}, 'leave days for training'),
    ({'kind': 'era5', 'start_date': '20200101'}, 'quoted ISO date'),
    ({'kind': 'hatpro', 'end_date': '2020-W01-1'}, 'quoted ISO date'),
    ({'kind': 'era5', 'start_date': '2020-01-02', 'end_date': '2020-01-01'}, 'must not be after'),
    ({'kind': 'era5', 'end_date': '2020-01-01', 'delay_days': 5}, 'delay_days applies only'),
    ({'kind': 'simulation', 'inputs': {'profiles': 'profiles'}, 'radiative_transfer': {'frequency_shift': [0.01]}}, 'nonempty table'),
    ({'kind': 'simulation', 'inputs': {'profiles': 'profiles'}, 'radiative_transfer': {'frequency_shift': {'999': 0.01}}}, 'selected frequencies'),
])
def test_invalid_graphs_fail_before_output_creation(dataset_project, values, message):
    recipe(dataset_project, 'bad', **values)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        DatasetWorkflow(dataset_project, 'test').build('bad', with_dependencies=True)
    assert list((dataset_project / 'data/sites/test').iterdir()) == []


@pytest.mark.parametrize(('name', 'section', 'key'), [
    ('source', None, 'end_year'),
    ('profiles', 'preparation', 'heights'),
    ('profiles', 'preparation', 'cloud'),
    ('profiles', 'preparation', 'quality'),
    ('profiles', 'preparation', 'incomplete_profiles'),
    ('r24', 'radiative_transfer', 'absorption_model'),
    ('r24', 'radiative_transfer', 'ray_tracing'),
    ('r24', 'radiative_transfer', 'infrared'),
    ('training', 'split', 'test_days_of_month'),
])
def test_missing_scientific_choices_fail_before_processing(dataset_project, name, section, key):
    path = dataset_project / 'sites/test/datasets' / f'{name}.toml'
    values = tomllib.loads(path.read_text())
    del (values[section] if section else values)[key]
    path.write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match=f'Missing .* settings: {key}'):
        DatasetWorkflow(dataset_project, 'test').build('training', with_dependencies=True)
    assert list((dataset_project / 'data/sites/test').iterdir()) == []


@pytest.mark.parametrize(('cloud', 'message'), [
    ({'rh_threshold': 0.96, 'min_temperature': 253.15, 'a': 0.7, 'b': 2.3}, 'Missing .*adiabatic_fraction_method'),
    ({'rh_threshold': 0.96, 'min_temperature': 253.15, 'adiabatic_fraction_method': 'unknown'}, 'must be power_law, karstens or fully_adiabatic'),
    ({'rh_threshold': 0.96, 'min_temperature': 253.15, 'adiabatic_fraction_method': 'power_law', 'a': 0.7}, 'Missing .*b'),
    ({'rh_threshold': 0.96, 'min_temperature': 253.15, 'adiabatic_fraction_method': 'karstens', 'a': 0.7}, 'remove them for this method'),
])
def test_cloud_method_choices_fail_before_processing(dataset_project, cloud, message):
    path = dataset_project / 'sites/test/datasets/profiles.toml'
    values = tomllib.loads(path.read_text())
    values['preparation']['cloud'] = cloud
    path.write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match=message):
        DatasetWorkflow(dataset_project, 'test').build('profiles')
    assert list((dataset_project / 'data/sites/test').iterdir()) == []


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('cloud', 'rh_threshold', 2.0),
    ('quality', 'min_top_height', -1),
    ('quality', 'min_levels', 0),
])
def test_invalid_profile_settings_fail_before_io(dataset_project, section, key, value):
    from openMWR.radiosonde import create_radiosonde_dataset_for_RT

    path = dataset_project / 'sites/test/datasets/profiles.toml'
    values = tomllib.loads(path.read_text())
    preparation = values['preparation']
    preparation[section][key] = value
    with pytest.raises(ValueError, match=rf'preparation\.{section}\.{key}'):
        path.write_text(tomli_w.dumps(values))
        DatasetRecipe.load(dataset_project, 'test', 'profiles')
    assert list((dataset_project / 'data/sites/test').iterdir()) == []


@pytest.mark.parametrize('radius', [2.49, 60.0])
def test_unsupported_infrared_radius_fails_before_processing(dataset_project, radius):
    path = dataset_project / 'sites/test/datasets/r24.toml'
    values = tomllib.loads(path.read_text())
    values['radiative_transfer']['infrared']['min_effective_radius'] = radius
    path.write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match='min_effective_radius must be at least 2.5 and below 60.*libRadtran'):
        DatasetWorkflow(dataset_project, 'test').build('r24', with_dependencies=True)
    assert list((dataset_project / 'data/sites/test').iterdir()) == []


def test_dependency_cycles_fail_before_output_creation(dataset_project):
    recipe(dataset_project, 'profiles', kind='profiles', inputs={'source': 'training'})
    with pytest.raises(ValueError, match='Circular dataset dependency'):
        DatasetWorkflow(dataset_project, 'test').build('training', with_dependencies=True)
    assert not list((dataset_project / 'data/sites/test').iterdir())
