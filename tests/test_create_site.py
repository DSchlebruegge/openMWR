"""Create an editable radiosonde starter without processing any data."""

import subprocess
import sys
import tomllib

import pytest
import tomli_w

from openMWR.datasets import DatasetWorkflow
from openMWR.recipes import RetrievalRecipe
from openMWR.site import create_site


def command(project, site):
    return subprocess.run([sys.executable, '-m', 'openMWR', 'create-site', site,
        '--project-dir', str(project)], cwd=project.parent, capture_output=True, text=True)


def test_command_creates_radiosonde_starter_and_requires_station_selection(tmp_path):
    project = tmp_path / 'new project'
    result = command(project, 'my_site')
    assert result.returncode == 0, result.stderr
    assert 'site.toml' in result.stdout and 'radiosondes.toml' in result.stdout
    assert 'grids.toml' in result.stdout
    grids = tomllib.loads((project / 'grids.toml').read_text())
    assert set(grids) == {'heights', 'frequencies', 'angles'}
    assert len(grids['heights']['levels_208_30km']) == 208
    assert 'openMWR data my_site training --with-dependencies --dry-run' in result.stdout
    directory = project / 'sites/my_site'
    datasets = {path.stem: tomllib.loads(path.read_text())
                for path in (directory / 'datasets').glob('*.toml')}
    assert set(datasets) == {'radiosondes', 'profiles', 'simulation', 'training',
                             'hatpro', 'analysis_radiosonde'}
    assert datasets['radiosondes']['stations'] == []
    assert datasets['simulation']['radiative_transfer']['absorption_model'] == 'R24'
    assert datasets['profiles']['preparation']['heights'] == 'levels_208_30km'
    assert all(settings['kind'] != 'era5' for settings in datasets.values())
    settings = tomllib.loads((directory / 'site.toml').read_text())
    assert settings == {'prediction': {'retrievals': ['NN_example']}}
    recipe = RetrievalRecipe.load(project, 'my_site', 'NN_example')
    assert recipe.section('model')['freqs'] == 'dataset'
    assert all(not isinstance(value, dict) for value in recipe.section('training').values())
    assert not list((project / 'data').rglob('*.*'))
    before = {path: path.read_bytes() for path in project.rglob('*.toml')}
    with pytest.raises(ValueError, match='stations must not be empty'):
        DatasetWorkflow(project, 'my_site').build('training', with_dependencies=True)
    assert before == {path: path.read_bytes() for path in project.rglob('*.toml')}
    assert not list((project / 'data').rglob('*.*'))


def test_existing_site_is_preserved_and_another_site_can_be_created(tmp_path):
    directory = create_site('first', tmp_path, with_examples=True)
    (directory / 'site.toml').write_text('# User edits\ngen = "G4"\n')
    grids_file = tmp_path / 'grids.toml'
    grids = tomllib.loads(grids_file.read_text())
    grids['heights']['custom'] = [0., 1000.]
    grids_file.write_text(tomli_w.dumps(grids))
    checkpoint = directory / 'retrieval/NN_example/model.pth'
    checkpoint.write_bytes(b'saved model')
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    rejected = command(tmp_path, 'first')
    assert rejected.returncode == 1 and 'Site already exists' in rejected.stderr
    assert before == {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    created = command(tmp_path, 'second')
    assert created.returncode == 0, created.stderr
    assert (tmp_path / 'sites/second/site.toml').is_file()
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('site', ['../outside', '.', 'nested/site'])
def test_invalid_site_names_create_nothing(tmp_path, site):
    result = command(tmp_path, site)
    assert result.returncode == 1 and 'site must be a nonempty name' in result.stderr
    assert not list(tmp_path.iterdir())


def test_python_configuration_overrides_and_both_dataset_chains(tmp_path):
    directory = create_site('test', tmp_path, with_examples=True,
        gen='G4', mwr_height=123, hatpro_data_dir='raw',
        mwr_measurement_start_date='2020-01-01', prediction={'retrievals': ['custom']})
    settings = tomllib.loads((directory / 'site.toml').read_text())
    assert settings['gen'] == 'G4' and settings['mwr_height'] == 123
    assert settings['prediction'] == {'retrievals': ['custom']}
    source = directory / 'datasets/radiosondes.toml'
    source.write_text(source.read_text().replace('stations = []', 'stations = ["03715"]'))
    workflow = DatasetWorkflow(tmp_path, 'test')
    assert workflow.build('training', with_dependencies=True, dry_run=True) == [
        'radiosondes', 'profiles', 'simulation', 'training']
    assert workflow.build('analysis_radiosonde', with_dependencies=True, dry_run=True) == [
        'hatpro', 'radiosondes', 'profiles', 'simulation', 'analysis_radiosonde']
    assert workflow.settings('simulation')['absorption_model'] == 'R24'
    from openMWR.train import TrainingWorkflow
    recipe = RetrievalRecipe.load(tmp_path, 'test', 'NN_example')
    training = TrainingWorkflow(recipe.create_model(), 'test', 'cpu', tmp_path)
    training.add_standard_training_hyper_params(**recipe.training_params(training))
    training.add_input_noise_params(**recipe.input_noise_params(training))
    assert not training.requires_optimization
    del settings['mwr_height']
    (directory / 'site.toml').write_text(tomli_w.dumps(settings))
    with pytest.raises(ValueError, match='set mwr_height in .*site.toml'):
        DatasetWorkflow(tmp_path, 'test').build('training', with_dependencies=True)
    assert not list((tmp_path / 'data').rglob('*.*'))
