"""Saved fixed models exercise the complete comparison and publication workflow."""

import numpy as np
import pytest
import torch
import xarray as xr

from openMWR.evaluation import EvaluationRecipe, evaluate, load_evaluation, save_evaluation
from openMWR.models import Model, SpectralConsistencyModel
from openMWR.site import create_site
from tests.fixtures.cli import run_cli


def save_fixed(model, root):
    specification = xr.merge([model.input, model.output])
    model.nor = xr.Dataset({
        name: xr.concat([xr.zeros_like(value), xr.ones_like(value), xr.ones_like(value)],
                        dim='stat').assign_coords(stat=['mean', 'std', 'max'])
        for name, value in specification.items()
    })
    # Older checkpoints can retain station labels in normalization coordinates.
    model.nor = model.nor.assign_coords(synop='03715')
    metadata = None
    if model.name in ('old', 'new'):
        first, last = (21, 21) if model.name == 'old' else (1, 20)
        metadata = {'setup': {'optimization': {'data_source': 'radiosonde',
                    'days_of_month': {'first': first, 'last': last}}}}
    model.save('test', root, {'training_dataset': 'old_name', 'sources': ['03715']}, metadata)


def comparison_project(root):
    create_site('test', root, mwr_height=550, invalid_measurements=[{
        'variables': ['surface_p'], 'start_date': '2024-01-22', 'end_date': '2024-01-22'}])
    times = np.arange('2024-01-20', '2024-01-24', dtype='datetime64[D]')
    heights = np.array([0., 1000.])
    frequencies = np.array([22.24, 31.4])
    angles = np.array([90., 30.])
    truth = xr.Dataset({
        'T': (('time', 'height'), np.repeat([[280.], [280.], [300.], [280.]], 2, axis=1)),
        'rh': (('time', 'height'), np.full((4, 2), 50.)),
        'lwc': (('time', 'height'), np.ones((4, 2))),
        'lwp': ('time', np.full(4, 20.)), 'iwv': ('time', np.full(4, 10.)),
        'surface_p': ('time', np.full(4, 740.)),
        'TB': (('time', 'ang', 'frq'), np.full((4, 2, 2), 100.)),
    }, coords={'time': times, 'height': heights, 'ang': angles, 'frq': frequencies})
    directory = root / 'data/sites/test/analysis_radiosonde'
    directory.mkdir(parents=True)
    for bls in (False, True):
        measured = truth.assign(T=xr.full_like(truth.T, 280.5), rh=xr.full_like(truth.rh, 50.5)).copy(deep=True)
        if bls:
            measured.TB.loc[{'time': times[-1]}] = np.nan
        paired = xr.concat([truth, measured], dim='data_source').assign_coords(
            data_source=['radiosonde', 'hatpro']).expand_dims(station=['03715'])
        if not bls:
            paired = paired.sel(ang=90, drop=True)
        paired.to_netcdf(directory / ('bls.nc' if bls else 'zenith.nc'))
    directory = root / 'data/sites/test/training_reference'
    directory.mkdir()
    truth.assign(T=xr.full_like(truth.T, 280.)).assign_coords(
        time=np.arange('2023-01-01', '2023-01-05', dtype='datetime64[D]')).to_netcdf(directory / 'test_03715.nc')
    for name, offset, bls, extra in [('old', 1, False, ['surface_p']),
                                     ('new', 2, False, []), ('scan', 3, True, [])]:
        model = Model(name, heights, frequencies, is_MLR=True, is_bls_model=bls,
                      angles=angles if bls else None, additional_input_vars=extra,
                      norm_functions={name: 'z_norm' for name in truth.data_vars})
        with torch.no_grad():
            model.stack.weight.zero_()
            model.stack.bias.copy_(torch.tensor([280+offset, 280+offset, 50+offset, 50+offset, 1, 1, 20, 10]))
        save_fixed(model, root)
    directory = root / 'sites/test/evaluations'
    directory.mkdir()
    (directory / 'comparison.toml').write_text('''schema_version = 1
retrievals = ["old", "new", "scan"]
[observations]
dataset = "analysis_radiosonde"
data_source = "radiosonde"
station_id = "03715"
variables = ["T", "rh"]
height_range = [0, 10000]
days_of_month = { first = 21, last = 31 }
include_rpg = true
[test]
dataset = "training_reference"
sources = ["03715"]
variables = ["T", "rh", "lwc", "lwp"]
height_range = "model"
''')
    return EvaluationRecipe.load(root, 'test', 'comparison')


def test_common_cases_roundtrip_and_command_preserve_inputs(tmp_path, monkeypatch):
    recipe = comparison_project(tmp_path)
    originals = {path: path.read_bytes() for path in tmp_path.rglob('*')
                 if path.suffix in ('.pth', '.nc', '.toml')}
    result = evaluate(recipe)
    observations, test = result.evaluations
    assert observations.samples['stations'] == ['03715']
    assert observations.samples['time_coordinates'] == 3
    for name, offset, count in [('old', 1, 2), ('new', 2, 3), ('scan', 3, 2)]:
        assert observations.metrics['errors'].T.sel(model=name, error='RMSE') == pytest.approx(offset)
        assert observations.metrics['available_profiles'].T.sel(model=name) == count
        assert observations.metrics['paired_profiles'].T.sel(model=name) == 1
        assert test.metrics['paired_profiles'].T.sel(model=name) == 4
    assert observations.metrics['errors'].T.sel(model='RPG', error='ME') == pytest.approx(0.5)
    assert np.isnan(observations.metrics['errors'].rh.sel(model='RPG_BLS', error='RMSE'))
    assert any('Independence' in note for note in result.metadata['models'][0]['notes'])
    assert any('dates overlap' in note for note in result.metadata['models'][0]['notes'])
    assert any('dates are outside' in note for note in result.metadata['models'][1]['notes'])
    single = evaluate(recipe, retrievals=['new'])
    assert single.evaluations[0].metrics['paired_profiles'].T.sel(model='new') == 3
    assert recipe.values['retrievals'] == ['old', 'new', 'scan']
    directory = save_evaluation(result)
    assert {path.name for path in directory.iterdir()} == {'report.html', 'metrics.nc'}
    html = (directory / 'report.html').read_text()
    assert 'data:image/' in html and 'common finite' in html and 'training_reference' in html
    restored = load_evaluation(directory)
    assert restored.metadata == result.metadata
    for original, loaded in zip(result.evaluations, restored.evaluations):
        for name in original.metrics:
            xr.testing.assert_identical(original.metrics[name], loaded.metrics[name])
    report = (directory / 'report.html').read_bytes()
    with pytest.raises(FileExistsError):
        save_evaluation(result, directory)
    assert (directory / 'report.html').read_bytes() == report
    command = run_cli('evaluate', 'test', 'comparison', '-p', str(tmp_path), '-o', str(tmp_path / 'cli run'))
    assert command.returncode == 0, command.stderr
    actual = load_evaluation(tmp_path / 'cli run')
    xr.testing.assert_equal(actual.evaluations[0].metrics['errors'], observations.metrics['errors'])
    assert all(path.read_bytes() == before for path, before in originals.items())
    def fail_render(analysis):
        raise RuntimeError('Rendering failed')
    monkeypatch.setattr('openMWR.evaluation.render_report', fail_render)
    with pytest.raises(RuntimeError, match='Rendering failed'):
        save_evaluation(result, tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists() and not list(tmp_path.glob('.evaluation-*'))


def test_explicit_station_period_and_height_selection_and_missing_coverage(tmp_path):
    recipe = comparison_project(tmp_path)
    recipe.values.pop('test')
    settings = recipe.values['observations']
    settings.update(start_date='2024-01-21', end_date='2024-01-21', height_range=[0, 500], include_rpg=False)
    for path in (tmp_path / 'data/sites/test/analysis_radiosonde').glob('*.nc'):
        original = xr.load_dataset(path)
        other = original.assign(T=original.T + 50).assign_coords(station=['02290'])
        xr.concat([other, original], dim='station').to_netcdf(path)
    result = evaluate(recipe, retrievals=['new']).evaluations[0]
    assert result.samples['time_coordinates'] == 1
    np.testing.assert_array_equal(result.metrics['error_profiles'].height, [0.])
    assert result.metrics['errors'].T.sel(error='ME') == pytest.approx(2.)
    settings['station_id'] = '02290'
    result = evaluate(recipe, retrievals=['new']).evaluations[0]
    assert result.metrics['errors'].T.sel(error='ME') == pytest.approx(-48.)
    settings.pop('station_id')
    with pytest.raises(ValueError, match='Select observations.station_id'):
        evaluate(recipe, retrievals=['new'])
    settings.update(station_id='03715', start_date='2024-01-22', end_date='2024-01-22')
    result = evaluate(recipe, retrievals=['old', 'new']).evaluations[0]
    assert result.metrics['available_profiles'].T.sel(model='old') == 0
    assert result.metrics['available_profiles'].T.sel(model='new') == 1
    assert np.isnan(result.metrics['errors'].T).all()
    assert result.metrics['paired_profiles'].T.sum() == 0
    settings['height_range'] = [2000, 3000]
    with pytest.raises(ValueError, match='height'):
        evaluate(recipe, retrievals=['new'])


def test_spectral_retrieval_uses_the_same_engine_and_result_format(tmp_path):
    recipe = comparison_project(tmp_path)
    model = SpectralConsistencyModel('spectral', np.array([22.24, 31.4]), hidden_layer_size=3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    save_fixed(model, tmp_path)
    recipe.values['retrievals'] = ['spectral']
    for section in ('observations', 'test'):
        recipe.values[section]['variables'] = ['TB']
        recipe.values[section].pop('height_range')
    recipe.values['observations'].pop('include_rpg')
    result = evaluate(recipe)
    for section in result.evaluations:
        np.testing.assert_allclose(section.metrics['errors'].TB.sel(error='RMSE'), 100.)
    restored = load_evaluation(save_evaluation(result))
    xr.testing.assert_identical(restored.evaluations[0].metrics['errors'], result.evaluations[0].metrics['errors'])
