"""Observation selection, OMB preparation and portable evaluation reports."""


import numpy as np
import pytest
import xarray as xr

from openMWR.evaluation import evaluate_retrieval, write_report
from openMWR.models import load_model
from openMWR.train import TrainingWorkflow


from tests.fixtures.retrievals import retrieval_project


@pytest.mark.parametrize('kind', ['mlr', 'spectral'], ids=['profiles', 'spectra'])
def test_evaluation_writes_portable_report_and_preserves_existing_files(tmp_path, kind):
    recipe, _ = retrieval_project(tmp_path, kind=kind, search=False)
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').train()
    analysis = evaluate_retrieval(tmp_path, 'test', 'example')
    report = write_report(analysis)
    assert report.parent == recipe.parent / 'evaluation'
    html = report.read_text()
    assert 'Observation days 21-31' in html and 'data:image/' in html
    with pytest.raises(FileExistsError):
        write_report(analysis, report)
    all_days = evaluate_retrieval(tmp_path, 'test', 'example', days_of_month={'first': 1, 'last': 31})
    assert all_days.evaluations[0].samples['time_coordinates'] == 6
    assert 'Observation days 1-31' in all_days.evaluations[0].note
    with pytest.raises(FileNotFoundError, match='missing/zenith.nc'):
        evaluate_retrieval(tmp_path, 'test', 'example', dataset='missing')


def test_station_id_is_independent_of_file_order_and_requires_a_choice(tmp_path):
    recipe, data = retrieval_project(tmp_path, search=False)
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
    with pytest.raises(ValueError, match='available stations'):
        evaluate_retrieval(tmp_path, 'test', 'example', station_id='missing')
    recipe.write_text(recipe.read_text().replace('station_id = "03715"', ''))
    with pytest.raises(ValueError, match='Select evaluation.station_id'):
        evaluate_retrieval(tmp_path, 'test', 'example')


def test_era5_only_recipe_reuses_bias_correction_without_station_ids(tmp_path):
    recipe, data = retrieval_project(tmp_path)
    recipe.write_text(recipe.read_text().replace('sources = ["03715"]', 'sources = ["era5"]')
                      + '\n[bias_correction]\ndays_of_month = { first = 1, last = 20 }\ndata_source = "era5"\ndataset = "analysis_era5"\nfactor = 0.5\n')
    (tmp_path / 'sites/test/datasets/radiosondes.toml').write_text('kind = "era5"\nstart_date = "2024-01-01"\nend_date = "2024-01-31"\nhour_interval = 6\n')
    profiles_file = tmp_path / 'sites/test/datasets/profiles.toml'
    profiles_file.write_text('kind = "profiles"\n[inputs]\nsource = "radiosondes"\n[preparation]\nheights = [0.0, 1000.0]\n')
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
    recipe, data = retrieval_project(tmp_path, search=False)
    # Use the normal 14-channel grid so the existing neighbor-based OMB method
    # has channels on both sides of its K/V-band split.
    from openMWR.config import load_grid
    import tomli_w
    import tomllib

    frequencies = load_grid(tmp_path, 'frequencies', 'hatpro')
    rng = np.random.default_rng(17)
    data = data.drop_vars('TB').drop_dims('frq')
    data['TB'] = (('time', 'ang', 'frq'), 100 + rng.normal(size=(6, 2, len(frequencies))))
    data = data.assign_coords(frq=frequencies)
    for split in ('training_data', 'validation_data'):
        data.to_netcdf(tmp_path / f'data/sites/test/training_variant/{split.removesuffix("_data")}_03715.nc')
    background = data.sel(ang=90, drop=True)
    measured = background.assign(TB=background.TB + rng.normal(size=(6, len(frequencies))))
    observations = xr.concat([background, measured], dim='data_source').assign_coords(data_source=['era5', 'hatpro'])
    observations.to_netcdf(tmp_path / 'data/sites/test/analysis_era5/zenith.nc')
    settings = tomllib.loads(recipe.read_text())
    settings['model']['freqs'] = 'hatpro'
    settings.pop('optimization')
    settings['input_noise'] = {'method': 'omb', 'data_source': 'era5', 'dataset': 'analysis_era5', 'factor': 1.0,
                               'start_date': '2024-01-11', 'end_date': '2024-01-21',
                               'days_of_month': {'first': 1, 'last': 20}}
    recipe.write_text(tomli_w.dumps(settings))
    first = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    np.testing.assert_array_equal(first.omb.diff.time.values, data.time.values[1:3])
    settings['input_noise']['factor'] = 2.0
    recipe.write_text(tomli_w.dumps(settings))
    second = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    for frequency in frequencies:
        key = f'input_noise_TB_{frequency}'
        assert np.isfinite(first.hyper_params[key])
        assert second.hyper_params[key] == pytest.approx(2 * first.hyper_params[key])
    settings['input_noise'].update({'TB_22.24': 0.25, 'TB_31.4': {'low': 0.1, 'high': 0.4}})
    settings['optimization'] = {'n_trials': 1, 'data_source': 'era5', 'dataset': 'analysis_era5', 'seed': 42, 'days_of_month': {'first': 1, 'last': 20}, 'score_weights': {'T': 1.0, 'rh': 0.1}}
    recipe.write_text(tomli_w.dumps(settings))
    overridden = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    assert overridden.hyper_params['input_noise_TB_22.24'] == 0.25
    assert overridden.hyper_params['input_noise_TB_31.4'] == overridden.suggest_float(0.1, 0.4)
    assert overridden.hyper_params['input_noise_TB_23.04'] == second.hyper_params['input_noise_TB_23.04']
    assert overridden.requires_optimization
    assert not (recipe.parent / 'model.pth').exists()


def test_observation_periods_are_shared_by_bias_optimization_and_evaluation(tmp_path):
    import tomllib
    import tomli_w
    from openMWR.predict import run_model_list

    recipe, data = retrieval_project(tmp_path)
    path = tmp_path / 'data/sites/test/analysis_radiosonde/zenith.nc'
    paired = xr.load_dataset(path)
    times = paired.time.values.copy()
    times[2] = np.datetime64('2024-01-12T23:59:59.999999999')
    times[4] = np.datetime64('2024-01-22T23:59:59.999999999')
    paired = paired.assign_coords(time=times)
    paired['TB'].loc[{'data_source': 'hatpro'}] += xr.DataArray(np.arange(6), dims='time', coords={'time': times})
    future = paired.assign(T=paired.T + 50).assign_coords(time=times + np.timedelta64(366, 'D'))
    # Unrelated years are present in the same reusable analysis dataset.
    xr.concat([paired, future], dim='time').to_netcdf(path)
    selection = {'start_date': '2024-01-11', 'end_date': '2024-01-22',
                 'days_of_month': {'first': 12, 'last': 22}}
    settings = tomllib.loads(recipe.read_text())
    settings['bias_correction'] = dict(data_source='radiosonde', dataset='analysis_radiosonde', factor=0.5, **selection)
    settings['optimization'].update(selection, score_weights={'T': 1.0, 'rh': 0.0})
    settings['evaluation'].update(selection)
    recipe.write_text(tomli_w.dumps(settings))
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    np.testing.assert_array_equal(workflow.omb.diff.time.values, times[[2, 3, 4]])
    np.testing.assert_allclose(workflow.hyper_params['mean_bias'], [1.5, 1.5])
    study = workflow.optimize(n_trials=1)
    model = load_model('example', 'test', tmp_path)
    selected = paired.isel(time=[2, 3, 4])
    predicted = run_model_list([model], selected.sel(data_source='hatpro', drop=True), 'cpu')
    residual = predicted.T - selected.sel(data_source='radiosonde', drop=True).T
    expected = np.sqrt((residual ** 2).mean(('time', 'station'))).mean().item()
    assert study.best_value == pytest.approx(expected)
    with pytest.raises(ValueError, match='different or unrecorded experiment'):
        TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').optimize(n_trials=1, run_mode='continue', end_date='2024-01-21')
    analysis = evaluate_retrieval(tmp_path, 'test', 'example')
    assert analysis.evaluations[0].samples['time_coordinates'] == 3
    assert analysis.evaluations[1].samples['time_coordinates'] == data.time.size
    assert analysis.metadata['filters']['observation_end_date'] == '2024-01-22'
    # A year/date selection can be used without any monthly filter.
    settings['optimization'].pop('days_of_month')
    settings['evaluation'].pop('days_of_month')
    recipe.write_text(tomli_w.dumps(settings))
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    workflow.validate_optimization()
    assert evaluate_retrieval(tmp_path, 'test', 'example').evaluations[0].samples['time_coordinates'] == 4


def test_evaluation_height_ranges_select_observation_and_test_profiles(tmp_path):
    retrieval_project(tmp_path, search=False)
    TrainingWorkflow.from_recipe(tmp_path, 'test', 'example').train()
    result = evaluate_retrieval(tmp_path, 'test', 'example', observation_height_range=[0, 500], test_height_range=[500, 1000])
    np.testing.assert_array_equal(result.evaluations[0].metrics['error_profiles'].height, [0.])
    np.testing.assert_array_equal(result.evaluations[1].metrics['error_profiles'].height, [1000.])
    assert result.metadata['filters']['observation_height_range'] == [0, 500]
    assert result.metadata['filters']['test_height_range'] == [500, 1000]
