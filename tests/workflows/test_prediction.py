"""Saved retrievals applied to small local RPG measurements."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from openMWR.predict import PredictionWorkflow, load_predictions, run_model_list

from tests.fixtures.prediction import prediction_project, raw_day, checkpoint
from tests.fixtures.cli import run_cli


def command(prediction_project, *args):
    return run_cli("predict", "test", *args, "-p", str(prediction_project))

def test_known_prediction_smoothing_and_netcdf_roundtrip(prediction_project, monkeypatch):
    workflow = PredictionWorkflow(prediction_project, 'test')
    assert workflow.device == 'cpu'
    # A later availability change must not make an existing workflow switch GPUs.
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    result = workflow.predict('2024-01-01')
    assert not (prediction_project / 'published').exists()  # Python prediction has no writes.
    assert not result.errors
    # TB samples average to 100.5, 102.5, ..., 110.5 K. Normalization
    # subtracts one, the known checkpoint applies 0.1*x + 290, and
    # denormalization adds one. The zenith series has a centered 5-point mean.
    np.testing.assert_allclose(result.zenith.T.sel(height=0).squeeze(),
                               [301.15, 301.25, 301.35, 301.55, 301.65, 301.75], rtol=0, atol=5e-5)
    np.testing.assert_allclose(result.bls.T.sel(height=0).squeeze(),
                               [300.9, 301., 301.1, 301.2, 301.3, 301.4], rtol=0, atol=5e-5)
    assert result.zenith.sizes['time'] == 6
    assert result.bls.T.attrs['units'] == 'K'
    paths = workflow.save(result)
    assert [path.name for path in paths] == ['20240101_retrieval.nc', '20240101_retrieval_bls.nc']
    assert all(xr.load_dataset(path).sizes['model'] == 1 for path in paths)
    cli = command(prediction_project, '-d', '2024-01-01', '-D', 'cpu')
    assert cli.returncode == 0, cli.stderr
    loaded = load_predictions(prediction_project, 'test', '2024-01-01')
    for name in ('zenith', 'bls'):
        expected = getattr(result, name).drop_vars(['doy_sin', 'doy_cos'])
        xr.testing.assert_identical(getattr(loaded, name), expected)
    assert loaded.observations is None
    assert not list((prediction_project / 'data/sites/test').iterdir())  # No prepared dataset generation.

    # Previously published single-model BLS files remain readable.
    historical = xr.load_dataset(paths[1]).squeeze('model')
    historical.to_netcdf(paths[1])
    restored = load_predictions(prediction_project, 'test', '2024-01-01')
    xr.testing.assert_identical(restored.bls, loaded.bls)

    # Missing data in one station must not discard a valid sample at another.
    stations = xr.concat([result.observations, result.observations], dim='station').assign_coords(station=['a', 'b'])
    stations['TB'][0, :3] = np.nan
    stations['TB'][1, 3:] = np.nan
    partial = run_model_list([workflow.models[0]], stations, 'cpu')
    assert int(partial.T.sel(height=0).notnull().sum()) == 6


def test_range_continues_after_missing_day_and_reports_partial_bls(prediction_project):
    raw_day(prediction_project, datetime(2024, 1, 3), bls=False)
    workflow = PredictionWorkflow(prediction_project, 'test')
    for day in ('2024-01-01', '2024-01-02', '2024-01-03'):
        workflow.save(workflow.predict(day))
    directory = prediction_project / 'published/2024/01'
    assert {path.name for path in directory.iterdir()} == {
        '20240101_retrieval.nc', '20240101_retrieval_bls.nc', '20240103_retrieval.nc'}
    partial = PredictionWorkflow(prediction_project, 'test').predict('2024-01-03')
    assert partial.zenith is not None and partial.bls is None
    assert set(partial.errors) == {'bls'}
    workflow = PredictionWorkflow(prediction_project, 'test')
    missing = workflow.predict('2024-01-02')
    assert missing.zenith is None and missing.bls is None
    assert set(missing.errors) == {'zenith', 'bls'}
    assert workflow.save(missing) == []
    assert not (directory / '20240102_retrieval.nc').exists()


def test_computation_errors_are_not_reported_as_missing_data(prediction_project, monkeypatch):
    workflow = PredictionWorkflow(prediction_project, 'test', ['primary'])

    def broken_forward(inputs):
        raise ValueError('Inference failed')

    monkeypatch.setattr(workflow.models[0], 'forward', broken_forward)
    with pytest.raises(ValueError, match='Inference failed'):
        workflow.predict('2024-01-01')


def test_override_paths_replacement_model_reuse_and_reading_without_checkpoints(prediction_project):
    raw_day(prediction_project, datetime(2024, 1, 2), bls=False)
    workflow = PredictionWorkflow(prediction_project, 'test', ['primary'], output_dir='experiment')
    first = workflow.predict('2024-01-01')
    paths = workflow.save(first)
    assert paths[0] == prediction_project / 'experiment/2024/01/20240101_retrieval.nc'
    previous = paths[0].read_bytes()
    absolute = prediction_project.parent / (prediction_project.name + '-absolute')
    other = PredictionWorkflow(prediction_project, 'test', ['primary'], output_dir=absolute)
    assert other.save(first)[0].is_relative_to(absolute)
    for path in prediction_project.glob('sites/test/retrieval/*/model.pth'):
        path.unlink()
    second = workflow.predict('2024-01-02')  # Reuses loaded models.
    workflow.save(second)
    workflow.models[0].stack.bias.data.add_(1)
    workflow.save(workflow.predict('2024-01-02'))
    assert paths[0].read_bytes() == previous
    loaded = load_predictions(prediction_project, 'test', '2024-01-02', output_dir='experiment')
    assert loaded.bls is None
    assert not loaded.zenith.T.identical(second.zenith.T)


def test_bls_only_and_multiple_bls_models(prediction_project):
    (prediction_project / 'raw/Y2024/M01/D01/240101.BRT.NC').unlink()
    checkpoint(prediction_project, 'scan2', bls=True)
    workflow = PredictionWorkflow(prediction_project, 'test', ['scan', 'scan2'])
    result = workflow.predict('2024-01-01')
    assert result.zenith is None and not result.errors
    path, = workflow.save(result)
    assert xr.load_dataset(path).model.values.tolist() == ['scan', 'scan2']


def test_spectral_prediction_and_unsmoothed_python_override(prediction_project):
    checkpoint(prediction_project, 'spectral', spectral=True)
    workflow = PredictionWorkflow(prediction_project, 'test', ['spectral'], smooth=False)
    result = workflow.predict(datetime(2024, 1, 1, 2, tzinfo=timezone(timedelta(hours=2))))
    assert result.date == datetime(2024, 1, 1)
    expected = run_model_list(workflow.models, result.observations, 'cpu')
    xr.testing.assert_allclose(result.zenith.TB, expected.TB)
    assert 'T' not in result.zenith


def test_configuration_and_checkpoint_selection_before_writing(prediction_project):
    config = prediction_project / 'sites/test/site.toml'
    config.write_text(config.read_text().replace('[prediction]', '[unused]'))
    with pytest.raises(ValueError, match='Select retrieval names'):
        PredictionWorkflow(prediction_project, 'test')
    with pytest.raises(ValueError, match='only once'):
        PredictionWorkflow(prediction_project, 'test', ['primary', 'primary'])
    with pytest.raises(FileNotFoundError):
        PredictionWorkflow(prediction_project, 'test', ['missing'])
    # A broken training recipe is irrelevant for saved-model inference.
    (prediction_project / 'sites/test/retrieval/primary/retrieval.toml').write_text('invalid = [')
    workflow = PredictionWorkflow(prediction_project, 'test', ['primary'], output_dir='chosen')
    workflow.save(workflow.predict('2024-01-01'))
    assert (prediction_project / 'chosen/2024/01/20240101_retrieval.nc').exists()
    assert not (prediction_project / 'published').exists()


def test_default_output_and_missing_saved_predictions(prediction_project):
    config = prediction_project / 'sites/test/site.toml'
    config.write_text(config.read_text().replace('retrieval_output_dir = "published"\n', ''))
    workflow = PredictionWorkflow(prediction_project, 'test', ['primary'])
    path, = workflow.save(workflow.predict('2024-01-01'))
    assert path == prediction_project / 'data/sites/test/predictions/2024/01/20240101_retrieval.nc'
    with pytest.raises(FileNotFoundError, match='No saved predictions'):
        load_predictions(prediction_project, 'test', '2024-01-02')


@pytest.mark.parametrize('weather', ['declared-outage', 'missing-met', 'unreadable-met'])
def test_prediction_without_weather_keeps_other_products_and_reuses_loaded_observations(prediction_project, weather):
    from openMWR.hatpro_data import import_hatpro_data

    checkpoint(prediction_project, 'without_weather', weather=False)
    raw = prediction_project / 'raw/Y2024/M01/D01'
    before = (raw / '240101.BRT.NC').read_bytes()
    if weather == 'missing-met':
        (raw / '240101.MET.NC').unlink()
    elif weather == 'unreadable-met':
        (raw / '240101.MET.NC').write_bytes(b'Broken auxiliary sensor file')
    else:
        config = prediction_project / 'sites/test/site.toml'
        config.write_text(config.read_text() + '''
[[invalid_measurements]]
variables = ["surface_T", "surface_p", "surface_rh"]
periods = [{ start_date = "2024-01-01", end_date = "2024-01-01" }]
reason = "Station unavailable"
''')
    observations = import_hatpro_data(datetime(2024, 1, 1), 'test', prediction_project,
                                     mean_one_min=True, import_retrieval_data=False)
    assert observations.time.size == 6 and np.isfinite(observations.TB).all()
    workflow = PredictionWorkflow(prediction_project, 'test', ['without_weather'])
    loaded = workflow.predict('2024-01-01', observations=observations)
    direct = workflow.predict('2024-01-01')
    assert not loaded.errors and loaded.zenith is not None
    xr.testing.assert_identical(loaded.zenith, direct.zenith)
    assert {'ah', 'TD', 'T_grad', 'lwp', 'iwv', 'lwp_from_lwc', 'iwv_from_ah'} <= set(loaded.zenith)
    assert 'p' not in loaded.zenith
    assert 'surface_p' not in loaded.zenith or not np.isfinite(loaded.zenith.surface_p).any()
    path, = workflow.save(loaded)
    assert xr.load_dataset(path).model.values.tolist() == ['without_weather']
    normal = PredictionWorkflow(prediction_project, 'test', ['primary']).predict('2024-01-01')
    assert normal.zenith is None and 'zenith' in normal.errors
    wrong_day = workflow.predict('2024-01-02', observations=observations)
    assert 'requested UTC day' in wrong_day.errors['zenith']
    assert (raw / '240101.BRT.NC').read_bytes() == before
    if weather == 'declared-outage':
        import tomllib
        import tomli_w

        values = tomllib.loads(config.read_text())
        values['invalid_measurements'] = {}
        config.write_text(tomli_w.dumps(values))
        with pytest.raises(ValueError, match='array of tables'):
            import_hatpro_data(datetime(2024, 1, 1), 'test', prediction_project)
        with pytest.raises(ValueError, match='array of tables'):
            workflow.predict('2024-01-01', observations=observations)
