"""Real checkpoints and small RPG files through Python and module prediction."""

from datetime import datetime, timedelta, timezone
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from openMWR.hatpro_data import calc_derived_quantities
from openMWR.models import Model, SpectralConsistencyModel
from openMWR.predict import PredictionWorkflow, load_predictions, run_model_list
from openMWR.site import create_site

FREQUENCIES = np.array([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.4,
                        51.26, 52.28, 53.86, 54.94, 56.66, 57.3, 58.0])
ANGLES = np.array([90.0, 30.0, 19.2, 14.4, 11.4, 8.4, 6.6, 5.4, 4.8, 4.2, 0.0])


def raw_day(project, date, *, bls=True):
    directory = project / 'raw' / date.strftime('Y%Y/M%m/D%d')
    directory.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(date, periods=12, freq='30s')
    sample = np.arange(len(times), dtype=float)
    metadata = {'radiometer_software_version': 'test'}
    brt = xr.Dataset({
        'TBs': (('time', 'number_frequencies'), 100 + sample[:, None] + np.zeros((12, 14))),
        'frequencies': ('number_frequencies', FREQUENCIES),
        'rain_flag': ('time', sample % 3 == 0),
        'azimuth_angle': ('time', sample * 0),
        'elevation_angle': ('time', sample * 0 + 90),
    }, coords={'time': times}, attrs=metadata)
    met = xr.Dataset({
        'env_pressure': ('time', 950 + sample / 10),
        'env_temperature': ('time', 290 + sample / 10),
        'env_relative_humidity': ('time', 50 + sample / 10),
    }, coords={'time': times})
    irt = xr.Dataset({'IRR_data': (('time', 'number_wavelength'), -20 + sample[:, None])},
                     coords={'time': times})
    files = {'BRT': brt, 'MET': met, 'IRT': irt}
    if bls:
        scan_times = times[::2]
        files['BLB'] = xr.Dataset({
            'TBs': (('time', 'number_scan_angles', 'number_frequencies'),
                    100 + np.arange(6)[:, None, None] + np.zeros((6, 11, 14))),
            'elevation_scan_angles': ('number_scan_angles', ANGLES),
            'frequencies': ('number_frequencies', FREQUENCIES),
            'rain_flag': ('time', np.zeros(6)),
            'azimuth_angle': ('time', np.zeros(6)),
        }, coords={'time': scan_times}, attrs=metadata)
    for suffix, data in files.items():
        data.to_netcdf(directory / f'{date:%y%m%d}.{suffix}.NC')


def checkpoint(project, name, *, bls=False, spectral=False):
    torch.manual_seed(2)
    if spectral:
        model = SpectralConsistencyModel(name, FREQUENCIES[:2], hidden_layer_size=3)
    else:
        model = Model(name, np.array([0., 1000., 2000.]), FREQUENCIES[:2],
            is_bls_model=bls, is_MLR=True, angles=[90., 30.] if bls else None,
            additional_input_vars=['surface_T', 'surface_rh', 'surface_p', 'doy_cos', 'doy_sin'])
        with torch.no_grad():
            model.stack.weight.fill_(0.0001)
            model.stack.bias.copy_(torch.tensor([290., 283., 276., 50., 50., 50., .01, .01, .01, 20., 10.]))
    specification = xr.merge([model.input, model.output])
    model.nor = xr.Dataset({name: xr.ones_like(values).expand_dims(stat=['mean', 'std', 'max'])
                            for name, values in specification.items()})
    model.save('test', project, {})


@pytest.fixture
def project(tmp_path):
    create_site('test', tmp_path, gen='G5', hatpro_data_dir='raw',
                retrieval_output_dir='published', mwr_height=550,
                prediction={'retrievals': ['primary', 'scan']})
    checkpoint(tmp_path, 'primary')
    checkpoint(tmp_path, 'scan', bls=True)
    raw_day(tmp_path, datetime(2024, 1, 1))
    return tmp_path


def command(project, *args):
    return subprocess.run([sys.executable, '-m', 'openMWR', 'predict', 'test',
        *args, '-p', str(project)],
        cwd=project.parent, text=True, capture_output=True)


def test_python_command_and_previous_profile_pipeline_agree(project):
    workflow = PredictionWorkflow(project, 'test')
    result = workflow.predict('2024-01-01')
    assert not (project / 'published').exists()  # Python prediction has no writes.
    assert not result.errors
    for name, observations, bls in (('zenith', result.observations, False),
                                     ('bls', result.observations_bls, True)):
        models = [model for model in workflow.models if model.is_bls_model == bls]
        expected = calc_derived_quantities(run_model_list(models, observations, 'cpu'))
        expected['rain_flag'] = observations.rain_flag.sel(time=expected.time)
        if not bls:
            expected = expected.rolling(time=5, center=True, min_periods=1).mean()
        xr.testing.assert_allclose(getattr(result, name), expected)
    assert result.zenith.sizes['time'] == 6
    assert result.bls.T.attrs['units'] == 'K'
    paths = workflow.save(result)
    assert [path.name for path in paths] == ['20240101_retrieval.nc', '20240101_retrieval_bls.nc']
    assert 'model' not in xr.load_dataset(paths[1]).dims  # Existing published BLS shape.
    cli = command(project, '-d', '2024-01-01', '-D', 'cpu')
    assert cli.returncode == 0, cli.stderr
    loaded = load_predictions(project, 'test', '2024-01-01')
    for name in ('zenith', 'bls'):
        expected = getattr(result, name).drop_vars(['doy_sin', 'doy_cos'])
        xr.testing.assert_identical(getattr(loaded, name), expected)
    assert loaded.observations is None
    assert not list((project / 'data/sites/test').iterdir())  # No prepared dataset generation.


def test_range_continues_after_missing_day_and_reports_partial_bls(project):
    raw_day(project, datetime(2024, 1, 3), bls=False)
    cli = command(project, '-s', '2024-01-01', '-e', '2024-01-03')
    assert cli.returncode == 1
    assert '2024-01-02' in cli.stderr and 'bls' in cli.stderr
    assert 'Prediction incomplete' in cli.stderr
    directory = project / 'published/2024/01'
    assert {path.name for path in directory.iterdir()} == {
        '20240101_retrieval.nc', '20240101_retrieval_bls.nc', '20240103_retrieval.nc'}
    partial = PredictionWorkflow(project, 'test').predict('2024-01-03')
    assert partial.zenith is not None and partial.bls is None
    assert set(partial.errors) == {'bls'}
    workflow = PredictionWorkflow(project, 'test')
    missing = workflow.predict('2024-01-02')
    assert missing.zenith is None and missing.bls is None
    assert set(missing.errors) == {'zenith', 'bls'}
    assert workflow.save(missing) == []
    assert not (directory / '20240102_retrieval.nc').exists()


def test_computation_errors_are_not_reported_as_missing_data(project, monkeypatch):
    workflow = PredictionWorkflow(project, 'test', ['primary'])

    def broken_forward(inputs):
        raise ValueError('Inference failed')

    monkeypatch.setattr(workflow.models[0], 'forward', broken_forward)
    with pytest.raises(ValueError, match='Inference failed'):
        workflow.predict('2024-01-01')


def test_override_paths_replacement_model_reuse_and_reading_without_checkpoints(project):
    raw_day(project, datetime(2024, 1, 2), bls=False)
    workflow = PredictionWorkflow(project, 'test', ['primary'], output_dir='experiment')
    first = workflow.predict('2024-01-01')
    paths = workflow.save(first)
    assert paths[0] == project / 'experiment/2024/01/20240101_retrieval.nc'
    previous = paths[0].read_bytes()
    absolute = project.parent / (project.name + '-absolute')
    other = PredictionWorkflow(project, 'test', ['primary'], output_dir=absolute)
    assert other.save(first)[0].is_relative_to(absolute)
    for path in project.glob('sites/test/retrieval/*/model.pth'):
        path.unlink()
    second = workflow.predict('2024-01-02')  # Reuses loaded models.
    workflow.save(second)
    workflow.models[0].stack.bias.data.add_(1)
    workflow.save(workflow.predict('2024-01-02'))
    assert paths[0].read_bytes() == previous
    loaded = load_predictions(project, 'test', '2024-01-02', output_dir='experiment')
    assert loaded.bls is None
    assert not loaded.zenith.T.identical(second.zenith.T)


def test_bls_only_and_multiple_bls_models(project):
    (project / 'raw/Y2024/M01/D01/240101.BRT.NC').unlink()
    checkpoint(project, 'scan2', bls=True)
    workflow = PredictionWorkflow(project, 'test', ['scan', 'scan2'])
    result = workflow.predict('2024-01-01')
    assert result.zenith is None and not result.errors
    path, = workflow.save(result)
    assert xr.load_dataset(path).model.values.tolist() == ['scan', 'scan2']
    cli = command(project, 'scan', 'scan2', '--date', '2024-01-01')
    assert cli.returncode == 0, cli.stderr
    assert xr.load_dataset(path).model.values.tolist() == ['scan', 'scan2']


def test_spectral_prediction_and_unsmoothed_python_override(project):
    checkpoint(project, 'spectral', spectral=True)
    workflow = PredictionWorkflow(project, 'test', ['spectral'], smooth=False)
    result = workflow.predict(datetime(2024, 1, 1, 2, tzinfo=timezone(timedelta(hours=2))))
    assert result.date == datetime(2024, 1, 1)
    expected = run_model_list(workflow.models, result.observations, 'cpu')
    xr.testing.assert_allclose(result.zenith.TB, expected.TB)
    assert 'T' not in result.zenith
    assert command(project, 'spectral', '--date', '2024-01-01').returncode == 0


@pytest.mark.parametrize('args', [
    [], ['--date', 'invalid'], ['--start-date', '2024-01-01'],
    ['--date', '2024-01-01', '--end-date', '2024-01-02'],
    ['--start-date', '2024-01-02', '--end-date', '2024-01-01'],
])
def test_invalid_dates_fail_before_loading_or_writing(tmp_path, args):
    result = command(tmp_path, *args)
    assert result.returncode == 2
    assert not list(tmp_path.iterdir())


def test_configuration_and_checkpoint_selection_before_writing(project):
    config = project / 'sites/test/site.toml'
    config.write_text(config.read_text().replace('[prediction]', '[unused]'))
    with pytest.raises(ValueError, match='Select retrieval names'):
        PredictionWorkflow(project, 'test')
    with pytest.raises(ValueError, match='only once'):
        PredictionWorkflow(project, 'test', ['primary', 'primary'])
    with pytest.raises(FileNotFoundError):
        PredictionWorkflow(project, 'test', ['missing'])
    # A broken training recipe is irrelevant for saved-model inference.
    (project / 'sites/test/retrieval/primary/retrieval.toml').write_text('invalid = [')
    assert command(project, 'primary', '--date', '2024-01-01',
                   '-o', 'chosen').returncode == 0
    assert (project / 'chosen/2024/01/20240101_retrieval.nc').exists()
    assert not (project / 'published').exists()


def test_default_output_and_missing_saved_predictions(project):
    config = project / 'sites/test/site.toml'
    config.write_text(config.read_text().replace('retrieval_output_dir = "published"\n', ''))
    workflow = PredictionWorkflow(project, 'test', ['primary'])
    path, = workflow.save(workflow.predict('2024-01-01'))
    assert path == project / 'data/sites/test/predictions/2024/01/20240101_retrieval.nc'
    with pytest.raises(FileNotFoundError, match='No saved predictions'):
        load_predictions(project, 'test', '2024-01-02')
