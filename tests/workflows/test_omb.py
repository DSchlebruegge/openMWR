"""OMB selection, portable reports and training's use of fitted physical shifts."""

import base64
from html import unescape
import json
from pathlib import Path
import re
from xml.etree import ElementTree

import numpy as np
import pytest
import xarray as xr

from openMWR.omb import Omb_Analysis, get_input_noise
from openMWR.normalization import normalized_input_noise
from openMWR.train import TrainingWorkflow
from tests.fixtures.cli import run_cli
from tests.fixtures.retrievals import retrieval_project


def test_report_command_preserves_selected_stations_angles_dates_and_existing_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr('openMWR.omb.utc_now', lambda: '2026-10-29T16:30:00.123456+00:00')
    retrieval_project(tmp_path, mode='bls', search=False)
    site_file = tmp_path / 'sites/test/site.toml'
    site_file.write_text(site_file.read_text() + '\ncalibration_dates = ["2024-01-12", "2025-03-12"]\n')
    path = tmp_path / 'data/sites/test/analysis_radiosonde/bls.nc'
    paired = xr.load_dataset(path)
    # Both stations and angles have distinct, exactly known offsets.
    offsets = xr.DataArray([2., -1.], dims='ang', coords={'ang': paired.ang})
    paired['TB'].loc[{'data_source': 'hatpro'}] += offsets
    second = paired.assign_coords(station=['other']).copy(deep=True)
    second['TB'].loc[{'data_source': 'hatpro'}] += 3
    paired = xr.concat([paired, second], dim='station')
    paired['lwp'].loc[{'data_source': 'hatpro', 'time': paired.time[1]}] = 50
    paired.to_netcdf(path)
    before = set(path.parent.iterdir())
    args = ('omb', 'test', 'analysis_radiosonde', '-p', str(tmp_path),
            '-B', '-S', '03715', 'other', '-s', '2024-01-11', '-e', '2024-01-22',
            '--days-of-month', '1', '20', '-l', '30')
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    out = Path(result.stdout.strip())
    assert out.parent == path.parent and out.name == 'omb_bls_2026-10-29_16-30-00.html'
    assert set(path.parent.iterdir()) == before | {out}
    html = out.read_text()
    tables = re.findall(r'<section><h2>(.*?)</h2>(<table.*?</table>)', html, re.S)
    assert len(tables) == 4
    for title, table in tables:
        expected = (2. if '90.0' in title else -1.) + (3. if 'other' in title else 0.)
        rows = ElementTree.fromstring(table).find('tbody').findall('tr')
        assert len(rows) == 2
        for row in rows:
            values = [cell.text for cell in row]
            # Jan 11 was excluded by LWP. Jan 12 remains. Jan 21–22 stay excluded.
            assert values[1] == '1' and float(values[2]) == expected and values[3] == '—'
    assert 'Monthly statistics and counts' in html and '2024-01-01' in html
    meta = json.loads(unescape(re.search(r'<pre>(.*?)</pre>', html, re.S)[1]))
    assert meta['selection']['days_of_month'] == {'first': 1, 'last': 20}
    assert meta['max_lwp'] == 30 and len(meta['data_id']) == 64
    assert meta['calibration_dates'] == ['2024-01-12', '2025-03-12']
    assert meta['training_input_noise'] == {}
    assert meta['frequency_bands'] == [[20., 40.], [50., 60.]]
    assert meta['regression']['predictors_ghz'] == {'22.24': [31.4], '31.4': [22.24]}
    assert 'Regression groups: 20–40, 50–60 GHz' in html
    assert 'Regression predictors [GHz]' in html
    assert html.count('<summary>22.24 GHz</summary>') == 4
    assert 'Monthly channel means and standard deviations.' in html
    assert 'in-sample regression residuals.' in html and 'empirical CDFs.' in html
    images = re.findall(r'<img src="([^"]+)"', html)
    assert images and all(source.startswith('data:image/svg+xml;base64,') for source in images)
    figures = [base64.b64decode(source.split(',', 1)[1]).decode('utf-8') for source in images]
    assert all('<svg' in figure and '<image' not in figure for figure in figures)
    assert 'omb.nc' not in html
    original = out.read_bytes()
    assert run_cli(*args, '--output', str(out)).returncode == 1
    assert out.read_bytes() == original
    missing = run_cli('omb', 'test', 'absent', '-p', str(tmp_path), '-o', str(tmp_path / 'missing.html'))
    assert missing.returncode == 1 and not (tmp_path / 'missing.html').exists()


def test_report_uses_paired_samples_calibration_markers_and_checkpoint_noise(tmp_path):
    from matplotlib import dates as mdates
    from openMWR.omb_report import _channel_figures, _training_noise

    retrieval_project(tmp_path, mode='bls', search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example', 'cpu')
    # Save a tiny checkpoint with known physical training noise, without fitting.
    workflow.model.nor = xr.Dataset()
    params = {**workflow.hyper_params, 'input_noise_TB_22.24': .5, 'input_noise_TB_31.4': 1.}
    workflow.model.save('test', tmp_path, params)
    checkpoint = tmp_path / 'sites/test/retrieval/example/model.pth'
    original = checkpoint.read_bytes()
    # Overlay settings come from the checkpoint, independently of a recipe edit.
    (checkpoint.parent / 'retrieval.toml').write_text('not a valid recipe')
    path = tmp_path / 'data/sites/test/analysis_radiosonde/bls.nc'
    paired = xr.load_dataset(path)
    paired['TB'].loc[{'frq': 31.4}] = np.nan
    paired.to_netcdf(path)
    analysis = Omb_Analysis('test', tmp_path, dataset='analysis_radiosonde', bls=True, angles=[30.])
    metadata_before = analysis.metadata()
    noise, records = _training_noise(analysis, ['example'])
    assert noise['example'].sel(frq=22.24, ang=30.) == 1.5
    assert len(records['example']['checkpoint_id']) == 64
    assert _training_noise(analysis, None) == ({}, {})
    out = tmp_path / 'noise.html'
    result = run_cli('omb', 'test', 'analysis_radiosonde', '-p', str(tmp_path),
                     '-B', '-a', '30', '-r', 'example', '-F', '20', '40', '-o', str(out))
    assert result.returncode == 0, result.stderr
    assert 'Training input noise: example' in out.read_text()
    assert '<summary>31.4 GHz</summary><p>No finite pairs.</p>' in out.read_text()
    metadata = json.loads(unescape(re.search(r'<pre>(.*?)</pre>', out.read_text(), re.S)[1]))
    assert metadata['training_input_noise'] == records
    assert metadata['frequency_bands'] == [[20., 40.]]
    # Python uses the same defaults, without adding fitted state to the caller.
    python_report = analysis.save(tmp_path / 'python.html')
    python_metadata = json.loads(unescape(re.search(r'<pre>(.*?)</pre>', python_report.read_text(), re.S)[1]))
    assert python_metadata['frequency_bands'] == [[20., 40.], [50., 60.]]
    assert not hasattr(analysis, 'resid_diff')
    assert checkpoint.read_bytes() == original and analysis.metadata() == metadata_before
    # Observed and simulated distributions use exactly the same finite pairs.
    observed = xr.DataArray([10., 20., np.nan, 40., 50., 60.], dims='time',
                            coords={'time': analysis.diff.time, 'frq': 22.24})
    simulated = observed.copy(data=[9., 19., 29., np.inf, 49., 59.])
    difference = (observed - simulated).where(lambda value: np.isfinite(value))
    figures = dict(_channel_figures(observed, simulated, difference, None, ['2024-01-11', '2025-03-12']))
    series, distributions = figures.values()
    axis = series.axes[0]
    data = next(line for line in axis.lines if line.get_label() == 'OMB')
    np.testing.assert_equal(data.get_ydata(), [1., 1., 1., 1.])
    marker = next(line for line in axis.lines if line.get_label() == 'Calibration')
    assert marker.get_xdata()[0] == np.datetime64('2024-01-11')
    assert axis.get_xlim()[1] < mdates.date2num(np.datetime64('2025-01-01'))
    scatter, histogram, cdf = distributions.axes
    np.testing.assert_equal(scatter.lines[0].get_xdata(), [9., 19., 49., 59.])
    np.testing.assert_equal(scatter.lines[0].get_ydata(), [10., 20., 50., 60.])
    for line, expected in zip(cdf.lines, ([10., 20., 50., 60.], [9., 19., 49., 59.])):
        np.testing.assert_equal(line.get_xdata()[1:-1], expected)
        np.testing.assert_equal(line.get_ydata(), [0, .25, .5, .75, 1., 1.])
    assert len(histogram.patches) == 2


def test_bias_uses_selected_channel_and_angle_offsets_without_mutating_diagnostics(tmp_path):
    recipe, _ = retrieval_project(tmp_path, mode='bls', search=False)
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example', 'cpu')
    path = tmp_path / 'data/sites/test/analysis_radiosonde/bls.nc'
    paired = xr.load_dataset(path)
    paired['TB'].loc[{'data_source': 'hatpro'}] += xr.DataArray([2., 5.], dims='ang', coords={'ang': paired.ang})
    paired.to_netcdf(path)
    original = workflow.training_data.TB.copy(deep=True)
    workflow.make_omb_analysis(['03715'], dataset='analysis_radiosonde', days_of_month={'first': 1, 'last': 20})
    workflow.omb.remove_explained_variance(frequency_bands=[[20., 40.]])
    means = workflow.omb.mean_diff.copy(deep=True)
    workflow.make_bias_correction(specific_freqs=[22.24], factor=1.)
    np.testing.assert_allclose((workflow.training_data.TB - original).sel(frq=22.24, ang=90), 2.)
    np.testing.assert_allclose((workflow.training_data.TB - original).sel(frq=22.24, ang=30), 5.)
    xr.testing.assert_equal(workflow.training_data.TB.sel(frq=31.4), original.sel(frq=31.4))
    xr.testing.assert_identical(workflow.omb.mean_diff, means)
    assert workflow.hyper_params['omb_bias_correction']['application'] == 'add_to_simulation'
    with pytest.raises(ValueError, match='already applied'):
        workflow.make_bias_correction(factor=1.)
    # Applying another filter retains the calendar selection and clears old fits.
    workflow.omb.filter_lwp(100.)
    assert workflow.omb.diff.time.dt.day.max() <= 20
    assert not hasattr(workflow.omb, 'resid_diff')
    # The same physical noise values drive training and public Python plots.
    workflow.hyper_params.update({'input_noise_TB_22.24': .5, 'input_noise_TB_31.4': 1.})
    workflow.model.hyper_params = workflow.hyper_params
    physical = get_input_noise(workflow.model)
    assert physical.TB.sel(frq=22.24, ang=90) == .5
    assert physical.TB.sel(frq=22.24, ang=30) == 1.5
    workflow.model.nor = xr.Dataset({'TB': xr.DataArray([2., 1.], dims='stat', coords={'stat': ['std', 'mean']})})
    normalized = normalized_input_noise(workflow.model, workflow.hyper_params)
    np.testing.assert_allclose(normalized, physical.TB.transpose('ang', 'frq').values.ravel() / 2)
    # The workflow keeps its selected grid. Bias correction must not modify the
    # source simulation, including channels unused by this model.
    recipe.write_text(recipe.read_text().replace('frequencies = [22.24, 31.4]', 'frequencies = [22.24]'))
    source = tmp_path / 'data/sites/test/training_variant/training_03715.nc'
    saved = source.read_bytes()
    subset = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example', 'cpu')
    original = subset.training_data.TB.copy(deep=True)
    np.testing.assert_array_equal(subset.training_data.frq, [22.24])
    subset.make_omb_analysis(['03715'], dataset='analysis_radiosonde')
    subset.make_bias_correction(factor=1.)
    np.testing.assert_allclose((subset.training_data.TB - original).sel(ang=90), 2.)
    np.testing.assert_allclose((subset.training_data.TB - original).sel(ang=30), 5.)
    assert source.read_bytes() == saved
    assert np.isfinite(subset.training_data.TB).all()


def test_omb_training_choices_are_explicit_and_persist_in_checkpoint(tmp_path):
    import tomllib
    import tomli_w
    from openMWR.models import load_model
    from openMWR.recipes import RetrievalRecipe

    recipe, _ = retrieval_project(tmp_path, search=False)
    paired_path = tmp_path / 'data/sites/test/analysis_radiosonde/zenith.nc'
    paired = xr.load_dataset(paired_path)
    paired['TB'].loc[{'data_source': 'hatpro'}] += xr.DataArray(np.arange(6.), dims='time', coords={'time': paired.time})
    paired.to_netcdf(paired_path)
    settings = tomllib.loads(recipe.read_text())
    grids_path = tmp_path / 'grids.toml'
    grids = tomllib.loads(grids_path.read_text())
    grids['frequencies']['bias_channels'] = [22.24]
    grids_path.write_text(tomli_w.dumps(grids))
    settings['input_noise'] = dict(method='omb', dataset='analysis_radiosonde',
                                   factor=1., statistic='residual_std', frequency_bands=[[20., 40.]])
    settings['bias_correction'] = dict(dataset='analysis_radiosonde', factor=1.,
                                       frequencies='bias_channels',
                                       days_of_month={'first': 1, 'last': 20})
    recipe.write_text(tomli_w.dumps(settings))
    workflow = TrainingWorkflow.from_recipe(tmp_path, 'test', 'example', 'cpu')
    assert workflow.hyper_params['input_noise_TB_22.24'] == pytest.approx(0., abs=1e-12)
    workflow.train()
    model = load_model('example', 'test', tmp_path)
    assert model.hyper_params['mean_bias'] == [1., 0.]
    assert model.hyper_params['omb_bias_correction']['specific_freqs'] == [22.24]
    assert model.hyper_params['omb_bias_correction']['data_source'] == 'radiosonde'
    assert model.hyper_params['omb_input_noise']['statistic'] == 'residual_std'
    assert model.hyper_params['omb_bias_correction']['selection']['days_of_month']['last'] == 20
    assert model.hyper_params['omb_bias_correction']['data_id'] != model.hyper_params['omb_input_noise']['data_id']
    for change, message in (({'statistic': None}, 'Missing input_noise settings: statistic'),
                            ({'frequency_bands': None}, 'Missing input_noise settings: frequency_bands'),
                            ({'statistic': 'unknown'}, 'statistic must'),
                            ({'frequency_bands': [[20., 30.], [25., 40.]]}, 'overlap')):
        invalid = tomllib.loads(recipe.read_text())
        invalid['input_noise'].update(change)
        for key, value in change.items():
            if value is None:
                invalid['input_noise'].pop(key)
        recipe.write_text(tomli_w.dumps(invalid))
        with pytest.raises(ValueError, match=message):
            RetrievalRecipe.load(tmp_path, 'test', 'example')
        recipe.write_text(tomli_w.dumps(settings))
