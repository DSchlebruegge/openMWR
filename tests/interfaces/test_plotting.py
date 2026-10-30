"""Analysis figures can be embedded while existing notebook calls still work."""

from unittest.mock import patch
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.quiver import Barbs
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openMWR.plot import SkewT, load_radiosonde_profile, plot_err, plot_scatter_lwp
from openMWR.evaluation import _profile_figure


def errors():
    return xr.Dataset(
        {
            "T": (("model", "height"), [[1.0, 2.0]]),
            "rh": (("model", "height"), [[3.0, 4.0]]),
        },
        coords={"model": ["test"], "height": [0.0, 1000.0]},
    )


def lwp():
    ds = xr.Dataset(
        {"lwp": (("station", "time"), [[1.0, 3.0]])},
        coords={"station": ["first"], "time": [0, 1]},
    )
    return ds.assign_coords(model="test"), ds


def test_supplied_figures_do_not_create_windows_or_display():
    profiles, scatter = Figure(), Figure()
    predicted, reference = lwp()
    with (
        patch("openMWR.plot.plt.figure") as create,
        patch("openMWR.plot.plt.show") as show,
    ):
        colors = plot_err(errors(), figure=profiles, show=False)
        plot_scatter_lwp(predicted, reference, figure=scatter, show=False)
    create.assert_not_called()
    show.assert_not_called()
    assert "test" in colors
    np.testing.assert_array_equal(profiles.axes[0].lines[0].get_xdata(), [1.0, 2.0])
    np.testing.assert_array_equal(profiles.axes[0].lines[0].get_ydata(), [0.0, 1.0])
    np.testing.assert_array_equal(
        scatter.axes[0].collections[0].get_offsets(), [[1.0, 1.0], [3.0, 3.0]]
    )


def test_notebook_calls_still_display_and_return_colors():
    predicted, reference = lwp()
    try:
        with patch("openMWR.plot.plt.show") as show:
            colors = plot_err(errors())
            assert isinstance(colors, dict)
            assert plot_scatter_lwp(predicted, reference) is None
        assert show.call_count == 2
    finally:
        plt.close("all")


def test_missing_lwp_still_produces_a_figure():
    predicted, reference = lwp()
    predicted["lwp"][:] = np.nan
    figure = Figure()
    plot_scatter_lwp(predicted, reference, figure=figure, show=False)
    assert len(figure.axes) == 2
    assert figure.axes[0].lines[0].get_xdata().size == 0


def test_report_error_axes_distinguish_signed_errors_from_nonnegative_errors():
    profiles = errors().assign(T=lambda ds: -ds.T)
    spectral = xr.Dataset({'TB': (('model', 'frq'), [[-1., 2.]])},
                          coords={'model': ['test'], 'frq': [22.24, 31.4]})
    for data, dimension in [(profiles, 'height'), (spectral, 'frq')]:
        for metric in ('RMSE', 'MAE', 'ME', 'Relative RMSE'):
            figure = _profile_figure(abs(data) if metric in ('RMSE', 'MAE') else data,
                                     metric, {'test': 'blue'})
            for ax in figure.axes:
                if metric in ('RMSE', 'MAE'):
                    assert len(ax.lines) == 1
                    assert (ax.get_xlim() if dimension == 'height' else ax.get_ylim())[0] == 0
                else:
                    assert len(ax.lines) == 2
                    zero_line = ax.lines[-1]
                    np.testing.assert_array_equal(
                        zero_line.get_xdata() if dimension == 'height' else zero_line.get_ydata(), [0, 0])


def test_radiosonde_selection_and_reusing_a_profile_without_file_reads(tmp_path):
    times = pd.to_datetime(['2024-01-01T00:00', '2024-01-01T12:00', '2024-01-02T12:00'])
    heights = np.tile([0., 500., 500., 1000., 2000.], (3, 1))
    raw = xr.Dataset({
        'height': (('synop', 'meastime'), heights),
        'T': (('synop', 'meastime'), 290 - heights * .006),
        'TD': (('synop', 'meastime'), 280 - heights * .006),
        'p': (('synop', 'meastime'), 950 * np.exp(-heights / 8000)),
        'wind_speed': (('synop', 'meastime'), np.full((3, 5), 5.)),
        'wind_dir': (('synop', 'meastime'), np.full((3, 5), 90.)),
        'starttime': ('synop', times + pd.Timedelta(minutes=5)),
        'endtime': ('synop', times + pd.Timedelta(minutes=50)),
    }, coords={'synop': times})
    directory = tmp_path / 'data/radiosondes/station_03715'
    directory.mkdir(parents=True)
    raw.to_netcdf(directory / 'raw_data_2024.nc')
    day = pd.Timestamp('2024-01-01T12:00')
    profile = load_radiosonde_profile(tmp_path, '03715', day)
    expected = raw.sel(synop=day).dropna(dim='meastime').swap_dims({'meastime': 'height'})
    expected = expected.sel(height=~expected.height.to_series().duplicated().values)
    xr.testing.assert_allclose(profile, expected)
    with pytest.raises(ValueError, match='No matching radiosondes'):
        load_radiosonde_profile(tmp_path, '03715', '2024-01-03')
    with patch('openMWR.plot.xr.open_dataset', side_effect=AssertionError('Unexpected file read')):
        for top in (100, 500):
            skew = SkewT(day, tmp_path, min_pressure=top)
            try:
                skew.plot_radiosonde('03715', data=profile)
                assert skew.radiosonde_plotted
                assert skew.retrieval_time == pd.Timestamp('2024-01-01T12:05')
                pressure = skew.ax.lines[0].get_ydata().to('hPa').magnitude
                np.testing.assert_array_equal(pressure, profile.p.values)
                skew.finalize()
                np.testing.assert_array_equal(skew.ax.get_ylim(), [1000, top])
            finally:
                plt.close(skew._fig)


@pytest.mark.parametrize(('station_alt', 'surface_p', 'limits'),
                         [(2656, 745.5, (800, 400)), (550, 963.7, (1000, 500))])
@pytest.mark.parametrize('zoomed', [False, True])
def test_skewt_layout_and_height_reference_are_independent_of_sounding(tmp_path, station_alt, surface_p, limits, zoomed):
    if not zoomed:
        limits = (1000, 100)
    day = pd.Timestamp('2024-01-01T12:05')
    heights = np.arange(0., 6001., 500.)
    local = xr.Dataset({
        'T': (('time', 'height'), (280 - .006 * heights)[None, :]),
        'TD': (('time', 'height'), (275 - .006 * heights)[None, :]),
        'p': (('time', 'height'), (surface_p - .06 * heights)[None, :]),
    }, coords={'time': [day], 'height': heights})
    sonde = xr.Dataset({
        'T': ('height', 290 - .006 * heights),
        'TD': ('height', 280 - .006 * heights),
        'p': ('height', 914 - .06 * heights),
        'wind_speed': ('height', np.full(heights.size, 5.)),
        'wind_dir': ('height', np.full(heights.size, 90.)),
        'starttime': day, 'endtime': day + pd.Timedelta(minutes=45),
    }, coords={'height': heights + 977, 'synop': day.normalize()})
    ticks = []
    layouts = []
    legend_positions = []
    dimensions = []
    for sounding in (False, True):
        skew = SkewT(day, tmp_path, station_alt=station_alt,
                     max_pressure=limits[0], min_pressure=limits[1], T_range=40 if zoomed else 70)
        try:
            if sounding:
                skew.plot_radiosonde('02290', data=sonde)
            label = 'MIM-Retrieval'
            if zoomed:
                label += ' (NN_opt_era5_no_met)\nPressure: UFS +1.50 hPa'
            skew.plot_retrieval(local, label)
            if zoomed:
                skew.plot_retrieval(local, 'MIM-Retrieval\nBoundary Layer Scan', color='purple')
            skew.finalize()
            skew._fig.canvas.draw()
            np.testing.assert_array_equal(skew.ax.get_ylim(), limits)
            if sounding:
                barbs, = (artist for artist in skew.ax.collections if isinstance(artist, Barbs))
                np.testing.assert_allclose(barbs.get_clip_box().intervaly, skew.ax.bbox.intervaly)
            positions = skew.ax.get_yticks(minor=True)
            labels = [tick.get_text().strip() for tick in skew.ax.get_yticklabels(minor=True)]
            assert positions[labels.index(f'{station_alt/1000:.2f}')] == pytest.approx(surface_p)
            for position, label in zip(positions, labels):
                height = station_alt if label == f'{station_alt/1000:.2f}' else float(label) * 1000
                assert position == pytest.approx(surface_p - .06 * (height - station_alt))
                assert station_alt <= height <= station_alt + heights[-1]
                assert limits[1] < position < limits[0]
            assert len(set(positions)) == len(positions)
            ticks.append(positions)
            legend = skew.ax.get_legend().get_window_extent()
            assert legend.x0 > skew.ax.bbox.x1
            assert legend.x1 < skew._fig.bbox.x1
            assert legend.y0 >= 0
            legend_positions.append((legend.x0, legend.y1))
            layouts.append(skew.ax.get_position().bounds)
            path = tmp_path / f'skew-{sounding}.svg'
            with plt.rc_context({'savefig.bbox': 'tight'}):
                skew.save(path)
            root = ET.parse(path).getroot()
            dimensions.append(tuple(root.get(key) for key in ('width', 'height', 'viewBox')))
        finally:
            plt.close(skew._fig)
    np.testing.assert_array_equal(*ticks)
    np.testing.assert_allclose(*layouts, rtol=0, atol=1e-12)
    np.testing.assert_allclose(*legend_positions, rtol=0, atol=1e-12)
    assert dimensions[0] == dimensions[1] == ('734.4pt', '561.6pt', '0 0 734.4 561.6')
