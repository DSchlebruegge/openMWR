"""Analysis figures can be embedded while existing notebook calls still work."""

from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import xarray as xr

from openMWR.plot import plot_err, plot_scatter_lwp


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
