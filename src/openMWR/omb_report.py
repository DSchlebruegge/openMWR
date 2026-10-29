"""Portable vector reports for the paired diagnostics in openMWR.omb."""

from copy import copy
from html import escape
from io import BytesIO
from itertools import product
import base64
import json
from pathlib import Path

from matplotlib import dates as mdates
from matplotlib.figure import Figure
import numpy as np

from openMWR.config import check_name, check_strings
from openMWR.models import load_model
from openMWR.omb import _regression_neighbors, get_input_noise, omb_statistics
from openMWR.paths import checkpoint_file
from openMWR.provenance import checkpoint_id, plain_value, software_identity, utc_now


def save_omb(analysis, output_file, *, retrievals=None, frequency_bands=None):
    """Write one self-contained HTML file, preserving existing reports."""
    output_file = Path(output_file).resolve()
    if output_file.exists():
        raise FileExistsError(f'OMB output already exists: {output_file}. Choose a new filename.')
    # Fit report diagnostics without changing the caller's analysis/training state.
    analysis = copy(analysis)
    if frequency_bands is not None or not hasattr(analysis, 'resid_diff'):
        analysis.remove_explained_variance(frequency_bands=frequency_bands if frequency_bands is not None
                                          else [[20., 40.], [50., 60.]])
    frequencies = analysis.diff.frq.values
    neighbors = _regression_neighbors(frequencies, analysis.frequency_bands)
    predictors = {str(float(frequency)): [float(frequencies[index]) for index in neighbors[target]]
                  for target, frequency in enumerate(frequencies)}
    noise, checkpoints = _training_noise(analysis, retrievals)
    calibrations = sorted(analysis.site_config.get('calibration_dates', []))
    metadata = {**analysis.metadata(), 'created_utc': utc_now(), 'software': software_identity(),
                'calibration_dates': calibrations, 'training_input_noise': checkpoints,
                'regression': {'method': 'Ordinary least squares with intercept on immediate neighbors within each band',
                               'predictors_ghz': predictors}}
    statistics = analysis.statistics.assign(residual_std=analysis.resid_std, residual_count=analysis.resid_count)
    residual = analysis.resid_diff
    monthly = analysis.diff.resample(time='MS').map(omb_statistics)
    sections = []
    dimensions = [dim for dim in ('station', 'ang') if dim in analysis.diff.dims]
    combinations = product(*(analysis.diff[dim].values for dim in dimensions))
    for values in combinations:
        selection = dict(zip(dimensions, values))
        title = ', '.join(f'{key}: {value}' for key, value in selection.items()) or 'Zenith'
        diff = analysis.diff.sel(selection, drop=True)
        stats = statistics.sel(selection, drop=True)
        months = monthly.sel(selection, drop=True)
        if not bool(stats['count'].any()):
            sections.append(f'<section><h2>{escape(title)}</h2><p>No finite pairs.</p></section>')
            continue
        angle = selection.get('ang', 90.)
        overlays = {name: value.sel(ang=angle) for name, value in noise.items() if angle in value.ang}
        images = ''.join(_image(figure, caption) for caption, figure in
                         _overview_figures(diff, stats, months, calibrations, overlays))
        channels = []
        for frequency in diff.frq.values:
            difference = diff.sel(frq=frequency)
            if not bool(difference.notnull().any()):
                channels.append(f'<details><summary>{frequency:g} GHz</summary><p>No finite pairs.</p></details>')
                continue
            observed = analysis.observation.TB.sel(selection, drop=True).sel(frq=frequency)
            simulated = analysis.background.TB.sel(selection, drop=True).sel(frq=frequency)
            unexplained = residual.sel(selection, drop=True).sel(frq=frequency)
            plots = ''.join(_image(figure, caption) for caption, figure in
                            _channel_figures(observed, simulated, difference, unexplained, calibrations))
            channels.append(f'<details><summary>{frequency:g} GHz</summary>{plots}</details>')
        summary_vars = ['count', 'mean', 'std', 'rmse', 'residual_count', 'residual_std']
        frame = stats[summary_vars].to_dataframe()[summary_vars]
        frame['Regression predictors [GHz]'] = [', '.join(f'{value:g}' for value in predictors[str(float(frequency))])
                                              or 'None (centered OMB)' for frequency in diff.frq.values]
        table = frame.to_html(float_format=lambda x: f'{x:.4g}', na_rep='—', border=0)
        month_table = months[['count', 'mean', 'std', 'rmse']].to_dataframe().to_html(float_format=lambda x: f'{x:.4g}', na_rep='—', border=0)
        sections.append(f'<section><h2>{escape(title)}</h2>{table}'
                        f'<p>Common pairs for covariance: {int(stats.covariance_count)}.</p>'
                        + images + '<h3>Individual channels</h3>' + ''.join(channels)
                        + '<details><summary>Monthly statistics and counts</summary>' + month_table + '</details></section>')
    calibration_note = ('<p>Calibration dates (UTC): ' + ', '.join(calibrations)
                        + '. Dashed lines mark dates within the selected period.</p>') if calibrations else ''
    noise_note = ('<p>Training input noise: ' + ', '.join(escape(name) for name in noise)
                  + '. Overlays use saved checkpoint settings at matching frequencies and viewing angles. '
                    'This is training augmentation, not measured instrument noise.</p>') if noise else ''
    bands = ', '.join(f'{low:g}–{high:g}' for low, high in analysis.frequency_bands)
    document = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>OMB: {escape(analysis.site)} / {escape(analysis.dataset)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}}
img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:.35em .7em;border-bottom:1px solid #ddd}}
pre{{white-space:pre-wrap}}details{{margin:1em 0}}summary{{cursor:pointer}}figure{{margin:1em 0}}
figcaption{{color:#444}}section{{margin:2em 0}}</style>
<h1>OMB: {escape(analysis.site)} / {escape(analysis.dataset)}</h1>
<p>Observed minus simulated brightness temperature. Mean, standard deviation and RMSE are in K.
Counts use finite observation/reference pairs per channel. Covariance uses common finite pairs
across channels. Standard deviations use ddof=0. Missing statistics appear as —.</p>
<p>OMB includes instrument, background-profile, radiative-transfer and sampling errors.
Unexplained regression scatter is experimental and does not isolate instrument noise.
Low LWP is a scene selection and does not independently establish clear skies.</p>
<p>Regression groups: {bands} GHz. Each channel uses only its nearest selected lower and upper
frequency within its group. Edge channels use one neighbor. The tables list the predictors.
A channel alone in its group uses centered OMB. Fits without residual degrees of freedom
are unavailable. Residuals are in-sample and are shown below the individual OMB time series.</p>
<p>All figures are vector graphics embedded in this HTML file, which can be shared on its own.
Daily and monthly summaries show changes in the mismatch. They are not a noise estimate.
Expand individual channels for all paired samples, with no thinning. Their histograms and empirical
CDFs compare distributions on the same finite pairs. Similar distributions do not imply agreement
of individual observation/reference pairs.</p>
{calibration_note}{noise_note}{''.join(sections)}
<details><summary>Settings and provenance</summary><pre>{escape(json.dumps(metadata, indent=2))}</pre></details>
</html>'''
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('x', encoding='utf-8') as stream:
        stream.write(document)
    return output_file


def _training_noise(analysis, retrievals):
    """Read physical TB noise on the CPU, without consulting mutable recipes."""
    if retrievals is None:
        return {}, {}
    check_strings(retrievals, 'OMB retrievals')
    noise, records = {}, {}
    angles = analysis.diff.ang.values if 'ang' in analysis.diff.dims else [90.]
    for name in retrievals:
        check_name(name, 'OMB retrieval')
        model = load_model(name, analysis.site, analysis.project_dir)
        try:
            value = get_input_noise(model).TB
        except KeyError as exc:
            raise ValueError(f'Retrieval {name!r} has no saved training noise setting {exc}.') from exc
        if not bool(np.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f'Retrieval {name!r} must have finite, nonnegative TB training noise.')
        if 'ang' not in value.dims:
            value = value.expand_dims(ang=[90.])
        if not np.isin(analysis.diff.frq, value.frq).any() or not np.isin(angles, value.ang).any():
            raise ValueError(f'Retrieval {name!r} has no matching OMB frequencies/viewing angles.')
        records[name] = {'checkpoint_id': checkpoint_id(checkpoint_file(analysis.project_dir, analysis.site, name)),
                         'TB': plain_value(value.to_dict())}
        noise[name] = value.reindex(frq=analysis.diff.frq)
    return noise, records


def _image(figure, caption):
    stream = BytesIO()
    figure.savefig(stream, format='svg', bbox_inches='tight')
    encoded = base64.b64encode(stream.getvalue()).decode('ascii')
    figure.clear()
    text = escape(caption)
    return f'<figure><img src="data:image/svg+xml;base64,{encoded}" alt="{text}"><figcaption>{text}</figcaption></figure>'


def _time_axis(axis, times, calibrations):
    """Show UTC dates and in-period events without extending the plotted period."""
    first = times.min().astype('datetime64[D]')
    last = times.max().astype('datetime64[D]') + np.timedelta64(1, 'D')
    for value in calibrations:
        if first <= np.datetime64(value) < last:
            axis.axvline(np.datetime64(value), color='black', linestyle='--', linewidth=.9,
                         label='Calibration' if not any(line.get_label() == 'Calibration' for line in axis.lines) else '_nolegend_')
    locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.set_xlabel('UTC date')
    axis.set_xlim(first, last)
    axis.grid(alpha=.3)


def _overview_figures(difference, statistics, monthly, calibrations, noise):
    labels = [f'{value:g}' for value in difference.frq.values]
    channels = np.arange(len(labels))
    figure = Figure(figsize=(11, 6), layout='constrained')
    axes = figure.subplots(2, 1, sharex=True)
    axes[0].plot(channels, statistics['mean'], 'o-')
    axes[0].axhline(0, color='black', linewidth=.8)
    axes[0].set_ylabel('Mean OMB [K]')
    for name, label in (('std', 'OMB standard deviation'), ('rmse', 'OMB RMSE'),
                        ('residual_std', 'Unexplained OMB standard deviation')):
        if name in statistics:
            axes[1].plot(channels, statistics[name], 'o-', label=label)
    for name, value in noise.items():
        axes[1].plot(channels, value, 'x--', label=f'Training noise: {name}')
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel('Scatter / RMSE [K]')
    axes[1].set_xticks(channels, labels, rotation=45)
    axes[1].set_xlabel('Frequency [GHz]')
    axes[1].legend(fontsize='small')
    for axis in axes:
        axis.grid(alpha=.3)
    yield 'Channel means, scatter and RMSE.', figure

    figure = Figure(figsize=(11, 5), layout='constrained')
    axis = figure.subplots()
    daily = difference.resample(time='1D').mean()
    for frequency in difference.frq.values:
        axis.plot(daily.time.values, daily.sel(frq=frequency), label=f'{frequency:g} GHz')
    axis.axhline(0, color='black', linewidth=.8)
    axis.set_ylabel('Daily mean OMB [K]')
    _time_axis(axis, difference.time.values, calibrations)
    axis.legend(ncol=4, fontsize='small')
    yield 'Daily mean OMB by channel. Missing days remain gaps.', figure

    figure = Figure(figsize=(11, 8), layout='constrained')
    axes = figure.subplots(2, 1, sharex=True)
    edges = np.append(monthly.time.values.astype('datetime64[M]'),
                      monthly.time.values[-1].astype('datetime64[M]') + np.timedelta64(1, 'M'))
    mean = monthly['mean'].values
    bound = max(float(np.nanmax(np.abs(mean))), .01)
    std = monthly['std'].values
    std_bound = max(float(std[np.isfinite(std)].max()), .01) if np.isfinite(std).any() else .01
    for axis, name, title, settings in zip(axes, ('mean', 'std'), ('Monthly mean OMB [K]', 'Monthly OMB std [K]'),
                                         ({'vmin': -bound, 'vmax': bound, 'cmap': 'RdBu_r'},
                                          {'vmin': 0, 'vmax': std_bound, 'cmap': 'viridis'})):
        mesh = axis.pcolormesh(mdates.date2num(edges), np.arange(len(labels) + 1),
                               monthly[name].transpose('frq', 'time'), **settings)
        colorbar = figure.colorbar(mesh, ax=axis, label=title)
        colorbar.solids.set_rasterized(False)
        axis.set_yticks(channels + .5, labels)
        axis.set_ylabel('Frequency [GHz]')
        _time_axis(axis, difference.time.values, calibrations)
        # Keep complete calendar-month cells visible, including partial end months.
        axis.set_xlim(edges[0].astype('datetime64[D]'), edges[-1].astype('datetime64[D]'))
    axes[0].set_xlabel('')
    yield 'Monthly channel means and standard deviations. Empty cells have no statistic. Counts are in the monthly table.', figure

    figure = Figure(figsize=(7, 6), layout='constrained')
    axis = figure.subplots()
    mesh = axis.pcolormesh(statistics.correlation.transpose('frq', 'frq2'), vmin=-1, vmax=1, cmap='RdBu_r')
    axis.set_xticks(channels + .5, labels, rotation=45)
    axis.set_yticks(channels + .5, labels)
    axis.set_xlabel('Frequency [GHz]')
    axis.set_ylabel('Frequency [GHz]')
    colorbar = figure.colorbar(mesh, ax=axis, label='OMB correlation')
    colorbar.solids.set_rasterized(False)
    yield 'Signed OMB correlations on common finite pairs across channels.', figure


def _channel_figures(observed, simulated, difference, residual, calibrations):
    """Full paired samples for one channel, station and viewing angle."""
    valid = np.isfinite(difference.values)
    obs, sim = observed.values[valid], simulated.values[valid]
    times = difference.time.values
    title = f'{float(difference.frq):g} GHz'
    figure = Figure(figsize=(11, 7 if residual is not None else 4), layout='constrained')
    axes = figure.subplots(2 if residual is not None else 1, 1, sharex=True, squeeze=False)[:, 0]
    axes[0].plot(times[valid], difference.values[valid], '.', markersize=3, label='OMB')
    axes[0].set_ylabel('OMB [K]')
    if residual is not None:
        finite = np.isfinite(residual.values)
        axes[1].plot(times[finite], residual.values[finite], '.', markersize=3, label='Unexplained OMB')
        axes[1].set_ylabel('Unexplained OMB [K]')
    for axis in axes:
        axis.axhline(0, color='black', linewidth=.8)
        _time_axis(axis, times, calibrations)
        axis.legend(fontsize='small')
    for axis in axes[:-1]:
        axis.set_xlabel('')
    figure.suptitle(title)
    yield f'{title}: individual OMB samples' + (' and in-sample regression residuals.' if residual is not None else '.'), figure

    figure = Figure(figsize=(12, 4), layout='constrained')
    scatter, histogram, cdf = figure.subplots(1, 3)
    combined = np.concatenate((obs, sim))
    lower, upper = combined.min(), combined.max()
    padding = max(float(upper - lower) * .05, .1)
    limits = (lower - padding, upper + padding)
    scatter.plot(sim, obs, '.', markersize=3)
    scatter.plot(limits, limits, color='black', linewidth=.8, label='1:1')
    scatter.set(xlim=limits, ylim=limits, xlabel='Simulated TB [K]', ylabel='Observed TB [K]')
    scatter.set_aspect('equal', adjustable='box')
    scatter.legend(fontsize='small')
    # Shared, equal-width bins in kelvin, with no pooling across channels.
    bins = np.linspace(*limits, 31)
    for values, label in ((obs, 'Observed'), (sim, 'Simulated')):
        histogram.hist(values, bins=bins, histtype='step', density=True, label=label)
        # Start at zero before the smallest value, including singleton samples.
        cdf.step(np.r_[limits[0], np.sort(values), limits[1]], np.r_[0., np.arange(1, len(values) + 1) / len(values), 1.],
                 where='post', label=label)
    histogram.set(xlabel='TB [K]', ylabel='Probability density [1/K]', xlim=limits, ylim=(0, None))
    cdf.set(xlabel='TB [K]', ylabel='Empirical cumulative probability', xlim=limits, ylim=(0, 1))
    for axis in (scatter, histogram, cdf):
        axis.grid(alpha=.3)
    histogram.legend(fontsize='small')
    cdf.legend(fontsize='small')
    figure.suptitle(f'{title}, {len(obs)} finite pairs')
    yield f'{title}: observed vs simulated TB, paired-sample histograms and empirical CDFs.', figure
