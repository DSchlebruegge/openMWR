"""Observed minus simulated TB diagnostics, in physical brightness temperatures."""

from datetime import datetime
import logging
from pathlib import Path
import tomllib

import numpy as np
import xarray as xr

from openMWR.config import check_coordinates, check_number, check_time_selection, select_time
from openMWR.observations import load_analysis_dataset
from openMWR.paths import dataset_dir, dataset_recipe_file
from openMWR.provenance import dataset_id, plain_value, utc_now
from openMWR.site import load_site_config

logger = logging.getLogger(__name__)


def omb_statistics(difference, *, sample_dims=('time',)):
    """Summarize finite O−B pairs, retaining station and viewing-angle axes.

    Channel moments use each channel's finite pairs. Covariance and correlation
    use common finite pairs across the selected channels. Population moments
    (ddof=0) give RMSE² = bias² + std². At least two pairs are required for
    standard deviations and covariances. Constant channels have undefined
    correlation. Standard deviations of residual fits describe in-sample scatter.
    """
    if 'frq' not in difference.dims or not set(sample_dims) <= set(difference.dims):
        raise ValueError('OMB statistics require frequency and sample dimensions.')
    difference = difference.where(np.isfinite(difference))
    count = difference.count(sample_dims)
    mean = difference.mean(sample_dims)
    centered = difference - mean
    std = np.sqrt((centered ** 2).mean(sample_dims)).where(count >= 2)
    common = difference.notnull().all('frq')
    common_count = common.sum(sample_dims)
    complete = difference.where(common)
    anomalies = (complete - complete.mean(sample_dims)).fillna(0)
    covariance = xr.dot(anomalies, anomalies.rename(frq='frq2'), dim=sample_dims) / common_count.where(common_count >= 2)
    scale = np.sqrt((anomalies ** 2).sum(sample_dims) / common_count.where(common_count >= 2))
    denominator = scale * scale.rename(frq='frq2')
    correlation = covariance / denominator.where(denominator > 0)
    result = xr.Dataset(dict(count=count, mean=mean, std=std,
                             rmse=np.sqrt((difference ** 2).mean(sample_dims)),
                             covariance_count=common_count,
                             covariance=covariance, correlation=correlation.clip(-1, 1)))
    for name in ('mean', 'std', 'rmse'):
        result[name].attrs['units'] = 'K'
    result.covariance.attrs['units'] = 'K2'
    result.correlation.attrs['units'] = '1'
    result.attrs.update(convention='observed minus simulated TB', ddof=0,
                        covariance_samples='common finite pairs across selected frequencies')
    return result


def check_frequency_bands(bands):
    """Validate explicit, non-overlapping regression bands in GHz."""
    if not isinstance(bands, list) or not bands:
        raise ValueError('frequency_bands must contain [lower, upper] bounds in GHz.')
    for band in bands:
        if not isinstance(band, list) or len(band) != 2:
            raise ValueError('Each frequency_bands entry must be [lower, upper] in GHz.')
        for value in band:
            check_number(value, 'frequency_bands bound', 0)
        if band[0] >= band[1]:
            raise ValueError('frequency_bands bounds must be increasing.')
    ordered = sorted(bands)
    if any(left[1] >= right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('frequency_bands must not overlap.')


def _regression_neighbors(frequencies, frequency_bands):
    """Select immediate neighbors by frequency without crossing band boundaries."""
    check_frequency_bands(frequency_bands)
    frequencies = np.asarray(frequencies)
    membership = [np.flatnonzero((frequencies >= low) & (frequencies <= high)) for low, high in frequency_bands]
    if set(np.concatenate(membership)) != set(range(len(frequencies))):
        raise ValueError('frequency_bands must cover every selected frequency.')
    neighbors = {}
    for group in membership:
        group = group[np.argsort(frequencies[group])]
        for position, target in enumerate(group):
            neighbors[target] = group[max(0, position - 1):position].tolist() + group[position + 1:position + 2].tolist()
    return neighbors


def unexplained_omb(difference, *, frequency_bands):
    """Regress OMB on nearest lower/upper frequencies within explicit bands.

    Fit an intercept and slopes in kelvin on finite target/predictor pairs.
    Station and angle axes are fitted separately. A lone channel uses centered
    OMB. Least squares handles rank-deficient predictors. Fits with no residual
    degrees of freedom are unavailable. These residuals are unexplained OMB
    scatter, not an instrument-noise measurement.
    """
    frequencies = np.asarray(difference.frq.values)
    neighbors = _regression_neighbors(frequencies, frequency_bands)
    other = [dim for dim in difference.dims if dim not in ('time', 'frq')]
    ordered = difference.transpose(*other, 'time', 'frq')
    values = np.asarray(ordered.values, dtype=float)
    residual = np.full(values.shape, np.nan)
    counts = np.zeros(values.shape[:-2] + (len(frequencies),), dtype=int)
    for target, predictors in neighbors.items():
        for index in np.ndindex(values.shape[:-2]):
            y = values[index][:, target]
            x = values[index][:, predictors]
            valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
            y, x = y[valid], x[valid]
            if not len(y):
                continue
            # Center on this fit's finite sample, not channel-wide means.
            x, y = x - x.mean(axis=0), y - y.mean()
            beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
            if len(y) <= rank + 1:
                continue
            residual[index][valid, target] = y - x @ beta
            counts[index + (target,)] = len(y)
    result = xr.DataArray(residual, dims=ordered.dims, coords=ordered.coords).transpose(*difference.dims)
    result.attrs.update(units='K', description='Unexplained OMB after neighboring-channel regression')
    count = xr.DataArray(counts, dims=(*other, 'frq'), coords={dim: ordered[dim] for dim in (*other, 'frq')})
    return result, count


class Omb_Analysis:
    """Inspect paired analysis data without running a retrieval.

    Retains all selected stations and BLS angles. Filters apply to the paired
    dataset before differences. Optional ``max_lwp`` selects low-LWP scenes in
    both sources, in g/m². It is not independent evidence of cloud-free skies.
    """

    def __init__(self, site, project_dir, data_source=None, station_ids_for_analysis=None,
                 load_existing_analysis_dataset=True, dataset=None, *, bls=False,
                 start_date=None, end_date=None, days_of_month=None, max_lwp=None,
                 frequencies=None, angles=None, site_config=None):
        if dataset is None:
            raise ValueError('Select a named analysis dataset for OMB.')
        self.site, self.project_dir, self.dataset = site, Path(project_dir), dataset
        self.site_config = load_site_config(site, project_dir, config=site_config)
        self.bls = bls
        self.station_ids_for_analysis = station_ids_for_analysis
        self.selection = {key: value for key, value in dict(start_date=start_date, end_date=end_date,
                          days_of_month=days_of_month).items() if value is not None}
        check_time_selection(self.selection, 'OMB selection')
        if not load_existing_analysis_dataset:
            from openMWR.datasets import DatasetWorkflow
            DatasetWorkflow(project_dir, site).build(dataset)
        data = load_analysis_dataset(project_dir, site, dataset, bls=bls, site_config=self.site_config)
        references = [str(value) for value in data.data_source.values if value != 'hatpro']
        if data_source is None:
            if len(references) != 1:
                raise ValueError('Select data_source when an analysis contains multiple references.')
            data_source = references[0]
        if data_source not in references:
            raise ValueError(f'Background {data_source!r} is absent from {dataset}.')
        self.data_source = data_source
        if station_ids_for_analysis is not None and data_source == 'radiosonde':
            if not station_ids_for_analysis:
                raise ValueError('Select at least one radiosonde station.')
            data = data.sel(station=station_ids_for_analysis)
        if frequencies is not None:
            check_coordinates(frequencies, 'OMB frequencies')
            data = data.sel(frq=frequencies)
        if angles is not None:
            check_coordinates(angles, 'OMB angles')
            data = data.sel(ang=angles)
        self.an_data = select_time(data, **self.selection)
        self.max_lwp = None
        self.tb_filters = []
        if max_lwp is not None:
            self.filter_lwp(max_lwp)
        else:
            self.calc_difference()

    def calc_difference(self):
        """Recalculate selected pairs and invalidate residual fits."""
        if not self.an_data.sizes.get('time', 0):
            raise ValueError('No OMB observations remain after selection.')
        self.observation = self.an_data.sel(data_source='hatpro', drop=True)
        self.background = self.an_data.sel(data_source=self.data_source, drop=True)
        self.diff = (self.observation.TB - self.background.TB).where(lambda x: np.isfinite(x))
        self.diff.attrs.update(units='K', description='Observed minus simulated brightness temperature')
        if not bool(self.diff.notnull().any()):
            raise ValueError('No finite OMB brightness-temperature pairs remain after selection.')
        for attr in ('resid_diff', 'resid_std', 'resid_count', 'frequency_bands'):
            self.__dict__.pop(attr, None)
        self.calc_mean_std()

    def calc_mean_std(self):
        """Calculate channel summaries and common-sample covariance."""
        self.statistics = omb_statistics(self.diff)
        self.mean_diff, self.std_diff = self.statistics['mean'], self.statistics['std']

    def filter_lwp(self, lwp_threshold):
        """Keep low-LWP pairs in both sources, retaining date selection."""
        check_number(lwp_threshold, 'max_lwp', 0)
        observed = self.an_data.sel(data_source='hatpro', drop=True)
        background = self.an_data.sel(data_source=self.data_source, drop=True)
        mask = (observed.lwp < lwp_threshold) & (background.lwp < lwp_threshold)
        self.an_data = self.an_data.where(mask)
        self.max_lwp = lwp_threshold
        self.calc_difference()

    def filter_tb(self, frequency, max_tb):
        """Select low-TB scenes at a frequency in GHz, rather than an index."""
        check_number(max_tb, 'max_tb', 0)
        tb = self.an_data.TB.sel(frq=frequency, drop=True)
        mask = (tb.sel(data_source='hatpro', drop=True) < max_tb) & (tb.sel(data_source=self.data_source, drop=True) < max_tb)
        self.an_data = self.an_data.where(mask)
        self.tb_filters.append(dict(frequency=frequency, max_tb=max_tb))
        self.calc_difference()

    def remove_explained_variance(self, *, frequency_bands):
        """Fit experimental unexplained OMB residuals in kelvin."""
        self.resid_diff, self.resid_count = unexplained_omb(self.diff, frequency_bands=frequency_bands)
        self.resid_std = omb_statistics(self.resid_diff)['std']
        self.frequency_bands = frequency_bands
        return self.resid_diff, self.resid_std

    def create_input_noise_from_diff(self, model, *, statistic, factor, frequency_bands=None):
        """Use explicit OMB scatter as training augmentation, anchored at zenith.

        Station/time samples are pooled. BLS training uses the zenith estimate
        and configured angle factors. Diagnostics retain all viewing angles.
        Neither statistic isolates instrument noise.
        """
        check_number(factor, 'OMB noise factor', 0)
        if statistic == 'residual_std':
            self.remove_explained_variance(frequency_bands=frequency_bands)
            differences = self.resid_diff
            logger.warning('Using experimental unexplained OMB scatter for training noise. This is not measured instrument noise.')
        elif statistic == 'std' and frequency_bands is None:
            differences = self.diff
        else:
            raise ValueError("OMB statistic must be 'std' or 'residual_std'. Only residual_std uses frequency_bands.")
        if 'ang' in differences.dims:
            if 90.0 not in differences.ang:
                raise ValueError('OMB-derived training noise requires the zenith angle (90 degrees).')
            differences = differences.sel(ang=90, drop=True)
        dimensions = tuple(dim for dim in ('time', 'station') if dim in differences.dims)
        std = omb_statistics(differences, sample_dims=dimensions)['std'].sel(frq=model.freqs)
        if not bool(np.isfinite(std).all()):
            raise ValueError('Insufficient finite OMB pairs for training noise in one or more frequencies.')
        return {f'input_noise_TB_{frequency}': float(value) * factor for frequency, value in zip(model.freqs, std.values)}

    def metadata(self):
        """Record selections, fitted data identity and named dataset recipes."""
        recipes = {}

        def read(name):
            path = dataset_recipe_file(self.project_dir, self.site, name)
            if name in recipes or not path.is_file():
                return
            with path.open('rb') as stream:
                recipes[name] = tomllib.load(stream)
            for dependency in recipes[name].get('inputs', {}).values():
                read(dependency)

        read(self.dataset)
        return plain_value(dict(site=self.site, dataset=self.dataset, data_source=self.data_source,
                          mode='bls' if self.bls else 'zenith', selection=self.selection,
                          max_lwp=self.max_lwp, tb_filters=self.tb_filters,
                          frequencies=self.diff.frq.values,
                          angles=self.diff.ang.values if 'ang' in self.diff.dims else [90.],
                          stations=self.diff.station.values if 'station' in self.diff.dims else [],
                          first_time=str(self.diff.time.min().values), last_time=str(self.diff.time.max().values),
                          data_id=dataset_id(self.an_data[['TB']]), dataset_recipes=recipes,
                          convention='observed minus simulated TB',
                          frequency_bands=getattr(self, 'frequency_bands', None)))

    def save(self, output_file=None, *, retrievals=None, frequency_bands=None):
        """Save one HTML report, optionally comparing saved retrievals' TB noise.

        ``retrievals`` selects checkpoint names at this site. Their physical
        training input noise is overlaid only at matching frequencies/angles.
        Reports include neighboring-channel regression. ``frequency_bands``
        overrides the HATPRO groups [[20, 40], [50, 60]] GHz. An existing explicit
        fit is reused when no override is supplied. Report-only fitting leaves
        the analysis object and training settings unchanged.
        """
        from openMWR.omb_report import save_omb
        if output_file is None:
            timestamp = datetime.fromisoformat(utc_now()).strftime('%Y-%m-%d_%H-%M-%S')
            prefix = 'omb_bls' if self.bls else 'omb'
            output_file = dataset_dir(self.project_dir, self.site, self.dataset) / f'{prefix}_{timestamp}.html'
        return save_omb(self, Path(output_file), retrievals=retrievals, frequency_bands=frequency_bands)


def get_input_noise(model, hyper_params=None):
    """Physical training noise, including the BLS angle factors."""
    params = model.hyper_params if hyper_params is None else hyper_params
    noise = xr.Dataset({'TB': xr.DataArray([params[f'input_noise_TB_{f}'] for f in model.freqs],
                                          dims='frq', coords={'frq': model.freqs})})
    if model.is_bls_model:
        angles = xr.DataArray(model.angles, dims='ang', coords={'ang': model.angles})
        if model.vary_input_noise_with_angle:
            factors = xr.DataArray([params[f'input_noise_TB_{a}'] for a in model.angles], dims='ang', coords={'ang': model.angles})
        elif model.vary_input_noise_with_angle_linear:
            factors = 1 + params['input_noise_TB_angle_factor'] * (90 - angles) / 90
        else:
            factors = xr.ones_like(angles)
        noise['TB'] = noise.TB * factors
    for variable in model.input_vars:
        if variable != 'TB':
            noise[variable] = params[f'input_noise_{variable}']
    return noise
