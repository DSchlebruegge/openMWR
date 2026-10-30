"""Validity of observation variables, independent of their source or consumer."""

import numpy as np
import xarray as xr

from openMWR.config import check_invalid_measurements, check_strings, check_name


def reference_source(data, data_source=None):
    """Select a background, inferring it only when the paired data has one."""
    if 'data_source' not in data.coords or 'hatpro' not in data.data_source.values:
        raise ValueError('Analysis data must contain HATPRO observations and reference data_source coordinates.')
    references = [str(value) for value in data.data_source.values if value != 'hatpro']
    if not references or len(set(references)) != len(references):
        raise ValueError('Analysis data must contain distinct reference sources.')
    if data_source is None:
        if len(references) != 1:
            raise ValueError(f'Select data_source when an analysis contains multiple references: {references}.')
        return references[0]
    if data_source not in references:
        raise ValueError(f'Background {data_source!r} is absent from the analysis dataset. Available references: {references}.')
    return data_source


def select_reference_sources(data, data_source, sources=None):
    """Select members of a reference dataset without pooling their samples."""
    if sources is None:
        return data
    check_strings(sources, 'sources')
    if not sources:
        raise ValueError('sources must not be empty.')
    for source in sources:
        check_name(source, 'sources')
    if data_source == 'radiosonde':
        if 'station' not in data.coords:
            raise ValueError('Missing radiosonde station coordinates in the reference dataset.')
        available = [str(value) for value in data.station.values]
    else:
        available = ['era5'] if data_source == 'era5' else []
    if len(set(available)) != len(available):
        raise ValueError('Duplicate reference source IDs.')
    if set(sources) - set(available):
        raise ValueError(f'sources must be a subset of the reference sources {available}.')
    return data.assign_coords(station=available).sel(station=sources) if data_source == 'radiosonde' else data


def mask_invalid_measurements(data, rules, *, time='time'):
    """Mask each rule's variables over the union of its declared UTC periods."""
    check_invalid_measurements(rules)
    result = data.copy()
    for rule in rules:
        invalid = xr.zeros_like(result[time], dtype=bool)
        for period in rule['periods']:
            if 'start_time' in period:
                # Dataset times are UTC without a timezone. Validation requires Z.
                selected = result[time] >= np.datetime64(period['start_time'][:-1])
                if 'end_time' in period:
                    selected = selected & (result[time] < np.datetime64(period['end_time'][:-1]))
            else:
                selected = result[time] >= np.datetime64(period['start_date'])
                if 'end_date' in period:
                    selected = selected & (result[time] < np.datetime64(period['end_date']) + np.timedelta64(1, 'D'))
            invalid = invalid | selected
        for name in rule['variables']:
            if name in result.data_vars:
                result[name] = result[name].where(~invalid)
    return result


def valid_observation_times(data, variables):
    """Return a time mask requiring finite values in each named variable.

    Station-indexed observations retain a separate mask for each station.
    Missing variables invalidate all times. Coordinates and model-specific
    grid selection remain the responsibility of the consumer.
    """
    valid = xr.ones_like(data.time, dtype=bool)
    for name in variables:
        if name not in data:
            return xr.zeros_like(data.time, dtype=bool)
        values = data[name]
        finite = np.isfinite(values)
        dimensions = [dim for dim in values.dims if dim not in ('time', 'station')]
        if dimensions:
            finite = finite.all(dimensions)
        valid = valid & finite
    return valid


def load_analysis_dataset(project_dir, site, dataset, *, bls=False, site_config=None):
    """Read paired data with current site validity and recipe sample requirements.

    Custom Python comparison files can omit a recipe. Named analysis recipes
    specify which observation variables must be finite. Reference values are
    preserved when an observation variable is masked.
    """
    from openMWR.paths import dataset_file, dataset_recipe_file
    from openMWR.datasets import DatasetRecipe
    from openMWR.site import get_invalid_measurements

    path = dataset_file(project_dir, site, dataset, bls=bls)
    data = xr.load_dataset(path)
    required = []
    exclude_rain = False
    if dataset_recipe_file(project_dir, site, dataset).is_file():
        recipe = DatasetRecipe.load(project_dir, site, dataset)
        if recipe.kind != 'analysis':
            raise ValueError(f'{dataset} must be an analysis dataset.')
        required = recipe.values['analysis']['required_observations']
        exclude_rain = recipe.values['analysis']['exclude_rain']
        if exclude_rain:
            required = [*required, 'rain_flag']
    rules = get_invalid_measurements(site, project_dir, site_config=site_config)
    measured = mask_invalid_measurements(data.sel(data_source='hatpro', drop=True), rules)
    if 'old_time' in measured:
        measured = mask_invalid_measurements(measured, rules, time='old_time')
    valid = valid_observation_times(measured, required)
    if exclude_rain and 'rain_flag' in measured:
        dry = measured.rain_flag == 0.0
        valid = valid & dry.all([dim for dim in dry.dims if dim not in ('time', 'station')])
    affected = {name for rule in rules for name in rule['variables']}
    if required and not bool(valid.all()):
        # Exclude individual station/time pairs as well as wholly invalid times.
        # A model without the failed sensor must not reuse these analysis pairs.
        affected.update(name for name in measured if 'time' in measured[name].dims)
        measured = measured.where(valid)
    for name in affected & set(measured.data_vars):
        dimensions = {dim: data[dim] for dim in measured[name].dims if dim not in data[name].dims}
        if 'data_source' not in data[name].dims:
            dimensions['data_source'] = data.data_source
        if dimensions:
            data[name] = data[name].expand_dims(dimensions).copy(deep=True)
        data[name].loc[{'data_source': 'hatpro'}] = measured[name]
    times = valid.any('station') if 'station' in valid.dims else valid
    return data.isel(time=times)
