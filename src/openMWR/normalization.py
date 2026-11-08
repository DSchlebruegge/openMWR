"""Model normalization and physical input noise on the model's feature grid.

Statistics retain the checkpoint layout (variable, stat, and feature coordinates).
Noise is specified in physical units and converted with the same linear scales
as model inputs. Its flattened order follows ``model.input``.
"""

import numpy as np
import xarray as xr


def create_stats(model, ds: xr.Dataset) -> xr.Dataset:
    """Compute mean, population standard deviation and maximum over samples."""
    stats = ['mean', 'std', 'max']
    return xr.Dataset({
        var: xr.concat([getattr(ds[var], stat)(dim=['time', 'station']) for stat in stats],
                       dim='stat').assign_coords(stat=stats)
        for var in ds.data_vars if var in model.input_vars or var in model.output_vars
    })


def _aligned_stats(ds, statistics):
    """Use zenith statistics for TB without a scan dimension."""
    if 'TB' in ds:
        if 'ang' not in ds.TB.dims and 'ang' in statistics.dims:
            return statistics.sel(ang=90, drop=True)
        if 'ang' not in ds.TB.coords and 'ang' in statistics.coords:
            return statistics.drop_vars('ang')
    return statistics


def _linear_scale(statistics, var, method):
    """Share the physical scale between value and noise normalization."""
    if method in ('z_norm', 'std_norm', 'max_norm'):
        stat = 'max' if method == 'max_norm' else 'std'
        return statistics[var].sel(stat=stat, drop=True)
    return 1


def _transform(ds, model, *, inverse):
    result = ds.copy()
    statistics = _aligned_stats(ds, model.nor)
    for var in model.input_vars + model.output_vars:
        if var not in result:
            continue
        method = model.norm_functions[var]
        if method == 'log_norm':
            result[var] = np.exp(result[var]) - 1 if inverse else np.log(result[var] + 1)
        elif method in ('z_norm', 'std_norm', 'max_norm'):
            scale = _linear_scale(statistics, var, method)
            offset = statistics[var].sel(stat='mean', drop=True) if method == 'z_norm' else 0
            result[var] = (result[var] * scale + offset if inverse
                           else xr.where(scale != 0, (result[var] - offset) / scale, 0))
    return result


def normalize(ds: xr.Dataset, model) -> xr.Dataset:
    """Normalize selected model variables without changing the source dataset.

    Constant linear channels become zero. Zenith TB selects the 90-degree
    statistics when the saved normalization has an angle dimension.
    """
    return _transform(ds, model, inverse=False)


def denormalize(ds: xr.Dataset, model) -> xr.Dataset:
    """Restore physical units and clip retrieved relative humidity to 1–100%."""
    result = _transform(ds, model, inverse=True)
    if 'rh' in result:
        result['rh'] = result.rh.clip(min=1, max=100)
    return result


def get_input_noise(model, hyper_params=None) -> xr.Dataset:
    """Physical training noise, including the BLS angle factors."""
    params = model.hyper_params if hyper_params is None else hyper_params
    noise = xr.Dataset({'TB': xr.DataArray([params[f'input_noise_TB_{f}'] for f in model.freqs],
                                          dims='frq', coords={'frq': model.freqs})})
    if model.is_bls_model:
        angles = xr.DataArray(model.angles, dims='ang', coords={'ang': model.angles})
        if model.vary_input_noise_with_angle:
            if any(f'input_noise_TB_angle_{angle}' not in params and f'input_noise_TB_{angle}' in params for angle in model.angles):
                raise ValueError('Saved per-angle noise requires conversion. Run openMWR migrate --project-dir <project>.')
            factors = xr.DataArray([params[f'input_noise_TB_angle_{a}'] for a in model.angles], dims='ang', coords={'ang': model.angles})
        elif model.vary_input_noise_with_angle_linear:
            factors = 1 + params['input_noise_TB_angle_factor'] * (90 - angles) / 90
        else:
            factors = xr.ones_like(angles)
        noise['TB'] = noise.TB * factors
    for variable in model.input_vars:
        if variable != 'TB':
            noise[variable] = params[f'input_noise_{variable}']
    return noise


def normalized_input_noise(model, hyper_params=None, *, normalization=None) -> np.ndarray:
    """Return noise standard deviations in model feature order, for NN or MLR.

    Optional statistics allow preflight validation without replacing model.nor.
    Nonzero noise requires a finite, nonzero linear scale. Log normalization
    supports only zero additive noise because its conversion depends on values.
    """
    noise = get_input_noise(model, hyper_params)
    statistics = _aligned_stats(noise, model.nor if normalization is None else normalization)
    flattened = []
    for var, specification in model.input.items():
        values = noise[var]
        method = model.norm_functions[var]
        if method == 'log_norm':
            if bool((values != 0).any()):
                raise ValueError(f'{var}: additive input noise is not supported with log_norm. '
                                 'Use zero noise or a linear input normalization.')
            values = xr.zeros_like(values)
        else:
            scale = abs(_linear_scale(statistics, var, method))
            invalid = ~np.isfinite(scale) | (scale == 0)
            if bool(((values != 0) & invalid).any()):
                stat = 'max' if method == 'max_norm' else 'std'
                raise ValueError(f'{var}: nonzero input noise requires a finite, nonzero '
                                 f'normalization {stat}. Check constant or missing training inputs.')
            values = values / xr.where(invalid, 1, scale)
        flattened.append(values.sel(**specification.coords).transpose(*specification.dims).values.ravel())
    return np.concatenate(flattened)
