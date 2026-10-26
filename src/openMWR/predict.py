"""Inference on xarray datasets and daily prediction from local measurements."""

from dataclasses import dataclass, field
from datetime import date as calendar_date, datetime, timezone, timedelta
from typing import Sequence

import torch
import xarray as xr
import numpy as np
import logging
from openMWR.models import BaseModel, load_model
from openMWR.paths import project_root
from openMWR.site import get_config_parameter, prediction_output_dir, get_invalid_measurements
from openMWR.devices import resolve_device

logger = logging.getLogger(__name__)


def validate_model_data(specification, data, label='data'):
    """Check variables, coordinates and complete samples without building tensors."""
    if not data.sizes.get('time', 0):
        raise ValueError(f'{label} contains no time samples after filtering.')
    for coordinate in ('time', 'height', 'frq', 'ang'):
        if coordinate in data.indexes and not data.indexes[coordinate].is_unique:
            raise ValueError(f'{label} has duplicate {coordinate} coordinates.')
    valid = None
    for name, expected in specification.items():
        if name not in data:
            raise ValueError(f'{label} is missing variable {name!r}.')
        values = data[name]
        if set(values.dims) - {'station', 'time'} != set(expected.dims):
            raise ValueError(f'{label}.{name} dimensions {values.dims} do not match model dimensions {expected.dims}.')
        try:
            values = values.sel(**expected.coords)
        except (KeyError, ValueError) as exc:
            raise ValueError(f'{label}.{name} coordinates do not match the model: {exc}') from exc
        finite = np.isfinite(values).all(list(expected.dims))
        valid = finite if valid is None else valid & finite
    if valid is None or not bool(valid.any()):
        raise ValueError(f'{label} contains no complete finite samples for this model.')

def create_tensor_from_ds(model_ds: xr.Dataset, data_ds: xr.Dataset, device: str, dim_0: str = 'station_time') -> torch.Tensor:
    """
    Create a PyTorch tensor by aligning and concatenating variables from two xarray Datasets.

    Parameters
    ----------
    model_ds : xr.Dataset
        Specification of expected variables and their non-leading dimensions. Could be either model.input or model.output.
    data_ds : xr.Dataset
        Dataset providing the actual data to convert into a tensor.
    device : str
        Target device for the resulting tensor (e.g., ``"cpu"``, ``"cuda"``).
    dim_0 : str, optional
        Name of the leading dimension. Defaults to ``"station_time"``.

    Returns
    -------
    torch.Tensor
        A tensor of shape ``(dim_0_size, total_features)``, where all variables are
        aligned, flattened, and concatenated along the feature axis.

    Raises
    ------
    ValueError
        If variable dimensions in `data_ds` do not match those specified in `model_ds`
        (excluding the leading dimension).
    """

    dim_0_size = data_ds.sizes[dim_0]

    x_list = []

    for var, da in model_ds.items():

        if set(data_ds[var].dims) - {dim_0} != set(da.dims):
            raise ValueError(f"Dimensions mismatch for variable {var}: {data_ds[var].dims} in data, but {da.dims} in model input.")

        input_da = data_ds[var].sel(**da.coords).transpose(dim_0, *da.dims)

        arr = input_da.values.reshape(dim_0_size, -1)

        x_list.append(arr)

    x_np = np.concatenate(x_list, axis=1)
    return torch.tensor(x_np, dtype=torch.float32).to(device)

def create_ds_from_y(y: np.ndarray, model: BaseModel, ds_orig: xr.Dataset, dim_0: str = 'station_time') -> xr.Dataset:
    """
    Create a new xarray.Dataset by inserting model output variables taken from a
    prediction array, while preserving all non-model variables from the original dataset.

    Parameters
    ----------
    y : np.ndarray
        Model output array of shape ``(N, features)``, typically produced by a
        forward pass.
    model : BaseModel
        Model object containing:
        - ``output``: an xr.Dataset describing output variable shapes.
        - ``output_slices``: dict mapping variable names to column slices in ``y``.
    ds_orig : xr.Dataset
        Original dataset used for metadata and coordinates.
    dim_0 : str, optional
        Name of the leading dimension. Defaults to ``"station_time"``.

    Returns
    -------
    xr.Dataset
        A dataset matching `ds_orig` in structure but with variables defined in
        ``model.output`` replaced by the corresponding reshaped outputs from ``y``.
    """


    coords_0 = {dim_0: ds_orig[dim_0]}

    # Keep the stacked MultiIndex coordinate only at dataset level to avoid
    # xarray's deprecated single-level MultiIndex drop path during merges.
    ds_new = xr.Dataset(coords=coords_0)
    for var, da in model.output.items():
        sl = model.output_slices[var]
        arr = y[:, sl].reshape((y.shape[0],) + da.shape)

        var_coords = {name: coord for name, coord in da.coords.items() if name != dim_0}
        ds_new[var] = xr.DataArray(arr, dims=(dim_0,) + da.dims, coords=var_coords)

    return ds_new

def normalize(ds: xr.Dataset, model: BaseModel) -> xr.Dataset:
    """
    Normalizes the dataset based on the model's normalization functions and statistics.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be normalized.
    model : BaseModel
        The model containing normalization functions and statistics.

    Returns
    -------
    xarray.Dataset
        The normalized dataset.

    Notes
    -----
    - Supports various normalization methods: z-score, standard deviation, max normalization, and logarithmic normalization.
    - Handles special cases for 'TB' variable with 'ang' dimension.
    """
    ds_norm = ds.copy()

    nor = model.nor

    if 'TB' in ds_norm.data_vars:
        if 'ang' not in ds_norm.TB.dims and 'ang' in nor.dims:
            nor = nor.sel(ang=90, drop=True)
        elif 'ang' not in ds_norm.TB.coords and 'ang' in nor.coords:
            nor = nor.drop_vars('ang')

    vars_to_normalize = [var for var in model.input_vars + model.output_vars if var in ds_norm.data_vars]

    for var in vars_to_normalize:
        if model.norm_functions[var] == 'z_norm':
            std = nor[var].sel(stat='std', drop=True)
            mean = nor[var].sel(stat='mean', drop=True)
            ds_norm[var] = xr.where(std != 0, (ds_norm[var] - mean) / std, 0)

        elif model.norm_functions[var] == 'std_norm':
            std = nor[var].sel(stat='std', drop=True)
            ds_norm[var] = xr.where(std != 0, ds_norm[var] / std, 0)

        elif model.norm_functions[var] == 'max_norm':
            max = nor[var].sel(stat='max', drop=True)
            ds_norm[var] = xr.where(max != 0, ds_norm[var] / max, 0)

        elif model.norm_functions[var] == 'log_norm':
            ds_norm[var] = np.log(ds_norm[var] + 1)

    return ds_norm

def denormalize(ds_norm: xr.Dataset, model: BaseModel) -> xr.Dataset:
    """
    Denormalizes the dataset based on the model's normalization functions and statistics.

    Parameters
    ----------
    ds_norm : xarray.Dataset
        The normalized dataset to be denormalized.
    model : BaseModel
        The model containing normalization functions and statistics.

    Returns
    -------
    xarray.Dataset
        The denormalized dataset.
    """
    ds = ds_norm.copy()

    nor = model.nor

    if 'TB' in ds_norm.data_vars:
        if 'ang' not in ds_norm.TB.dims and 'ang' in nor.dims:
            nor = nor.sel(ang=90, drop=True)
        elif 'ang' not in ds_norm.TB.coords and 'ang' in nor.coords:
            nor = nor.drop_vars('ang')

    vars_to_denormalize = [var for var in model.input_vars + model.output_vars if var in ds_norm.data_vars]

    for var in vars_to_denormalize:
        if model.norm_functions[var] == 'z_norm':
            ds[var] = ds[var] * nor[var].sel(stat='std', drop=True) + nor[var].sel(stat='mean', drop=True)

        elif model.norm_functions[var] == 'std_norm':
            ds[var] = ds[var] * nor[var].sel(stat='std', drop=True)

        elif model.norm_functions[var] == 'max_norm':
            ds[var] = ds[var] * nor[var].sel(stat='max', drop=True)

        elif model.norm_functions[var] == 'log_norm':
            ds[var] = np.exp(ds[var]) - 1

    if 'rh' in ds.data_vars:
        ds['rh'] = ds['rh'].where(ds['rh'] > 1, 1)
        ds['rh'] = ds['rh'].where(ds['rh'] < 100, 100)

    return ds

def create_stats(model: BaseModel, ds: xr.Dataset) -> xr.Dataset:
    """
    Creates a dataset containing statistics (mean, std, max) for each variable in the model's input and output variables.

    Parameters
    ----------
    model : BaseModel
        The model containing input and output variable specifications.
    ds : xarray.Dataset
        The dataset from which to compute statistics.

    Returns
    -------
    xarray.Dataset
        A dataset containing the computed statistics for each variable.
    """

    dims = ["time", "station"]
    stats = ["mean", "std", "max"]

    nor = xr.Dataset({
        var: xr.concat(
            [getattr(ds[var], stat)(dim=dims) for stat in stats], #[ds[var].mean(dim=dims), ds[var].std(dim=dims), ds[var].max(dim=dims)],
            dim="stat"
        ).assign_coords(stat=stats)
        for var in ds.data_vars if var in model.input_vars or var in model.output_vars
    })

    return nor

def run_model(
    model: BaseModel,
    ds: xr.Dataset,
    device: str = 'auto',
    include_input_vars: bool = True,
) -> xr.Dataset:
    """
    Run a model on an input dataset and return denormalized predictions.

    Parameters
    ----------
    model : BaseModel
        Trained model with input/output specs and normalization info.
    ds : xr.Dataset
        Input dataset containing the model input variables.
    device : str
        Torch device name. ``"auto"`` selects CUDA, then MPS, then CPU.
    include_input_vars : bool, optional
        Whether to merge input variables into the output dataset.

    Returns
    -------
    xr.Dataset
        Dataset with predicted output variables (and optionally inputs).

    Raises
    ------
    ValueError
        If no valid input samples remain after dropping NaNs.
    """
    if device == 'auto':
        device = resolve_device(device)
    model.to(device)

    ds_sel = ds[model.input_vars]
    if not model.is_bls_model and 'ang' in ds_sel.dims:
        ds_sel = ds_sel.sel(ang=90, drop=True)
    # Normalize only the channels/angles expected by this checkpoint. Extra
    # measured channels otherwise acquire NaNs or fail xarray alignment.
    ds_sel = ds_sel.sel({name: values for name, values in model.input.coords.items()
                         if name in ds_sel.dims})
    ds_sel = ds_sel.where(np.isfinite(ds_sel))

    if ds_sel["time"].size == 0:
        raise ValueError("No Data with all not Nan input values.")

    ds_nor = normalize(ds_sel, model)

    if not model.is_bls_model and 'ang' in ds_nor.dims:
        ds_nor = ds_nor.sel(ang = 90, drop=True)

    # If multiple stations are present, stack station and time dimensions
    dims_to_stack = tuple(dim for dim in ("station", "time") if dim in ds.dims)
    ds_stacked = ds_nor.stack(station_time=dims_to_stack).dropna(dim='station_time', subset=model.input_vars)
    if not ds_stacked.sizes['station_time']:
        raise ValueError('No Data with all finite input values.')
    #x = create_x_tensor(model, ds_stacked, device)
    x = create_tensor_from_ds(model.input, ds_stacked, device)

    # Run model
    model.eval()
    with torch.no_grad():
        y = model(x).detach().cpu().numpy()

    model.to('cpu')

    pred_nor_stacked = create_ds_from_y(y, model, ds_stacked)

    pred_nor = pred_nor_stacked.unstack("station_time")

    pred_de = denormalize(pred_nor, model)

    if include_input_vars:
        pred_de = xr.merge([ds_sel.drop_vars(model.output_vars, errors='ignore'), pred_de], join='outer', compat='no_conflicts')
    
    return pred_de

def run_model_list(
    model_list: Sequence[BaseModel],
    ds: xr.Dataset,
    device: str = 'auto',
    add_not_predicted_vars: bool = False,
    include_input_vars: bool = True,
) -> xr.Dataset:
    """
    Run multiple models and concatenate predictions along a ``model`` dimension.

    Parameters
    ----------
    model_list : list
        Models to apply.
    ds : xr.Dataset
        Input dataset.
    device : str
        Torch device name. ``"auto"`` selects CUDA, then MPS, then CPU.
    add_not_predicted_vars : bool, optional
        If True, merge variables not predicted by any model from `ds`.
    include_input_vars : bool, optional
        Whether to include input variables in each model prediction.

    Returns
    -------
    xr.Dataset
        Concatenated predictions with a ``model`` dimension.
    """
    if device == 'auto':
        device = resolve_device(device)
    pred_list = []

    for model in model_list:

        pred = run_model(model, ds, device, include_input_vars=include_input_vars)

        pred_list.append(pred.expand_dims(model=[model.name]))

    ds_predicted = xr.concat(pred_list, dim='model')

    if add_not_predicted_vars:
        # Add all variables which are not predicted by any model
        total_output_vars = set()
        for model in model_list:
            total_output_vars.update(model.output_vars)

        ds_not_predicted = ds.drop_vars(total_output_vars, errors='ignore')

        return xr.merge([ds_not_predicted, ds_predicted], join='outer', compat='override')
    else:
        return ds_predicted

def _day(value):
    """Normalize a calendar day or datetime to midnight UTC (timezone naive)."""
    if isinstance(value, str):
        value = calendar_date.fromisoformat(value)
    if isinstance(value, datetime) and value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    if not isinstance(value, calendar_date):
        raise TypeError('A prediction date must be a date, datetime or YYYY-MM-DD string.')
    return datetime(value.year, value.month, value.day)


def _daily_file(directory, date, bls=False):
    suffix = '_bls' if bls else ''
    return directory / date.strftime('%Y/%m') / f'{date:%Y%m%d}_retrieval{suffix}.nc'


@dataclass
class PredictionResult:
    """One UTC day's predictions and optional imported observation datasets.

    Zenith and BLS predictions have a ``model`` dimension and retain their own
    time/height grids. ``errors`` describes unavailable requested groups; usable
    predictions from the other group can still be saved and plotted.
    """

    site: str
    date: datetime
    zenith: xr.Dataset | None = None
    bls: xr.Dataset | None = None
    observations: xr.Dataset | None = None
    observations_bls: xr.Dataset | None = None
    errors: dict[str, str] = field(default_factory=dict)


class PredictionWorkflow:
    """Load saved retrievals once and apply them to local daily measurements.

    ``retrievals=None`` selects ``[prediction].retrievals`` from site.toml.
    Training recipes are never read. Prediction does not write files; call
    ``save(result)`` explicitly. Python callers can disable zenith smoothing or
    include RPG products in the returned observations for comparison plots.
    """

    def __init__(self, project_dir, site, retrievals=None, device='auto', *,
                 output_dir=None, smooth=True, import_retrieval_data=False):
        self.project_dir = project_root(project_dir)
        self.site = site
        self.device = resolve_device(device)
        if retrievals is None:
            settings = get_config_parameter(site, 'prediction', self.project_dir) or {}
            if not isinstance(settings, dict) or set(settings) - {'retrievals'}:
                raise ValueError('[prediction] supports only a retrievals list.')
            retrievals = settings.get('retrievals')
        if isinstance(retrievals, str):
            retrievals = [retrievals]
        if not isinstance(retrievals, (list, tuple)) or not retrievals:
            raise ValueError('Select retrieval names or set [prediction].retrievals in site.toml.')
        if any(not isinstance(name, str) or not name or name in ('.', '..')
               or '/' in name or '\\' in name for name in retrievals):
            raise ValueError('Retrievals must be nonempty directory names without separators.')
        if len(set(retrievals)) != len(retrievals):
            raise ValueError('Select each retrieval only once.')
        self.models = tuple(load_model(name, site, self.project_dir) for name in retrievals)
        self._zenith_models = tuple(model for model in self.models if not model.is_bls_model)
        self._bls_models = tuple(model for model in self.models if model.is_bls_model)
        self.output_dir = prediction_output_dir(self.project_dir, site, output_dir)
        self.smooth = smooth
        self.import_retrieval_data = import_retrieval_data
        logger.info('Prediction for %s: %s; device=%s', site, ', '.join(retrievals), self.device)

    def predict(self, date, *, observations=None, observations_bls=None):
        """Return predictions without writing or downloading; retain partial groups.

        Optional standardized observation datasets bypass daily file loading.
        They must belong to the requested UTC day. Site validity periods still
        apply. Missing/invalid input is always reported in ``result.errors``, including
        days when neither group succeeds. Configuration and computation errors
        still raise exceptions.
        """
        date = _day(date)
        result = PredictionResult(self.site, date)
        result.zenith, result.observations, zenith_error = self._predict_group(
            date, self._zenith_models, bls=False, observations=observations)
        result.bls, result.observations_bls, bls_error = self._predict_group(
            date, self._bls_models, bls=True, observations=observations_bls)
        if zenith_error is not None:
            result.errors['zenith'] = zenith_error
        if bls_error is not None:
            result.errors['bls'] = bls_error
        return result

    def _predict_group(self, date, models, *, bls, observations=None):
        from openMWR.hatpro_data import import_hatpro_data, import_hatpro_data_bls
        from openMWR.observations import mask_invalid_measurements

        if not models:
            return None, None, None
        periods = get_invalid_measurements(self.site, self.project_dir)
        group = 'bls' if bls else 'zenith'
        logger.info('Predict %s %s %s: %s', self.site, date.date(), group,
                    ', '.join(model.name for model in models))
        try:
            if observations is not None:
                if bool(((observations.time < np.datetime64(date)) |
                         (observations.time >= np.datetime64(date + timedelta(days=1)))).any()):
                    raise ValueError('Observations must belong to the requested UTC day.')
            elif bls:
                observations = import_hatpro_data_bls(date, self.site, self.project_dir,
                    import_retrieval_data=self.import_retrieval_data)
            else:
                observations = import_hatpro_data(date, self.site, self.project_dir,
                    import_retrieval_data=self.import_retrieval_data, mean_one_min=True)
            observations = mask_invalid_measurements(observations, periods)
            for model in models:
                validate_model_data(model.input, observations, f'{group} observations for {model.name}')
        except (OSError, ValueError) as exc:
            logger.warning('%s %s %s: %s', self.site, date.date(), group, exc)
            return None, None, str(exc)
        predicted = self._predict_observations(models, observations, bls)
        return predicted, observations, None

    def _predict_observations(self, models, observations, bls):
        from openMWR.hatpro_data import calc_derived_quantities

        predicted = run_model_list(models, observations, self.device)
        # Surface pressure is needed for derived profile quantities even when a
        # custom model did not use it as an input. Spectral outputs need no profiles.
        if {'T', 'rh', 'lwc'} <= set(predicted):
            if 'surface_p' not in predicted and 'surface_p' in observations:
                predicted['surface_p'] = observations.surface_p.sel(time=predicted.time)
            predicted = calc_derived_quantities(predicted)
        if 'rain_flag' in observations:
            predicted['rain_flag'] = observations.rain_flag.sel(time=predicted.time)
        if self.smooth and not bls:
            predicted = predicted.rolling(time=5, center=True, min_periods=1).mean(keep_attrs=True)
        units = {'T': 'K', 'rh': '%', 'surface_T': 'K', 'surface_rh': '%',
                 'surface_p': 'hPa', 'lwc': 'g/m^3', 'lwp': 'g/m^2', 'iwv': 'kg/m^2', 'TB': 'K'}
        for name, unit in units.items():
            if name in predicted:
                predicted[name].attrs['units'] = unit
        return predicted

    def save(self, result):
        """Replace selected daily NetCDF files and return their paths.

        Retain the historical scalar BLS model coordinate for a single BLS
        retrieval. In-memory predictions always keep the model dimension.
        """
        if result.site != self.site:
            raise ValueError('Cannot save predictions from another site with this workflow.')
        paths = []
        for predicted, bls in ((result.zenith, False), (result.bls, True)):
            if predicted is None:
                continue
            path = _daily_file(self.output_dir, result.date, bls)
            path.parent.mkdir(parents=True, exist_ok=True)
            output = predicted.drop_vars(['doy_sin', 'doy_cos'], errors='ignore')
            if bls and output.sizes.get('model') == 1:
                output = output.squeeze('model')
            output.to_netcdf(path)
            logger.info('Saved %s', path)
            paths.append(path)
        return paths


def load_predictions(project_dir, site, date, *, output_dir=None):
    """Read daily prediction files without loading models or running inference.

    Observations are not stored in these files; comparison plots can import
    their local RPG inputs separately. Missing groups remain None.
    """
    date = _day(date)
    directory = prediction_output_dir(project_dir, site, output_dir)
    result = PredictionResult(site, date,
        zenith=_load_prediction_file(_daily_file(directory, date)),
        bls=_load_prediction_file(_daily_file(directory, date, bls=True)))
    if result.zenith is None and result.bls is None:
        raise FileNotFoundError(f'No saved predictions for {site} on {date:%Y-%m-%d} in {directory}.')
    return result


def _load_prediction_file(path):
    if not path.exists():
        return None
    predicted = xr.load_dataset(path)
    if 'model' in predicted.coords and 'model' not in predicted.dims:
        rain_flag = predicted.get('rain_flag')
        predicted = predicted.expand_dims('model')
        if rain_flag is not None:
            predicted['rain_flag'] = rain_flag
    return predicted
