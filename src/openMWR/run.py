import torch
import xarray as xr
import numpy as np
import logging
from datetime import datetime
from typing import Sequence, Tuple

from openMWR.models import BaseModel

logger = logging.getLogger(__name__)

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
    device: str,
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
        Torch device name (e.g., ``"cpu"``, ``"cuda"``).
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
    model.to(device)

    ds_sel = ds[model.input_vars].dropna(dim='time')

    if ds_sel["time"].size == 0:
        raise ValueError("No Data with all not Nan input values.")

    ds_nor = normalize(ds_sel, model)

    if not model.is_bls_model and 'ang' in ds_nor.dims:
        ds_nor = ds_nor.sel(ang = 90, drop=True)

    # If multiple stations are present, stack station and time dimensions
    dims_to_stack = tuple(dim for dim in ("station", "time") if dim in ds.dims)
    ds_stacked = ds_nor.stack(station_time=dims_to_stack).dropna(dim='station_time', subset=model.input_vars)
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
        pred_de = xr.merge([ds_sel, pred_de], join='outer', compat='no_conflicts')
    
    return pred_de

def run_model_list(
    model_list: Sequence[BaseModel],
    ds: xr.Dataset,
    device: str,
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
        Torch device name.
    add_not_predicted_vars : bool, optional
        If True, merge variables not predicted by any model from `ds`.
    include_input_vars : bool, optional
        Whether to include input variables in each model prediction.

    Returns
    -------
    xr.Dataset
        Concatenated predictions with a ``model`` dimension.
    """
    pred_list = []

    for model in model_list:

        pred = run_model(model, ds, device, include_input_vars=include_input_vars)

        for var in pred.data_vars:
            pred[var] = pred[var].expand_dims(model=[model.name])

        pred_list.append(pred)

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

def import_data_and_run_retrieval(
    date: datetime,
    site: str,
    data_dir: str,
    model_list: Sequence[BaseModel],
    import_retrieval_data: bool = True,
    bls: bool = False,
    smooth: bool = True,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Import MWR data, run retrieval models, and post-process outputs.

    Parameters
    ----------
    date : datetime
        Date to process.
    site : str
        Site name used for data import.
    data_dir : str
        Base directory containing all openMWR-managed data.
    model_list : list
        Models to run.
    import_retrieval_data : bool, optional
        Whether to include RPG retrieval data when importing.
    bls : bool, optional
        If True, import boundary-layer-scan data and skip smoothing.
    smooth : bool, optional
        Whether to apply temporal smoothing to outputs (ignored if `bls` is True).

    Returns
    -------
    tuple
        ``(pred, ds_hatpro)`` where `pred` contains retrieval outputs.
    """
    from openMWR.hatpro_data import import_hatpro_data, import_hatpro_data_bls, calc_derived_quantities
    
    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")

    # Import MWR measurement data
    if not bls:
        ds_hatpro = import_hatpro_data(date, site, data_dir, import_retrieval_data=import_retrieval_data, mean_one_min=True)
    else:
        ds_hatpro = import_hatpro_data_bls(date, site, data_dir, import_retrieval_data=import_retrieval_data)

    # Apply model to measurement data
    pred = run_model_list(model_list, ds_hatpro, device)

    units_dict = {
        "T": "K",
        "rh": "%",
        "surface_T": "K",
        "surface_rh": "%",
        "surface_p": "hPa",
        "lwc": "g/m^3",
        "lwp": "g/m^2",
        "iwv": "kg/m^2",
    }

    for var, unit in units_dict.items():
        pred[var].attrs["units"] = unit

    pred = calc_derived_quantities(pred)

    pred['rain_flag'] = ds_hatpro['rain_flag'].sel(time=pred.time)
    
    # Smooth result
    if smooth and not bls:
        pred = pred.rolling(time=5, center=True, min_periods=1).mean()

    return pred, ds_hatpro
