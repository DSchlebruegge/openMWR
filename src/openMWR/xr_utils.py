import xarray as xr
import numpy as np
from openMWR.paths import site_subdir

def _get_indices_to_interpolate(old_coord: np.ndarray, new_coord: np.ndarray, extrapolate_lower_end=False, extrapolate_upper_end=False):
    delta_new_coord = np.diff(new_coord) / 2
    mid_level = np.concatenate(([new_coord[0] - delta_new_coord[0]], 
                                new_coord[:-1] + delta_new_coord, 
                                [new_coord[-1] + delta_new_coord[-1]]))
    
    indices_to_interpolate = []

    # Iterate over height layers along the specified axis
    for i, h in enumerate(new_coord):
        # Nearest larger and smaller values
        indices_lower = np.where(old_coord <= h)[0]
        indices_upper = np.where(old_coord > h)[0]
        
        if indices_lower.size == 0 and not extrapolate_lower_end:
            indices_to_interpolate.append(np.array([]))
            continue
        if indices_upper.size == 0 and not extrapolate_upper_end:
            indices_to_interpolate.append(np.array([]))
            continue

        index_lower = np.max(indices_lower) if indices_lower.size else np.nan
        index_upper = np.min(indices_upper) if indices_upper.size else np.nan
        indices_neighbor = np.array([i for i in (index_lower, index_upper) if not np.isnan(i)])

        # Values within the layer
        indices_in_level = np.where((old_coord >= mid_level[i]) & (old_coord < mid_level[i + 1]))[0]

        # Two closest values
        indices_closest = np.argsort(np.abs(old_coord - h))[:2]

        # Combine indices
        indices = np.unique(np.concatenate((indices_in_level, indices_neighbor, indices_closest)))
        indices_to_interpolate.append(indices)

    return indices_to_interpolate

def _interpolate_array_along_dim(old_data: np.ndarray, old_coord: np.ndarray, new_coord: np.ndarray, indices_to_interpolate, axis_num):
    from scipy import interpolate
    
    new_shape = list(old_data.shape)
    new_shape[axis_num] = len(new_coord)
    new_data = np.empty(new_shape)

    def slices(indices):
        """Helper function to create slices."""
        return tuple(indices if ax == axis_num else slice(None) for ax in range(old_data.ndim))

    # Iterate over height layers along the specified axis
    for i, h in enumerate(new_coord):
        indices = indices_to_interpolate[i]
        if indices.size == 0:
            new_data[slices(i)] = np.nan
            continue

        # Interpolation along the specified axis
        f = interpolate.interp1d(old_coord[indices], old_data[slices(indices)], axis=axis_num, fill_value="extrapolate")
        new_data[slices(i)] = f(h)

    return new_data

def change_coord_of_da(da, new_coord, dim='height', extrapolate_lower_end=False, extrapolate_upper_end=False, indices_to_interpolate=None):
    """
    Change the coordinate of a DataArray to a new coordinate, interpolating values along the specified dimension.

    Parameters
    --------------
    da: xarray.DataArray
        The input DataArray with the original coordinate.
    new_coord: array-like
        The new coordinate values to which the DataArray should be changed.
    dim: str, optional
        The dimension along which to change the coordinate (default is 'height').
    extrapolate_lower_end: bool, optional
        Whether to extrapolate values for lower end of the new coordinate (default is False).
    extrapolate_upper_end: bool, optional
        Whether to extrapolate values for upper end of the new coordinate (default is False).
    indices_to_interpolate: list of np.ndarray, optional
        Precomputed indices to use for interpolation (default is None, which computes them).

    Returns
    --------------
    xarray.DataArray
        A new DataArray with the updated coordinate and interpolated values.
    """
    
    old_coord = da[dim].values
    axis_num = da.get_axis_num(dim)

    if indices_to_interpolate is None:
        indices_to_interpolate = _get_indices_to_interpolate(old_coord, new_coord, extrapolate_lower_end, extrapolate_upper_end)

    new_data = _interpolate_array_along_dim(da.values, old_coord, new_coord, indices_to_interpolate, axis_num)

    coords = {**{d: da[d] for d in da.dims if d != dim}, dim: new_coord }

    return xr.DataArray(new_data, coords, attrs=da.attrs.copy())


def change_coord_of_ds(ds, new_coord, dim='height', extrapolate_lower_end=False, extrapolate_upper_end=False, indices_to_interpolate=None):
    """
    Change the coordinate of all DataArrays in a Dataset to a new coordinate, interpolating values along the specified dimension.
    
    Parameters
    --------------
    ds: xarray.Dataset
        The input Dataset with DataArrays having the original coordinate.
    new_coord: array-like
        The new coordinate values to which the DataArrays should be changed.
    dim: str, optional
        The dimension along which to change the coordinate (default is 'height').
    extrapolate_lower_end: bool, optional
        Whether to extrapolate values for lower end of the new coordinate (default is False).
    extrapolate_upper_end: bool, optional
        Whether to extrapolate values for upper end of the new coordinate (default is False).
    indices_to_interpolate: list of np.ndarray, optional
        Precomputed indices to use for interpolation (default is None, which computes them).

    Returns
    --------------
    xarray.Dataset
        A new Dataset with updated coordinates and interpolated values in its DataArrays.
    """
    
    old_coord = ds[dim].values
    
    vars_with_dim = [var for var in ds.data_vars if dim in ds[var].dims]
    vars_without_dim = [var for var in ds.data_vars if dim not in ds[var].dims]

    if indices_to_interpolate is None:
        indices_to_interpolate = _get_indices_to_interpolate(old_coord, new_coord, extrapolate_lower_end, extrapolate_upper_end)

    updated_vars = {   
        var: change_coord_of_da(ds[var], new_coord, dim, extrapolate_lower_end=extrapolate_lower_end, extrapolate_upper_end=extrapolate_upper_end, indices_to_interpolate=indices_to_interpolate)    
        for var in vars_with_dim
    }

    updated_vars.update({
        var: ds[var] for var in vars_without_dim
    })
    
    return xr.Dataset(updated_vars, attrs=ds.attrs.copy())

def fill_with_nan(ds, freq="1min"):
    """
    Reindex an xarray Dataset with a complete time range.

    Creates a full time index from the dataset's time coordinate using the specified 
    frequency (default: "1min") and reindexes the dataset. Missing values are filled 
    with NaN.

    Parameters
    --------------
    ds : xarray.Dataset
        Input dataset with a time coordinate.
    freq : str, optional
        Frequency for the new time index (default: "1min").

    Returns
    --------------
    xarray.Dataset
        Reindexed dataset with missing values filled as NaN.
    """

    import pandas as pd
    
    full_time_index = pd.date_range(ds.time.min().item(), ds.time.max().item(), freq=freq)

    ds = ds.reindex(time=full_time_index)

    return ds

def calc_gradient(da: xr.DataArray, height_dim='height'):
    """
    Calculate the gradient of a DataArray along a specified height dimension.
    """
    return da.diff(height_dim, label='lower') / da[height_dim].diff(height_dim, label='lower')

def integrate(da: xr.DataArray, dim: str):
    """
    Integrate a DataArray along a specified dimension using the trapezoidal rule.
    """
    return (da * da[dim].diff(dim, label='lower')).sum(dim=dim)

def swap_dims_and_rename(ds, old_dim="altitude_layer", old_var="altitude", new_dim="height"):
    """
    Swap dimensions and rename a variable in an xarray Dataset.
    
    Parameters
    --------------
    ds : xarray.Dataset
        The input xarray Dataset.
    old_dim : str
        The name of the dimension to be swapped (default is "altitude_layer").
    old_var : str
        The name of the variable to be used as the new dimension (default is "altitude").
    new_dim : str
        The new name for the dimension after swapping (default is "height").
    
    Returns
    --------------
    xarray.Dataset
        The modified Dataset with the specified dimension swapped and renamed.
    """
    ds = ds.swap_dims({old_dim: old_var})
    ds = ds.rename({old_var: new_dim})   
    return ds

def mean_n_min(ds: xr.Dataset, n_min: int):
    """
    Resamples variables with a time dimension in the given xarray Dataset to the specified interval (in minutes),
    computing the mean for each interval. Variables without a time dimension are retained unchanged.

    Parameters
    --------------
    ds : xr.Dataset
        The input xarray Dataset containing data variables, some of which may have a 'time' dimension.
    n_min : int
        The resampling interval in minutes.
    
    Returns
    --------------
    xr.Dataset
        A new Dataset with time-dimensioned variables resampled and averaged over the specified interval,
        and non-time-dimensioned variables included unchanged. Only intervals with at least one data point are retained.
    """

    # Variables with a time dimension
    vars_with_time = [v for v in ds.data_vars if 'time' in ds[v].dims]
    # Separate: variables without a time dimension
    vars_without_time = [v for v in ds.data_vars if 'time' not in ds[v].dims]
    
    # Resample only variables with a time dimension
    mean = ds[vars_with_time].resample(time=f'{n_min}min').mean(skipna=True) # , origin='start'

    count = ds["time"].resample(time=f"{n_min}min").count()

    mean = mean.where(count >= 1, drop=True)

    # Append remaining variables unchanged
    mean = mean.assign(**{v: ds[v] for v in vars_without_time})

    return mean

def load_combined_dataset(site, data_dir, file, sources, file_extension=''):
    """
    Loads and combines multiple NetCDF datasets for specified station IDs into a single xarray Dataset.

    Parameters
    --------------
    site : str
        The name of the site, used to construct the data directory path.
    file : str
        The base filename (without station ID and extension) of the NetCDF files to load. 
        Options: 'training_data', 'validation_data', 'test_data', 'total_data'
    sources : list of str
        List of data sources. Can include radiosonde station IDs and 'era5'.
    file_extension : str, optional
        Optional suffix for the file names (default is '').
    
    Returns
    --------------
    ds : xarray.Dataset
        The concatenated xarray Dataset containing data from all specified stations along a new 'station' dimension.
    """
    forward_calc_dir = site_subdir(data_dir, site, "training")

    if file_extension != '':
        file_extension = '_' + file_extension

    ds_list = [xr.load_dataset(forward_calc_dir / f"{file}_{s}{file_extension}.nc") for s in sources]

    ds = xr.concat(ds_list, dim='station', join='outer')

    return ds

def to_1d_tensor(ds, device):
    """Convert xarray Dataset to 1D torch tensor."""
    import torch
    arrays = []
    for var in ds:
        arr = ds[var].values
        if arr.ndim > 1:
            arr = arr.flatten()
        elif arr.ndim == 0:
            arr = np.array([arr])
        arrays.append(arr)
    tensor = torch.tensor(np.concatenate(arrays), dtype=torch.float32, device=device)
    return tensor

def add_time_data(ds):
    """
    Adds time-related features to the dataset based on the 'time' coordinate. 
    
    Notes
    -----
    Features added are:

    - 'doy_cos': Cosine of the day of year (to capture seasonal cycles)
    - 'doy_sin': Sine of the day of year
    - 'years_since_1970': Time in years since 1970-01-01
    """
    ds['doy_cos'] = np.cos(2*np.pi/365.25 * (ds.time.dt.dayofyear+10)) # +10 so 21.12 is the zero point
    ds['doy_sin'] = np.sin(2*np.pi/365.25 * (ds.time.dt.dayofyear+10))
    ds['years_since_1970'] = ds.time.astype('int64').astype('float64') * 1e-9 / 60 / 60 / 24 / 365.25

    return ds


def add_variables_at_closest_time(ds1: xr.Dataset, ds2: xr.Dataset, max_diff: int = 2) -> xr.Dataset:
    """
    Add variables from a second dataset to a first dataset by matching timestamps,
    using exact matches when possible and nearest-neighbor matching otherwise.

    Parameters
    ----------
    ds1 : xr.Dataset
        The base dataset to which new variables will be added. Its time coordinate
        defines the reference timeline.
    ds2 : xr.Dataset
        The dataset providing additional variables. Variables are matched to ds1
        based on timestamps.
    max_diff : int, optional
        Maximum allowed time difference (in minutes) for nearest-neighbor matching.
        Only nearest timestamps within this tolerance are used. Default is 2 minutes.

    Returns
    -------
    xr.Dataset
        A copy of `ds1` containing additional variables from `ds2`, aligned by time.
        Timestamps with exact matches take precedence. Non-exact matches are
        included only when within the allowed tolerance; otherwise, NaN values
        are inserted.

    Notes
    -----
    - Exact timestamp matches are detected first.
    - Non-matching timestamps from `ds1` are paired with the closest timestamps
      in `ds2` if the time difference is below `max_diff` minutes.
    - The function creates new variables with the same names as in `ds2`.
    - Values outside the tolerance window are filled with NaN.
    """

    # Time axes
    time1 = ds1.time
    time2 = ds2.time

    # Direct matching times (exact matches)
    common_times, idx1, idx2 = np.intersect1d(time1.values, time2.values, return_indices=True)

    # Mask: which timestamps in ds1 are exact matches?
    is_exact_match = np.isin(time1.values, common_times)

    # Indices of the remaining (non-exact) timestamps in ds1
    non_exact_indices = np.where(~is_exact_match)[0]

    # Time alignment only for the remaining timestamps
    time1_non_exact = time1.values[non_exact_indices]
    time2_values = time2.values

    # Index in ds2 with the minimal distance
    nearest_indices = np.abs(time2_values[:, np.newaxis] - time1_non_exact).argmin(axis=0)
    nearest_times = time2_values[nearest_indices]

    # Apply tolerance mask
    max_delta = np.timedelta64(max_diff, 'm')
    time_deltas = np.abs(nearest_times - time1_non_exact)
    valid_mask = time_deltas < max_delta

    # Empty dictionary to collect matched variables
    new_vars = {}

    for var in ds2.data_vars:
        # Prepare empty array with NaNs, same length as ds1
        full_array = np.full_like(ds1.time.values, np.nan, dtype=float)

        # Part 1: insert exact matches directly
        full_array[idx1] = ds2[var].values[idx2]

        # Part 2: insert non-exact but valid matches (within tolerance)
        valid_indices_in_ds1 = non_exact_indices[valid_mask]
        valid_indices_in_ds2 = nearest_indices[valid_mask]

        full_array[valid_indices_in_ds1] = ds2[var].values[valid_indices_in_ds2]

        # Store variable
        new_vars[var] = (("time",), full_array)

    # Add new variables to ds1
    ds1 = ds1.assign(**new_vars)

    return ds1


def select_closest_times(
    ds_search: xr.Dataset,
    ds_goal: xr.Dataset,
    diff_min: int = 10,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Select nearest times from one dataset to match another within a tolerance.

    Parameters
    ----------
    ds_search : xr.Dataset
        Dataset from which to select nearest times.
    ds_goal : xr.Dataset
        Dataset providing the target time stamps.
    diff_min : int, optional
        Maximum allowed time difference in minutes.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        `(ds_search_sel, ds_goal_sel)` with matching `time` coordinates. The
        selected search dataset includes an `old_time` variable with the
        original time stamps.
    """

    # Ensure both datasets have datetime64[ns] time coordinates
    # ds_search['time'] = xr.decode_cf(ds_search).time
    # ds_goal['time']   = xr.decode_cf(ds_goal).time

    goal_times = ds_goal.time.values
    #goal_times = np.array(ds_goal.time.values, dtype='datetime64[ns]')

    nearest_times = ds_search.time.sel(time=goal_times, method='nearest').values

    diff = abs(nearest_times - goal_times)

    mask = diff <= np.timedelta64(diff_min, 'm')

    valid_nearest_times = nearest_times[mask]
    valid_goal_times = goal_times[mask]

    ds_search_sel = ds_search.sel(time=valid_nearest_times).copy(deep=True)
    ds_goal_sel = ds_goal.sel(time=valid_goal_times).copy(deep=True)

    ds_search_sel['old_time'] = ds_search_sel['time']
    ds_search_sel['time'] = valid_goal_times

    return ds_search_sel, ds_goal_sel
