import numpy as np
import xarray as xr

def cal_ME(ds1, ds2, dims_to_mean):
    r"""
    Calculate Mean Error (ME) between two xarray datasets.

    .. math::

        \text{ME} = \frac{1}{n} \sum_{i=1}^n \left( x_i - \hat{x}_i \right) 

    """
    return (ds1 - ds2).mean(dim=dims_to_mean, skipna=True)

def cal_MRE(ds1, ds2, dims_to_mean):
    r"""
    Calculate Mean Relative Error (MRE) between two xarray datasets.

    .. math::

        \text{MRE} = \frac{1}{n} \sum_{i=1}^n \frac{x_i - \hat{x}_i}{\hat{x}_i}

    """
    return ((ds1 - ds2)/ds2).mean(dim=dims_to_mean, skipna=True)

def cal_MAE(ds1, ds2, dims_to_mean):
    r"""
    Calculate Mean Absolute Error (MAE) between two xarray datasets.

    .. math::

        \text{MAE} = \frac{1}{n} \sum_{i=1}^n \left| x_i - \hat{x}_i \right|

    """
    return np.abs(ds1 - ds2).mean(dim=dims_to_mean, skipna=True)

def cal_MARE(ds1, ds2, dims_to_mean):
    r"""
    Calculate Mean Absolute Relative Error (MARE) between two xarray datasets.

    .. math::

        \text{MARE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{x_i - \hat{x}_i}{\hat{x}_i} \right|

    """
    return np.abs((ds1 - ds2)/ds2).mean(dim=dims_to_mean, skipna=True)

def cal_MSE(ds1, ds2, dims_to_mean):
    r"""
    Calculate Mean Square Error (MSE) between two xarray datasets.

    .. math::

        \text{MSE} = \frac{1}{n} \sum_{i=1}^n \left( x_i - \hat{x}_i \right)^2

    """
    return ((ds1 - ds2) ** 2).mean(dim=dims_to_mean, skipna=True)

def cal_RMSE(ds1, ds2, dims_to_mean):
    r"""
    Calculate Root Mean Square Error (RMSE) between two xarray datasets.

    .. math::

        \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( x_i - \hat{x}_i \right)^2}

    """
    return np.sqrt(((ds1 - ds2) ** 2).mean(dim=dims_to_mean, skipna=True))

def _get_error_func(name: str):
    fn = globals().get(f"cal_{name}")
    if fn is None:
        raise ValueError(f"Unknown error type: {name}")
    return fn

def calculate_errors(ds1, ds2, error_types = ['ME', 'MAE', 'RMSE']):
    """
    Compute error metrics between two xarray datasets.

    Parameters
    ----------
    ds1 : xarray.Dataset
        First dataset (typically predictions or estimates).
    ds2 : xarray.Dataset
        Second dataset (typically reference or truth).
    error_types : list of str, optional
        List of error metric names to compute. Each name must correspond
        to an existing ``cal_<name>`` function. Default is ['ME', 'MAE', 'RMSE'].
        Options include: 'ME', 'MRE', 'MAE', 'MARE', 'RMSE'.

    Returns
    -------
    tuple of xarray.Dataset
        A pair of datasets:
        
        - ``errors`` : Dataset containing error metrics for all common variables
          (excluding predefined input variables). Metrics are averaged over 
          ``time`` or ``time`` and ``station`` depending on dataset dims.
        - ``errors_height`` : Dataset containing error metrics for variables
          with a ``height`` dimension, computed both with and without this 
          dimension. Variables without ``height`` are omitted.

    Notes
    -----
    Only variables present in both datasets (and not listed as input variables)
    are evaluated. Error functions are dynamically retrieved via
    :func:`get_error_func`.
    """
    error_func = [_get_error_func(err) for err in error_types]
    # Avoid pandas StringDtype indexes; xarray alignment can fail on concat with them.
    error_labels = np.asarray(error_types, dtype=str)
    
    error_dict = {}
    error_dict_height = {}

    common_vars = list(set(ds1.data_vars).intersection(ds2.data_vars))
    
    input_vars = ['input', 'TB', 'TB_IR', 'surface_T', 'surface_p', 'surface_rh', 'doy_cos', 'doy_sin']
    common_vars = [var for var in common_vars if var not in input_vars]


    dims_to_mean = ['time', 'station'] if ('station' in ds1.dims) or ('station' in ds2.dims) else ['time']

    
    for var in common_vars:
        if 'height' in ds1[var].dims:
            errors = [f(ds1[var], ds2[var], dims_to_mean + ['height']) for f in error_func]
            
            errors_height = [f(ds1[var], ds2[var], dims_to_mean) for f in error_func]
            
            error_dict_height[var] = xr.concat(errors_height, dim='error').assign_coords(error=('error', error_labels))
        
        else:
            errors = [f(ds1[var], ds2[var], dims_to_mean) for f in error_func]
                  
        
        error_dict[var] = xr.concat(errors, dim='error', join='outer').assign_coords(error=('error', error_labels))
        

    return xr.Dataset(error_dict), xr.Dataset(error_dict_height)
