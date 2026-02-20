"""
Functions for thermodynamic calculations such as saturation vapor pressure,
latent heat, absolute humidity, and dewpoint temperature.
"""

import numpy as np
import xarray as xr

from openMWR import consts

def saturation_vapor_pressure(T):
    """
    Calculate saturation vapor pressure using the Magnus equation.

    Parameters
    ----------
    T : float or ndarray
        Air temperature in kelvin.

    Returns
    -------
    e_s : float or ndarray
        Saturation vapor pressure in pascals.
    """
    # Convert temperature from kelvin to degrees Celsius
    T_C = T - consts.T_0

    # Apply Magnus equation to compute saturation vapor pressure
    e_s = 611.2 * np.exp(17.62 * T_C / (T_C + 243.12))

    return e_s

def latent_heat(T: np.ndarray) -> np.ndarray:
    """
    Calculate the latent heat of vaporization as a function of temperature.

    Parameters
    ----------
    T : ndarray
        Temperature in kelvin.

    Returns
    -------
    L : ndarray
        Latent heat of vaporization in joules per kilogram (J/kg).

    Notes
    -----
    Formula from Rogers and Yau, 1989: A Short Course in Cloud Physics
    (III.Ed.), Pergamon Press, 293p. Page 15
    """
    L = consts.L_0 - (consts.c_w - consts.c_pv) * (T - consts.T_0)

    return L


def calc_absolute_humidity(T, rh):
    """
    Calculate absolute humidity from temperature and relative humidity.

    Parameters
    ----------
    T : float or ndarray
        Air temperature in kelvin.
    rh : float or ndarray
        Relative humidity in percent (%).

    Returns
    -------
    ah : float or ndarray
        Absolute humidity in kg/m³.
    """
    ah = rh / 100 * saturation_vapor_pressure(T) / (consts.R_v * T)

    return ah


def dewpoint(e):
    """
    Calculate dewpoint temperature from vapor pressure using the reversed
    Magnus equation.

    Parameters
    ----------
    e : float, ndarray, or xr.DataArray
        Water vapor pressure in pascals (Pa).

    Returns
    -------
    T_D : float, ndarray, or xr.DataArray
        Dewpoint temperature in kelvin (K).
    """
    # Ensure non-positive vapor pressures become NaN
    if isinstance(e, xr.DataArray):
        e = e.where(e > 0)
    else:
        e = np.where(e > 0, e, np.nan)

    # Compute dewpoint using inverse Magnus formula
    log = np.log(e / 611.2)
    T_D = 243.12 * log / (17.62 - log) + consts.T_0

    return T_D

def dewpoint_from_relative_humidity(T, rh):
    """
    Calculate dewpoint temperature from temperature and relative humidity.

    Parameters
    ----------
    T : float or ndarray
        Air temperature in kelvin.
    rh : float or ndarray
        Relative humidity in percent (%).

    Returns
    -------
    T_D : float or ndarray
        Dewpoint temperature in kelvin (K).
    """
    # Convert RH to fraction and compute vapor pressure
    e = rh / 100 * saturation_vapor_pressure(T)

    # Convert vapor pressure to dewpoint temperature
    T_D = dewpoint(e)

    return T_D


def dewpoint_from_absolute_humidity(T, ah):
    """
    Calculate dewpoint temperature from temperature and absolute humidity.

    Parameters
    ----------
    T : float or ndarray
        Air temperature in kelvin.
    ah : float or ndarray
        Absolute humidity in kg/m³.

    Returns
    -------
    T_D : float or ndarray
        Dewpoint temperature in kelvin (K).
    """
    # Convert absolute humidity to vapor pressure
    e = ah * consts.R_v * T

    # Convert vapor pressure to dewpoint temperature
    T_D = dewpoint(e)

    return T_D

def calc_pressure(surface_p, T, z):
    r"""
    Compute the hydrostatic pressure profile from surface pressure, temperature,
    and geometric height. Supports both NumPy arrays (1D calculation) and
    xarray.DataArray inputs (vectorized), using a self-dispatching mechanism.

    Parameters
    ----------
    surface_p : float or xr.DataArray
        Surface pressure at the lowest level in pascals (Pa).
    T : ndarray or xr.DataArray
        Temperature profile in kelvin (K). Must contain a vertical dimension
        named ``'height'`` when used with xarray.
    z : ndarray or xr.DataArray
        Geometric height levels in meters (m). Must share the ``'height'``
        dimension with `T` when using xarray.

    Returns
    -------
    p : ndarray or xr.DataArray
        Pressure profile in pascals (Pa), evaluated at each height level.

    Notes
    -----
    - For xarray inputs, the function automatically switches into a vectorized
      mode using :func:`xarray.apply_ufunc`. Inside this process, it calls
      itself again, but with NumPy arrays extracted from the DataArrays.
    - The underlying 1D computation follows the hydrostatic equation:

      .. math::
        p(z_i) = p_0\, \exp\!\left(
            - \sum_{k=0}^{i-1}
            \frac{g\, \Delta z_k}{R_d\, T_{\text{mean},k}}
        \right)

      where :math:`T_{mean}` is the layer-mean temperature.
    """

    # -------- CASE 1: xarray input → vectorized dispatch via apply_ufunc --------
    if isinstance(surface_p, xr.DataArray) or isinstance(T, xr.DataArray) or isinstance(z, xr.DataArray):
        return xr.apply_ufunc(
            calc_pressure,       # self-dispatching call
            surface_p,
            T,
            z,
            input_core_dims=[[], ['height'], ['height']],
            output_core_dims=[['height']],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    # -------- CASE 2: NumPy inputs → perform actual 1D hydrostatic computation --------
    delta_z = np.diff(z)

    # Layer-mean temperature
    T_mean = (T[:-1] + T[1:]) / 2

    # Hydrostatic exponent for each layer
    exponent = -consts.g * delta_z / (consts.R_d * T_mean)

    # Initialize output array and integrate upward
    p = np.empty(z.size)
    p[0] = surface_p
    p[1:] = surface_p * np.exp(np.cumsum(exponent))

    return p