from __future__ import annotations

"""Thermodynamic helper functions for atmospheric moisture variables."""

import torch

from . import consts


def saturation_vapor_pressure(T: torch.Tensor, formula: str = "magnus", ice: bool = False) -> torch.Tensor:
    """Compute saturation vapor pressure.

    Parameters
    ----------
    T : torch.Tensor
        Temperature in K.
    formula : str, optional
        Saturation formula to use. Supported values are ``"magnus"`` and
        ``"goff_gratch"``.
    ice : bool, optional
        If ``True``, apply ice coefficients/branch where supported by the
        selected formula.

    Returns
    -------
    torch.Tensor
        Saturation vapor pressure in Pa.

    Raises
    ------
    ValueError
        If ``formula`` is not recognized.

    Notes
    -----
    The Goff-Gratch helper computes in hPa and is converted to Pa in this
    dispatcher.
    """
    formula_l = formula.lower()
    if formula_l == "magnus":
        return saturation_vapor_pressure_magnus(T, ice=ice)
    if formula_l == "goff_gratch":
        # goff_gratch_saturation_vapor_pressure returns hPa
        return goff_gratch_saturation_vapor_pressure(T, ice=ice) * 100.0
    raise ValueError(f"Unknown saturation vapor pressure formula '{formula}'.")

def vapor_pressure_from_relative_humidity(T: torch.Tensor, rh: torch.Tensor, formula: str = "goff_gratch", ice: bool = False) -> torch.Tensor:
    """Compute vapor pressure from relative humidity.

    Parameters
    ----------
    T : torch.Tensor
        Temperature in K.
    rh : torch.Tensor
        Relative humidity as a fraction (0-1).
    formula : str, optional
        Saturation-vapor-pressure formula passed to
        :func:`saturation_vapor_pressure`.
    ice : bool, optional
        Ice handling flag passed to :func:`saturation_vapor_pressure`.

    Returns
    -------
    torch.Tensor
        Vapor pressure in Pa.
    """
    return rh * saturation_vapor_pressure(T, formula=formula, ice=ice)

def goff_gratch_saturation_vapor_pressure(T: torch.Tensor, ice: bool = False) -> torch.Tensor:
    """Compute saturation vapor pressure using the Goff-Gratch formula.

    Parameters
    ----------
    T : torch.Tensor
        Temperature in K.
    ice : bool, optional
        If ``True``, apply the ice branch below 263.16 K, matching legacy
        behavior.

    Returns
    -------
    torch.Tensor
        Saturation vapor pressure in hPa.

    Notes
    -----
    This implementation mirrors the legacy pyrtlib ``RTEquation.vapor``
    formulation.
    """
    y = 373.16 / T
    es = (
        -7.90298 * (y - 1.0)
        + 5.02808 * torch.log10(y)
        - 1.3816e-07 * (torch.pow(10.0, 11.344 * (1.0 - (1.0 / y))) - 1.0)
        + 0.0081328 * (torch.pow(10.0, -3.49149 * (y - 1.0)) - 1.0)
        + torch.log10(torch.tensor(1013.246, dtype=T.dtype, device=T.device))
    )
    if ice:
        indx = torch.nonzero(T < 263.16, as_tuple=True)
        y = 273.16 / T[indx]
        es[indx] = (
            -9.09718 * (y - 1.0)
            - 3.56654 * torch.log10(y)
            + 0.876793 * (1.0 - (1.0 / y))
            + torch.log10(torch.tensor(6.1071, dtype=T.dtype, device=T.device))
        )

    return torch.pow(10.0, es)

def saturation_vapor_pressure_magnus(T: torch.Tensor, ice: bool = False) -> torch.Tensor:
    """Compute saturation vapor pressure using a Magnus-type formula.

    Parameters
    ----------
    T : torch.Tensor
        Temperature in K.
    ice : bool, optional
        If ``True``, use ice coefficients ``a=22.46`` and ``b=272.62``.
        Otherwise use water coefficients ``a=17.62`` and ``b=243.12``.

    Returns
    -------
    torch.Tensor
        Saturation vapor pressure in Pa.
    """
    T_c = T - consts.T_0
    if ice:
        a = torch.tensor(22.46, device=T.device, dtype=T.dtype)
        b = torch.tensor(272.62, device=T.device, dtype=T.dtype)
    else:
        a = torch.tensor(17.62, device=T.device, dtype=T.dtype)
        b = torch.tensor(243.12, device=T.device, dtype=T.dtype)
    es0 = torch.tensor(611.2, device=T.device, dtype=T.dtype)
    return es0 * torch.exp(a * T_c / (T_c + b))

def vapor_pressure_from_specific_humidity(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Compute vapor pressure from specific humidity.

    Parameters
    ----------
    q : torch.Tensor
        Specific humidity in kg/kg.
    p : torch.Tensor
        Total air pressure in Pa.

    Returns
    -------
    torch.Tensor
        Vapor pressure in Pa.

    Notes
    -----
    Uses :math:`e = q p / (\\epsilon + (1-\\epsilon)q)` with
    :math:`\\epsilon = R_d / R_v`.
    """

    eps = torch.tensor(consts.eps, device=q.device, dtype=q.dtype)
    return q * p / (eps + (1 - eps) * q)
