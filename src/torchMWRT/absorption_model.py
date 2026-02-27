# -*- coding: utf-8 -*-
"""
Torch implementations of microwave absorption models.

This module ports pyrtlib absorption formulations to PyTorch so they can be
evaluated on batched tensors and used in differentiable radiative-transfer
workflows.
"""

from typing import Optional, Tuple, Union

import torch

from . import consts
from .lineshape import H2OLL, O2LL, O3LL

def _dilec12(f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the complex dielectric constant of liquid water.

    Parameters
    ----------
    f : torch.Tensor
        Frequency in GHz.
    t : torch.Tensor
        Temperature in K.

    Returns
    -------
    torch.Tensor
        Complex dielectric constant with a negative imaginary part for
        dissipation.

    Notes
    -----
    This is the torchMWRT port of ``pyrtlib.utils.dilec12`` (formerly public
    ``dilec12`` in ``utils.py``). It keeps the same three-term structure:
    static dielectric behavior, Debye relaxation, and the Rosenkranz B-band
    correction using complex logarithms. The original pyrtlib note reports
    validation ranges of approximately 20-220 GHz for 248-273 K and 1-1000 GHz
    for 273-330 K.

    Provenance: the physical description, validation ranges, and reference
    mapping are taken from the docstring of ``pyrtlib.utils.dilec12``.

    References
    ----------
    - :cite:alp:`Rosenkranz-2015`, IEEE Trans. Geosci. Remote Sens., v.53(3), pp.1387-1393
    """
    tc = t - 273.15
    z = torch.complex(torch.zeros_like(f), f)
    theta = 300.0 / t

    kappa = (
        -43.7527 * theta ** 0.05
        + 299.504 * theta ** 1.47
        - 399.364 * theta ** 2.11
        + 221.327 * theta ** 2.31
    )

    delta = 80.69715 * torch.exp(-tc / 226.45)
    sd = 1164.023 * torch.exp(-651.4728 / (tc + 133.07))
    kappa = kappa - (delta * z) / (sd + z)

    delta = 4.008724 * torch.exp(-tc / 103.05)
    hdelta = delta / 2.0
    f1 = (
        10.46012
        + (0.1454962 * tc)
        + (0.063267156 * tc ** 2)
        + (0.00093786645 * tc ** 3)
    )
    z1 = torch.complex(torch.full_like(f, -0.75), torch.ones_like(f)) * f1
    z2 = torch.complex(torch.full_like(f, -4500.0), torch.full_like(f, 2000.0))
    cnorm = torch.log(z2 / z1)
    chip = (hdelta * torch.log((z - z2) / (z - z1))) / cnorm
    chij = (hdelta * torch.log((z - torch.conj(z2)) / (z - torch.conj(z1)))) / torch.conj(
        cnorm
    )
    dchi = chip + chij - delta
    return kappa + dchi


def _dcerror(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Evaluate a sixth-order approximation of the complex error function.

    Parameters
    ----------
    x : torch.Tensor
        Real part of the complex argument.
    y : torch.Tensor
        Imaginary part of the complex argument.

    Returns
    -------
    torch.Tensor
        Approximation of :math:`w(z) = e^{-z^2}\mathrm{erfc}(-iz)` where
        :math:`z = x + iy`.

    Notes
    -----
    This function ports ``pyrtlib.utils._dcerror`` to torch operations. In
    pyrtlib, the user-visible helper lived in ``utils.py`` as ``dcerror``.
    The implementation uses the same rational polynomial coefficients and
    applies quadrant handling with ``torch.where``. The approximation follows
    the sixth-order form from Hui, Armstrong, and Wray (1978).

    Provenance: the function definition statement and cited approximation
    source are taken from the docstring of ``pyrtlib.utils._dcerror``.

    """
    dtype = x.dtype
    device = x.device
    a = torch.tensor(
        [
            122.607931777104326,
            214.382388694706425,
            181.928533092181549,
            93.155580458138441,
            30.180142196210589,
            5.912626209773153,
            0.564189583562615,
        ],
        dtype=dtype,
        device=device,
    )
    b = torch.tensor(
        [
            122.607931773875350,
            352.730625110963558,
            457.334478783897737,
            348.703917719495792,
            170.354001821091472,
            53.992906912940207,
            10.479857114260399,
        ],
        dtype=dtype,
        device=device,
    )

    zh = torch.complex(torch.abs(y), -x)
    asum = (
        (((((a[6] * zh + a[5]) * zh + a[4]) * zh + a[3]) * zh + a[2]) * zh + a[1])
        * zh
        + a[0]
    )
    bsum = (
        (((((zh + b[6]) * zh + b[5]) * zh + b[4]) * zh + b[3]) * zh + b[2]) * zh
        + b[1]
    ) * zh + b[0]
    w = asum / bsum

    z = torch.complex(x, y)
    alt = 2.0 * torch.exp(-(z ** 2)) - w
    return torch.where(y >= 0, w, alt)


class AbsModelError(Exception):
    """Error raised for invalid absorption-model configuration.

    Parameters
    ----------
    model : str
        Model identifier that triggered the error.
    message : str
        Human-readable description of the issue.
    """

    def __init__(self, model, message):
        self.model = model
        self.message = message

        super().__init__(self.message)


class AbsModel:
    """Base class with shared absorption-model configuration.

    Parameters
    ----------
    model : str
        Absorption model identifier (for example ``"R24"`` or ``"MWL24"``).
    dtype : torch.dtype
        Default dtype for internal constants.
    device : torch.device or str
        Device for internal constants and line-list tensors.
    freqs : torch.Tensor
        Frequency grid in GHz associated with this model instance.
    amu : dict, optional
        Optional uncertainty/tuning parameter overrides.
    """

    def __init__(self, model: str, dtype: torch.dtype, device, freqs: torch.Tensor, amu=None) -> None:
        super(AbsModel, self).__init__()
        self.model = model
        self.dtype = dtype
        self.device = device
        self.amu = amu
        self.freqs = freqs
        self._db2np = torch.log(torch.tensor(10.0, device=self.device, dtype=self.dtype)) * 0.1
        self._rvap = torch.tensor((0.01 * 8.31451) / 18.01528, device=self.device, dtype=self.dtype)

class LiqAbsModel(AbsModel):
    """Liquid-water absorption model family."""

    def __init__(self, model: str, dtype: torch.dtype, device, freqs: torch.Tensor, amu=None) -> None:
        super(LiqAbsModel, self).__init__(model, dtype=dtype, device=device, freqs=freqs, amu=amu)

    def liquid_water_absorption(self, water: torch.Tensor, freq: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        """Compute absorption by suspended liquid water droplets in Np/km.

        Parameters
        ----------
        water : torch.Tensor
            Liquid water content in g/m^3 (mass of liquid water per volume of
            dry air).
        freq : torch.Tensor
            Frequency in GHz.
        temp : torch.Tensor
            Temperature in K.

        Returns
        -------
        torch.Tensor
            Liquid-water absorption in Np/km.

        Raises
        ------
        ValueError
            If ``self.model`` is not implemented for liquid water absorption.

        Notes
        -----
        Model selection is consistent with pyrtlib:
        ``R03``/``R98`` use the older double-Debye form, while newer models use
        ``_dilec12``. In torchMWRT, non-positive ``water`` values are masked to
        zero with ``torch.where`` instead of scalar early return.

        Revision history from the original Rosenkranz routine:

        - PWR 08/03/92 Original Version
        - PWR 12/14/98 Temp dependence of eps2 eliminated to agree with MPM93
        - PWR 06/05/15 Using ``dilec12`` for complex dielectric constant

        Provenance: the revision-history bullets and reference list are taken
        from the docstring of
        ``pyrtlib.absorption_model.LiqAbsModel.liquid_water_absorption``.

        References
        ----------
        - :cite:alp:`Liebe-Hufford-Manabe`
        - :cite:alp:`Liebe-Hufford-Cotton`
        - :cite:alp:`Rosenkranz-1988`
        - :cite:alp:`Rosenkranz-2015`, IEEE Trans. Geosci. Remote Sens., v.53(3), pp.1387-1393
        """
        water_t = water
        freq_t = freq
        temp_t = temp

        has_water = water_t > 0

        if self.model in ['R03', 'R98']:
            theta1 = 1.0 - 300.0 / temp_t
            eps0 = 77.66 - 103.3 * theta1
            eps1 = 0.0671 * eps0
            eps2 = 3.52
            fp = (316.0 * theta1 + 146.4) * theta1 + 20.2
            if self.model == 'R03':
                fp = 20.1 * torch.exp(7.88 * theta1)
            fs = 39.8 * fp
            den_fp = torch.complex(torch.ones_like(freq_t), freq_t / fp)
            den_fs = torch.complex(torch.ones_like(freq_t), freq_t / fs)
            eps = (eps0 - eps1) / den_fp + (eps1 - eps2) / den_fs + eps2
        elif self.model in ['R17', 'R16', 'R19', 'R20', 'R19SD', 'R22SD', 'R23', 'R23SD', 'R24']:
            eps = _dilec12(freq_t, temp_t)
        else:
            raise ValueError(
                '[AbsLiq] No model available with this name: {} . Sorry...'.format(self.model))

        re = (eps - 1.0) / (eps + 2.0)

        abliq = -0.06286 * torch.imag(re) * freq_t * water_t
        abliq = torch.where(has_water, abliq, torch.zeros_like(abliq))

        return abliq


class N2AbsModel(AbsModel):
    """Nitrogen collision-induced absorption model family."""

    def __init__(self, model: str, dtype: torch.dtype, device, freqs: torch.Tensor, amu=None) -> None:
        super(N2AbsModel, self).__init__(model, dtype=dtype, device=device, freqs=freqs, amu=amu)
        base_device = self.device if self.device is not None else self.freqs.device
        base_dtype = self.dtype if self.dtype is not None else self.freqs.dtype
        self._n2_mwl24_coeffs = {
            "a_e": torch.tensor([1.4359, -0.0005385, 1.6182e-7, 1.8845e-10], device=base_device, dtype=base_dtype),
            "b_e": torch.tensor([0.07988, -0.0005165, 1.131e-6, -5.738e-10], device=base_device, dtype=base_dtype),
            "a0e": torch.tensor(1.754e-18, device=base_device, dtype=base_dtype),
            "a1e": torch.tensor(1.033, device=base_device, dtype=base_dtype),
            "a2e": torch.tensor(0.1510, device=base_device, dtype=base_dtype),
            "a3e": torch.tensor(0.1420, device=base_device, dtype=base_dtype),
            "f1e": torch.tensor(44.0, device=base_device, dtype=base_dtype),
            "f2e": torch.tensor(568.0, device=base_device, dtype=base_dtype),
            "f3e": torch.tensor(358.0, device=base_device, dtype=base_dtype),
            "f4e": torch.tensor(127.0, device=base_device, dtype=base_dtype),
            "f5e": torch.tensor(92.0, device=base_device, dtype=base_dtype),
        }

    def n2_absorption_mwl24(self, t: torch.Tensor, p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Evaluate the MWL24 :math:`N_2-N_2` collision-induced absorption term.

        Parameters
        ----------
        t : torch.Tensor
            Air temperature in K.
        p : torch.Tensor
            Dry-air pressure in mbar.
        f : torch.Tensor
            Frequency in GHz.

        Returns
        -------
        torch.Tensor
            Pair-absorption coefficient in 1/cm.

        Notes
        -----
        The implementation follows the MWL24 parametrization based on classical
        trajectory calculations and keeps pyrtlib constants. torchMWRT
        pre-stores polynomial coefficients as tensors to avoid repeated host
        allocations.

        To convert to Np/km multiply by ``1e5``. To convert pair absorption to
        dry-air absorption, multiply by ``0.84``.

        Provenance: the model description and conversion guidance are taken from
        the docstring of
        ``pyrtlib.absorption_model.N2AbsModel.n2_absorption_mwl24``.

        References
        ----------
        - :cite:alp:`Serov-2024`
        - :cite:alp:`Meshkov-DeLucia-2007`
        """
        t0 = 300.
        ti = t0 / t
        p_torr = p * 0.7500616827

        kbcm = 0.6950354549  # boltzmann in (1/cm)/K

        d0 = 108. / (kbcm * t)
        d_ref = 108. / (kbcm * 300.)
        potential = (torch.exp(d0) - (1 + d0)) / (torch.exp(t.new_tensor(d_ref)) - (1 + d_ref))

        coeffs = self._n2_mwl24_coeffs
        f_powers = torch.stack(
            [torch.ones_like(f), f, f * f, f * f * f]
        )
        m0 = torch.tensordot(coeffs["a_e"], f_powers, dims=1)
        aa = torch.tensordot(coeffs["b_e"], f_powers, dims=1)
        t2 = coeffs["a3e"] * torch.exp(-(f / coeffs["f5e"] - 1) ** 2)
        t1_denom = 1. + ((f - coeffs["f1e"]) / coeffs["f2e"]) ** 2
        t1_num_denom = 1. + ((f - coeffs["f3e"]) / coeffs["f4e"]) ** 2
        t1_num = 1 + coeffs["a2e"] / t1_num_denom
        spec_fun = coeffs["a0e"] * 0.5 * (1 + coeffs["a1e"] * t1_num / t1_denom + t2)

        t_dep = ti ** (m0 + aa * torch.log(ti))

        return spec_fun * t_dep * potential * p_torr**2 * f**2

    def n2_absorption(self, t: torch.Tensor, p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Compute nitrogen continuum absorption in Np/km.

        Parameters
        ----------
        t : torch.Tensor
            Temperature in K.
        p : torch.Tensor
            Dry-air pressure in mbar.
        f : torch.Tensor
            Frequency in GHz.

        Returns
        -------
        torch.Tensor
            Nitrogen continuum absorption in Np/km.

        Raises
        ------
        ValueError
            If ``self.model`` is not implemented.

        Notes
        -----
        The factor ``n`` accounts for additional pair processes
        (:math:`O_2-O_2` and :math:`O_2-N_2`) for model families where that
        correction is prescribed. ``MWL24`` uses
        :meth:`n2_absorption_mwl24`; other models use the Rosenkranz-style
        empirical power law with optional high-frequency taper ``fdepen``.

        Provenance: the model intent and the Eq. 2.6 pointer in References are
        taken from the docstring of
        ``pyrtlib.absorption_model.N2AbsModel.n2_absorption``.

        References
        ----------
        - :cite:alp:`Maetzler-Rosenkranz-2006`, Eq. 2.6
        - :cite:alp:`Borysow-Frommhold-1986`
        - :cite:alp:`Boissoles-2003`
        """
        fdepen = 0.5 + 0.5 / (1.0 + (f / 450.0) ** 2)

        th = 300.0 / t
        if self.model in ['R16', 'R17', 'R18', 'R19', 'R19SD']:
            l, m, n = 6.5e-14, 3.6, 1.34
        elif self.model in ['R20', 'R20SD', 'R21SD', 'R22', 'R22SD', 'R23', 'R23SD', 'R24']:
            l, m, n = 9.95e-14, 3.22, 1
        elif self.model == 'R03':
            l, m, n = 6.5e-14, 3.6, 1.29
        elif self.model == 'MWL24':
            l, m, n = 9.95e-14, 3.22, 0.84
        elif self.model in ['R98']:
            l, m, n = 6.4e-14, 3.55, 1
            fdepen = 1
        else:
            raise ValueError(
                '[AbsN2] No model available with this name: {} . Sorry...'.format(self.model))

        if self.model == 'MWL24':
            bf = self.n2_absorption_mwl24(t, p, f) * 1.E5
        else:
            bf = l * fdepen * p * p * f * f * th ** m

        abs_n2 = n * bf

        return abs_n2


class H2OAbsModel(AbsModel):
    """Water-vapor absorption model family."""

    def __init__(self, model: str, dtype: torch.dtype, device, freqs: torch.Tensor, amu=None) -> None:
        super(H2OAbsModel, self).__init__(model, dtype=dtype, device=device, freqs=freqs, amu=amu)

        self._implemented_models = H2OLL.available_models()

        if model not in self._implemented_models:
            raise AbsModelError(model, f"Model {model} is not available for H2O absorption. Please choose from {self._implemented_models}")

        self.h2oll = H2OLL.load(self.model, device=self.device, dtype=self.dtype)

    def h2o_continuum(self, frq: torch.Tensor, vx: torch.Tensor, nfreq: int) -> torch.Tensor:
        """Compute the MT-CKD-based self-continuum helper term.

        Parameters
        ----------
        frq : torch.Tensor
            Frequency in GHz.
        vx : torch.Tensor
            Normalized temperature :math:`300/T`.
        nfreq : int
            Kept for API compatibility with pyrtlib. The torch implementation
            is fully vectorized and does not need this value for iteration.

        Returns
        -------
        torch.Tensor
            Self-continuum coefficient in ``(1/cm)/mbar^2``.

        Notes
        -----
        The implementation uses a cubic interpolation over six tabulated nodes
        adapted from ``mt_ckd_h2o_module.f90`` (MT-CKD 4.1 fit).

        Provenance: this description is adapted from the docstring of
        ``pyrtlib.absorption_model.H2OAbsModel.h2o_continuum``.
        """
        nf = 6
        deltaf = 299.792458
        selfcon = torch.tensor([2.877e-21, 2.855e-21, 2.731e-21,
                                2.49e-21, 2.178e-21, 1.863e-21],
                               dtype=frq.dtype, device=frq.device)
        selftexp = torch.tensor([6.413, 6.414, 6.275, 6.049, 5.789, 5.557],
                                dtype=frq.dtype, device=frq.device)
        t = 300.0 / vx
        theta = 296./t

        # Build interpolation coefficients with an explicit leading node axis
        # so this works for scalar, profile, and fully-broadcasted frequency grids.
        node_shape = (nf,) + (1,) * theta.dim()
        theta_exp = theta.unsqueeze(0)
        selfcon_exp = selfcon.view(node_shape)
        selftexp_exp = selftexp.view(node_shape)

        a = 6.532e+12 * selfcon_exp * theta_exp ** (selftexp_exp + 3.0)
        a = torch.cat([a[1:2], a], dim=0)

        fj = frq / deltaf
        j_idx = torch.floor(fj).to(torch.long)
        j_idx = torch.clamp(j_idx, min=0, max=nf - 2)
        p = fj - j_idx.to(fj.dtype)
        c = (3. - 2. * p) * p * p
        b = 0.5 * p * (1. - p)
        b1 = b * (1. - p)
        b2 = b * p

        gather_idx = j_idx.unsqueeze(0)
        a0 = torch.take_along_dim(a, gather_idx, dim=0).squeeze(0)
        a1 = torch.take_along_dim(a, gather_idx + 1, dim=0).squeeze(0)
        a2 = torch.take_along_dim(a, gather_idx + 2, dim=0).squeeze(0)
        a3 = torch.take_along_dim(a, gather_idx + 3, dim=0).squeeze(0)
        cs = -a0 * b1 + a1 * (1. - c + b2) + a2 * (c + b1) - a3 * b2
        return cs

    def h2o_continuum_mwl24(self, frq: torch.Tensor, vx: torch.Tensor) -> torch.Tensor:
        """Compute the MWL24 water-vapor self-continuum parametrization.

        Parameters
        ----------
        frq : torch.Tensor
            Frequency in GHz.
        vx : torch.Tensor
            Normalized temperature :math:`300/T`.

        Returns
        -------
        torch.Tensor
            Self-continuum coefficient scaled to ``(Np/km)/mbar^2``.

        Notes
        -----
        The model combines bound-dimer, metastable-dimer, and far-wing
        contributions. This helper already applies the ``1e5`` conversion used
        in pyrtlib, so callers should multiply only by ``pvap**2`` to obtain the
        continuum absorption contribution in Np/km.

        Provenance: the physical summary and reference selection are taken from
        the docstring of
        ``pyrtlib.absorption_model.H2OAbsModel.h2o_continuum_mwl24``.

        References
        ----------
        - :cite:alp:`Tretyakov-2025`
        - :cite:alp:`Odintsova-2022`
        - :cite:alp:`Galanina-2022`
        """

        atm2mbar = 1013.25
        t = 300.0 / vx
        ti_bd = 268.0 / t
        ti_md = 266.0 / t
        ti_fw = 296.0 / t
        # constants of bound dimers absorption (A0-5) and other components (see Tretyakov et al 2024)
        (A0, A1, A2, A3, A4, A5, N_D, C_W, N_BD, N_MD, N_FW, PF_FW) = (5.55E-8, 434.8,
                                                                       219.5, 4.93E-10,
                                                                       21.06, 1.34E-14,
                                                                       0.7, 8.5E-15,
                                                                       10.0, 2.9, 1.5, 2.2)
        # calculations for dimer equilibrium constants
        # second virial coef-t parameters
        b_svc = (-7.804242E+6, 8.345651E+4, -4.212794E+2, 1.242946, -2.409822E-3,
                 3.017768E-6, -2.518957E-9, 1.350628E-12, -4.134191E-16, 5.530774E-20)
        b_svc_t = torch.tensor(b_svc, device=frq.device, dtype=frq.dtype)
        power_shape = (len(b_svc),) + (1,) * t.dim()
        powers = torch.arange(len(b_svc), device=frq.device).reshape(power_shape)
        t_powers = t.unsqueeze(0) ** powers
        b_t = torch.tensordot(b_svc_t, t_powers, dims=1)
        b_t *= -(100./t)**6
        r_t = 82.05746 * t  # molar gas constant multiplied by temperature
        # K_BD from paper, in 1/atm
        k_bd = 4.7856E-4 * torch.exp(1669.8 / t - 5.10485E-3 * t)
        k_md = ((b_t - 30.5) / r_t - k_bd) / atm2mbar  # K_MD, in 1/mbar
        k_bd = k_bd / atm2mbar  # recalc to 1/mbar

        # bound dimers contribution

        CON_BD = A0 / (A1 * A1 + (frq - A2) * (frq - A2))
        CON_BD += A3 / (A4 * A4 + frq * frq) + A5
        CON_BD = CON_BD * ti_bd**N_BD

        # metastable dimers contribution
        S2MON = 1.E-13 * (4.06 + 7.12E-3 * frq) * (1 / atm2mbar)
        CON_MD = (S2MON * (1 - N_D) * ti_md**N_MD + N_D * CON_BD / k_bd) * k_md

        # far wings contribution (note that it uses not f**2 but some different power)
        CON_FW = C_W * frq**PF_FW * ti_fw**N_FW

        cs_mwl24 = ((CON_BD + CON_MD) * frq**2 + CON_FW)

        return cs_mwl24 * 1.E5

    def h2o_absorption(self, pdrykpa: torch.Tensor, vx: torch.Tensor, ekpa: torch.Tensor, frq: torch.Tensor, amu: Optional[dict] = None) -> Union[
            Tuple[torch.Tensor, torch.Tensor], None]:
        """Compute water-vapor line and continuum absorption terms.

        Parameters
        ----------
        pdrykpa : torch.Tensor
            Dry-air pressure in kPa.
        vx : torch.Tensor
            Normalized temperature :math:`300/T`.
        ekpa : torch.Tensor
            Water-vapor partial pressure in kPa.
        frq : torch.Tensor
            Frequency in GHz.
        amu : dict, optional
            Optional parameter overrides. Expected entries have a ``.value``
            attribute, matching the uncertainty workflow used in torchMWRT.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(npp, ncpp)`` where ``npp`` is the line term and ``ncpp`` is the
            continuum term, both in ppm.

        Notes
        -----
        Inputs are broadcast with ``torch.broadcast_tensors`` and line-list
        coefficients are moved to input device/dtype before computation.
        Compared with pyrtlib's scalar flow, torchMWRT avoids hard early returns
        for non-positive vapor density and instead masks outputs, preserving
        differentiability and shape consistency.

        Provenance: the base description and primary reference are taken from
        the docstring of
        ``pyrtlib.absorption_model.H2OAbsModel.h2o_absorption``. Additional torchMWRT notes
        document behavior changes in this port.

        References
        ----------
        - :cite:alp:`Rosenkranz-2017`
        - :cite:alp:`Tretyakov-2025`
        - :cite:alp:`Odintsova-2022`
        - :cite:alp:`Galanina-2022`
        """
        amu = self.amu if amu is None else amu
        pdrykpa, vx, ekpa, frq = torch.broadcast_tensors(pdrykpa, vx, ekpa, frq)
        f2 = frq * frq
        self.h2oll.to(device=pdrykpa.device, dtype=pdrykpa.dtype)
        if amu:
            tval = lambda value: pdrykpa.new_tensor(value)
            if self.model > 'R17':
                self.h2oll.cf = tval(amu['con_Cf'].value)
                self.h2oll.cs = tval(amu['con_Cs'].value)
            else:
                self.h2oll.cf = tval(amu['con_Cf'].value * amu['con_Cf_factr'].value)
                self.h2oll.cs = tval(amu['con_Cs'].value * amu['con_Cs_factr'].value)
            self.h2oll.xcf = tval(amu['con_Xf'].value)
            self.h2oll.xcs = tval(amu['con_Xs'].value)
            self.h2oll.s1 = tval(amu['S'].value)
            self.h2oll.b2 = tval(amu['B2'].value)
            self.h2oll.w0 = tval(amu['gamma_a'].value / 1000.0)
            self.h2oll.w0s = tval(amu['gamma_w'].value / 1000.0)
            self.h2oll.x = tval(amu['n_a'].value)
            self.h2oll.xs = tval(amu['n_w'].value)
            if self.model > 'R17':
                self.h2oll.sh = tval(amu['delta_a'].value / 1000.0)
                self.h2oll.shs = tval(amu['delta_w'].value / 1000.0)
                self.h2oll.xh = tval(amu['n_da'].value)
                self.h2oll.xhs = tval(amu['n_dw'].value)
            else:
                self.h2oll.sr = tval(amu['SR'].value)
            self.h2oll.fl = tval(amu['FL'].value)

        # the best-fit voigt are given in koshelev et al. 2018, table 2 (rad,
        # mhz/torr). these correspond to w3(1) and ws(1) in h2o_list_r18 (mhz/mb)

        # cyh ***********************************************
        db2np = self._db2np
        rvap = self._rvap
        factor = 0.182 * frq
        t = 300.0 / vx
        p = (pdrykpa + ekpa) * 10.0
        rho = ekpa * 10.0 / (rvap * t)
        f = frq
        # cyh ***********************************************

        rho_pos = rho > 0.0
        rho = torch.where(rho_pos, rho, torch.full_like(rho, 1e-12))

        pvap = (rho * t) / 216.68
        if self.model in ['R03', 'R16', 'R17', 'R98']:
            pvap = (rho * t) / 217.0
        if self.model in ['R22SD', 'R23SD', 'R24', 'MWL24']:
            pvap = (consts.R_v * 1e-05) * rho * t
        pda = p - pvap
        if self.model in ['R03', 'R16', 'R98']:
            den = 3.335e+16 * rho
        else:
            den = 3.344e+16 * rho
        # continuum terms
        ti = self.h2oll.reftcon / t
        # xcf and xcs include 3 for conv. to density & stimulated emission
        if self.model in ['R03', 'R98']:
            con = (5.43e-10 * pda * ti ** 3 + 1.8e-08 *
                   pvap * ti ** 7.5) * pvap * f2
        elif self.model == 'MWL24':
            con_self = self.h2o_continuum_mwl24(f, vx) * pvap * pvap
            # con_frgn = self.h2oll.cf * pda * pvap * f * f * ti**self.h2oll.xcf  # foreign continuum from earlier version
            con_frgn = 7.1e-10 * (300./t)**4.4 * f**1.96 * pda * pvap
            con = con_self + con_frgn
        else:
            con = (self.h2oll.cf * pda * ti ** self.h2oll.xcf + self.h2oll.cs * pvap * ti ** self.h2oll.xcs) * \
                pvap * f2
        # 2019/03/18 *********************************************************
        # add resonances
        nlines = len(self.h2oll.fl)
        line_shape = (nlines,) + (1,) * f.dim()

        def _line_param(param) -> torch.Tensor:
            return param.view(line_shape)

        ti = self.h2oll.reftline / t
        summ = torch.zeros_like(f, dtype=f.dtype)
        npp_cs = None
        f_exp = f.unsqueeze(0)
        pda_exp = pda.unsqueeze(0)
        pvap_exp = pvap.unsqueeze(0)
        ti_exp = ti.unsqueeze(0)
        w0 = _line_param(self.h2oll.w0)
        w0s = _line_param(self.h2oll.w0s)
        x = _line_param(self.h2oll.x)
        xs = _line_param(self.h2oll.xs)
        s1 = _line_param(self.h2oll.s1)
        b2 = _line_param(self.h2oll.b2)
        fl = _line_param(self.h2oll.fl)
        if self.model.startswith(('R19SD', 'R20SD', 'R21SD', 'R22SD', 'R23SD', 'R24', 'MWL24')):
            tiln = torch.log(ti)
            ti2 = torch.exp(2.5 * tiln)
            if self.model in ['R23SD', 'R24']:
                if self.h2oll.cs > 0:
                    con = self.h2oll.cs * ti * self.h2oll.xcs
                    npp_cs = con
                else:
                    npp_cs = self.h2o_continuum(frq, vx, 1)
            tiln_exp = tiln.unsqueeze(0)
            ti2_exp = ti2.unsqueeze(0)

            w2 = _line_param(self.h2oll.w2)
            w2s = _line_param(self.h2oll.w2s)
            sh = _line_param(self.h2oll.sh)
            shs = _line_param(self.h2oll.shs)
            xh = _line_param(self.h2oll.xh)
            xhs = _line_param(self.h2oll.xhs)
            aair = _line_param(self.h2oll.aair)
            aself = _line_param(self.h2oll.aself)

            width0 = w0 * pda_exp * ti_exp ** x + w0s * pvap_exp * ti_exp ** xs
            if self.model in ['R21SD', 'R22SD', 'R23SD', 'R24', 'MWL24']:
                xw2 = _line_param(self.h2oll.xw2)
                xw2s = _line_param(self.h2oll.xw2s)
                width2_alt = w2 * pda_exp * ti_exp ** xw2 + w2s * pvap_exp * ti_exp ** xw2s
                width2 = torch.where(w2 > 0, width2_alt, torch.zeros_like(width0))
            else:
                width2 = w2 * pda_exp + w2s * pvap_exp

            shiftf = sh * pda_exp * (1. - aair * tiln_exp) * ti_exp ** xh
            shifts = shs * pvap_exp * (1. - aself * tiln_exp) * ti_exp ** xhs
            shift = shiftf + shifts
            wsq = width0 ** 2
            s = s1 * ti2_exp * torch.exp(b2 * (1. - ti_exp))
            df0 = f_exp - fl - shift
            df1 = f_exp + fl + shift
            df_vec = torch.stack([df0, df1], dim=0)
            base = width0 / (562500.0 + wsq)

            delta2 = torch.zeros_like(width0)
            if self.model in ["R21SD", 'R22SD', 'R23SD', 'R24', 'MWL24']:
                d2 = _line_param(self.h2oll.d2)
                d2s = _line_param(self.h2oll.d2s)
                delta2 = d2 * pda_exp + d2s * pvap_exp
            if self.model == 'R20SD':
                idx_mask = (torch.arange(nlines, device=f.device).view(line_shape) == 1)
                special = (self.h2oll.d2air * pda + self.h2oll.d2self * pvap).unsqueeze(0)
                delta2 = torch.where(idx_mask, special, delta2)

            sd_mask = (width2 > 0) & (torch.abs(df_vec) < (10 * width0))
            if self.model in ["R21SD", 'R22SD', 'R23SD', 'R24', 'MWL24', 'R20SD']:
                xc = torch.complex(width0 - 1.5 * width2, df_vec + 1.5 * delta2) / torch.complex(width2, -delta2)
            else:
                xc = torch.complex(width0 - 1.5 * width2, df_vec) / width2
            xrt = torch.sqrt(xc)
            pxw = 1.77245385090551603 * xrt * _dcerror(-torch.imag(xrt), torch.real(xrt))
            denom = torch.complex(width2, -delta2) if self.model in ['R20SD', 'R21SD', 'R22SD', 'R23SD', 'R24', 'MWL24'] else width2
            sd = 2.0 * (1.0 - pxw) / denom
            res_sd = torch.real(sd) - base
            mask_local = torch.abs(df_vec) < 750.0
            res_lorentz = mask_local * (width0 / (df_vec ** 2 + wsq) - base)
            res = torch.where(sd_mask, res_sd, res_lorentz).sum(dim=0)
            summ = (s * res * (f_exp / fl) ** 2).sum(dim=0)
        elif self.model in ['R16', 'R03', 'R17', 'R98']:
            ti2 = ti ** 2.5
            ti2_exp = ti2.unsqueeze(0)

            widthf = w0 * pda_exp * ti_exp ** x
            widths = w0s * pvap_exp * ti_exp ** xs
            width = widthf + widths
            if self.model == 'R98':
                shift = torch.zeros_like(width)
            else:
                sr = _line_param(self.h2oll.sr)
                if self.model == 'R03':
                    shift = sr * width
                else:
                    shift = sr * widthf
            wsq = width ** 2
            s = s1 * ti2_exp * torch.exp(b2 * (1.0 - ti_exp))
            df0 = f_exp - fl - shift
            df1 = f_exp + fl + shift
            df_vec = torch.stack([df0, df1], dim=0)
            base = width / (562500.0 + wsq)
            mask = torch.abs(df_vec) <= 750.0
            res = (mask * (width / (df_vec ** 2 + wsq) - base)).sum(dim=0)
            summ = (s * res * (f_exp / fl) ** 2).sum(dim=0)
        elif self.model in ['R19', 'R20']:
            tiln = torch.log(ti)
            ti2 = ti ** 2.5
            tiln_exp = tiln.unsqueeze(0)
            ti2_exp = ti2.unsqueeze(0)

            sh = _line_param(self.h2oll.sh)
            shs = _line_param(self.h2oll.shs)
            xh = _line_param(self.h2oll.xh)
            xhs = _line_param(self.h2oll.xhs)
            aair = _line_param(self.h2oll.aair)
            aself = _line_param(self.h2oll.aself)

            widthf = w0 * pda_exp * ti_exp ** x
            widths = w0s * pvap_exp * ti_exp ** xs
            width = widthf + widths
            shiftf = sh * pda_exp * (1. - aair * tiln_exp) * ti_exp ** xh
            shifts = shs * pvap_exp * (1. - aself * tiln_exp) * ti_exp ** xhs
            shift = shiftf + shifts
            wsq = width ** 2
            s = s1 * ti2_exp * torch.exp(b2 * (1. - ti_exp))
            df0 = f_exp - fl - shift
            df1 = f_exp + fl + shift
            df_vec = torch.stack([df0, df1], dim=0)
            base = width / (562500.0 + wsq)
            mask = torch.abs(df_vec) < 750.0
            res = (mask * (width / (df_vec ** 2 + wsq) - base)).sum(dim=0)
            summ = (s * res * (f_exp / fl) ** 2).sum(dim=0)
        elif self.model == 'R18':
            ti2 = ti ** 2.5
            ti2_exp = ti2.unsqueeze(0)

            sh = _line_param(self.h2oll.sh)
            shs = _line_param(self.h2oll.shs)
            xh = _line_param(self.h2oll.xh)
            xhs = _line_param(self.h2oll.xhs)

            widthf = w0 * pda_exp * ti_exp ** x
            widths = w0s * pvap_exp * ti_exp ** xs
            width = widthf + widths
            shiftf = sh * pda_exp * ti_exp ** xh
            shifts = shs * pvap_exp * ti_exp ** xhs
            shift = shiftf + shifts
            wsq = width ** 2
            s = s1 * ti2_exp * torch.exp(b2 * (1. - ti_exp))
            df0 = f_exp - fl - shift
            df1 = f_exp + fl + shift
            df_vec = torch.stack([df0, df1], dim=0)
            base = width / (562500.0 + wsq)
            mask = torch.abs(df_vec) < 750.0
            res = (mask * (width / (df_vec ** 2 + wsq) - base)).sum(dim=0)
            summ = (s * res * (f_exp / fl) ** 2).sum(dim=0)
        # separate the following original equ. into line and continuum
        # terms, and change the units from np/km to ppm
        # abh2o = .3183e-4*den*sum + con

        if self.model in ['R23SD', 'R24']:
            conf = self.h2oll.cf * ti**self.h2oll.xcf
            con = (conf * pda + npp_cs * pvap) * pvap * f2

        h20m = frq.new_tensor(2.9915075E-23)  # mass of water molecule (g)
        if self.model in ['R22SD', 'R23SD']:
            npp = 1.e-10 * rho * summ / (torch.pi * h20m) / db2np / factor
        elif self.model == 'R24':
            npp = 1.e-10 * (rho/h20m) * (summ/torch.pi) / db2np / factor
        else:
            npp = (3.1831e-05 * den * summ / db2np) / factor

        ncpp = (con / db2np) / factor

        # zero-out where rho was non-positive to avoid hard return that would
        # break gradients
        npp = torch.where(rho_pos, npp, torch.zeros_like(npp))
        ncpp = torch.where(rho_pos, ncpp, torch.zeros_like(ncpp))

        return npp, ncpp


class O2AbsModel(AbsModel):
    """Oxygen absorption model family."""

    def __init__(self, model: str, dtype: torch.dtype, device, freqs: torch.Tensor, amu=None) -> None:
        super(O2AbsModel, self).__init__(model, dtype=dtype, device=device, freqs=freqs, amu=amu)

        self._implemented_models = O2LL.available_models()

        if model not in self._implemented_models:
            raise AbsModelError(model, f"Model {model} is not available for O2 absorption. Please choose from {self._implemented_models}")

        self.o2ll = O2LL.load(self.model, device=self.device, dtype=self.dtype)

    def o2_absorption(self, pdrykpa: torch.Tensor, vx: torch.Tensor, ekpa: torch.Tensor, frq: torch.Tensor, amu: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute oxygen line and continuum absorption terms.

        Parameters
        ----------
        pdrykpa : torch.Tensor
            Dry-air pressure in kPa.
        vx : torch.Tensor
            Normalized temperature :math:`300/T`.
        ekpa : torch.Tensor
            Water-vapor partial pressure in kPa.
        frq : torch.Tensor
            Frequency in GHz.
        amu : dict, optional
            Optional parameter overrides with ``.value`` fields.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(npp, ncpp)`` where ``npp`` is the oxygen line term and ``ncpp``
            is the non-resonant continuum term, both in ppm.

        Notes
        -----
        This routine keeps the Rosenkranz lineage and later revisions used in
        pyrtlib, including line-mixing and line-shape updates across
        R03/R16/R17/R18/R19/R20/R22/R23/R24 families.

        History (as documented in the original pyrtlib routine):

        - 5/1/95  P. Rosenkranz
        - 11/5/97  P. Rosenkranz - 1- line modification.
        - 12/16/98  pwr - updated submm freq's and intensities from HITRAN96
        - 8/21/02  pwr - revised width at 425
        - 3/20/03  pwr - 1- line mixing and width revised
        - 9/29/04  pwr - new widths and mixing, using HITRAN intensities for all lines
        - 6/12/06  pwr - chg. T dependence of 1- line to 0.8
        - 10/14/08  pwr - moved isotope abundance back into intensities, added selected O16O18 lines.
        - 5/30/09  pwr - remove common block, add weak lines.
        - 12/18/14  pwr - adjust line broadening due to water vapor.
        - 9/29/18  pwr - 2nd-order line mixing
        - 8/20/19  pwr - adjust intensities according to Koshelev meas.

        Line intensities follow HITRAN2004; non-resonant intensity follows JPL
        catalog values. For models where the continuum is folded into the line
        formulation (R19+ variants), ``ncpp`` is returned as zeros.

        The mm line-width coefficients are from Tretyakov et al. (2005),
        Makarov et al. (2008), and Koshelev et al. (2016); submm line-widths
        are from Golubiatnikov and Krupnov, except the 234-GHz line width from
        Drouin. Mixing coefficients follow Makarov's 2018 revision. The same
        temperature dependence ``(1/T)**X`` is used for submillimeter line
        widths as in the 60-GHz band.

        Provenance: the full history block and detailed spectroscopy notes are
        taken from the docstring of
        ``pyrtlib.absorption_model.O2AbsModel.o2_absorption``.

        References
        ----------
        - :cite:alp:`Rosenkranz-1993`, chapter 2 and appendix
        - :cite:alp:`Golubiatnikov-Krupnov-2003`, pp.282-287
        - :cite:alp:`Tretyakov-2004`, pp.31-38
        - :cite:alp:`Tretyakov-2005`, pp.1-14
        - :cite:alp:`Drouin-2007`, pp.450-458
        - :cite:alp:`Makarov-2008`, pp.242-243
        - :cite:alp:`Makarov-2011`, pp.1420-1428
        - :cite:alp:`Koshelev-2015`, pp.24-27
        - :cite:alp:`Koshelev-2016`, pp.91-95
        - :cite:alp:`Koshelev-2017`, pp.78-86
        - :cite:alp:`Makarov-2020`
        """

        amu = self.amu if amu is None else amu
        self.o2ll.to(device=pdrykpa.device, dtype=pdrykpa.dtype)

        if amu:
            tval = lambda value: pdrykpa.new_tensor(value)
            self.o2ll.w2a = tval(amu['w2a'].value)
            # self.o2ll.apu = amu['APU'].value
            self.o2ll.w300[0:38] = tval(amu['O2gamma'].value[0:38])
            self.o2ll.w300[34:49] = tval(amu['O2gamma_NL'].value[34:49])
            self.o2ll.wb300 = tval(amu['WB300'].value)
            self.o2ll.f = tval(amu['O2FL'].value)
            self.o2ll.s300 = tval(amu['O2S'].value)
            self.o2ll.be = tval(amu['O2BE'].value)
            self.o2ll.x11 = tval(amu['X11'].value)
            self.o2ll.x16 = tval(amu['X16'].value)
            self.o2ll.x05 = tval(amu['X05'].value)
            self.o2ll.x = tval(amu['X05'].value)
            self.o2ll.snr = tval(amu['Snr'].value)
            self.o2ll.ns = tval(amu['O2_nS'].value)
            if self.model > 'R19':
                self.o2ll.y0 = tval(amu['y0'].value)
                self.o2ll.y1 = tval(amu['y1'].value)
                self.o2ll.dnu0 = tval(amu['dnu0'].value)
                self.o2ll.dnu1 = tval(amu['dnu1'].value)
            else:
                self.o2ll.y300 = tval(amu['Y300'].value)
                self.o2ll.y300[34:49] = tval(amu['Y300_NL'].value[34:49])
                self.o2ll.v = tval(amu['O2_V'].value)
                self.o2ll.v[34:49] = tval(amu['O2_V_NL'].value[34:49])

        # *** add the following lines *************************
        db2np = self._db2np
        rvap = self._rvap
        factor = 0.182 * frq
        temp = 300.0 / vx
        pres = (pdrykpa + ekpa) * 10.0
        vapden = (ekpa * 10.0) / (rvap * temp)
        freq = frq
        freq2 = freq * freq
        # *****************************************************

        th = 300.0 / temp
        th1 = th - 1.0
        b = th**self.o2ll.x
        preswv = vapden * temp / 216.68
        if self.model in ['R03', 'R16', 'R17', 'R18', 'R98']:
            preswv = vapden * temp / 217.0
        if self.model in ['R22', 'R22SD', 'R23', 'R23SD', 'R24']:
            preswv = 4.615228e-3 * vapden * temp
        presda = pres - preswv
        den = .001 * (presda * b + 1.2 * preswv * th)
        if self.model in ['R03', 'R16', 'R98']:
            den = 0.001 * (presda * b + 1.1 * preswv * th)
        if self.model == 'R03':
            den = 0.001 * (presda * th ** 0.9 + 1.1 * preswv * th)
        if self.model == 'R98':
            den = 0.001 * (presda + 1.1 * preswv) * th
        pe2 = den * den
        if self.model in ['R23', 'R23SD']:
            pe1 = .99 * den
            pe2 = pe1**2
        if self.model in ['R24']:
            pe1 = den  # [8] includes common TEMP-dependence
            pe2 = pe1**2
        dfnr = self.o2ll.wb300 * den

        # intensities of the non-resonant transitions for o16-o16 and o16-o18, from jpl's line compilation
        # 1.571e-17 (o16-o16) + 1.3e-19 (o16-o18) = 1.584e-17
        summ = 1.584e-17 * freq2 * dfnr / \
            (th * (freq2 + dfnr * dfnr))
        if self.model in ['R03', 'R16', 'R17', 'R18', 'R98']:
            summ = torch.zeros_like(freq, dtype=freq.dtype)
        nlines = len(self.o2ll.f)
        line_shape = (nlines,) + (1,) * den.dim()

        def _line_param(param) -> torch.Tensor:
            return param.view(line_shape)

        freq_exp = freq.unsqueeze(0)
        den_exp = den.unsqueeze(0)
        th1_exp = th1.unsqueeze(0)
        if self.model in ['R03', 'R98']:
            pres_exp = pres.unsqueeze(0)
            b_exp = b.unsqueeze(0)
            w300_l = _line_param(self.o2ll.w300)
            y300_l = _line_param(self.o2ll.y300)
            v_l = _line_param(self.o2ll.v)
            s300_l = _line_param(self.o2ll.s300)
            be_l = _line_param(self.o2ll.be)
            fcen = _line_param(self.o2ll.f)

            df = w300_l * den_exp
            y = 0.001 * pres_exp * b_exp * (y300_l + v_l * th1_exp)
            strr = s300_l * torch.exp(-be_l * th1_exp)

            sf1 = (df + (freq_exp - fcen) * y) / ((freq_exp - fcen) ** 2 + df * df)
            sf2 = (df - (freq_exp + fcen) * y) / ((freq_exp + fcen) ** 2 + df * df)
            contrib = (strr * (sf1 + sf2) * (freq_exp / fcen) ** 2).sum(dim=0)
            summ = summ + contrib
        elif self.model in ['R17', 'R18', 'R19', 'R19SD']:
            fcen = _line_param(self.o2ll.f)
            df = _line_param(self.o2ll.w300) * den_exp
            y = den_exp * (_line_param(self.o2ll.y300) + _line_param(self.o2ll.v) * th1_exp)
            strr = _line_param(self.o2ll.s300) * torch.exp(-_line_param(self.o2ll.be) * th1_exp)

            sf1 = (df + (freq_exp - fcen) * y) / ((freq_exp - fcen) ** 2 + df * df)
            sf2 = (df - (freq_exp + fcen) * y) / ((freq_exp + fcen) ** 2 + df * df)
            contrib = (strr * (sf1 + sf2) * (freq_exp / fcen) ** 2).sum(dim=0)
            summ = summ + contrib
        elif self.model in ['R23']:
            line_idx = torch.arange(nlines, device=freq.device).view(line_shape)
            pe1_exp = pe1.unsqueeze(0)
            pe2_exp = pe2.unsqueeze(0)

            f_l = _line_param(self.o2ll.f)
            s300_l = _line_param(self.o2ll.s300)
            be_l = _line_param(self.o2ll.be)
            g0_l = _line_param(self.o2ll.g0)
            g1_l = _line_param(self.o2ll.g1)
            w300_l = _line_param(self.o2ll.w300)
            y300_l = _line_param(self.o2ll.y300)
            y1_l = _line_param(self.o2ll.y1)
            dnu0_l = _line_param(self.o2ll.dnu0)
            dnu1_l = _line_param(self.o2ll.dnu1)

            a = s300_l * torch.exp(-be_l * th1_exp) / (f_l ** 2)
            g = g0_l + g1_l * th1_exp
            mask_1_37 = (line_idx > 0) & (line_idx <= 37)
            summ_off = (a * g * mask_1_37).sum(dim=0)
            anorm = (a * mask_1_37).sum(dim=0)
            anorm_safe = torch.where(anorm == 0, torch.ones_like(anorm), anorm)
            off = summ_off / anorm_safe
            summ = (1.584e-17/th) * dfnr / (freq2 + dfnr * dfnr)

            width = w300_l * den_exp
            y = pe1_exp * (y300_l + y1_l * th1_exp)
            gfac = torch.where(mask_1_37, 1.0 + pe2_exp * (g - off.unsqueeze(0)), torch.ones_like(g))
            fcen = f_l + pe2_exp * (dnu0_l + dnu1_l * th1_exp)
            sf1_base = (width*gfac + (freq_exp-fcen)*y) / ((freq_exp-fcen)**2 + width**2)
            sf2 = (width*gfac - (freq_exp+fcen)*y) / ((freq_exp+fcen)**2 + width**2)

            cond0 = torch.abs(freq_exp-fcen) < 10. * width
            width2 = .076 * width
            xc = torch.complex(width-1.5*width2, freq_exp-fcen)/width2
            xrt = torch.sqrt(xc)
            pxw = 1.77245385090551603 * xrt * _dcerror(-torch.imag(xrt), torch.real(xrt))
            asd = torch.complex(torch.ones_like(y), y) * 2. * (1.-pxw) / width2
            sf1_alt = torch.where(cond0, torch.real(asd), sf1_base)
            mask0 = line_idx == 0
            sf1 = torch.where(mask0, sf1_alt, sf1_base)

            contrib = a * (sf1 + sf2)
            if nlines > 37:
                contrib_first = contrib[:38].sum(dim=0)
                contrib_rest = contrib[38:].sum(dim=0) if nlines > 38 else torch.zeros_like(contrib_first)
                summ = summ + contrib_first
                summ = torch.maximum(summ, torch.zeros_like(summ))
                summ = summ + contrib_rest
            else:
                summ = summ + contrib.sum(dim=0)
        elif self.model in ['R24']:
            line_idx = torch.arange(nlines, device=freq.device).view(line_shape)
            th_exp = th.unsqueeze(0)
            pe1_exp = pe1.unsqueeze(0)
            pe2_exp = pe2.unsqueeze(0)

            f_l = _line_param(self.o2ll.f)
            s300_l = _line_param(self.o2ll.s300)
            be_l = _line_param(self.o2ll.be)
            y300_l = _line_param(self.o2ll.y300)
            y1_l = _line_param(self.o2ll.y1)
            g0_l = _line_param(self.o2ll.g0)
            g1_l = _line_param(self.o2ll.g1)
            w300_l = _line_param(self.o2ll.w300)
            dnu0_l = _line_param(self.o2ll.dnu0)
            dnu1_l = _line_param(self.o2ll.dnu1)

            wnr = self.o2ll.wb300 * pe1
            sumy = frq.new_tensor(1.584e-17 * self.o2ll.wb300)
            adjy = .99  # adjustment following update of line intensities

            a = s300_l * torch.exp(-be_l * th1_exp) * th_exp / (f_l ** 2)
            y = adjy * (y300_l + y1_l * th1_exp)
            g = g0_l + g1_l * th1_exp

            mask_le37 = line_idx <= 37
            mask_1_37 = (line_idx > 0) & (line_idx <= 37)
            sumy = sumy + (2. * a * (w300_l + y * f_l) * mask_le37).sum(dim=0)
            sumg = (a * g * mask_1_37).sum(dim=0)
            asq = ((a ** 2) * mask_1_37).sum(dim=0)
            anorm = (a * mask_1_37).sum(dim=0)
            anorm_safe = torch.where(anorm == 0, torch.ones_like(anorm), anorm)
            asq_safe = torch.where(asq == 0, torch.ones_like(asq), asq)
            sumy2 = sumy / (2. * anorm_safe)
            ratio = sumg / asq_safe
            y = torch.where(mask_1_37, y - sumy2.unsqueeze(0) / f_l, y)
            g = torch.where(mask_1_37, g - a * ratio.unsqueeze(0), g)

            summ = 1.584e-17 * wnr/(freq2 + wnr * wnr)
            width = w300_l * pe1_exp
            yk = pe1_exp * y
            gfac = torch.where(mask_1_37, 1.0 + pe2_exp * g, torch.ones_like(g))
            fcen = f_l + pe2_exp * (dnu0_l + dnu1_l * th1_exp)
            sf1_base = (width*gfac + (freq_exp-fcen)*yk) / ((freq_exp-fcen)**2 + width**2)
            sf2 = (width*gfac - (freq_exp+fcen)*yk) / ((freq_exp+fcen)**2 + width**2)

            cond0 = torch.abs(freq_exp-fcen) < 10. * width
            width2 = .076 * width
            xc = torch.complex(width - 1.5*width2, (freq_exp-fcen))/width2
            xrt = torch.sqrt(xc)
            pxw = 1.77245385090551603 * xrt * _dcerror(-torch.imag(xrt), torch.real(xrt))
            asd = torch.complex(torch.ones_like(yk), yk) * 2. * (1.-pxw)/width2
            sf1_alt = torch.where(cond0, torch.real(asd), sf1_base)
            mask0 = line_idx == 0
            sf1 = torch.where(mask0, sf1_alt, sf1_base)

            contrib = a * (sf1 + sf2)
            summ = summ + contrib.sum(dim=0)
        else:
            pe2_exp = pe2.unsqueeze(0)

            f_l = _line_param(self.o2ll.f)
            w300_l = _line_param(self.o2ll.w300)
            y0_l = _line_param(self.o2ll.y0)
            y1_l = _line_param(self.o2ll.y1)
            dnu0_l = _line_param(self.o2ll.dnu0)
            dnu1_l = _line_param(self.o2ll.dnu1)
            s300_l = _line_param(self.o2ll.s300)
            be_l = _line_param(self.o2ll.be)
            g0_l = _line_param(self.o2ll.g0)
            g1_l = _line_param(self.o2ll.g1)

            df = w300_l * den_exp
            y = den_exp * (y0_l + y1_l * th1_exp)
            dnu = pe2_exp * (dnu0_l + dnu1_l * th1_exp)
            strr = s300_l * torch.exp(-be_l * th1_exp)
            del1 = freq_exp - f_l - dnu
            del2 = freq_exp + f_l + dnu
            gfac = 1. + pe2_exp * (g0_l + g1_l * th1_exp)
            d1 = del1 * del1 + df * df
            d2 = del2 * del2 + df * df
            sf1 = (df * gfac + del1 * y) / d1
            sf2 = (df * gfac - del2 * y) / d2
            summ = summ + (strr * (sf1 + sf2) * (freq_exp / f_l) ** 2).sum(dim=0)

        o2abs = 1.6097e+11 * summ * presda * th ** 3
        if self.model in ['R23']:
            o2abs = 1.6097e+11 * summ * presda * freq2 * th**3
        if self.model in ['R24']:
            o2abs = 1.6097e+11 * summ * presda * (freq * th)**2
        if self.model in ['R03', 'R98']:
            o2abs = 5.034e+11 * summ * presda * th ** 3 / 3.14159
        # o2abs = 1.004 * torch.maximum(o2abs, 0.0)
        if self.model != 'R98':
            o2abs = torch.maximum(o2abs, torch.zeros_like(o2abs))
        if self.model in ['R20', 'R20SD', 'R22', 'R22SD']:
            o2abs = 1.004 * torch.maximum(o2abs, torch.zeros_like(o2abs))

        # *** ********************************************************
        # separate the equ. into line and continuum
        # terms, and change the units from np/km to ppm

        # intensities of the non-resonant transitions for o16-o16 and o16-o18, from jpl's line compilation
        # 1.571e-17 (o16-o16) + 1.3e-19 (o16-o18) = 1.584e-17

        ncpp = 1.584e-17 * freq2 * dfnr / \
            (th * (freq2 + dfnr * dfnr))
        #  .20946e-4/(3.14159*1.38065e-19*300) = 1.6097e11
        # a/(pi*k*t_0) = 0.20946/(3.14159*1.38065e-23*300) = 1.6097e19  - then it needs a factor 1e-8 to account
        # for units conversion (pa->hpa, hz->ghz)
        # pa2hpa=1e-2; hz2ghz=1e-9; m2cm=1e2; m2km=1e-3; pa2hpa^-1 * hz2ghz * m2cm^-2 * m2km^-1 = 1e-8
        # th^3 = th(from ideal gas law 2.13) * th(from the mw approx of stimulated emission 2.16 vs. 2.14) *
        # th(from the partition sum 2.20)
        if self.model in ['R03', 'R98']:
            ncpp = 1.6e-17 * freq2 * dfnr / \
                (th * (freq2 + dfnr * dfnr))
            ncpp *= 5.034e+11 * presda * th ** 3 / 3.14159
        else:
            ncpp *= 1.6097e11 * presda * th ** 3  # n/pi*sum0
        # change the units from np/km to ppm
        npp = (o2abs / db2np) / factor
        ncpp = (ncpp / db2np) / factor
        if self.model in [
            'R19', 'R19SD', 'R20', 'R20SD', 'R22', 'R22SD', 'R23', 'R24']:
            ncpp = torch.zeros_like(ncpp)

        return npp, ncpp


class O3AbsModel(AbsModel):
    """Ozone absorption model family."""

    def __init__(self, model: str, dtype: torch.dtype, device, freqs: torch.Tensor, amu=None) -> None:
        super(O3AbsModel, self).__init__(model, dtype=dtype, device=device, freqs=freqs, amu=amu)

        self._implemented_models = O3LL.available_models()

        if model not in self._implemented_models:
            raise AbsModelError(model, f"Model {model} is not available for O3 absorption. Please choose from {self._implemented_models}")
        
        self.o3ll = O3LL.load(self.model, device=self.device, dtype=self.dtype)

    def o3_absorption(self, t: torch.Tensor, p: torch.Tensor, f: torch.Tensor, o3n: torch.Tensor, amu: Optional[dict] = None) -> torch.Tensor:
        """Compute ozone absorption from selected :math:`O_3` rotational lines.

        Parameters
        ----------
        t : torch.Tensor
            Temperature in K.
        p : torch.Tensor
            Total pressure in mbar.
        f : torch.Tensor
            Frequency in GHz.
        o3n : torch.Tensor
            Ozone number density in molecules/m^3.
        amu : dict, optional
            Optional line-parameter overrides with ``.value`` fields.

        Returns
        -------
        torch.Tensor
            Ozone absorption coefficient in Np/km.

        Notes
        -----
        Inputs are broadcast to a common shape. For ``R22``/``R22SD`` the
        routine evaluates a Doppler-Lorentz form through ``_dcerror``; other
        models use the Lorentz/Voigt-width approximation retained from pyrtlib.
        For scalar frequency input, the calculation is limited to lines within
        +/-1 GHz around ``f``; tensor frequency grids evaluate all lines.

        Provenance: the functional description is adapted from the docstring of
        ``pyrtlib.absorption_model.O3AbsModel.o3_absorption``.
        """

        amu = self.amu if amu is None else amu
        t, p, f, o3n = torch.broadcast_tensors(t, p, f, o3n)
        self.o3ll.to(device=t.device, dtype=t.dtype)

        if amu:
            tval = lambda value: t.new_tensor(value)
            self.o3ll.fl = tval(amu['O3_FL'].value)
            self.o3ll.s1 = tval(amu['O3_S1'].value)
            self.o3ll.b = tval(amu['O3_B'].value)
            self.o3ll.w = tval(amu['O3_W'].value)
            self.o3ll.x = tval(amu['O3_X'].value)
            self.o3ll.sr = tval(amu['O3_SR'].value)

        den = 1e-06 * o3n
        ti = self.o3ll.reftline / t
        ti2 = ti ** 2.5
        qvinv = 1.0 - torch.exp(-1008.0 / t)
        # add resonances within 1 ghz of f.  most of the ozone is in the
        # stratosphere, so lines are relatively narrow, and lorentz shape
        # factor is ok.
        summ = torch.zeros_like(f, dtype=f.dtype)
        nlines = len(self.o3ll.fl)
        freq_scalar = float(f) if not isinstance(f, torch.Tensor) or f.dim() == 0 else None
        line_shape = (nlines,) + (1,) * f.dim()

        def _line_param(param) -> torch.Tensor:
            return param.view(line_shape)

        f_exp = f.unsqueeze(0)
        t_exp = t.unsqueeze(0)
        p_exp = p.unsqueeze(0)
        ti_exp = ti.unsqueeze(0)
        fl = _line_param(self.o3ll.fl)
        s1 = _line_param(self.o3ll.s1)
        b = _line_param(self.o3ll.b)
        w = _line_param(self.o3ll.w)
        x = _line_param(self.o3ll.x)
        if freq_scalar is not None:
            mask = (fl >= (freq_scalar - 1.0)) & (fl <= (freq_scalar + 1.0))
        else:
            mask = torch.ones_like(fl, dtype=torch.bool)

        if self.model in ["R22", "R22SD"]:
            widthc = w * p_exp * ti_exp ** x
            betad = .62065e-7 * fl * torch.sqrt(t_exp)
            arg1 = (fl-f_exp)/betad
            arg2 = widthc/betad
            s = s1 * torch.exp(b * (1.0 - ti_exp))
            summ = (s * torch.real(_dcerror(arg1, arg2)) / betad * mask).sum(dim=0)
            abs_o3 = .56419e-4 * summ * qvinv * ti2 * den
        else:
            widthc = w * p_exp * ti_exp ** x
            betad2 = 3.85e-15 * t_exp * fl ** 2
            width = 0.5346 * widthc + torch.sqrt(0.2166 * widthc * widthc + 0.6931 * betad2)
            s = s1 * torch.exp(b * (1.0 - ti_exp))
            shape = (f_exp / fl) ** 2 * width / ((f_exp - fl) ** 2 + width * width)
            summ = (s * shape * mask).sum(dim=0)
            abs_o3 = 3.183e-05 * summ * qvinv * ti2 * den

        return abs_o3
