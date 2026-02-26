# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Derived from SatCloP/pyrtlib (GPL-3.0): https://github.com/SatCloP/pyrtlib
# Ported to PyTorch by David von Schlebrügge, 2026.
"""
Torch radiative-transfer model utilities.

This module ports pyrtlib radiative-transfer routines to PyTorch and combines
gas/cloud absorption with layer-by-layer radiance integration for batched
profiles, frequencies, and viewing angles.
"""


import warnings
from typing import Tuple, Optional, Union

import numpy as np
import torch
import xarray as xr

from . import consts
from .absorption_model import H2OAbsModel, O2AbsModel, N2AbsModel, LiqAbsModel, O3AbsModel
from .lineshape import ensure_lineshape_data_available

from .profile import AtmProfile


class RTModel:
    """Combined radiative-transfer and absorption model.

    Parameters
    ----------
    freqs : array-like
        Simulation frequencies in GHz.
    angles : array-like
        Viewing elevation angles in degrees. 
        ``90`` denotes zenith-pointing (upward looking).
    absmdl : str, optional
        Absorption-model identifier (for example ``"R98"`` or ``"R17"``).
    ray_tracing : bool, optional
        If ``True``, use refractivity-dependent ray tracing. If ``False``, use
        plane-parallel slant-path geometry.
    from_sat : bool, optional
        If ``True``, integrate top-down (satellite view). If ``False``,
        integrate bottom-up (ground-based view).
    dtype : torch.dtype, optional
        Default dtype used for internal frequency/angle tensors and absorption
        model setup.
    device : str or torch.device, optional
        Device for internal tensors and absorption-model instances.
    amu : optional
        Optional line-shape parameter forwarded to absorption-model
        constructors.

    Raises
    ------
    ValueError
        If ``freqs`` is ``None``.

    Notes
    -----
    - Absorption-model instances are initialized during construction and reused
      for subsequent calls to :meth:`execute`.
    - Cloud absorption is inferred from the presence of ``lwc`` and/or ``iwc``
      in the provided :class:`AtmProfile`.
    - Input ``freqs`` and ``angles`` are converted to torch tensors on the
      configured dtype/device.

    Provenance: the overall radiative-transfer and absorption workflow is
    adapted from pyrtlib (especially ``TbCloudRTE`` and ``RTEquation``) and
    ported to PyTorch for differentiable workflows.
    """

    def __init__(
        self,
        freqs,
        angles,
        absmdl: str = "R98",
        ray_tracing: bool = False,
        from_sat: bool = False,
        dtype: torch.dtype = torch.float64,
        device=None,
        amu=None,
    ) -> None:
        if freqs is None:
            raise ValueError("RTModel requires freqs to initialise absorption models.")
        ensure_lineshape_data_available()
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else None
        self.freqs = torch.as_tensor(freqs, dtype=self.dtype, device=self.device)
        self.angles = torch.as_tensor(angles, dtype=self.dtype, device=self.device)
        self.absmdl = absmdl
        self.ray_tracing = ray_tracing
        self.from_sat = from_sat

        self._from_sat = from_sat

        self.h2o = H2OAbsModel(self.absmdl, freqs=self.freqs, dtype=self.dtype, device=self.device, amu=amu)
        self.o2 = O2AbsModel(self.absmdl, freqs=self.freqs, dtype=self.dtype, device=self.device, amu=amu)
        self.n2 = N2AbsModel(self.absmdl, freqs=self.freqs, dtype=self.dtype, device=self.device, amu=amu)
        self.liq = LiqAbsModel(self.absmdl, freqs=self.freqs, dtype=self.dtype, device=self.device, amu=amu)
        self.o3 = None

    def _tk2b_mod(self, hvk: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Compute the modified Planck function.

        Parameters
        ----------
        hvk : torch.Tensor
            Frequency-dependent factor :math:`h\nu/k_B` in K.
        t : torch.Tensor
            Temperature in K.

        Returns
        -------
        torch.Tensor
            Modified Planck function
            :math:`\tilde{B} = (e^{h\nu/(k_B T)} - 1)^{-1}`.

        Notes
        -----
        This is the torch equivalent of ``pyrtlib.utils.tk2b_mod`` and is kept
        differentiable for autograd workflows.

        Provenance: the physical definition and equation framing are taken from
        the docstring of ``pyrtlib.utils.tk2b_mod``.

        References
        ----------
        :cite:alp:`Schroeder-Westwater-1991`, Eq. (4)
        """
        return torch.reciprocal(torch.exp(hvk / t) - 1.0)

    def execute(
        self,
        profile: "AtmProfile",
        return_intermediate: bool = False,
        return_ds: bool = False,
    ) -> dict[str, torch.Tensor] | xr.Dataset:
        """Run radiative-transfer integration for one atmospheric profile object.

        Parameters
        ----------
        profile : AtmProfile
            Atmospheric profile container with thermodynamic state, cloud
            fields, geometry, and optional batch coordinates.
        return_intermediate : bool, optional
            If ``True``, include intermediate absorption and integrated-path
            diagnostics in the output.
        return_ds : bool, optional
            If ``True``, return an ``xarray.Dataset``; otherwise return a
            dictionary of tensors.

        Returns
        -------
        dict[str, torch.Tensor] or xarray.Dataset
            Brightness-temperature outputs and, optionally, intermediate
            absorption/path diagnostics.

        Raises
        ------
        TypeError
            If ``profile`` is not an ``AtmProfile`` instance.
        RuntimeError
            If ray tracing fails or required intermediate arrays are missing.

        Notes
        -----
        The computation is vectorized over leading batch dimensions, frequency,
        and viewing angle. Absorption is computed once per frequency/state and
        reused across angles, then integrated along each slant path.

        Provenance: output intent and naming conventions are adapted from
        ``pyrtlib.tb_spectrum.TbCloudRTE.execute`` and the underlying
        ``pyrtlib.rt_equation.RTEquation`` routines.
        """

        if not isinstance(profile, AtmProfile):
            raise TypeError(
                f"profile must be an AtmProfile instance, got {type(profile).__name__}.",
            )

        tk = profile.tk
        p = profile.p
        e = profile.e
        z = profile.z
        z0 = profile.z0
        denliq = profile.denliq
        denice = profile.denice
        o3n = profile.o3n
        has_liq = denliq is not None
        has_ice = denice is not None
        has_cloud = has_liq or has_ice

        frq = self.freqs.to(device=tk.device, dtype=tk.dtype)
        angles = self.angles.to(device=tk.device, dtype=tk.dtype)

        nl = profile.nl
        batch_shape = profile.batch_shape
        nf = int(frq.shape[0])
        nang = int(angles.shape[0])
        emissivity = profile._load_emissivity(nf)

        # compute refractivity
        dryn, wetn, refindx = self._refractivity(p, tk, e)

        # Absorption profiles depend only on the atmospheric state and frequency,
        # not on the look angle; compute once per frequency and reuse for all angles.
        awet_base, adry_base = self._clearsky_absorption(p, tk, e, o3n)
        aliq_base = aice_base = None
        if has_cloud:
            aliq_base, aice_base = self._cloudy_absorption(tk, denliq, denice)

        # Compute distance per angle once
        if self.ray_tracing:
            ds_all = self._ray_tracing_path(z, refindx, angles, z0)
            if ds_all is None:
                raise RuntimeError("ray_tracing_path returned None (invalid refractivity profile).")
            ds = ds_all
        else:
            angle_rad_all = angles * torch.pi / 180.0
            amass_all = 1 / torch.sin(angle_rad_all)
            base_ds = torch.cat(
                [torch.zeros(1, device=z.device, dtype=z.dtype), torch.diff(z)]
            )
            ds_all = (base_ds.unsqueeze(0) * amass_all.unsqueeze(1))  # (nang, nl)
            # Broadcast ds across any leading batch dims (e.g. time).
            ds = ds_all.reshape((1,) * len(batch_shape) + ds_all.shape)  # (..., nang, nl)

        # These intermediates are used only in optional outputs.
        swet = sdry = sliq = sice = None
        if return_intermediate:
            swet, _ = self._exponential_integration(True, wetn.unsqueeze(-2), ds, 0, nl, 0.1)
            sdry, _ = self._exponential_integration(True, dryn.unsqueeze(-2), ds, 0, nl, 0.1)
            if has_liq:
                sliq = self._cloud_integrated_density(denliq.unsqueeze(-2), ds)
            if has_ice:
                sice = self._cloud_integrated_density(denice.unsqueeze(-2), ds)

        # Cache per-angle copies for optional output (broadcast across angle).
        # awet/adry: (..., nf, nang, nl)
        awet = adry = None
        if return_intermediate:
            awet = awet_base.unsqueeze(-2).expand(*awet_base.shape[:-1], nang, nl)
            adry = adry_base.unsqueeze(-2).expand(*adry_base.shape[:-1], nang, nl)

        # Integrate absorption along each slant path: vectorized over batch, angles, and frequencies.
        # Arrange singleton axes to get frequency-major outputs directly (..., nf, nang[, nl]).
        ds_abs = ds.unsqueeze(-3)  # (..., 1, nang, nl)
        sptauwet, ptauwet = self._exponential_integration(
            True, awet_base.unsqueeze(-2), ds_abs, 1, nl, 1
        )
        sptaudry, ptaudry = self._exponential_integration(
            True, adry_base.unsqueeze(-2), ds_abs, 1, nl, 1
        )

        sptauliq = sptauice = ptauliq = ptauice = None
        ptaulay = ptauwet + ptaudry
        if aliq_base is not None:
            sptauliq, ptauliq = self._exponential_integration(
                False, aliq_base.unsqueeze(-2), ds_abs, 1, nl, 1
            )
            ptaulay = ptaulay + ptauliq
        if aice_base is not None:
            sptauice, ptauice = self._exponential_integration(
                False, aice_base.unsqueeze(-2), ds_abs, 1, nl, 1
            )
            ptaulay = ptaulay + ptauice

        # planck expects (ang, batch..., nf, nl) so that t/emissivity broadcast cleanly.
        taulay_for_planck = ptaulay.movedim(-2, 0)
        boftotl, boftatm, boftmr, _, hvk, _, _ = self._planck(
            tk,
            taulay_for_planck,
            emissivity=emissivity,
        )

        tbtotal_ang = self._bright(hvk, boftotl)  # (ang, batch..., nf)
        tbatm_ang = self._bright(hvk, boftatm[..., nl - 1])
        tmr_ang = self._bright(hvk, boftmr)

        # move ang dim to the end -> (batch..., nf, ang)
        tbtotal = tbtotal_ang.movedim(0, -1)
        tbatm = tbatm_ang.movedim(0, -1)
        tmr = tmr_ang.movedim(0, -1)

        if return_intermediate:
            if sptauliq is None or sptauice is None:
                sptauliq = torch.zeros_like(sptauwet)
                sptauice = torch.zeros_like(sptauwet)
            if ptauliq is None or ptauice is None:
                ptauliq = torch.zeros_like(ptauwet)
                ptauice = torch.zeros_like(ptauwet)
            if swet is None or sdry is None:
                raise RuntimeError("Internal error: clear-sky integrated refractivity was not computed.")
            if sliq is None or sice is None:
                sliq = torch.zeros_like(swet)
                sice = torch.zeros_like(swet)
            if awet is None or adry is None:
                raise RuntimeError("Internal error: expanded absorption fields were not computed.")

        if not return_ds:
            output: dict[str, torch.Tensor] = {
                "tbtotal": tbtotal,
                "tbatm": tbatm,
                "tmr": tmr,
            }
            if return_intermediate:
                output["tauwet"] = sptauwet
                output["taudry"] = sptaudry
                output["tauliq"] = sptauliq
                output["tauice"] = sptauice

                output["swet"] = swet
                output["sdry"] = sdry
                output["sliq"] = sliq
                output["sice"] = sice

                output["awet"] = awet
                output["adry"] = adry
                output["taulaywet"] = ptauwet
                output["taulaydry"] = ptaudry
                output["taulayliq"] = ptauliq
                output["taulayice"] = ptauice

            return output

        def _np(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().cpu().numpy()

        frq_t = _np(frq)
        ang_t = _np(angles)
        batch_dims = list(profile.batch_coords.keys())
        fa_dims = (*batch_dims, "frq", "ang")
        ang_dims = (*batch_dims, "ang")
        level_dims = (*batch_dims, "frq", "ang", profile._level_dim)

        coords: dict[str, np.ndarray] = {**profile.batch_coords, "frq": frq_t, "ang": ang_t}
        if return_intermediate:
            coords[profile._level_dim] = _np(z)

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
            "tbtotal": (fa_dims, _np(tbtotal)),
            "tbatm": (fa_dims, _np(tbatm)),
            "tmr": (fa_dims, _np(tmr)),
        }
        if return_intermediate:
            data_vars["tauwet"] = (fa_dims, _np(sptauwet))
            data_vars["taudry"] = (fa_dims, _np(sptaudry))
            data_vars["tauliq"] = (fa_dims, _np(sptauliq))
            data_vars["tauice"] = (fa_dims, _np(sptauice))

            data_vars["swet"] = (ang_dims, _np(swet))
            data_vars["sdry"] = (ang_dims, _np(sdry))
            data_vars["sliq"] = (ang_dims, _np(sliq))
            data_vars["sice"] = (ang_dims, _np(sice))

            data_vars["awet"] = (level_dims, _np(awet))
            data_vars["adry"] = (level_dims, _np(adry))
            data_vars["taulaywet"] = (level_dims, _np(ptauwet))
            data_vars["taulaydry"] = (level_dims, _np(ptaudry))
            data_vars["taulayliq"] = (level_dims, _np(ptauliq))
            data_vars["taulayice"] = (level_dims, _np(ptauice))

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def _bright(self, hvk: torch.Tensor, boft: torch.Tensor) -> torch.Tensor:
        r"""Convert modified Planck radiance to brightness temperature.

        Parameters
        ----------
        hvk : torch.Tensor
            Frequency-dependent factor :math:`h\nu/k_B` in K.
        boft : torch.Tensor
            Modified Planck radiance
            :math:`\tilde{B} = (e^{h\nu/(k_B T)} - 1)^{-1}`.

        Returns
        -------
        torch.Tensor
            Brightness temperature in K.

        Notes
        -----
        For zero radiance values, the output is masked to zero to avoid
        singularities in ``log(1 + 1/boft)``.

        Provenance: the inverse-Planck relation is taken from the docstring of
        ``pyrtlib.rt_equation.RTEquation.bright``.

        References
        ----------
        :cite:alp:`Schroeder-Westwater-1991`, Eq. (4)
        """

        Tb = hvk / torch.log(1.0 + (1.0 / boft))
        Tb = torch.where(boft != 0, Tb, torch.zeros_like(Tb))
        return Tb

    def _refractivity(self, p: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute dry/wet refractivity and refractive index profiles.

        Parameters
        ----------
        p : torch.Tensor
            Total pressure profile in mbar.
        t : torch.Tensor
            Temperature profile in K.
        e : torch.Tensor
            Water-vapor partial pressure profile in mbar.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(dryn, wetn, refindx)`` where ``dryn`` and ``wetn`` are
            refractivity components and ``refindx`` is refractive index.

        Notes
        -----
        The formulation follows Thayer and was originally intended for
        frequencies below 20 GHz.

        Provenance: the physical description and intended-frequency note are
        taken from the docstring of ``pyrtlib.rt_equation.RTEquation.refractivity``.

        References
        ----------
        :cite:alp:`Thayer-1974`
        """

        p_t, t_t, e_t = torch.broadcast_tensors(
            torch.as_tensor(p, device=t.device, dtype=t.dtype),
            torch.as_tensor(t, device=t.device, dtype=t.dtype),
            torch.as_tensor(e, device=t.device, dtype=t.dtype),
        )

        pa = p_t - e_t
        tc = t_t - 273.16
        tk2 = t_t * t_t
        tc2 = tc * tc
        rza = 1.0 + pa * (5.79e-07 * (1.0 + 0.52 / t_t) - (0.00094611 * tc) / tk2)
        rzw = 1.0 + 1650.0 * (e_t / (t_t * tk2)) * (
            1.0 - 0.01317 * tc + 0.000175 * tc2 + 1.44e-06 * (tc2 * tc)
        )

        wetn = (64.79 * (e_t / t_t) + 377600.0 * (e_t / tk2)) * rzw
        dryn = 77.6036 * (pa / t_t) * rza
        refindx = 1.0 + (dryn + wetn) * 1e-06

        return dryn, wetn, refindx

    def _ray_tracing_path(
        self,
        z: torch.Tensor,
        refindx: torch.Tensor,
        angle: Union[float, torch.Tensor],
        z0: float,
    ) -> Union[torch.Tensor, None]:
        """Compute slant-path layer distances with refractive ray tracing.

        Parameters
        ----------
        z : torch.Tensor
            Height profile in meters above observation height ``z0``.
        refindx : torch.Tensor
            Refractive-index profile, with last dimension matching ``z``.
        angle : float or torch.Tensor
            Elevation angle(s) in degrees.
        z0 : float
            Observation height in meters above mean sea level.

        Returns
        -------
        torch.Tensor or None
            Slant-path layer lengths in meters. With multiple angles, shape is
            ``(..., nang, nl)``. With a scalar angle, shape is ``(..., nl)``.
            Returns ``None`` if refractive index is invalid.

        Notes
        -----
        This is a vectorized port of the Dutton-Thayer-Westwater algorithm.
        The method assumes exponential decay of refractivity over each layer.

        Provenance: the algorithm description and the exponential-layer
        assumption are taken from the docstring of
        ``pyrtlib.rt_equation.RTEquation.ray_tracing``.

        References
        ----------
        :cite:alp:`Bean-Dutton`, Fig. 3.20 and surrounding text
        """

        z_t = torch.as_tensor(z)
        z_t = z_t / 1000.0
        angle_t = torch.as_tensor(angle, dtype=z_t.dtype, device=z_t.device)
        scalar_out = False
        if angle_t.dim() == 0:
            angle_t = angle_t.unsqueeze(0)
            scalar_out = True

        ref_t = torch.as_tensor(refindx, device=z_t.device, dtype=z_t.dtype)
        nl = int(z_t.numel())
        if ref_t.shape[-1] != nl:
            raise ValueError(
                f"refindx last dimension must match len(z)={nl}; got refindx shape {tuple(ref_t.shape)}."
            )
        batch_shape = tuple(ref_t.shape[:-1])

        deg2rad = torch.tensor(torch.pi / 180.0, dtype=z_t.dtype, device=z_t.device)
        re = torch.tensor(consts.EarthRadius, dtype=z_t.dtype, device=z_t.device) / 1000.0
        z0_t = torch.as_tensor(z0, dtype=z_t.dtype, device=z_t.device) / 1000.0
        nang = int(angle_t.shape[0])
        ds = torch.zeros(batch_shape + (nang, nl), dtype=z_t.dtype, device=z_t.device)

        # Check for refractive index values that will blow up calculations.
        if torch.any(ref_t < 1):
            warnings.warn('ray_tracing: Negative rafractive index')
            return None

        # If angle is close to 90 degrees, make ds a height difference profile.
        mask_90 = ((angle_t >= 89) & (angle_t <= 91)) | ((angle_t >= -91) & (angle_t <= -89))
        if torch.any(mask_90):
            dz = torch.zeros_like(z_t)
            dz[0] = 0.0
            dz[1:] = torch.diff(z_t)
            ds[..., mask_90, :] = dz

        # The rest of the subroutine applies only to angle other than 90 degrees.
        # Convert angle degrees to radians.  Initialize constant values.
        mask_non90 = ~mask_90
        if torch.any(mask_non90):
            theta0 = angle_t[mask_non90] * deg2rad  # (nang_non90,)
            nang_non90 = int(theta0.shape[0])

            rs = re + z_t[0] + z0_t
            rl = re + z_t[0] + z0_t

            costh0 = torch.cos(theta0)
            sina = torch.sin(theta0 * 0.5)
            a0 = 2.0 * (sina ** 2)

            # Broadcast angle-dependent terms over any batch dims.
            ang_shape = (1,) * len(batch_shape) + (nang_non90,)
            theta0_b = theta0.view(ang_shape)
            costh0_b = costh0.view(ang_shape)
            sina_b = sina.view(ang_shape)
            a0_b = a0.view(ang_shape)

            phil = torch.zeros(batch_shape + (nang_non90,), dtype=z_t.dtype, device=z_t.device)
            taul = torch.zeros_like(phil)
            tanthl = torch.tan(theta0_b)
            one = torch.tensor(1.0, dtype=z_t.dtype, device=z_t.device)

            ds_non = torch.zeros(batch_shape + (nang_non90, nl), dtype=z_t.dtype, device=z_t.device)
            for i in range(1, nl):
                r = re + z_t[i] + z0_t

                ri = ref_t[..., i].unsqueeze(-1)
                r_prev = ref_t[..., i - 1].unsqueeze(-1)
                r0 = ref_t[..., 0].unsqueeze(-1)
                close = (
                    torch.isclose(ri, r_prev)
                    | torch.isclose(ri, one)
                    | torch.isclose(r_prev, one)
                )
                denom = torch.where(close, torch.ones_like(ri), ri - one)
                ratio = (r_prev - one) / denom
                safe_ratio = torch.where(close, torch.ones_like(ratio), ratio)
                safe_log = torch.log(safe_ratio)
                safe_log = torch.where(close, torch.ones_like(safe_log), safe_log)
                refbar_alt = one + (r_prev - ri) / safe_log
                refbar = torch.where(close, 0.5 * (ri + r_prev), refbar_alt)
                argdth = z_t[i] / rs - ((r0 - ri) * costh0_b / ri)
                argth = 0.5 * (a0_b + argdth) / r
                if torch.any(argth <= 0):
                    warnings.warn('ray_tracing: Ducting encountered (argth <= 0)')
                    break
                sint = torch.sqrt(r * argth)
                theta = 2.0 * torch.asin(sint)
                cond = theta - 2.0 * theta0_b <= 0.0
                dendth = 2.0 * (sint + sina_b) * torch.cos((theta + theta0_b) * 0.25)
                sind4 = (0.5 * argdth - z_t[i] * argth) / dendth
                dtheta = torch.where(cond, 4.0 * torch.asin(sind4), theta - theta0_b)
                theta = theta0_b + dtheta
                tanth = torch.tan(theta)
                cthbar = 0.5 * ((1.0 / tanth) + (1.0 / tanthl))
                dtau = cthbar * (r_prev - ri) / refbar
                tau = taul + dtau
                phi = dtheta + tau
                ds_i = torch.sqrt(
                    (z_t[i] - z_t[i - 1]) ** 2 + 4.0 * r * rl * (torch.sin((phi - phil) * 0.5) ** 2))
                mask_dtau = dtau != 0.0
                if torch.any(mask_dtau):
                    dtaua = torch.abs(tau - taul)
                    ds_i = torch.where(
                        mask_dtau,
                        ds_i * (dtaua / (2.0 * torch.sin(dtaua * 0.5))),
                        ds_i
                    )
                ds_non[..., i] = ds_i
                phil = phi
                taul = tau
                rl = r
                tanthl = tanth

            ds[..., mask_non90, :] = ds_non

        if scalar_out:
            return ds[..., 0, :] * 1000.0
        return ds * 1000.0

    def _exponential_integration(self, zeroflg: bool, x: torch.Tensor, ds: torch.Tensor, ibeg: int, iend: int, factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate exponentially varying layer quantities.

        Parameters
        ----------
        zeroflg : bool
            Zero-handling flag from pyrtlib logic. If ``True``, zero crossings
            use layer-average values; otherwise they contribute zero.
        x : torch.Tensor
            Quantity profile(s) to integrate.
        ds : torch.Tensor
            Layer thickness profile(s) in meters.
        ibeg : int
            Inclusive lower layer index.
        iend : int
            Exclusive upper layer index.
        factor : float
            Multiplicative scaling applied to integrated sums.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(sxds, xds)`` where ``sxds`` is the integrated profile sum and
            ``xds`` holds per-layer contributions.

        Notes
        -----
        This is a vectorized torch port of pyrtlib's exponential integration.
        The near-equality threshold is scaled from the original Np/km context
        to Np/m to preserve branching behavior in this code path.

        Provenance: algorithm intent and control-flag behavior are taken from
        the docstring of ``pyrtlib.rt_equation.RTEquation.exponential_integration``.
        """
        x_t = torch.as_tensor(x)
        ds_t = torch.as_tensor(ds, device=x_t.device, dtype=x_t.dtype)
        x_t, ds_t = torch.broadcast_tensors(x_t, ds_t)

        # build shifted version with wrap for ibeg=0 along last dimension
        x_prev_full = torch.cat([x_t[..., -1:].clone(), x_t[..., :-1]], dim=-1)
        x_curr = x_t[..., ibeg:iend]
        x_prev = x_prev_full[..., ibeg:iend]
        ds_slice = ds_t[..., ibeg:iend]

        neg_mask = (x_curr < 0.0) | (x_prev < 0.0)
        if torch.any(neg_mask):
            warnings.warn('Error encountered in exponential_integration')

        # pyrtlib's close-value branch uses 1e-9 with absorption in Np/km.
        # Here absorption is in Np/m, so use the scaled threshold to preserve
        # the original branching behavior and avoid integration bias.
        mask_close = torch.abs(x_curr - x_prev) < 1e-12
        mask_zero = (x_curr == 0.0) | (x_prev == 0.0)
        mask_bad = mask_zero | neg_mask

        safe_prev = torch.where(mask_bad, torch.ones_like(x_prev), x_prev)
        safe_curr = torch.where(mask_bad, torch.ones_like(x_curr), x_curr)
        log_ratio = torch.log(safe_curr / safe_prev)
        log_ratio = torch.where(mask_bad, torch.ones_like(log_ratio), log_ratio)
        xlayer = (safe_curr - safe_prev) / log_ratio
        xlayer = torch.where(mask_close, x_curr, xlayer)
        if zeroflg:
            xlayer = torch.where(mask_zero, 0.5 * (x_curr + x_prev), xlayer)
        else:
            xlayer = torch.where(mask_zero, torch.zeros_like(xlayer), xlayer)

        xlayer = torch.where(neg_mask, torch.zeros_like(xlayer), xlayer)

        xds_slice = xlayer * ds_slice

        xds = torch.zeros_like(x_t)
        xds[..., ibeg:iend] = xds_slice
        sxds = torch.sum(xds, dim=-1) * factor

        return sxds, xds

    def _cloud_radiating_temperature(self, ibase: float, itop: float, hvk: torch.Tensor, tauprof: torch.Tensor, boftatm: torch.Tensor) -> Union[torch.Tensor, None]:
        """Compute mean radiating temperature for the lowest cloud layer.

        Parameters
        ----------
        ibase : float
            Profile index of cloud base.
        itop : float
            Profile index of cloud top.
        hvk : torch.Tensor
            Frequency factor :math:`h\nu/k_B`.
        tauprof : torch.Tensor
            Integrated absorption profile.
        boftatm : torch.Tensor
            Integrated atmospheric modified Planck radiance profile.

        Returns
        -------
        torch.Tensor or None
            Mean cloud radiating temperature in K.

        Notes
        -----
        This routine is designed for the lowest (or only) cloud layer.
        torchMWRT keeps the original logic but avoids hard early return by
        clamping exponent arguments when optical depth is large.

        Provenance: cloud-layer assumptions and formulation description are
        taken from the docstring of
        ``pyrtlib.rt_equation.RTEquation.cloud_radiating_temperature``.
        """

        # maximum absolute value for exponential function argument
        expmax = 125.0
        ibase = int(ibase)
        itop = int(itop)
        # check if absorption too large to exponentiate
        tau_base = tauprof[ibase]
        if torch.any(tau_base > expmax):
            warnings.warn(
                'from cloud_radiating_temperature: absorption too large to exponentiate for tmr of lowest cloud layer')

        # compute radiance (batmcld) and absorption (taucld) for cloud layer.
        # (if taucld is too large to exponentiate, treat it as infinity.)
        batmcld = boftatm[itop] - boftatm[ibase]
        taucld = tauprof[itop] - tauprof[ibase]
        tau_base_safe = torch.clamp(tau_base, max=expmax)
        exp_tau_base = torch.exp(tau_base_safe)
        taucld_safe = torch.clamp(taucld, max=expmax)
        exp_neg_taucld = torch.exp(-taucld_safe)
        denom = 1.0 - exp_neg_taucld
        boftcld = (batmcld * exp_tau_base) / denom
        boftcld = torch.where(taucld > expmax, batmcld * exp_tau_base, boftcld)

        # compute cloud mean radiating temperature (tmrcld)
        tmrcld = self._bright(hvk, boftcld)

        return tmrcld

    def _cloud_integrated_density(self, dencld: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        """Integrate cloud water/ice density along the propagation path.

        Parameters
        ----------
        dencld : torch.Tensor
            Cloud condensate density profile in g/m^3.
        ds : torch.Tensor
            Layer path lengths in meters.

        Returns
        -------
        torch.Tensor
            Path-integrated cloud density in cm (water-equivalent convention).

        Notes
        -----
        Unlike pyrtlib's version that integrates between explicit cloud
        base/top indices, this torchMWRT implementation integrates over all
        layers and supports optional leading batch dimensions.

        Provenance: base physical quantity and unit convention are taken from
        the docstring of ``pyrtlib.rt_equation.RTEquation.cloud_integrated_density``.
        """

        den_t = torch.as_tensor(dencld, device=ds.device if isinstance(ds, torch.Tensor) else None,
                                dtype=ds.dtype if isinstance(ds, torch.Tensor) else None)
        ds_t = torch.as_tensor(ds, device=den_t.device, dtype=den_t.dtype)
        den_b, ds_b = torch.broadcast_tensors(den_t, ds_t)
        avg = 0.5 * (den_b + torch.roll(den_b, shifts=1, dims=-1))
        term = ds_b * avg

        scld = torch.sum(term[..., 1:], dim=-1) * 0.1  # skip first to mirror original start

        return scld

    def _planck(self, t: torch.Tensor, taulay: torch.Tensor, emissivity: Optional[torch.Tensor] = None) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute modified-Planck radiance terms for RTE integration.

        Parameters
        ----------
        t : torch.Tensor
            Temperature profile in K, with last dimension ``nl``.
        taulay : torch.Tensor
            Layer optical-depth profile with shape ``(..., nf, nl)``.
        emissivity : torch.Tensor, optional
            Surface emissivity, scalar or broadcastable to ``(..., nf)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(boftotl, boftatm, boftmr, tauprof, hvk, boft, bakgrnd)`` where
            each term follows pyrtlib semantics in torch tensor form.

        Raises
        ------
        ValueError
            If input shapes are inconsistent with ``(..., nf, nl)`` handling.

        Notes
        -----
        The routine computes atmospheric and background contributions for both
        upwelling and downwelling configurations and preserves the original
        optical-depth clipping strategy (``expmax = 125``).

        Provenance: algorithm purpose and returned physical terms are taken
        from the docstring of ``pyrtlib.rt_equation.RTEquation.planck``.

        References
        ----------
        :cite:alp:`Schroeder-Westwater-1992`, Eq. (4)
        """

        t_t = torch.as_tensor(t)
        taulay_t = torch.as_tensor(taulay, device=t_t.device, dtype=t_t.dtype)

        if taulay_t.dim() < 2:
            raise ValueError("taulay must have at least 2 dimensions (..., nf, nl).")

        nl = int(taulay_t.shape[-1])
        nf = int(taulay_t.shape[-2])
        leading_shape = taulay_t.shape[:-2]

        if t_t.shape[-1] != nl:
            raise ValueError(
                f"Temperature last dimension must match taulay nl={nl}; got t shape {tuple(t_t.shape)} "
                f"and taulay shape {tuple(taulay_t.shape)}."
            )

        Tc = torch.tensor(consts.Tcosmicbkg, dtype=t_t.dtype, device=t_t.device)
        h = torch.tensor(consts.planck, dtype=t_t.dtype, device=t_t.device)
        k = torch.tensor(consts.boltzmann, dtype=t_t.dtype, device=t_t.device)

        frq_t = self.freqs.to(device=t_t.device, dtype=t_t.dtype)
        if frq_t.dim() == 0:
            frq_t = frq_t.reshape(1)
        if frq_t.numel() != nf:
            raise ValueError(
                f"taulay has nf={nf} but RTModel.freqs has {frq_t.numel()} elements."
            )

        # Broadcast temperature to the taulay leading dimensions (angle, batch, ...).
        while t_t.dim() < len(leading_shape) + 1:
            t_t = t_t.unsqueeze(0)
        t_b = t_t.expand(leading_shape + (nl,))
        t_bf = t_b.unsqueeze(-2)  # (..., 1, nl) -> broadcast across nf

        # Frequency-dependent constants (broadcast across leading dims and nl).
        frq_b = frq_t.reshape((1,) * len(leading_shape) + (nf,))
        hvk = (frq_b * 1e9) * h / k  # (1, ..., 1, nf)
        hvk = hvk.expand(leading_shape + (nf,))  # (..., nf)
        hvk_full = hvk.unsqueeze(-1)  # (..., nf, 1)

        expmax = 125.0

        # Modified Planck function evaluated at each level.
        boft = self._tk2b_mod(hvk_full, t_bf)  # (..., nf, nl)

        # Surface emissivity: allow scalar, per-frequency, or batched emissivity.
        emissivity_t = torch.as_tensor(
            1.0 if emissivity is None else emissivity,
            device=t_t.device,
            dtype=t_t.dtype,
        )
        # If the last dimension is not a frequency axis, treat emissivity as a scalar
        # per leading profile and broadcast across frequency.
        if emissivity_t.dim() != 0 and emissivity_t.shape[-1] not in (1, nf):
            emissivity_t = emissivity_t.unsqueeze(-1)
        while emissivity_t.dim() < len(leading_shape) + 1:
            emissivity_t = emissivity_t.unsqueeze(0)
        emissivity_f = emissivity_t.expand(leading_shape + (nf,))

        tauprof = torch.zeros_like(taulay_t)
        boftatm = torch.zeros_like(taulay_t)
        bakgrnd = torch.zeros(leading_shape + (nf,), dtype=t_t.dtype, device=t_t.device)

        if self._from_sat:
            boftotl = torch.zeros(leading_shape + (nf,), dtype=t_t.dtype, device=t_t.device)
            Ts = t_b[..., 0]  # (...,)

            if nl > 1:
                taulay_tail = taulay_t[..., 1:]
                tauprof_tail = torch.cumsum(taulay_tail.flip(-1), dim=-1).flip(-1)
                tauprof = torch.cat([tauprof_tail, torch.zeros_like(taulay_t[..., :1])], dim=-1)

                exp_tau = torch.exp(-taulay_tail)
                boftlay = (boft[..., 1:] + boft[..., :-1] * exp_tau) / (1.0 + exp_tau)
                tauprof_next = tauprof[..., 1:]
                batmlay = boftlay * torch.exp(-tauprof_next) * (1.0 - exp_tau)
                boftatm_tail = torch.cumsum(batmlay.flip(-1), dim=-1).flip(-1)
                boftatm = torch.cat([boftatm_tail, torch.zeros_like(boftatm_tail[..., :1])], dim=-1)
            else:
                tauprof.zero_()
                boftatm.zero_()

            tau0 = tauprof[..., 0]  # (..., nf)
            boftbg = emissivity_f * self._tk2b_mod(hvk, Ts) + (1 - emissivity_f) * boftotl
            mask = tau0 < expmax
            bakgrnd = torch.where(mask, boftbg * torch.exp(-tau0), torch.zeros_like(boftbg))
            boftotl = torch.where(mask, bakgrnd + boftatm[..., 0], boftatm[..., 0])
            denom = 1.0 - torch.exp(-tau0)
            boftmr = torch.where(mask, boftatm[..., 0] / denom, boftatm[..., 0])
        else:
            if nl > 1:
                exp_tau = torch.exp(-taulay_t[..., 1:])
                boftlay = (boft[..., :-1] + boft[..., 1:] * exp_tau) / (1.0 + exp_tau)
                tauprof = torch.cumsum(taulay_t, dim=-1)
                batmlay = boftlay * torch.exp(-tauprof[..., :-1]) * (1.0 - exp_tau)
                boftatm_tail = torch.cumsum(batmlay, dim=-1)
                boftatm = torch.cat([torch.zeros_like(boftatm_tail[..., :1]), boftatm_tail], dim=-1)
            else:
                tauprof.zero_()
                boftatm.zero_()

            tau_last = tauprof[..., nl - 1]  # (..., nf)
            boftbg = self._tk2b_mod(hvk, Tc)
            mask = tau_last < expmax
            bakgrnd = torch.where(mask, boftbg * torch.exp(-tau_last), torch.zeros_like(tau_last))
            boftatm_last = boftatm[..., nl - 1]
            boftotl = torch.where(mask, bakgrnd + boftatm_last, boftatm_last)
            denom = 1.0 - torch.exp(-tau_last)
            boftmr = torch.where(mask, boftatm_last / denom, boftatm_last)

        return boftotl, boftatm, boftmr, tauprof, hvk, boft, bakgrnd

    def _cloudy_absorption(
        self,
        t: torch.Tensor,
        denl: Optional[torch.Tensor],
        deni: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute cloud liquid and ice absorption profiles.

        Parameters
        ----------
        t : torch.Tensor
            Temperature profile in K.
        denl : torch.Tensor, optional
            Cloud liquid water density in g/m^3.
        deni : torch.Tensor, optional
            Cloud ice density in g/m^3.

        Returns
        -------
        tuple[torch.Tensor or None, torch.Tensor or None]
            ``(aliq, aice)`` cloud liquid and ice absorption in Np/m.

        Raises
        ------
        ValueError
            If absorption models are not initialized.

        Notes
        -----
        The formulation follows pyrtlib cloud absorption components, then
        converts from Np/km to Np/m. Negative values are warned and clamped to
        zero for numerical robustness.

        Provenance: physical model intent is taken from the docstring of
        ``pyrtlib.rt_equation.RTEquation.cloudy_absorption``.

        References
        ----------
        :cite:alp:`Westwater-1972`

        See Also
        --------
        pyrtlib.absorption_model.LiqAbsModel.liquid_water_absorption
        """
        if denl is None and deni is None:
            return None, None

        ref = next((arg for arg in (t, denl, deni) if isinstance(arg, torch.Tensor)), None)
        t_t = torch.as_tensor(t, device=ref.device if ref is not None else None,
                              dtype=ref.dtype if ref is not None else None)
        denl_t = torch.as_tensor(denl, device=t_t.device, dtype=t_t.dtype) if denl is not None else None
        deni_t = torch.as_tensor(deni, device=t_t.device, dtype=t_t.dtype) if deni is not None else None

        if denl_t is not None and deni_t is not None:
            t_t, denl_t, deni_t = torch.broadcast_tensors(t_t, denl_t, deni_t)
        elif denl_t is not None:
            t_t, denl_t = torch.broadcast_tensors(t_t, denl_t)
        elif deni_t is not None:
            t_t, deni_t = torch.broadcast_tensors(t_t, deni_t)

        if self.liq is None:
            raise ValueError("RTModel absorption models not initialised.")

        batch_shape = t_t.shape[:-1]
        frq_t = self.freqs.to(device=t_t.device, dtype=t_t.dtype)
        nf = int(frq_t.numel())
        frq_grid = frq_t.reshape((1,) * len(batch_shape) + (nf, 1))  # (..., nf, 1)

        c = torch.tensor(consts.light * 100, dtype=t_t.dtype, device=t_t.device)
        ghz2hz = torch.as_tensor(1e9, dtype=t_t.dtype, device=t_t.device)
        db2np = torch.log(torch.tensor(10.0, dtype=t_t.dtype, device=t_t.device)) * 0.1

        t_exp = t_t.unsqueeze(-2)  # (..., 1, nl)

        wave = c / (frq_grid * ghz2hz)  # (..., nf, 1) -> broadcast to (..., nf, nl)

        aliq = aice = None
        if denl_t is not None:
            denl_exp = denl_t.unsqueeze(-2)  # (..., 1, nl)
            liq_abs = self.liq.liquid_water_absorption(denl_exp, frq_grid, t_exp)
            has_liq = denl_exp > 0
            aliq = torch.where(has_liq, liq_abs, torch.zeros_like(liq_abs))

        if deni_t is not None:
            deni_exp = deni_t.unsqueeze(-2)  # (..., 1, nl)
            ice_abs = ((8.18645 / wave) * deni_exp) * 0.000959553 * db2np
            has_ice = deni_exp > 0
            aice = torch.where(has_ice, ice_abs, torch.zeros_like(ice_abs))

        # clamp any negative absorption early and signal clearly
        neg_liq = aliq < 0 if aliq is not None else None
        neg_ice = aice < 0 if aice is not None else None
        has_neg_liq = bool(torch.any(neg_liq)) if neg_liq is not None else False
        has_neg_ice = bool(torch.any(neg_ice)) if neg_ice is not None else False
        if has_neg_liq or has_neg_ice:
            liq_min = float(aliq.min().detach().cpu()) if has_neg_liq and aliq is not None else 0.0
            ice_min = float(aice.min().detach().cpu()) if has_neg_ice and aice is not None else 0.0
            warnings.warn(
                f"Negative cloud absorption detected (liq_min={liq_min:.3e}, ice_min={ice_min:.3e}); clamping to zero.",
            )
            if aliq is not None:
                aliq = aliq.clamp_min(0.0)
            if aice is not None:
                aice = aice.clamp_min(0.0)

        return (
            aliq / 1000.0 if aliq is not None else None,
            aice / 1000.0 if aice is not None else None,
        )

    def _clearsky_absorption(self, p: torch.Tensor, t: torch.Tensor, e: torch.Tensor, o3n: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute clear-sky water-vapor and dry-air absorption profiles.

        Parameters
        ----------
        p : torch.Tensor
            Pressure profile in mbar.
        t : torch.Tensor
            Temperature profile in K.
        e : torch.Tensor
            Water-vapor partial pressure profile in mbar.
        o3n : torch.Tensor, optional
            Ozone number density profile in molecules/m^3.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(awet, adry)`` in Np/m.

        Raises
        ------
        ValueError
            If required absorption models are not initialized.

        Notes
        -----
        ``awet`` is computed from water-vapor line and continuum terms.
        ``adry`` combines :math:`O_2`, :math:`N_2`, and optional :math:`O_3`.
        Internal ppm outputs from absorption-model routines are converted to
        Np/m in this method.

        Provenance: base clear-sky formulation statement is taken from the
        docstring of ``pyrtlib.rt_equation.RTEquation.clearsky_absorption``.

        References
        ----------
        :cite:alp:`Liebe-Layton`
        :cite:alp:`Rosenkranz-1988`

        See Also
        --------
        pyrtlib.absorption_model.H2OAbsModel.h2o_absorption
        pyrtlib.absorption_model.O2AbsModel.o2_absorption
        """
        h2o_model = self.h2o
        o2_model = self.o2
        if h2o_model is None or o2_model is None:
            raise ValueError("RTModel absorption models not initialised.")
        n2_model = self.n2
        o3_model = self.o3

        ref = next((arg for arg in (p, t, e) if isinstance(arg, torch.Tensor)), None)
        p_t = torch.as_tensor(p, device=ref.device if ref is not None else None,
                              dtype=ref.dtype if ref is not None else None)
        t_t = torch.as_tensor(t, device=p_t.device, dtype=p_t.dtype)
        e_t = torch.as_tensor(e, device=p_t.device, dtype=p_t.dtype)
        p_t, t_t, e_t = torch.broadcast_tensors(p_t, t_t, e_t)

        batch_shape = p_t.shape[:-1]
        frq_t = self.freqs.to(device=p_t.device, dtype=p_t.dtype)
        nf = int(frq_t.numel())
        frq_grid = frq_t.reshape((1,) * len(batch_shape) + (nf, 1))  # (..., nf, 1)

        factor = 0.182 * frq_grid
        db2np = torch.log(torch.tensor(10.0, dtype=p_t.dtype, device=p_t.device)) * 0.1

        ekpa_lvl = e_t / 10.0
        pdrykpa_lvl = p_t / 10.0 - ekpa_lvl
        pdrykpa = pdrykpa_lvl.unsqueeze(-2)  # (..., 1, nl)
        vx = (300.0 / t_t).unsqueeze(-2)     # (..., 1, nl)
        ekpa = ekpa_lvl.unsqueeze(-2)        # (..., 1, nl)

        npp, ncpp = h2o_model.h2o_absorption(pdrykpa, vx, ekpa, frq_grid)
        awet = factor * (npp + ncpp) * db2np

        npp, ncpp = o2_model.o2_absorption(pdrykpa, vx, ekpa, frq_grid)
        aO2 = factor * (npp + ncpp) * db2np

        if n2_model:
            aN2 = n2_model.n2_absorption(t_t.unsqueeze(-2), (pdrykpa_lvl * 10.0).unsqueeze(-2), frq_grid)
        else:
            aN2 = torch.zeros_like(aO2)

        if isinstance(o3n, torch.Tensor) and o3_model and hasattr(o3_model, "o3_absorption"):
            o3n_t = torch.as_tensor(o3n, device=p_t.device, dtype=p_t.dtype)
            o3n_t = torch.broadcast_to(o3n_t, p_t.shape).unsqueeze(-2)
            aO3 = o3_model.o3_absorption(t_t.unsqueeze(-2), p_t.unsqueeze(-2), frq_grid, o3n_t)
        else:
            aO3 = torch.zeros_like(aO2)

        adry = aO2 + aN2 + aO3

        return awet / 1000.0, adry / 1000.0
