from __future__ import annotations

"""Atmospheric profile container and input-normalization utilities."""

import numpy as np
import torch
import xarray as xr

from .atm import vapor_pressure_from_relative_humidity, vapor_pressure_from_specific_humidity

ProfileInput = torch.Tensor | np.ndarray | xr.DataArray


class AtmProfile(object):
    """Atmospheric profile container for torchMWRT radiative transfer.

    Notes
    -----
    Expected units:
    ``height`` in m above mean sea level, ``pressure`` in hPa,
    ``temperature`` in K, and cloud water contents in g/m^3.
    Exactly one humidity input must be provided: relative humidity ``rh``,
    specific humidity ``q``, or vapor pressure ``e``.

    Provenance: profile-orientation handling and antenna-relative height
    normalization are adapted from the profile setup logic in
    ``pyrtlib.tb_spectrum.TbCloudRTE``.
    """

    def __init__(
        self,
        *,
        temperature: ProfileInput,
        height: ProfileInput,
        pressure: ProfileInput,
        rh: ProfileInput | None = None,
        q: ProfileInput | None = None,
        e: ProfileInput | None = None,
        lwc: ProfileInput | None = None,
        iwc: ProfileInput | None = None,
        emissivity: ProfileInput | None = None,
    ):
        """Build a validated atmospheric profile object.

        Parameters
        ----------
        temperature : torch.Tensor or numpy.ndarray or xarray.DataArray
            Atmospheric temperature profile in K.
        height : torch.Tensor or numpy.ndarray or xarray.DataArray
            Geometric height profile in m above mean sea level.
        pressure : torch.Tensor or numpy.ndarray or xarray.DataArray
            Atmospheric pressure profile in hPa.
        rh : torch.Tensor or numpy.ndarray or xarray.DataArray, optional
            Relative humidity as fraction (0-1).
        q : torch.Tensor or numpy.ndarray or xarray.DataArray, optional
            Specific humidity in kg/kg.
        e : torch.Tensor or numpy.ndarray or xarray.DataArray, optional
            Vapor pressure in hPa.
        lwc : torch.Tensor or numpy.ndarray or xarray.DataArray, optional
            Liquid water content in g/m^3.
        iwc : torch.Tensor or numpy.ndarray or xarray.DataArray, optional
            Ice water content in g/m^3.
        emissivity : torch.Tensor or numpy.ndarray or xarray.DataArray, optional
            Surface emissivity, scalar or frequency-dependent array.

        Raises
        ------
        ValueError
            If humidity inputs are not provided exactly once.
        SystemExit
            If height is not strictly monotonic.

        Notes
        -----
        All profile inputs are converted to torch tensors on a common
        dtype/device. If vertical levels are descending, the profile is flipped
        so internal computations use ascending height.
        """
        humidity_inputs = [input_ is not None for input_ in (rh, q, e)]
        if sum(humidity_inputs) != 1:
            raise ValueError("Provide exactly one of rh, q, or e.")

        dtype, device = self._resolve_dtype_device(temperature, height, pressure, rh, q, e, lwc, iwc, emissivity)

        self.temperature = self._as_profile_tensor(temperature, name="temperature", device=device, dtype=dtype)
        self.height = self._as_profile_tensor(height, name="height", device=device, dtype=dtype)
        self.pressure = self._as_profile_tensor(pressure, name="pressure", device=device, dtype=dtype)
        rh_profile = self._as_profile_tensor(rh, name="rh", device=device, dtype=dtype) if rh is not None else None
        q_profile = self._as_profile_tensor(q, name="q", device=device, dtype=dtype) if q is not None else None
        e_profile = self._as_profile_tensor(e, name="e", device=device, dtype=dtype) if e is not None else None
        self.lwc = self._as_profile_tensor(lwc, name="lwc", device=device, dtype=dtype) if lwc is not None else None
        self.iwc = self._as_profile_tensor(iwc, name="iwc", device=device, dtype=dtype) if iwc is not None else None
        self.emissivity_arr = (
            self._as_profile_tensor(emissivity, name="emissivity", device=device, dtype=dtype)
            if emissivity is not None
            else None
        )

        # Keep batch coordinates for output mapping/dataset. If temperature carries
        # labeled leading coordinates (e.g. "time"), preserve them.
        self.batch_coords, self._level_dim = self._infer_batch_coords(temperature)

        self.z = self.height
        self.p = self.pressure

        rh_data = rh_profile
        q_data = q_profile
        e_data = e_profile

        if self.iwc is not None:
            self.denice = self.iwc
        else:
            self.denice = torch.zeros_like(self.z, device=device, dtype=dtype)

        if self.lwc is not None:
            self.denliq = self.lwc
        else:
            self.denliq = torch.zeros_like(self.z, device=device, dtype=dtype)

        self.tk = self.temperature
        self.o3n = None

        self._validate_vertical_shapes(rh_data, q_data, e_data)

        dz = torch.diff(self.z)
        if torch.all(dz > 0):
            pass
        elif torch.all(dz < 0):
            self.z = torch.flip(self.z, dims=[0])
            self.p = torch.flip(self.p, dims=[-1])
            self.tk = torch.flip(self.tk, dims=[-1])
            if rh_data is not None:
                rh_data = torch.flip(rh_data, dims=[-1])
            if q_data is not None:
                q_data = torch.flip(q_data, dims=[-1])
            if e_data is not None:
                e_data = torch.flip(e_data, dims=[-1])
            self.denliq = torch.flip(self.denliq, dims=[-1])
            self.denice = torch.flip(self.denice, dims=[-1])
        else:
            raise SystemExit("ERROR: input profile seems incorrect. "
                             "It must be monotonically increasing or decreasing")

        if rh_data is not None:
            e_pa = vapor_pressure_from_relative_humidity(self.tk, rh_data)
        elif q_data is not None:
            p_pa = self.p * 100.0
            e_pa = vapor_pressure_from_specific_humidity(q_data, p_pa)
        else:  # e_data is not None
            e_pa = e_data * 100.0

        self.e = e_pa / 100.0  # hPa

        self.nl = int(self.z.shape[-1])
        self.batch_shape = tuple(self.tk.shape[:-1]) if self.tk.dim() > 1 else ()

        self.ice = False

        self.z0 = self.z[0]
        self.z = self.z - self.z0

    def _resolve_dtype_device(
        self,
        *values: ProfileInput | None,
    ) -> tuple[torch.dtype, torch.device]:
        """Determine default dtype/device from provided inputs.

        Parameters
        ----------
        *values : ProfileInput or None
            Candidate profile inputs.

        Returns
        -------
        tuple[torch.dtype, torch.device]
            Dtype/device inferred from the first torch tensor input, or
            ``(torch.float64, cpu)`` if none are torch tensors.
        """
        for value in values:
            if isinstance(value, torch.Tensor):
                return value.dtype, value.device
        return torch.float64, torch.device("cpu")

    def _as_profile_tensor(
        self,
        value: ProfileInput,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert a profile input to a torch tensor on target dtype/device.

        Parameters
        ----------
        value : torch.Tensor or numpy.ndarray or xarray.DataArray
            Input profile data.
        name : str
            Variable name used in error messages.
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.

        Returns
        -------
        torch.Tensor
            Converted tensor.

        Raises
        ------
        TypeError
            If input type is not supported.
        """
        if isinstance(value, xr.DataArray):
            value = value.values

        if isinstance(value, np.ndarray):
            if not value.flags.writeable:
                value = value.copy()
            tensor = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            tensor = value
        else:
            raise TypeError(
                f"{name} must be a torch.Tensor, numpy.ndarray, or xarray.DataArray, got {type(value).__name__}."
            )
        return tensor.to(device=device, dtype=dtype)

    def _infer_batch_coords(self, ref_var: ProfileInput) -> tuple[dict[str, np.ndarray], str]:
        """Infer output batch coordinates from the reference input variable.

        Parameters
        ----------
        ref_var : torch.Tensor or numpy.ndarray or xarray.DataArray
            Reference variable used to define leading dimensions.

        Returns
        -------
        tuple[dict[str, numpy.ndarray], str]
            Batch-coordinate mapping and level-dimension name.

        Notes
        -----
        The last dimension is interpreted as vertical level; all leading
        dimensions are propagated as batch dimensions.
        """

        if isinstance(ref_var, xr.DataArray):
            level_dim = str(ref_var.dims[-1]) if ref_var.ndim > 0 else "level"
            if ref_var.ndim <= 1:
                return {}, level_dim

            coords: dict[str, np.ndarray] = {}
            for dim_name, size in zip(ref_var.dims[:-1], ref_var.shape[:-1]):
                if dim_name in ref_var.coords and ref_var.coords[dim_name].size == size:
                    coords[str(dim_name)] = np.asarray(ref_var.coords[dim_name].values)
                else:
                    coords[str(dim_name)] = np.arange(size)
            return coords, level_dim

        ref_shape = tuple(ref_var.shape)
        if len(ref_shape) <= 1:
            return {}, "level"

        coords: dict[str, np.ndarray] = {}
        for dim_index, size in enumerate(ref_shape[:-1]):
            coords[f"batch_{dim_index}"] = np.arange(size)
        return coords, "level"

    def _load_emissivity(self, nf: int) -> torch.Tensor:
        """Return emissivity as a tensor ready for model execution.

        Parameters
        ----------
        nf : int
            Number of frequencies in the radiative-transfer simulation.

        Returns
        -------
        torch.Tensor
            Emissivity tensor. Defaults to ones when emissivity input is not
            provided.
        """
        if self.emissivity_arr is None:
            return torch.ones((nf,), device=self.tk.device, dtype=self.tk.dtype)

        emissivity = self.emissivity_arr.to(device=self.tk.device, dtype=self.tk.dtype)
        return emissivity

    def _validate_vertical_shapes(self, rh: torch.Tensor | None, q: torch.Tensor | None, e: torch.Tensor | None) -> None:
        """Validate that all profile variables share the same vertical length.

        Parameters
        ----------
        rh : torch.Tensor, optional
            Relative-humidity profile.
        q : torch.Tensor, optional
            Specific-humidity profile.
        e : torch.Tensor, optional
            Vapor-pressure profile.

        Raises
        ------
        ValueError
            If any variable's last dimension does not match the height profile.
        """
        nl = self.z.shape[-1]
        expected = {
            "temperature": self.tk,
            "pressure": self.p,
            "liquid water content": self.denliq,
            "ice water content": self.denice,
        }
        if rh is not None:
            expected["relative humidity"] = rh
        if q is not None:
            expected["specific humidity"] = q
        if e is not None:
            expected["vapor pressure"] = e
        for name, tensor in expected.items():
            if tensor.shape[-1] != nl:
                raise ValueError(
                    f"{name} last dimension must match height profile (expected {nl}, got {tensor.shape[-1]})."
                )
