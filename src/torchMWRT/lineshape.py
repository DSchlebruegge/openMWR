# -*- coding: utf-8 -*-

"""Line-shape parameter loaders for microwave absorption models.

This module consolidates the original pyrtlib line-shape loaders for
water vapor, oxygen, and ozone into a single torch-friendly interface.
In pyrtlib, these definitions were split across ``pyrtlib._lineshape.h2oll``,
``pyrtlib._lineshape.o2ll``, and ``pyrtlib._lineshape.o3ll``.
"""

from contextlib import contextmanager
from importlib.resources import as_file, files
from typing import Any, List, Optional

from netCDF4 import Dataset
import torch


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


@contextmanager
def _open_lineshape(nc_path: str):
    """Read a coefficient table supplied by the installed pyrtlib package."""
    resource = files("pyrtlib").joinpath(nc_path)
    if not resource.is_file():
        raise FileNotFoundError(f"Missing line-shape table {nc_path!r} in the installed pyrtlib package.")
    with as_file(resource) as path, Dataset(path, mode='r') as nc:
        yield nc


class _LineShape:
    """Container for line-shape coefficients and metadata."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    @classmethod
    def available_models(cls) -> List[str]:
        """Return model identifiers from this gas's coefficient table."""
        with _open_lineshape(cls.NC_PATH) as nc:
            return list(nc.groups.keys())

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "_LineShape":
        """Move/convert numeric attributes to torch tensors.

        Parameters
        ----------
        device : torch.device, optional
            Target device.
        dtype : torch.dtype, optional
            Target dtype.

        Returns
        -------
        _LineShape
            The same object with numeric attributes converted in place.

        Notes
        -----
        Non-numeric metadata (for example ``model``) are preserved unchanged.
        """
        for name, value in vars(self).items():
            if name == "model":
                continue
            if torch.is_tensor(value):
                if device is not None or dtype is not None:
                    setattr(self, name, value.to(device=device, dtype=dtype))
                continue
            if isinstance(value, (list, tuple, int, float)) or hasattr(value, "__array__"):
                setattr(self, name, torch.as_tensor(value, device=device, dtype=dtype))
        return self


class H2OLL(_LineShape):
    """Water-vapor line-shape coefficient container.

    Notes
    -----
    Frequencies are in GHz. Strengths and broadening/shift coefficients follow
    the conventions documented in the original pyrtlib H2O line-shape module.

    Provenance: reference text and coefficient intent are taken from
    ``pyrtlib._lineshape.h2oll`` and consolidated here.

    References
    ----------
    - M. Koshelev et al., JQSRT, v.205, pp.51-58 (2018)
    - V. Payne et al., IEEE Trans. Geosci. Remote Sens., v.46, pp.3601-3617 (2008)
    - G. Golubiatnikov, J. Mol. Spectrosc., v.230, pp.196-198 (2005)
    - M. Koshelev et al., J. Mol. Spectrosc., v.241, pp.101-108 (2007)
    - J.-M. Colmont et al., J. Mol. Spectrosc., v.193, pp.233-243 (1999)
    - M. Tretyakov et al., JQSRT, v.114, pp.109-121 (2013)
    - G. Golubiatnikov et al., JQSRT, v.109, pp.1828-1833 (2008)
    - V. Podobedov et al., JQSRT, v.87, pp.377-385 (2004)
    - M. Koshelev, JQSRT, v.112, pp.550-552 (2011)
    - M. Tretyakov, JQSRT, v.328, pp.7-26 (2016)
    - D. Turner et al., IEEE Trans. Geosci. Remote Sens., v.47, pp.3326-3337 (2009)
    - M. Koshelev et al., JQSRT, doi:10.1016/j.jqsrt.2020.107472

    Other parameters from HITRAN2020.
    """

    NC_PATH = "_lineshape/h2o_lineshape.nc"

    @classmethod
    def load(
        cls,
        model: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "H2OLL":
        """Load H2O line-shape parameters for a model.

        Parameters
        ----------
        model : str
            H2O absorption-model identifier.
        device : torch.device, optional
            Target device for returned tensors.
        dtype : torch.dtype, optional
            Target dtype for returned tensors.

        Returns
        -------
        H2OLL
            Loaded and converted line-shape parameter object.

        Raises
        ------
        AbsModelError
            If ``model`` is not available in the NetCDF file.

        Notes
        -----
        Model-dependent field mapping mirrors the original pyrtlib logic,
        including special handling for R03/R16/R17 and SD-family variants.

        Provenance: variable mapping rules were consolidated from
        ``pyrtlib._lineshape.h2oll``.
        """
        with _open_lineshape(cls.NC_PATH) as nc:
            if model not in nc.groups:
                raise AbsModelError(
                    model,
                    f"Model {model} is not available for H2O absorption. "
                    f"Please choose from {list(nc.groups)}",
                )
            d = nc.groups[model]
            mtx = d.variables['mtx'][:].data
            ctr = d.variables['ctr'][:].data
            reftline = d.variables['reftline'][:].data.item()
            extra = {}
            if model == 'R20SD':
                extra['d2air'] = d.variables['d2air'][:].data.item()
                extra['d2self'] = d.variables['d2self'][:].data.item()

        ls = cls(
            model=model,
            mtx=mtx,
            ctr=ctr,
            reftline=reftline,
            reftcon=ctr[0],
            cf=ctr[1],
            xcf=ctr[2],
            cs=ctr[3],
            xcs=ctr[4],
            fl=mtx[:, 1],
            s1=mtx[:, 2],
            b2=mtx[:, 3],
            w0=mtx[:, 4] / 1000.0,
            x=mtx[:, 5],
            w0s=mtx[:, 6] / 1000.0,
            xs=mtx[:, 7],
        )

        if model == 'R03':
            ls.w0s = mtx[:, 7] / 1000.0
            ls.xs = mtx[:, 8]
            ls.sr = mtx[:, 6]
        elif model in ['R16', 'R17']:
            ls.sr = mtx[:, 6]
            ls.w0s = mtx[:, 7] / 1000.0
            ls.xs = mtx[:, 8]

        if model not in ['R98', 'R03', 'R16', 'R17']:
            ls.sh = mtx[:, 8] / 1000.0
            ls.xh = mtx[:, 9]
            ls.shs = mtx[:, 10] / 1000.0
            ls.xhs = mtx[:, 11]

        if model in ['R19', 'R19SD', 'R20', 'R20SD', 'R21SD', 'R22SD', 'R23', 'R23SD', 'R24', 'MWL24']:
            ls.aair = mtx[:, 12]
            ls.aself = mtx[:, 13]

        if model in ['R19SD', 'R20SD', 'R21SD', 'R22SD', 'R23', 'R23SD', 'R24', 'MWL24']:
            ls.w2 = mtx[:, 14] / 1000.0

        if model in ['R19SD', 'R20SD', 'R22SD', 'R23', 'R23SD', 'R24', 'MWL24']:
            ls.w2s = mtx[:, 15] / 1000.0

        if model in ['R21SD', 'R22SD', 'R23', 'R23SD', 'R24', 'MWL24']:
            ls.xw2 = mtx[:, 15]
            ls.w2s = mtx[:, 16] / 1000.0
            ls.xw2s = mtx[:, 17]
            ls.d2 = mtx[:, 18] / 1000.0
            ls.d2s = mtx[:, 19] / 1000.0

        if model == 'R20SD':
            # Provide defaults so downstream code expecting d2/d2s still works.
            ls.d2 = mtx[:, 14] * 0.0
            ls.d2s = mtx[:, 15] * 0.0
            ls.d2air = extra['d2air']
            ls.d2self = extra['d2self']

        return ls.to(device=device, dtype=dtype)


class O2LL(_LineShape):
    """Oxygen line-shape coefficient container.

    Notes
    -----
    This class loads line-mixing and broadening coefficients used by the O2
    absorption models.

    Provenance: data-field selection rules are taken from
    ``pyrtlib._lineshape.o2ll`` and consolidated here.
    """

    NC_PATH = "_lineshape/o2_lineshape.nc"

    @classmethod
    def load(
        cls,
        model: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "O2LL":
        """Load O2 line-shape parameters for a model.

        Parameters
        ----------
        model : str
            O2 absorption-model identifier.
        device : torch.device, optional
            Target device for returned tensors.
        dtype : torch.dtype, optional
            Target dtype for returned tensors.

        Returns
        -------
        O2LL
            Loaded and converted line-shape parameter object.

        Raises
        ------
        AbsModelError
            If ``model`` is not available in the NetCDF file.

        Notes
        -----
        Optional parameter arrays (for example ``v``, ``y0``, ``y1``, ``g*``,
        ``dnu*``) are attached only for model families where they exist.

        Provenance: model-conditional field extraction was consolidated from
        ``pyrtlib._lineshape.o2ll``.
        """
        with _open_lineshape(cls.NC_PATH) as nc:
            if model not in nc.groups:
                raise AbsModelError(
                    model,
                    f"Model {model} is not available for O2 absorption. "
                    f"Please choose from {list(nc.groups)}",
                )
            d = nc.groups[model]
            f = d.variables['f'][:].data
            s300 = d.variables['s300'][:].data
            be = d.variables['be'][:].data
            wb300 = d.variables['wb300'][:].data.item()
            x = d.variables['x'][:].data.item()
            w300 = d.variables['w300'][:].data
            v = None
            y300 = None
            y0 = None
            y1 = g0 = g1 = dnu0 = dnu1 = None

            if model in ['R98', 'R03', 'R17', 'R18', 'R19', 'R19SD']:
                v = d.variables['v'][:].data
            if model in ['R98', 'R03', 'R17', 'R18', 'R19', 'R19SD', 'R23', 'R24']:
                y300 = d.variables['y300'][:].data
            else:
                y0 = d.variables['y0'][:].data
            if model in ['R16', 'R20', 'R20SD', 'R21', 'R22', 'R23', 'R24']:
                y1 = d.variables['y1'][:].data
                g0 = d.variables['g0'][:].data
                g1 = d.variables['g1'][:].data
                dnu0 = d.variables['dnu0'][:].data
                dnu1 = d.variables['dnu1'][:].data

        ls = cls(
            model=model,
            f=f,
            s300=s300,
            be=be,
            wb300=wb300,
            x=x,
            w300=w300,
        )

        if v is not None:
            ls.v = v
        if y300 is not None:
            ls.y300 = y300
        if y0 is not None:
            ls.y0 = y0
        if y1 is not None:
            ls.y1 = y1
            ls.g0 = g0
            ls.g1 = g1
            ls.dnu0 = dnu0
            ls.dnu1 = dnu1

        return ls.to(device=device, dtype=dtype)


class O3LL(_LineShape):
    """Ozone line-shape coefficient container.

    Notes
    -----
    Coefficients are loaded from the ozone NetCDF line-shape table and exposed
    with convenient derived views (``fl``, ``s1``, ``b``, ``w``, ``x``, ``sr``).

    Provenance: loading/mapping behavior is taken from
    ``pyrtlib._lineshape.o3ll`` and consolidated here.
    """

    NC_PATH = "_lineshape/o3_lineshape.nc"

    @classmethod
    def load(
        cls,
        model: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "O3LL":
        """Load O3 line-shape parameters for a model.

        Parameters
        ----------
        model : str
            O3 absorption-model identifier.
        device : torch.device, optional
            Target device for returned tensors.
        dtype : torch.dtype, optional
            Target dtype for returned tensors.

        Returns
        -------
        O3LL
            Loaded and converted line-shape parameter object.

        Raises
        ------
        AbsModelError
            If ``model`` is not available in the NetCDF file.

        Notes
        -----
        Provenance: model loading and variable mapping are consolidated from
        ``pyrtlib._lineshape.o3ll``.
        """
        with _open_lineshape(cls.NC_PATH) as nc:
            if model not in nc.groups:
                raise AbsModelError(
                    model,
                    f"Model {model} is not available for O3 absorption. "
                    f"Please choose from {list(nc.groups)}",
                )
            d = nc.groups[model]
            mtx = d.variables['mtx'][:].data
            reftline = d.variables['reftline'][:].data.item()

        return cls(
            model=model,
            mtx=mtx,
            reftline=reftline,
            fl=mtx[:, 1],
            s1=mtx[:, 2],
            b=mtx[:, 3],
            w=mtx[:, 4] / 1000.0,
            x=mtx[:, 5],
            sr=mtx[:, 6],
        ).to(device=device, dtype=dtype)
