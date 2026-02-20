import warnings

import numpy as np
import xarray as xr
from pyrtlib.tb_spectrum import TbCloudRTE

from torchMWRT import AtmProfile, RTModel
from openMWR.utils import patch_pyrtlib_numpy_compat

HATPRO_14 = np.array(
    [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40, 51.26, 52.28, 53.86, 54.94, 56.66, 57.30, 58.00],
    dtype=float,
)

patch_pyrtlib_numpy_compat()


def run_torchMWRT(ds: xr.Dataset, angles) -> xr.DataArray:

    rtmodel = RTModel(freqs=HATPRO_14, angles=angles, absmdl="R17")
    atm_profile = AtmProfile(
        temperature=ds["T"],
        height=ds["height"],
        pressure=ds["p"],
        rh=ds["rh"] / 100.0,
        lwc=ds["lwc"],
    )
    tb_ds = rtmodel.execute(atm_profile, return_ds=True)
    return tb_ds["tbtotal"]


def _get_cloud_top_base(lwc: np.ndarray):
    cloud = lwc > 0
    i_base = np.where(np.diff(cloud.astype(int), prepend=0) == 1)[0]
    i_top = np.where(np.diff(cloud.astype(int)) == -1)[0] + 1
    return i_top, i_base


def run_pyrtlib(ds: xr.Dataset, angles) -> xr.DataArray:

    T_K = ds["T"].values
    z_km = ds["height"].values / 1000.0
    p_hPa = ds["p"].values
    rh_100 = ds["rh"].values
    lwc_gpm3 = ds["lwc"].values

    i_top, i_base = _get_cloud_top_base(lwc_gpm3)
    rte = TbCloudRTE(z_km, p_hPa, T_K, rh_100 / 100.0, HATPRO_14, angles)
    rte.satellite = False

    if len(i_top) != 0:
        rte.cloudy = True
        rte.beglev = i_base
        rte.endlev = i_top
        rte.denice = np.zeros(z_km.shape)
        rte.denliq = lwc_gpm3

    rte.init_absmdl("R17")
    df_rt = rte.execute()
    df_tb = df_rt.pivot(columns="angle", values="tbtotal")
    return xr.DataArray(df_tb.values, dims=["frq", "ang"], coords={"frq": HATPRO_14, "ang": angles})


def test_radiosonde():
    ds = xr.load_dataset("tests/test_data/radiosonde.nc")

    ANGLES = np.array([90.0], dtype=float)

    tb_torch = run_torchMWRT(ds, angles=ANGLES)
    tb_pyrtlib = run_pyrtlib(ds, angles=ANGLES)

    diff = tb_torch.values - tb_pyrtlib.values
    abs_diff = np.abs(diff)
    warnings.warn(
        f"torchMWRT - pyrtlib TB differences [K]: max={abs_diff.max():.6g}, "
        f"mean={abs_diff.mean():.6g}, per-channel={diff.ravel().tolist()}",
        UserWarning,
    )

    np.testing.assert_allclose(tb_torch.values, tb_pyrtlib.values, rtol=1e-10, atol=1e-12)


def test_radiosonde_2():
    ds = xr.load_dataset("tests/test_data/radiosonde_2.nc")

    ANGLES = np.array([4.2], dtype=float)

    tb_torch = run_torchMWRT(ds, angles=ANGLES)
    tb_pyrtlib = run_pyrtlib(ds, angles=ANGLES)

    diff = tb_torch.values - tb_pyrtlib.values
    abs_diff = np.abs(diff)
    warnings.warn(
        f"torchMWRT - pyrtlib TB differences [K]: max={abs_diff.max():.6g}, "
        f"mean={abs_diff.mean():.6g}, per-channel={diff.ravel().tolist()}",
        UserWarning,
    )

    np.testing.assert_allclose(tb_torch.values, tb_pyrtlib.values, rtol=0.0, atol=1e-8)
