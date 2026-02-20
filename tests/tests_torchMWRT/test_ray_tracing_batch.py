import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchMWRT import RTModel, AtmProfile


def _build_batched_profile(nt: int = 3, nl: int = 12) -> xr.Dataset:
    heights_km = np.linspace(0.0, 3.0, nl, dtype=np.float64)
    heights_m = heights_km * 1000.0
    times = np.arange(nt, dtype=int)

    temperature0 = 290.0 - 6.5 * heights_km  # K
    temperature = np.stack([temperature0 + 0.5 * i for i in range(nt)], axis=0).astype(np.float64)

    pressure0_pa = 101325.0 * np.exp(-heights_m / 8000.0)  # Pa
    pressure0 = pressure0_pa / 100.0  # hPa
    pressure = np.stack([pressure0 for _ in range(nt)], axis=0).astype(np.float64)

    rh0 = np.linspace(0.60, 0.20, nl, dtype=np.float64)  # fraction
    rh = np.stack([rh0 for _ in range(nt)], axis=0).astype(np.float64)

    lwc = np.zeros((nt, nl), dtype=np.float64)
    iwc = np.zeros((nt, nl), dtype=np.float64)

    return xr.Dataset(
        {
            "temperature": (("time", "height"), temperature),
            "pressure": (("time", "height"), pressure),
            "rh": (("time", "height"), rh),
            "LWC": (("time", "height"), lwc),
            "IWC": (("time", "height"), iwc),
        },
        coords={"time": times, "height": heights_km},
    )


def test_ray_tracing_supports_time_batches():
    ds = _build_batched_profile(nt=3, nl=12)
    height = ds["height"].values

    rtmodel = RTModel(
        freqs=np.array([22.24, 31.40], dtype=float),
        angles=np.array([90.0, 50.0], dtype=float),
        absmdl="R17",
        ray_tracing=True,
        from_sat=False,
        cloudy=False,
    )

    profile_batched = AtmProfile(
        temperature=ds["temperature"].values,
        height=height,
        pressure=ds["pressure"].values,
        rh=ds["rh"].values,
        lwc=ds["LWC"].values,
        iwc=ds["IWC"].values,
    )
    tb_batched = rtmodel.execute(profile_batched, return_ds=True)["tbtotal"].to_numpy()
    assert tb_batched.shape == (ds.sizes["time"], 2, 2)

    tb_seq = []
    for ti in range(ds.sizes["time"]):
        ds_i = ds.isel(time=ti)
        height_i = ds_i["height"].values
        profile_i = AtmProfile(
            temperature=ds_i["temperature"].values,
            height=height_i,
            pressure=ds_i["pressure"].values,
            rh=ds_i["rh"].values,
            lwc=ds_i["LWC"].values,
            iwc=ds_i["IWC"].values,
        )
        tb_seq.append(
            rtmodel.execute(profile_i, return_ds=True)["tbtotal"].to_numpy()
        )
    tb_seq = np.stack(tb_seq, axis=0)

    assert np.allclose(tb_batched, tb_seq, rtol=1e-6, atol=1e-6)


def test_ray_tracing_time_batches_are_differentiable():
    ds = _build_batched_profile(nt=2, nl=10)
    temperature = torch.tensor(ds["temperature"].values, dtype=torch.float64, requires_grad=True)
    height = torch.tensor(ds["height"].values, dtype=torch.float64)
    pressure = torch.tensor(ds["pressure"].values, dtype=torch.float64)
    rh = torch.tensor(ds["rh"].values, dtype=torch.float64)
    lwc = torch.tensor(ds["LWC"].values, dtype=torch.float64)
    iwc = torch.tensor(ds["IWC"].values, dtype=torch.float64)

    rtmodel = RTModel(
        freqs=np.array([22.24], dtype=float),
        angles=np.array([50.0], dtype=float),
        absmdl="R17",
        ray_tracing=True,
        from_sat=False,
        cloudy=False,
    )

    profile = AtmProfile(
        temperature=temperature,
        height=height,
        pressure=pressure,
        rh=rh,
        lwc=lwc,
        iwc=iwc,
    )
    tb = rtmodel.execute(profile)["tbtotal"]
    loss = tb.mean()
    loss.backward()

    grad = temperature.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
