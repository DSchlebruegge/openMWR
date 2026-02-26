import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchMWRT import RTModel, AtmProfile


def _build_dataset(
    lwc: np.ndarray,
    iwc: np.ndarray,
    temperature: np.ndarray | None = None,
    rh: np.ndarray | None = None,
) -> xr.Dataset:
    levels = np.arange(5)
    height = np.array([0.0, 0.5, 1.0, 2.0, 3.0])  # km
    if temperature is None:
        temperature = np.array([290.0, 286.0, 282.0, 276.0, 270.0])
    pressure = np.array([1010.0, 950.0, 900.0, 800.0, 700.0])  # hPa
    if rh is None:
        rh = np.array([0.55, 0.60, 0.65, 0.70, 0.75])  # fraction

    return xr.Dataset(
        {
            "height": ("level", height),
            "temperature": ("level", temperature),
            "pressure": ("level", pressure),
            "rh": ("level", rh),
            "LWC": ("level", lwc),
            "IWC": ("level", iwc),
        },
        coords={"level": levels},
    )


def _check_grad(tensor: torch.Tensor, name: str) -> None:
    grad = tensor.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


@pytest.mark.parametrize(
    "config",
    [
        {
            "name": "clear_nadir",
            "freqs": np.array([22.24, 31.40], dtype=float),
            "angles": np.array([90.0], dtype=float),
            "cloudy": False,
            "ray_tracing": False,
            "from_sat": False,
            "return_intermediate": False,
            "require_grad": ["temperature", "rh"],
        },
        {
            "name": "cloudy_multi_angle",
            "freqs": np.array([22.24, 31.40], dtype=float),
            "angles": np.array([50.0, 70.0], dtype=float),
            "cloudy": True,
            "ray_tracing": False,
            "from_sat": False,
            "return_intermediate": True,
            "require_grad": ["LWC", "IWC"],
        },
        {
            "name": "raytrace_sat",
            "freqs": np.array([51.26, 52.28], dtype=float),
            "angles": np.array([55.0], dtype=float),
            "cloudy": True,
            "ray_tracing": True,
            "from_sat": True,
            "return_intermediate": False,
            "require_grad": ["temperature"],
        },
    ],
)
def test_tb_gradients(config):
    if config["cloudy"]:
        lwc = np.array([0.0, 0.02, 0.01, 0.0, 0.0])
        iwc = np.array([0.0, 0.0, 0.005, 0.002, 0.0])
    else:
        lwc = np.zeros(5)
        iwc = np.zeros(5)

    ds = _build_dataset(lwc=lwc, iwc=iwc)
    requires = set(config["require_grad"])
    temperature = torch.tensor(ds["temperature"].values, dtype=torch.float64, requires_grad="temperature" in requires)
    height = torch.tensor(ds["height"].values, dtype=torch.float64)
    pressure = torch.tensor(ds["pressure"].values, dtype=torch.float64)
    rh = torch.tensor(ds["rh"].values, dtype=torch.float64, requires_grad="rh" in requires)
    lwc_t = torch.tensor(ds["LWC"].values, dtype=torch.float64, requires_grad="LWC" in requires)
    iwc_t = torch.tensor(ds["IWC"].values, dtype=torch.float64, requires_grad="IWC" in requires)

    rtmodel = RTModel(
        freqs=config["freqs"],
        angles=config["angles"],
        absmdl="R17",
        ray_tracing=config["ray_tracing"],
        from_sat=config["from_sat"],
    )
    atm_profile_kwargs = dict(
        temperature=temperature,
        height=height,
        pressure=pressure,
        rh=rh,
    )
    if config["cloudy"]:
        atm_profile_kwargs["lwc"] = lwc_t
        atm_profile_kwargs["iwc"] = iwc_t
    atm_profile = AtmProfile(**atm_profile_kwargs)

    tb_out = rtmodel.execute(atm_profile, return_intermediate=config["return_intermediate"])
    loss = tb_out["tbtotal"].mean()
    loss.backward()

    tensors = {
        "temperature": temperature,
        "rh": rh,
        "LWC": lwc_t,
        "IWC": iwc_t,
    }
    for name in config["require_grad"]:
        _check_grad(tensors[name], name)


def _finite_difference_check(
    tensor: torch.Tensor,
    idx: int,
    eps: float,
    loss_fn,
    rtol: float,
    atol: float,
) -> None:
    loss = loss_fn(tensor)
    loss.backward()

    grad_ad = tensor.grad[idx].item()
    assert np.isfinite(grad_ad)

    with torch.no_grad():
        plus = tensor.detach().clone()
        minus = tensor.detach().clone()
        plus[idx] += eps
        minus[idx] -= eps
        loss_plus = loss_fn(plus)
        loss_minus = loss_fn(minus)
        grad_fd = ((loss_plus - loss_minus) / (2 * eps)).item()

    assert np.isfinite(grad_fd)
    assert np.isclose(grad_ad, grad_fd, rtol=rtol, atol=atol)


def _compute_loss_for_temperature(temp_tensor: torch.Tensor) -> torch.Tensor:
    lwc = np.zeros(5)
    iwc = np.zeros(5)
    ds = _build_dataset(lwc=lwc, iwc=iwc, temperature=temp_tensor.detach().cpu().numpy())
    height = torch.tensor(ds["height"].values, dtype=torch.float64)
    pressure = torch.tensor(ds["pressure"].values, dtype=torch.float64)
    rh = torch.tensor(ds["rh"].values, dtype=torch.float64)
    rtmodel = RTModel(
        freqs=np.array([22.24, 31.40], dtype=float),
        angles=np.array([90.0], dtype=float),
        absmdl="R17",
        ray_tracing=False,
        from_sat=False,
    )
    atm_profile = AtmProfile(
        temperature=temp_tensor,
        height=height,
        pressure=pressure,
        rh=rh,
    )
    tb_out = rtmodel.execute(atm_profile)
    return tb_out["tbtotal"].mean()


def _compute_loss_for_rh(rh_tensor: torch.Tensor) -> torch.Tensor:
    lwc = np.zeros(5)
    iwc = np.zeros(5)
    ds = _build_dataset(lwc=lwc, iwc=iwc, rh=rh_tensor.detach().cpu().numpy())
    temperature = torch.tensor(ds["temperature"].values, dtype=torch.float64)
    height = torch.tensor(ds["height"].values, dtype=torch.float64)
    pressure = torch.tensor(ds["pressure"].values, dtype=torch.float64)
    rtmodel = RTModel(
        freqs=np.array([22.24, 31.40], dtype=float),
        angles=np.array([90.0], dtype=float),
        absmdl="R17",
        ray_tracing=False,
        from_sat=False,
    )
    atm_profile = AtmProfile(
        temperature=temperature,
        height=height,
        pressure=pressure,
        rh=rh_tensor,
    )
    tb_out = rtmodel.execute(atm_profile)
    return tb_out["tbtotal"].mean()


def _compute_loss_for_lwc(lwc_tensor: torch.Tensor) -> torch.Tensor:
    iwc = np.zeros(5)
    ds = _build_dataset(lwc=lwc_tensor.detach().cpu().numpy(), iwc=iwc)
    temperature = torch.tensor(ds["temperature"].values, dtype=torch.float64)
    height = torch.tensor(ds["height"].values, dtype=torch.float64)
    pressure = torch.tensor(ds["pressure"].values, dtype=torch.float64)
    rh = torch.tensor(ds["rh"].values, dtype=torch.float64)

    rtmodel = RTModel(
        freqs=np.array([22.24, 31.40], dtype=float),
        angles=np.array([90.0], dtype=float),
        absmdl="R17",
        ray_tracing=False,
        from_sat=False,
    )
    atm_profile = AtmProfile(
        temperature=temperature,
        height=height,
        pressure=pressure,
        rh=rh,
        lwc=lwc_tensor,
    )
    tb_out = rtmodel.execute(atm_profile)
    return tb_out["tbtotal"].mean()


def _compute_loss_for_iwc(iwc_tensor: torch.Tensor) -> torch.Tensor:
    lwc = np.zeros(5)
    ds = _build_dataset(lwc=lwc, iwc=iwc_tensor.detach().cpu().numpy())
    temperature = torch.tensor(ds["temperature"].values, dtype=torch.float64)
    height = torch.tensor(ds["height"].values, dtype=torch.float64)
    pressure = torch.tensor(ds["pressure"].values, dtype=torch.float64)
    rh = torch.tensor(ds["rh"].values, dtype=torch.float64)

    rtmodel = RTModel(
        freqs=np.array([22.24, 31.40], dtype=float),
        angles=np.array([90.0], dtype=float),
        absmdl="R17",
        ray_tracing=False,
        from_sat=False,
    )
    atm_profile = AtmProfile(
        temperature=temperature,
        height=height,
        pressure=pressure,
        rh=rh,
        iwc=iwc_tensor,
    )
    tb_out = rtmodel.execute(atm_profile)
    return tb_out["tbtotal"].mean()


def test_tb_gradient_finite_difference_temperature():
    temp = torch.tensor([290.0, 286.0, 282.0, 276.0, 270.0], dtype=torch.float64, requires_grad=True)
    _finite_difference_check(
        temp,
        idx=2,
        eps=1e-3,
        loss_fn=_compute_loss_for_temperature,
        rtol=5e-2,
        atol=1e-3,
    )


def test_tb_gradient_finite_difference_rh():
    rh = torch.tensor([0.55, 0.60, 0.65, 0.70, 0.75], dtype=torch.float64, requires_grad=True)
    _finite_difference_check(
        rh,
        idx=2,
        eps=1e-4,
        loss_fn=_compute_loss_for_rh,
        rtol=5e-2,
        atol=1e-3,
    )


def test_tb_gradient_finite_difference_lwc():
    lwc = torch.tensor([0.0, 0.02, 0.01, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    _finite_difference_check(
        lwc,
        idx=1,
        eps=1e-5,
        loss_fn=_compute_loss_for_lwc,
        rtol=1e-1,
        atol=1e-4,
    )


def test_tb_gradient_finite_difference_iwc():
    iwc = torch.tensor([0.0, 0.0, 0.005, 0.002, 0.0], dtype=torch.float64, requires_grad=True)
    _finite_difference_check(
        iwc,
        idx=2,
        eps=1e-5,
        loss_fn=_compute_loss_for_iwc,
        rtol=1e-1,
        atol=1e-4,
    )
