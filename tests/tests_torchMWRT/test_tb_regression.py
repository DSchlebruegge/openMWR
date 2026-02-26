import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchMWRT import RTModel, AtmProfile
from pyrtlib.absorption_model import H2OAbsModel
from pyrtlib.tb_spectrum import TbCloudRTE

warnings.filterwarnings("ignore", message="Number of levels too low", module="pyrtlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyrtlib")


def _patch_pyrtlib_scalar_h2o_absorption() -> None:
    """Compatibility shim for pyrtlib versions returning vector outputs for scalar frq."""
    original = H2OAbsModel.h2o_absorption
    if getattr(original, "_openmwr_scalar_patch", False):
        return

    def _wrapped(self, pdrykpa, vx, ekpa, frq, amu=None):
        npp, ncpp = original(self, pdrykpa, vx, ekpa, frq, amu)
        if np.ndim(frq) == 0:
            npp = np.asarray(npp).reshape(-1)[0]
            ncpp = np.asarray(ncpp).reshape(-1)[0]
        return npp, ncpp

    _wrapped._openmwr_scalar_patch = True  # type: ignore[attr-defined]
    H2OAbsModel.h2o_absorption = _wrapped


_patch_pyrtlib_scalar_h2o_absorption()

# Frequencies used for HATPRO 14-channel configuration.
HATPRO_14 = np.array([
    22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40,
    51.26, 52.28, 53.86, 54.94, 56.66, 57.30, 58.00
], dtype=float)

FILE_PATH = Path(__file__).resolve().parent
DATASET_PATH = FILE_PATH / ".." / "test_data" / "20240427_model.nc"
BASELINE_PATH = FILE_PATH / ".." / "baselines" / "pyrtlib_hatpro14.json"

# Cover multiple time steps, angles, and physics configurations.
SCENARIOS = [
    {"time_idx": 11, "angles": np.array([90.0]), "abs_model": "R17"},
    {"time_idx": 12, "angles": np.array([90.0, 50.0]), "abs_model": "R17"},
    {"time_idx": 24, "angles": np.array([60.0]), "abs_model": "R17"},
    # Additional coverage
    {"time_idx": 6, "angles": np.array([60.0]), "abs_model": "R98"},  # alternate absorption model
    {"time_idx": 12, "angles": np.array([50.0]), "abs_model": "R17", "ray_tracing": True},
    {"time_idx": 19, "angles": np.array([30.0, 50.0, 70.0]), "abs_model": "R17"},
    {"time_idx": 5, "angles": np.array([90.0]), "abs_model": "R17", "cloudy": False, "force_clear": True},
    {"time_idx": 12, "angles": np.array([20.0]), "abs_model": "R17", "from_sat": True},
]

_BASELINE_CACHE = None

def _ensure_dataset_available() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Required dataset missing: {DATASET_PATH}")


def _dataset_signature() -> dict:
    stat = DATASET_PATH.stat()
    return {
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_mtime_ns": stat.st_mtime_ns,
    }


def _cache_key(time_idx: int, angles: np.ndarray, abs_model: str, ray_tracing: bool, from_sat: bool, cloudy: bool, force_clear: bool) -> str:
    ang_key = ",".join(f"{float(a):.2f}" for a in np.asarray(angles).ravel())
    return (
        f"time{int(time_idx)}|abs{abs_model}|ang[{ang_key}]|"
        f"rt{'1' if ray_tracing else '0'}|sat{'1' if from_sat else '0'}|"
        f"cld{'1' if cloudy else '0'}|clr{'1' if force_clear else '0'}"
    )


def _load_baseline_cache() -> dict:
    sig = _dataset_signature()
    if not BASELINE_PATH.exists():
        return {"cases": {}, **sig}
    try:
        with BASELINE_PATH.open("r") as f:
            cached = json.load(f)
    except json.JSONDecodeError:
        return {"cases": {}, **sig}

    if cached.get("dataset_path") != sig["dataset_path"] or cached.get("dataset_mtime_ns") != sig["dataset_mtime_ns"]:
        return {"cases": {}, **sig}
    return cached


def _write_baseline_cache(cache: dict) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BASELINE_PATH.open("w") as f:
        json.dump(cache, f, indent=2)


def _get_cloud_top_base(lwc: np.ndarray):
    cloud = lwc > 0
    i_base = np.where(np.diff(cloud.astype(int), prepend=0) == 1)[0]
    i_top = np.where(np.diff(cloud.astype(int)) == -1)[0] + 1
    return i_top, i_base


def _cloud_active(lwc: np.ndarray, cloudy: bool, force_clear: bool) -> bool:
    return bool(cloudy and not force_clear and np.count_nonzero(lwc > 0) > 0)


def _calc_pyrtlib_tb(ds: xr.Dataset, angles: np.ndarray, abs_model: str, ray_tracing: bool, from_sat: bool, cloudy: bool, force_clear: bool) -> np.ndarray:
    T_K = ds.temperature.values
    z_km = ds.height.values / 1000
    p_hPa = ds.pressure.values / 100
    rh_100 = ds.rh.values
    lwc = ds.LWC.values
    iwc = ds.IWC.values

    if np.any(rh_100 < 0):
        raise ValueError(f"Relative humidity has negative values: {rh_100}")

    if force_clear:
        lwc = np.zeros_like(lwc)
        iwc = np.zeros_like(iwc)

    i_top, i_base = _get_cloud_top_base(lwc)
    cloud_flag = _cloud_active(lwc, cloudy, force_clear)
    rte = TbCloudRTE(
        z_km, p_hPa, T_K, rh_100 / 100, HATPRO_14, angles,
        ray_tracing=ray_tracing, from_sat=from_sat, cloudy=cloud_flag,
    )
    rte.satellite = from_sat
    # Always set cloud attributes so pyrtlib stays happy even if no cloud is present.
    rte.beglev = i_base
    rte.endlev = i_top
    rte.denice = iwc
    rte.denliq = lwc
    if cloud_flag and len(i_top) != 0:
        rte.cloudy = True

    rte.init_absmdl(abs_model)
    df_rt = rte.execute()

    df_tb = df_rt.pivot(columns="angle", values="tbtotal").reindex(columns=list(angles))
    tb = df_tb.values
    return tb


def _to_rt_units(ds: xr.Dataset) -> xr.Dataset:
    """Convert an xarray.Dataset with SI-ish units to AtmProfile expected units."""
    ds_rt = ds.copy()
    ds_rt["pressure"] = ds_rt["pressure"] / 100.0
    ds_rt["rh"] = ds_rt["rh"] / 100.0
    return ds_rt


def _get_expected_tb(ds: xr.Dataset, time_idx: int, angles: np.ndarray, abs_model: str, ray_tracing: bool, from_sat: bool, cloudy: bool, force_clear: bool) -> np.ndarray:
    global _BASELINE_CACHE
    if _BASELINE_CACHE is None:
        _BASELINE_CACHE = _load_baseline_cache()

    key = _cache_key(time_idx, angles, abs_model, ray_tracing, from_sat, cloudy, force_clear)
    cases = _BASELINE_CACHE.setdefault("cases", {})

    if key not in cases:
        tb = _calc_pyrtlib_tb(ds, angles, abs_model, ray_tracing, from_sat, cloudy, force_clear)
        cases[key] = {
            "time_idx": int(time_idx),
            "angles": [float(a) for a in np.asarray(angles).ravel()],
            "abs_model": abs_model,
            "ray_tracing": bool(ray_tracing),
            "from_sat": bool(from_sat),
            "cloudy": bool(cloudy),
            "force_clear": bool(force_clear),
            "tb": tb.tolist(),
        }
        _write_baseline_cache(_BASELINE_CACHE)
        print(f"[baseline] stored pyrtlib TB for {key} at {BASELINE_PATH}")
    else:
        tb = np.asarray(cases[key]["tb"], dtype=float)

    return tb


def _run_torchrt(ds: xr.Dataset, angles: np.ndarray, abs_model: str, ray_tracing: bool, from_sat: bool, cloudy: bool, force_clear: bool):
    ds_rt = ds.copy()
    if force_clear:
        ds_rt["LWC"] = xr.zeros_like(ds_rt["LWC"])
        ds_rt["IWC"] = xr.zeros_like(ds_rt["IWC"])
    ds_rt = _to_rt_units(ds_rt)

    rtmodel = RTModel(
        freqs=HATPRO_14,
        angles=angles,
        absmdl=abs_model,
        from_sat=from_sat,
        ray_tracing=ray_tracing,
    )
    emissivity_var = ds_rt.data_vars.get("emissivity")
    atm_profile_kwargs = dict(
        temperature=ds_rt["temperature"].values,
        height=ds_rt["height"].values,
        pressure=ds_rt["pressure"].values,
        rh=ds_rt["rh"].values,
        emissivity=emissivity_var.values if emissivity_var is not None else None,
    )
    if cloudy and not force_clear:
        atm_profile_kwargs["lwc"] = ds_rt["LWC"].values
        atm_profile_kwargs["iwc"] = ds_rt["IWC"].values
    atm_profile = AtmProfile(**atm_profile_kwargs)

    start = perf_counter()
    tb_torch = rtmodel.execute(atm_profile, return_ds=True)
    duration = perf_counter() - start
    return tb_torch["tbtotal"].to_numpy(), duration


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_tb_matches_pyrtlib(scenario):
    _ensure_dataset_available()
    ds = xr.load_dataset(DATASET_PATH).isel(time=scenario["time_idx"])

    angles = scenario["angles"]
    abs_model = scenario["abs_model"]
    ray_tracing = scenario.get("ray_tracing", False)
    from_sat = scenario.get("from_sat", False)
    cloudy = scenario.get("cloudy", True)
    force_clear = scenario.get("force_clear", False)

    expected_tb = _get_expected_tb(ds, scenario["time_idx"], angles, abs_model, ray_tracing, from_sat, cloudy, force_clear)
    torch_tb, duration = _run_torchrt(ds, angles, abs_model, ray_tracing, from_sat, cloudy, force_clear)

    assert torch_tb.shape == expected_tb.shape, (
        f"Shape mismatch for time {scenario['time_idx']} and angles {angles}. "
        f"torchRT {torch_tb.shape} vs pyrtlib {expected_tb.shape}"
    )
    assert np.allclose(torch_tb, expected_tb, rtol=5e-5, atol=1e-6), (
        f"TB outputs differ for time {scenario['time_idx']} and angles {angles}.\n"
        f"Expected (pyrtlib): {expected_tb.tolist()}\n"
        f"Got (torchRT): {torch_tb.tolist()}\n"
        f"Execution time: {duration:.3f}s"
    )

    print(f"[timing] torchRT tb_spectrum execute: {duration:.3f}s for time={scenario['time_idx']} angles={angles.tolist()}")


def test_tb_full_dataset_parallel_torchrt():
    """Attempt a truly batched torchRT run over the time dimension.

    This test is meant to validate *parallel* execution: we pass a full xarray dataset that
    still contains the ``time`` dimension directly into :class:`torchRT.AtmProfile`
    and expect all operations to broadcast over time (resulting TB should include
    a leading time dimension).
    """
    _ensure_dataset_available()
    ds_full = xr.load_dataset(DATASET_PATH)

    angles = np.array([90.0, 50.0])
    abs_model = "R17"
    ray_tracing = False
    from_sat = False
    cloudy = True
    force_clear = False

    ds_full_rt = ds_full.copy()
    if force_clear:
        ds_full_rt["LWC"] = xr.zeros_like(ds_full_rt["LWC"])
        ds_full_rt["IWC"] = xr.zeros_like(ds_full_rt["IWC"])

    ntime = ds_full.sizes.get("time", ds_full["time"].size if "time" in ds_full else 0)
    nf = int(HATPRO_14.shape[0])
    nang = int(np.asarray(angles).size)

    # --- Batched torchRT call over time --------------------------------------------
    ds_full_rt = _to_rt_units(ds_full_rt)
    rtmodel = RTModel(
        freqs=HATPRO_14,
        angles=angles,
        absmdl=abs_model,
        from_sat=from_sat,
        ray_tracing=ray_tracing,
    )
    emissivity_var = ds_full_rt.data_vars.get("emissivity")
    atm_profile = AtmProfile(
        temperature=ds_full_rt["temperature"].values,
        height=ds_full_rt["height"].values,
        pressure=ds_full_rt["pressure"].values,
        rh=ds_full_rt["rh"].values,
        lwc=ds_full_rt["LWC"].values,
        iwc=ds_full_rt["IWC"].values,
        emissivity=emissivity_var.values if emissivity_var is not None else None,
    )
    tb_ds = rtmodel.execute(atm_profile, return_ds=True)
    tb_parallel = tb_ds["tbtotal"].to_numpy()

    assert tb_parallel.ndim == 3, (
        "Expected tbtotal to include a time dimension when passing inputs with time "
        f"(got shape {tb_parallel.shape})."
    )
    try:
        axis_time = tb_parallel.shape.index(ntime)
        axis_freq = tb_parallel.shape.index(nf)
        axis_ang = tb_parallel.shape.index(nang)
    except ValueError as exc:
        raise AssertionError(
            "tbtotal output shape does not match expected (time, frq, ang) sizes. "
            f"Expected sizes time={ntime}, frq={nf}, ang={nang}; got {tb_parallel.shape}."
        ) from exc
    tb_parallel = np.transpose(tb_parallel, (axis_time, axis_freq, axis_ang))

    tb_expected_all = []
    for ti in range(ntime):
        ds = ds_full.isel(time=ti)
        expected_tb = _get_expected_tb(ds, ti, angles, abs_model, ray_tracing, from_sat, cloudy, force_clear)
        tb_expected_all.append(expected_tb)

    tb_expected_all = np.stack(tb_expected_all, axis=0)

    assert tb_parallel.shape == tb_expected_all.shape, (
        f"Shape mismatch for batched dataset. torchRT {tb_parallel.shape} vs pyrtlib {tb_expected_all.shape}"
    )
    assert np.allclose(tb_parallel, tb_expected_all, rtol=5e-5, atol=1e-6), (
        "TB outputs differ for batched dataset run.\n"
        f"torchRT sample {tb_parallel[0].tolist()}\n"
        f"pyrtlib sample {tb_expected_all[0].tolist()}"
    )
