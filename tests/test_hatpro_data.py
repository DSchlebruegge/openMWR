import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openMWR import hatpro_data


def daily_dataset(times, values, bls=False):
    dims = ("time", "frq", "ang") if bls else ("time", "frq")
    coords = {"time": times, "frq": [22.24]}
    if bls:
        coords["ang"] = [90.0, 30.0]
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if bls:
        values = np.repeat(values[:, :, None], 2, axis=2)
    return xr.Dataset({"TB": (dims, values)}, coords=coords)


@pytest.fixture
def site(tmp_path):
    today = pd.Timestamp.now().normalize()
    site_dir = tmp_path / "sites" / "test"
    out_dir = site_dir / "hatpro"
    out_dir.mkdir(parents=True)
    (site_dir / "config.json").write_text(json.dumps({
        "mwr_measurement_start_date": str((today - pd.Timedelta(days=1)).date()),
        "rpg_retrieval_exists": False,
    }))
    return str(tmp_path), out_dir, today


def test_modes_select_and_clean_only_their_daily_files(site, monkeypatch):
    data_dir, out_dir, today = site
    bls_file = out_dir / "hatpro_data_bls.nc"
    daily_dataset([today], [290], bls=True).to_netcdf(bls_file)
    original_bls = bls_file.read_bytes()
    leftover = out_dir / f"hatpro_data_bls_{today - pd.Timedelta(days=2):%Y%m%d}.nc"
    daily_dataset([today - pd.Timedelta(days=2)], [280], bls=True).to_netcdf(leftover)

    def create_day(date, bls, suffix, site, data_dir, out_dir, import_retrieval_data):
        if bls:
            assert bls_file.read_bytes() == original_bls
            assert leftover.exists()
        daily_dataset([date], [250], bls).to_netcdf(
            out_dir / f"hatpro_data{suffix}_{date:%Y%m%d}.nc"
        )

    monkeypatch.setattr(hatpro_data, "_create_hatpro_day_file", create_day)
    hatpro_data.create_hatpro_dataset("test", data_dir)

    with xr.open_dataset(out_dir / "hatpro_data.nc") as zenith:
        assert zenith.TB.dims == ("time", "frq")
        assert "ang" not in zenith.dims
        assert zenith.sizes["time"] == 2
    with xr.open_dataset(bls_file) as bls:
        assert bls.TB.dims == ("time", "frq", "ang")
        assert bls.sizes["time"] == 3
    assert not leftover.exists()
    assert set(out_dir.iterdir()) == {out_dir / "hatpro_data.nc", bls_file}


def test_daily_and_update_duplicates_keep_original_first_values(site, monkeypatch):
    data_dir, out_dir, today = site
    yesterday = today - pd.Timedelta(days=1)
    updating = False

    def create_day(date, bls, suffix, site, data_dir, out_dir, import_retrieval_data):
        if updating:
            times = [today, today + pd.Timedelta(hours=12), today + pd.Timedelta(hours=18)]
            values = [32, 33, 34]
        elif date == yesterday:
            times = [yesterday, yesterday + pd.Timedelta(hours=12), today]
            values = [10, 11, 12]
        else:
            times = [today, today + pd.Timedelta(hours=12)]
            values = [22, 23]
        daily_dataset(times, values, bls).to_netcdf(
            out_dir / f"hatpro_data{suffix}_{date:%Y%m%d}.nc"
        )

    monkeypatch.setattr(hatpro_data, "_create_hatpro_day_file", create_day)
    hatpro_data.create_hatpro_dataset("test", data_dir)
    with xr.open_dataset(out_dir / "hatpro_data.nc") as initial:
        assert initial.TB.sel(time=today).item() == 12

    updating = True
    hatpro_data.create_hatpro_dataset("test", data_dir, update_only=True)
    with xr.open_dataset(out_dir / "hatpro_data.nc") as updated:
        np.testing.assert_array_equal(updated.TB.values[:, 0], [10, 11, 12, 23, 34])
        assert updated.indexes["time"].is_unique
        assert updated.indexes["time"].is_monotonic_increasing


def test_combined_failure_propagates_and_preserves_existing_update(site, monkeypatch):
    data_dir, out_dir, today = site
    combined = out_dir / "hatpro_data.nc"
    daily_dataset([today], [210]).to_netcdf(combined)
    original = combined.read_bytes()

    def create_day(date, bls, suffix, site, data_dir, out_dir, import_retrieval_data):
        assert not bls
        daily_dataset([date], [220]).to_netcdf(out_dir / f"hatpro_data_{date:%Y%m%d}.nc")

    def fail_combination(*args, **kwargs):
        raise ValueError("cannot align duplicate time values")

    monkeypatch.setattr(hatpro_data, "_create_hatpro_day_file", create_day)
    monkeypatch.setattr(hatpro_data.xr, "open_mfdataset", fail_combination)
    with pytest.raises(ValueError, match="duplicate time values"):
        hatpro_data.create_hatpro_dataset("test", data_dir, update_only=True)
    assert combined.read_bytes() == original
    assert set(out_dir.iterdir()) == {combined}
