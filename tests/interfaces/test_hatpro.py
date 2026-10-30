
import numpy as np
import pandas as pd
import pytest
import xarray as xr


from openMWR import hatpro_data
from openMWR.site import create_site


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
    create_site("test", tmp_path,
                mwr_measurement_start_date=str((today - pd.Timedelta(days=1)).date()),
                hatpro_data_dir="raw")
    out_dir = tmp_path / "data/sites/test/hatpro"
    out_dir.mkdir(parents=True)
    return str(tmp_path), out_dir, today


def test_modes_select_and_clean_only_their_daily_files(site, monkeypatch):
    project_dir, out_dir, today = site
    bls_file = out_dir / "bls.nc"
    daily_dataset([today], [290], bls=True).to_netcdf(bls_file)
    original_bls = bls_file.read_bytes()
    leftover = out_dir / f"hatpro_data_bls_{today - pd.Timedelta(days=2):%Y%m%d}.nc"
    daily_dataset([today - pd.Timedelta(days=2)], [280], bls=True).to_netcdf(leftover)

    def create_day(date, bls, suffix, site, project_dir, out_dir, include_rpg, site_config):
        if bls:
            assert bls_file.read_bytes() == original_bls
            assert leftover.exists()
        daily_dataset([date], [250], bls).to_netcdf(
            out_dir / f"hatpro_data{suffix}_{date:%Y%m%d}.nc"
        )

    monkeypatch.setattr(hatpro_data, "_create_hatpro_day_file", create_day)
    hatpro_data.create_hatpro_dataset("test", project_dir)

    with xr.open_dataset(out_dir / "zenith.nc") as zenith:
        assert zenith.TB.dims == ("time", "frq")
        assert "ang" not in zenith.dims
        assert zenith.sizes["time"] == 2
    with xr.open_dataset(bls_file) as bls:
        assert bls.TB.dims == ("time", "frq", "ang")
        assert bls.sizes["time"] == 3
    assert not leftover.exists()
    assert set(out_dir.iterdir()) == {out_dir / "zenith.nc", bls_file}


def test_overlapping_days_and_updates_keep_first_values_with_missing_profiles(site, monkeypatch):
    project_dir, out_dir, today = site
    first_day = today - pd.Timedelta(days=2)
    yesterday = today - pd.Timedelta(days=1)
    updating = False

    def create_day(date, bls, suffix, site, project_dir, out_dir, include_rpg, site_config):
        if updating:
            times = [today, today + pd.Timedelta(hours=12), today + pd.Timedelta(hours=18)]
            values = [42, 43, 44]
        elif date == first_day:
            times = [first_day, first_day + pd.Timedelta(hours=12), yesterday]
            values = [10, 11, 12]
        elif date == yesterday:
            times = [yesterday, yesterday + pd.Timedelta(hours=12), today]
            values = [22, 23, 24]
        else:
            times = [today, today + pd.Timedelta(hours=12)]
            values = [32, 33]
        data = daily_dataset(times, values, bls)
        if date != today:
            data = data.assign_coords(height=[0., 1000.])
            data['T'] = (('time', 'height'), np.full((len(times), 2),
                                                   280. if date == first_day else 281.))
        data.to_netcdf(
            out_dir / f"hatpro_data{suffix}_{date:%Y%m%d}.nc"
        )

    monkeypatch.setattr(hatpro_data, "_create_hatpro_day_file", create_day)
    # Directory entries need not be in date order.
    original_glob = hatpro_data.glob.glob
    monkeypatch.setattr(hatpro_data.glob, 'glob', lambda pattern: original_glob(pattern)[::-1])
    hatpro_data.create_hatpro_dataset("test", project_dir, start_date=str(first_day.date()))
    with xr.open_dataset(out_dir / "zenith.nc") as initial:
        np.testing.assert_array_equal(initial.TB.values[:, 0], [10, 11, 12, 23, 24, 33])
        np.testing.assert_array_equal(initial.T.sel(time=yesterday), [280., 280.])
        assert np.isnan(initial.T.sel(time=today + pd.Timedelta(hours=12))).all()

    updating = True
    hatpro_data.create_hatpro_dataset("test", project_dir, update_only=True)
    for name in ('zenith.nc', 'bls.nc'):
        with xr.open_dataset(out_dir / name) as updated:
            values = updated.TB.isel(ang=0) if 'ang' in updated.TB.dims else updated.TB
            np.testing.assert_array_equal(values.values[:, 0], [10, 11, 12, 23, 24, 33, 44])
            assert updated.indexes["time"].is_unique
            assert updated.indexes["time"].is_monotonic_increasing
            assert np.isnan(updated.T.sel(time=today + pd.Timedelta(hours=18))).all()


def test_combined_failure_propagates_and_preserves_existing_update(site, monkeypatch):
    project_dir, out_dir, today = site
    combined = out_dir / "zenith.nc"
    daily_dataset([today], [210]).to_netcdf(combined)
    original = combined.read_bytes()

    def create_day(date, bls, suffix, site, project_dir, out_dir, include_rpg, site_config):
        assert not bls
        daily_dataset([date], [220]).to_netcdf(out_dir / f"hatpro_data_{date:%Y%m%d}.nc")

    def fail_combination(*args, **kwargs):
        raise ValueError("cannot align duplicate time values")

    monkeypatch.setattr(hatpro_data, "_create_hatpro_day_file", create_day)
    monkeypatch.setattr(hatpro_data.xr, "open_mfdataset", fail_combination)
    with pytest.raises(ValueError, match="duplicate time values"):
        hatpro_data.create_hatpro_dataset("test", project_dir, update_only=True)
    assert combined.read_bytes() == original
    assert set(out_dir.iterdir()) == {combined}
