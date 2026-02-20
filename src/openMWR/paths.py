"""Central path builders for openMWR data files and directories."""

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def data_root(data_dir: PathLike) -> Path:
    """Return the resolved root directory for openMWR data."""
    return Path(data_dir).expanduser()


def site_root(data_dir: PathLike, site: str) -> Path:
    """Return the root directory for a specific site."""
    return data_root(data_dir) / "sites" / site


def site_subdir(data_dir: PathLike, site: str, subdir: str) -> Path:
    """Return a named subdirectory under a site root."""
    return site_root(data_dir, site) / subdir


def site_config_file(data_dir: PathLike, site: str) -> Path:
    """Return the path to a site's config.json file."""
    return site_root(data_dir, site) / "config.json"


def radiosonde_station_dir(data_dir: PathLike, station_id: str) -> Path:
    """Return the directory containing raw files for one station."""
    return data_root(data_dir) / "radiosondes" / f"station_{station_id}"


def radiosonde_year_file(data_dir: PathLike, station_id: str, year: int) -> Path:
    """Return the path to one yearly raw radiosonde NetCDF file."""
    return radiosonde_station_dir(data_dir, station_id) / f"raw_data_{year}.nc"


def libRadtran_dir(data_dir: PathLike) -> Path:
    """Return the directory used for temporary libRadtran files."""
    return data_root(data_dir) / "libRadtran"
