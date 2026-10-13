"""Paths within an openMWR working project."""

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def project_root(project_dir: PathLike) -> Path:
    """Return the absolute working-project root."""
    return Path(project_dir).expanduser().resolve()


def data_root(project_dir: PathLike) -> Path:
    """Return the root for downloaded and prepared datasets."""
    return project_root(project_dir) / "data"


def site_root(project_dir: PathLike, site: str) -> Path:
    """Return the root for site settings and retrieval artifacts."""
    return project_root(project_dir) / "sites" / site


def site_data_dir(project_dir: PathLike, site: str, subdir: str = "") -> Path:
    """Return a site's bulk-data directory or one of its subdirectories."""
    return data_root(project_dir) / "sites" / site / subdir


def site_config_file(project_dir: PathLike, site: str) -> Path:
    """Return the site's authoritative TOML configuration."""
    return site_root(project_dir, site) / "site.toml"


def retrieval_dir(project_dir: PathLike, site: str, name: str) -> Path:
    """Return the artifact directory for one named retrieval."""
    return site_root(project_dir, site) / "retrieval" / name


def checkpoint_file(project_dir: PathLike, site: str, name: str) -> Path:
    """Return the trained checkpoint for a named retrieval."""
    return retrieval_dir(project_dir, site, name) / "model.pth"


def radiosonde_station_dir(project_dir: PathLike, station_id: str) -> Path:
    """Return the directory containing raw files for one station."""
    return data_root(project_dir) / "radiosondes" / f"station_{station_id}"


def radiosonde_year_file(project_dir: PathLike, station_id: str, year: int) -> Path:
    """Return the path to one yearly raw radiosonde NetCDF file."""
    return radiosonde_station_dir(project_dir, station_id) / f"raw_data_{year}.nc"


def libRadtran_dir(project_dir: PathLike) -> Path:
    """Return the directory used for temporary libRadtran files."""
    return data_root(project_dir) / "libRadtran"
