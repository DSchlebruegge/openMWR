"""Paths within an openMWR working project."""

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]
PREDICTIONS_DIR = "predictions"


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


def dataset_recipe_file(project_dir: PathLike, site: str, name: str) -> Path:
    """Return the editable recipe for a named dataset."""
    _path_name(site)
    _dataset_name(name)
    return site_root(project_dir, site) / "datasets" / f"{name}.toml"


def dataset_dir(project_dir: PathLike, site: str, name: str) -> Path:
    """Return the generated-data directory owned by a dataset."""
    _path_name(site)
    _dataset_name(name)
    return site_data_dir(project_dir, site) / name


def dataset_file(project_dir: PathLike, site: str, name: str, source: str | None = None,
                 *, split: str | None = None, bls: bool = False) -> Path:
    """Resolve the one storage convention for profiles, simulations and pairs.

    Profiles/simulations use a station/source file. Split datasets contain three
    files per source; observation and analysis datasets contain zenith/BLS files.
    """
    if source is not None:
        _path_name(source)
        if split is not None and split not in ("training", "validation", "test"):
            raise ValueError("split must be training, validation or test.")
        filename = f"{split + '_' if split else ''}{source}.nc"
    else:
        if split is not None:
            raise ValueError("A split file requires a source.")
        filename = "bls.nc" if bls else "zenith.nc"
    return dataset_dir(project_dir, site, name) / filename


def era5_cache_file(project_dir: PathLike, latitude: float, longitude: float,
                    hour_interval: int = 6) -> Path:
    """Shared ERA5 cache identity includes location and temporal sampling."""
    lat, lon = round(latitude * 4) / 4, round(longitude * 4) / 4
    return data_root(project_dir) / "era5" / f"era5_{lat}_{lon}_{hour_interval}h.nc"


def _path_name(value: str) -> None:
    if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError("Site, dataset and source names must be nonempty names without directory separators.")


def _dataset_name(value: str) -> None:
    _path_name(value)
    if value.casefold() == PREDICTIONS_DIR:
        raise ValueError(f"Dataset name {value!r} is reserved for prediction outputs. Choose another dataset name.")


def retrieval_dir(project_dir: PathLike, site: str, name: str) -> Path:
    """Return the artifact directory for one named retrieval."""
    return site_root(project_dir, site) / "retrieval" / name


def checkpoint_file(project_dir: PathLike, site: str, name: str) -> Path:
    """Return the trained checkpoint for a named retrieval."""
    return retrieval_dir(project_dir, site, name) / "model.pth"


def evaluation_recipe_file(project_dir: PathLike, site: str, name: str) -> Path:
    """Return the configuration for one reproducible comparison."""
    _path_name(site)
    _path_name(name)
    return site_root(project_dir, site) / 'evaluations' / f'{name}.toml'


def evaluation_log_file(project_dir: PathLike, site: str, name: str) -> Path:
    """Return the append log beside an evaluation recipe and its HTML reports."""
    return evaluation_recipe_file(project_dir, site, name).with_suffix('.log')


def optimization_file(project_dir: PathLike, site: str, name: str) -> Path:
    """Return the Optuna database owned by one named retrieval."""
    return retrieval_dir(project_dir, site, name) / "optimization.db"


def radiosonde_station_dir(project_dir: PathLike, station_id: str) -> Path:
    """Return the directory containing raw files for one station."""
    return data_root(project_dir) / "radiosondes" / f"station_{station_id}"


def radiosonde_year_file(project_dir: PathLike, station_id: str, year: int) -> Path:
    """Return the path to one yearly raw radiosonde NetCDF file."""
    return radiosonde_station_dir(project_dir, station_id) / f"raw_data_{year}.nc"


def libRadtran_dir(project_dir: PathLike) -> Path:
    """Return the directory used for temporary libRadtran files."""
    return data_root(project_dir) / "libRadtran"
