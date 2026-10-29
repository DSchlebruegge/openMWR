import logging
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import tomllib
import tomli_w
from typing import Any, Union
from openMWR.config import (check_name, check_invalid_measurements, check_keys,
                            check_number, check_strings, check_time_selection, check_hatpro_format)
from openMWR.paths import project_root, site_root, site_data_dir, site_config_file, retrieval_dir, PREDICTIONS_DIR

logger = logging.getLogger(__name__)

SITE_KEYS = {'gen', 'mwr_height', 'latitude', 'longitude', 'hatpro_data_dir',
             'hatpro_format', 'hatpro_time_offset_hours', 'rpg_retrieval_exists',
             'mwr_measurement_start_date', 'plot_dir', 'retrieval_output_dir',
             'prediction', 'invalid_measurements', 'calibration_dates', 'custom'}
_PATH_KEYS = {'hatpro_data_dir', 'plot_dir', 'retrieval_output_dir'}


@dataclass(frozen=True)
class SiteConfig(Mapping):
    """Validated site settings shared by one workflow, including its workers.

    Core settings are optional until an operation requires them. Project-specific
    settings belong in [custom]. Paths are resolved against the project root.
    """

    site: str
    project_dir: Path
    _values: dict

    def __getitem__(self, key):
        if key not in SITE_KEYS:
            raise ValueError(f'Unknown site setting {key!r}. Project-specific settings belong in [custom].')
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def require(self, key):
        value = self.get(key)
        if value is None:
            raise ValueError(f'set {key} in {site_config_file(self.project_dir, self.site)} before this operation.')
        return value


def _validate_site(values):
    check_keys(values, SITE_KEYS, 'site')
    if 'gen' in values and values['gen'] not in ('G4', 'G5'):
        raise ValueError('site.gen must be "G4" or "G5".')
    for key, limits in (('mwr_height', None), ('latitude', (-90, 90)), ('longitude', (-180, 180))):
        if key in values:
            check_number(values[key], f'site.{key}')
            if limits and not limits[0] <= values[key] <= limits[1]:
                raise ValueError(f'site.{key} must be between {limits[0]} and {limits[1]}.')
    for key in _PATH_KEYS:
        if key in values and (not isinstance(values[key], str) or not values[key].strip()):
            raise ValueError(f'site.{key} must be a nonempty path.')
    check_hatpro_format(values.get('hatpro_format', 'auto'), values.get('hatpro_time_offset_hours'))
    if 'rpg_retrieval_exists' in values and type(values['rpg_retrieval_exists']) is not bool:
        raise ValueError('site.rpg_retrieval_exists must be true or false.')
    if 'mwr_measurement_start_date' in values:
        check_time_selection({'start_date': values['mwr_measurement_start_date']}, 'site')
    if 'calibration_dates' in values:
        check_strings(values['calibration_dates'], 'site.calibration_dates')
        for value in values['calibration_dates']:
            check_time_selection({'start_date': value}, 'site.calibration_dates')
    if 'prediction' in values:
        prediction = values['prediction']
        check_keys(prediction, ('retrievals',), 'prediction')
        if 'retrievals' in prediction:
            check_strings(prediction['retrievals'], 'prediction.retrievals')
            if not prediction['retrievals']:
                raise ValueError('prediction.retrievals must not be empty.')
            for name in prediction['retrievals']:
                check_name(name, 'prediction.retrievals')
    check_invalid_measurements(values.get('invalid_measurements', []))
    if 'custom' in values and not isinstance(values['custom'], dict):
        raise ValueError('site.custom must be a TOML table.')


def load_site_config(site, project_dir, *, config=None):
    """Read and validate site.toml, or reuse an explicitly supplied snapshot."""
    root = project_root(project_dir)
    check_name(site, 'site')
    if config is not None:
        if not isinstance(config, SiteConfig) or config.site != site or config.project_dir != root:
            raise ValueError('Site configuration must belong to the requested site and project.')
        return config
    path = site_config_file(root, site)
    try:
        with path.open('rb') as stream:
            values = tomllib.load(stream)
        _validate_site(values)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc
    for key in _PATH_KEYS & values.keys():
        values[key] = str(root / Path(values[key]).expanduser())
    return SiteConfig(site, root, values)


def _example_files(directory, relative=Path()):
    """Read bundled examples from a source checkout or an installed package."""
    for entry in directory.iterdir():
        path = relative / entry.name
        if entry.is_dir():
            yield from _example_files(entry, path)
        else:
            yield path, entry.read_text(encoding='utf-8')


def retrieval_examples():
    """Return the names of the bundled fixed and optimization recipes."""
    directory = files('openMWR').joinpath('templates', 'site', 'retrieval')
    return sorted(entry.name.removesuffix('_example') for entry in directory.iterdir() if entry.is_dir())


def create_retrieval(site, name, project_dir, *, example='NN'):
    """Copy one bundled recipe into a new named retrieval directory.

    Review its scientific settings and station selection before training.
    Existing retrieval directories are never overwritten.
    """
    load_site_config(site, project_dir)
    check_name(name, 'retrieval')
    if example not in retrieval_examples():
        raise ValueError(f'Unknown retrieval example {example!r}. Choose from {", ".join(retrieval_examples())}.')
    text = files('openMWR').joinpath('templates', 'site', 'retrieval', f'{example}_example', 'retrieval.toml').read_text(encoding='utf-8')
    directory = retrieval_dir(project_dir, site, name)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'retrieval.toml').write_text(text, encoding='utf-8')
    return directory


def create_site(site: str, project_dir: str, *, with_examples: bool = False, **config):
    """
    Create directory structure and config file for a new site.

    Parameters
    ----------
    site : str
        Name of the site (used to create directory).
    project_dir : str
        Working-project root containing data/ and sites/.

    with_examples : bool, default=False
        Include commented site settings, a radiosonde dataset chain, HATPRO and
        radiosonde-analysis recipes, one zenith NN retrieval and its evaluation.
        Other retrieval examples are available through ``create_retrieval``.
        Site details and radiosonde station IDs must be filled in before processing data.

    **config : dict
        Configuration parameters to be saved in ``sites/{site}/site.toml``.
        Optional parameters with value ``None`` are omitted.
        The following keys are commonly used:

        hatpro_data_dir : str
            Directory where the MWR (HATPRO) data is stored.  
            Must contain year-based subfolders (e.g. ``Y2024``, ``Y2025``).

        gen : str
            Generation of the HATPRO microwave radiometer (e.g. ``"G4"``, ``"G5"``).

        hatpro_format : str, optional
            ``"auto"`` prefers NetCDF observations per product and uses binary
            when NetCDF is absent. ``"netcdf"`` and ``"binary"`` require that
            observation format. RPG retrieval products always need NetCDF.

        hatpro_time_offset_hours : float, optional
            Fixed local time minus UTC in hours for binary files marked as local
            time. Required for those files. UTC files ignore this offset.

        mwr_height : float or int
            Height of the MWR above sea level (in meters).

        rpg_retrieval_exists : bool
            Indicates whether an operational RPG retrieval product exists for the site.

        mwr_measurement_start_date : str
            Start date of the MWR measurement time series  
            (format: ``"YYYY-MM-DD"``).

        calibration_dates : list of str, optional
            Instrument calibration dates (``"YYYY-MM-DD"``, UTC). Informational
            markers in OMB reports, with no effect on data selection or training.

        latitude : float
            Geographic latitude of the site in decimal degrees.

        longitude : float
            Geographic longitude of the site in decimal degrees.

        plot_dir : str, optional
            Directory where plots should be saved (e.g. when running operational scripts).

        Project-specific parameters can be supplied in a ``custom`` dictionary
        and are saved in the ``[custom]`` table.

    Returns
    -------
    pathlib.Path
        Created site directory. An existing site is never overwritten.

    Notes
    -----
    The function creates the directory structure::

        sites/{site}/
            site.toml
            datasets/
            retrieval/
            evaluations/
        data/sites/{site}/
        grids.toml                  shared across sites; existing file is kept

    Relative input/output paths in the configuration are resolved from
    ``project_dir`` when read. Absolute paths retain their meaning.
    """

    check_name(site, 'site')
    directory = site_root(project_dir, site)
    if directory.exists():
        raise FileExistsError(f'Site already exists: {directory}. Choose a new site name.')
    documents = {}
    if with_examples:
        for path, text in _example_files(files('openMWR').joinpath('templates', 'site')):
            if path.parts[0] == 'retrieval' and path.parts[1] != 'NN_example':
                continue
            if path == Path('evaluations/spectral.toml'):
                continue
            documents[path] = text
        config = {'prediction': {'retrievals': ['NN_example']}, **config}
    config = {key: value for key, value in config.items() if value is not None}
    _validate_site(config)
    document = tomli_w.dumps(config)
    documents[Path('site.toml')] = documents.get(Path('site.toml'), '') + document

    directory.mkdir(parents=True)
    ignore_path = project_root(project_dir) / '.gitignore'
    if not ignore_path.exists():
        ignore_path.write_text(files('openMWR').joinpath('templates', 'project.gitignore').read_text(encoding='utf-8'), encoding='utf-8')
    grids_path = project_root(project_dir) / 'grids.toml'
    if not grids_path.exists():
        grids_path.write_text(files('openMWR').joinpath('templates', 'grids.toml').read_text(encoding='utf-8'), encoding='utf-8')
    (directory / 'retrieval').mkdir()
    (directory / 'datasets').mkdir()
    site_data_dir(project_dir, site).mkdir(parents=True, exist_ok=True)

    for relative, text in documents.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    logger.info(f"Created site config for {site} with: {config}")
    return directory

def get_invalid_measurements(site, project_dir, *, site_config=None):
    """Read and validate the site's optional measurement validity declarations."""
    return load_site_config(site, project_dir, config=site_config).get('invalid_measurements', [])


def prediction_output_dir(project_dir, site, output_dir=None, *, site_config=None):
    """Resolve the shared directory for daily prediction files and their log."""
    if output_dir is None:
        output_dir = load_site_config(site, project_dir, config=site_config).get('retrieval_output_dir')
    if output_dir is None:
        return site_data_dir(project_dir, site, PREDICTIONS_DIR)
    return project_root(project_dir) / Path(output_dir).expanduser()


def get_config_parameter(site: str, parameter: Union[str, list[str]], project_dir: str) -> Any:
    """Retrieve one or multiple configuration parameters for a given site.

    Args:
        site (str): Name of the site (used to locate config file).
        parameter (str | list[str]): Single key or list of keys to fetch.
        project_dir (str): Working-project root containing data/ and sites/.

    Returns:
        Any: Single value if parameter is str, tuple of values if list.
    """
    config = load_site_config(site, project_dir)

    if isinstance(parameter, str):
        return config.get(parameter)
    elif isinstance(parameter, list):
        return tuple(config.get(p) for p in parameter)
    else:
        raise TypeError("parameter must be str or list[str]")
