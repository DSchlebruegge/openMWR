import os
import logging
from importlib.resources import files
from pathlib import Path
import tomllib
import tomli_w
from typing import Any, Union
from openMWR.config import check_name
from openMWR.paths import project_root, site_root, site_data_dir, site_config_file

logger = logging.getLogger(__name__)

def _example_files(directory, relative=Path()):
    """Read bundled examples from a source checkout or an installed package."""
    for entry in directory.iterdir():
        path = relative / entry.name
        if entry.is_dir():
            yield from _example_files(entry, path)
        else:
            yield path, entry.read_text(encoding='utf-8')


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
        radiosonde-analysis recipes, and fixed/optimization examples for zenith
        NN/MLR, BLS NN and spectral-consistency retrievals.
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

        mwr_height : float or int
            Height of the MWR above sea level (in meters).

        rpg_retrieval_exists : bool
            Indicates whether an operational RPG retrieval product exists for the site.

        mwr_measurement_start_date : str
            Start date of the MWR measurement time series  
            (format: ``"YYYY-MM-DD"``).

        latitude : float
            Geographic latitude of the site in decimal degrees.

        longitude : float
            Geographic longitude of the site in decimal degrees.

        plot_dir : str, optional
            Directory where plots should be saved (e.g. when running operational scripts).

        Additional user-defined parameters can also be passed and will
        be included unchanged in the generated TOML configuration.

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
        documents.update(_example_files(files('openMWR').joinpath('templates', 'site')))
        config = {'prediction': {'retrievals': ['NN_example']}, **config}
    document = tomli_w.dumps({key: value for key, value in config.items() if value is not None})
    documents[Path('site.toml')] = documents.get(Path('site.toml'), '') + document

    directory.mkdir(parents=True)
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

def prediction_output_dir(project_dir, site, output_dir=None):
    """Resolve the shared directory for daily prediction files and their log."""
    if output_dir is None:
        output_dir = get_config_parameter(site, 'retrieval_output_dir', project_dir)
    if output_dir is None:
        return site_data_dir(project_dir, site, 'predictions')
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
    with site_config_file(project_dir, site).open("rb") as file:
        config = tomllib.load(file)

    def value_for(key):
        value = config.get(key)
        if key in {'hatpro_data_dir', 'plot_dir', 'retrieval_output_dir'} and value is not None:
            return str(project_root(project_dir) / os.path.expanduser(value))
        return value

    if isinstance(parameter, str):
        return value_for(parameter)
    elif isinstance(parameter, list):
        return tuple(value_for(p) for p in parameter)
    else:
        raise TypeError("parameter must be str or list[str]")
