import os
import logging
import tomllib
import tomli_w
from typing import Any, Union
from openMWR.paths import project_root, site_root, site_data_dir, site_config_file

logger = logging.getLogger(__name__)

def create_site(site: str, project_dir: str, **config):
    """
    Create directory structure and config file for a new site.

    Parameters
    ----------
    site : str
        Name of the site (used to create directory).
    project_dir : str
        Working-project root containing data/ and sites/.

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

    Notes
    -----
    The function creates the directory structure::

        sites/{site}/
            site.toml
            retrieval/
        data/sites/{site}/
            analysis/
            radiosonde/
            hatpro/

    Relative input/output paths in the configuration are resolved from
    ``project_dir`` when read. Absolute paths retain their meaning.
    """

    document = tomli_w.dumps({key: value for key, value in config.items() if value is not None})
    (site_root(project_dir, site) / 'retrieval').mkdir(parents=True, exist_ok=True)
    subdirs = ['analysis', 'radiosonde', 'hatpro', 'training', 'era5']
    for dir in subdirs:
        path = site_data_dir(project_dir, site, dir)
        os.makedirs(path, exist_ok=True)

    site_config_file(project_dir, site).write_text(document, encoding='utf-8')

    logger.info(f"Created site config for {site} with: {config}")

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
