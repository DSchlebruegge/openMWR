import os
import logging
import json
from typing import Any, Union
from openMWR.paths import site_subdir, site_config_file

logger = logging.getLogger(__name__)

def create_site(site: str, data_dir: str, **config):
    """
    Create directory structure and config file for a new site.

    Parameters
    ----------
    site : str
        Name of the site (used to create directory).
    data_dir : str
        Base directory containing all openMWR-managed data.

    **config : dict
        Configuration parameters to be saved in ``config.json``.
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
        be included unchanged in the generated ``config.json``.

    Notes
    -----
    The function creates the directory structure::

        data/sites/{site}/
            retrieval/
            analysis/
            radiosonde/
            hatpro/

    and stores all provided configuration parameters in
    ``data/sites/{site}/config.json``.
    """

    subdirs = ['retrieval', 'analysis', 'radiosonde', 'hatpro', 'training', 'era5']
    for dir in subdirs:
        path = site_subdir(data_dir, site, dir)
        os.makedirs(path, exist_ok=True)

    # Config speichern
    with open(site_config_file(data_dir, site), "w") as f:
        json.dump(config, f, indent=4)

    logger.info(f"Created site config for {site} with: {config}")

def get_config_parameter(site: str, parameter: Union[str, list[str]], data_dir: str) -> Any:
    """Retrieve one or multiple configuration parameters for a given site.

    Args:
        site (str): Name of the site (used to locate config file).
        parameter (str | list[str]): Single key or list of keys to fetch.
        data_dir (str): Base directory containing all openMWR-managed data.

    Returns:
        Any: Single value if parameter is str, tuple of values if list.
    """
    with open(site_config_file(data_dir, site), "r") as file:
        config = json.load(file)

    if isinstance(parameter, str):
        return config.get(parameter)
    elif isinstance(parameter, list):
        return tuple(config.get(p) for p in parameter)
    else:
        raise TypeError("parameter must be str or list[str]")
