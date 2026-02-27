import xarray as xr
import numpy as np
import pandas as pd
import logging
import time
import os
from datetime import datetime
from typing import Optional, Union
import glob

from openMWR.utils import (
    progress_log,
)
from openMWR.parallel import run_pool_date_range
from openMWR.xr_utils import change_coord_of_ds, swap_dims_and_rename, mean_n_min, calc_gradient, integrate, add_time_data, add_variables_at_closest_time
from openMWR.atm import dewpoint_from_absolute_humidity, calc_absolute_humidity, calc_pressure
from openMWR.site import get_config_parameter
from openMWR.paths import site_subdir

logger = logging.getLogger(__name__)

def _set_time_encoding_to_seconds(ds: xr.Dataset, time_dim: str = "time") -> None:
    """
    Encode datetime coordinates as integer seconds to avoid minute-unit warnings.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a datetime coordinate.
    time_dim : str, optional
        Name of the datetime coordinate, by default "time".
    """
    if time_dim not in ds.coords:
        return
    if not np.issubdtype(ds[time_dim].dtype, np.datetime64):
        return

    ds[time_dim].encoding.update(
        {
            "units": "seconds since 1970-01-01 00:00:00",
            "dtype": "int64",
            "calendar": "proleptic_gregorian",
        }
    )

def calc_derived_quantities(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute and append derived meteorological quantities to a retrieval dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing retrieval variables such as `T`, `rh`, `surface_p`,
        `height`, and `lwc`. Units should be
        (K, %, hPa, m, and g/m^3 for `lwc`).

    Returns
    -------
    xr.Dataset
        The input dataset with added variables: `ah` (absolute humidity,
        g/m^3), `TD` (dew point, K), `T_grad` (temperature gradient, K/km),
        `p` (pressure, hPa), `lwp_from_lwc` (g/m^2), and `iwv_from_ah`
        (kg/m^2).

    Notes
    -----
    - The dataset is modified in place and returned for convenience.
    - Relative humidity values below 0.1% are clipped to 0.1% before calculations.
    - Pressure is derived from hydrostatic equilibrium using the surface
      pressure and temperature profile.
    """
    # Compute absolute humidity
    rh = ds['rh'].where(ds['rh'] > 0.1, 0.1)

    ds['ah'] = calc_absolute_humidity(ds.T, rh) * 1000
    ds['ah'].attrs["units"] = "g/m^3"
    # ds['ah'] = ds['ah'].where(ds['ah'] != 0, 0.1)  # Set zeros to a small value so dew point calculation works

    # Compute dew point temperature
    ds['TD'] = dewpoint_from_absolute_humidity(ds.T, ds.ah / 1000)
    ds['TD'].attrs["units"] = "K"

    # Compute temperature gradient
    ds['T_grad'] = calc_gradient(ds.T) * 1000
    ds['T_grad'].attrs["units"] = "K/km"

    # Compute pressure from hydrostatic equilibrium
    ds['p'] = calc_pressure(ds.surface_p, ds.T, ds.height)
    ds['p'].attrs["units"] = "hPa"

    # Compute LWP from LWC
    ds['lwp_from_lwc'] = integrate(ds.lwc, 'height')
    ds['lwp_from_lwc'].attrs["units"] = "g/m^2"

    # Compute IWV from absolute humidity
    ds['iwv_from_ah'] = integrate(ds.ah, 'height') / 1000
    ds['iwv_from_ah'].attrs["units"] = "kg/m^2"

    return ds

def _get_naming_convention(ds_BRT: xr.Dataset) -> tuple[dict[str, str], dict[str, str]]:
    """
    Identify the variable and dimension naming convention used by RPG files.

    Parameters
    ----------
    ds_BRT : xr.Dataset
        Brightness temperature dataset used to inspect software version
        attributes.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Mapping dictionaries `(var_names, dim_names)` where keys are openMWR
        internal names (e.g., `TB`, `surface_p`, `frequencie_dim`) and values
        are the corresponding names in the input dataset.

    Notes
    -----
    The function expects either `radiometer_software_version` or
    `Radiometer_Software_Version` to be present in `ds_BRT.attrs`.
    """

    if 'radiometer_software_version' in ds_BRT.attrs:  # Different naming convention from v10+; check attribute capitalization
        var_names = {'TB': 'TBs',
                    'surface_p': 'env_pressure',
                    'surface_T': 'env_temperature',
                    'surface_rh': 'env_relative_humidity',
                    'TB_IR': 'IRR_data',
                    'rain_flag': 'rain_flag',
                    'azimuth_angle': 'azimuth_angle',
                    'elevation_angle': 'elevation_angle',
                    'lwp': 'LWP_data',
                    'lwc': 'liquid_water_profiles',
                    'T': 'temperature_profiles',
                    'ah': 'Absolute_Humidity_Profiles',
                    'rh': 'relative_humidity_profiles',
                    'iwv': 'IWV_data',
        }
        dim_names = {
            'altitude_dim': 'number_altitude_layers',
            'altitude_var': 'altitude_layers',
            'frequencie_dim': 'number_frequencies',
            'frequencie_var': 'frequencies',
            'scan_angles_dim': 'number_scan_angles',
            'scan_angles_var': 'elevation_scan_angles',
        }

    elif 'Radiometer_Software_Version' in ds_BRT.attrs:
        var_names = {'TB': 'TBs',
                    'surface_p': 'Surf_P',
                    'surface_T': 'Surf_T',
                    'surface_rh': 'Surf_RH',
                    'TB_IR': 'IRR_Map',
                    'rain_flag': 'RF',
                    'azimuth_angle': 'AziAng',
                    'elevation_angle': 'ElAng',
                    'lwp': 'LWP',
                    'lwc': 'LWC_prof',
                    'T': 'T_prof',
                    'ah': 'AH_Prof',
                    'rh': 'RH_prof',
                    'iwv': 'IWV',
        }
        dim_names = {
            'altitude_dim': 'altitude_layer',
            'altitude_var': 'altitude',
            'frequencie_dim': 'number_frequencies',
            'frequencie_var': 'Freq',
            'scan_angles_dim': 'number_scan_angles',
            'scan_angles_var': 'ElAngs',
        }
    return var_names, dim_names

def import_hatpro_data(
    date: Union[datetime, pd.Timestamp],
    site: str,
    data_dir: str,
    mean_one_min: bool = False,
    import_retrieval_data: bool = True,
    out_file: Optional[Union[str, os.PathLike]] = None,
) -> Optional[xr.Dataset]:
    """
    Load and assemble zenith-mode Hatpro measurements for a single day.

    Parameters
    ----------
    date : datetime or pandas.Timestamp
        Date used to construct RPG file paths like
        `{hatpro_data_dir}/Y%Y/M%m/D%d/%y%m%d.<suffix>.NC`.
    site : str
        Site name used to resolve `hatpro_data_dir` from the site config.
    data_dir : str
        Base directory containing all openMWR-managed data.
    mean_one_min : bool, optional
        If True, average all variables to 1-minute resolution.
    import_retrieval_data : bool, optional
        If True, include RPG retrieval products (LWP, LPR, TPC, HPC, IWV).
    out_file : str or os.PathLike, optional
        If provided, save the resulting daily dataset to this NetCDF file.

    Returns
    -------
    xr.Dataset or None
        Combined dataset with standardized variable names and coordinates,
        filtered to zenith scans (`elevation_angle` ~ 90 deg). When retrieval
        data is included, profile and column products are merged as well.
        Returns `None` when `out_file` is provided.

    Raises
    ------
    FileNotFoundError
        If any expected RPG netCDF file is missing.
    ValueError
        If no valid times remain after filtering.

    Notes
    -----
    - Variables are renamed to openMWR conventions (e.g., `TB`, `surface_T`).
    - Surface meteorology from `MET` is added by nearest time (max 2 minutes).
    - IR brightness temperatures are converted to Kelvin.
    - A standard frequency grid is enforced if unexpected values are found.
    """
    hatpro_data_dir = get_config_parameter(site, 'hatpro_data_dir', data_dir)

    file_base = f"{hatpro_data_dir}/{date:Y%Y/M%m/D%d}/{date:%y%m%d}"

    file_suffixes = ["BRT", "MET", "IRT"]
    if import_retrieval_data:
        file_suffixes += ["LWP", "LPR", "TPC", "HPC", "IWV"]

    # Build file paths
    files = {suffix: f"{file_base}.{suffix}.NC" for suffix in file_suffixes}

    # Check existence
    for suffix, file in files.items():
        if not os.path.exists(file):
            raise FileNotFoundError(f"Hatpro measurement file not found: '{file}'")

    # Load all files as datasets
    datasets = {
        suffix: xr.load_dataset(file, decode_timedelta=True).drop_duplicates('time').dropna(dim='time')
        for suffix, file in files.items()
    }

    ds_BRT = datasets["BRT"]

    var_names, dim_names = _get_naming_convention(ds_BRT)

    ds_BRT = swap_dims_and_rename(ds_BRT, old_dim=dim_names['frequencie_dim'], old_var=dim_names['frequencie_var'], new_dim="frq")
    ds_BRT = ds_BRT[[var_names['TB'], var_names['rain_flag'], var_names['azimuth_angle'], var_names['elevation_angle']]]
    ds_BRT[var_names['rain_flag']] = ds_BRT[var_names['rain_flag']].astype("float64")

    ds_BRT["frq"] = ds_BRT["frq"].astype("float64").round(2)
    frq = np.array([22.24,23.04,23.84,25.44,26.24,27.84,31.40,51.26,52.28,53.86,54.94,56.66,57.30,58.00])
    if not np.array_equal(ds_BRT["frq"].values, frq):
        logger.error(f'Frequency variable has unexpected values: {ds_BRT["frq"].values}. Using standard frequencies.')
        ds_BRT["frq"] = frq

    ds_MET = datasets["MET"]
    ds_MET = ds_MET[[var_names['surface_p'], var_names['surface_T'], var_names['surface_rh']]]

    ds_IRT = datasets["IRT"]
    ds_IRT = ds_IRT[var_names['TB_IR']]
    ds_IRT = ds_IRT.isel(number_wavelength=0)
    ds_IRT += 273.15
    ds_IRT.attrs["units"] = "K"

    if import_retrieval_data:
        ds_LWP = datasets["LWP"]
        ds_LPR = datasets["LPR"]
        ds_TPC = datasets["TPC"]
        ds_HPC = datasets["HPC"]
        ds_IWV = datasets["IWV"]

        ds_LWP = ds_LWP[var_names['lwp']]

        ds_IWV = ds_IWV[var_names['iwv']]

        ds_LPR = swap_dims_and_rename(ds_LPR, old_dim=dim_names['altitude_dim'], old_var=dim_names['altitude_var'], new_dim="height_lwc")
        ds_LPR = ds_LPR[var_names['lwc']]

        ds_TPC = swap_dims_and_rename(ds_TPC, old_dim=dim_names['altitude_dim'], old_var=dim_names['altitude_var'], new_dim="height")
        ds_TPC = ds_TPC[var_names['T']]

        ds_HPC = swap_dims_and_rename(ds_HPC, old_dim=dim_names['altitude_dim'], old_var=dim_names['altitude_var'], new_dim="height")
        ds_HPC = ds_HPC[[var_names['ah'], var_names['rh']]]

        # Merge datasets into a single dataset
        ds_hatpro = xr.merge([ds_BRT, ds_IRT, ds_LWP, ds_LPR, ds_TPC, ds_HPC, ds_IWV], join='outer')
    else:
        ds_hatpro = xr.merge([ds_BRT, ds_IRT], join='outer')

    # Add the nearest surface met data in time
    ds_hatpro = add_variables_at_closest_time(ds_hatpro, ds_MET, max_diff=2)

    # Rename variables
    ds_hatpro = ds_hatpro.rename({v: k for k, v in var_names.items() if v in ds_hatpro})

    # Filter
    #ds_hatpro = ds_hatpro.where((ds_hatpro['elevation_angle'].round(0) == 90.) & (ds_hatpro['azimuth_angle'].round(0) == 0.), drop=True)
    ds_hatpro = ds_hatpro.where(ds_hatpro['elevation_angle'].round(0) == 90., drop=True)

    if mean_one_min:
        ds_hatpro = mean_n_min(ds_hatpro, 1)  # Useful to avoid many NaNs from mismatched time coords

    if import_retrieval_data:
        # Compute pressure from hydrostatic equilibrium
        ds_hatpro['p'] = calc_pressure(ds_hatpro.surface_p, ds_hatpro.T, ds_hatpro.height)

        # Compute dew point temperature
        ds_hatpro['TD'] = dewpoint_from_absolute_humidity(ds_hatpro.T, ds_hatpro.ah / 1000)

    # Check if any values remain in the dataset
    if ds_hatpro["time"].size == 0:
        raise ValueError("The dataset has an empty 'time' coordinate.")

    ds_hatpro = add_time_data(ds_hatpro)

    if out_file is not None:
        if ds_hatpro["time"].size > 0:
            ds_hatpro.to_netcdf(out_file)
        ds_hatpro.close()
        return None

    return ds_hatpro

def import_hatpro_data_bls(
    date: Union[datetime, pd.Timestamp],
    site: str,
    data_dir: str,
    mean_one_min: bool = False,
    import_retrieval_data: bool = True,
    out_file: Optional[Union[str, os.PathLike]] = None,
) -> Optional[xr.Dataset]:
    """
    Load and assemble boundary layer scan (BLS) Hatpro measurements for a day.

    Parameters
    ----------
    date : datetime or pandas.Timestamp
        Date used to construct RPG file paths like
        `{hatpro_data_dir}/Y%Y/M%m/D%d/%y%m%d.<suffix>.NC`.
    site : str
        Site name used to resolve `hatpro_data_dir` from the site config.
    data_dir : str
        Base directory containing all openMWR-managed data.
    mean_one_min : bool, optional
        If True, average surface meteorology to 1-minute resolution.
    import_retrieval_data : bool, optional
        If True, include RPG retrieval temperature profiles (TPB).
    out_file : str or os.PathLike, optional
        If provided, save the resulting daily dataset to this NetCDF file.

    Returns
    -------
    xr.Dataset or None
        Combined dataset with standardized variable names and coordinates,
        including scan angle (`ang`) and frequency (`frq`) dimensions.
        Returns `None` when `out_file` is provided.

    Raises
    ------
    FileNotFoundError
        If any expected RPG netCDF file is missing.
    ValueError
        If the elevation angle dimension does not match the expected length.

    Notes
    -----
    - A standard elevation angle grid and frequency grid are enforced if
      unexpected values are found.
    - The smallest elevation angle (0 deg) is dropped after sorting.
    - Surface meteorology from `MET` is added by nearest time (max 2 minutes).
    """
    hatpro_data_dir = get_config_parameter(site, 'hatpro_data_dir', data_dir)

    file_base = f"{hatpro_data_dir}/{date:Y%Y/M%m/D%d}/{date:%y%m%d}"

    file_suffixes = ["MET", "BLB"]
    if import_retrieval_data:
        file_suffixes += ["TPB"]

    # Build file paths
    files = {suffix: f"{file_base}.{suffix}.NC" for suffix in file_suffixes}

    # Check existence
    for suffix, file in files.items():
        if not os.path.exists(file):
            raise FileNotFoundError(f"Hatpro measurement file not found: '{file}'")

    # Load all files as datasets
    datasets = {
        suffix: xr.load_dataset(file).drop_duplicates('time').dropna(dim='time')
        for suffix, file in files.items()
    }

    ds_MET = datasets["MET"]
    ds_BLB = datasets["BLB"]

    if import_retrieval_data:
        ds_TPB = datasets["TPB"]

    if mean_one_min:
        ds_MET = mean_n_min(ds_MET, 1)

    var_names, dim_names = _get_naming_convention(ds_BLB)

    ds_MET = ds_MET[[var_names['surface_p'], var_names['surface_T'], var_names['surface_rh']]]

    ds_BLB = swap_dims_and_rename(ds_BLB, old_dim=dim_names['frequencie_dim'], old_var=dim_names['frequencie_var'], new_dim="frq")
    ds_BLB = swap_dims_and_rename(ds_BLB, old_dim=dim_names['scan_angles_dim'], old_var=dim_names['scan_angles_var'], new_dim='ang')

    ds_BLB["ang"] = ds_BLB["ang"].astype("float64").round(2)
    ang = np.array([90.0, 30.0, 19.2, 14.4, 11.4, 8.4, 6.6, 5.4, 4.8, 4.2, 0])
    if len(ds_BLB["ang"].values) != len(ang):
        raise ValueError(f'Elevation angle variable for BLS has unexpected length: {len(ds_BLB["ang"].values)}. Expected: {len(ang)}.')
    if not np.array_equal(ds_BLB["ang"].values, ang):
        logger.error(f'Elevation angle variable for BLS has unexpected values: {ds_BLB["ang"].values}. Using standard angles.')
        ds_BLB["ang"] = ang


    ds_BLB["frq"] = ds_BLB["frq"].astype("float64").round(2)
    frq = np.array([22.24,23.04,23.84,25.44,26.24,27.84,31.40,51.26,52.28,53.86,54.94,56.66,57.30,58.00])
    if not np.array_equal(ds_BLB["frq"].values, frq):
        logger.error(f'Frequency variable for BLS has unexpected values: {ds_BLB["frq"].values}. Using standard frequencies.')
        ds_BLB["frq"] = frq

    if 'AzAng' in ds_BLB:
        ds_BLB = ds_BLB.rename({'AzAng' :var_names['azimuth_angle']})

    ds_BLB = ds_BLB[[var_names['TB'], var_names['rain_flag'], var_names['azimuth_angle']]]
    ds_BLB[var_names['rain_flag']] = ds_BLB[var_names['rain_flag']].astype("float64")

    ds_BLB = ds_BLB.sortby('ang')
    ds_BLB = ds_BLB.sel(ang=slice(1, None))

    if import_retrieval_data:
        ds_TPB = swap_dims_and_rename(ds_TPB, old_dim=dim_names['altitude_dim'], old_var=dim_names['altitude_var'], new_dim="height")
        ds_TPB = ds_TPB[var_names['T']]

        ds_hatpro_bls = xr.merge([ds_BLB, ds_TPB], join='outer')
    else:
        ds_hatpro_bls = ds_BLB

    # Add the nearest surface met data in time
    ds_hatpro_bls = add_variables_at_closest_time(ds_hatpro_bls, ds_MET, max_diff=2)

    # Rename variables
    ds_hatpro_bls = ds_hatpro_bls.rename({v: k for k, v in var_names.items() if v in ds_hatpro_bls})

    # Filter
    #ds_hatpro_bls = ds_hatpro_bls.where(ds_hatpro_bls['azimuth_angle'].round(0) == 0., drop=True)


    if import_retrieval_data:
        # Compute pressure from hydrostatic equilibrium
        ds_hatpro_bls['p'] = calc_pressure(ds_hatpro_bls.surface_p, ds_hatpro_bls.T, ds_hatpro_bls.height)

    ds_hatpro_bls = add_time_data(ds_hatpro_bls)

    if out_file is not None:
        if not ds_hatpro_bls["time"].size > 0:
            raise ValueError("The hatpro dataset has an empty 'time' coordinate before writing.")

        ds_hatpro_bls.to_netcdf(out_file)
        ds_hatpro_bls.close()
        return None

    return ds_hatpro_bls

def _create_hatpro_day_file(
    date: Union[datetime, pd.Timestamp],
    bls: bool,
    bls_attribute: str,
    site: str,
    data_dir: str,
    hatpro_out_dir: Union[str, os.PathLike],
    import_retrieval_data: bool,
) -> None:

    daily_file = hatpro_out_dir / f'hatpro_data{bls_attribute}_{date:%Y%m%d}.nc'

    try:
        if bls:
            import_hatpro_data_bls(
                date,
                site,
                data_dir,
                import_retrieval_data=import_retrieval_data,
                out_file=daily_file,
            )
        else:
            import_hatpro_data(
                date,
                site,
                data_dir,
                mean_one_min=True,
                import_retrieval_data=import_retrieval_data,
                out_file=daily_file,
            )
    except FileNotFoundError as e:
        logger.info(f"Date skipped due to missing file: {date} – {e}")
    except ValueError as e:
        if "empty 'time' coordinate" in str(e):
            logger.info(f"Error importing hatpro data for {date} – {e}")
        else:
            logger.error(f"Error importing hatpro data for {date}: {e}")
            logger.info(e, exc_info=True)
    except Exception as e:
        logger.error(f"Error importing hatpro data for {date}: {e}")
        logger.info(e, exc_info=True)

def create_hatpro_dataset(
    site: str,
    data_dir: str,
    import_retrieval_data: Optional[bool] = None,
    num_of_processes: int = 1,
    update_only: bool = False,
) -> None:
    """
    Build site-level Hatpro datasets from daily RPG files.

    Parameters
    ----------
    site : str
        Site name used for configuration and output paths.
    data_dir : str
        Base directory containing all openMWR-managed data.
    import_retrieval_data : bool, optional
        If None, the site config parameter `rpg_retrieval_exists` is used.
        When True, RPG retrieval products are included in the output.
    num_of_processes : int, optional
        Number of processes used to create daily files.
    update_only : bool, optional
        If True and a combined Hatpro file already exists, concatenate the
        newly created data to the existing dataset and drop duplicate times.

    Returns
    -------
    None

    Notes
    -----
    - Creates both zenith-mode and BLS datasets.
    - Writes temporary daily files to `data/sites/{site}/hatpro/` and combines
      them into `hatpro_data(_bls).nc`, removing the daily files after a
      successful merge.
    - The date range is derived from `mwr_measurement_start_date` up to today.
    """

    if import_retrieval_data is None:
        import_retrieval_data = get_config_parameter(site, 'rpg_retrieval_exists', data_dir)

    mwr_measurement_start_date = get_config_parameter(site, 'mwr_measurement_start_date', data_dir)
    dr_full = pd.date_range(mwr_measurement_start_date, pd.Timestamp.now().normalize())
    hatpro_out_dir = site_subdir(data_dir, site, "hatpro")

    for bls in [False, True]:

        logger.info(f'{"Updating" if update_only else "Creating"} Hatpro data for site {site} with BLS: {bls}')

        bls_attribute = '_bls' if bls else ''
        combined_file = hatpro_out_dir / f'hatpro_data{bls_attribute}.nc'
        tmp_file = None

        if update_only and os.path.exists(combined_file):
            update_only_bls = True
        elif update_only and not os.path.exists(combined_file):
            logger.warning(
                f"Hatpro data file {combined_file} does not exist. "
                "Creating dataset from scratch although update_only=True."
            )
            update_only_bls = False
        else:
            update_only_bls = False


        try:
            if update_only_bls:
                with xr.open_dataset(combined_file) as ds_hatpro_old:
                    last_date = pd.Timestamp(ds_hatpro_old.time.max().values).normalize()
                logger.info(f"Last date in dataset: {last_date}")
                dr = pd.date_range(start=last_date, end=pd.Timestamp.now())
            else:
                dr = dr_full

            if num_of_processes == 1:
                total = len(dr)
                start_time = time.time()

                for i, date in enumerate(dr):
                    if i % 5 == 0:
                        logger.info(progress_log(i, total, start_time))

                    _create_hatpro_day_file(
                        date,
                        bls,
                        bls_attribute,
                        site,
                        data_dir,
                        hatpro_out_dir,
                        import_retrieval_data,
                    )
            else:
                run_pool_date_range(
                    dr,
                    num_of_processes,
                    _create_hatpro_day_file,
                    bls,
                    bls_attribute,
                    site,
                    data_dir,
                    hatpro_out_dir,
                    import_retrieval_data,
                )

            logger.info(f'Finished creating daily files for Hatpro data with BLS: {bls}')

            files = glob.glob(str(hatpro_out_dir / f'hatpro_data{bls_attribute}_*.nc'))

            if len(files) == 0:
                logger.info(f'No Hatpro daily files created for site {site} with BLS: {bls}')
                continue

            if update_only_bls:
                tmp_file = combined_file.with_name(f"{combined_file.name}.tmp")
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

            with xr.open_mfdataset(
                files,
                combine='by_coords',
                join='outer',
                compat='no_conflicts',
            ) as ds_hatpro_new:
                ds_hatpro = ds_hatpro_new.drop_duplicates(dim='time').sortby("time")

                if update_only_bls:
                    with xr.open_dataset(combined_file) as ds_hatpro_old:
                        ds_hatpro = xr.concat([ds_hatpro_old, ds_hatpro], dim='time').drop_duplicates(dim='time').sortby("time")
                        _set_time_encoding_to_seconds(ds_hatpro)
                        ds_hatpro.to_netcdf(tmp_file)
                else:
                    _set_time_encoding_to_seconds(ds_hatpro)
                    ds_hatpro.to_netcdf(combined_file)

            if update_only_bls and tmp_file is not None:
                os.replace(tmp_file, combined_file)

            logger.info(f'{combined_file} saved')

        except Exception as e:
            logger.error(f"Error creating Hatpro dataset for site {site} with BLS: {bls}: {e}", exc_info=True)

        finally:
            if tmp_file is not None and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    logger.exception(f"Failed to remove temporary file: {tmp_file}")

            files = glob.glob(str(hatpro_out_dir / f'hatpro_data{bls_attribute}_*.nc'))
            # Delete old files
            for f in files:
                try:
                    os.remove(f)
                except OSError:
                    logger.exception(f"Failed to remove temporary file: {f}")

            logger.info(f'Cleaned up daily files for site {site} with BLS: {bls}')

def change_height_coord_of_ds_hatpro(ds_hatpro: xr.Dataset, new_heights: np.ndarray) -> xr.Dataset:
    """
    Regrid Hatpro height coordinates to a new vertical grid.

    Parameters
    ----------
    ds_hatpro : xr.Dataset
        Hatpro dataset containing `height` and `height_lwc` dimensions.
    new_heights : np.ndarray
        Target height grid in meters.

    Returns
    -------
    xr.Dataset
        Dataset interpolated to `new_heights` with `height_lwc` merged into
        `height` for `lwc`.
    """
    ds_hatpro = change_coord_of_ds(ds_hatpro, new_heights, extrapolate_lower_end=True)
    ds_hatpro = change_coord_of_ds(ds_hatpro, new_heights, extrapolate_lower_end=True, dim='height_lwc')
    ds_hatpro['lwc'] = ds_hatpro['lwc'].rename({'height_lwc': 'height'})
    ds_hatpro = ds_hatpro.drop_vars('height_lwc')
    return ds_hatpro
