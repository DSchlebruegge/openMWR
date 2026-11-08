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
from openMWR.xr_utils import change_coord_of_ds, mean_n_min, calc_gradient, integrate, add_time_data, add_variables_at_closest_time
from openMWR.atm import dewpoint_from_absolute_humidity, calc_absolute_humidity, calc_pressure
from openMWR.site import load_site_config, get_invalid_measurements
from openMWR.paths import dataset_dir, dataset_file
from openMWR.observations import mask_invalid_measurements
from openMWR.hatpro_io import load_hatpro_day

logger = logging.getLogger(__name__)


class _NoUsableObservations(ValueError):
    """A daily source has no observations supported by the importer."""


def _standard_frequency_order(data):
    """Reorder measured channels by frequency without changing their identity."""
    frequencies = np.array([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40,
                            51.26, 52.28, 53.86, 54.94, 56.66, 57.30, 58.00])
    data = data.assign_coords(frq=data.frq.astype('float64').round(2))
    if not np.array_equal(np.sort(data.frq.values), frequencies):
        raise ValueError(f'Unsupported HATPRO frequencies: {data.frq.values.tolist()}. '
                         f'Expected channels: {frequencies.tolist()}.')
    return data.sel(frq=frequencies)


def _select_utc_day(data, date):
    """Keep the requested day, excluding the following midnight."""
    day = np.datetime64(date.strftime('%Y-%m-%d'))
    return data.isel(time=(data.time >= day) & (data.time < day + np.timedelta64(1, 'D')))


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
        The input dataset with available derived variables: `ah` (absolute humidity,
        g/m^3), `TD` (dew point, K), `T_grad` (temperature gradient, K/km),
        `p` (pressure, hPa), `lwp_from_lwc` (g/m^2), and `iwv_from_ah`
        (kg/m^2).

    Notes
    -----
    - Each quantity requires its corresponding predicted inputs.
    - The dataset is modified in place and returned for convenience.
    - Relative humidity values below 0.1% are clipped to 0.1% before calculations.
    - Pressure is derived from hydrostatic equilibrium when a finite surface
      pressure is available. Other derived products do not require pressure.
    """
    if 'T' in ds:
        ds['T_grad'] = calc_gradient(ds.T) * 1000
        ds['T_grad'].attrs['units'] = 'K/km'
        if 'surface_p' in ds and bool(np.isfinite(ds.surface_p).any()):
            ds['p'] = calc_pressure(ds.surface_p, ds.T, ds.height).where(np.isfinite(ds.T).all('height'))
            ds['p'].attrs['units'] = 'hPa'

    if {'T', 'rh'} <= set(ds):
        # Preserve missing outputs when different retrievals share one dataset.
        rh = ds.rh.clip(min=0.1)
        ds['ah'] = calc_absolute_humidity(ds.T, rh) * 1000
        ds['ah'].attrs['units'] = 'g/m^3'
        ds['TD'] = dewpoint_from_absolute_humidity(ds.T, ds.ah / 1000)
        ds['TD'].attrs['units'] = 'K'
        ds['iwv_from_ah'] = (integrate(ds.ah, 'height') / 1000).where(np.isfinite(ds.ah).all('height'))
        ds['iwv_from_ah'].attrs['units'] = 'kg/m^2'

    if 'lwc' in ds:
        ds['lwp_from_lwc'] = integrate(ds.lwc, 'height').where(np.isfinite(ds.lwc).all('height'))
        ds['lwp_from_lwc'].attrs['units'] = 'g/m^2'

    return ds

def import_hatpro_data(
    date: Union[datetime, pd.Timestamp],
    site: str,
    project_dir: str,
    mean_one_min: bool = False,
    include_rpg: bool = True,
    out_file: Optional[Union[str, os.PathLike]] = None,
    *, site_config=None,
) -> Optional[xr.Dataset]:
    """
    Load and assemble zenith-mode Hatpro measurements for a single day.

    Parameters
    ----------
    date : datetime or pandas.Timestamp
        Date used to construct RPG file paths like
        `{hatpro_data_dir}/Y%Y/M%m/D%d/%y%m%d.<suffix>[.NC]`.
        The site hatpro_format selects auto, netcdf or binary inputs for
        measurements and stored RPG retrieval products alike.
    site : str
        Site name used to resolve `hatpro_data_dir` from the site config.
    project_dir : str
        Working-project root containing data/ and sites/.
    mean_one_min : bool, optional
        If True, average all variables to 1-minute resolution.
    include_rpg : bool, optional
        If True, include available RPG retrieval products (LWP, LPR, TPC, HPC,
        IWV). Missing or unreadable products are logged and omitted.
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
        If the required microwave file is missing. MET, IRT and RPG retrievals
        are optional. Consumers validate their required variables.
    ValueError
        If no valid times remain after filtering.

    Notes
    -----
    - Variables are renamed to openMWR conventions (e.g., `TB`, `surface_T`).
    - Surface meteorology from `MET` is added by nearest time (max 2 minutes).
    - IR brightness temperatures are converted to Kelvin.
    - Measured channels are ordered by their frequencies. Unsupported grids fail.
    - Zenith microwave and IR records are averaged independently when requested.
    - Returned observations belong to the requested UTC day.
    """
    site_config = load_site_config(site, project_dir, config=site_config)
    periods = get_invalid_measurements(site, project_dir, site_config=site_config)
    products = ['BRT', 'MET', 'IRT']
    if include_rpg:
        products += ['LWP', 'LPR', 'TPC', 'HPC', 'IWV']
    datasets = load_hatpro_day(date, site, project_dir, products, site_config=site_config)
    ds_BRT = _standard_frequency_order(datasets['BRT'])
    ds_BRT['rain_flag'] = ds_BRT.rain_flag.astype('float64')
    ds_BRT = ds_BRT.where(ds_BRT['elevation_angle'].round(0) == 90., drop=True)
    ds_BRT = _select_utc_day(ds_BRT, date)
    if not ds_BRT.sizes['time']:
        raise _NoUsableObservations('No zenith observations remain in the requested UTC day after selecting elevation 90 degrees.')

    ds_MET = mask_invalid_measurements(datasets['MET'], periods) if 'MET' in datasets else None
    extra = []
    if include_rpg:
        extra.extend(datasets[key] for key in ('LWP', 'LPR', 'TPC', 'HPC', 'IWV') if key in datasets)
    ds_hatpro = xr.merge([ds_BRT, *extra], join='outer').sel(time=ds_BRT.time)
    if ds_MET is not None and ds_MET.sizes.get('time', 0):
        ds_hatpro = add_variables_at_closest_time(ds_hatpro, ds_MET, max_diff=2)
    ds_hatpro = mask_invalid_measurements(ds_hatpro, periods)

    if mean_one_min:
        ds_hatpro = mean_n_min(ds_hatpro, 1)

    if 'IRT' in datasets:
        infrared = _select_utc_day(mask_invalid_measurements(datasets['IRT'], periods), date)
        if infrared.sizes['time']:
            if mean_one_min:
                infrared = mean_n_min(infrared, 1)
            # Assignment aligns on microwave times/minutes. It neither drops
            # staggered samples before averaging nor creates IR-only minutes.
            ds_hatpro['TB_IR'] = infrared.TB_IR.isel(wavelength=0, drop=True)

    if 'T' in ds_hatpro:
        # Compute pressure from hydrostatic equilibrium
        if 'surface_p' in ds_hatpro and bool(np.isfinite(ds_hatpro.surface_p).any()):
            ds_hatpro['p'] = calc_pressure(ds_hatpro.surface_p, ds_hatpro.T, ds_hatpro.height)

        # Compute dew point temperature
        if 'ah' in ds_hatpro:
            ds_hatpro['TD'] = dewpoint_from_absolute_humidity(ds_hatpro.T, ds_hatpro.ah / 1000)

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
    project_dir: str,
    mean_one_min: bool = False,
    include_rpg: bool = True,
    out_file: Optional[Union[str, os.PathLike]] = None,
    *, site_config=None,
) -> Optional[xr.Dataset]:
    """
    Load and assemble boundary layer scan (BLS) Hatpro measurements for a day.

    Parameters
    ----------
    date : datetime or pandas.Timestamp
        Date used to construct RPG file paths like
        `{hatpro_data_dir}/Y%Y/M%m/D%d/%y%m%d.<suffix>[.NC]`.
        The site hatpro_format selects auto, netcdf or binary inputs for
        measurements and stored RPG retrieval products alike.
    site : str
        Site name used to resolve `hatpro_data_dir` from the site config.
    project_dir : str
        Working-project root containing data/ and sites/.
    mean_one_min : bool, optional
        If True, average surface meteorology to 1-minute resolution.
    include_rpg : bool, optional
        If True, include available RPG retrieval temperature profiles (TPB).
        Missing or unreadable products are logged and omitted.
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
        If the required scan file is missing. MET and RPG retrievals are
        optional. Consumers validate their required variables.
    ValueError
        If required scan angles or time samples are missing.

    Notes
    -----
    - Scans must contain all standard HATPRO BLS angles. Extra angles are
      retained. Measured angle labels are preserved and sorted.
    - A standard frequency grid is enforced if unexpected values are found.
    - The reader excludes the NetCDF surface-temperature channel from scan angles.
    - Surface meteorology from `MET` is added by nearest time (max 2 minutes).
    - Returned observations belong to the requested UTC day.
    """
    site_config = load_site_config(site, project_dir, config=site_config)
    periods = get_invalid_measurements(site, project_dir, site_config=site_config)
    products = ['MET', 'BLB'] + (['TPB'] if include_rpg else [])
    datasets = load_hatpro_day(date, site, project_dir, products, site_config=site_config)
    ds_MET = mask_invalid_measurements(datasets['MET'], periods) if 'MET' in datasets else None
    if ds_MET is not None and mean_one_min:
        ds_MET = mean_n_min(ds_MET, 1)
    ds_BLB = _standard_frequency_order(datasets['BLB'])
    ds_BLB['ang'] = ds_BLB.ang.astype('float64').round(2)
    ang = np.array([90.0, 30.0, 19.2, 14.4, 11.4, 8.4, 6.6, 5.4, 4.8, 4.2])
    missing_angles = np.setdiff1d(ang, ds_BLB.ang.values)
    if missing_angles.size:
        raise _NoUsableObservations(
            f'Incompatible BLS scan angles: missing required angles {missing_angles.tolist()}. '
            f'Measured angles: {ds_BLB.ang.values.tolist()}.')
    ds_BLB['rain_flag'] = ds_BLB.rain_flag.astype('float64')
    ds_BLB = ds_BLB.sortby('ang')
    ds_hatpro_bls = xr.merge([ds_BLB, datasets['TPB']], join='outer') if 'TPB' in datasets else ds_BLB
    if ds_MET is not None and ds_MET.sizes.get('time', 0):
        ds_hatpro_bls = add_variables_at_closest_time(ds_hatpro_bls, ds_MET, max_diff=2)
    ds_hatpro_bls = mask_invalid_measurements(ds_hatpro_bls, periods)
    ds_hatpro_bls = _select_utc_day(ds_hatpro_bls, date)
    if not ds_hatpro_bls.sizes['time']:
        raise _NoUsableObservations('No BLS time samples remain in the requested UTC day.')

    if 'T' in ds_hatpro_bls and 'surface_p' in ds_hatpro_bls and bool(np.isfinite(ds_hatpro_bls.surface_p).any()):
        # Compute pressure from hydrostatic equilibrium
        ds_hatpro_bls['p'] = calc_pressure(ds_hatpro_bls.surface_p, ds_hatpro_bls.T, ds_hatpro_bls.height)

    ds_hatpro_bls = add_time_data(ds_hatpro_bls)

    if out_file is not None:
        ds_hatpro_bls.to_netcdf(out_file)
        ds_hatpro_bls.close()
        return None

    return ds_hatpro_bls

def _create_hatpro_day_file(
    date: Union[datetime, pd.Timestamp],
    bls: bool,
    bls_attribute: str,
    site: str,
    project_dir: str,
    hatpro_out_dir: Union[str, os.PathLike],
    include_rpg: bool,
    site_config,
) -> None:

    daily_file = hatpro_out_dir / f'hatpro_data{bls_attribute}_{date:%Y%m%d}.nc'

    try:
        if bls:
            import_hatpro_data_bls(
                date,
                site,
                project_dir,
                include_rpg=include_rpg,
                out_file=daily_file, site_config=site_config,
            )
        else:
            import_hatpro_data(
                date,
                site,
                project_dir,
                mean_one_min=True,
                include_rpg=include_rpg,
                out_file=daily_file, site_config=site_config,
            )
    except FileNotFoundError as e:
        logger.info(f"Date skipped due to missing file: {date} – {e}")
    except _NoUsableObservations as e:
        logger.info('Skipping HATPRO %s for %s %s: %s',
                    'BLS' if bls else 'zenith', site, date.strftime('%Y-%m-%d'), e)
    except Exception as e:
        logger.error(f"Error importing hatpro data for {date}: {e}")
        logger.info(e, exc_info=True)

def create_hatpro_dataset(
    site: str,
    project_dir: str,
    include_rpg: bool = False,
    num_of_processes: int = 1,
    update_only: bool = False,
    dataset: str = 'hatpro',
    start_date: str = None,
    end_date: str = None,
    *, site_config=None,
) -> None:
    """
    Build site-level Hatpro datasets from daily RPG files.

    Parameters
    ----------
    site : str
        Site name used for configuration and output paths.
    project_dir : str
        Working-project root containing data/ and sites/.
    include_rpg : bool, optional
        When True, available RPG retrieval products are included in the output.
        Missing or unreadable products do not prevent observation processing.
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
    - Writes temporary daily files to `data/sites/{site}/{dataset}/`
      and combines them into `zenith.nc` and `bls.nc`, removing daily files after a
      successful merge.
    - Optional start_date/end_date override measurement-start through today.
    - Combined-build failures are logged and re-raised to the caller.
    """

    site_config = load_site_config(site, project_dir, config=site_config)
    site_config.require('hatpro_data_dir')
    start_date = pd.Timestamp(start_date or site_config.require('mwr_measurement_start_date'))
    end_date = pd.Timestamp(end_date or pd.Timestamp.now().normalize())
    dr_full = pd.date_range(start_date, end_date)
    hatpro_out_dir = dataset_dir(project_dir, site, dataset)
    hatpro_out_dir.mkdir(parents=True, exist_ok=True)

    for bls in [False, True]:

        logger.info(f'{"Updating" if update_only else "Creating"} Hatpro data for site {site} with BLS: {bls}')

        bls_attribute = '_bls' if bls else ''
        daily_pattern = f'hatpro_data{bls_attribute}_' + '[0-9]' * 8 + '.nc'
        combined_file = dataset_file(project_dir, site, dataset, bls=bls)
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
                dr = pd.date_range(start=max(start_date, last_date), end=end_date)
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
                        project_dir,
                        hatpro_out_dir,
                        include_rpg, site_config,
                    )
            else:
                run_pool_date_range(
                    dr,
                    num_of_processes,
                    _create_hatpro_day_file,
                    bls,
                    bls_attribute,
                    site,
                    project_dir,
                    hatpro_out_dir,
                    include_rpg, site_config,
                )

            logger.info(f'Finished creating daily files for Hatpro data with BLS: {bls}')

            files = sorted(glob.glob(str(hatpro_out_dir / daily_pattern)))

            if len(files) == 0:
                logger.info(f'No Hatpro daily files created for site {site} with BLS: {bls}')
                continue

            if update_only_bls:
                tmp_file = combined_file.with_name(f"{combined_file.name}.tmp")
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

            # Days can overlap and contain different optional RPG products.
            with xr.open_mfdataset(
                files,
                combine='nested',
                concat_dim='time',
                coords='minimal',
                join='outer',
                compat='no_conflicts',
            ) as ds_hatpro_new:
                ds_hatpro = ds_hatpro_new.drop_duplicates(dim='time').sortby("time")

                if update_only_bls:
                    with xr.open_dataset(combined_file) as ds_hatpro_old:
                        ds_hatpro = xr.concat(
                            [ds_hatpro_old, ds_hatpro], dim='time', coords='minimal',
                            join='outer', compat='no_conflicts',
                        ).drop_duplicates(dim='time').sortby("time")
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
            raise

        finally:
            if tmp_file is not None and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    logger.exception(f"Failed to remove temporary file: {tmp_file}")

            files = glob.glob(str(hatpro_out_dir / daily_pattern))
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
