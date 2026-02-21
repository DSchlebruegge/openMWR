import numpy as np
import xarray as xr
import logging
import os
import glob
from typing import Optional, Sequence

from openMWR.cloud import CloudColumn, CloudModelConfig
from openMWR.atm import dewpoint_from_absolute_humidity, calc_absolute_humidity
from openMWR.xr_utils import change_coord_of_ds, integrate, add_time_data, select_closest_times
from openMWR.site import get_config_parameter
from openMWR.run_RT import run_RT
from openMWR.hatpro_data import change_height_coord_of_ds_hatpro
from openMWR.paths import site_subdir, radiosonde_station_dir

logger = logging.getLogger(__name__)

def add_cloud_to_ds(ds_date):
    """
    Add adiabatic liquid water content profile to a radiosonde slice.

    Parameters
    ----------
    ds_date : xarray.Dataset
        Single-profile dataset containing ``height`` (m), ``T`` (K), ``rh`` (%),
        and ``p`` (hPa).

    Returns
    -------
    xarray.Dataset
        Dataset with additional ``lwc`` variable (g/m^3) on the ``height`` grid.
    """
    z_m = ds_date.height.values
    T_K = ds_date.T.values
    rh_1 = ds_date.rh.values / 100
    p_Pa = ds_date.p.values * 100

    config = CloudModelConfig(rh_thres=0.96)
    cloud_column = CloudColumn(z_m, p_Pa, T_K, rh_1, config)
    lwc_gpm3 = cloud_column.calculate_lwc()
    
    ds_date['lwc'] = xr.DataArray(lwc_gpm3, dims=['height'])
    ds_date['lwc'].attrs["units"] = "g/m^3"

    return ds_date

def add_derived_data(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds derived data to the radiosonde dataset.
    This function calculates absolute humidity, dew point, liquid water path (LWP), and integrated water vapor (IWV)
    from the existing temperature, relative humidity, and liquid water content variables in the dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing temperature, relative humidity, and liquid water content variables.
    
    Returns
    -------
    xr.Dataset
        The updated dataset with additional derived variables.
    """

    ds['ah'] = calc_absolute_humidity(ds.T, ds.rh)
    ds['ah'].attrs["units"] = "kg/m^3"

    ds['TD'] = dewpoint_from_absolute_humidity(ds.T, ds.ah)
    ds['TD'].attrs["units"] = "K"

    ds['lwp'] = integrate(ds['lwc'], 'height')
    ds['lwp'].attrs["units"] = "g/m^2"

    ds['iwv'] = integrate(ds['ah'], 'height')
    ds['iwv'].attrs["units"] = "kg/m^2"

    return ds

def _scale_rh_profiles_for_clear_sky_cases(
    ds: xr.Dataset,
    fraction: float = 0.1,
    scale_factor: float = 0.5,
    random_seed: Optional[int] = None,
    lwc_reference: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """
    Scales RH profiles for a random subset of clear-sky cases (lwc == 0).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'rh' and 'lwc' variables with a 'synop' dimension.
    fraction : float, optional
        Fraction of clear-sky cases to scale (default is 0.1).
    scale_factor : float, optional
        Multiplicative scaling factor for RH (default is 0.5).
    random_seed : int, optional
        Seed for reproducible random selection.
    lwc_reference : xr.DataArray, optional
        Optional LWC reference used to determine clear-sky cases. Can be a
        profile (with 'height') or a precomputed max over height.

    Returns
    -------
    xr.Dataset
        Dataset with scaled RH profiles for selected clear-sky cases.
    """
    if 'rh' not in ds:
        logger.warning("Dataset has no 'rh' variable. Skipping RH scaling.")
        return ds

    if fraction <= 0:
        logger.info("RH scaling fraction <= 0. Skipping RH scaling.")
        return ds
    if fraction > 1:
        logger.warning("RH scaling fraction > 1. Clipping to 1.")
        fraction = 1.0
    if scale_factor <= 0:
        logger.warning("RH scaling factor must be positive. Skipping RH scaling.")
        return ds

    if lwc_reference is None:
        if 'lwc' not in ds:
            logger.warning("Dataset has no 'lwc' variable. Skipping RH scaling.")
            return ds
        lwc_reference = ds['lwc']

    if 'height' in lwc_reference.dims:
        lwc_max = lwc_reference.max(dim='height', skipna=True)
    else:
        lwc_max = lwc_reference

    if 'synop' not in lwc_max.dims:
        logger.warning("LWC reference is missing 'synop' dimension. Skipping RH scaling.")
        return ds

    clear_mask = lwc_max <= 0
    clear_synops = ds['synop'].where(clear_mask, drop=True).values

    if clear_synops.size == 0:
        logger.info("No clear-sky profiles found for RH scaling.")
        return ds

    n_select = int(np.round(clear_synops.size * fraction))
    if n_select <= 0:
        logger.info("No clear-sky profiles selected for RH scaling.")
        return ds

    rng = np.random.default_rng(random_seed)
    selected_synops = rng.choice(clear_synops, size=n_select, replace=False)

    logger.info(f"Before scaling: {ds['rh'].sel(synop=selected_synops)}")

    synop_mask = ds['synop'].isin(selected_synops)
    ds['rh'] = ds['rh'].where(~synop_mask, ds['rh'] * scale_factor)

    logger.info(f"After scaling: {ds['rh'].sel(synop=selected_synops)}")

    logger.info(
        "Scaled RH by %s for %s clear-sky profiles (fraction=%s).",
        scale_factor,
        n_select,
        fraction,
    )

    return ds

def create_radiosonde_dataset_for_RT(
    site: str,
    station_id: str,
    data_dir: str,
    heights: np.ndarray,
    update_only: bool = False,
    first_year: Optional[int] = None,
    cut_off_at_mwr_height: bool = False,
    scale_rh_clear_sky: bool = False,
    scale_rh_clear_sky_seed: Optional[int] = None,
    file_extension: str = ''
) -> None:
    """
    Creates a processed radiosonde dataset for the radiative transfer (RT) calculation for a given site and station.
    This function loads raw radiosonde data files for the specified station, processes and filters the profiles,
    interpolates them to fixed heights, fills missing values with monthly means, adds derived variables, and saves
    the resulting dataset to a NetCDF file.

    Parameters
    ----------
    site: str
        The site identifier used for configuration and output path.
    station_id: str
        The identifier of the radiosonde station.
    data_dir: str
        Base directory containing all openMWR-managed data.
    heights: np.array
        Array of heights (in meters) to which the profiles will be interpolated.
    update_only: bool, optional
        If True, only new profiles not already in the existing dataset will be added.
    first_year: int, optional
        If specified, only data from this year onwards will be processed.
    cut_off_at_mwr_height: bool, optional
        If True, profiles will be cut off at the MWR height specified in the site configuration.
        If False, profiles will be cut off at the radiosonde station height.
    scale_rh_clear_sky: bool, optional
        If True, scale RH by 0.5 for a random 10% of clear-sky profiles (lwc == 0).
    scale_rh_clear_sky_seed: int, optional
        Seed for reproducible random selection of clear-sky profiles.

    Returns
    -------
        None

    Notes
    -----
    - The function expects raw radiosonde NetCDF files to be located under
        ``{data_dir}/radiosondes/station_{station_id}/raw_data_{year}.nc``.
            -> Can be created using :func:`openMWR.radiosonde.download_all_radiosondes`
    - The processed dataset is saved to
        ``{data_dir}/sites/{site}/radiosonde/radiosonde_data_{station_id}.nc``.
    - Profiles are filtered based on minimum and maximum height, number of measurements, and presence of NaN values.
    - Derived variables such as absolute humidity, dew point, LWP, and IWV are added.
    - Surface variables (temperature, relative humidity, pressure at ground level) are extracted.
    """
    radiosonde_dir = site_subdir(data_dir, site, "radiosonde")

    if file_extension != '':
        target_file = radiosonde_dir / f'radiosonde_data_{station_id}_{file_extension}.nc'
    else:
        target_file = radiosonde_dir / f'radiosonde_data_{station_id}.nc'

    if update_only and not os.path.exists(target_file):
        logger.warning(f"File {target_file} does not exist. Cannot update. Creating new dataset instead.")
        update_only = False

    if update_only:
        logger.info(f"Updating radiosonde dataset for site {site} and station {station_id}")
    else:
        logger.info(f"Creating radiosonde dataset for site {site} and station {station_id}")

    if update_only:
        ds_old = xr.load_dataset(target_file)

    #Find all files for the station
    station_dir = radiosonde_station_dir(data_dir, station_id)
    raw_data_files = sorted(glob.glob(str(station_dir / 'raw_data_*.nc')))
    if not raw_data_files:
        logger.error(f"No raw radiosonde data files found for station {station_id}.")
        return
    
    if first_year is not None:
        raw_data_files = [f for f in raw_data_files if int(os.path.basename(f).split('_')[-1].split('.')[0]) >= first_year]
    
    mwr_height = get_config_parameter(site, 'mwr_height', data_dir)

    ds_list = []

    for raw_data_file in raw_data_files:
        logger.info(f"Processing file: {raw_data_file}")

        # Load the dataset
        ds_year = xr.load_dataset(raw_data_file)

        if update_only:
            # Filter out already existing synops
            new_synops = np.setdiff1d(ds_year['synop'].values, ds_old['synop'].values)
            if new_synops.size == 0:
                logger.info("No new synops found in this file. Skipping.")
                continue
            else:
                ds_year = ds_year.sel(synop=new_synops)
                logger.info(f"Found {ds_year.sizes['synop']} possible new synops to add.")

        
        station_height = int(ds_year.attrs['station_height'])
        if cut_off_at_mwr_height:
            if station_height <= mwr_height:
                cut_off_height = mwr_height
            else:
                logger.warning(f"Station height {station_height} m is higher than MWR height {mwr_height} m. Using station height as zero height.")
                cut_off_height = station_height
        else:
            cut_off_height = station_height

        #Drop dewpoint, because it is recalculated later
        if 'TD' in ds_year.data_vars:
            ds_year = ds_year.drop_vars('TD')

        for i in range(ds_year.sizes['synop']):
            ds_date = ds_year.isel(synop=i)

            # Filter height min and max
            min_height = ds_date['height'].min().item()
            max_height = ds_date['height'].max().item()

            if min_height < station_height or min_height > station_height + 15:
                continue
            if max_height < 15000:
                continue

            # Filter out unrealistic height values, when meastime is less than 100 seconds
            vars_to_filter = [var for var in ds_date.data_vars if 'meastime' in ds_date[var].coords]
            ds_date[vars_to_filter] = ds_date[vars_to_filter].where((ds_date.meastime > 100) | (ds_date.height < 20000))

            # Filter naN values
            ds_date = ds_date.dropna(dim="meastime", subset=['height', 'p', 'T', 'rh']) 

            # Swap dimension to height and sort by height
            ds_date = ds_date.swap_dims({'meastime': 'height'}).sortby('height')

            # Filter out values with same height
            _, index = np.unique(ds_date.height.values, return_index=True)
            ds_date = ds_date.isel(height=index)

            if ds_date.sizes['height'] < 500:
                continue

            # Add adiabatic lwc
            try:
                ds_date = add_cloud_to_ds(ds_date)
            except Exception as e:
                logger.error(f"Fehler bei Cloud: {ds_date.synop.values} {e}")
                logger.debug(e, exc_info=True)
                continue

            # Change height coordinate to height above ground
            ds_date['height'] = ds_date['height'] - cut_off_height

            # Interpolate to fixed heights
            ds_date = change_coord_of_ds(ds_date, heights, extrapolate_lower_end=True, extrapolate_upper_end=False)

            ds_date = ds_date.expand_dims(synop=[ds_date.synop.values])
            ds_list.append(ds_date)

        logger.info(f"Finished processing file: {raw_data_file}")

    if not ds_list:
        logger.warning("No valid radiosonde profiles found after processing all files.")
        return

    ds_new = xr.concat(ds_list, dim='synop').sortby('synop')

    # If updating, concatenate with old data and sort
    if update_only:
        ds_new = xr.concat([ds_old, ds_new], dim='synop').sortby('synop')

    if update_only:
        logger.info(f"Updated radiosonde dataset with {len(ds_list)} total profiles")
    else:
        logger.info("Created radiosonde dataset")
        logger.info(f"Number of radiosonde profiles: {ds_new.sizes['synop']}")

    lwc_reference = None
    if scale_rh_clear_sky:
        if 'lwc' in ds_new.data_vars:
            lwc_reference = ds_new['lwc'].max(dim='height', skipna=True)
        else:
            logger.warning("Requested RH scaling for clear-sky cases, but 'lwc' is missing.")
        
    # Fill nan values with mean of month (only the above the last measured height)
    vars_to_fill = [var for var in ds_new.data_vars if 'height' in ds_new[var].coords]
    ds_new[vars_to_fill] = ds_new[vars_to_fill].groupby('synop.month').apply(lambda x: x.fillna(x.mean(dim='synop')))

    logger.info("Filled NaN values with monthly mean")

    #Warning if not all values are filled
    for var in vars_to_fill:
        n_nan = ds_new[var].isnull().sum().item()
        if n_nan > 0:
            logger.warning(f"Variable {var} still has {n_nan} NaN values after filling with monthly mean.")

    if scale_rh_clear_sky:
        ds_new = _scale_rh_profiles_for_clear_sky_cases(
            ds_new,
            fraction=0.1,
            scale_factor=0.2,
            random_seed=scale_rh_clear_sky_seed,
            lwc_reference=lwc_reference,
        )

    # Replace too-small or negative RH values with 0.1% (otherwise causes problems with TD calculations)
    ds_new['rh'] = ds_new['rh'].where(ds_new['rh'] > 0.1, 0.1)

    # Add derived data
    ds_new = add_derived_data(ds_new)

    logger.info("Added derived data (absolute humidity, dew point, LWP, IWV)")

    # Extract surface variables
    ds_new['surface_T'] = ds_new['T'].sel(height=0)
    ds_new['surface_rh'] = ds_new['rh'].sel(height=0)
    ds_new['surface_p'] = ds_new['p'].sel(height=0)

    logger.info("Added surface variables to dataset")

    # Add time data
    ds_new = add_time_data(ds_new, time_dim='synop')
    logger.info("Added time-related features to dataset like 'doy_cos', 'doy_sin'")

    # Save the dataset
    ds_new.to_netcdf(target_file)
    logger.info(f"Radiosonde dataset saved to {target_file}")

def create_radiosonde_dataset_with_RT(station_id: str, site: str, data_dir: str, freqs: np.ndarray, angles: np.ndarray = np.array([90.]), save: bool = True, num_of_processes: int = 12,
           freq_shift: np.ndarray = None, update_only: bool = False, test: bool = False, absmdl: str = 'R20', file_extension_in: str = '', 
           file_extension_out: str = '', RT_model: str = 'torchMWRT'):
    """
    Run RT for a station's processed radiosonde profiles.

    Parameters
    ----------
    station_id : str
        Radiosonde station identifier used in file names.
    site : str
        Site name used to resolve input and output paths.
    data_dir : str
        Base directory containing all openMWR-managed data.
    freqs : numpy.ndarray
        Microwave channel frequencies.
    angles : numpy.ndarray, optional
        Elevation angles in degrees.
    save : bool, optional
        If True, write the output dataset to disk.
    num_of_processes : int, optional
        Number of worker processes for RT and IR calculations.
    freq_shift : numpy.ndarray, optional
        Per-channel frequency offsets subtracted before RT.
    update_only : bool, optional
        If True, process only times not yet present in the output file.
    test : bool, optional
        If True, limit processing to the first ``num_of_processes`` profiles.
    absmdl : str, optional
        Absorption model passed to the selected RT backend.
    file_extension_in : str, optional
        Optional suffix of the input radiosonde file.
    file_extension_out : str, optional
        Optional suffix of the output dataset file.
    RT_model : {"torchMWRT", "pyrtlib"}, optional
        RT backend used in :func:`openMWR.run_RT.run_RT`.

    Returns
    -------
    xarray.Dataset or None
        RT dataset if ``save`` is False, otherwise ``None``.
    """
    logger.info(f'calc RT for station {station_id}:')

    radiosonde_dir = site_subdir(data_dir, site, "radiosonde")

    if file_extension_out != '':
        file = radiosonde_dir / f"radiosonde_data_with_RT_{station_id}_{file_extension_out}.nc"
    else:
        file = radiosonde_dir / f"radiosonde_data_with_RT_{station_id}.nc"

    if update_only and not os.path.exists(file):
        logger.warning(f"File {file} does not exist. Cannot update. Creating new dataset instead.")
        update_only = False

    if file_extension_in != '':
        RS_file = radiosonde_dir / f'radiosonde_data_{station_id}_{file_extension_in}.nc'
    else:
        RS_file = radiosonde_dir / f'radiosonde_data_{station_id}.nc'

    ds_RS = xr.load_dataset(RS_file)

    ds_RS = ds_RS.rename(synop = 'time')

    if update_only:
        ds_old = xr.load_dataset(file)
        ds_RS = ds_RS.sel(time = np.setdiff1d(ds_RS.time.values, ds_old.time.values))
        if ds_RS.time.size == 0:
            logger.info("No new radiosonde data found. Skipping RT calculation.")
            return
        else:
            logger.info(f"Found {ds_RS.time.size} new radiosonde profiles. Calculating RT for these.")

    if test:
        ds_RS = ds_RS.isel(time=slice(None, num_of_processes))

    ds_new = run_RT(
        ds_RS,
        site,
        data_dir,
        freqs,
        angles,
        num_of_processes=num_of_processes,
        freq_shift=freq_shift,
        absmdl=absmdl,
        RT_model=RT_model,
    )

    ds_new = ds_new.expand_dims(station=[station_id])

    if update_only:
        ds_new = xr.concat([ds_old, ds_new], dim='time').sortby('time')

    if save:
        ds_new.to_netcdf(file)

        logger.info(f'Saved radiosonde data with RT to {file}')

        logger.info(ds_new)

    else:
        return ds_new


def create_analysis_dataset(
    site: str,
    station_ids: Sequence[str],
    data_dir: str,
    from_forward_calc: bool = True,
    save: bool = True,
    file_extension: str = ''
) -> Optional[tuple[xr.Dataset, xr.Dataset]]:
    """
    Create analysis datasets pairing Hatpro measurements with radiosonde data.

    Parameters
    ----------
    site : str
        Site name used to locate Hatpro and radiosonde files.
    station_ids : Sequence[str]
        Radiosonde station IDs to include (strings preserve leading zeros).
    data_dir : str
        Base directory containing all openMWR-managed data.
    from_forward_calc : bool, optional
        If True, use radiosonde datasets with forward-calculated brightness
        temperatures (`radiosonde_data_with_RT_*`). If False, use the prepared
        radiosonde datasets without RT.
    save : bool, optional
        If True, write outputs to `data/sites/{site}/analysis/` and return
        None. If False, return the assembled datasets.
    file_extension : str, optional
        Additional string appended to radiosonde filenames before the `.nc`
        extension (e.g., to indicate a specific processing version).

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset] or None
        `(analysis, analysis_bls)` when `save=False`, otherwise None.

    Raises
    ------
    FileNotFoundError
        If required Hatpro or radiosonde files are missing.

    Notes
    -----
    - Requires Hatpro datasets from `create_hatpro_dataset`.
    - Matches times by nearest neighbor within 10 minutes and filters out
      rainy scenes using `rain_flag`.
    - Hatpro heights are interpolated to the radiosonde height grid for
      consistent comparisons.
    - Adds a `data_source` dimension with entries `radiosonde` and `hatpro`.
    """

    logger.info(f'Creating analysis dataset for site {site} from radiosonde and hatpro data')

    # Hatpro data
    hatpro_dir = site_subdir(data_dir, site, "hatpro")
    analysis_dir = site_subdir(data_dir, site, "analysis")
    radiosonde_dir = site_subdir(data_dir, site, "radiosonde")

    ds_hatpro = xr.load_dataset(hatpro_dir / 'hatpro_data.nc').dropna(dim='time')
    ds_hatpro_bls = xr.load_dataset(hatpro_dir / 'hatpro_data_bls.nc').dropna(dim='time')

    an_data_raso_list = []
    an_data_hatpro_list = []
    an_data_raso_bls_list = []
    an_data_hatpro_bls_list = []

    n_raso = 0
    n_raso_bls = 0

    if file_extension != '':
        file_extension = '_' + file_extension

    # Radiosonde data
    for station_id in station_ids:

        if from_forward_calc:
            file = radiosonde_dir / f'radiosonde_data_with_RT_{station_id}{file_extension}.nc'
        else:
            file = radiosonde_dir / f'radiosonde_data_{station_id}{file_extension}.nc'

        if not os.path.exists(file):
            raise FileNotFoundError(f"Radiosonde data file {file} does not exist. Please create it first.")
        
        raso_data = xr.load_dataset(file)

        # Keep heights below 10 km
        raso_data = raso_data.sel(height=slice(0, 10001))

        if from_forward_calc:
            raso_data = raso_data.sel(station=station_id, drop=True)
            raso_data = raso_data.rename(time = 'synop')

        raso_data = raso_data.transpose('synop', 'height', ...)

        raso_data = raso_data.swap_dims({'synop': 'starttime'})
        raso_data = raso_data.rename({'starttime': 'time'})   

        # Select matching times
        ds_hatpro_sel, raso_data_sel = select_closest_times(ds_hatpro, raso_data)
        ds_hatpro_bls_sel, raso_data_bls_sel = select_closest_times(ds_hatpro_bls, raso_data)

        if from_forward_calc:
            raso_data_sel = raso_data_sel.sel(ang=90.0, drop=True)

        # Keep only non-rainy data
        raso_data_sel = raso_data_sel.where(ds_hatpro_sel['rain_flag'] == 0.0, drop=True)
        ds_hatpro_sel = ds_hatpro_sel.where(ds_hatpro_sel['rain_flag'] == 0.0, drop=True)

        raso_data_bls_sel = raso_data_bls_sel.where(ds_hatpro_bls_sel['rain_flag'] == 0.0, drop=True)
        ds_hatpro_bls_sel = ds_hatpro_bls_sel.where(ds_hatpro_bls_sel['rain_flag'] == 0.0, drop=True)

        an_data_raso_list.append(raso_data_sel.expand_dims({'station': [raso_data.attrs['station_id']]}))
        an_data_hatpro_list.append(ds_hatpro_sel.expand_dims({'station': [raso_data.attrs['station_id']]}))
        an_data_raso_bls_list.append(raso_data_bls_sel.expand_dims({'station': [raso_data.attrs['station_id']]}))
        an_data_hatpro_bls_list.append(ds_hatpro_bls_sel.expand_dims({'station': [raso_data.attrs['station_id']]}))

        n_raso += raso_data_sel.sizes['time']
        n_raso_bls += raso_data_bls_sel.sizes['time']

    # Concatenate data
    an_data_raso = xr.concat(an_data_raso_list, dim='station', join='outer')
    an_data_hatpro = xr.concat(an_data_hatpro_list, dim='station', join='outer')
    an_data_raso_bls = xr.concat(an_data_raso_bls_list, dim='station', join='outer')
    an_data_hatpro_bls = xr.concat(an_data_hatpro_bls_list, dim='station', join='outer')

    # Adjust height coordinates
    if 'height' in an_data_hatpro.dims:
        an_data_hatpro = change_height_coord_of_ds_hatpro(an_data_hatpro, an_data_raso.height.values) #heights[heights <= 10000]
    if 'height' in an_data_hatpro_bls.dims:
        an_data_hatpro_bls = change_coord_of_ds(an_data_hatpro_bls, an_data_raso_bls.height.values)

    # Log number of radiosonde data points
    logging.info(f'Number of Radiosonde data points: {n_raso}')
    logging.info(f'Number of Radiosonde data points for BLS: {n_raso_bls}')

    an_data_raso = an_data_raso.expand_dims({'data_source': ['radiosonde']})
    an_data_hatpro = an_data_hatpro.expand_dims({'data_source': ['hatpro']})
    an_data_raso_bls = an_data_raso_bls.expand_dims({'data_source': ['radiosonde']})
    an_data_hatpro_bls = an_data_hatpro_bls.expand_dims({'data_source': ['hatpro']})

    an_data = xr.concat([an_data_raso, an_data_hatpro], dim='data_source')
    an_data_bls = xr.concat([an_data_raso_bls, an_data_hatpro_bls], dim='data_source')

    # Save
    if save:
        file = analysis_dir / f'analysis_data_radiosonde{file_extension}.nc'
        an_data.to_netcdf(file)
        logging.info(f'{file} saved')

        file = analysis_dir / f'analysis_data_radiosonde_bls{file_extension}.nc'
        an_data_bls.to_netcdf(file)
        logging.info(f'{file} saved')
    
    else:
        return an_data, an_data_bls
