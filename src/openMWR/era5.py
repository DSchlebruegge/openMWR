import os
import logging
import xarray as xr
import pandas as pd
import cdsapi
import tempfile

from openMWR.site import get_config_parameter
from openMWR.run_RT import run_RT
from openMWR.xr_utils import change_coord_of_ds, add_time_data, select_closest_times
from openMWR.parallel import run_pool
from openMWR.hatpro_data import change_height_coord_of_ds_hatpro
from openMWR.radiosonde import add_derived_data
from openMWR.paths import data_root, site_subdir

# deactivate cdsapi-Logger
# cds_logger = logging.getLogger("cdsapi")
# for h in cds_logger.handlers[:]:
#     cds_logger.removeHandler(h)
# cds_logger.propagate = True


class ERA5Processor:
    """
    Download, process, and prepare ERA5 data for radiative transfer calculations
    and retrieval analysis in the openMWR framework.

    This class retrieves ERA5 pressure-level data for a given site, performs
    interpolation and radiative transfer forward calculations for microwave
    radiometer frequencies and viewing angles, and prepares matched analysis
    datasets combining HATPRO measurements with ERA5 atmospheric profiles.

    Parameters
    ----------
    site : str
        Name of the site for which ERA5 data should be processed. Must exist in
        the openMWR site configuration.
    data_dir : str
        Base directory containing all openMWR-managed data.
    start_date : str or None, optional
        Start date of the ERA5 data range (`'YYYY-MM-DD'`).  
        If None, uses the site's configured ``mwr_measurement_start_date``.
    end_date : str or None, optional
        End date of the ERA5 data range (`'YYYY-MM-DD'`).  
        If None, uses current day minus 5 days (recommended to avoid incomplete ERA5 data).
    update_only : bool, optional
        If True (default), only missing dates are downloaded or processed.
    file_extension : str, optional
        Additional string to append to the filenames before the file extension,
        e.g. to distinguish different processing versions.

    Notes
    -----
    ERA5 data is downloaded from the Copernicus Data Store using the `cdsapi`
    client. The processed datasets are stored under:

    * ``data/era5/`` – raw ERA5 pressure-level data  
    * ``data/sites/{site}/era5/`` – forward-calculated brightness temperatures  
    * ``data/sites/{site}/analysis/`` – matched analysis datasets with HATPRO

    Examples
    --------
    Create an ERA5 processor and generate raw and forward-calculated datasets:

    .. code-block:: python

        from openMWR.era5 import ERA5Processor
        from openMWR.consts import std_freqs, std_heights, std_angles
        era5 = ERA5Processor("munich_G5", data_dir="data")
        era5.create_original_dataset(h_freq=2)
        era5.create_forward_calc_dataset(std_freqs, std_heights, std_angles, num_of_processes=12)

    Create an analysis dataset matched to HATPRO measurements:

    .. code-block:: python

        era5.create_analysis_dataset_from_forward_calc()

    Run the entire processing chain:

    .. code-block:: python

        era5.create_analysis_dataset_from_scratch(num_of_processes=16)
    """
    def __init__(self, site: str, data_dir: str, start_date: str = None, end_date: str = None, update_only: bool = True, file_extension: str = '') -> None:
        self.site = site
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.update_only = update_only
        self.logger = logging.getLogger(__name__)

        # Load site parameters
        self.lat, self.lon, start_date_str = get_config_parameter(
            site, ["latitude", "longitude", "mwr_measurement_start_date"], data_dir
        )

        if start_date is None:
            self.start_date = pd.Timestamp(start_date_str)
        else:
            self.start_date = pd.Timestamp(start_date)

        if end_date is None:
            self.end_date = pd.Timestamp.now().floor("D") - pd.Timedelta(days=5)
        else:
            self.end_date = pd.Timestamp(end_date)

        # ERA5 Raster rounding
        self.lat_round = round(self.lat * 4) / 4
        self.lon_round = round(self.lon * 4) / 4

        # Prepare paths
        if file_extension != '':
            file_extension = '_' + file_extension

        self.raw_data_file = data_root(data_dir) / f"era5/era5_{self.lat_round}_{self.lon_round}.nc"
        self.forward_calc_file = site_subdir(data_dir, site, "era5") / f"forward_calc_era5{file_extension}.nc"
        self.analysis_file = site_subdir(data_dir, site, "analysis") / f"analysis_data_era5{file_extension}.nc"

    def download_era5(self, start_time, end_time, h_freq=6):
        """
        Download ERA5 pressure-level profiles for a given time range.

        Parameters
        ----------
        start_time : str or pandas.Timestamp
            Start timestamp of the download interval.
        end_time : str or pandas.Timestamp
            End timestamp of the download interval.
        h_freq : int, optional
            Temporal sampling frequency in hours (default: 6).

        Returns
        -------
        xarray.Dataset
            ERA5 dataset interpolated to the site coordinates, sorted by time,
            and containing variables:
            ``T``, ``rh``, ``lwc``, ``p``, ``height``.

        Notes
        -----
        Downloads one NetCDF file per day, concatenates them, and converts
        geopotential heights to meters.
        """

        dr = pd.date_range(start=start_time, end=end_time, freq="1D")
        ds_list = []

        client = cdsapi.Client()

        for date in dr:
            self.logger.info(f"Downloading ERA5 data for {date.date()} at lat={self.lat_round}, lon={self.lon_round}")

            request = {
                "product_type": ["reanalysis"],
                "variable": [
                    "relative_humidity",
                    "specific_cloud_liquid_water_content",
                    "temperature",
                    "geopotential",
                ],
                "year": [str(date.year)],
                "month": [f"{date.month:02d}"],
                "day": [f"{date.day:02d}"],
                "time": [f"{h:02d}:00" for h in range(0, 24, h_freq)],
                "pressure_level": [str(p) for p in [
                    1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,
                    225,250,300,350,400,450,500,550,600,650,700,
                    750,775,800,825,850,875,900,925,950,975,1000
                ]],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [self.lat_round, self.lon_round, self.lat_round, self.lon_round],
            }

            tmpfile = tempfile.mktemp(suffix=".nc")
            client.retrieve("reanalysis-era5-pressure-levels", request).download(tmpfile)

            ds_date = xr.load_dataset(tmpfile)
            os.remove(tmpfile)

            ds_date = ds_date.rename({"valid_time": "time"})
            ds_list.append(ds_date)

        ds = xr.concat(ds_list, dim="time").sortby("time")
        ds["height"] = ds["z"] / 9.81
        ds = ds.drop_vars(["z", "expver", "number"])
        ds = ds.rename({"r": "rh", "clwc": "lwc", "t": "T", 'pressure_level': 'p'})
        ds.attrs = {"latitude": self.lat_round, "longitude": self.lon_round}

        return ds.isel(latitude=0, longitude=0, drop=True)

    def create_original_dataset(self, h_freq=6):
        """
        Create or update the locally stored raw ERA5 dataset.

        Parameters
        ----------
        h_freq : int, optional
            Temporal sampling frequency in hours for ERA5 download.

        Returns
        -------
        xarray.Dataset
            The full raw ERA5 dataset covering the configured date range.

        Notes
        -----
        If the file already exists and ``update_only=True``, only missing dates
        are downloaded.
        """

        os.makedirs(os.path.dirname(self.raw_data_file), exist_ok=True)

        if os.path.exists(self.raw_data_file) and self.update_only:
            ds_old = xr.load_dataset(self.raw_data_file)
            old_end_date = pd.Timestamp(ds_old.time.values[-1]).floor("D")

            if old_end_date >= self.end_date:
                self.logger.info("ERA5 file is up to date.")
                return ds_old

            start_date = old_end_date + pd.Timedelta(days=1)
            ds_new = self.download_era5(start_date, self.end_date, h_freq=h_freq)
            ds = xr.concat([ds_old, ds_new], dim="time").sortby("time")

        else:
            ds = self.download_era5(self.start_date, self.end_date, h_freq=h_freq)

        ds.to_netcdf(self.raw_data_file)
        return ds

    def create_forward_calc_dataset(self, freqs, new_heights, angles, num_of_processes=10, freq_shift=None, absmdl='R17', RT_model: str = 'torchMWRT') -> xr.Dataset:
        """
        Create or update the ERA5 forward-calculation dataset.

        Parameters
        ----------
        freqs : array_like
            Frequencies (GHz) for radiative transfer calculation.
        new_heights : array_like
            Target height grid for interpolation (m).
        angles : array_like
            Viewing elevation angles (degrees).
        num_of_processes : int, optional
            Number of processes to use for parallel RT computation.
        freq_shift : array_like or None, optional
            Per-channel frequency offsets subtracted before RT calculation.
        absmdl : str, optional
            Absorption model used by the RT backend.
        RT_model : {"torchMWRT", "pyrtlib"}, optional
            Radiative transfer backend used in :func:`openMWR.run_RT.run_RT`.

        Returns
        -------
        xarray.Dataset
            Dataset including interpolated ERA5 profiles, derived variables,
            time metadata, and brightness temperatures.

        Notes
        -----
        If ``self.update_only`` is enabled and an output file exists, only
        missing timestamps are processed and merged with the existing file.
        """

        self.logger.info(f'Creating forward calculation dataset for site {self.site} from era5 data')

        # Load original data
        if not os.path.exists(self.raw_data_file):
            raise FileNotFoundError("ERA5 base dataset missing. Run create_original_dataset first.")

        ds_raw = xr.load_dataset(self.raw_data_file)

        # Select time range
        ds_raw = ds_raw.sel(time=slice(self.start_date, self.end_date))

        os.makedirs(os.path.dirname(self.analysis_file), exist_ok=True)

        update = True if os.path.exists(self.forward_calc_file) and self.update_only else False

        if update:
            # Update only the missing times
            ds_old = xr.load_dataset(self.forward_calc_file)
            times_to_add = pd.Index(ds_raw.time.values).difference(ds_old.time.values)

            if len(times_to_add) == 0:
                self.logger.info("Forward calculation dataset is up to date.")
                return ds_old

            ds_sel = ds_raw.sel(time=times_to_add)
        else:
            ds_sel = ds_raw

        # Change height to be relative to MWR
        mwr_height = get_config_parameter(self.site, "mwr_height", self.data_dir)
        ds_sel["height"] = ds_sel["height"] - mwr_height

        def _change_height_coord(ds_date):
            ds_date = ds_date.swap_dims({'p': 'height'})
            ds_date = ds_date.reset_coords("p")
            ds_date = change_coord_of_ds(ds_date, new_heights, extrapolate_lower_end=True, extrapolate_upper_end=False)
            return ds_date

        # Process RT in parallel
        ds_new = run_pool(ds_sel, num_of_processes, _change_height_coord)

        # Replace too-small or negative RH values with 0.1% (otherwise causes problems with TD calculations)
        ds_new['rh'] = ds_new['rh'].where(ds_new['rh'] > 0.1, 0.1)
        
        ds_new = run_RT(
            ds_new,
            site=self.site,
            data_dir=self.data_dir,
            freqs=freqs,
            angles=angles,
            absmdl=absmdl,
            num_of_processes=num_of_processes,
            freq_shift=freq_shift,
            RT_model=RT_model,
        )

        # Add derived data
        ds_new = add_derived_data(ds_new)

        # Extract surface variables
        ds_new['surface_T'] = ds_new['T'].sel(height=0)
        ds_new['surface_rh'] = ds_new['rh'].sel(height=0)
        ds_new['surface_p'] = ds_new['p'].sel(height=0)

        ds_new = add_time_data(ds_new)

        if update:
            ds_new = xr.concat([ds_old, ds_new], dim="time").sortby("time")

        ds_new.to_netcdf(self.forward_calc_file)

        self.logger.info(f'Forward calculation dataset created with {ds_new.time.size} times and saved to {self.forward_calc_file}')

        return ds_new
    
    def create_analysis_dataset_from_forward_calc(self, save=True):
        """
        Create an analysis dataset by matching HATPRO measurements with
        forward-calculated ERA5 brightness temperatures and profiles.

        Parameters
        ----------
        save : bool, optional
            If True (default), saves the analysis datasets to disk.  
            If False, returns them instead of saving.

        Returns
        -------
        tuple of xarray.Dataset, optional
            Only returned if ``save=False``:
            ``(analysis_dataset, analysis_dataset_bls)`` for zenith and BLS modes.

        Notes
        -----
        - Matches ERA5 times with HATPRO measurements using nearest timestamps.
        - Removes rainy cases based on ``rain_flag`` in HATPRO data.
        - Merges ERA5 and HATPRO profiles into a joint dataset with dimension
          ``data_source = ['era5', 'hatpro']``.
        """

        self.logger.info(f'Creating analysis dataset for site {self.site} from era5 data and hatpro data')

        #Hatpro Data
        hatpro_dir = site_subdir(self.data_dir, self.site, "hatpro")
        ds_hatpro = xr.load_dataset(hatpro_dir / 'hatpro_data.nc').dropna(dim='time')
        ds_hatpro_bls = xr.load_dataset(hatpro_dir / 'hatpro_data_bls.nc').dropna(dim='time')

        # Era5 Data
        if not os.path.exists(self.forward_calc_file):
            raise FileNotFoundError("ERA5 forward calculation dataset missing. Run create_forward_calc_dataset first.")
        ds_era5 = xr.load_dataset(self.forward_calc_file).dropna(dim='time')

        # Select only heights up to 10km
        ds_era5 = ds_era5.sel(height=slice(0, 10001))

        # Select closest times
        ds_hatpro_sel, ds_era5_sel = select_closest_times(ds_hatpro, ds_era5)
        ds_hatpro_bls_sel, ds_era5_bls_sel = select_closest_times(ds_hatpro_bls, ds_era5)

        # Select only 90 degree angle
        ds_era5_sel = ds_era5_sel.sel(ang=90.0, drop=True)

        # Select only dates where rain_flag is 0
        ds_era5_sel = ds_era5_sel.where(ds_hatpro_sel['rain_flag'] == 0.0, drop=True)
        ds_hatpro_sel = ds_hatpro_sel.where(ds_hatpro_sel['rain_flag'] == 0.0, drop=True)

        ds_era5_bls_sel = ds_era5_bls_sel.where(ds_hatpro_bls_sel['rain_flag'] == 0.0, drop=True)
        ds_hatpro_bls_sel = ds_hatpro_bls_sel.where(ds_hatpro_bls_sel['rain_flag'] == 0.0, drop=True)

        # Change height coordinate
        if 'height' in ds_hatpro_sel.dims:
            ds_hatpro_sel = change_height_coord_of_ds_hatpro(ds_hatpro_sel, ds_era5_sel.height.values) #heights[heights <= 10000]
        if 'height' in ds_hatpro_bls_sel.dims:
            ds_hatpro_bls_sel = change_coord_of_ds(ds_hatpro_bls_sel, ds_era5_bls_sel.height.values)

        # Merge datasets
        ds_era5_sel = ds_era5_sel.expand_dims({'data_source': ['era5']})
        ds_hatpro_sel = ds_hatpro_sel.expand_dims({'data_source': ['hatpro']})
        ds_era5_bls_sel = ds_era5_bls_sel.expand_dims({'data_source': ['era5']})
        ds_hatpro_bls_sel = ds_hatpro_bls_sel.expand_dims({'data_source': ['hatpro']})

        an_data = xr.concat([ds_era5_sel, ds_hatpro_sel], dim='data_source')
        an_data_bls = xr.concat([ds_era5_bls_sel, ds_hatpro_bls_sel], dim='data_source')


        self.logger.info(f'Analysis dataset created with {an_data.time.size} times')
        self.logger.info(f'Analysis dataset bls created with {an_data_bls.time.size} times')

        # Save datasets
        if save:
            if an_data.time.size > 0:
                file = site_subdir(self.data_dir, self.site, "analysis") / 'analysis_data_era5.nc'
                an_data.to_netcdf(file)
                self.logger.info(f'{file} saved')
            else:
                self.logger.info('an_data is empty, not saved')

            if an_data_bls.time.size > 0:
                file = site_subdir(self.data_dir, self.site, "analysis") / 'analysis_data_era5_bls.nc'
                an_data_bls.to_netcdf(file)
                self.logger.info(f'{file} saved')
            else:
                self.logger.info('an_data_bls is empty, not saved')

        else:
            return an_data, an_data_bls

    def create_analysis_dataset_from_scratch(self, save=True, num_of_processes=10):
        """
        Run the full ERA5 processing chain: download, forward calculation,
        and creation of analysis datasets.

        Parameters
        ----------
        save : bool, optional
            If True, saves analysis datasets to disk.
        num_of_processes : int, optional
            Number of processes for forward calculation.

        Returns
        -------
        tuple of xarray.Dataset or None
            Returns the analysis datasets if ``save=False``.
        """

        self.create_original_dataset()
        self.create_forward_calc_dataset(num_of_processes=num_of_processes)
        return self.create_analysis_dataset_from_forward_calc(save=save)
