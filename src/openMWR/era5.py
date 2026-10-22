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
from openMWR.paths import dataset_file, era5_cache_file

# deactivate cdsapi-Logger
# cds_logger = logging.getLogger("cdsapi")
# for h in cds_logger.handlers[:]:
#     cds_logger.removeHandler(h)
# cds_logger.propagate = True


def _change_height_coord_for_forward_calc(ds_date, new_heights):
    ds_date = ds_date.swap_dims({'p': 'height'})
    ds_date = ds_date.reset_coords("p")
    ds_date = change_coord_of_ds(
        ds_date,
        new_heights,
        extrapolate_lower_end=True,
        extrapolate_upper_end=False,
    )
    return ds_date


class ERA5Processor:
    """Shared ERA5 acquisition, named profile preparation, RT and matching.

    All dates are inclusive calendar dates. Raw caches are shared by grid point
    and sampling interval; each prepared-profile or simulation dataset has its
    own site-local name. Methods can be called independently from Python.
    """

    def __init__(self, site, project_dir, start_date=None, end_date=None,
                 update_only=True, hour_interval=6, profiles_dataset='profiles_era5',
                 simulation_dataset='simulation_era5'):
        self.site, self.project_dir = site, project_dir
        self.update_only, self.hour_interval = update_only, hour_interval
        self.logger = logging.getLogger(__name__)
        self.lat, self.lon, site_start = get_config_parameter(
            site, ['latitude', 'longitude', 'mwr_measurement_start_date'], project_dir)
        self.start_date = pd.Timestamp(start_date or site_start)
        self.end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize() - pd.Timedelta(days=5)
        self.lat_round, self.lon_round = round(self.lat * 4) / 4, round(self.lon * 4) / 4
        self.raw_data_file = era5_cache_file(project_dir, self.lat, self.lon, hour_interval)
        self.profile_file = dataset_file(project_dir, site, profiles_dataset, 'era5')
        self.forward_calc_file = dataset_file(project_dir, site, simulation_dataset, 'era5')

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

    def create_original_dataset(self):
        """Append days after the latest cached day, leaving historical gaps alone.

        A source rebuild refreshes its requested coverage and preserves cached
        dates outside that coverage for other recipes sharing the same cache.
        """
        self.raw_data_file.parent.mkdir(parents=True, exist_ok=True)
        old = xr.load_dataset(self.raw_data_file) if self.raw_data_file.is_file() else None
        start_date = self.start_date
        if old is not None and self.update_only and old.time.size:
            last_day = pd.Timestamp(old.time.max().values).normalize()
            start_date = max(start_date, last_day + pd.Timedelta(days=1))
        if start_date > self.end_date:
            self.logger.info('ERA5 source is up to date.')
            return old
        new = self.download_era5(start_date, self.end_date, h_freq=self.hour_interval)
        if old is not None:
            if not self.update_only:
                # An explicit rebuild refreshes only its requested coverage.
                old = old.sel(time=(old.time < start_date) | (old.time >= self.end_date + pd.Timedelta(days=1)))
            new = xr.concat([old, new], dim='time').sortby('time').drop_duplicates('time')
        new.to_netcdf(self.raw_data_file)
        return new

    def create_profile_dataset(self, new_heights, num_of_processes=1):
        """Prepare only missing ERA5 profiles, independently of RT settings."""
        if not self.raw_data_file.is_file():
            raise FileNotFoundError(f'ERA5 source missing: {self.raw_data_file}')
        raw = xr.load_dataset(self.raw_data_file).sel(
            time=slice(self.start_date, self.end_date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)))
        old = xr.load_dataset(self.profile_file) if self.update_only and self.profile_file.is_file() else None
        if old is not None:
            raw = raw.sel(time=~raw.time.isin(old.time))
        if raw.time.size == 0:
            if old is not None:
                return old
            raise ValueError('No ERA5 profiles in the selected source coverage.')
        raw['height'] = raw['height'] - get_config_parameter(self.site, 'mwr_height', self.project_dir)
        new = run_pool(raw, num_of_processes, _change_height_coord_for_forward_calc, new_heights)
        new['rh'] = new['rh'].where(new['rh'] > 0.1, 0.1)
        new = add_derived_data(new)
        for var in ('T', 'rh', 'p'):
            new[f'surface_{var}'] = new[var].sel(height=0)
        new = add_time_data(new)
        if old is not None:
            new = xr.concat([old, new], dim='time').sortby('time')
        self.profile_file.parent.mkdir(parents=True, exist_ok=True)
        new.to_netcdf(self.profile_file)
        return new

    def create_forward_calc_dataset(self, freqs, angles, num_of_processes=10,
                                    freq_shift=None, absmdl='R17', RT_model='torchMWRT'):
        """Run RT only for prepared profiles missing from this simulation."""
        if not self.profile_file.is_file():
            raise FileNotFoundError(f'ERA5 prepared profiles missing: {self.profile_file}')
        profiles = xr.load_dataset(self.profile_file)
        old = xr.load_dataset(self.forward_calc_file) if self.update_only and self.forward_calc_file.is_file() else None
        if old is not None:
            profiles = profiles.sel(time=~profiles.time.isin(old.time))
        if profiles.time.size == 0:
            if old is not None:
                return old
            raise ValueError('No prepared ERA5 profiles available for RT.')
        new = run_RT(profiles, self.site, self.project_dir, freqs, angles,
                     num_of_processes=num_of_processes, freq_shift=freq_shift,
                     absmdl=absmdl, RT_model=RT_model)
        if old is not None:
            new = xr.concat([old, new], dim='time').sortby('time')
        self.forward_calc_file.parent.mkdir(parents=True, exist_ok=True)
        new.to_netcdf(self.forward_calc_file)
        return new

    def create_analysis_dataset_from_forward_calc(self, save=True, observations_dataset='hatpro', dataset='analysis_era5'):
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
        ds_hatpro = xr.load_dataset(dataset_file(self.project_dir, self.site, observations_dataset)).dropna(dim='time')
        ds_hatpro_bls = xr.load_dataset(dataset_file(self.project_dir, self.site, observations_dataset, bls=True)).dropna(dim='time')

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
                file = dataset_file(self.project_dir, self.site, dataset)
                file.parent.mkdir(parents=True, exist_ok=True)
                an_data.to_netcdf(file)
                self.logger.info(f'{file} saved')
            else:
                self.logger.info('an_data is empty, not saved')

            if an_data_bls.time.size > 0:
                file = dataset_file(self.project_dir, self.site, dataset, bls=True)
                file.parent.mkdir(parents=True, exist_ok=True)
                an_data_bls.to_netcdf(file)
                self.logger.info(f'{file} saved')
            else:
                self.logger.info('an_data_bls is empty, not saved')

        else:
            return an_data, an_data_bls
