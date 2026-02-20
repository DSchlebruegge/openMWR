import requests
import zipfile
import io
from io import StringIO
import pandas as pd
from datetime import datetime
import numpy as np
import xarray as xr
import logging
import os
from openMWR.paths import radiosonde_station_dir, radiosonde_year_file

logger = logging.getLogger(__name__)

def _filter_data(df):
    """
    filter data from data file

    Parameters
    ----------
    df : pandas.Dataframe
        data from dwd radiosonde as dataframe
   
    Returns
    -------
    pandas.Dataframe
        Dataframe with filtered data
    """
    # Filter for outliers
    df["AE_DD"] = df.AE_DD.where((df.AE_DD <= 360.0) & (df.AE_DD >= 0.0))
    df["AE_GB_POS"] = df.AE_GB_POS.where((df.AE_GB_POS <= 60.0) & (df.AE_GB_POS >= 40))  # must be in Germany
    df["AE_GL_POS"] = df.AE_GL_POS.where((df.AE_GL_POS <= 20.0) & (df.AE_GL_POS >= 0))  # must be in Germany
    df["AE_P"] = df.AE_P.where((df.AE_P <= 1200.0) & (df.AE_P >= 0))
    df["AE_RF"] = df.AE_RF.where((df.AE_RF <= 120.0) & (df.AE_RF >= 0))
    df["AE_TT"] = df.AE_TT.where((df.AE_TT <= 100.0) & (df.AE_TT >= -150.0))
    df["AE_TD"] = df.AE_TD.where((df.AE_TD <= 60) & (df.AE_TD >= -150.0))
    df["AE_GPM"] = df.AE_GPM.where((df.AE_GPM <= 50000.0) & (df.AE_GPM >= -100.0))
    return df

def _download_file(url: str) -> pd.DataFrame:
    """Download a file from the given URL and return its content as a pandas DataFrame."""
    response = requests.get(url)
    
    if response.status_code == 404:
        raise FileNotFoundError(f"The requested file was not found (404): {url}")

    # ZIP-Datei im Speicher entpacken
    with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        #logger.info(thezip.namelist())

        with thezip.open(thezip.namelist()[-1]) as file:
            df = pd.read_csv(file, delimiter=';', skipinitialspace=True, dtype=str)

    return df

def _download_radiosondes_year(station_id: str, year: int, df_stations: pd.DataFrame) -> xr.Dataset:
    """
    Downloads and processes radiosonde data for a given station and year.
    This function retrieves radiosonde data from the DWD (Deutscher Wetterdienst) open data portal,
    either from the recent or historical archive depending on the specified year. The data is filtered,
    cleaned, and converted into an xarray.Dataset with standardized variable names and additional
    station metadata.

    Parameters
    ----------
    station_id : str
        The identifier of the radiosonde station.
    year : int
        The year for which to download the radiosonde data.
    df_stations : pd.DataFrame
        DataFrame containing station metadata, indexed by station_id.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the processed radiosonde data, including variables for temperature,
        dew point, wind speed and direction, pressure, relative humidity, height, latitude, longitude,
        and time information, as well as station metadata.

    Raises
    ------
    FileNotFoundError
        If no data is found for the specified year and station.

    Notes
    -----
    - Temperature and dew point are converted to Kelvin.
    """
    
    if datetime.now().year == year:
        url = f'https://opendata.dwd.de/climate_environment/CDC/observations_germany/radiosondes/high_resolution/recent/sekundenwerte_aero_{station_id}_akt.zip'
    else:
        url = f'https://opendata.dwd.de/climate_environment/CDC/observations_germany/radiosondes/high_resolution/historical/{year}/sekundenwerte_aero_{station_id}_{year}0101_{year}1231_hist.zip'

    df = _download_file(url)

    cloumns_to_use = ['BEZUGSDATUM_SYNOP', 'MESSZEITPUNKT', 'AE_GB_POS', 'AE_GL_POS', 'AE_GPM', 'AE_P', 'AE_TT', 'AE_TD', 'AE_FF', 'AE_DD', 'AE_RF']
    df = df[cloumns_to_use]

    df['BEZUGSDATUM_SYNOP'] = pd.to_datetime(df['BEZUGSDATUM_SYNOP'], format='%Y%m%d%H', errors='coerce')
    df = df.dropna(subset=['BEZUGSDATUM_SYNOP', 'MESSZEITPUNKT'])#.copy()
    df = df.drop_duplicates(subset=['BEZUGSDATUM_SYNOP', 'MESSZEITPUNKT'])#.copy()

    if df.empty:
        raise FileNotFoundError(f"No data in file for {year}")

    numeric_columns = ['MESSZEITPUNKT', 'AE_GB_POS', 'AE_GL_POS', 'AE_GPM', 'AE_P', 'AE_TT', 'AE_TD', 'AE_FF', 'AE_DD', 'AE_RF']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    nan_values = [-999]
    df = df.replace(nan_values, np.nan) 

    df = _filter_data(df)

    df['MESSZEITPUNKT'] = df['MESSZEITPUNKT'].astype(int)
    ds = df.set_index(['BEZUGSDATUM_SYNOP', 'MESSZEITPUNKT']).to_xarray()

    ds = ds.rename({"MESSZEITPUNKT": "meastime",'BEZUGSDATUM_SYNOP': 'synop', "AE_P": "p", "AE_TT": "T", "AE_TD": "TD", "AE_FF": "wind_speed", "AE_DD": "wind_dir", "AE_GB_POS": "lat", "AE_GL_POS": "lon", "AE_GPM": "height", "AE_RF": "rh"})

    ds["T"] += 273.15
    ds["TD"] += 273.15

    ds['starttime'] = ds.synop - pd.Timedelta(minutes=75)
    ds['endtime'] = ds.synop + pd.Timedelta(minutes=15)

    ds.attrs['station_id'] = station_id
    ds.attrs['station_name'] = df_stations.loc[station_id, 'Stationsname']
    ds.attrs['station_height'] = df_stations.loc[station_id, 'Stationshoehe']
    ds.attrs['station_lat'] = df_stations.loc[station_id, 'geoBreite']
    ds.attrs['station_lon'] = df_stations.loc[station_id, 'geoLaenge']

    # Units
    ds['meastime'].attrs["units"] = 's'
    ds['T'].attrs["units"] = "K"
    ds['TD'].attrs["units"] = "K"
    ds['p'].attrs["units"] = "hPa"
    ds['rh'].attrs["units"] = "%"
    ds['wind_speed'].attrs["units"] = "m/s"
    ds['wind_dir'].attrs["units"] = "°"
    ds['lat'].attrs["units"] = "°N"
    ds['lon'].attrs["units"] = "°E"
    ds['height'].attrs["units"] = "m"

    return ds

def get_df_stations(log: bool = False) -> pd.DataFrame:
    """
    Fetch and parse DWD radiosonde station metadata.

    Parameters
    ----------
    log : bool, optional
        If True, log the full station table for inspection.

    Returns
    -------
    pandas.DataFrame
        Station information indexed by station ID with coordinates, heights,
        names, and validity periods.
    """
    response = requests.get('https://opendata.dwd.de/climate_environment/CDC/observations_germany/radiosondes/high_resolution/recent/sec_aero_Beschreibung_Stationen.txt')
    response.raise_for_status()

    column_names = [
        "Stations_id", "von_datum", "bis_datum", "Stationshoehe", "geoBreite", 
        "geoLaenge", "Stationsname", "Bundesland"
    ]
    column_widths = [6, 9, 9, 15, 12, 10, 41, 41]

    # Erstelle ein DataFrame aus den festen Spaltenbreiten
    df_stations = pd.read_fwf(StringIO(response.text), widths=column_widths, header=1, names=column_names)

    # Index
    df_stations = df_stations.set_index('Stations_id')
    df_stations.index = df_stations.index.map(lambda x: f"{x:05d}")

    # Convert date columns to datetime
    df_stations['von_datum'] = pd.to_datetime(df_stations['von_datum'], format='%Y%m%d', errors='coerce')
    df_stations['bis_datum'] = pd.to_datetime(df_stations['bis_datum'], format='%Y%m%d', errors='coerce')

    # Corrections
    df_stations.loc['00125', "Stationshoehe"] = 760 # Apparently stored incorrectly

    if log:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False) 

        logger.info(f'Station information:\n{df_stations}') # .loc[[int(id) for id in station_ids]] information about the stations

    return df_stations


def download_radiosondes(station_id: str, data_dir: str, start_year = 1990, end_year = datetime.now().year) -> None:
    """
    Downloads and saves radiosonde data for a given station across all available years.

    For the specified station ID, this function:

    - Creates a directory to store the downloaded data if it does not exist.
    - Determines the range of years to download data for (from 1990 to the current year).
    - Iterates through each year in the range, downloading radiosonde data for that year.
    - Saves each year's data as a NetCDF file in data/radiosondes/.
    - Logs the progress and skips years where data is not found.

    Parameters
    ----------
        station_id: str 
            The identifier of the station for which to download radiosonde data.
        data_dir: str
            Base directory containing all openMWR-managed data.
        start_year: int, optional
            The starting year for downloading data (default is 1990).
        end_year: int, optional
            The ending year for downloading data (default is the current year).
    """

    logger.info(f"Downloading radiosonde data for station {station_id}")

    df_stations = get_df_stations(log=False)

    station_dir = radiosonde_station_dir(data_dir, station_id)
    os.makedirs(station_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):

        try:
            ds_year = _download_radiosondes_year(station_id, year, df_stations)

            file = radiosonde_year_file(data_dir, station_id, year)
            ds_year.to_netcdf(file)
            logger.info(f"Downloaded data for {year} and saved to {file}.")

        except FileNotFoundError as e:
            logger.info(f"Skipping {year}: {e}")

def update_radiosonde_data(station_id: str, year: int, data_dir: str):
    """
    Refresh radiosonde data for a specific station and year.

    Parameters
    ----------
    station_id : str
        Identifier of the radiosonde station.
    year : int
        Year to update.
    data_dir : str
        Base directory containing all openMWR-managed data.

    Returns
    -------
    list
        Sorted list of newly added synop timestamps.

    Notes
    -----
    Creates the yearly NetCDF file if it does not exist and overwrites it with
    the latest download. Logs how many new launches were added.
    """

    logger.info(f"Updating radiosonde data for station {station_id} for year {year}")

    file = radiosonde_year_file(data_dir, station_id, year)

    if os.path.exists(file):
        ds_old = xr.load_dataset(file)
        old_synop = set(pd.to_datetime(ds_old['synop'].values))
    else:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        logger.info(f"No existing data found for {year}. Downloading new data.")
        ds_old = None
        old_synop = set()

    df_stations = get_df_stations()
    try:
        ds_new = _download_radiosondes_year(station_id, year, df_stations)
    except FileNotFoundError as e:
        logger.info(f"Skipping {year}: {e}")
        return []
    ds_new.to_netcdf(file)

    new_synop = set(pd.to_datetime(ds_new['synop'].values))
    added_synop = sorted(list(new_synop - old_synop))

    if added_synop:
        logger.info(f"Updated radiosonde data for {year} and saved to {file}.")
        logger.info(f'Added {len(added_synop)} synops: {added_synop}')
    else:
        logger.info(f'No new synops found for {station_id} in {year}')

    return added_synop
