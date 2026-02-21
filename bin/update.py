from datetime import datetime

from openMWR.dwd_opendata import update_radiosonde_data
from openMWR.radiosonde import create_radiosonde_dataset_for_RT, create_radiosonde_dataset_with_RT, create_analysis_dataset
from openMWR.hatpro_data import update_hatpro_dataset
from openMWR.utils import setup_logging
from openMWR.era5 import ERA5Processor
from openMWR.consts import std_freqs, std_angles, std_heights

logger = setup_logging()
DATA_DIR = "../data"

def update():
    site = 'munich_G5'

    station_ids = ['03715']
    station_ids_for_analysis = ['03715']

    freqs, angles, heights = std_freqs, std_angles, std_heights

    ### Update raw radiosonde data
    first_year = datetime.now().year - 1
    for raso_station_id in station_ids:
        for year in [first_year, datetime.now().year]:
            update_radiosonde_data(raso_station_id, year, DATA_DIR)

    ### Prepare the the downloaded raw radiosonde data for the RT calculation. (Specific for each site)
    cut_off_at_mwr_height = False
    for station_id in station_ids: 
        create_radiosonde_dataset_for_RT(site, station_id, DATA_DIR, heights, update_only=True, cut_off_at_mwr_height=cut_off_at_mwr_height, first_year=first_year)

    ### Calculate TBs with RT models for each station.
    for station_id in station_ids: 
        create_radiosonde_dataset_with_RT(station_id, site, DATA_DIR, freqs, angles, num_of_processes = 12, update_only=True)

    ### Update the hatpro dataset with new data
    update_hatpro_dataset(site, DATA_DIR)

    ### Create an analysis dataset with Radiosonde data and hatpro data
    create_analysis_dataset(site, station_ids_for_analysis, DATA_DIR)

    ### Create an analysis dataset with ERA5 data and hatpro data
    era5 = ERA5Processor(site, DATA_DIR, update_only=True)
    era5.create_original_dataset(h_freq=2)
    era5.create_forward_calc_dataset(freqs, heights, angles, num_of_processes=12)
    era5.create_analysis_dataset_from_forward_calc()

def main():
    update()

if __name__ == '__main__':
    main()
