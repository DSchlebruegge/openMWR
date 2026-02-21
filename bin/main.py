from openMWR.utils import setup_logging
from openMWR.consts import std_freqs, std_angles, std_heights

logger = setup_logging()

DATA_DIR = "../data"

def create_munich_G5_site_config():
    from openMWR.site import create_site
    create_site('munich_G5', data_dir=DATA_DIR,
               hatpro_data_dir="/project/meteo/data/hatpro-g5", 
               plot_dir='/project/meteo/homepages/quicklooks/hatpro/G5', 
               retrieval_output_dir="/project/meteo/data/hatpro-g5/mim_retrieval",
               gen='G5', 
               mwr_height=550, 
               rpg_retrieval_exists = True,
               mwr_measurement_start_date = '2024-11-01',
               latitude = 48.148,
               longitude = 11.573)

def radiosonde_data_generation():

    site = 'munich_G5'

    # ___ Step 2: Radiosonde Data ___
    # ### Download all availabe radiosonde data for the stations

    station_ids = ['03715']

    from openMWR.dwd_opendata import download_radiosondes
    for station_id in station_ids: 
        download_radiosondes(station_id, DATA_DIR, start_year = 1990)

    ### Prepare the downloaded raw radiosonde data for the RT calculation. (Specific for each site)
    cut_off_at_mwr_height = False
    from openMWR.radiosonde import create_radiosonde_dataset_for_RT
    for station_id in station_ids: 
        create_radiosonde_dataset_for_RT(site, station_id, DATA_DIR, std_heights, cut_off_at_mwr_height=cut_off_at_mwr_height)

    # ___ Step 3: Radiative Transfer Calculation ___
    ### Calculate TBs with RT models for each station.
    from openMWR.radiosonde import create_radiosonde_dataset_with_RT
    for station_id in station_ids: 
        create_radiosonde_dataset_with_RT(station_id, site, DATA_DIR, std_freqs, std_angles, num_of_processes = 12)

    # ___ Step 4: Separate Training Data into Training, Test, Validation ___
    from openMWR.train import separate_training_data
    for station_id in station_ids: 
        separate_training_data(site, station_id, DATA_DIR)

def analysis_data_generation():

    site = 'munich_G5'

    # ___ Step 5: Hatpro Data for Retrieval Evaluation ___
    ## Create nc file with minutly hatpro measurements and rpg retrieval results
    from openMWR.hatpro_data import create_hatpro_dataset
    create_hatpro_dataset(site, DATA_DIR)

    # ___ Step 6: Create Analysis Dataset ___
    ### Create an analysis dataset with Radiosonde data and hatpro data
    from openMWR.radiosonde import create_analysis_dataset
    station_ids_for_analysis = ['03715']
    create_analysis_dataset(site, station_ids_for_analysis, DATA_DIR)

def era5_data_generation():

    site = 'munich_G5'

    ### Create an analysis dataset with ERA5 data and hatpro data
    from openMWR.era5 import ERA5Processor
    era5 = ERA5Processor(site, DATA_DIR, update_only=False, start_date='2025-01-01', end_date='2025-02-01')
    era5.create_original_dataset(h_freq=2)
    era5.create_forward_calc_dataset(std_freqs, std_heights, std_angles, num_of_processes=12)
    era5.create_analysis_dataset_from_forward_calc() # Expects call of create_hatpro_dataset(site) before, to get the hatpro data for the analysis dataset.

def train():
    from openMWR.models import Model
    from openMWR.train import TrainingWorkflow

    site = 'munich_G5'
    sources = ['03715'] # ['era5']
    end_date_training = '2024-10-31' #Use data until the end of October for training, so that the data from November can be used for evaluation.
    device = "cuda" 

    model = Model('NN', std_heights, std_freqs, is_bls_model=False, is_MLR=False)

    workflow = TrainingWorkflow(model, site, device, DATA_DIR)
    workflow.load_training_data(sources, end_date_training)
    workflow.add_standard_training_hyper_params()
    workflow.train()

def train_SPC_model():
    # ___ Train Spectral Consistency Model ___
    from openMWR.models import SpectralConsistencyModel
    from openMWR.train import TrainingWorkflow

    site = 'munich_G5'
    sources = ['03715']
    end_date_training = '2024-10-31'
    device = "cuda" 

    spc = SpectralConsistencyModel('SPC', std_freqs)

    workflow = TrainingWorkflow(spc, site, device, DATA_DIR)
    workflow.load_training_data(sources, end_date_training)

    nn_model = 'NN'
    workflow.get_hyper_params_from_existing_model(nn_model)

    workflow.change_hyper_params(loss_fn='MSEPlusBiasLoss', lr_height=0.01, batch_size=1024, epochs=1000, epoch_max_lr=100)
    workflow.train()

def main():
    create_munich_G5_site_config()
    radiosonde_data_generation()
    analysis_data_generation()
    era5_data_generation()
    train()
    train_SPC_model()

if __name__ == '__main__':
    main()



