#!/home/m/met-hatpro/scripts/mwr_retrieval/venv/bin/python

import pandas as pd
from datetime import datetime, timedelta
import sys 
import os
import pytz

from openMWR.models import Model
from openMWR.run import import_data_and_run_retrieval
from openMWR.plot import TimeSeriesCreator, SkewT
from openMWR.dwd_opendata import update_radiosonde_data
from openMWR.site import get_config_parameter
from openMWR.utils import setup_logging

#logger = setup_daily_logger()
logger = setup_logging()
DATA_DIR = "../data"

def run_retrival_and_plot(date, site, plt_bls=True, plot_rbg_retrieval=True):
    logger.info(f"Run retrival and plot for the {date.strftime('%Y-%m-%d')} for {site}")

    plot_dir = get_config_parameter(site, 'plot_dir', DATA_DIR)

    plot_dir_month = plot_dir + '/' + date.strftime('%Y/%m/')
    if not os.path.exists(plot_dir_month):
        os.makedirs(plot_dir_month)

    file_name = f"{plot_dir_month}{date.strftime('%Y%m%d')}_"

    #Select models
    if site == 'munich_G5':
        NN_model_name = 'NN_opt'
        MLR_model_name = 'MLR_opt'
    elif site == 'zugspitze':
        NN_model_name = 'NN_opt_era5_biased_05'
        MLR_model_name = 'MLR'

    model_list = [Model.load(NN_model_name, site, DATA_DIR), Model.load(MLR_model_name, site, DATA_DIR)]

    if site == 'zugspitze':
        model_list.append(Model.load('NN_opt_era5', site, DATA_DIR))


    try:
        pred, ds_hatpro = import_data_and_run_retrieval(date, site, DATA_DIR, model_list, import_retrieval_data=plot_rbg_retrieval)
    except FileNotFoundError as e:
        logger.error(e)
        logger.debug(e, exc_info=True)
        return None
    except ValueError as e:
        logger.error(e)
        logger.debug(e, exc_info=True)
        return None
    
    retrieval_output_dir = get_config_parameter(site, 'retrieval_output_dir', DATA_DIR)
    ret_dir_day = retrieval_output_dir + f'/{date.strftime("%Y/%m/")}'

    os.makedirs(ret_dir_day, exist_ok=True)

    vars_to_save = [var for var in pred.data_vars if var not in ['doy_sin', 'doy_cos']]

    pred[vars_to_save].to_netcdf(f'{ret_dir_day}{date.strftime("%Y%m%d")}_retrieval.nc')

    #BLS model
    if plt_bls:
        if site == 'munich_G5':
            NN_BLS_name = 'NN_BLS_opt_var_ang_lin_3'
        else:
            NN_BLS_name = 'NN_BLS_opt_var_ang_lin'

        model_list_bls = [Model.load(NN_BLS_name, site, DATA_DIR)]

        try:
            pred_bls, ds_hatpro_bls = import_data_and_run_retrieval(date, site, DATA_DIR, model_list_bls, import_retrieval_data=plot_rbg_retrieval, bls=True)
        except FileNotFoundError as e:
            logger.error(e)
            logger.debug(e, exc_info=True)
            plt_bls = False
            pred_bls, ds_hatpro_bls = None, None
        except ValueError as e:
            logger.error(e)
            logger.debug(e, exc_info=True)
            plt_bls = False
            pred_bls, ds_hatpro_bls = None, None
    else:
        pred_bls, ds_hatpro_bls = None, None

    if pred_bls is not None:
        pred_bls = pred_bls.sel(model=NN_BLS_name)
        
        vars_to_save = [var for var in pred_bls.data_vars if var not in ['doy_sin', 'doy_cos']]

        pred_bls[vars_to_save].to_netcdf(f'{ret_dir_day}{date.strftime("%Y%m%d")}_retrieval_bls.nc')

    #Split predictions
    pred_NN = pred.sel(model=NN_model_name)
    pred_MLR = pred.sel(model=MLR_model_name)

    if not plot_rbg_retrieval:
        ds_hatpro = None
        ds_hatpro_bls = None

    #Skew_T
    if site == 'zugspitze':
        raso_station_id = '02290'
        min_pressure_la=400
        max_pressure_la=800
    elif site == 'munich_G5' or site == 'munich_G4':
        raso_station_id = '03715'
        min_pressure_la=500
        max_pressure_la=1000

    station_alt = get_config_parameter(site, 'mwr_height', DATA_DIR)

    if date == datetime.now(tz=pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None):
        skew_T_date = pd.to_datetime(pred.time.values[-1])
        plot_radiosonde=False
    else:
        skew_T_date = date.replace(hour=12)
        plot_radiosonde=True

    retrieval_label = 'MIM-Retrieval'


    # Full atmosphere
    skew = SkewT(skew_T_date, DATA_DIR, station_alt=station_alt)
    if plot_radiosonde:
        skew.plot_radiosonde(raso_station_id)
    if plot_rbg_retrieval:
        skew.plot_retrieval(ds_hatpro.dropna(dim='time', subset=['T', 'TD']), 'RPG-Retrieval', color='green')
    skew.plot_retrieval(pred_NN, retrieval_label, color='red')
    skew.finalize()
    skew.save(file_name + 'radiosonde.svg')

    #Lower atmosphere
    skew = SkewT(skew_T_date, DATA_DIR, station_alt=station_alt, min_pressure=min_pressure_la, max_pressure=max_pressure_la, T_range=40)
    if plot_radiosonde:
        skew.plot_radiosonde(raso_station_id)
    if plot_rbg_retrieval:
        skew.plot_retrieval(ds_hatpro.dropna(dim='time', subset=['T', 'TD']), 'RPG-Retrieval', color='green')
        skew.plot_retrieval(ds_hatpro_bls, 'RPG-Retrieval\nBoundary Layer Scan', color='orange', plot_TD=False)
    skew.plot_retrieval(pred_NN, retrieval_label, color='red')
    if plt_bls:
        skew.plot_retrieval(pred_bls, f'{retrieval_label}\nBoundary Layer Scan', color='purple')
    skew.finalize()
    skew.save(file_name + 'radiosonde_la.svg')

    ###Time series

    if site == 'zugspitze':
        pre_title = 'Umweltforschungstation Schneefernerhaus (Zugspitze, Germany; 47.417 N / 10.980 E)'
    else:
        pre_title = 'Meteorological Institute of Ludwig-Maximilians-Universität (München, Germany; 48.148 N / 11.573 E)'

    #LWP
    lwp = TimeSeriesCreator.lwp(date, pred_NN, pred_MLR, ds_hatpro, pre_title=pre_title, retrieval_label=retrieval_label)
    lwp.save(file_name + 'lwp.svg')

    #IWV
    iwv = TimeSeriesCreator.iwv(date, pred_NN, ds_hatpro, pre_title=pre_title, retrieval_label=retrieval_label)
    iwv.save(file_name + 'iwv.svg')

    #Colormaps
    data = pred_NN.sel(height=slice(0, 10000)).transpose('time', ...)

    temp = TimeSeriesCreator.temp(date, data, pre_title=pre_title, retrieval_label=retrieval_label)
    temp.save(file_name + 'temp.svg') #, dpi=300

    temp_an = TimeSeriesCreator.temp_an(date, data, pre_title=pre_title, retrieval_label=retrieval_label)
    temp_an.save(file_name + 'temp_an.svg')

    dT = TimeSeriesCreator.dT(date, data, pre_title=pre_title, retrieval_label=retrieval_label)
    dT.save(file_name + 'temp_grad.svg')

    rh = TimeSeriesCreator.rh(date, data, pre_title=pre_title, retrieval_label=retrieval_label)
    rh.save(file_name + 'rh.svg')

    ah = TimeSeriesCreator.ah(date, data, pre_title=pre_title, retrieval_label=retrieval_label)
    ah.save(file_name + 'ah.svg')

    lwc = TimeSeriesCreator.lwc(date, data, pre_title=pre_title, retrieval_label=retrieval_label)
    lwc.save(file_name + 'lwc.svg')

def run_today_and_update(site, raso_station_id, plt_bls=True, plot_rbg_retrieval=True):
    #Current date
    now = datetime.now(tz=pytz.utc).replace(tzinfo=None)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    run_retrival_and_plot(today, site, plt_bls=plt_bls, plot_rbg_retrieval=plot_rbg_retrieval)

    #plot yesterday again
    if now.hour <= 1:
        yesterday = today - timedelta(days=1)
        run_retrival_and_plot(yesterday, site, plt_bls=plt_bls, plot_rbg_retrieval=plot_rbg_retrieval)

    #Update radiosonde data
    if now.hour == 4:

        years = [now.year - 1, now.year] if (now.month == 1 and now.day < 4) or (now.month <= 3 and now.day == 1) else [now.year]

        for year in years:
            added_synop = update_radiosonde_data(raso_station_id, year, DATA_DIR)
            new_dates = pd.Series(added_synop).dt.normalize().drop_duplicates()

            for date in new_dates:
                run_retrival_and_plot(date, site, plt_bls=plt_bls, plot_rbg_retrieval=plot_rbg_retrieval)

def main():
    #Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    args=sys.argv

    if len(args) > 1:

        if len(args) != 5 and len(args) != 8:
            print("Usage: cron_job.py [site] [year] [month] [day] or [site] [start_year] [start_month] [start_day] [end_year] [end_month] [end_day] (end_date is inclusive)")
            sys.exit(1)

        site = args[1] #'munich_G5'

        plt_bls = True #if site == 'munich_G5' else False
        plot_rbg_retrieval = True if site == 'munich_G5' else False

        if len(args) == 5:
            date = datetime(int(args[2]), int(args[3]), int(args[4]))
            run_retrival_and_plot(date, site, plt_bls=plt_bls, plot_rbg_retrieval=plot_rbg_retrieval)

        elif len(args) == 8:
            start_date = datetime(int(args[2]), int(args[3]), int(args[4]))
            end_date = datetime(int(args[5]), int(args[6]), int(args[7]))

            for date in pd.date_range(start=start_date, end=end_date, freq="D"):
                run_retrival_and_plot(date, site, plt_bls=plt_bls, plot_rbg_retrieval=plot_rbg_retrieval)

    else:
        run_today_and_update('munich_G5', '03715', plt_bls = True, plot_rbg_retrieval = True)
        run_today_and_update('zugspitze', '02290', plt_bls = True, plot_rbg_retrieval = False)


if __name__ == '__main__':
    main()
