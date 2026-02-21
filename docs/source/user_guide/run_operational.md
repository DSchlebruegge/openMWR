# Run Retrieval Operational

## Cron Job

To run the retrieval operationally, we set up a cron job that executes the `cron_job.py` script located in the `bin` directory. This script is run every hour.

The cron job performs the following tasks:
1. It retrieves the current date and time.
2. It runs the retrieval for the current day.
3. It saves the results of the retrieval.
4. It generates and saves plots of the retrieval results such as Skew-T diagrams and time-height plots.
5. If the current time is within the first hour of the day, it also re-runs the retrieval for the previous day to include any late-arriving data.
6. At 4:00 a.m. UTC, it updates the locally saved radiosonde data to ensure that the most recent atmospheric profiles are available for plotting.

The plots are saved in the directory specified in the site's configuration file as `plot_dir`.
The retrieval results are saved in the directory specified in the site's configuration file as `retrieval_output_dir`.

The `cron_job.py` script can also be run manually for past dates by providing the following arguments:

Run just one day:

    ./cron_job.py [site] [year] [month] [day]

Or, run a range of days:

    ./cron_job.py [site] [start_year] [start_month] [start_day] [end_year] [end_month] [end_day]

Note that the end date is inclusive.

The `cron_job.py` script needs to be personalized. Site name, plot titles, and retrieval names are set in the script itself.

## Update Radiosonde and ERA5 Data

There is also an `update.py` script in the `bin` directory that can be run like this:

    ./run_py_script.sh update.py

This script updates all the datasets created in [Create a Retrieval](#create_retrieval) or with `bin/main.py` with the latest available data.

In the list below, `DATA_DIR` refers to the data root configured in your scripts and passed to the openMWR functions.

Specifically, the script updates the following datasets:
- The raw, unchanged radiosonde data:<br>
 `{DATA_DIR}/radiosondes/station_{station}/raw_data_{year}.nc`.
- The filtered and interpolated radiosonde data:<br>
`{DATA_DIR}/sites/{site}/radiosonde/radiosonde_data_{station}.nc`.
- The radiosonde data with forward-calculated brightness temperatures:<br>
`{DATA_DIR}/sites/{site}/radiosonde/radiosonde_data_with_RT_{station}.nc`.
- The Hatpro dataset with all the measurements:<br>
`{DATA_DIR}/sites/{site}/hatpro/hatpro_data.nc`.
- The radiosonde analysis dataset:<br>
`{DATA_DIR}/sites/{site}/analysis/analysis_data_radiosonde.nc`.
- The raw, unchanged ERA5 data:<br>
`{DATA_DIR}/era5/era5_{lat}_{lon}.nc`.
- The interpolated ERA5 data with forward calculation:<br>
`{DATA_DIR}/sites/{site}/era5/forward_calc_era5.nc`.
- The ERA5 analysis dataset:<br>
`{DATA_DIR}/sites/{site}/analysis/analysis_data_era5.nc`.

The `update.py` script also needs to be personalized. Site names and radiosonde station IDs are set in the script itself.