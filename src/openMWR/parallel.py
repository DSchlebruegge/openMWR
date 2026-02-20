from multiprocessing import Pool
import xarray as xr
import numpy as np
import time
import logging

from openMWR.utils import progress_log

logger = logging.getLogger(__name__)

def _kernal_function(ds_orig, process_number, function, args_for_function, kwargs_for_function):
    ds_list = []

    n_times = ds_orig.time.size
    start_time = time.time()

    div = max(n_times // 10, 1)

    for i in range(n_times):
        if i % div == 0:
            logger.info(f'Process {process_number}: {progress_log(i, n_times, start_time)}')

        try:
            ds_date = function(ds_orig.isel(time=i), *args_for_function, **kwargs_for_function)
            ds_date = ds_date.expand_dims({'time': [ds_orig.time.isel(time=i).values]})

        except Exception as e:
            logger.error(f'Process {process_number}: Error processing time {ds_orig.time.isel(time=i).values}:', exc_info=True)
            continue

        if ds_date is not None:
            ds_list.append(ds_date)

    if len(ds_list) != 0:
        ds = xr.concat(ds_list, dim='time')
        return ds

def run_pool(ds_orig, num_of_processes, function, *args_for_function, **kwargs_for_function):
    """
    Runs a given function in parallel across multiple processes, each operating on a shuffled split of the input dataset along the 'time' dimension.
    
    Parameters
    ----------
    ds_orig : xarray.Dataset
        The original dataset to be processed, expected to have a 'time' dimension.
    num_of_processes : int
        The number of parallel processes to use.
    function : callable
        The function to apply to each split of the dataset.
    *args_for_function
        Additional arguments to pass to the function.
    **kwargs_for_function
        Additional keyword arguments to pass to the function.
    Returns
    -------
    xarray.Dataset
        The concatenated and time-sorted result of applying the function to each split of the dataset.
    Notes
    -----
    - The 'time' dimension of the dataset is shuffled before splitting to ensure random distribution across processes.
    - Each process receives a contiguous chunk of the shuffled dataset.
    - The function is expected to return an xarray.Dataset or None; None results are filtered out.
    """

    shuffled_time = np.random.permutation(ds_orig['time'].values)
    ds_shuffled = ds_orig.reindex(time=shuffled_time)

    n_dates = ds_shuffled.sizes['time']

    split_datasets = [ds_shuffled.isel(time=slice(int(n_dates/num_of_processes*i), int(n_dates/num_of_processes*(i+1)))) for i in range(num_of_processes)]

    args = [
        (split_datasets[i], i, function, args_for_function, kwargs_for_function)
        for i in range(num_of_processes)
    ]

    with Pool(processes=num_of_processes) as pool:
        ds_list = pool.starmap(_kernal_function, args)

    ds_list = [ds for ds in ds_list if ds is not None]
    if len(ds_list) == 0:
        raise ValueError("No data processed successfully in any of the parallel processes.")
    
    ds_new = xr.concat(ds_list, dim='time')
    ds_new = ds_new.sortby('time')

    return ds_new