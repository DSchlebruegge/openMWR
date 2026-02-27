from multiprocessing import Pool
import multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener
import xarray as xr
import numpy as np
import pandas as pd
import time
import logging

from openMWR.utils import progress_log

logger = logging.getLogger(__name__)


def _default_log_formatter() -> logging.Formatter:
    return logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def _default_stream_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(_default_log_formatter())
    return handler


def _start_multiprocessing_log_listener():
    """Start a queue listener that forwards worker log records."""
    root_logger = logging.getLogger()
    handlers = [h for h in root_logger.handlers if not isinstance(h, QueueHandler)]

    if not handlers:
        handlers = [_default_stream_handler()]

    log_queue = mp.Queue()
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    return log_queue, listener


def _configure_multiprocessing_worker_logging(log_queue, level=logging.INFO) -> None:
    """Route worker-process logs into the shared multiprocessing queue."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(QueueHandler(log_queue))

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

        logger.info(f'Process {process_number}: Completed processing.')
        return ds

    logger.info(f'Process {process_number}: Completed processing.')

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

    log_queue, log_listener = _start_multiprocessing_log_listener()
    try:
        with Pool(
            processes=num_of_processes,
            initializer=_configure_multiprocessing_worker_logging,
            initargs=(log_queue,),
        ) as pool:
            ds_list = pool.starmap(_kernal_function, args)
    finally:
        log_listener.stop()

    ds_list = [ds for ds in ds_list if ds is not None]
    if len(ds_list) == 0:
        raise ValueError("No data processed successfully in any of the parallel processes.")
    
    ds_new = xr.concat(ds_list, dim='time')
    ds_new = ds_new.sortby('time')

    return ds_new


def _kernal_function_date_range(
    dr,
    process_number,
    function,
    args_for_function,
    kwargs_for_function,
):
    
    n_times = len(dr)
    start_time = time.time()

    div = max(n_times // 10, 1)

    for i, date in enumerate(dr):
        if i % div == 0:
            logger.info(f'Process {process_number}: {progress_log(i, n_times, start_time)}')

        function(
            date,
            *args_for_function,
            **kwargs_for_function,
        )

    logger.info(f'Process {process_number}: Completed processing date range.')


def run_pool_date_range(
    dr: pd.DatetimeIndex,
    num_of_processes: int,
    function,
    *args_for_function,
    **kwargs_for_function,
) -> None:
    """
    Run a function in parallel over a shuffled pandas date range.

    Parameters
    ----------
    dr : pandas.DatetimeIndex
        Date range to be processed.
    num_of_processes : int
        Number of parallel processes to use.
    function : callable
        Function called once per date inside each worker. It must accept the
        date (``pandas.Timestamp``) as the first argument.
    *args_for_function
        Additional positional arguments passed to ``function``.
    **kwargs_for_function
        Additional keyword arguments passed to ``function``.

    Returns
    -------
    None

    Notes
    -----
    - The input date range is shuffled before splitting to preserve randomized
      processing order across workers.
    - Each worker receives one split of the shuffled date range and iterates
      over its dates, calling ``function(date, *args_for_function, **kwargs_for_function)``.
    """

    shuffled_idx = np.random.permutation(len(dr))
    dr_shuffled = dr.take(shuffled_idx)

    n_dates = len(dr_shuffled)
    split_date_ranges = [
        dr_shuffled[int(n_dates / num_of_processes * i): int(n_dates / num_of_processes * (i + 1))]
        for i in range(num_of_processes)
    ]

    args = [
        (split_date_ranges[i], i, function, args_for_function, kwargs_for_function)
        for i in range(num_of_processes)
    ]

    log_queue, log_listener = _start_multiprocessing_log_listener()
    try:
        with Pool(
            processes=num_of_processes,
            initializer=_configure_multiprocessing_worker_logging,
            initargs=(log_queue,),
        ) as pool:
            pool.starmap(_kernal_function_date_range, args)
    finally:
        log_listener.stop()
