import sys
import time
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import warnings
import traceback

def progress_bar(current, total, start_time, bar_length=50):
    """
    Write a progress bar with ETA information to stdout.

    Parameters
    ----------
    current : int
        Current step (0-based).
    total : int
        Total number of steps.
    start_time : float
        Start timestamp from ``time.time()``.
    bar_length : int, optional
        Width of the progress bar in characters, by default 50.
    """
    progress = current / (total - 1)
    block = int(bar_length * progress)
    bar = "#" * block + " " * (bar_length - block)
    
    elapsed_time = time.time() - start_time
    if current > 0:
        estimated_total_time = elapsed_time / current * total
        estimated_total_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_total_time))
        remaining_time = estimated_total_time - elapsed_time
        remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
    else:
        estimated_total_time_str = ""
        remaining_time_str = ""
    
    sys.stdout.write(f"\r[{bar}] {progress * 100:.2f}% - Est. Remaining Time: {remaining_time_str} - Est. Total Time: {estimated_total_time_str}")
    sys.stdout.flush()

def _format_time(seconds):
    """
    Format seconds into a human-readable ``DDd HH:MM:SS`` string.

    Parameters
    ----------
    seconds : float
        Time interval in seconds.

    Returns
    -------
    str
        Formatted duration string.
    """
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def progress_log(current, total, start_time):
    """
    Build a textual progress status with estimated timing.

    Parameters
    ----------
    current : int
        Current step (0-based).
    total : int
        Total number of steps.
    start_time : float
        Start timestamp from ``time.time()``.

    Returns
    -------
    str
        Progress string including percentage, remaining, and total time.
    """
    progress = current / total #(total - 1)
    
    elapsed_time = time.time() - start_time
    if current > 0:
        estimated_total_time = elapsed_time / current * total
        estimated_total_time_str = _format_time(estimated_total_time)
        remaining_time = estimated_total_time - elapsed_time
        remaining_time_str = _format_time(remaining_time)
    else:
        estimated_total_time_str = ""
        remaining_time_str = ""

    return f"{progress * 100:.2f}% - Est. Remaining Time: {remaining_time_str} - Est. Total Time: {estimated_total_time_str}"

def is_called_from_notebook():
    """
    Check whether the current interpreter runs inside a Jupyter notebook.

    Returns
    -------
    bool
        ``True`` if executed from an IPython kernel, otherwise ``False``.
    """
    try:
        # Check whether an IPython environment exists
        from IPython import get_ipython
        ipython_env = get_ipython()
        # Check whether the shell corresponds to a notebook kernel
        if ipython_env and 'IPKernelApp' in ipython_env.config:
            return True
        return False
    except ImportError:
        return False

def setup_daily_logger(name=__name__, file=f"{__file__}.log", level=logging.INFO):
    """
    Create a daily rotating file logger.

    Parameters
    ----------
    name : str, optional
        Logger name, by default ``__name__``.
    file : str, optional
        Path to the log file, by default ``"{__file__}.log"``.
    level : int, optional
        Logging level (e.g., ``logging.INFO``), by default ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    Existing handlers are cleared to prevent duplicate log entries. The handler
    rotates at midnight UTC without keeping backups.
    """


    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Doppelte Handler verhindern
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = TimedRotatingFileHandler(
        file,
        when="midnight",
        interval=1,
        backupCount=0,
        encoding='utf-8',
        utc=True
    )

    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def remove_files_in_dir(path: str):
    """
    Remove all files inside a directory.

    Parameters
    ----------
    path : str
        Directory whose files should be removed.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    PermissionError
        If a file cannot be removed due to permissions.
    OSError
        On other filesystem errors while removing files.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"The directory {path} does not exist.")
    
    for element in os.listdir(path):
        element_path = os.path.join(path, element)
        if os.path.isfile(element_path):
            os.remove(element_path)

def setup_logging():
    """
    Configure root logging with a stream handler and warning capture.

    Returns
    -------
    logging.Logger
        Root logger configured with formatter and custom warning hook.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    ))
    logger.addHandler(handler)

    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        stack = '\n'.join(traceback.format_stack())
        logger.warning(
            f'{category.__name__}: {message} (in {filename} at line {lineno})\nTraceback (most recent call last):\n{stack}'
        )

    warnings.showwarning = custom_warning_handler

    return logger

def round_down(n, k): 
    """
    Round a number down to the nearest multiple of ``k``.

    Parameters
    ----------
    n : int or float
        Value to round.
    k : int or float
        Target multiple.

    Returns
    -------
    int or float
        Rounded value.
    """
    return n - (n%k)

def round_up(n, k):
    """
    Round a number up to the nearest multiple of ``k``.

    Parameters
    ----------
    n : int or float
        Value to round.
    k : int or float
        Target multiple.

    Returns
    -------
    int or float
        Rounded value.
    """
    return n - (n%k) + k


def patch_pyrtlib_numpy_compat() -> None:
    """
    Patch pyrtlib absorption behavior for NumPy>=2 compatibility.

    Returns
    -------
    None
        Applies monkey patches in place. Repeated calls are safe.

    Notes
    -----
    Uses lazy imports so modules that do not run pyrtlib are unaffected.
    """
    import numpy as np
    import pyrtlib.absorption_model as absorption_model
    from pyrtlib.absorption_model import H2OAbsModel, O2AbsModel

    if not getattr(H2OAbsModel.h2o_absorption, "_openmwr_numpy_compat_patch", False):
        original_h2o_absorption = H2OAbsModel.h2o_absorption

        def _h2o_absorption_safe(self, *args, **kwargs):
            frq = kwargs.get("frq", args[3] if len(args) > 3 else None)
            needs_sd_df_shape_patch = (
                H2OAbsModel.model in {"R20SD", "R21SD", "R22SD", "R23SD", "R24", "MWL24"}
                and frq is not None
                and np.ndim(frq) == 0
            )

            if needs_sd_df_shape_patch:
                # pyrtlib SD-family scalar-frequency path can create ``df`` with
                # shape (2, 1) and later pass ``df[j]`` into ``complex(...)``,
                # which fails in newer NumPy/Python combinations.
                original_zeros = absorption_model.np.zeros

                def _zeros_safe(shape, *zargs, **zkwargs):
                    if shape == (2, 1):
                        return original_zeros(2, *zargs, **zkwargs)
                    return original_zeros(shape, *zargs, **zkwargs)

                absorption_model.np.zeros = _zeros_safe
                try:
                    npp, ncpp = original_h2o_absorption(self, *args, **kwargs)
                finally:
                    absorption_model.np.zeros = original_zeros
            else:
                npp, ncpp = original_h2o_absorption(self, *args, **kwargs)

            npp_arr = np.asarray(npp)
            ncpp_arr = np.asarray(ncpp)
            if npp_arr.ndim == 0:
                npp = float(npp_arr)
            elif npp_arr.size == 1:
                npp = float(npp_arr.reshape(()))
            if ncpp_arr.ndim == 0:
                ncpp = float(ncpp_arr)
            elif ncpp_arr.size == 1:
                ncpp = float(ncpp_arr.reshape(()))
            return npp, ncpp

        _h2o_absorption_safe._openmwr_numpy_compat_patch = True  # type: ignore[attr-defined]
        H2OAbsModel.h2o_absorption = _h2o_absorption_safe

    if not getattr(O2AbsModel.o2_absorption, "_openmwr_numpy_compat_patch", False):
        original_o2_absorption = O2AbsModel.o2_absorption

        def _o2_absorption_safe(self, *args, **kwargs):
            npp, ncpp = original_o2_absorption(self, *args, **kwargs)
            npp_arr = np.asarray(npp)
            ncpp_arr = np.asarray(ncpp)
            if npp_arr.ndim == 0:
                npp = float(npp_arr)
            elif npp_arr.size == 1:
                npp = float(npp_arr.reshape(()))
            if ncpp_arr.ndim == 0:
                ncpp = float(ncpp_arr)
            elif ncpp_arr.size == 1:
                ncpp = float(ncpp_arr.reshape(()))
            return npp, ncpp

        _o2_absorption_safe._openmwr_numpy_compat_patch = True  # type: ignore[attr-defined]
        O2AbsModel.o2_absorption = _o2_absorption_safe
