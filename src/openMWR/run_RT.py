import xarray as xr
import numpy as np
import pandas as pd
import os
import time
import logging
from pathlib import Path
import warnings

from pyrtlib.tb_spectrum import TbCloudRTE
from torchMWRT import RTModel, AtmProfile

from openMWR.libRadtran import libRadtran
from openMWR.utils import remove_files_in_dir, patch_pyrtlib_numpy_compat
from openMWR.parallel import run_pool
from openMWR.site import get_config_parameter
from openMWR.paths import libRadtran_dir

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"Number of levels too low .* or minimum pressure value lower than 10 hPa.*",
    category=UserWarning,
    module="pyrtlib.tb_spectrum"
) # Probably only important for satellites

warnings.filterwarnings(
    "ignore",
    message=r"from cloud_radiating_temperature: absorption too large to exponentiate.*",
    category=UserWarning,
    module="pyrtlib.rt_equation"
) #only important for cloud radiating temperature, which is not used here

def calculate_IR_band_TB(site, data_dir, z_km, T_K, rh_100, p_hPa, lwc_gpm3, effective_droplet_radius_mum, ang=np.array([90])):
    """
    Calculate band-integrated infrared brightness temperatures with libRadtran.

    Parameters
    ----------
    site : str
        Site name used to read configuration (determines wavelength band).
    data_dir : str
        Base directory containing all openMWR-managed data.
    z_km : array-like
        Heights in kilometers for each model level.
    T_K : array-like
        Air temperature profile in kelvin.
    rh_100 : array-like
        Relative humidity in percent.
    p_hPa : array-like
        Pressure profile in hectopascals; used for sorting and deduplication.
    lwc_gpm3 : array-like
        Liquid water content in g m^-3.
    effective_droplet_radius_mum : array-like
        Effective droplet radius in micrometers at each level.
    ang : array-like, optional
        Elevation angles in degrees; zenith-only by default.

    Returns
    -------
    float or pandas.Series
        Brightness temperatures for the requested angles.

    Notes
    -----
    Temporary libRadtran input files are written to ``data/libRadtran`` and
    removed after execution. The wavelength band is selected based on the
    site's Hatpro generation.
    """
    zeitstempel_ns = time.time_ns()

    libRadtran_data_dir = Path(libRadtran_dir(data_dir)).resolve()
    os.makedirs(libRadtran_data_dir, exist_ok=True)
    
    radio_file = libRadtran_data_dir / f"radio_{zeitstempel_ns}.dat"

    wc_file = libRadtran_data_dir / f"wc_{zeitstempel_ns}.dat"
    
    df = pd.DataFrame({
        'z_km': z_km,
        'T_K': T_K,
        'rh_100': rh_100,
        'p_hPa': p_hPa,
        'lwc_gpm3': lwc_gpm3,
        'effective_droplet_radius_mum': effective_droplet_radius_mum,
    })
    
    df = df.sort_values(by='p_hPa')
    df = df.drop_duplicates(subset=['p_hPa'])

    df.to_csv(radio_file, index=False, columns=['p_hPa','T_K','rh_100'], sep=' ', header=False)

    df = df.sort_values(by='z_km', ascending=False)
    df = df[df['z_km'] < 15]

    df.to_csv(wc_file, index=False, columns=['z_km', 'lwc_gpm3', 'effective_droplet_radius_mum'], sep=' ', header=False)

    umu = np.cos(np.radians(90 + ang))

    gen = get_config_parameter(site, 'gen', data_dir)

    if gen == 'G5':
        wavelengths = '9600 11500'
    elif gen == 'G4':
        wavelengths = '9200 10600'
    
    try:
        df_rad, stderr = libRadtran(
            atmosphere_file = '../data/atmmod/afglus.dat',
            source = 'thermal',
            mol_abs_param =  'reptran medium',
            wavelength = wavelengths,
            output_process = 'integrate',
            radiosonde = f'{str(radio_file)} h2o RH',
            radiosonde_levels_only = True,
            wc_file = f'1D {str(wc_file)}',
            umu = umu[::-1], # -0.5 np.array([-1, -0.5]
            output_quantity = 'brightness',
            output_user = 'uu',
            quiet = True
            )
    except Exception as e:
        raise e
    finally:
        for f in [radio_file, wc_file]:
            os.remove(f)

    if len(umu) > 1:
        return df_rad['uu'].iloc[0].loc[umu]
    else:
        return df_rad['uu'].iloc[0].item()

def _get_cloud_top_base(lwc):
    """
    Identify cloud base and top indices from a liquid water content profile.

    Parameters
    ----------
    lwc : array-like
        Liquid water content profile; cloud presence is defined by values > 0.

    Returns
    -------
    tuple of numpy.ndarray
        Arrays with indices for cloud tops and bases on the input grid.
    """
    cloud = lwc > 0
    i_base = np.where(np.diff(cloud.astype(int), prepend = 0) == 1)[0]
    i_top = np.where(np.diff(cloud.astype(int)) == -1)[0] + 1
    
    return i_top, i_base

def _effective_droplet_radius_mum(lwc_gpm3: np.ndarray) -> np.ndarray:
    """
    Estimate droplet effective radius from liquid water content.

    Parameters
    ----------
    lwc_gpm3 : numpy.ndarray
        Liquid water content profile in g m^-3.

    Returns
    -------
    numpy.ndarray
        Effective droplet radius in micrometers, clipped to a minimum of 2.5.
    """
    N_cloud_drops = 200e6  # 1/m^3
    rho_w = 1e6  # g/m^3
    effective_droplet_radius_mum = np.cbrt(lwc_gpm3 / (N_cloud_drops * 4 / 3 * np.pi * rho_w)) * 1e6
    effective_droplet_radius_mum[effective_droplet_radius_mum < 2.5] = 2.5
    return effective_droplet_radius_mum

def _run_IR_date(ds_date: xr.Dataset, site: str, data_dir: str) -> xr.DataArray:
    """
    Calculate infrared brightness temperature for one profile.

    Parameters
    ----------
    ds_date : xarray.Dataset
        Single-profile dataset. Required data variables are ``T`` (K),
        ``rh`` (%), ``p`` (hPa), and ``lwc`` (g m^-3), each on the
        ``height`` dimension. A ``height`` coordinate (m) is required.
    site : str
        Site name used to resolve libRadtran wavelength settings.
    data_dir : str
        Base directory containing all openMWR-managed data.

    Returns
    -------
    xarray.DataArray
        Infrared brightness temperature for the requested viewing angle(s).
    """
    z_m = ds_date.height.values
    z_km = z_m / 1000
    T_K = ds_date.T.values
    rh_100 = ds_date.rh.values
    p_hPa = ds_date.p.values
    lwc_gpm3 = ds_date.lwc.values

    effective_droplet_radius_mum = _effective_droplet_radius_mum(lwc_gpm3)
    TB_IR = calculate_IR_band_TB(site, data_dir, z_km, T_K, rh_100, p_hPa, lwc_gpm3, effective_droplet_radius_mum)
    return xr.DataArray(TB_IR)

def _run_pyrtlib_date(ds_date, site, data_dir, freqs, angles, freq_shift: np.ndarray = None, absmdl: str = 'R20'):
    """
    Run radiative transfer for a single time slice of radiosonde data.

    Parameters
    ----------
    ds_date : xarray.Dataset
        Dataset for one timestamp. Required data variables are ``T`` (K),
        ``rh`` (%), ``p`` (hPa), and ``lwc`` (g m^-3), all on ``height``.
        A ``height`` coordinate (m) is required.
    site : str
        Site name used to resolve configuration for the infrared calculation.
    data_dir : str
        Base directory containing all openMWR-managed data.
    freqs : numpy.ndarray
        Microwave channel frequencies.
    angles : numpy.ndarray
        Elevation angles in degrees.
    freq_shift : numpy.ndarray, optional
        Frequency offsets to subtract from ``freqs`` before the calculation.
    absmdl : str, optional
        Absorption model passed to pyrtlib.

    Returns
    -------
    xarray.Dataset
        Dataset with added ``TB`` (microwave) and ``TB_IR`` (infrared) fields.
    """

    patch_pyrtlib_numpy_compat()

    z_m = ds_date.height.values
    z_km = z_m / 1000
    T_K = ds_date.T.values
    rh_100 = ds_date.rh.values
    p_hPa = ds_date.p.values
    lwc_gpm3 = ds_date.lwc.values

    if np.any(rh_100 < 0):
        raise ValueError(f"Relative humidity has negative values: {rh_100}")

    #Cloud top and base indices for pyrtlib
    i_top, i_base = _get_cloud_top_base(lwc_gpm3)
    
    #Initialize the RT model
    if freq_shift is not None:
        used_freqs = freqs - freq_shift
    else:
        used_freqs = freqs

    rh_1 = rh_100 / 100

    rte = TbCloudRTE(z_km, p_hPa, T_K, rh_1, used_freqs, angles)
    rte.satellite = False

    if len(i_top) != 0:
        rte.cloudy = True
        rte.beglev = i_base
        rte.endlev = i_top
        rte.denice = np.zeros(z_km.shape)
        rte.denliq = lwc_gpm3

    #run the RT model and get TBs
    rte.init_absmdl(absmdl)
    df_rt = rte.execute()
    
    df_tb = df_rt.pivot(columns="angle", values="tbtotal")

    ds_date['TB'] = xr.DataArray(df_tb.values, dims=['frq', 'ang'], coords={"frq": freqs, "ang": angles})

    #IR with libRadtran    
    ds_date['TB_IR'] = _run_IR_date(ds_date, site, data_dir)

    return ds_date

def run_pyrtlib(ds_RS, site, data_dir, freqs, angles, freq_shift, absmdl, num_of_processes: int = 12):
    """
    Run pyrtlib-based radiative transfer for one or many profiles.

    Parameters
    ----------
    ds_RS : xarray.Dataset
        Input dataset for pyrtlib. Required data variables are ``T`` (K),
        ``rh`` (%), ``p`` (hPa), and ``lwc`` (g m^-3), defined on ``height``.
        A ``height`` coordinate (m) is required. If multiple profiles are
        provided, a ``time`` dimension is expected.
    site : str
        Site name used for infrared calculations.
    data_dir : str
        Base directory containing all openMWR-managed data.
    freqs : numpy.ndarray
        Microwave channel frequencies.
    angles : numpy.ndarray
        Elevation angles in degrees.
    freq_shift : numpy.ndarray or None
        Optional per-channel frequency offsets.
    absmdl : str
        Absorption model identifier for pyrtlib.
    num_of_processes : int, optional
        Number of worker processes when multiple time steps are present.

    Returns
    -------
    xarray.Dataset
        Dataset with added ``TB`` and ``TB_IR``.
    """

    if 'time' in ds_RS.dims and ds_RS.sizes['time'] > 1:
        ds_new = run_pool(ds_RS, num_of_processes, _run_pyrtlib_date, site, data_dir, freqs, angles, freq_shift, absmdl)
    else:
        ds_new = _run_pyrtlib_date(ds_RS, site, data_dir, freqs, angles, freq_shift, absmdl)

    return ds_new

def run_torchMWRT(ds_RS, site, data_dir, freqs, angles, freq_shift, absmdl, num_of_processes: int = 12):
    """
    Run torchMWRT for microwave TB and libRadtran for infrared TB.

    Parameters
    ----------
    ds_RS : xarray.Dataset
        Input dataset for torchMWRT. Required data variables are ``T`` (K),
        ``rh`` (%), ``p`` (hPa), and ``lwc`` (g m^-3), defined on ``height``.
        A ``height`` coordinate (m) and a ``time`` dimension are required.
    site : str
        Site name used for infrared calculations.
    data_dir : str
        Base directory containing all openMWR-managed data.
    freqs : numpy.ndarray
        Microwave channel frequencies.
    angles : numpy.ndarray
        Elevation angles in degrees.
    freq_shift : numpy.ndarray or None
        Optional per-channel frequency offsets.
    absmdl : str
        Absorption model identifier for torchMWRT.
    num_of_processes : int, optional
        Number of processes for the infrared parallel pass.

    Returns
    -------
    xarray.Dataset
        Copy of ``ds_RS`` with added ``TB`` and ``TB_IR``.
    """

    used_freqs = freqs - np.asarray(freq_shift, dtype=float) if freq_shift is not None else freqs

    if np.any(ds_RS["rh"].values < 0):
        raise ValueError("Relative humidity contains negative values.")
    
    logger.info(f"Running torchMWRT for {ds_RS.time.size} profiles")

    rtmodel = RTModel(freqs=used_freqs, angles=angles, absmdl=absmdl)

    n_times = ds_RS.sizes["time"]
    chunk_size = 200
    tb_chunks = []

    for start in range(0, n_times, chunk_size):
        stop = min(start + chunk_size, n_times)
        logger.info(f"Running torchMWRT chunk {start}:{stop} of {n_times}")

        ds_RS_chunk = ds_RS.isel(time=slice(start, stop))

        atm_profile = AtmProfile(
            temperature=ds_RS_chunk["T"],
            height=ds_RS_chunk["height"],
            pressure=ds_RS_chunk["p"],
            rh=ds_RS_chunk["rh"] / 100.0,
            lwc=ds_RS_chunk["lwc"],
        )

        tb_chunk_ds = rtmodel.execute(atm_profile, return_ds=True)
        tb_chunk_da = tb_chunk_ds["tbtotal"]

        tb_chunks.append(tb_chunk_da)

    logger.info("Finished torchMWRT forward calculation")

    tb_da = xr.concat(tb_chunks, dim="time")#.sel(time=ds_RS.time)

    ds_new = ds_RS.copy()
    ds_new["TB"] = tb_da

    n_proc = max(1, min(int(num_of_processes), n_times))

    logger.info(f"Running IR calculation in parallel with {n_proc} processes")
    TB_IR = run_pool(ds_RS, n_proc, _run_IR_date, site, data_dir)

    logger.info("Finished IR calculation")
    
    ds_new["TB_IR"] = TB_IR

    return ds_new

def run_RT(ds: xr.Dataset, site: str, data_dir: str, freqs: np.ndarray, angles: np.ndarray = np.array([90.]), num_of_processes: int = 12,
           freq_shift: np.ndarray = None, absmdl: str = 'R20', RT_model: str = 'torchMWRT'):
    """
    Run forward radiative transfer with the selected backend.

    Parameters
    ----------
    ds : xarray.Dataset
        Input atmospheric profiles. Required data variables are ``T`` (K),
        ``rh`` (%), ``p`` (hPa), and ``lwc`` (g m^-3), defined on ``height``.
        A ``height`` coordinate (m) is required. For multi-profile execution,
        a ``time`` dimension is expected.
    site : str
        Site name used for infrared calculations.
    data_dir : str
        Base directory containing all openMWR-managed data.
    freqs : numpy.ndarray
        Microwave channel frequencies.
    angles : numpy.ndarray, optional
        Elevation angles in degrees.
    num_of_processes : int, optional
        Number of worker processes for parallel parts of the workflow.
    freq_shift : numpy.ndarray, optional
        Per-channel frequency offsets to subtract from ``freqs``.
    absmdl : str, optional
        Absorption model name used by the selected RT backend.
    RT_model : {"torchMWRT", "pyrtlib"}, optional
        Radiative transfer backend.

    Returns
    -------
    xarray.Dataset
        Dataset with forward-calculated ``TB`` and ``TB_IR``.
    """

    logger.info(f'strating RT for {ds.time.size} profiles with model {RT_model}')
    logger.info(f'Number of Processes: {num_of_processes}')

    if RT_model == 'pyrtlib':
        ds_new = run_pyrtlib(ds, site, data_dir, freqs, angles, freq_shift, absmdl, num_of_processes=num_of_processes)
    elif RT_model == 'torchMWRT':
        ds_new = run_torchMWRT(ds, site, data_dir, freqs, angles, freq_shift, absmdl, num_of_processes=num_of_processes)
    else:
        raise ValueError(f"Unknown RT model: {RT_model}")
    
    if freq_shift is not None:
        ds_new.attrs['freq_shift'] = freq_shift.tolist()

    logger.info('Finished RT')

    # Remove temporary libRadtran files
    libRadtran_data_dir = libRadtran_dir(data_dir)
    if os.path.isdir(libRadtran_data_dir):
        remove_files_in_dir(libRadtran_data_dir)

    return ds_new
