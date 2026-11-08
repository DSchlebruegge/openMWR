import xarray as xr
import numpy as np
import pandas as pd
import tempfile
import logging
from pathlib import Path
import warnings

from pyrtlib.tb_spectrum import TbCloudRTE
from torchMWRT import RTModel, AtmProfile

from openMWR.libRadtran import get_libradtran_paths, libRadtran
from openMWR.parallel import run_pool
from openMWR.config import check_infrared
from openMWR.site import load_site_config
from openMWR.paths import libRadtran_dir
from openMWR.xr_utils import valid_profile_pressure

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

def calculate_IR_band_TB(site, project_dir, z_km, T_K, rh_100, p_hPa, lwc_gpm3, effective_droplet_radius_mum, *, ang, surface_altitude, site_config=None):
    """
    Calculate band-integrated infrared brightness temperatures with libRadtran.

    Parameters
    ----------
    site : str
        Site name used to read configuration (determines wavelength band).
    project_dir : str
        Working-project root containing data/ and sites/.
    z_km : array-like
        Heights in kilometers above the profile base, strictly increasing.
    T_K : array-like
        Air temperature profile in kelvin.
    rh_100 : array-like
        Relative humidity in percent.
    p_hPa : array-like
        Pressure profile in hectopascals, strictly decreasing with height.
    lwc_gpm3 : array-like
        Liquid water content in g m^-3.
    effective_droplet_radius_mum : array-like
        Effective droplet radius in micrometers at each level.
    ang : array-like, optional
        Elevation angles in degrees; zenith-only by default.
    surface_altitude : float
        Altitude of the profile base in metres above sea level. This is the
        launch-station altitude for uncut radiosondes, or the instrument
        altitude for profiles rebased to the instrument.

    Returns
    -------
    float or pandas.Series
        Brightness temperatures for the requested angles.

    Notes
    -----
    Temporary libRadtran input files are written to ``data/libRadtran`` and
    removed after execution. The wavelength band is selected based on the
    site's Hatpro generation. Atmosphere and cloud files use the supplied
    geometric heights above sea level. Background trace gases follow the
    U.S. standard atmosphere, and libRadtran converts water-relative RH.
    """
    gen = load_site_config(site, project_dir, config=site_config).require('instrument_generation')
    _, installation_data_dir = get_libradtran_paths()
    background_file = installation_data_dir / "atmmod" / "afglus.dat"
    if not background_file.is_file():
        raise FileNotFoundError(f"libRadtran standard atmosphere not found at: {background_file}")

    libRadtran_data_dir = Path(libRadtran_dir(project_dir)).resolve()
    libRadtran_data_dir.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(surface_altitude):
        raise ValueError('surface_altitude must be finite (metres above sea level).')
    
    df = pd.DataFrame({
        'z_km': z_km,
        'T_K': T_K,
        'rh_100': rh_100,
        'p_hPa': p_hPa,
        'lwc_gpm3': lwc_gpm3,
        'effective_droplet_radius_mum': effective_droplet_radius_mum,
    })
    
    if (len(df) < 2 or not np.isfinite(df['z_km']).all()
            or not (np.diff(df['z_km']) > 0).all()):
        raise ValueError('Infrared RT requires at least two finite, strictly increasing heights.')
    if (not np.isfinite(df['p_hPa']).all() or not (df['p_hPa'] > 0).all()
            or not (np.diff(df['p_hPa']) < 0).all()):
        raise ValueError('Infrared RT requires positive pressure strictly decreasing with height. '
                         'Reject inconsistent profiles during preparation instead of sorting by pressure.')

    altitude_km = surface_altitude / 1000
    bottom_altitude_km = altitude_km + df.iloc[0]['z_km']
    cloud = df.loc[df['z_km'] < 15, ['z_km', 'lwc_gpm3', 'effective_droplet_radius_mum']].copy()
    df['z_km'] += altitude_km
    cloud['z_km'] += altitude_km

    z = df['z_km'].to_numpy()
    background = np.loadtxt(background_file)[::-1]  # AFGL columns are top-down.
    if z[0] < background[0, 0] or z[-1] > background[-1, 0]:
        raise ValueError('Infrared profile heights must lie within the libRadtran standard atmosphere.')
    air = df['p_hPa'].to_numpy() * 1e-4 / (1.380649e-23 * df['T_K'].to_numpy())  # cm^-3
    background_air = np.exp(np.interp(z, background[:, 0], np.log(background[:, 3])))
    # Match libRadtran 2.0.6 read_radiosonde_file and default LINMIX interpolation.
    # O2/CO2 use the new air density. O3/NO2 retain the background air density.
    gases = [np.interp(z, background[:, 0], background[:, column] / background[:, 3])
             * (air if column in (5, 7) else background_air) for column in (4, 5, 7, 8)]
    # atmosphere_file columns: z, p, T, air, O3, O2, H2O, CO2, NO2.
    atmosphere = np.column_stack([z, df['p_hPa'], df['T_K'], air, gases[0], gases[1],
                                  np.zeros_like(z), gases[2], gases[3]])

    umu = np.cos(np.radians(90 + ang))

    if gen == 'G5':
        wavelengths = '9600 11500'
    elif gen == 'G4':
        wavelengths = '9200 10600'
    
    # Each invocation owns its scratch files, including concurrent site builds.
    with tempfile.TemporaryDirectory(dir=libRadtran_data_dir) as directory:
        atmosphere_file = Path(directory) / 'atmosphere.dat'
        humidity_file = Path(directory) / 'humidity.dat'
        wc_file = Path(directory) / 'cloud.dat'
        # H2O is supplied separately in RH units to retain libRadtran's native
        # saturation formula. CH4/N2O/CO/N2 retain its standard gas files.
        np.savetxt(atmosphere_file, atmosphere[::-1], fmt='%.12g')
        np.savetxt(humidity_file, df[['z_km', 'rh_100']].to_numpy()[::-1], fmt='%.12g')
        cloud.iloc[::-1].to_csv(wc_file, index=False, sep=' ', header=False)
        df_rad, stderr = libRadtran(
            atmosphere_file = str(atmosphere_file),
            data_files_path = str(installation_data_dir),
            source = 'thermal',
            mol_abs_param =  'reptran medium',
            wavelength = wavelengths,
            output_process = 'integrate',
            mol_file = f'H2O {humidity_file} RH',
            z_interpolate = 'linmix',
            wc_file = f'1D {str(wc_file)}',
            altitude = bottom_altitude_km,
            zout = 0,
            umu = umu[::-1], # -0.5 np.array([-1, -0.5]
            output_quantity = 'brightness',
            output_user = 'uu',
            quiet = True
            )

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
    if cloud[-1]:
        i_top = np.append(i_top, len(cloud) - 1)
    
    return i_top, i_base

def _effective_droplet_radius_mum(lwc_gpm3: np.ndarray, *, droplet_number_density, min_effective_radius) -> np.ndarray:
    """
    Estimate droplet effective radius from liquid water content.

    Parameters
    ----------
    lwc_gpm3 : numpy.ndarray
        Liquid water content profile in g m^-3.

    Returns
    -------
    numpy.ndarray
        Effective droplet radius in micrometers, clipped to min_effective_radius.
    """
    rho_w = 1e6  # g/m^3
    effective_droplet_radius_mum = np.cbrt(lwc_gpm3 / (droplet_number_density * 4 / 3 * np.pi * rho_w)) * 1e6
    effective_droplet_radius_mum = np.maximum(effective_droplet_radius_mum, min_effective_radius)
    if np.any(effective_droplet_radius_mum >= 60):
        raise ValueError("Calculated droplet effective radii must be below 60 micrometres for libRadtran's Hu-Stamnes cloud optics. Check droplet_number_density and liquid water content.")
    return effective_droplet_radius_mum

def _run_IR_date(ds_date: xr.Dataset, site: str, project_dir: str, infrared: dict, site_config) -> xr.DataArray:
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
    project_dir : str
        Working-project root containing data/ and sites/.

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

    effective_droplet_radius_mum = _effective_droplet_radius_mum(lwc_gpm3, **infrared)
    TB_IR = calculate_IR_band_TB(site, project_dir, z_km, T_K, rh_100, p_hPa, lwc_gpm3, effective_droplet_radius_mum,
                               ang=np.array([90.]), surface_altitude=ds_date.surface_altitude.item(), site_config=site_config)
    return xr.DataArray(TB_IR)

def _rt_context(ds, angles):
    """Describe the input selection when a microwave calculation fails."""
    altitude = np.asarray(ds.surface_altitude)
    context = (f'profile-base altitude {altitude.min():g} to {altitude.max():g} m ASL, '
               f'elevation angles {np.asarray(angles).tolist()} degrees')
    if 'time' in ds:
        times = np.asarray(ds.time)
        context = f'UTC time {times.min()} to {times.max()}, {context}'
    return context


def _run_pyrtlib_date(ds_date, site, project_dir, freqs, angles, freq_shift, absmdl, infrared, ray_tracing, site_config):
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
    project_dir : str
        Working-project root containing data/ and sites/.
    freqs : numpy.ndarray
        Microwave channel frequencies.
    angles : numpy.ndarray
        Elevation angles in degrees.
    freq_shift : numpy.ndarray, optional
        Frequency offsets to subtract from ``freqs`` before the calculation.
    absmdl : str
        Absorption model passed to pyrtlib.

    Returns
    -------
    xarray.Dataset
        Dataset with added ``TB`` (microwave) and ``TB_IR`` (infrared) fields.
    """

    z_m = ds_date.height.values + ds_date.surface_altitude.item()
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

    rte = TbCloudRTE(z_km, p_hPa, T_K, rh_1, used_freqs, angles, ray_tracing=ray_tracing)
    rte.satellite = False

    if len(i_top) != 0:
        rte.cloudy = True
        rte.beglev = i_base
        rte.endlev = i_top
        rte.denice = np.zeros(z_km.shape)
        rte.denliq = lwc_gpm3

    #run the RT model and get TBs
    rte.init_absmdl(absmdl)
    # PyRTlib can otherwise warn and return an incomplete ray on ducting.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message=r'ray_tracing:.*', category=UserWarning)
            df_rt = rte.execute()
    except (ValueError, UserWarning) as exc:
        raise ValueError(f'PyRTlib failed ({_rt_context(ds_date, angles)}): {exc}') from exc
    
    df_tb = df_rt.pivot(columns="angle", values="tbtotal").reindex(columns=angles)

    ds_date['TB'] = xr.DataArray(df_tb.values, dims=['frq', 'ang'], coords={"frq": freqs, "ang": angles})

    #IR with libRadtran    
    ds_date['TB_IR'] = _run_IR_date(ds_date, site, project_dir, infrared, site_config)

    return ds_date

def run_pyrtlib(ds_RS, site, project_dir, freqs, angles, freq_shift, absmdl, num_of_processes: int = 12, *, infrared, ray_tracing, site_config=None):
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
    project_dir : str
        Working-project root containing data/ and sites/.
    freqs : numpy.ndarray
        Microwave channel frequencies.
    angles : numpy.ndarray
        Elevation angles in degrees.
    freq_shift : numpy.ndarray or None
        Optional per-channel frequency offsets.
    absmdl : str
        Absorption model identifier for pyrtlib.
    ray_tracing : bool
        Use spherical refractive microwave paths, or plane-parallel geometry if False.
    num_of_processes : int, optional
        Number of worker processes when multiple time steps are present.

    Returns
    -------
    xarray.Dataset
        Dataset with added ``TB`` and ``TB_IR``.
    """

    site_config = load_site_config(site, project_dir, config=site_config)
    site_config.require('instrument_generation')
    if 'time' in ds_RS.dims:
        try:
            ds_new = run_pool(ds_RS, num_of_processes, _run_pyrtlib_date, site, project_dir, freqs, angles, freq_shift, absmdl, infrared, ray_tracing, site_config)
        except ValueError as exc:
            raise ValueError(f'PyRTlib failed ({_rt_context(ds_RS, angles)}): {exc}') from exc
    else:
        ds_new = _run_pyrtlib_date(ds_RS, site, project_dir, freqs, angles, freq_shift, absmdl, infrared, ray_tracing, site_config)

    return ds_new

def run_torchMWRT(ds_RS, site, project_dir, freqs, angles, freq_shift, absmdl, num_of_processes: int = 12, *, infrared, ray_tracing, site_config=None):
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
    project_dir : str
        Working-project root containing data/ and sites/.
    freqs : numpy.ndarray
        Microwave channel frequencies.
    angles : numpy.ndarray
        Elevation angles in degrees.
    freq_shift : numpy.ndarray or None
        Optional per-channel frequency offsets.
    absmdl : str
        Absorption model identifier for torchMWRT.
    ray_tracing : bool
        Use spherical refractive microwave paths, or plane-parallel geometry if False.
    num_of_processes : int, optional
        Number of processes for the infrared parallel pass.

    Returns
    -------
    xarray.Dataset
        Copy of ``ds_RS`` with added ``TB`` and ``TB_IR``.
    """

    site_config = load_site_config(site, project_dir, config=site_config)
    site_config.require('instrument_generation')
    used_freqs = freqs - np.asarray(freq_shift, dtype=float) if freq_shift is not None else freqs

    if np.any(ds_RS["rh"].values < 0):
        raise ValueError("Relative humidity contains negative values.")
    
    logger.info(f"Running torchMWRT for {ds_RS.time.size} profiles")

    rtmodel = RTModel(freqs=used_freqs, angles=angles, absmdl=absmdl, ray_tracing=ray_tracing)

    n_times = ds_RS.sizes["time"]
    chunk_size = 200
    tb_chunks = []

    for start in range(0, n_times, chunk_size):
        stop = min(start + chunk_size, n_times)
        logger.info(f"Running torchMWRT chunk {start}:{stop} of {n_times}")

        ds_RS_chunk = ds_RS.isel(time=slice(start, stop))

        # AtmProfile uses one absolute height grid per batch. Keep distinct
        # profile-base altitudes separate without changing the input order.
        for altitude in np.unique(ds_RS_chunk.surface_altitude):
            group = ds_RS_chunk
            if group.surface_altitude.ndim:
                group = group.sel(time=group.surface_altitude == altitude)
            atm_profile = AtmProfile(
                temperature=group['T'], height=group.height + altitude,
                pressure=group['p'], rh=group['rh'] / 100.0, lwc=group['lwc'])
            try:
                tb_chunks.append(rtmodel.execute(atm_profile, return_ds=True)['tbtotal'])
            except ValueError as exc:
                raise ValueError(f'torchMWRT failed ({_rt_context(group, angles)}): {exc}') from exc

    logger.info("Finished torchMWRT forward calculation")

    tb_da = xr.concat(tb_chunks, dim="time").sel(time=ds_RS.time)

    ds_new = ds_RS.copy()
    ds_new["TB"] = tb_da

    n_proc = max(1, min(int(num_of_processes), n_times))

    logger.info(f"Running IR calculation in parallel with {n_proc} processes")
    TB_IR = run_pool(ds_RS, n_proc, _run_IR_date, site, project_dir, infrared, site_config)

    logger.info("Finished IR calculation")
    
    ds_new["TB_IR"] = TB_IR

    return ds_new

def run_RT(ds: xr.Dataset, site: str, project_dir: str, freqs: np.ndarray, angles: np.ndarray, num_of_processes: int = 12,
           freq_shift: np.ndarray = None, *, absmdl: str, RT_model: str, infrared, ray_tracing: bool, site_config=None):
    """
    Run forward radiative transfer with the selected backend.

    Parameters
    ----------
    ds : xarray.Dataset
        Input atmospheric profiles. Required data variables are ``T`` (K),
        ``rh`` (%), ``p`` (hPa), and ``lwc`` (g m^-3), defined on ``height``.
        A ``height`` coordinate (m) is required. For multi-profile execution,
        a ``time`` dimension is expected. ``surface_altitude`` gives the profile
        base in metres above sea level, as a scalar or per-time coordinate.
    site : str
        Site name used for infrared calculations.
    project_dir : str
        Working-project root containing data/ and sites/.
    freqs : numpy.ndarray
        Microwave channel frequencies.
    angles : numpy.ndarray
        Elevation angles in degrees.
    num_of_processes : int, optional
        Number of worker processes for parallel parts of the workflow.
    freq_shift : numpy.ndarray, optional
        Per-channel frequency offsets to subtract from ``freqs``.
    absmdl : str
        Absorption model name used by the selected RT backend.
    RT_model : {"torchMWRT", "pyrtlib"}
        Radiative transfer backend.
    ray_tracing : bool
        Use spherical refractive ray paths, or plane-parallel geometry if False.
    infrared : dict
        Explicit droplet_number_density (m^-3) and min_effective_radius
        (micrometres). Atmospheric T, p, rh and lwc must be finite on every level.

    Returns
    -------
    xarray.Dataset
        Dataset with forward-calculated ``TB`` and ``TB_IR``.
    """

    check_infrared(infrared)
    if not isinstance(ray_tracing, bool):
        raise ValueError('ray_tracing must be true or false.')
    if 'surface_altitude' not in ds or not bool(np.isfinite(ds.surface_altitude).all()):
        raise ValueError('RT requires finite surface_altitude (profile-base metres above sea level). '
                         'Rebuild prepared profiles or set this coordinate for custom Python inputs.')
    if ds.surface_altitude.dims not in ((), ('time',)):
        raise ValueError('surface_altitude must be scalar or have only the time dimension. '
                         f'Got dimensions {ds.surface_altitude.dims}.')
    for var in ('T', 'p', 'rh', 'lwc'):
        if not bool(np.isfinite(ds[var]).all()):
            raise ValueError(f'RT requires complete profiles: {var} contains missing or non-finite values.')
    if (ds.sizes['height'] < 2 or not bool(np.isfinite(ds.height).all())
            or not bool((ds.height.diff('height') > 0).all())):
        raise ValueError('RT requires at least two finite, strictly increasing heights.')
    valid_pressure = valid_profile_pressure(ds.p)
    if not bool(valid_pressure.all()):
        invalid = ds.sel(time=~valid_pressure) if valid_pressure.dims == ('time',) else ds
        raise ValueError('RT requires positive pressure strictly decreasing with height '
                         f'({_rt_context(invalid, angles)}). Reject inconsistent profiles during preparation.')
    # Radius increases with LWC; check the maximum before starting RT or workers.
    _effective_droplet_radius_mum(ds.lwc.max().item(), **infrared)
    logger.info(f'Starting RT for {ds.time.size} profiles with model {RT_model}')

    if RT_model == 'pyrtlib':
        ds_new = run_pyrtlib(ds, site, project_dir, freqs, angles, freq_shift, absmdl, num_of_processes=num_of_processes, infrared=infrared, ray_tracing=ray_tracing, site_config=site_config)
    elif RT_model == 'torchMWRT':
        ds_new = run_torchMWRT(ds, site, project_dir, freqs, angles, freq_shift, absmdl, num_of_processes=num_of_processes, infrared=infrared, ray_tracing=ray_tracing, site_config=site_config)
    else:
        raise ValueError(f"Unknown RT model: {RT_model}")
    
    if freq_shift is not None:
        ds_new.attrs['freq_shift'] = freq_shift.tolist()

    if 'time' in ds.dims:
        missing = np.setdiff1d(ds.time.values, ds_new.time.values)
        if missing.size:
            raise ValueError(f'RT failed to produce complete results for {missing.size} profiles '
                             f'({_rt_context(ds.sel(time=missing), angles)}). See the worker errors above.')
    for variable in ('TB', 'TB_IR'):
        if not bool(np.isfinite(ds_new[variable]).all()):
            raise ValueError(f'RT produced non-finite {variable}; incomplete results cannot be saved.')
    ds_new.attrs['ray_tracing'] = int(ray_tracing)
    ds_new.attrs['infrared_atmosphere'] = 'geometric_height'

    logger.info('Finished RT')

    return ds_new
