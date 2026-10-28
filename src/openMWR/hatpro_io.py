"""Read RPG NetCDF and MWRpy binary observations into common variables and units.

File reading preserves sample order and duplicate timestamps. Daily importers
own duplicate handling, validity masks, matching and averaging.
"""

from datetime import timedelta
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from openMWR.config import check_number
from openMWR.site import get_config_parameter

logger = logging.getLogger(__name__)

_BINARY_PRODUCTS = {'BRT', 'BLB', 'MET', 'IRT'}
_VARIABLES = {
    'TB': ('TBs',),
    'surface_p': ('env_pressure', 'Surf_P'),
    'surface_T': ('env_temperature', 'Surf_T'),
    'surface_rh': ('env_relative_humidity', 'Surf_RH'),
    'TB_IR': ('IRR_data', 'IRR_Map'),
    'rain_flag': ('rain_flag', 'RF'),
    'azimuth_angle': ('azimuth_angle', 'AziAng', 'AzAng'),
    'elevation_angle': ('elevation_angle', 'ElAng'),
    'lwp': ('LWP_data', 'LWP'),
    'lwc': ('liquid_water_profiles', 'LWC_prof'),
    'T': ('temperature_profiles', 'T_prof'),
    'ah': ('Absolute_Humidity_Profiles', 'AH_Prof'),
    'rh': ('relative_humidity_profiles', 'RH_prof'),
    'iwv': ('IWV_data', 'IWV'),
}
_PRODUCT_VARIABLES = {
    'BRT': ('TB', 'rain_flag', 'azimuth_angle', 'elevation_angle'),
    'BLB': ('TB', 'rain_flag', 'azimuth_angle'),
    'MET': ('surface_p', 'surface_T', 'surface_rh'),
    'IRT': ('TB_IR',),
    'LWP': ('lwp',), 'LPR': ('lwc',), 'TPC': ('T',),
    'HPC': ('ah', 'rh'), 'IWV': ('iwv',), 'TPB': ('T',),
}
_PRODUCT_COORDS = {
    'BRT': ('frq',), 'BLB': ('frq', 'ang'), 'IRT': ('wavelength',),
    'LPR': ('height_lwc',), 'TPC': ('height',),
    'HPC': ('height',), 'TPB': ('height',),
}


def check_hatpro_format(format, time_offset_hours=None):
    """Validate the site-wide input choice used by datasets and prediction."""
    if format not in ('auto', 'netcdf', 'binary'):
        raise ValueError('hatpro_format must be "auto", "netcdf" or "binary".')
    if time_offset_hours is not None:
        check_number(time_offset_hours, 'hatpro_time_offset_hours')
        if abs(time_offset_hours) > 24:
            raise ValueError('hatpro_time_offset_hours must be between -24 and 24.')


def _validate_netcdf(data, path, product):
    """Reject incomplete products before daily matching or averaging starts."""
    variables = _PRODUCT_VARIABLES[product]
    required = set(variables)
    if product == 'BLB':
        required.remove('azimuth_angle')
    elif product == 'MET':
        required.clear()  # A partially available weather station remains usable.
        if not any(name in data for name in variables):
            raise ValueError(f'{path}: no surface meteorology variables found.')
    missing = sorted(required - set(data.data_vars))
    if missing:
        raise ValueError(f'{path}: missing required variables: {", ".join(missing)}.')
    axes = _PRODUCT_COORDS.get(product, ())
    for name in ('time', *axes):
        if name not in data.coords:
            raise ValueError(f'{path}: missing required coordinate {name}.')
        if data[name].dims != (name,):
            raise ValueError(f'{path}: coordinate {name} must be one-dimensional along {name}.')
    if not np.issubdtype(data.time.dtype, np.datetime64):
        raise ValueError(f'{path}: time must contain decoded datetime values.')
    for name in variables:
        if name not in data:
            continue
        dimensions = ('time', *axes) if name in ('TB', 'TB_IR', 'T', 'ah', 'rh', 'lwc') else ('time',)
        if sorted(data[name].dims) != sorted(dimensions):
            raise ValueError(f'{path}: {name} must use dimensions {dimensions}, got {data[name].dims}.')


def _netcdf(path, product):
    data = xr.load_dataset(path, decode_timedelta=False)
    rename = {alias: key for key in _PRODUCT_VARIABLES[product]
              for alias in _VARIABLES[key] if alias in data and alias != key}
    data = data.rename(rename)
    coordinates = {
        'frq': ('frequencies', 'Freq'),
        'ang': ('elevation_scan_angles', 'ElAngs'),
        'height_lwc' if product == 'LPR' else 'height': ('altitude_layers', 'altitude'),
    }
    if product == 'IRT':
        coordinates = {'wavelength': ('frequencies', 'WavLens')}
    for name, aliases in coordinates.items():
        original = next((alias for alias in aliases if alias in data), None)
        if original is not None:
            if data[original].ndim != 1:
                raise ValueError(f'{path}: coordinate {original} must be one-dimensional.')
            dimension, = data[original].dims
            data = data.swap_dims({dimension: original}).rename({original: name})
    _validate_netcdf(data, path, product)
    data = data[[key for key in _PRODUCT_VARIABLES[product] if key in data]]
    if product == 'IRT':
        data['TB_IR'] = (data.TB_IR + 273.15).transpose('time', 'wavelength')
        data.TB_IR.attrs['units'] = 'K'
    elif product == 'BLB':
        # The NetCDF-only 0-degree entry contains surface temperature, not a scan.
        data = data.isel(ang=data.ang != 0)
    return data


def _binary(path, product, time_offset_hours):
    # Import only when binary data is needed. MWRpy owns binary layout decoding.
    from mwrpy.level1 import rpg_bin
    from mwrpy import __version__
    from mwrpy.exceptions import InvalidFileError
    from mwrpy.utils import epoch2unix

    if product not in _BINARY_PRODUCTS:
        raise ValueError(f'{path}: binary {product} is not supported. RPG retrieval '
                         'products require NetCDF.')
    try:
        header, values = getattr(rpg_bin, 'read_' + product.lower())(str(path))
    except (InvalidFileError, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{path}: {exc}') from exc
    reference = int(header['_time_ref'])
    if reference not in (0, 1):
        raise ValueError(f'{path}: unsupported binary time reference {reference}.')
    if reference == 0 and time_offset_hours is None:
        raise ValueError(f'{path}: binary timestamps are local time. Set '
                         'hatpro_time_offset_hours to the explicit local UTC offset.')
    offset = None if time_offset_hours is None else timedelta(hours=time_offset_hours)
    times = epoch2unix(values['time'], reference, time_offset=offset).astype('datetime64[s]').astype('datetime64[ns]')
    data = xr.Dataset(coords={'time': times}, attrs={
        'source_format': 'binary', 'source_file': str(path), 'mwrpy_version': __version__})
    if product in ('BRT', 'BLB'):
        data = data.assign_coords(frq=np.asarray(header['_f']))
        data.frq.attrs['units'] = 'GHz'
        if product == 'BRT':
            data['TB'] = (('time', 'frq'), values['tb'])
            for angle in ('elevation_angle', 'azimuth_angle'):
                data[angle] = ('time', values[angle])
        else:
            data = data.assign_coords(ang=np.asarray(header['_ang']))
            data.ang.attrs['units'] = 'degree'
            data['TB'] = (('time', 'frq', 'ang'), values['tb'])
        # The remaining BLB bits describe scan status, not rainfall.
        data['rain_flag'] = ('time', (values['rain'] & 1).astype(float))
        data.TB.attrs['units'] = 'K'
    elif product == 'MET':
        for name, source, factor, unit in (
            ('surface_p', 'air_pressure', 1, 'hPa'),
            ('surface_T', 'air_temperature', 1, 'K'),
            ('surface_rh', 'relative_humidity', 100, '%'),
        ):
            data[name] = ('time', values[source] * factor)
            data[name].attrs['units'] = unit
    else:
        data = data.assign_coords(wavelength=np.atleast_1d(header['_f']))
        data.wavelength.attrs['units'] = 'um'
        data['TB_IR'] = (('time', 'wavelength'), values['irt'])
        data.TB_IR.attrs['units'] = 'K'
    return data


def read_hatpro_file(path, *, time_offset_hours=None):
    """Read one RPG product without filtering, sorting or removing duplicates.

    NetCDF uses its encoded timestamps. Binary local timestamps require an
    explicit fixed UTC offset in hours (local time minus UTC). Observation units
    are K, hPa, percent, GHz and degrees. IR wavelengths are in micrometres.
    BLB contains only scan angles. The NetCDF-only 0-degree surface-temperature
    channel is removed. Binary BLB has no explicit azimuth field.
    """
    path = Path(path)
    netcdf = path.suffix.upper() == '.NC'
    product = path.with_suffix('').suffix[1:].upper() if netcdf else path.suffix[1:].upper()
    if product not in _PRODUCT_VARIABLES:
        raise ValueError(f'{path}: unsupported HATPRO product {product!r}.')
    check_hatpro_format('netcdf' if netcdf else 'binary', time_offset_hours)
    return _netcdf(path, product) if netcdf else _binary(path, product, time_offset_hours)


def load_hatpro_day(date, site, project_dir, products):
    """Load daily observations and available auxiliary/comparison products.

    BRT and BLB are required when requested. Missing or unreadable auxiliary
    sensors and NetCDF RPG retrievals are logged and omitted independently.
    Discard missing timestamps and dates more than a day outside the file's
    nominal day before matching or averaging. This tolerance retains midnight
    overlaps and local-to-UTC shifts. The interval boundaries are included.
    Retain the first duplicate and sort timestamps for matching and averaging.
    """
    root, format, offset = get_config_parameter(site,
        ['hatpro_data_dir', 'hatpro_format', 'hatpro_time_offset_hours'], project_dir)
    format = 'auto' if format is None else format
    check_hatpro_format(format, offset)
    if root is None:
        raise ValueError('Set hatpro_data_dir in site.toml before reading HATPRO observations.')
    base = Path(root) / date.strftime('Y%Y/M%m/D%d/%y%m%d')
    day = np.datetime64(date.strftime('%Y-%m-%d'))
    earliest, latest = day - np.timedelta64(1, 'D'), day + np.timedelta64(2, 'D')
    datasets = {}
    selected = {}
    for product in products:
        nc, binary = Path(f'{base}.{product}.NC'), Path(f'{base}.{product}')
        # Manufacturer retrievals are available through the NetCDF backend only.
        use_netcdf = product not in _BINARY_PRODUCTS or format == 'netcdf' or (format == 'auto' and nc.is_file())
        path = nc if use_netcdf else binary
        optional = product in _PRODUCT_VARIABLES and product not in ('BRT', 'BLB')
        try:
            data = read_hatpro_file(path, time_offset_hours=offset)
            valid = (data.time.values >= earliest) & (data.time.values <= latest)
            invalid_count = int((~valid).sum())
            if invalid_count:
                logger.warning('Discarding %s samples from %s: missing timestamps or timestamps '
                               'outside [%s, %s].', invalid_count, path, earliest, latest)
                data = data.isel(time=valid)
                if not data.sizes['time']:
                    raise ValueError(f'{path}: no usable timestamps remain in the daily time window.')
            datasets[product] = data.drop_duplicates('time').sortby('time')
            selected[product] = 'netcdf' if use_netcdf else 'binary'
            logger.debug('Read HATPRO %s from %s', 'NetCDF' if use_netcdf else 'binary', path)
        except (OSError, ValueError) as exc:
            if not optional:
                raise
            selected[product] = 'unavailable'
            logger.warning('Optional HATPRO input is unavailable: %s: %s', path, exc)
    logger.info('HATPRO inputs for %s %s: %s', site, date.strftime('%Y-%m-%d'),
                ', '.join(f'{product}={format}' for product, format in selected.items()))
    return datasets
