"""Decode stored RPG retrieval products, without applying retrievals or QA.

Layouts follow RPG's Instrument Operation and Software Guide (27 April 2023),
Appendix A1/A2/A8/A9/A11/A12, pp. 141–155:
https://www.radiometer-physics.de/download/PDF/Radiometers/HATPRO/RPG_MWR_STD_Software_Manual%20G5_2023.pdf

Records are packed little-endian. Modern TPC/HPC records append pointing and
astronomical coordinates. HPC relative humidity, when present, is a separate
block with its own min/max and time/rain fields, without pointing coordinates.
"""

import struct

import numpy as np


_CODES = {
    'LWP': (934501978, 934501000),
    'IWV': (594811068, 594811000),
    'TPC': (780798065, 780798066),
    'TPB': (459769847,),
    'HPC': (117343672, 117343673, 117343674, 117343675),
    'LPR': (4567,),
}


def read_rpg_product(path, product):
    """Return header, time, values and optional relative_humidity in file order.

    Values retain native RPG units. Naming variables and attaching units and
    dimensions belong to hatpro_io for both input formats.
    """
    content = path.read_bytes()
    if len(content) < 24:
        raise ValueError(f'{path}: incomplete RPG header (need at least 24 bytes).')
    code, count, _, _, reference, retrieval = struct.unpack_from('<iiffii', content)
    if code not in _CODES[product]:
        raise ValueError(f'{path}: {product} file code {code} is not supported.')
    if count <= 0:
        raise ValueError(f'{path}: number of records must be positive, got {count}.')
    header = {'_code': code, '_time_ref': reference, '_retrieval': retrieval}
    fields = [('time', '<i4'), ('rain', 'u1')]
    column = product in ('LWP', 'IWV')
    if column:
        offset = 24
        fields += [('values', '<f4'), ('angles', '<f4' if code == _CODES[product][0] else '<i4')]
    else:
        if len(content) < 28:
            raise ValueError(f'{path}: incomplete profile header.')
        levels, = struct.unpack_from('<i', content, 24)
        offset = 28 + 4 * levels
        if levels <= 0 or offset > len(content):
            raise ValueError(f'{path}: invalid or incomplete height grid ({levels} levels).')
        heights = np.frombuffer(content, dtype='<i4', count=levels, offset=28).copy()
        if np.any(heights < 0) or np.any(np.diff(heights.astype('int64')) <= 0):
            raise ValueError(f'{path}: heights must be nonnegative and strictly increasing.')
        header['_height'] = heights
        fields += [('values', '<f4', (levels,))]
        if code in (780798066, 117343674, 117343675):
            fields += [('angles', '<i4'), ('right_ascension', '<f4'), ('declination', '<f4')]
    dtype = np.dtype(fields)
    has_rh = code in (117343673, 117343675)
    rh_dtype = np.dtype(fields[:3])  # RH has no pointing fields.
    expected = offset + count * dtype.itemsize
    if has_rh:
        expected += 8 + count * rh_dtype.itemsize
    if len(content) != expected:
        raise ValueError(f'{path}: inconsistent records: header declares {count}, '
                         f'expected {expected} bytes, found {len(content)}.')
    records = np.frombuffer(content, dtype=dtype, count=count, offset=offset)
    values = {'time': records['time'].copy(), 'values': records['values'].copy()}
    if has_rh:
        rh_offset = offset + count * dtype.itemsize + 8
        humidity = np.frombuffer(content, dtype=rh_dtype, count=count, offset=rh_offset)
        # The two blocks describe the same N profiles. RPG's NetCDF export pairs
        # them by record index on the absolute-humidity timestamps. The appended
        # RH timestamps can be slightly later (observed up to 21 seconds).
        values['relative_humidity'] = humidity['values'].copy()
    return header, values
