"""Extract small paired RPG files from the measured sources in provenance.json.

Run deliberately with ``python -m tests.tools.generate_hatpro_rpg_reference``.
Binary record bytes and manufacturer NetCDF values are copied independently.
This tool does not call the reader being tested.
"""

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import xarray as xr


DATA = Path(__file__).resolve().parents[1] / 'data/hatpro'


def generate():
    provenance = json.loads((DATA / 'provenance.json').read_text())
    for variant, entry in provenance.items():
        target = DATA / variant / 'rpg'
        target.mkdir(exist_ok=True)
        comparison_times = []
        for product in ('LWP', 'IWV', 'TPC', 'TPB', 'HPC', 'LPR'):
            source = Path(entry['source'] + '.' + product)
            content = source.read_bytes()
            code, count = struct.unpack_from('<2i', content)
            header_size, record_size = 24, 13
            if product not in ('LWP', 'IWV'):
                levels, = struct.unpack_from('<i', content, 24)
                header_size, record_size = 28 + 4 * levels, 5 + 4 * levels
            rh_size = record_size
            if code in (780798066, 117343674, 117343675):
                record_size += 12
            indices = sorted(set([*range(8), count // 3, count // 2, *range(count - 3, count)]))
            if variant == 'modern' and product == 'HPC':
                # Include measured differences between AH/RH block timestamps.
                indices = sorted(set(indices + [137, 719, 816, 1062]))
            header = bytearray(content[:header_size])
            struct.pack_into('<i', header, 4, len(indices))
            subset = header + b''.join(content[header_size + i * record_size:
                                              header_size + (i + 1) * record_size] for i in indices)
            if code in (117343673, 117343675):
                offset = header_size + count * record_size
                subset += content[offset:offset + 8]
                subset += b''.join(content[offset + 8 + i * rh_size:
                                           offset + 8 + (i + 1) * rh_size] for i in indices)
            (target / source.name).write_bytes(subset)
            nc = Path(str(source) + '.NC')
            original = xr.load_dataset(nc, decode_timedelta=False)
            assert original.sizes['time'] == count, source
            original.isel(time=indices).to_netcdf(target / nc.name)
            comparison_times.extend(original.time.values[indices])
            entry['products'][product] = {
                'code': code, 'records': indices,
                'binary_sha256': hashlib.sha256(content).hexdigest(),
                'netcdf_sha256': hashlib.sha256(nc.read_bytes()).hexdigest(),
            }
        # Keep measured BRT samples at comparison times too. Daily zenith
        # selection uses BRT pointing, so profiles alone would be discarded.
        source = Path(entry['source'] + '.BRT')
        content = source.read_bytes()
        _, _, _, frequencies = struct.unpack_from('<4i', content)
        header_size, record_size = 16 + 12 * frequencies, 9 + 4 * frequencies
        original = xr.load_dataset(str(source) + '.NC', decode_timedelta=False)
        indices = sorted(set(entry['products']['BRT']['records']) |
                         set(np.flatnonzero(np.isin(original.time.values, comparison_times)).tolist()))
        header = bytearray(content[:header_size])
        struct.pack_into('<i', header, 4, len(indices))
        subset = header + b''.join(content[header_size + i * record_size:
                                          header_size + (i + 1) * record_size] for i in indices)
        (DATA / variant / source.name).write_bytes(subset)
        original.isel(time=indices).to_netcdf(DATA / variant / (source.name + '.NC'))
        entry['products']['BRT']['records'] = indices
    (DATA / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    generate()
