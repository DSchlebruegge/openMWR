"""Explicitly regenerate independent PyRTlib reference values; never called by pytest.

Run from the repository: python -m tests.tools.generate_rt_reference
Review every changed value and the recorded dependency/input provenance.
"""

from hashlib import sha256
from importlib.metadata import version
import json

import numpy as np
from pyrtlib.tb_spectrum import TbCloudRTE

from tests.fixtures.radiative_transfer import DATA, FREQUENCIES, profile_inputs
from torchMWRT.lineshape import H2OLL, O2LL


def pyrtlib_brightness(case):
    inputs = profile_inputs(case)
    z = inputs['height'].values / 1000  # PyRTlib requires km; AtmProfile requires m.
    liquid = inputs.get('lwc')
    liquid = np.zeros_like(z) if liquid is None else liquid.values
    ice = inputs.get('iwc')
    ice = np.zeros_like(z) if ice is None else ice.values
    cloud = liquid > 0
    starts = np.where(np.diff(cloud.astype(int), prepend=0) == 1)[0]
    ends = np.where(np.diff(cloud.astype(int)) == -1)[0] + 1
    rte = TbCloudRTE(z, inputs['pressure'].values, inputs['temperature'].values,
                    inputs['rh'].values, np.array(FREQUENCIES), np.array(case['angles']),
                    ray_tracing=case.get('ray_tracing', False),
                    from_sat=case.get('from_sat', False), cloudy=bool(cloud.any()))
    rte.satellite = case.get('from_sat', False)
    rte.beglev, rte.endlev = starts, ends
    rte.denliq, rte.denice = liquid, ice
    rte.init_absmdl(case['absorption_model'])
    result = rte.execute().pivot(columns='angle', values='tbtotal')
    return result.reindex(columns=case['angles']).values


def cases():
    model = '20240427_model.nc'
    configurations = [
        (11, [90.], 'R17', {}),
        (12, [90., 50.], 'R17', {}),
        (24, [60.], 'R17', {}),
        (6, [60.], 'R98', {}),
        (12, [50.], 'R17', {'ray_tracing': True}),
        (19, [30., 50., 70.], 'R17', {}),
        (5, [90.], 'R17', {'clear': True}),
        (12, [20.], 'R17', {'from_sat': True}),
        (11, [90.], 'R24', {}),
    ]
    configurations += [(11, [90.], name, {'clear': True})
                       for name in H2OLL.available_models() if name in O2LL.available_models()]
    # Three distinct profiles prove batch handling for straight and curved paths.
    configurations += [(i, [90., 50.], 'R17', {'ray_tracing': ray})
                       for ray in (False, True) for i in (11, 12, 24)]
    result = [dict(source=model, time_idx=i, angles=angles, absorption_model=name, **flags)
              for i, angles, name, flags in configurations]
    result += [dict(source=source, angles=[angle], absorption_model='R17')
               for source, angle in [('radiosonde.nc', 90.), ('radiosonde_2.nc', 4.2)]]
    # Configuration overlap is intentional above; store each physical case only once.
    result = [{key: value for key, value in case.items()
               if key not in ('ray_tracing', 'from_sat', 'clear') or value}
              for case in result]
    return list({json.dumps(case, sort_keys=True): case for case in result}.values())


if __name__ == '__main__':
    values = cases()
    for i, case in enumerate(values):
        case['tb_K'] = pyrtlib_brightness(case).tolist()
        print(f'{i + 1}/{len(values)}: {case["source"]} {case["absorption_model"]} {case["angles"]}')
    reference = {
        'generator': 'python -m tests.tools.generate_rt_reference',
        'packages': {name: version(name) for name in ('pyrtlib', 'numpy')},
        'inputs_sha256': {source: sha256((DATA / source).read_bytes()).hexdigest()
                          for source in sorted({case['source'] for case in values})},
        'units': 'brightness temperature in K; rows = frequencies, columns = angles',
        'frequencies_GHz': FREQUENCIES,
        'cases': values,
    }
    (DATA / 'rt_reference.json').write_text(json.dumps(reference, indent=2) + '\n')
