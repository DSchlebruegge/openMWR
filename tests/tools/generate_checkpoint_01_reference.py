"""Generate the checkpoint reference with the actual 0.1 code, without fitting.

Run with PYTHONPATH=/path/to/openMWR-0.1/src. Do not regenerate references to
make a failing compatibility test pass. Review any change to the expected data.
"""

from pathlib import Path
import tempfile

import numpy as np
import torch
import xarray as xr

from openMWR.models import Model, SpectralConsistencyModel
from openMWR.run import run_model


def main():
    torch.set_num_threads(1)
    cases = []
    with tempfile.TemporaryDirectory() as directory:
        for name in ('zenith', 'bls', 'mlr', 'spectral'):
            kwargs = {'freqs': np.array([22.24, 58.0])}
            if name == 'spectral':
                model = SpectralConsistencyModel(name, hidden_layer_size=4, **kwargs)
            else:
                model = Model(name, heights=np.array([0., 2000., 2001., 10000., 10001.]),
                              hidden_layers=[4], is_bls_model=name == 'bls', is_MLR=name == 'mlr', **kwargs)
            with torch.no_grad():
                for index, parameter in enumerate(model.parameters()):
                    parameter.copy_(torch.linspace(-0.02, 0.02, parameter.numel()).reshape(parameter.shape) + index * 0.001)
            mean_std = {'TB': (250., 20.), 'TB_IR': (250., 20.), 'surface_T': (280., 10.),
                        'surface_p': (900., 50.), 'surface_rh': (50., 20.),
                        'T': (280., 10.), 'rh': (50., 20.), 'iwv': (15., 5.)}
            model.nor = xr.Dataset()
            for variable, template in {**model.input.data_vars, **model.output.data_vars}.items():
                mean, std = mean_std.get(variable, (0., 1.))
                model.nor[variable] = xr.concat(
                    [xr.full_like(template, value, dtype=float) for value in (mean, std, 300.)],
                    dim=xr.IndexVariable('stat', ['mean', 'std', 'max']))
            inputs = xr.Dataset()
            for variable, template in model.input.items():
                mean, std = mean_std.get(variable, (0., 1.))
                shape = (2, *template.shape)
                values = mean + std * np.linspace(-0.5, 0.5, np.prod(shape)).reshape(shape)
                inputs[variable] = xr.DataArray(values, dims=('time', *template.dims),
                    coords={'time': np.array(['2025-01-12T12:00', '2025-01-12T12:01'], dtype='datetime64[ns]'),
                            **template.coords})
            model.save('test', str(Path(directory) / 'data'), {'file_extension': 'R17'})
            checkpoint = (Path(directory) / 'data/sites/test/retrieval' / f'{name}.pth').read_bytes()
            prediction = run_model(model, inputs, 'cpu', include_input_vars=False)
            cases.append({'name': name, 'checkpoint': checkpoint, 'inputs': inputs.to_dict(),
                          'prediction': prediction.to_dict(),
                          'loss_weights': model.loss_weights.to_dict() if hasattr(model, 'loss_weights') else None})
    output = Path(__file__).resolve().parents[1] / 'data/checkpoint_01.pt'
    torch.save({'source_commit': '3a366973a7356b811eb93b39e416c2519113a88f', 'cases': cases}, output)
    print(output)


if __name__ == '__main__':
    main()
