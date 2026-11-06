from importlib.resources import files
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest
import torch

import torchMWRT
import torchMWRT.lineshape as lineshape
from torchMWRT.absorption_model import AbsModelError, H2OAbsModel, O2AbsModel, O3AbsModel


def test_coefficients_load_from_readonly_packages_without_local_copies(tmp_path):
    installed = tmp_path / 'site-packages'
    package = installed / 'torchMWRT'
    shutil.copytree(Path(torchMWRT.__file__).parent, package,
                    ignore=shutil.ignore_patterns('_lineshape', '__pycache__'))
    dependency = installed / 'pyrtlib'
    (dependency / '_lineshape').mkdir(parents=True)
    (dependency / '__init__.py').write_text('')
    for loader in (lineshape.H2OLL, lineshape.O2LL, lineshape.O3LL):
        resource = files('pyrtlib').joinpath(loader.NC_PATH)
        (dependency / loader.NC_PATH).write_bytes(resource.read_bytes())
    before = {path.relative_to(installed): path.read_bytes()
              for path in installed.rglob('*') if path.is_file()}
    try:
        for path in installed.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
        installed.chmod(0o555)
        result = subprocess.run(
            [sys.executable, '-c', '''
from pathlib import Path
import torchMWRT
import pyrtlib
from torchMWRT.lineshape import H2OLL, O2LL, O3LL

assert Path(torchMWRT.__file__).parent == Path('site-packages/torchMWRT').resolve()
assert Path(pyrtlib.__file__).parent == Path('site-packages/pyrtlib').resolve()
for loader in (H2OLL, O2LL, O3LL):
    name = loader.available_models()[0]
    assert loader.load(name).model == name
torchMWRT.RTModel(freqs=[22.24], angles=[90.0], absmdl='R24')
assert not Path(torchMWRT.__file__).with_name('_lineshape').exists()
'''], cwd=tmp_path,
            env={**os.environ, 'PYTHONPATH': str(installed), 'PYTHONDONTWRITEBYTECODE': '1'},
            capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert {path.relative_to(installed): path.read_bytes()
                for path in installed.rglob('*') if path.is_file()} == before
    finally:
        installed.chmod(0o755)
        for path in installed.rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o644)


@pytest.mark.parametrize('loader', [lineshape.H2OLL, lineshape.O2LL, lineshape.O3LL],
                         ids=['water-vapour', 'oxygen', 'ozone'])
def test_missing_table_identifies_pyrtlib_and_the_file(tmp_path, monkeypatch, loader):
    monkeypatch.setattr(lineshape, 'files', lambda package: tmp_path)
    with pytest.raises(FileNotFoundError, match=f'{Path(loader.NC_PATH).name}.*pyrtlib'):
        loader.load('R24')


@pytest.mark.parametrize('loader, absorption_model, gas, model', [
    (lineshape.H2OLL, H2OAbsModel, 'H2O', 'R24'),
    (lineshape.O2LL, O2AbsModel, 'O2', 'R24'),
    (lineshape.O3LL, O3AbsModel, 'O3', 'R23'),
], ids=['water-vapour', 'oxygen', 'ozone'])
def test_model_loading_reads_once_and_preserves_configuration_errors(monkeypatch, loader, absorption_model, gas, model):
    opened = []
    open_table = lineshape._open_lineshape

    def track_table(nc_path):
        opened.append(nc_path)
        return open_table(nc_path)

    monkeypatch.setattr(lineshape, '_open_lineshape', track_table)
    options = dict(dtype=torch.float64, device=None, freqs=torch.tensor([22.24], dtype=torch.float64))
    assert absorption_model(model, **options).model == model
    assert opened == [loader.NC_PATH]

    opened.clear()
    with pytest.raises(AbsModelError, match=f'unsupported.*{gas} absorption.*{model}') as error:
        absorption_model('unsupported', **options)
    assert opened == [loader.NC_PATH]
    assert type(error.value) is lineshape.AbsModelError
    assert error.value.model == 'unsupported'
    assert error.value.message == str(error.value)
    assert not isinstance(error.value, ValueError)
