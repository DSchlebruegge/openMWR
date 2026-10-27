"""Binary and NetCDF observations share one dataset and prediction workflow."""

from datetime import datetime
import json
from pathlib import Path
import shutil
import struct
import tomllib

import numpy as np
import pytest
import tomli_w
import xarray as xr

from openMWR.datasets import DatasetWorkflow
from openMWR.hatpro_data import import_hatpro_data, import_hatpro_data_bls
from openMWR.hatpro_io import read_hatpro_file
from openMWR.predict import PredictionWorkflow
from openMWR.site import create_site
from tests.fixtures.prediction import checkpoint

DATA = Path(__file__).resolve().parents[1] / 'data/hatpro'


def project(tmp_path, variant='modern'):
    day = json.loads((DATA / 'provenance.json').read_text())[variant]['date']
    date = datetime.fromisoformat(day)
    raw = tmp_path / 'raw' / date.strftime('Y%Y/M%m/D%d')
    shutil.copytree(DATA / variant, raw)
    create_site('test', tmp_path, gen='G5', hatpro_data_dir='raw',
                rpg_retrieval_exists=False, mwr_height=550)
    return date, raw


def configure(tmp_path, **changes):
    path = tmp_path / 'sites/test/site.toml'
    values = tomllib.loads(path.read_text())
    values.update(changes)
    path.write_text(tomli_w.dumps(values))


@pytest.mark.parametrize('variant', ['legacy', 'modern'])
def test_real_binary_and_manufacturer_exports_give_the_same_daily_observations(tmp_path, variant):
    date, raw = project(tmp_path, variant)
    for importer in (import_hatpro_data, import_hatpro_data_bls):
        configure(tmp_path, hatpro_format='netcdf')
        expected = importer(date, 'test', tmp_path, mean_one_min=True, import_retrieval_data=False)
        configure(tmp_path, hatpro_format='binary')
        actual = importer(date, 'test', tmp_path, mean_one_min=True, import_retrieval_data=False)
        # Binary BLB has no explicit azimuth field. Every common observation and
        # derived time input must agree, including unpacked rain and MET humidity.
        variables = {'TB', 'rain_flag', 'surface_p', 'surface_T', 'surface_rh',
                     'doy_cos', 'doy_sin', 'years_since_1970'}
        if importer is import_hatpro_data:
            variables |= {'TB_IR', 'elevation_angle', 'azimuth_angle'}
            assert set(expected.data_vars) == variables
        else:
            assert set(expected.data_vars) == variables | {'azimuth_angle'}
        assert set(actual.data_vars) == variables
        xr.testing.assert_allclose(actual[sorted(variables)], expected[sorted(variables)], rtol=1e-6, atol=4e-5)
        assert float(actual.years_since_1970.min()) > 40
        assert actual.time.dtype == np.dtype('datetime64[ns]')
        assert np.isfinite(actual.TB).any() and np.isfinite(actual.surface_p).any()
        assert float(actual.surface_rh.max()) <= 100
        assert actual.attrs['mwrpy_version']
    for suffix in ('', '.NC'):
        irt = read_hatpro_file(next(raw.glob(f'*.IRT{suffix}')))
        assert irt.TB_IR.dims == ('time', 'wavelength')
        assert irt.TB_IR.attrs['units'] == 'K' and float(irt.TB_IR.min()) > 100
        blb = read_hatpro_file(next(raw.glob(f'*.BLB{suffix}')))
        assert blb.sizes['ang'] == 10 and bool((blb.ang > 0).all())
    assert irt.wavelength.values.tolist() == [10.5]


def test_auto_selects_each_product_and_never_hides_a_broken_selected_file(tmp_path):
    date, raw = project(tmp_path)
    brt = next(raw.glob('*.BRT.NC'))
    reference = xr.load_dataset(brt)
    reference['TBs'] += 10
    reference.to_netcdf(brt)
    # Missing IRT NetCDF uses binary, while BRT still uses the modified NetCDF.
    next(raw.glob('*.IRT.NC')).unlink()
    mixed = import_hatpro_data(date, 'test', tmp_path, import_retrieval_data=False)
    np.testing.assert_allclose(mixed.TB.isel(time=0), reference.TBs.isel(time=0))
    assert np.isfinite(mixed.TB_IR).any()
    configure(tmp_path, hatpro_format='netcdf')
    nc = import_hatpro_data(date, 'test', tmp_path, import_retrieval_data=False)
    assert 'TB_IR' not in nc
    configure(tmp_path, hatpro_format='auto')
    brt.write_bytes(b'Broken selected NetCDF, even though valid binary exists')
    with pytest.raises((ValueError, OSError)):
        import_hatpro_data(date, 'test', tmp_path, import_retrieval_data=False)
    configure(tmp_path, hatpro_format='binary')
    next(raw.glob('*.MET')).unlink()
    binary = import_hatpro_data(date, 'test', tmp_path, import_retrieval_data=False)
    np.testing.assert_allclose(binary.TB.isel(time=0), reference.TBs.isel(time=0) - 10)
    assert 'surface_p' not in binary  # Forced binary never uses the MET NetCDF.
    # RPG comparison products still require their manufacturer's NetCDF exports.
    next(raw.glob('*.IRT')).with_suffix('.LWP').write_bytes(b'Binary RPG product')
    with pytest.raises(ValueError, match='requires NetCDF'):
        import_hatpro_data(date, 'test', tmp_path, import_retrieval_data=True)
    configure(tmp_path, hatpro_format='unknown')
    with pytest.raises(ValueError, match='hatpro_format'):
        import_hatpro_data(date, 'test', tmp_path, import_retrieval_data=False)
    config = tmp_path / 'sites/test/site.toml'
    config.write_text('hatpro_format = "auto"\n')
    with pytest.raises(ValueError, match='Set hatpro_data_dir'):
        import_hatpro_data(date, 'test', tmp_path, import_retrieval_data=False)


def test_malformed_netcdf_names_missing_fields_and_preserves_partial_weather(tmp_path):
    date, raw = project(tmp_path)
    configure(tmp_path, hatpro_format='netcdf')
    path = next(raw.glob('*.BRT.NC'))
    original = xr.load_dataset(path)
    expected_temperature = import_hatpro_data(date, 'test', tmp_path,
        import_retrieval_data=False).surface_T
    for variable, diagnostic in [('TBs', 'missing required variables: TB'),
                                 ('Freq', 'missing required coordinate frq'),
                                 ('time', 'missing required coordinate time')]:
        original.drop_vars(variable).to_netcdf(path)
        with pytest.raises(ValueError, match=f'BRT.NC.*{diagnostic}'):
            import_hatpro_data(date, 'test', tmp_path, import_retrieval_data=False)
    invalid = original.copy()
    invalid['Freq'] = original.Freq.expand_dims(time=original.time)
    invalid.to_netcdf(path)
    with pytest.raises(ValueError, match='BRT.NC.*coordinate Freq must be one-dimensional'):
        read_hatpro_file(path)
    invalid = original.copy()
    invalid['TBs'] = original.TBs.rename({'number_frequencies': 'wrong_axis'})
    invalid.to_netcdf(path)
    with pytest.raises(ValueError, match='BRT.NC.*TB must use dimensions'):
        read_hatpro_file(path)
    original.to_netcdf(path)
    # A malformed optional IR file is reported and omitted. Valid station fields
    # still work even when other station variables are absent.
    irt = next(raw.glob('*.IRT.NC'))
    infrared = xr.load_dataset(irt)
    infrared.drop_vars('WavLens').to_netcdf(irt)
    with pytest.raises(ValueError, match='IRT.NC.*missing required coordinate wavelength'):
        read_hatpro_file(irt)
    infrared.drop_vars('IRR_Map').to_netcdf(irt)
    met = next(raw.glob('*.MET.NC'))
    xr.load_dataset(met).drop_vars(['Surf_P', 'Surf_RH']).to_netcdf(met)
    result = import_hatpro_data(date, 'test', tmp_path, import_retrieval_data=False)
    assert {'TB', 'surface_T'} <= set(result.data_vars)
    assert not {'TB_IR', 'surface_p', 'surface_rh'} & set(result.data_vars)
    xr.testing.assert_identical(result.surface_T, expected_temperature)


def test_binary_errors_and_local_timestamps_have_explicit_meaning(tmp_path):
    date, raw = project(tmp_path)
    path = next(raw.glob('*.BRT'))
    original = path.read_bytes()
    utc = read_hatpro_file(path)
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match='BRT.*records'):
        read_hatpro_file(path)
    content = bytearray(original)
    struct.pack_into('<i', content, 0, 999)
    path.write_bytes(content)
    with pytest.raises(ValueError, match='BRT.*not supported'):
        read_hatpro_file(path)
    content = bytearray(original)
    struct.pack_into('<i', content, 8, 0)  # BRT header's local/UTC reference.
    path.write_bytes(content)
    with pytest.raises(ValueError, match='hatpro_time_offset_hours'):
        read_hatpro_file(path)
    local = read_hatpro_file(path, time_offset_hours=1)
    np.testing.assert_array_equal(local.time, utc.time - np.timedelta64(1, 'h'))
    np.testing.assert_array_equal(local.TB, utc.TB)
    path.write_bytes(original)
    xr.testing.assert_identical(read_hatpro_file(path, time_offset_hours=1), utc)


def test_binary_dataset_build_update_and_prediction_use_the_same_inputs(tmp_path):
    date, raw = project(tmp_path)
    recipe = tmp_path / 'sites/test/datasets/hatpro.toml'
    recipe.write_text(tomli_w.dumps(dict(kind='hatpro', start_date=str(date.date()),
        end_date=str(date.date()), observations={'import_retrieval_data': False},
        execution={'processes': 1})))
    assert DatasetWorkflow(tmp_path, 'test').settings('hatpro')['hatpro_format'] == 'auto'
    configure(tmp_path, hatpro_format='binary')
    workflow = DatasetWorkflow(tmp_path, 'test')
    workflow.build('hatpro')
    assert workflow.settings('hatpro')['hatpro_format'] == 'binary'
    workflow.require('hatpro')
    outputs = workflow.files('hatpro')
    before = {path: xr.load_dataset(path) for path in outputs}
    workflow.build('hatpro')
    for path, original in before.items():
        xr.testing.assert_identical(xr.load_dataset(path), original)
    checkpoint(tmp_path, 'primary')
    checkpoint(tmp_path, 'scan', bls=True)
    result = PredictionWorkflow(tmp_path, 'test', ['primary', 'scan'], device='cpu').predict(date)
    assert not result.errors and result.zenith is not None and result.bls is not None
    saved = PredictionWorkflow(tmp_path, 'test', ['primary', 'scan'], device='cpu').save(result)
    assert len(saved) == 2 and all(path.is_file() for path in saved)
    configure(tmp_path, hatpro_format='netcdf')
    expected = PredictionWorkflow(tmp_path, 'test', ['primary', 'scan'], device='cpu').predict(date)
    xr.testing.assert_allclose(result.zenith, expected.zenith, rtol=1e-6, atol=4e-5)
    xr.testing.assert_allclose(result.bls, expected.bls, rtol=1e-6, atol=4e-5)
    with pytest.raises(ValueError, match='hatpro_format'):
        DatasetWorkflow(tmp_path, 'test').require('hatpro')
