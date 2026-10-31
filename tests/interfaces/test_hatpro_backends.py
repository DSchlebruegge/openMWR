"""Binary and NetCDF observations share one dataset and prediction workflow."""

from datetime import datetime, timedelta
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
from openMWR import hatpro_data
from openMWR.hatpro_data import import_hatpro_data, import_hatpro_data_bls
from openMWR.hatpro_io import load_hatpro_day, read_hatpro_file
from openMWR.predict import PredictionWorkflow
from openMWR.site import create_site, load_site_config
from tests.fixtures.prediction import checkpoint

DATA = Path(__file__).resolve().parents[1] / 'data/hatpro'


def project(tmp_path, variant='modern', *, rpg=False):
    day = json.loads((DATA / 'provenance.json').read_text())[variant]['date']
    date = datetime.fromisoformat(day)
    raw = tmp_path / 'raw' / date.strftime('Y%Y/M%m/D%d')
    shutil.copytree(DATA / variant, raw)
    if rpg:
        for path in (DATA / variant / 'rpg').iterdir():
            shutil.copy(path, raw)
    create_site('test', tmp_path, instrument_generation='G5', hatpro_data_dir='raw',
                mwr_height=550)
    return date, raw


def configure(tmp_path, **changes):
    path = tmp_path / 'sites/test/site.toml'
    values = tomllib.loads(path.read_text())
    values.update(changes)
    path.write_text(tomli_w.dumps(values))


@pytest.mark.parametrize('variant', ['legacy', 'modern'])
@pytest.mark.parametrize('include_rpg', [False, True])
def test_real_binary_and_manufacturer_exports_give_the_same_daily_observations(tmp_path, variant, include_rpg):
    date, raw = project(tmp_path, variant, rpg=include_rpg)
    if include_rpg:
        # Independent manufacturer exports verify every stored value, timestamp
        # and height, before matching, filtering or derived calculations.
        for product in ('LWP', 'IWV', 'TPC', 'TPB', 'HPC', 'LPR'):
            path = next(raw.glob(f'*.{product}'))
            binary = read_hatpro_file(path)
            netcdf = read_hatpro_file(Path(str(path) + '.NC'))
            xr.testing.assert_allclose(binary, netcdf, rtol=0, atol=0)
            assert binary.attrs['rpg_file_code'] > 0
            assert all('units' in value.attrs for value in binary.data_vars.values())
    for importer in (import_hatpro_data, import_hatpro_data_bls):
        configure(tmp_path, hatpro_format='netcdf')
        expected = importer(date, 'test', tmp_path, mean_one_min=True, include_rpg=include_rpg)
        configure(tmp_path, hatpro_format='binary')
        actual = importer(date, 'test', tmp_path, mean_one_min=True, include_rpg=include_rpg)
        # Binary BLB has no explicit azimuth field. Every common observation and
        # derived time input must agree, including unpacked rain and MET humidity.
        variables = {'TB', 'rain_flag', 'surface_p', 'surface_T', 'surface_rh',
                     'doy_cos', 'doy_sin', 'years_since_1970'}
        if include_rpg:
            variables |= {'T', 'p'}
        if importer is import_hatpro_data:
            variables |= {'TB_IR', 'elevation_angle', 'azimuth_angle'}
            if include_rpg:
                variables |= {'ah', 'rh', 'TD', 'lwp', 'iwv', 'lwc'}
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
        if include_rpg:
            comparisons = ('T', 'p') if importer is import_hatpro_data_bls else ('T', 'p', 'TD', 'ah', 'rh', 'lwp', 'iwv', 'lwc')
            assert all(np.isfinite(actual[name]).any() for name in comparisons)
    for suffix in ('', '.NC'):
        irt = read_hatpro_file(next(raw.glob(f'*.IRT{suffix}')))
        assert irt.TB_IR.dims == ('time', 'wavelength')
        assert irt.TB_IR.attrs['units'] == 'K' and float(irt.TB_IR.min()) > 100
        blb = read_hatpro_file(next(raw.glob(f'*.BLB{suffix}')))
        assert blb.sizes['ang'] == 10 and bool((blb.ang > 0).all())
    assert irt.wavelength.values.tolist() == [10.5]


@pytest.mark.parametrize('format', ['netcdf', 'binary'])
def test_daily_import_discards_corrupt_times_and_limits_output_to_requested_day(tmp_path, caplog, format):
    date, raw = project(tmp_path)
    configure(tmp_path, hatpro_format=format)

    def append_times(path, times):
        times = np.asarray(times, dtype='datetime64[ns]')
        if path.suffix == '.NC':
            original = xr.load_dataset(path)
            extra = original.isel(time=[0] * len(times)).assign_coords(time=times)
            combined = xr.concat([original, extra], dim='time', data_vars='minimal', coords='minimal',
                                 compat='override')
            combined.time.encoding = dict(units='seconds since 1970-01-01', dtype='int64',
                                          _FillValue=np.iinfo('int64').min)
            combined.to_netcdf(path)
        else:
            content = bytearray(path.read_bytes())
            _, count, _, frequencies = struct.unpack_from('<4i', content)
            header_size = 16 + 12 * frequencies
            record_size = 9 + 4 * frequencies
            record = content[header_size:header_size + record_size]
            origin = read_hatpro_file(path).time.values[0]
            original_time, = struct.unpack_from('<i', record)
            for time in times:
                struct.pack_into('<i', record, 0, original_time + int((time - origin) / np.timedelta64(1, 's')))
                content.extend(record)
            struct.pack_into('<i', content, 4, count + len(times))
            path.write_bytes(content)

    suffix = '.NC' if format == 'netcdf' else ''
    brt = next(raw.glob(f'*.BRT{suffix}'))
    expected = {importer: importer(date, 'test', tmp_path, mean_one_min=True, include_rpg=False)
                for importer in (import_hatpro_data, import_hatpro_data_bls)}
    overlaps = [np.datetime64(date) - np.timedelta64(1, 's'),
                np.datetime64(date + timedelta(days=1)),
                np.datetime64(date + timedelta(days=1)) + np.timedelta64(1, 's')]
    append_times(brt, overlaps)
    if format == 'netcdf':
        append_times(next(raw.glob('*.BLB.NC')), overlaps)
    expected_times = read_hatpro_file(brt).drop_duplicates('time').sortby('time').time
    products = ['BRT', 'BLB', 'MET', 'IRT']
    for product in products if format == 'netcdf' else ['BRT']:
        bad_times = ['1940-01-01', '2037-01-01', 'NaT'] if format == 'netcdf' else ['1940-01-01', '2037-01-01']
        append_times(next(raw.glob(f'*.{product}{suffix}')), bad_times)

    # Keep adjacent-day samples available for matching auxiliary sensors. Only
    # the assembled daily observations exclude them. Check before resampling so
    # corrupt timestamps cannot exhaust CI memory.
    with caplog.at_level('WARNING', logger='openMWR.hatpro_io'):
        cleaned = load_hatpro_day(date, 'test', tmp_path, products)
    xr.testing.assert_identical(cleaned['BRT'].time, expected_times)
    for data in cleaned.values():
        assert bool(((data.time >= np.datetime64(date - timedelta(days=1))) &
                     (data.time <= np.datetime64(date + timedelta(days=2)))).all())
    assert str(brt) in caplog.text and 'Discarding' in caplog.text
    for importer, original in expected.items():
        actual = importer(date, 'test', tmp_path, mean_one_min=True, include_rpg=False)
        xr.testing.assert_identical(actual, original)
        assert bool(((actual.time >= np.datetime64(date)) &
                     (actual.time < np.datetime64(date + timedelta(days=1)))).all())

    # A readable neighbouring-day file is not usable data for the requested day.
    if format == 'netcdf':
        for product, importer in [('BRT', import_hatpro_data), ('BLB', import_hatpro_data_bls)]:
            path = next(raw.glob(f'*.{product}.NC'))
            data = xr.load_dataset(path).isel(time=[0])
            data.assign_coords(time=[overlaps[0]]).to_netcdf(path)
            with pytest.raises(ValueError, match='requested UTC day'):
                importer(date, 'test', tmp_path, mean_one_min=True, include_rpg=False)


def test_days_with_only_invalid_timestamps_report_the_file_and_omit_optional_products(tmp_path, caplog):
    date, raw = project(tmp_path)
    configure(tmp_path, hatpro_format='netcdf')
    original = import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
    for product in ['MET', 'IRT', 'BRT']:
        path = next(raw.glob(f'*.{product}.NC'))
        data = xr.load_dataset(path)
        data.assign_coords(time=np.full(data.sizes['time'], np.datetime64('NaT', 'ns'))).to_netcdf(path)
        if product == 'BRT':
            with pytest.raises(ValueError, match='BRT.NC.*no usable timestamps'):
                load_hatpro_day(date, 'test', tmp_path, ['BRT'])
        else:
            assert not load_hatpro_day(date, 'test', tmp_path, [product])
            result = import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
            xr.testing.assert_identical(result.TB, original.TB)
            assert ('surface_p' if product == 'MET' else 'TB_IR') not in result
        assert str(path) in caplog.text


def test_auto_selects_each_product_and_never_hides_a_broken_selected_file(tmp_path):
    date, raw = project(tmp_path)
    brt = next(raw.glob('*.BRT.NC'))
    reference = xr.load_dataset(brt)
    reference['TBs'] += 10
    reference.to_netcdf(brt)
    # Missing IRT NetCDF uses binary, while BRT still uses the modified NetCDF.
    next(raw.glob('*.IRT.NC')).unlink()
    mixed = import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
    np.testing.assert_allclose(mixed.TB.isel(time=0), reference.TBs.isel(time=0))
    assert np.isfinite(mixed.TB_IR).any()
    configure(tmp_path, hatpro_format='netcdf')
    nc = import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
    assert 'TB_IR' not in nc
    configure(tmp_path, hatpro_format='auto')
    brt.write_bytes(b'Broken selected NetCDF, even though valid binary exists')
    with pytest.raises((ValueError, OSError)):
        import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
    configure(tmp_path, hatpro_format='binary')
    next(raw.glob('*.MET')).unlink()
    binary = import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
    np.testing.assert_allclose(binary.TB.isel(time=0), reference.TBs.isel(time=0) - 10)
    assert 'surface_p' not in binary  # Forced binary never uses the MET NetCDF.
    configure(tmp_path, hatpro_format='unknown')
    with pytest.raises(ValueError, match='hatpro_format'):
        import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
    config = tmp_path / 'sites/test/site.toml'
    config.write_text('hatpro_format = "auto"\n')
    with pytest.raises(ValueError, match='set hatpro_data_dir'):
        import_hatpro_data(date, 'test', tmp_path, include_rpg=False)


@pytest.mark.parametrize('angles', [
    [90., 42., 30., 19.2, 10.2, 5.4],
    [90., 30., 19.2, 14.4, 11.4, 8.4, 6.6, 5.4, 4.8, 42.],
], ids=['six-angle-scan', 'ten-angles-with-a-different-direction'])
def test_incompatible_scans_are_skipped_with_the_missing_and_measured_angles(tmp_path, caplog, angles):
    date, raw = project(tmp_path)
    path = next(raw.glob('*.BLB.NC'))
    data = xr.load_dataset(path).isel(number_scan_angles=slice(0, len(angles) + 1))
    data['ElAngs'] = ('number_scan_angles', [*angles, 0.])
    data.to_netcdf(path)
    output = tmp_path / 'processed'
    output.mkdir()
    with pytest.raises(ValueError, match='Incompatible BLS scan angles.*missing required angles'):
        import_hatpro_data_bls(date, 'test', tmp_path, include_rpg=False,
                              out_file=output / 'rejected.nc')
    with caplog.at_level('INFO', logger='openMWR.hatpro_data'):
        hatpro_data._create_hatpro_day_file(date, True, '_bls', 'test', tmp_path, output, False, load_site_config('test', tmp_path))
    assert f'Skipping HATPRO BLS for test {date:%Y-%m-%d}' in caplog.text
    assert '4.2' in caplog.text and f'Measured angles: {angles}' in caplog.text
    assert not any(record.levelname == 'ERROR' or record.exc_info for record in caplog.records)
    assert not list(output.iterdir())


def test_complete_scans_keep_extra_angles_and_their_actual_brightness_temperatures(tmp_path):
    date, raw = project(tmp_path)
    path = next(raw.glob('*.BLB.NC'))
    original = xr.load_dataset(path)
    extra = original.isel(number_scan_angles=[0]).copy(deep=True)
    extra['ElAngs'] = ('number_scan_angles', [42.])
    extra['TBs'] += 10
    data = xr.concat([original, extra], dim='number_scan_angles', data_vars='minimal',
                     coords='minimal', compat='override').isel(number_scan_angles=slice(None, None, -1))
    data.to_netcdf(path)
    expected = read_hatpro_file(path)
    expected = expected.assign_coords(ang=expected.ang.astype('float64').round(2),
                                      frq=expected.frq.astype('float64').round(2)).sortby('ang')
    actual = import_hatpro_data_bls(date, 'test', tmp_path, include_rpg=False)
    xr.testing.assert_identical(actual.TB, expected.TB)
    assert actual.sizes['ang'] == 11 and 42. in actual.ang


def test_empty_zenith_selection_skips_averaging_and_preserves_independent_scans(tmp_path, caplog):
    date, raw = project(tmp_path)
    path = next(raw.glob('*.BRT.NC'))
    original = xr.load_dataset(path)
    data = original.copy(deep=True)
    data['ElAng'] = data.ElAng * -1
    data.to_netcdf(path)
    with pytest.raises(ValueError, match='No zenith observations remain'):
        import_hatpro_data(date, 'test', tmp_path, mean_one_min=True, include_rpg=False)
    output = tmp_path / 'processed'
    output.mkdir()
    with caplog.at_level('INFO', logger='openMWR.hatpro_data'):
        hatpro_data._create_hatpro_day_file(date, False, '', 'test', tmp_path, output, False, load_site_config('test', tmp_path))
    assert 'Skipping HATPRO zenith' in caplog.text and 'No zenith observations remain' in caplog.text
    assert not any(record.levelname == 'ERROR' or record.exc_info for record in caplog.records)
    assert not list(output.iterdir())
    hatpro_data._create_hatpro_day_file(date, True, '_bls', 'test', tmp_path, output, False, load_site_config('test', tmp_path))
    assert (output / f'hatpro_data_bls_{date:%Y%m%d}.nc').is_file()

    # A day with both target and sky views retains the sky portion.
    original['ElAng'] = original.ElAng.where(original.time >= original.time[-3], -90.)
    original.to_netcdf(path)
    partial = import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
    np.testing.assert_array_equal(partial.time, original.time[-3:])
    np.testing.assert_array_equal(partial.TB, original.TBs[-3:])
    averaged = import_hatpro_data(date, 'test', tmp_path, mean_one_min=True, include_rpg=False)
    assert np.isfinite(averaged.TB).any() and bool((averaged.elevation_angle.dropna('time') == 90).all())


@pytest.mark.parametrize('importer,profile_product', [
    (import_hatpro_data, 'TPC'), (import_hatpro_data_bls, 'TPB'),
])
def test_unavailable_rpg_products_preserve_observations_and_other_comparisons(tmp_path, caplog, importer, profile_product):
    date, raw = project(tmp_path)
    configure(tmp_path, hatpro_format='netcdf')
    expected = importer(date, 'test', tmp_path, mean_one_min=True, include_rpg=False)
    primary = 'BRT' if profile_product == 'TPC' else 'BLB'
    times = read_hatpro_file(next(raw.glob(f'*.{primary}.NC'))).time.values
    # Hourly source blocks need not arrive in chronological order.
    met = next(raw.glob('*.MET.NC'))
    xr.load_dataset(met).isel(time=slice(None, None, -1)).to_netcdf(met)
    height = ('number_altitude_layers', [0., 1000.])
    values = np.tile([280., 275.], (len(times), 1))
    temperature = xr.Dataset({
        'temperature_profiles': (('time', 'number_altitude_layers'), values),
        'altitude_layers': height,
    }, coords={'time': times})
    profiles = raw / f'{date:%y%m%d}.{profile_product}.NC'
    temperature.to_netcdf(profiles)
    if profile_product == 'TPC':
        humidity = xr.Dataset({
            'Absolute_Humidity_Profiles': (('time', 'number_altitude_layers'), values * 0 + 3),
            'relative_humidity_profiles': (('time', 'number_altitude_layers'), values * 0 + 50),
            'altitude_layers': height,
        }, coords={'time': times})
        humidity.to_netcdf(raw / f'{date:%y%m%d}.HPC.NC')
        xr.Dataset({'LWP_data': ('time', np.full(len(times), 20.))},
                   coords={'time': times}).to_netcdf(raw / f'{date:%y%m%d}.LWP.NC')

    def read():
        result = importer(date, 'test', tmp_path, mean_one_min=True, include_rpg=True)
        for name in expected.data_vars:
            xr.testing.assert_identical(result[name], expected[name])
        return result

    complete = read()
    assert np.isfinite(complete.T).any() and np.isfinite(complete.p).any()
    if profile_product == 'TPC':
        assert np.isfinite(complete.TD).any() and float(complete.lwp.min()) == 20

    profiles.unlink()
    caplog.clear()
    partial = read()
    assert not {'T', 'p', 'TD'} & set(partial)
    assert profiles.name in caplog.text
    if profile_product == 'TPC':
        xr.testing.assert_identical(partial.ah, complete.ah)
        xr.testing.assert_identical(partial.lwp, complete.lwp)
        temperature.to_netcdf(profiles)
        broken = raw / f'{date:%y%m%d}.HPC.NC'
    else:
        broken = profiles
    broken.write_bytes(b'Unreadable RPG profile')
    partial = read()
    if profile_product == 'TPC':
        xr.testing.assert_identical(partial.T, complete.T)
        xr.testing.assert_identical(partial.p, complete.p)
        assert not {'ah', 'rh', 'TD'} & set(partial)
    else:
        assert 'T' not in partial and 'p' not in partial

    for product in ('TPC', 'TPB', 'HPC', 'LWP'):
        (raw / f'{date:%y%m%d}.{product}.NC').unlink(missing_ok=True)
    binary_rpg = raw / f'{date:%y%m%d}.LWP'
    binary_rpg.write_bytes(b'Broken RPG product')
    xr.testing.assert_identical(read(), expected)
    with pytest.raises(ValueError, match='incomplete RPG header'):
        read_hatpro_file(binary_rpg)


def test_malformed_netcdf_names_missing_fields_and_preserves_partial_weather(tmp_path):
    date, raw = project(tmp_path)
    configure(tmp_path, hatpro_format='netcdf')
    path = next(raw.glob('*.BRT.NC'))
    original = xr.load_dataset(path)
    expected_temperature = import_hatpro_data(date, 'test', tmp_path,
        include_rpg=False).surface_T
    for variable, diagnostic in [('TBs', 'missing required variables: TB'),
                                 ('Freq', 'missing required coordinate frq'),
                                 ('time', 'missing required coordinate time')]:
        original.drop_vars(variable).to_netcdf(path)
        with pytest.raises(ValueError, match=f'BRT.NC.*{diagnostic}'):
            import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
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
    result = import_hatpro_data(date, 'test', tmp_path, include_rpg=False)
    assert {'TB', 'surface_T'} <= set(result.data_vars)
    assert not {'TB_IR', 'surface_p', 'surface_rh'} & set(result.data_vars)
    xr.testing.assert_identical(result.surface_T, expected_temperature)


@pytest.mark.parametrize('product,reference_offset', [('BRT', 8), ('HPC', 16)])
def test_binary_errors_and_local_timestamps_have_explicit_meaning(tmp_path, product, reference_offset):
    date, raw = project(tmp_path, rpg=True)
    path = next(raw.glob(f'*.{product}'))
    original = path.read_bytes()
    utc = read_hatpro_file(path)
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match=f'{product}.*records'):
        read_hatpro_file(path)
    content = bytearray(original)
    struct.pack_into('<i', content, 0, 999)
    path.write_bytes(content)
    with pytest.raises(ValueError, match=f'{product}.*not supported'):
        read_hatpro_file(path)
    content = bytearray(original)
    struct.pack_into('<i', content, reference_offset, 0)
    path.write_bytes(content)
    with pytest.raises(ValueError, match='hatpro_time_offset_hours'):
        read_hatpro_file(path)
    local = read_hatpro_file(path, time_offset_hours=1)
    np.testing.assert_array_equal(local.time, utc.time - np.timedelta64(1, 'h'))
    for name in utc.data_vars:
        np.testing.assert_array_equal(local[name], utc[name])
    configure(tmp_path, hatpro_format='binary', hatpro_time_offset_hours=1)
    xr.testing.assert_identical(load_hatpro_day(date, 'test', tmp_path, [product])[product],
                                local.drop_duplicates('time').sortby('time'))
    struct.pack_into('<i', content, reference_offset, 2)
    path.write_bytes(content)
    with pytest.raises(ValueError, match='unsupported binary time reference 2'):
        read_hatpro_file(path)
    path.write_bytes(original)
    xr.testing.assert_identical(read_hatpro_file(path, time_offset_hours=1), utc)


@pytest.mark.parametrize('product', ['BRT', 'HPC'])
def test_binary_times_survive_the_unix_int32_boundary(tmp_path, product):
    _, raw = project(tmp_path, rpg=True)
    path = next(raw.glob(f'*.{product}'))
    original = read_hatpro_file(path)
    content = bytearray(path.read_bytes())
    expected = np.array(['2038-01-19T03:14:07', '2038-01-19T03:14:08',
                         '2040-01-01T00:00:00', '2069-01-19T03:14:07'], dtype='datetime64[ns]')
    if product == 'BRT':
        frequencies, = struct.unpack_from('<i', content, 12)
        start, size, reference_offset = 16 + 12 * frequencies, 9 + 4 * frequencies, 8
    else:
        levels, = struct.unpack_from('<i', content, 24)
        start, size, reference_offset = 28 + 4 * levels, 17 + 4 * levels, 16
    for index, time in enumerate(expected):
        seconds = int((datetime.fromisoformat(str(time)) - datetime(2001, 1, 1)).total_seconds())
        struct.pack_into('<i', content, start + index * size, seconds)
    path.write_bytes(content)
    utc = read_hatpro_file(path, time_offset_hours=5.75)
    np.testing.assert_array_equal(utc.time[:4], expected)
    for name in original.data_vars:
        np.testing.assert_array_equal(utc[name], original[name])
    # Offsets are applied only to local-time files, including fractional-hour
    # zones. The last sample exercises the binary epoch's largest signed value.
    struct.pack_into('<i', content, reference_offset, 0)
    path.write_bytes(content)
    for offset, seconds in [(5.75, 20700), (-3.5, -12600)]:
        local = read_hatpro_file(path, time_offset_hours=offset)
        np.testing.assert_array_equal(local.time[:4], expected - np.timedelta64(seconds, 's'))


@pytest.mark.parametrize('variant', ['legacy', 'modern'])
def test_humidity_only_formats_preserve_absolute_profiles_without_inventing_rh(tmp_path, variant):
    date, raw = project(tmp_path, variant, rpg=True)
    path = next(raw.glob('*.HPC'))
    content = bytearray(path.read_bytes())
    code, count = struct.unpack_from('<2i', content)
    levels, = struct.unpack_from('<i', content, 24)
    record_size = 5 + 4 * levels + (12 if variant == 'modern' else 0)
    content = content[:28 + 4 * levels + count * record_size]
    struct.pack_into('<i', content, 0, code - 1)
    path.write_bytes(content)
    nc = Path(str(path) + '.NC')
    original = xr.load_dataset(nc)
    original.drop_vars('RH_prof' if variant == 'modern' else 'relative_humidity_profiles').to_netcdf(nc)
    binary = read_hatpro_file(path)
    assert set(binary.data_vars) == {'ah'}
    xr.testing.assert_allclose(binary, read_hatpro_file(nc), rtol=0, atol=0)
    configure(tmp_path, hatpro_format='binary')
    day = import_hatpro_data(date, 'test', tmp_path, include_rpg=True, mean_one_min=True)
    assert 'rh' not in day and np.isfinite(day.ah).any() and np.isfinite(day.TD).any()


def test_rpg_format_selection_and_invalid_optional_files_preserve_other_products(tmp_path, caplog):
    date, raw = project(tmp_path, rpg=True)
    products = ['LWP', 'IWV', 'TPC', 'HPC', 'LPR', 'TPB']
    reference = load_hatpro_day(date, 'test', tmp_path, products)
    nc = next(raw.glob('*.LWP.NC'))
    data = xr.load_dataset(nc)
    data['LWP'] += 100
    data.to_netcdf(nc)
    auto = load_hatpro_day(date, 'test', tmp_path, products)
    xr.testing.assert_allclose(auto['LWP'].lwp, reference['LWP'].lwp + 100)
    configure(tmp_path, hatpro_format='binary')
    binary = load_hatpro_day(date, 'test', tmp_path, products)
    xr.testing.assert_allclose(binary['LWP'], reference['LWP'], rtol=0, atol=0)
    configure(tmp_path, hatpro_format='auto')
    nc.unlink()
    xr.testing.assert_identical(load_hatpro_day(date, 'test', tmp_path, products)['LWP'], binary['LWP'])
    nc.write_bytes(b'Broken selected NetCDF')
    assert set(load_hatpro_day(date, 'test', tmp_path, products)) == set(products) - {'LWP'}
    configure(tmp_path, hatpro_format='binary')
    path = next(raw.glob('*.TPC'))
    original = path.read_bytes()
    for content, diagnostic in [(original[:25], 'incomplete profile header'),
                                (original + b'extra', 'inconsistent records')]:
        path.write_bytes(content)
        with pytest.raises(ValueError, match=diagnostic):
            read_hatpro_file(path)
    for offset, value, diagnostic in [(4, 0, 'records must be positive'),
                                      (24, 0, 'height grid'),
                                      (24, 2**31 - 1, 'height grid'),
                                      (28, -1, 'strictly increasing'),
                                      (32, 0, 'strictly increasing')]:
        content = bytearray(original)
        struct.pack_into('<i', content, offset, value)
        path.write_bytes(content)
        with pytest.raises(ValueError, match=diagnostic):
            read_hatpro_file(path)
    path.write_bytes(original[:-1])
    next(raw.glob('*.HPC')).unlink()
    partial = load_hatpro_day(date, 'test', tmp_path, products)
    assert set(partial) == set(products) - {'TPC', 'HPC'}
    for product, values in partial.items():
        xr.testing.assert_identical(values, binary[product])
    assert str(path) in caplog.text


def test_binary_dataset_build_update_and_prediction_use_the_same_inputs(tmp_path):
    date, raw = project(tmp_path, rpg=True)
    recipe = tmp_path / 'sites/test/datasets/hatpro.toml'
    recipe.write_text(tomli_w.dumps(dict(kind='hatpro', start_date=str(date.date()),
        end_date=str(date.date()), observations={'include_rpg': True},
        execution={'processes': 1})))
    assert DatasetWorkflow(tmp_path, 'test').settings('hatpro')['hatpro_format'] == 'auto'
    configure(tmp_path, hatpro_format='binary')
    workflow = DatasetWorkflow(tmp_path, 'test')
    workflow.build('hatpro')
    assert workflow.settings('hatpro')['hatpro_format'] == 'binary'
    workflow.require('hatpro')
    outputs = workflow.files('hatpro')
    before = {path: xr.load_dataset(path) for path in outputs}
    assert all(np.isfinite(data.T).any() for data in before.values())
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


def test_combining_days_keeps_observations_and_missing_comparisons_as_nan(tmp_path):
    date, raw = project(tmp_path)
    configure(tmp_path, hatpro_format='netcdf')
    next_date = date + timedelta(days=1)
    following = tmp_path / 'raw' / next_date.strftime('Y%Y/M%m/D%d')
    following.mkdir(parents=True)
    for path in raw.glob('*.NC'):
        data = xr.load_dataset(path)
        data = data.assign_coords(time=data.time + np.timedelta64(1, 'D'))
        data.to_netcdf(following / path.name.replace(f'{date:%y%m%d}', f'{next_date:%y%m%d}'))
    times = xr.load_dataset(next(raw.glob('*.BRT.NC'))).time.values
    xr.Dataset({'T_prof': (('time', 'height'), np.full((len(times), 2), 280.))},
        coords={'time': times, 'height': [0., 1000.]}).to_netcdf(raw / f'{date:%y%m%d}.TPC.NC')
    recipe = tmp_path / 'sites/test/datasets/hatpro.toml'
    recipe.write_text(tomli_w.dumps(dict(kind='hatpro', start_date=str(date.date()),
        end_date=str(next_date.date()), observations={'include_rpg': True},
        execution={'processes': 1})))
    workflow = DatasetWorkflow(tmp_path, 'test')
    workflow.build('hatpro')
    zenith, bls = (xr.load_dataset(path) for path in workflow.files('hatpro'))
    for data in (zenith, bls):
        for day in (date, next_date):
            assert np.isfinite(data.TB.sel(time=str(day.date()))).any()
    for name in ('T', 'p'):
        assert np.isfinite(zenith[name].sel(time=str(date.date()))).any()
        assert np.isnan(zenith[name].sel(time=str(next_date.date()))).all()
    assert 'TD' not in zenith and 'T' not in bls
