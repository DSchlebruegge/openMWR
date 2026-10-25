"""External acquisition boundaries preserve and extend local caches."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openMWR.datasets import DatasetWorkflow
from openMWR.era5 import ERA5Processor
from openMWR.paths import era5_cache_file


from tests.fixtures.datasets import dataset_project, recipe, raw_profiles

def test_shared_era5_cache_appends_after_last_day_and_preserves_other_coverage(dataset_project, monkeypatch):
    requested = []
    def download(self, start, end, h_freq=6):
        requested.append((start, end, h_freq))
        times = pd.date_range(start, end + pd.Timedelta(days=1), freq=f'{h_freq}h', inclusive='left')
        return xr.Dataset({'T': ('time', np.full(len(times), 280.))}, coords={'time': times})
    monkeypatch.setattr(ERA5Processor, 'download_era5', download)
    cache = era5_cache_file(dataset_project, 48.148, 11.573, 2)
    processor = ERA5Processor('test', dataset_project, start_date='2020-01-01', end_date='2020-01-03', hour_interval=2)
    created = processor.create_original_dataset()
    assert requested == [(pd.Timestamp(day), pd.Timestamp(day), 2)
                         for day in ('2020-01-01', '2020-01-02', '2020-01-03')]
    assert created.time.size == 36
    requested.clear()
    # Historical six-hour samples and an incomplete last day must not trigger
    # backfilling when the recipe requests two-hour data for routine updates.
    old_times = pd.date_range('2020-01-02', periods=4, freq='6h').append(pd.DatetimeIndex(['2020-01-05']))
    old = xr.Dataset({'T': ('time', np.full(len(old_times), 270.))}, coords={'time': old_times})
    old.to_netcdf(cache)
    processor.end_date = pd.Timestamp('2020-01-07')
    updated = processor.create_original_dataset()
    assert requested == [(pd.Timestamp(day), pd.Timestamp(day), 2)
                         for day in ('2020-01-06', '2020-01-07')]
    assert updated.time.size == 29
    xr.testing.assert_identical(updated.sel(time=old_times), old)
    original = cache.read_bytes()
    processor.create_original_dataset()
    assert len(requested) == 2
    assert cache.read_bytes() == original
    processor.end_date = pd.Timestamp('2020-01-03')
    processor.create_original_dataset()
    assert len(requested) == 2  # An older requested window also leaves gaps alone.
    assert cache.read_bytes() == original
    processor.update_only = False
    rebuilt = processor.create_original_dataset()
    assert requested[-3:] == [(pd.Timestamp(day), pd.Timestamp(day), 2)
                             for day in ('2020-01-01', '2020-01-02', '2020-01-03')]
    assert rebuilt.time.size == 61
    assert rebuilt.T.sel(time='2020-01-02T00:00').item() == 280
    assert rebuilt.T.sel(time='2020-01-05').item() == 270
    assert era5_cache_file(dataset_project, 48.148, 11.573, 6) != cache


@pytest.mark.parametrize('failure', ['download', 'write'])
def test_interrupted_era5_update_keeps_completed_days_and_resumes(dataset_project, monkeypatch, failure):
    requested = []
    interrupted = False
    failed_day = pd.Timestamp('2020-01-03')

    def retrieve(product, request):
        nonlocal interrupted
        day = pd.Timestamp('-'.join(request[key][0] for key in ('year', 'month', 'day')))
        requested.append(day)
        if failure == 'download' and day == failed_day and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return SimpleNamespace(download=lambda path: _era5_response(day).to_netcdf(path))

    monkeypatch.setattr('openMWR.era5.cdsapi.Client', lambda: SimpleNamespace(retrieve=retrieve))
    write = xr.Dataset.to_netcdf

    def interrupted_write(data, path, *args, **kwargs):
        nonlocal interrupted
        if (failure == 'write' and 'T' in data
                and pd.Timestamp(data.time.max().values).normalize() == failed_day and not interrupted):
            interrupted = True
            Path(path).write_bytes(b'Interrupted NetCDF write')
            raise KeyboardInterrupt
        return write(data, path, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, 'to_netcdf', interrupted_write)
    processor = ERA5Processor('test', dataset_project, start_date='2020-01-01',
                              end_date='2020-01-01', hour_interval=12)
    original = processor.create_original_dataset()
    processor.end_date = pd.Timestamp('2020-01-04')
    with pytest.raises(KeyboardInterrupt):
        processor.create_original_dataset()

    saved = xr.load_dataset(processor.raw_data_file)
    np.testing.assert_array_equal(saved.time, pd.date_range('2020-01-01', periods=4, freq='12h'))
    xr.testing.assert_identical(saved.sel(time=original.time), original)
    requested.clear()
    resumed = processor.create_original_dataset()
    assert requested == [pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-04')]
    np.testing.assert_array_equal(resumed.time, pd.date_range('2020-01-01', periods=8, freq='12h'))
    xr.testing.assert_identical(resumed.sel(time=saved.time), saved)


def test_source_creation_remembers_unavailable_historical_years(dataset_project, monkeypatch):
    import openMWR.dwd_opendata as dwd

    calls = []
    monkeypatch.setattr(dwd, 'get_df_stations', lambda: None)
    def download(station, year, stations):
        calls.append(year)
        if year == 2020:
            raise FileNotFoundError('No historical soundings')
        return raw_profiles(np.array([f'{year}-01-01'], dtype='datetime64[ns]'))
    monkeypatch.setattr(dwd, '_download_radiosondes_year', download)
    recipe(dataset_project, 'source', kind='radiosonde', stations=['03715'], start_year=2020, end_year=2021)
    workflow = DatasetWorkflow(dataset_project, 'test')
    workflow.build('source')
    assert calls == [2020, 2021]
    workflow.build('source')
    assert calls == [2020, 2021]
    workflow.build('source', rebuild=True)
    assert calls == [2020, 2021, 2020, 2021]


def test_dwd_archive_decoding_units_quality_and_missing_year_preserves_cache(tmp_path, monkeypatch):
    """Exercise the actual station/ZIP/CSV readers; replace HTTP responses only."""
    from io import BytesIO
    from types import SimpleNamespace
    from zipfile import ZipFile
    from openMWR.dwd_opendata import update_radiosonde_data
    from openMWR.paths import radiosonde_year_file

    station = ['03715', '20000101', '20201231', '484', '48.25', '11.5', 'Munich', 'Bavaria']
    widths = [6, 9, 9, 15, 12, 10, 41, 41]
    metadata = 'Station table\nColumn descriptions\n' + ''.join(
        value.ljust(width) for value, width in zip(station, widths)) + '\n'
    columns = ['BEZUGSDATUM_SYNOP', 'MESSZEITPUNKT', 'AE_GB_POS', 'AE_GL_POS',
               'AE_GPM', 'AE_P', 'AE_TT', 'AE_TD', 'AE_FF', 'AE_DD', 'AE_RF']
    rows = [
        ['2020010100', 0, 48.25, 11.5, 500, 950, 10, 5, 4, 180, 70],
        ['2020010100', 0, 48.25, 11.5, 500, 950, 99, 5, 4, 180, 70],  # duplicate
        ['2020010100', 10, 48.25, 11.5, 600, 940, 8, 3, 5, 190, 75],
        ['2020010100', 20, 39, 11.5, 700, -999, 200, -999, 6, 400, 130],
        ['invalid', 30, 48.25, 11.5, 800, 920, 4, 1, 7, 200, 80],
    ]
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        archive.writestr('readme.txt', 'Data is in the final archive entry.')
        archive.writestr('soundings.txt', pd.DataFrame(rows, columns=columns).to_csv(sep=';', index=False))
    unavailable = False

    def get(url):
        if url.endswith('.txt'):
            return SimpleNamespace(text=metadata, raise_for_status=lambda: None)
        assert '/historical/2020/' in url and '03715' in url
        return SimpleNamespace(status_code=404 if unavailable else 200, content=buffer.getvalue())

    monkeypatch.setattr('openMWR.dwd_opendata.requests.get', get)
    assert update_radiosonde_data('03715', 2020, tmp_path) == [pd.Timestamp('2020-01-01')]
    path = radiosonde_year_file(tmp_path, '03715', 2020)
    data = xr.load_dataset(path)
    assert data.T.dims == ('synop', 'meastime')
    np.testing.assert_array_equal(data.meastime, [0, 10, 20])
    np.testing.assert_allclose(data.T.values[0], [283.15, 281.15, np.nan])
    np.testing.assert_allclose(data.TD.values[0], [278.15, 276.15, np.nan])
    for name in ('p', 'rh', 'lat', 'wind_dir'):
        assert np.isnan(data[name].values[0, 2])
    assert data.T.attrs['units'] == 'K' and data.p.attrs['units'] == 'hPa'
    assert data.attrs['station_id'] == '03715' and data.attrs['station_height'] == 484
    assert data.starttime.item() == pd.Timestamp('2019-12-31T22:45').value
    assert update_radiosonde_data('03715', 2020, tmp_path) == []
    original = path.read_bytes()
    unavailable = True
    assert update_radiosonde_data('03715', 2020, tmp_path) == []
    assert path.read_bytes() == original


def test_cds_request_and_netcdf_decoding_keep_site_time_and_units(dataset_project, monkeypatch):
    """Exercise the real ERA5 decoder; replace only the CDS client transport."""
    requested = []

    def retrieve(product, request):
        assert product == 'reanalysis-era5-pressure-levels'
        requested.append(request)

        def download(path):
            day = pd.Timestamp('-'.join(request[key][0] for key in ('year', 'month', 'day')))
            _era5_response(day).to_netcdf(path)
        return SimpleNamespace(download=download)

    monkeypatch.setattr('openMWR.era5.cdsapi.Client', lambda: SimpleNamespace(retrieve=retrieve))
    processor = ERA5Processor('test', dataset_project, start_date='2020-01-01', end_date='2020-01-02')
    data = processor.download_era5('2020-01-01', '2020-01-02', h_freq=12)
    assert [request['day'] for request in requested] == [['01'], ['02']]
    assert all(request['area'] == [48.25, 11.5, 48.25, 11.5]
               and request['time'] == ['00:00', '12:00'] for request in requested)
    assert set(data.data_vars) == {'T', 'rh', 'lwc', 'height'}
    assert data.T.dims == ('time', 'p') and data.attrs == {'latitude': 48.25, 'longitude': 11.5}
    np.testing.assert_array_equal(data.time, pd.date_range('2020-01-01', periods=4, freq='12h').values)
    np.testing.assert_allclose(data.height, [[550., 1550.]] * 4)
    np.testing.assert_array_equal(data.p, [950., 850.])
    np.testing.assert_allclose(data.T, [[280., 274.]] * 4)


def _era5_response(day):
    # Deliberately reverse the times to check the decoder's ordering.
    times = (day + pd.to_timedelta([12, 0], unit='h')).values
    shape = (2, 2, 1, 1)
    values = {'t': [280., 274.], 'r': [50., 40.], 'clwc': [1e-4, 0.],
              'z': np.array([550., 1550.]) * 9.81}
    return xr.Dataset({name: (('valid_time', 'pressure_level', 'latitude', 'longitude'),
                              np.broadcast_to(np.array(value)[None, :, None, None], shape))
                       for name, value in values.items()},
                      coords={'valid_time': times, 'pressure_level': [950., 850.],
                              'latitude': [48.25], 'longitude': [11.5], 'expver': 1, 'number': 0})
