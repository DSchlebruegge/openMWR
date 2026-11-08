"""Profile completion, cloud choices and RT microphysics."""

import tomllib

import numpy as np
import pandas as pd
import pytest
import tomli_w
import xarray as xr

from openMWR.datasets import DatasetWorkflow
from openMWR.paths import dataset_file, radiosonde_year_file
from openMWR.xr_utils import change_coord_of_da, integrate


from tests.fixtures.datasets import dataset_project, raw_profiles


def test_profile_interpolation_preserves_endpoints_and_trapezoidal_columns():
    profile = xr.DataArray([0., 1., 0.], dims='height', coords={'height': [0., 100., 300.]})
    xr.testing.assert_identical(change_coord_of_da(profile, profile.height.values), profile)
    np.testing.assert_allclose(change_coord_of_da(profile, [100.]), [1.])
    assert integrate(profile, 'height').item() == 150.
    assert np.isnan(integrate(profile.where(profile.height != 100.), 'height'))
    target = [-10., 0., 300., 310.]
    np.testing.assert_allclose(change_coord_of_da(profile, target), [np.nan, 0., 0., np.nan])
    np.testing.assert_allclose(change_coord_of_da(profile, target, extrapolate_lower_end=True,
                                                extrapolate_upper_end=True), [-.1, 0., 0., -.05])


def test_era5_moisture_uses_water_relative_humidity_and_volumetric_cloud_water():
    from openMWR.era5 import _convert_era5_moisture

    raw = xr.Dataset({'T': ('p', [283.16, 263.16, 243.16]),
                      'rh': ('p', [80., 80., 80.]),
                      'lwc': ('p', [0.0001, 0.0002, 0.])},
                     coords={'p': [900., 700., 500.]})
    raw.lwc.attrs['units'] = 'kg kg**-1'
    original = raw.copy(deep=True)
    result = _convert_era5_moisture(raw)
    # Frozen IFS mixed-phase Tetens values and moist ideal-gas density,
    # evaluated independently with 45-digit decimal arithmetic.
    np.testing.assert_allclose(result.rh, [80., 74.90484720953636, 59.86739415346871], rtol=1e-12)
    np.testing.assert_allclose(result.lwc, [0.11026709944912659, 0.1851121700408729, 0.], rtol=1e-12)
    assert result.rh.attrs['reference'] == 'water'
    assert result.lwc.attrs['units'] == 'g/m^3'
    xr.testing.assert_identical(raw, original)


def test_cloud_threshold_and_adiabatic_parameters_change_liquid_water():
    from openMWR.cloud import CloudColumn, CloudModelConfig
    from openMWR.radiosonde import add_cloud_to_ds

    heights = np.arange(0., 2100., 100.)
    profile = xr.Dataset({'T': ('height', 285 - 0.006 * heights),
                          'p': ('height', 1000 * np.exp(-heights / 7500)),
                          'rh': ('height', np.where((heights >= 500) & (heights <= 1500), 97., 50.))},
                         coords={'height': heights})
    standard = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15, adiabatic_fraction="power_law", a=0.7, b=2.3).lwc
    assert standard.max() > 0
    clear = add_cloud_to_ds(profile.copy(), rh_threshold=0.98, min_temperature=253.15, adiabatic_fraction="power_law", a=0.7, b=2.3).lwc
    assert clear.max() == 0
    halved = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15, adiabatic_fraction="power_law", a=0.35, b=2.3).lwc
    np.testing.assert_allclose(halved, standard / 2)
    different_shape = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15, adiabatic_fraction="power_law", a=0.7, b=1.0).lwc
    assert not np.allclose(different_shape, standard)
    karstens = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15, adiabatic_fraction='karstens').lwc
    adiabatic = CloudColumn(heights, profile.p.values * 100, profile.T.values,
                           profile.rh.values / 100, CloudModelConfig(rh_thres=0.96, use_f_ad=False)).calculate_lwc()
    h = np.maximum(heights - 500., 0.)
    np.testing.assert_allclose(karstens, adiabatic * np.maximum(0., 1.239 - 0.145 * np.log(h + 1)))
    assert not np.allclose(karstens, standard)
    fully_adiabatic = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=253.15,
                                    adiabatic_fraction='none').lwc
    np.testing.assert_allclose(fully_adiabatic, adiabatic)
    warmer_threshold = add_cloud_to_ds(profile.copy(), rh_threshold=0.96, min_temperature=283.,
                                      adiabatic_fraction='none').lwc
    assert warmer_threshold.max() == 0
    # The first clear level can also be the final level of a shorter profile.
    short = add_cloud_to_ds(profile.isel(height=slice(0, 17)).copy(), rh_threshold=0.96,
                           min_temperature=253.15, adiabatic_fraction='power_law', a=0.7, b=2.3).lwc
    xr.testing.assert_equal(short, standard.sel(height=short.height))


@pytest.mark.parametrize('completion', ['monthly_mean', 'filter'])
def test_profile_quality_limits_select_source_profiles_before_interpolation(dataset_project, completion):
    path = dataset_project / 'sites/test/datasets/profiles.toml'
    values = tomllib.loads(path.read_text())
    values['preparation']['incomplete_profiles'] = completion
    path.write_text(tomli_w.dumps(values))
    times = pd.date_range('2020-01-01', periods=3, freq='12h').values
    raw = raw_profiles(times)
    # One shallow ascent, one ascent with too few valid levels, one valid ascent.
    raw['height'][0] = np.linspace(484, 12000, raw.sizes['meastime'])
    raw['T'][1, 450:] = np.nan
    path = radiosonde_year_file(dataset_project, '03715', 2020)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_netcdf(path)
    workflow = DatasetWorkflow(dataset_project, 'test')
    workflow.build('profiles')
    selected = xr.load_dataset(dataset_file(dataset_project, 'test', 'profiles', '03715'))
    np.testing.assert_array_equal(selected.synop, times[2:])
    values = workflow.recipe('profiles').values
    values['preparation']['quality'] = {'min_top_height': 10000, 'min_levels': 400}
    values['preparation']['cut_off_at_mwr_height'] = True
    (dataset_project / 'sites/test/datasets/profiles.toml').write_text(tomli_w.dumps(values))
    with pytest.raises(ValueError, match='scientific settings'):
        DatasetWorkflow(dataset_project, 'test').build('profiles')
    DatasetWorkflow(dataset_project, 'test').build('profiles', rebuild=True)
    selected = xr.load_dataset(dataset_file(dataset_project, 'test', 'profiles', '03715'))
    assert selected.surface_altitude.item() == 550.
    # The compressed shallow ascent has pressure below the monthly completion
    # at its join, so the pressure filter also rejects it after filling.
    np.testing.assert_array_equal(selected.synop, times[1:] if completion == 'monthly_mean' else times[2:])
    assert np.isfinite(selected[['T', 'p', 'rh', 'lwc']].to_array()).all()


def test_monthly_completion_rejects_profiles_without_upper_level_reference(dataset_project):
    times = pd.date_range('2020-01-01', periods=2, freq='12h').values
    raw = raw_profiles(times).isel(meastime=slice(0, 520))
    path = radiosonde_year_file(dataset_project, '03715', 2020)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_netcdf(path)
    # Both ascents pass the source-quality checks but neither reaches the top grid level.
    with pytest.raises(ValueError, match='No complete profiles remain for RT'):
        DatasetWorkflow(dataset_project, 'test').build('profiles')
    assert not dataset_file(dataset_project, 'test', 'profiles', '03715').is_file()


def test_preparation_rejects_pressure_increases_plateaus_and_nonpositive_values(dataset_project, caplog):
    times = pd.date_range('2020-01-01', periods=4, freq='12h').values
    raw = raw_profiles(times)
    raw['p'][0, :20] = np.linspace(900., 1000., 20)
    raw['p'][1, :20] = 950.
    raw['p'][2, :20] = -1.
    path = radiosonde_year_file(dataset_project, '03715', 2020)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_netcdf(path)
    DatasetWorkflow(dataset_project, 'test').build('profiles')
    selected = xr.load_dataset(dataset_file(dataset_project, 'test', 'profiles', '03715'))
    np.testing.assert_array_equal(selected.synop, times[3:])
    assert 'Rejected 3 profiles: pressure must be positive and strictly decreasing' in caplog.text


@pytest.mark.parametrize('backend', ['torchMWRT', 'pyrtlib'])
def test_infrared_microphysics_reaches_rt_and_incomplete_inputs_are_rejected(dataset_project, monkeypatch, backend):
    import openMWR.run_RT as rt

    altitudes = []
    def infrared(*args, **kwargs):
        altitudes.append(kwargs['surface_altitude'])
        return float(np.mean(args[-1]))
    monkeypatch.setattr(rt, 'calculate_IR_band_TB', infrared)
    profiles = xr.Dataset({
        'T': (('time', 'height'), [[285., 282., 279.], [285., 282., 279.]]),
        'p': (('time', 'height'), [[1000., 950., 900.], [1000., 950., 900.]]),
        'rh': (('time', 'height'), [[80., 80., 80.], [80., 80., 80.]]),
        'lwc': (('time', 'height'), [[0., 0.5, 0.], [0., 0.5, 0.]]),
    }, coords={'time': pd.date_range('2020-01-01', periods=2), 'height': [0., 500., 1000.],
               'surface_altitude': ('time', [484., 2656.])})
    scratch = dataset_project / 'data/libRadtran'
    scratch.mkdir(exist_ok=True)
    other_build = scratch / 'other-build.dat'
    other_build.write_text('Input owned by another running build')
    options = dict(site='test', project_dir=dataset_project, freqs=np.array([22.24]),
                   angles=np.array([4.2, 90.]), num_of_processes=1, absmdl='R24', RT_model=backend,
                   ray_tracing=True)
    first = rt.run_RT(profiles, **options, infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
    second = rt.run_RT(profiles, **{**options, 'angles': options['angles'][::-1].copy()},
                       infrared={'droplet_number_density': 1600e6, 'min_effective_radius': 3.0})
    cloudy_radius = (0.5 / (200e6 * 4 / 3 * np.pi * 1e6)) ** (1 / 3) * 1e6
    np.testing.assert_allclose(first.TB_IR, (2 * 2.5 + cloudy_radius) / 3)
    np.testing.assert_allclose(second.TB_IR, (2 * 3.0 + cloudy_radius / 2) / 3)
    xr.testing.assert_equal(first.TB, second.TB.sel(ang=first.ang))
    flat = rt.run_RT(profiles, **{**options, 'ray_tracing': False},
                     infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
    xr.testing.assert_allclose(first.TB.sel(ang=90.), flat.TB.sel(ang=90.))
    assert bool((abs(first.TB.sel(ang=4.2)-flat.TB.sel(ang=4.2)) > 1e-4).all())
    assert first.attrs['ray_tracing'] == 1 and flat.attrs['ray_tracing'] == 0
    assert altitudes == [484., 2656., 484., 2656., 484., 2656.]
    assert other_build.read_text() == 'Input owned by another running build'
    with pytest.raises(ValueError, match='RT requires finite surface_altitude'):
        rt.run_RT(profiles.drop_vars('surface_altitude'), **options,
                  infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
    for altitude in (xr.DataArray([484., 500., 600.], dims='height'),
                     xr.DataArray(np.full((2, 3), 484.), dims=('time', 'height'))):
        with pytest.raises(ValueError, match='surface_altitude must be scalar or have only the time dimension'):
            rt.run_RT(profiles.assign_coords(surface_altitude=altitude), **options,
                      infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
    invalid = profiles.copy(deep=True)
    invalid['T'][0, -1] = np.nan
    with pytest.raises(ValueError, match='RT requires complete profiles'):
        rt.run_RT(invalid, **options, infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
    invalid = profiles.copy(deep=True)
    invalid['p'][0, :2] = [950., 1000.]
    with pytest.raises(ValueError, match='positive pressure strictly decreasing with height.*2020-01-01'):
        rt.run_RT(invalid, **options, infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
    invalid = profiles.copy(deep=True)
    invalid['lwc'][0, 1] = 1.0  # At 1 cm^-3, this gives about 62 micrometres.
    with pytest.raises(ValueError, match='Calculated droplet effective radii must be below 60.*libRadtran'):
        rt.run_RT(invalid, **options, infrared={'droplet_number_density': 1e6, 'min_effective_radius': 2.5})

    # One unsupported ray must not be saved as a partial atmosphere or silently
    # dropped by the per-profile multiprocessing wrapper.
    duct = profiles.copy(deep=True).assign_coords(height=[0., 10., 100.])
    duct['T'][0] = [285., 285., 285.]
    duct['p'][0] = [1000., 999.9, 999.]
    duct['p'][1] = [1000., 300., 100.]
    low_angles = {**options, 'angles': np.array([0.1, 90.])}
    valid = rt.run_RT(duct.isel(time=slice(0, 1)).assign_coords(surface_altitude=484.), **low_angles,
                      infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
    assert np.isfinite(valid.TB).all()
    for failing in (duct, duct.isel(time=slice(1, 2))):
        with pytest.raises(ValueError, match='ducting|complete results|No data processed') as error:
            rt.run_RT(failing, **low_angles,
                      infrared={'droplet_number_density': 200e6, 'min_effective_radius': 2.5})
        message = str(error.value)
        assert '2020-01-02' in message
        assert 'profile-base altitude 2656' in message
        assert 'elevation angles [0.1, 90.0]' in message
