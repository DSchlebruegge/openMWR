"""Six-sample retrieval experiments with real recipes, checkpoints and study storage."""


import numpy as np
import xarray as xr

from openMWR.recipes import RetrievalRecipe
from openMWR.site import create_site
from openMWR.train import TrainingWorkflow


def retrieval_project(tmp_path, kind='mlr', mode='zenith', suffix='variant', search=True):
    create_site('test', tmp_path, gen='G5', mwr_height=550)
    directory = tmp_path / 'sites/test/retrieval/example'
    directory.mkdir()
    settings = f'''schema_version = 1
[model]
kind = "{kind}"
mode = "{mode}"
freqs = [22.24, 31.4]
norm_functions = {{ TB = "z_norm", T = "z_norm", rh = "z_norm", lwc = "{'max_norm' if kind == 'mlr' else 'log_norm'}", lwp = "{'max_norm' if kind == 'mlr' else 'log_norm'}", iwv = "z_norm" }}
'''
    if kind != 'spectral':
        settings += 'heights = [0.0, 1000.0]\nadditional_input_vars = []\n'
        if kind == 'nn':
            bounds = '[2000, 10000]' if mode == 'bls' else '[10000]'
            values = '[1.0, 0.5, 0.25]' if mode == 'bls' else '[1.0, 0.5]'
            profile = f'{{ upper_bounds = {bounds}, values = {values} }}'
            weights = ', '.join(f'{name} = {profile}' for name in ('T', 'rh', 'lwc'))
            settings += f'hidden_layers = [4]\ndropout_rate = 0.0\nloss_weights = {{ {weights}, lwp = 1.0, iwv = 1.0 }}\n'
    else:
        settings += 'hidden_layer_size = 4\n'
    if mode == 'bls':
        settings += 'angles = [90.0, 30.0]\nvary_input_noise_with_angle_linear = true\n'
    training = ('repeats_for_linear_regression = 1\n' if kind == 'mlr' else
                'epochs = 1\nbatch_size = 4\nlr_height = 0.002\nlr_low = 1e-7\nepoch_max_lr = 50\nloss_fn = "MAE"\nweight_decay = 1e-5\n')
    variant = f'dataset = "training_{suffix}"\n' if suffix else 'dataset = "training"\n'
    settings += f'''
[data]
sources = ["03715"]
{variant}

[training]
seed = 42
{training}

[input_noise]
method = "standard"
TB = 0.0
{'TB_angle_factor = 3.0' if mode == 'bls' else ''}

[optimization]
seed = 42
days_of_month = {{ first = 1, last = 20 }}
score_weights = {{ T = 1.0, rh = 0.1 }}
n_trials = 1
data_source = "radiosonde"
dataset = "analysis_radiosonde"
'''
    if kind == 'spectral':
        settings = settings.replace('score_weights = { T = 1.0, rh = 0.1 }', 'score_weights = { TB = 1.0 }')
    if search:
        if kind == 'mlr':
            settings = settings.replace('TB = 0.0', 'TB = { low = 0.001, high = 0.002 }')
        else:
            settings = settings.replace('lr_height = 0.002', 'lr_height = { low = 0.001, high = 0.002 }')
    recipe = directory / 'retrieval.toml'
    recipe.write_text(settings)
    import tomli_w
    dataset_name = f'training_{suffix}' if suffix else 'training'
    evaluation = {'schema_version': 1, 'retrievals': ['example'],
                  'observations': {'dataset': 'analysis_radiosonde', 'data_source': 'radiosonde',
                                   'station_id': '03715', 'variables': ['TB'] if kind == 'spectral' else ['T', 'rh'],
                                   'days_of_month': {'first': 21, 'last': 31}},
                  'test': {'dataset': dataset_name, 'sources': ['03715'],
                           'variables': ['TB'] if kind == 'spectral' else ['T', 'rh', 'lwc', 'lwp']}}
    if kind != 'spectral':
        evaluation['observations']['height_range'] = [0, 10000]
        evaluation['test']['height_range'] = 'model'
    evaluation_dir = tmp_path / 'sites/test/evaluations'
    evaluation_dir.mkdir(exist_ok=True)
    (evaluation_dir / 'example.toml').write_text(tomli_w.dumps(evaluation))
    for name, definition in {
        'radiosondes': {'kind': 'radiosonde', 'stations': ['03715'], 'start_year': 1990, 'end_year': 'current_year'},
        'profiles': {'kind': 'profiles', 'inputs': {'source': 'radiosondes'}, 'preparation': {'heights': [0.0, 1000.0], 'incomplete_profiles': 'monthly_mean', 'cloud': {'rh_threshold': 0.96, 'min_temperature': 253.15, 'adiabatic_fraction': 'power_law', 'a': 0.7, 'b': 2.3}, 'quality': {'min_top_height': 15000, 'min_levels': 500}}},
        'simulation_r20': {'kind': 'simulation', 'inputs': {'profiles': 'profiles'},
                          'radiative_transfer': {'backend': 'torchMWRT', 'absorption_model': 'R20',
                                                 'frequencies': [22.24, 31.4], 'angles': [90.0, 30.0],
                                                 'infrared': {'droplet_number_density': 200e6, 'min_effective_radius': 2.5}}},
        dataset_name: {'kind': 'split', 'inputs': {'simulation': 'simulation_r20'},
                      'split': {'test_days_of_month': {'first': 1, 'last': 3}, 'validation_days_of_month': {'first': 15, 'last': 17}}},
    }.items():
        (tmp_path / f'sites/test/datasets/{name}.toml').write_text(tomli_w.dumps(definition))
    for name in (dataset_name, 'analysis_radiosonde', 'analysis_era5'):
        (tmp_path / f'data/sites/test/{name}').mkdir(parents=True, exist_ok=True)
    times = np.array(['2024-01-10', '2024-01-11', '2024-01-12', '2024-01-21', '2024-01-22', '2024-01-23'], dtype='datetime64[ns]')
    data = retrieval_samples(times)
    split_dates = {
        'training': ['2023-01-04', '2023-01-05', '2023-01-06', '2023-02-04', '2023-02-05', '2023-02-06'],
        'validation': ['2023-01-15', '2023-01-16', '2023-01-17', '2023-02-15', '2023-02-16', '2023-02-17'],
        'test': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-02-01', '2023-02-02', '2023-02-03'],
    }
    for seed, (split, dates) in enumerate(split_dates.items(), start=20):
        values = retrieval_samples(np.array(dates, dtype='datetime64[ns]'), seed=seed)
        values.to_netcdf(tmp_path / f'data/sites/test/{dataset_name}/{split}_03715.nc')
    observations = xr.concat([data, data.assign(T=data.T + 0.5)], dim='data_source').assign_coords(data_source=['radiosonde', 'hatpro']).expand_dims(station=['03715'])
    observations.sel(ang=90, drop=True).to_netcdf(tmp_path / 'data/sites/test/analysis_radiosonde/zenith.nc')
    observations.to_netcdf(tmp_path / 'data/sites/test/analysis_radiosonde/bls.nc')
    from openMWR.datasets import DatasetWorkflow
    DatasetWorkflow(tmp_path, 'test').record(dataset_name)
    return recipe, data


def training_workflow(tmp_path, manual=False):
    if not manual:
        return TrainingWorkflow.from_recipe(tmp_path, 'test', 'example')
    recipe = RetrievalRecipe.load(tmp_path, 'test', 'example')
    workflow = TrainingWorkflow(recipe.create_model(), 'test', 'cpu', tmp_path)
    workflow.change_hyper_params(**recipe.training_params(workflow))
    workflow.add_input_noise_params(**recipe.input_noise_params(workflow))
    data = recipe.section('data')
    workflow.load_training_data(sources=data.get('sources'), dataset=data['dataset'])
    return workflow


def retrieval_samples(times, seed=12):
    """A learnable linear atmosphere; independent samples for each partition."""
    rng = np.random.default_rng(seed)
    tb = 100 + rng.normal(size=(len(times), 2, 2))
    x, y = tb[:, 0, 0] - 100, tb[:, 0, 1] - 100
    return xr.Dataset({
        'T': (('time', 'height'), np.stack([280 + x + 2*y, 274 + 2*x - y], axis=1)),
        'rh': (('time', 'height'), np.stack([50 + x, 45 + y], axis=1)),
        'lwc': (('time', 'height'), np.stack([1 + .01*x, .5 + .01*y], axis=1)),
        'lwp': ('time', 20 + x), 'iwv': ('time', 10 + y),
        'TB': (('time', 'ang', 'frq'), tb),
    }, coords={'time': times, 'height': [0., 1000.], 'ang': [90., 30.], 'frq': [22.24, 31.4]})
