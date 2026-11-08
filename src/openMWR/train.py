from openMWR.observations import load_analysis_dataset, select_reference_sources, reference_source
from openMWR.site import load_site_config
import torch
from torch import nn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
import copy
import os
import optuna
import logging
from functools import wraps
from collections import namedtuple

from typing import Union, List

from openMWR.utils import progress_bar, is_called_from_notebook, progress_log
from openMWR.xr_utils import load_combined_dataset, to_1d_tensor
from openMWR.predict import create_tensor_from_ds, create_ds_from_y, validate_model_data
from openMWR.normalization import normalize, denormalize, create_stats, get_input_noise, normalized_input_noise
from openMWR.devices import resolve_device, device_name
from openMWR.calc_errors import cal_RMSE
from openMWR.models import Model, BaseModel, SpectralConsistencyModel
from openMWR.paths import dataset_file, retrieval_dir, checkpoint_file, optimization_file
from openMWR.provenance import plain_value, retrieval_log, software_identity, utc_now, dataset_id
from openMWR.config import (TRAINING_PARAMETERS, TRAINING_SETTINGS, TRAINING_NAMES, check_keys, check_name, require_keys,
                            training_parameters, input_noise_parameters, resolve_parameter, load_grid,
                            validate_training_settings, validate_parameter_usage, check_learning_rates,
                            check_seed, apply_seed, check_time_selection, check_split_days, select_time,
                            TIME_SELECTION_KEYS, check_score_weights)

logger = logging.getLogger(__name__)
_MLR_FITTING = 'least squares with expected input-noise penalty'


def _experiment_definition(experiment, variables):
    """Compare scientific settings, excluding names, data identities and unused normalizers."""
    if experiment is None:
        return None
    definition = {key: value for key, value in experiment.items() if key != 'data'}
    structure = {key: value for key, value in definition['model'].items() if key != 'name'}
    if 'norm_functions' in structure:
        structure['norm_functions'] = {
            key: value for key, value in structure['norm_functions'].items() if key in variables}
    definition['model'] = structure
    return definition


def _logged_run(action):
    def decorate(method):
        @wraps(method)
        def run(self, *args, **kwargs):
            with retrieval_log(self.project_dir, self.site, self.model.name, action):
                self._run_record = {'action': action, 'started_utc': utc_now(),
                                    'software': software_identity(), 'device': self.device,
                                    'device_name': device_name(self.device)}
                self.logger.info('Training device: %s (%s).', self.device, self._run_record['device_name'])
                try:
                    return method(self, *args, **kwargs)
                finally:
                    self._run_record = None
        return run
    return decorate

#__all__ = ["separate_training_data", "TrainingWorkflow"]

class TrainingWorkflow:
    """
    Training workflow for openMWR retrieval models.

    This class manages the full workflow for training a retrieval model,
    including loading and preprocessing training data, configuring
    hyperparameters, adding input noise, model training, and optionally
    performing Optuna-based hyperparameter optimization.

    Parameters
    ----------
    model : Model
        Model instance derived from `BaseModel` used for the retrieval.
    site : str
        Site identifier for which the model is trained.
    device : str, default='auto'
        Device identifier. Auto selects CUDA, then MPS, then CPU.
    project_dir : str
        Working-project root containing data/ and sites/.

    Attributes
    ----------
    model : Model
        The retrieval model to be trained.
    site : str
        Site for which the training data is loaded.
    device : str
        Device used for training computations.
    project_dir : str
        Working-project root containing data/ and sites/.
    hyper_params : dict
        Dictionary of training and input-noise hyperparameters.
    training_data : xr.Dataset
        Preprocessed training dataset.
    validation_data : xr.Dataset
        Preprocessed validation dataset.
    omb : Omb_Analysis
        OMB (observation-minus-background) analysis instance, available
        after calling `make_omb_analysis()`.

    Examples
    --------

    Creating a model and a workflow for training:

    .. code-block:: python

        from openMWR.models import Model
        from openMWR.train import TrainingWorkflow
        from openMWR.config import load_grid
        profile_weights = {'upper_bounds': [10000], 'values': [1., .5]}
        model = Model('NN_1', load_grid('.', 'heights', 'levels_208_30km'),
                      load_grid('.', 'frequencies', 'hatpro'),
                      output_vars=['T', 'rh', 'lwc', 'lwp', 'iwv'],
                      loss_weights={'T': profile_weights, 'rh': profile_weights,
                                    'lwc': profile_weights, 'lwp': 1., 'iwv': 1.})
        workflow = TrainingWorkflow(model, site='munich', device='cuda', project_dir='.')
        workflow.load_training_data(['03715'], dataset='training')

    Training a model with default hyperparameters and input noise:

    .. code-block:: python

        workflow.add_standard_training_hyper_params()
        workflow.add_input_noise_params()
        workflow.train()

    Or optimizing input noise using Optuna:

    .. code-block:: python

        workflow.add_standard_training_hyper_params()
        workflow.add_input_noise_params(optimization=True)
        workflow.optimize(n_trials=50)

    Or use input noise and hyperparameters from an existing model:

    .. code-block:: python

        workflow.get_hyper_params_from_existing_model('NN_opt_era5')
        workflow.train()

    Or use input noise and hyperparameters from an existing model but optimize some parameters:

    .. code-block:: python

        workflow.get_hyper_params_from_existing_model('NN_opt_era5')
        workflow.change_hyper_params(
            batch_size=workflow.suggest_int(256, 2048, step=256),
            lr_height=workflow.suggest_float(0.0001, 0.01, log=True)
        )
        workflow.optimize(n_trials=20)

    Or create an OMB analysis and use it for input noise selection:

    .. code-block:: python

        workflow.make_omb_analysis(['03715'], data_source='radiosonde', dataset='analysis_radiosonde', days_of_month={'first': 1, 'last': 20})
        workflow.select_omb_input_noise_hyper_params(statistic='std', factor=1.0)
        workflow.train()

    """
    def __init__(self, model: BaseModel, site: str, device: str = 'auto', project_dir: str = '.'):
        self.model = model
        self.site = site
        self.device = resolve_device(device)
        self.project_dir = project_dir

        self.hyper_params = {}
        self.recipe = None
        self.site_config = None

        self.logger = logging.getLogger(__name__)

        self.suggest_int = namedtuple('suggest_int', ['low', 'high', 'step', 'log'], defaults=[None, None, 1, False])
        self.suggest_float = namedtuple('suggest_float', ['low', 'high', 'step', 'log'], defaults=[None, None, None, False])
        self.suggest_categorical = namedtuple('suggest_categorical', ['choices'])

    @classmethod
    def from_recipe(cls, project_dir, site, retrieval, device='auto'):
        """Load a recipe and configure a workflow; no training or download runs.

        The returned datasets and settings remain available for custom Python
        processing before calling train() or optimize().
        """
        from openMWR.recipes import RetrievalRecipe

        recipe = RetrievalRecipe.load(project_dir, site, retrieval)
        workflow = cls(recipe.create_model(), site, device, recipe.project_dir)
        workflow.recipe = recipe
        workflow.site_config = recipe.site_config
        workflow.change_hyper_params(**recipe.training_params(workflow))
        noise_settings = recipe.section('input_noise')
        method = noise_settings['method']
        noise = recipe.input_noise_params(workflow)
        workflow.add_input_noise_params(**noise)
        data = recipe.section('data')
        workflow.load_training_data(sources=data.get('sources'), dataset=data['dataset'])
        bias = recipe.section('bias_correction')
        if bias:
            sources = bias.get('sources')
            workflow.make_omb_analysis(sources, data_source=bias.get('data_source'), dataset=bias['dataset'],
                **{key: bias[key] for key in (*TIME_SELECTION_KEYS, 'max_lwp') if key in bias})
            frequencies = (load_grid(recipe.project_dir, 'frequencies', bias['frequencies'], label='bias_correction.frequencies')
                           if 'frequencies' in bias else None)
            workflow.make_bias_correction(factor=bias['factor'], specific_freqs=frequencies)
        if method == 'omb':
            source = noise_settings.get('data_source')
            sources = noise_settings.get('sources')
            selection = {key: noise_settings[key] for key in TIME_SELECTION_KEYS if key in noise_settings}
            if hasattr(workflow, 'omb') and workflow.omb.dataset == noise_settings['dataset']:
                source = reference_source(workflow.omb.an_data, source)
            if (not hasattr(workflow, 'omb') or workflow.omb.data_source != source
                    or workflow.omb.station_ids_for_analysis != sources or workflow.omb.dataset != noise_settings['dataset']
                    or workflow.omb.selection != selection or workflow.omb.max_lwp != noise_settings.get('max_lwp')):
                workflow.make_omb_analysis(sources, data_source=source, dataset=noise_settings['dataset'],
                    max_lwp=noise_settings.get('max_lwp'), **selection)
            workflow.select_omb_input_noise_hyper_params(factor=noise_settings['factor'],
                statistic=noise_settings['statistic'], frequency_bands=noise_settings.get('frequency_bands'))
            workflow.hyper_params.update({f'input_noise_{key}': value for key, value in noise.items()})
        workflow.validate()
        return workflow

    @property
    def requires_optimization(self):
        """Whether the current parameters contain unresolved search definitions."""
        suggestions = (self.suggest_int, self.suggest_float, self.suggest_categorical)
        return any(isinstance(value, suggestions) for value in self.hyper_params.values())

    def validate(self):
        """Check current Python settings and prepared datasets before computation."""
        self.device = resolve_device(self.device, log=False)
        require_keys(self.hyper_params, training_parameters(self.model.is_MLR, self.hyper_params.get('loss_fn')), 'training')
        searches = {key: value for key, value in self.hyper_params.items()
                    if isinstance(value, (self.suggest_int, self.suggest_float, self.suggest_categorical))}
        if not self.model.is_MLR:
            searches['loss_fn'] = self.hyper_params['loss_fn']
        validate_parameter_usage(self.model.is_MLR, searches)
        for key, value in self.hyper_params.items():
            candidates = (value.choices if isinstance(value, self.suggest_categorical)
                          else (value.low, value.high) if isinstance(value, (self.suggest_int, self.suggest_float))
                          else (value,))
            for candidate in candidates:
                validate_training_settings({key: candidate})
        check_learning_rates(self.hyper_params)
        noise_keys = {f'input_noise_{key}' for key in input_noise_parameters(self.model)}
        missing_noise = noise_keys - self.hyper_params.keys()
        if missing_noise:
            raise ValueError(f"Missing input noise settings: {', '.join(sorted(missing_noise))}. Call add_input_noise_params() first.")
        expected = xr.merge([self.model.input, self.model.output], compat='override')
        missing_norm = set(expected.data_vars) - self.model.norm_functions.keys()
        if missing_norm:
            raise ValueError(f"Missing normalization functions: {', '.join(sorted(missing_norm))}.")
        for label in ('training_data', 'validation_data'):
            if not hasattr(self, label):
                raise ValueError('Load training and validation data before training.')
            data = getattr(self, label)
            validate_model_data(expected, data, label)
        # All noise and angle-factor ranges are nonnegative. Their maxima
        # cover every possible nonzero-noise/normalization combination.
        normalized_input_noise(
            self.model, self._input_noise_limits(),
            normalization=create_stats(self.model, self.training_data[self.model.input_vars]))
        return self

    def _input_noise_limits(self):
        """Resolve the largest allowed noise values without sampling a trial."""
        limits = {}
        for key, value in self.hyper_params.items():
            if key.startswith('input_noise_'):
                if isinstance(value, self.suggest_categorical):
                    value = max(value.choices)
                elif isinstance(value, (self.suggest_int, self.suggest_float)):
                    value = value.high
                limits[key] = value
        return limits

    def _save_model(self, model, hyper_params, **details):
        metadata = {**self._run_record, 'saved_utc': utc_now(),
                    'site': self.site, 'retrieval': model.name,
                    'model_class': type(model).__name__,
                    'effective': plain_value({'model': model.checkpoint_structure(), 'hyper_params': hyper_params}),
                    **details}
        if self.recipe is not None:
            metadata['recipe'] = plain_value(self.recipe.values)
        model.save(self.site, self.project_dir, hyper_params, training_metadata=metadata)

    def _prepare_run(self, run_mode, optimization=False):
        """Apply the selected retrieval's fresh/continue choice after validation."""
        check_name(self.site, 'site')
        check_name(self.model.name, 'retrieval')
        if run_mode not in (None, 'fresh', 'continue'):
            raise ValueError("run_mode must be 'fresh' or 'continue'.")
        if run_mode == 'continue' and not optimization:
            raise ValueError('Continuing is available only for optimization. Use --fresh or run_mode="fresh" to train one new model.')
        study_path = optimization_file(self.project_dir, self.site, self.model.name)
        storage = f'sqlite:///{study_path}'
        has_database = study_path.is_file()
        path = checkpoint_file(self.project_dir, self.site, self.model.name)
        if run_mode is None and (path.is_file() or has_database):
            choice = 'Choose --fresh to replace its model and study'
            if optimization:
                choice += ', or --continue to add optimization trials'
            raise ValueError(f'Results already exist for this retrieval. {choice}; Python callers can set run_mode explicitly.')
        if run_mode == 'continue' and not has_database:
            raise ValueError('No optimization study exists for this retrieval. Use --fresh or run_mode="fresh" to start one.')
        if run_mode == 'fresh':
            for artifact in (path, path.with_name('training_progress.pdf'), study_path):
                artifact.unlink(missing_ok=True)
        self.logger.info('%s retrieval %s.',
                         'Continuing' if run_mode == 'continue' else 'Starting fresh', self.model.name)
        return storage

    def _log_training_setup(self, optimization=None):
        """Record the actual fitting data and the separate trial-ranking rule."""
        setup = {'training_sources': self.hyper_params.get('sources', self.hyper_params.get('station_ids', [])),
                 'training_selection': self.hyper_params.get('training_selection', {}),
                 'training_dataset': self.hyper_params.get('training_dataset', 'training'),
                 'fitting': _MLR_FITTING if self.model.is_MLR else self.hyper_params['loss_fn']}
        self.logger.info('Training sources: %s; selection: %s; dataset: %s.', setup['training_sources'], setup['training_selection'], setup['training_dataset'])
        self.logger.info('Fitting: %s.', setup['fitting'])
        if optimization is not None:
            score = f"sum of variable weights {optimization['score_weights']} times mean RMSE"
            if hasattr(self.model, 'loss_weights'):
                score += ', with model height weights on matching reference heights'
            setup['optimization'] = {**optimization, 'score': score}
            self.logger.info('Optimization reference: %s; time selection: %s; sources: %s.', optimization['data_source'], {key: optimization[key] for key in TIME_SELECTION_KEYS if key in optimization}, optimization.get('sources', 'all sources in the analysis dataset'))
            self.logger.info('Optimization score (separate from fitting loss): %s.', score)
        self._run_record['setup'] = plain_value(setup)

    def validate_optimization(self, data_source=None, dataset=None, **options):
        """Check search definitions and analysis inputs without starting trials."""
        self._optimization_inputs(data_source=data_source, dataset=dataset, **options)
        return self

    def _optimization_inputs(self, *, n_trials=None, data_source=None, dataset=None, seed=None,
                             start_date=None, end_date=None, days_of_month=None, score_weights=None, sources=None):
        """Load, select and validate the reference data used to rank trials."""
        self.validate()
        options = {'n_trials': n_trials, 'data_source': data_source, 'dataset': dataset,
                   'seed': seed, 'start_date': start_date, 'end_date': end_date,
                   'days_of_month': days_of_month, 'score_weights': score_weights, 'sources': sources}
        settings = {}
        if self.recipe is not None:
            settings = dict(self.recipe.section('optimization'))
        elif isinstance(self.model, (Model, SpectralConsistencyModel)):
            settings = dict(self._standard_recipe(optimization=True)['optimization'])
            settings['score_weights'] = {key: value for key, value in settings['score_weights'].items() if key in self.model.output}
        settings.update({key: value for key, value in options.items() if value is not None})
        if not self.requires_optimization:
            raise ValueError('Optimization needs a search definition in training or input_noise: use low/high or choices on a parameter.')
        require_keys(settings, ('n_trials', 'dataset', 'seed', 'score_weights'), 'optimization')
        check_seed(settings['seed'], 'optimization.seed')
        check_time_selection(settings, 'optimization')
        check_score_weights(settings['score_weights'], self.model.output.data_vars)
        if type(settings['n_trials']) is not int or settings['n_trials'] < 1:
            raise ValueError('n_trials must be a positive integer.')
        analysis_name = settings['dataset']
        path = dataset_file(self.project_dir, self.site, analysis_name, bls=self.model.is_bls_model)
        if not path.is_file():
            raise FileNotFoundError(f'Missing optimization analysis dataset: {path}. Prepare it before optimization.')
        self.site_config = load_site_config(self.site, self.project_dir, config=self.site_config)
        data = load_analysis_dataset(self.project_dir, self.site, analysis_name, bls=self.model.is_bls_model, site_config=self.site_config)
        settings['data_source'] = source = reference_source(data, settings.get('data_source'))
        data = select_reference_sources(data, source, settings.get('sources'))
        data = select_time(data, **{key: settings[key] for key in TIME_SELECTION_KEYS if key in settings})
        if not self.model.is_bls_model and 'ang' in data.dims:
            data = data.sel(ang=90, drop=True)
        validate_model_data(self.model.input, data.sel(data_source='hatpro', drop=True), 'optimization observations')
        variables = [key for key, weight in settings['score_weights'].items() if weight > 0]
        output = self.model.output[variables]
        if 'height' in output.dims and data.sizes.get('height', 0):
            # Scoring aligns model/reference heights. Require the model levels
            # inside the reference range without demanding unobserved levels.
            output = output.sel(height=slice(data.height.min().item(), data.height.max().item()))
            if not output.sizes['height']:
                raise ValueError('Optimization reference has no heights within the model height range.')
        validate_model_data(output, data.sel(data_source=source, drop=True), 'optimization reference')
        return settings, data

    def load_training_data(self, sources: List[str] = None, dataset: str = 'training'):
        """
        Load and prepare training and validation data for the retrieval model.

        Parameters
        ----------
        sources : list of str
            List of data sources used for training (e.g., radiosonde station IDs
            and optionally 'era5').
        dataset : str, optional
            Named split dataset, including its training period. Omitted sources
            use all sources in its recipe.

        Notes
        -----
        - Selects only model-relevant variables.
        - For non-BLS models, selects only 90° elevation angle if present.
        - Stores datasets in `training_data` and `validation_data`.
        """
        from openMWR.datasets import DatasetWorkflow
        self.site_config = load_site_config(self.site, self.project_dir, config=self.site_config)
        datasets = DatasetWorkflow(self.project_dir, self.site, site_config=self.site_config)
        datasets.require(dataset, sources=sources, splits=('training', 'validation'))
        if sources is None:
            sources = datasets.sources(dataset)
        settings = datasets.settings(dataset)
        training = load_combined_dataset(self.site, self.project_dir, 'training', sources, dataset=dataset)
        validation = load_combined_dataset(self.site, self.project_dir, 'validation', sources, dataset=dataset)
        self.set_training_data(training, validation)
        self.hyper_params.update({
            'training_selection': {key: settings[key] for key in (*TIME_SELECTION_KEYS, 'test_days_of_month', 'validation_days_of_month') if key in settings},
            'sources': sources,
            'training_dataset': dataset,
            'station_ids': [s for s in sources if s != 'era5']
        })

    def set_training_data(self, training: xr.Dataset, validation: xr.Dataset):
        """Prepare supplied, unnormalized data without dataset recipes.

        Keep model variables and select zenith for non-BLS models. Each dataset
        needs a time dimension. A missing station dimension becomes one source.
        Both datasets are checked before replacing the current data. Dataset
        selection and applied bias-correction records are cleared on replacement.
        Apply bias correction after setting data. Returns this workflow.
        """
        expected = xr.merge([self.model.input, self.model.output], compat='override')

        def prepare(data, label):
            if not isinstance(data, xr.Dataset):
                raise TypeError(f'{label} must be an xarray.Dataset.')
            if not self.model.is_bls_model and 'ang' in data.dims:
                if 'ang' not in data.coords or 90. not in data.ang:
                    raise ValueError(f'{label} must contain the zenith angle (90 degrees).')
                data = data.sel(ang=90., drop=True)
            if 'station' not in data.dims:
                data = data.expand_dims('station') if 'station' in data.coords else data.expand_dims(station=[0])
            validate_model_data(expected, data, label)
            return data[list(expected.data_vars)].copy(deep=False)

        training = prepare(training, 'training_data')
        validation = prepare(validation, 'validation_data')
        self.training_data, self.validation_data = training, validation
        for key in ('training_selection', 'sources', 'station_ids', 'mean_bias', 'omb_bias_correction'):
            self.hyper_params.pop(key, None)
        self.hyper_params['training_dataset'] = None
        return self

    def _standard_recipe(self, optimization=False):
        """Read applicable recommendations from a bundled example recipe."""
        from openMWR.recipes import load_retrieval_template

        if isinstance(self.model, SpectralConsistencyModel):
            family = 'SPC'
        elif isinstance(self.model, Model):
            family = 'MLR' if self.model.is_MLR else 'NN_BLS' if self.model.is_bls_model else 'NN'
        else:
            raise ValueError(f'No bundled recommendations for {type(self.model).__name__}; supply parameters explicitly.')
        name = f'{family}_opt_example' if optimization else f'{family}_example'
        return load_retrieval_template(name)

    def _standard_parameters(self, optimization=False):
        from openMWR.recipes import noise_parameters

        recipe = self._standard_recipe(optimization)
        noise = noise_parameters(recipe['input_noise'])
        if 'TB' in noise:
            tb = noise.pop('TB')
            noise = {**{f'TB_{frequency}': tb for frequency in self.model.freqs}, **noise}
        training = {TRAINING_SETTINGS[key]: value for key, value in recipe['training'].items()}
        training_keys = training_parameters(self.model.is_MLR, training.get('loss_fn'))
        noise_keys = input_noise_parameters(self.model)
        return {**{key: value for key, value in training.items() if key in training_keys},
                **{f'input_noise_{key}': value for key, value in noise.items() if key in noise_keys}}

    def add_standard_training_hyper_params(self,
                        batch_size=None,
                        epochs=None,
                        lr_height=None,
                        lr_low=None,
                        epoch_max_lr=None,
                        loss_fn=None,
                        weight_decay=None,
                        seed=None,
                        bias_weight=None):
        """Use the matching bundled example's training settings, with explicit overrides.

        Omitted arguments use its [training] values. Only settings used by the
        model's fitting method are accepted; no project files or data are read.
        """
        parameters = {key: value for key, value in self._standard_parameters().items()
                      if key in TRAINING_PARAMETERS}
        overrides = {'batch_size': batch_size, 'epochs': epochs, 'lr_height': lr_height,
                     'lr_low': lr_low, 'epoch_max_lr': epoch_max_lr, 'loss_fn': loss_fn,
                     'weight_decay': weight_decay, 'seed': seed, 'bias_weight': bias_weight}
        parameters.update({key: value for key, value in overrides.items() if value is not None})
        validate_parameter_usage(self.model.is_MLR, parameters, complete=True)
        self.change_hyper_params(**{key: resolve_parameter(value, key, f'training.{key}', self)
                                   for key, value in parameters.items()})

    def change_hyper_params(self, **kwargs):
        """
        Update existing hyperparameters with new values.

        Parameters
        ----------
        **kwargs
            Arbitrary hyperparameters to modify or add.

        Notes
        -----
        Only updates provided keys; others remain unchanged.
        """
        self.hyper_params.update(kwargs)

    def get_hyper_params_from_existing_model(self, old_model_name, site_old_model=None):
        """
        Load hyperparameters from an already trained model and merge them
        into the current `hyper_params`.

        Parameters
        ----------
        old_model_name : str
            Name of the existing trained model.
        site_old_model : str, optional
            Site for the old model. Defaults to `self.site`.

        Notes
        -----
        Uses `Model.load_hyper_params`.
        """
        if site_old_model is None:
            site_old_model = self.site

        hyper_params_old = Model.load_hyper_params(site=site_old_model, name=old_model_name, project_dir=self.project_dir)

        self.hyper_params.update(hyper_params_old)

    def get_hyper_params_from_existing_study(self, study_site, old_model_name):
        """
        Import hyperparameters from an existing Optuna study.

        Parameters
        ----------
        study_site : str
            Site where the Optuna study is stored.
        old_model_name : str
            Retrieval directory whose study parameters should be loaded.

        Notes
        -----
        Loads best parameters from the Optuna study database.
        """
        study_file = optimization_file(self.project_dir, study_site, old_model_name)
        if not study_file.is_file():
            raise FileNotFoundError(f'Missing optimization database: {study_file}.')
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{study_file}")

        self.hyper_params.update(study.best_params)

    def default_search_spaces(self):
        """Return active searches from the matching bundled optimization example.

        Inputs absent from the example have no recommended range; their noise
        settings must be supplied explicitly before fitting or optimization.
        """
        return {key: resolve_parameter(value, key, key, self)
                for key, value in self._standard_parameters(optimization=True).items()
                if isinstance(value, dict)}

    def add_input_noise_params(self, optimization=False, **kwargs):
        """Set explicit noise values/ranges, filling omissions from examples.

        optimization=True selects the matching optimization example's ranges.
        TB applies to every frequency; per-frequency overrides take precedence.
        Custom models and inputs absent from an example require explicit
        values/ranges. Complete explicit settings do not read an example.
        """
        required = input_noise_parameters(self.model)
        check_keys(kwargs, required | {'TB'}, 'input_noise')
        parameters = {}
        if 'TB' in kwargs:
            tb = kwargs.pop('TB')
            parameters.update({f'TB_{frequency}': tb for frequency in self.model.freqs})
        parameters.update(kwargs)
        if required - parameters.keys() and isinstance(self.model, (Model, SpectralConsistencyModel)):
            defaults = {key.removeprefix('input_noise_'): value
                        for key, value in self._standard_parameters(optimization).items()
                        if key.startswith('input_noise_')}
            parameters = defaults | parameters
        require_keys(parameters, required, 'input_noise')
        resolved = {f'input_noise_{key}': resolve_parameter(value, f'input_noise_{key}', f'input_noise.{key}', self)
                    for key, value in parameters.items()}
        self.change_hyper_params(**resolved)

    def make_omb_analysis(self, station_ids_for_analysis, data_source=None, dataset=None, *,
                          start_date=None, end_date=None, days_of_month=None, max_lwp=None):
        """
        Create and process an OMB (observation-minus-background) analysis.

        Parameters
        ----------
        station_ids_for_analysis : list of str
            Radiosonde station IDs used for the OMB analysis.
        data_source : str, optional
            Background source. Inferred when the analysis has one reference.

        Notes
        -----
        - Intersects optional date bounds and an inclusive days_of_month range.
        - Computes mean and standard deviation of OMB differences.
        """
        from openMWR.omb import Omb_Analysis

        selection = {key: value for key, value in dict(start_date=start_date, end_date=end_date, days_of_month=days_of_month).items() if value is not None}
        check_time_selection(selection, 'OMB selection')
        self.site_config = load_site_config(self.site, self.project_dir, config=self.site_config)
        self.omb = Omb_Analysis(
            self.site,
            self.project_dir,
            data_source=data_source,
            station_ids_for_analysis=station_ids_for_analysis,
            load_existing_analysis_dataset=True,
            dataset=dataset, site_config=self.site_config,
            bls=self.model.is_bls_model, frequencies=list(self.model.freqs),
            angles=list(self.model.angles) if self.model.is_bls_model else None,
            max_lwp=max_lwp, **selection,
        )

    def make_bias_correction(self, specific_freqs=None, factor=None):
        """
        Apply bias correction to TB inputs based on OMB analysis.

        Parameters
        ----------
        specific_freqs : list of float, optional
            Frequencies for which bias correction should be applied.
            All others are set to zero.
        factor : float, optional
            Scale factor applied to the bias values.

        Raises
        ------
        ValueError
            If OMB analysis has not been created via `make_omb_analysis`.

        Notes
        -----
        Adds mean(observed − simulated) to synthetic training and validation
        TB. Prediction uses measured TB directly. The fitted shift and its
        provenance are stored in the checkpoint. Fit on separate dates from
        the final observation evaluation.
        """
        if not hasattr(self, 'omb'):
            raise ValueError("Omb analysis not made yet. Call make_omb_analysis() first.")
        
        from openMWR.omb import omb_statistics
        if 'mean_bias' in self.hyper_params:
            raise ValueError('Bias correction is already applied. Start a new workflow to refit it.')
        factor = 1.0 if factor is None else factor
        from openMWR.config import check_number, check_coordinates
        check_number(factor, 'bias correction factor', 0)
        dimensions = tuple(dim for dim in ('time', 'station') if dim in self.omb.diff.dims)
        stats = omb_statistics(self.omb.diff, sample_dims=dimensions)
        mean_bias = stats['mean'].copy(deep=True)
        if specific_freqs is not None:
            check_coordinates(specific_freqs, 'bias correction frequencies')
            if not set(specific_freqs) <= set(mean_bias.frq.values):
                raise ValueError('Bias correction frequencies must be selected OMB channels.')
            mean_bias = mean_bias.where(mean_bias.frq.isin(specific_freqs), 0.)
            counts = stats['count'].where(stats.frq.isin(specific_freqs), 2)
        else:
            counts = stats['count']
        if not bool(np.isfinite(mean_bias).all()) or not bool((counts >= 2).all()):
            raise ValueError('Bias correction requires at least two finite OMB pairs per selected channel and angle.')
        mean_bias = mean_bias * factor
        for data in (self.training_data, self.validation_data):
            # Simulation files can contain channels/angles unused by this model.
            shift = mean_bias.reindex({dim: data[dim] for dim in mean_bias.dims}, fill_value=0.)
            corrected = data['TB'] + shift
            corrected.attrs = data['TB'].attrs.copy()
            data['TB'] = corrected
        self.hyper_params['mean_bias'] = mean_bias.values.tolist()
        self.hyper_params['omb_bias_correction'] = plain_value({**self.omb.metadata(),
            'application': 'add_to_simulation', 'factor': factor, 'specific_freqs': specific_freqs,
            'dimensions': list(mean_bias.dims), 'count': stats['count'].values})

    def select_omb_input_noise_hyper_params(self, *, statistic, factor, frequency_bands=None):
        """
        Use explicit total or experimental unexplained OMB scatter for input noise.

        Parameters
        ----------
        statistic : {'std', 'residual_std'}
            Total OMB scatter or experimental unexplained regression scatter.
        factor : float
            Scaling factor applied to derived noise values.
        frequency_bands : list of [float, float], optional
            Explicit bounds in GHz, required for residual_std.

        Raises
        ------
        ValueError
            If OMB analysis has not been computed.

        Notes
        -----
        Updates `hyper_params` with OMB-based noise estimates.
        """
        if not hasattr(self, 'omb'):
            raise ValueError("Omb analysis not made yet. Call make_omb_analysis() first.")

        self.hyper_params.update(self.omb.create_input_noise_from_diff(self.model,
            statistic=statistic, factor=factor, frequency_bands=frequency_bands))
        self.hyper_params['omb_input_noise'] = {**self.omb.metadata(), 'statistic': statistic,
            'factor': factor, 'angle_application': 'zenith estimate with configured angle factors'}

    def add_factor_to_input_noise_TB(self, factor):
        """
        Multiply all TB-related input noise parameters by a scaling factor.

        Parameters
        ----------
        factor : float
            Multiplicative factor applied to all `input_noise_TB_*` parameters.
        """
        for f in self.model.freqs:
            self.hyper_params[f"input_noise_TB_{f}"] *= factor

    def _create_training_tensors(self):
        """
        Prepare normalized and stacked training/validation tensors.

        Calculates statistics like mean and std from training data,
        normalizes both training and validation datasets, stacks them
        over station and time dimensions, and converts to PyTorch tensors.

        Returns
        -------
        x_training : torch.Tensor
        y_training : torch.Tensor
        x_val : torch.Tensor
        y_val : torch.Tensor
        """
        # Normalize the data
        self.model.nor = create_stats(self.model, self.training_data)
        self.logger.info('Created normalization dataset')

        variables = list(dict.fromkeys(self.model.input_vars + self.model.output_vars))
        training_data = normalize(self.training_data[variables], self.model)
        validation_data = normalize(self.validation_data[variables], self.model)
        
        # Stack the data to create a single dimension for station and time
        # This is necessary to create a 2D tensor for training
        training_data = training_data.stack(station_time=("station", "time")).dropna(dim='station_time')
        validation_data = validation_data.stack(station_time=("station", "time")).dropna(dim='station_time')

        # Create tensors for training and validation
        x_training = create_tensor_from_ds(self.model.input, training_data, self.device)
        y_training = create_tensor_from_ds(self.model.output, training_data, self.device)
        x_val = create_tensor_from_ds(self.model.input, validation_data, self.device)
        y_val = create_tensor_from_ds(self.model.output, validation_data, self.device)

        return x_training, y_training, x_val, y_val
    
    def _adjust_lr(self, optimizer, epoch, epochs, lr_height, lr_low, epoch_max_lr):
        """
        Adjusts the learning rate of the optimizer according to a two-phase linear schedule.
        In the first phase (before `epoch_max_lr`), the learning rate increases linearly from `lr_low` to `lr_height`.
        In the second phase (after `epoch_max_lr`), the learning rate decreases linearly from `lr_height` back to `lr_low`.
        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
            epoch (int): The current epoch number.
            epochs (int): The total number of training epochs.
            lr_height (float): The maximum learning rate to reach at `epoch_max_lr`.
            lr_low (float): The minimum learning rate (starting and ending value).
            epoch_max_lr (int): The epoch at which the learning rate peaks at `lr_height`.
        Returns:
            None
        """
        
        if epoch < epoch_max_lr:
            lr = lr_low - (lr_low - lr_height) * (epoch / epoch_max_lr)
        else:
            lr = lr_height - (lr_height - lr_low) * ((epoch - epoch_max_lr) / (epochs - epoch_max_lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def _plot_training_progress(self, epochs, eval_losses, train_losses=None):
        """
        Plot training and validation loss curves.

        Parameters
        ----------
        epochs : int
            Total number of epochs.
        eval_losses : list of float
            Validation loss per epoch.
        train_losses : list of float, optional
            Training loss per epoch.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Plot for the first subplot
        if train_losses is not None:
            ax[0].plot(train_losses, label='training_data')
        ax[0].plot(eval_losses, label='validation_data')
        ax[0].legend()
        ax[0].grid()

        # Plot for the second subplot
        start = int(epochs / 3)
        if train_losses is not None:
            ax[1].plot(range(start, len(train_losses)), train_losses[start:], label='training_data')
        ax[1].plot(range(start, len(eval_losses)), eval_losses[start:], label='validation_data')
        ax[1].legend()
        ax[1].grid()

        if is_called_from_notebook():
            plt.show()

        else:
            plot_dir = retrieval_dir(self.project_dir, self.site, self.model.name)
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            fig.savefig(plot_dir / 'training_progress.pdf')
            plt.close(fig)
    
    def _training_loop(self, model: BaseModel, x_training: torch.Tensor, y_training: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, hyper_params: dict, device: str, quiet: bool = False):
        """
        Internal training loop for neural network models.

        Parameters
        ----------
        model : BaseModel
            Model instance to train.
        x_training, y_training : torch.Tensor
            Training inputs and targets.
        x_val, y_val : torch.Tensor
            Validation inputs and targets.
        hyper_params : dict
            Current hyperparameter dictionary.
        device : str
            Target device for computation.
        quiet : bool, default=False
            If True, suppress logging and progress output.

        Returns
        -------
        BaseModel
            Best-performing model based on validation loss.

        Notes
        -----
        - Input noise is injected during training.
        - Learning rate follows a triangular schedule (see `_adjust_lr` method).
        - Evaluation is performed on a validation set each epoch.
        """
        model = model.to(device)

        logger = logging.getLogger(__name__)

        epochs = hyper_params['epochs']
        batch_size = hyper_params['batch_size']
        weight_decay = hyper_params['weight_decay']

        # Learning rate
        lr_height = hyper_params['lr_height']
        lr_low = hyper_params['lr_low']
        epoch_max_lr = hyper_params['epoch_max_lr']

        if hyper_params['loss_fn'] == 'MSE':
            loss_fn = nn.MSELoss(reduction='none')
        elif hyper_params['loss_fn'] == 'MAE':
            loss_fn = nn.L1Loss(reduction='none')
        elif hyper_params['loss_fn'] == 'MAEPlusBiasLoss':
            loss_fn = MAEPlusBiasLoss(bias_weight=hyper_params['bias_weight'])
        elif hyper_params['loss_fn'] == 'MSEPlusBiasLoss':
            loss_fn = MSEPlusBiasLoss(bias_weight=hyper_params['bias_weight'])
        else:
            raise ValueError(f"Unknown loss function: {hyper_params['loss_fn']}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_low, weight_decay=weight_decay)

        if hasattr(model, 'loss_weights'):
            loss_weight = to_1d_tensor(model.loss_weights, device)
        else:
            loss_weight = None

        training_size = x_training.shape[0]

        # Errors
        std_dev_np = normalized_input_noise(model, hyper_params)
        std_dev = torch.tensor(std_dev_np, dtype=torch.float32).to(device)

        train_losses = []
        eval_losses = []


        idxs = np.arange(0, training_size)

        start_time = time.time()

        for epoch in range(epochs):

            # Compute linear learning rate
            self._adjust_lr(optimizer, epoch, epochs, lr_height, lr_low, epoch_max_lr)

            # Shuffle
            np.random.shuffle(idxs)

            # Training
            model.train()

            for i in range(0, training_size, batch_size):
                idx = idxs[i: i+batch_size]
                x = x_training[idx,:]
                y = y_training[idx,:]

                ### Add random noise
                x_noisy = x + torch.randn(x.size(), device=device) * std_dev

                # Compute prediction and loss
                pred = model(x_noisy)
                loss_per_output = loss_fn(pred, y)

                if loss_weight is None:
                    loss = loss_per_output.mean()
                else:
                    loss = (loss_per_output * loss_weight).mean()

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            #Testing
            model.eval()
            # pred = model(x_training)
            # loss_per_output = loss_fn(pred, y_training)
            # if loss_weight is None:
            #     loss = loss_per_output.mean()
            # else:
            #     loss = (loss_per_output * loss_weight).mean()
            # train_losses.append(loss.item())
            train_losses=None

            pred = model(x_val)
            loss_per_output = loss_fn(pred, y_val)
            if loss_weight is None:
                loss = loss_per_output.mean()
            else:
                loss = (loss_per_output * loss_weight).mean()
            eval_losses.append(loss.item())

            if epoch == 0 or loss.item() < best_loss:
                best_model = copy.deepcopy(model)
                best_loss = loss.item()
            
            if not quiet:
                if epoch % max(1, int(epochs / 100)) == 0:
                    if is_called_from_notebook() and epochs > 1:
                        progress_bar(epoch, epochs, start_time)
                    else:
                        logger.info(progress_log(epoch, epochs, start_time))
        
        #if not quiet:
        self._plot_training_progress(epochs, eval_losses, train_losses)

        logger.info(f'Finished training loop with best loss on validation dataset: {best_loss}')

        return best_model
    
    def _calc_linear_regression(self, model: BaseModel, x_training, y_training, hyper_params):
        """
        Fit MLR coefficients using the exact expected input-noise loss.

        Parameters
        ----------
        model : BaseModel
            Model with a linear (stack) structure.
        x_training : torch.Tensor
            Training inputs.
        y_training : torch.Tensor
            Training targets.
        hyper_params : dict
            Input-noise hyperparameters in the original input units.

        Returns
        -------
        BaseModel
            Model with fitted linear coefficients.

        Notes
        -----
        Independent zero-mean input noise with normalized standard deviations s
        adds n * diag(s**2) to the centered least-squares normal equations.
        An augmented least-squares solve avoids forming those normal equations
        and supports zero noise and rank-deficient inputs. The intercept is
        unpenalized. This minimizes the expected loss, replacing random repeats.
        """
        x = x_training.cpu().numpy().astype(np.float64)
        y = y_training.cpu().numpy().astype(np.float64)
        x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
        std_dev = normalized_input_noise(model, hyper_params)
        penalty = np.diag(np.sqrt(len(x)) * std_dev)
        coefficients, *_ = np.linalg.lstsq(
            np.vstack((x - x_mean, penalty)),
            np.vstack((y - y_mean, np.zeros((x.shape[1], y.shape[1])))), rcond=None)
        intercept = y_mean - x_mean @ coefficients
        with torch.no_grad():
            model.stack.weight.copy_(torch.as_tensor(coefficients.T))
            model.stack.bias.copy_(torch.as_tensor(intercept))

        return model

    @_logged_run('train')
    def train(self, run_mode=None):
        """
        Run training using current hyperparameters and data.

        For workflows with existing results, run_mode="fresh" explicitly
        replaces only this retrieval's checkpoint and study after validation.

        Notes
        -----
        - Prepares data tensors.
        - Runs either linear regression (MLR) or neural network training.
        - Saves the trained model to disk.
        """
        if self.requires_optimization:
            raise ValueError('Training one model needs fixed parameter values. Call workflow.optimize(), use the train command for automatic selection, or replace the search definitions with fixed values.')
        self.logger.info(f'Using hyper parameters: {self.hyper_params}')

        self.validate()
        self._prepare_run(run_mode)
        apply_seed(self.hyper_params['seed'])
        self._log_training_setup()

        x_training, y_training, x_val, y_val = self._create_training_tensors()

        if self.model.is_MLR:
            self.model = self._calc_linear_regression(self.model, x_training, y_training, self.hyper_params)
        else:
            self.model = self._training_loop(self.model, x_training, y_training, x_val, y_val, self.hyper_params, device=self.device, quiet=False)

        # Save model
        self._save_model(self.model, self.hyper_params)
        return self.model

    def _external_stop_callback(self, study, trial):
        """
        Optuna callback to externally stop optimization if 'stop.txt' exists.

        Used to gracefully terminate long-running optimization jobs. Optuna will
        check for the presence of 'stop.txt' in the current directory at the end
        of each trial and stop the study if found.

        """
        if os.path.exists("stop.txt"):
            self.logger.error("Stop-Datei gefunden. Optuna-Optimierung wird abgebrochen.")
            study.stop()

    def _suggest_values(self, trial: optuna.Trial, value: any, name: str = None):
        """
        Recursively convert dictionaries/lists/namedtuples of suggestion
        definitions into Optuna parameter suggestions.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object used to sample values.
        value : any
            Nested structure of suggest_* objects, lists, dicts, or fixed values.
        name : str, optional
            Parameter name (used for nested structures).

        Returns
        -------
        any
            Sampled value or structure of sampled values.

        Notes
        -----
        Supports:
        - suggest_int(low, high)
        - suggest_float(low, high)
        - suggest_categorical(choices)
        - nested lists and dictionaries.
        """
        if isinstance(value, dict):
            return {
                k: self._suggest_values(trial, v, name=k)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [
                self._suggest_values(trial, v, name=f'{name}_{i}')
                for i, v in enumerate(value)
            ]
        elif isinstance(value, self.suggest_int):
            return trial.suggest_int(name, value.low, value.high, step=value.step, log=value.log)
        elif isinstance(value, self.suggest_float):
            return trial.suggest_float(name, value.low, value.high, step=value.step, log=value.log)
        elif isinstance(value, self.suggest_categorical):
            return trial.suggest_categorical(name, value.choices)
        else:
            return value

    @_logged_run('optimize')
    def optimize(self, n_trials=None, data_source=None, run_mode=None, dataset=None, *, seed=None,
                 start_date=None, end_date=None, days_of_month=None, score_weights=None, sources=None):
        """
        Perform hyperparameter optimization using Optuna.

        Parameters
        ----------
        n_trials : int, optional
            Number of new optimization trials. Uses the recipe or bundled
            recommendation when omitted.
        data_source : str, optional
            Reference source used to score trials. Inferred when the analysis
            has one reference, otherwise select 'radiosonde' or 'era5'.
        sources : list of str, optional
            Reference source IDs used for scoring. Omitted sources select all
            members of the named analysis dataset.
        run_mode : {'fresh', 'continue'}, optional
            All workflows require a choice when results already exist.
            Fresh clears this retrieval's checkpoint/study after validation;
            continue adds trials to a compatible study.

        Notes
        -----
        - Models without a recipe or bundled recommendations require explicit
          n_trials, dataset, seed and score_weights. Time and source filters
          are optional.
        - Each trial trains a model using suggested parameters.
        - Evaluation is performed on real atmospheric profiles.
        - Best models are saved automatically.
        - Study is stored beside the retrieval in `optimization.db`.
        """
        settings, an_data = self._optimization_inputs(n_trials=n_trials, data_source=data_source, dataset=dataset,
                                                     seed=seed, start_date=start_date, end_date=end_date,
                                                     days_of_month=days_of_month, score_weights=score_weights, sources=sources)
        n_trials, data_source, analysis_name = settings['n_trials'], settings['data_source'], settings['dataset']
        hyper_params = dict(self.hyper_params)
        self.logger.info('Optimizing %s (trial budget: %d, reference: %s).', self.model.name, n_trials, data_source)
        for key, value in hyper_params.items():
            if isinstance(value, (self.suggest_int, self.suggest_float, self.suggest_categorical)):
                label = f'input_noise.{key.removeprefix("input_noise_")}' if key.startswith('input_noise_') else f'training.{TRAINING_NAMES.get(key, key)}'
                self.logger.info('Search %s: %s', label, value)

        variables = list(set(self.model.input_vars + self.model.output_vars))
        scored = [key for key, weight in settings['score_weights'].items() if weight > 0]
        data_identity = {
            'training': dataset_id(self.training_data[variables]),
            'validation': dataset_id(self.validation_data[variables]),
            'observations': dataset_id(an_data.sel(data_source='hatpro', drop=True)[self.model.input_vars]),
            'reference': dataset_id(an_data.sel(data_source=data_source, drop=True)[scored]),
        }
        self._run_record['data'] = data_identity
        storage = self._prepare_run(run_mode, optimization=True)
        retrieval_dir(self.project_dir, self.site, self.model.name).mkdir(parents=True, exist_ok=True)
        sampler = optuna.samplers.TPESampler(seed=None if settings['seed'] == 'random' else settings['seed'])
        if run_mode == 'continue':
            study = optuna.load_study(study_name=None, storage=storage, sampler=sampler)
        else:
            study = optuna.create_study(study_name=self.model.name, storage=storage,
                                       direction='minimize', sampler=sampler)
        unused = TRAINING_PARAMETERS - training_parameters(self.model.is_MLR, hyper_params.get('loss_fn'))
        # Applied bias corrections are already recorded as hyper_params['mean_bias'].
        experiment = plain_value({'model': self.model.checkpoint_structure(), 'hyper_params': hyper_params,
                                  'data_source': data_source, 'analysis_dataset': analysis_name,
                                  'optimization': {key: settings[key] for key in ('seed', 'score_weights', 'sources', *TIME_SELECTION_KEYS) if key in settings}})
        experiment['hyper_params'] = {key: value for key, value in experiment['hyper_params'].items() if key not in unused}
        if self.model.is_MLR:
            experiment['fitting'] = _MLR_FITTING
        # Historical trials always divided noise by std. Only noisy max_norm
        # inputs change meaning, so unaffected histories need no new marker.
        maximum_noise = get_input_noise(self.model, self._input_noise_limits())
        scaling = {var: 'max' for var in self.model.input_vars
                   if self.model.norm_functions[var] == 'max_norm'
                   and bool((maximum_noise[var] != 0).any())}
        if scaling:
            experiment['input_noise_scaling'] = scaling
        experiment['data'] = data_identity
        previous = study.user_attrs.get('openMWR_experiment')
        definition = _experiment_definition(experiment, variables)
        previous_definition = _experiment_definition(previous, variables)
        if (previous_definition is not None
                and previous_definition.get('input_noise_scaling', {}) != definition.get('input_noise_scaling', {})):
            raise ValueError('This study uses different input-noise normalization. '
                             'Use --fresh (run_mode="fresh" in Python) or a new retrieval name to restart.')
        if ((previous is None and study.trials) or
                (previous is not None and ('data' not in previous or previous_definition != definition))):
            raise ValueError('This study has a different or unrecorded experiment definition. Use --fresh (run_mode="fresh" in Python) or a new retrieval name to restart.')
        if previous is not None and previous['data'] != data_identity:
            changed = ', '.join(key for key, value in data_identity.items() if previous['data'].get(key) != value)
            raise ValueError(f'Optimization data changed ({changed}). Use --fresh (run_mode="fresh" in Python) or a new retrieval name to restart.')
        try:
            best_trial = study.best_trial
        except ValueError:
            best_trial = None
        if best_trial is not None:
            path = checkpoint_file(self.project_dir, self.site, self.model.name)
            metadata = type(self.model).load(self.model.name, self.site, self.project_dir).training_metadata if path.is_file() else None
            expected = {'study_name': study.study_name, 'trial_number': best_trial.number}
            if (metadata or {}).get('optimization') != expected:
                raise ValueError('The best study checkpoint is missing or has been replaced. Use --fresh (run_mode="fresh" in Python) or a new retrieval name to restart.')
        study.set_user_attr('openMWR_experiment', experiment)

        self._log_training_setup(settings)
        self.logger.info(f'Using hyper parameters: {self.hyper_params}')

        x_training, y_training, x_val, y_val = self._create_training_tensors()

        ds_hatpro = an_data.sel(data_source='hatpro', drop=True)
        truth = an_data.sel(data_source=data_source, drop=True)

        ds_hatpro = ds_hatpro[self.model.input_vars]

        ds_nor = normalize(ds_hatpro, self.model)

        use_station_time = True if 'station' in ds_nor.dims else False
        dim_0='station_time' if use_station_time else 'time'

        if use_station_time:
            ds_nor = ds_nor.stack(station_time=("station", "time")).dropna(dim='station_time')

        x_real = create_tensor_from_ds(self.model.input, ds_nor, self.device, dim_0=dim_0)

        def objective(trial):

            if settings['seed'] != 'random':
                # Trial numbers survive restarts, so continuation follows the same sequence.
                trial.study.sampler = optuna.samplers.TPESampler(seed=(settings['seed'] + trial.number) % 2**32)
            training_seed = hyper_params['seed']
            if training_seed != 'random':
                training_seed = (training_seed + trial.number) % 2**32
            apply_seed(training_seed)
            model_trial = type(self.model)(**self.model.structure)

            model_trial.nor = self.model.nor.copy(deep=True)

            hyper_params_trial = self._suggest_values(trial, hyper_params)

            if model_trial.is_MLR:
                model_trial = self._calc_linear_regression(model_trial, x_training, y_training, hyper_params_trial)
                model_trial = model_trial.to(self.device)
            else:
                model_trial = self._training_loop(model_trial, x_training, y_training, x_val, y_val, hyper_params_trial, device=self.device, quiet=True)

            # Run model
            model_trial.eval()
            with torch.no_grad():
                y_real = model_trial(x_real).detach().cpu().numpy()

            pred_nor = create_ds_from_y(y_real, model_trial, ds_nor, dim_0=dim_0)

            if use_station_time:
                pred_nor = pred_nor.unstack("station_time")

            pred = denormalize(pred_nor, model_trial)

            # Calc real loss
            total_raso_loss = 0
            for var, score_weight in settings['score_weights'].items():
                if score_weight > 0:
                    raso_loss_height = cal_RMSE(pred[var], truth[var], dims_to_mean=['time', 'station'] if use_station_time else ['time'])

                    weights = model_trial.loss_weights[var] if hasattr(model_trial, 'loss_weights') and var in model_trial.loss_weights else 1
                    raso_loss = (raso_loss_height * weights).mean().item()

                    self.logger.info(f'{var}: {raso_loss}')

                    total_raso_loss += score_weight * raso_loss

            if not np.isfinite(total_raso_loss):
                raise ValueError('Optimization produced a non-finite score; check analysis data and model outputs.')

            # save the model if it is better than the best one so far
            try:
                best_value = trial.study.best_value
            except ValueError:
                best_value = float('inf')   

            if total_raso_loss < best_value:
                self._save_model(model_trial, hyper_params_trial,
                                 training_seed=training_seed,
                                 optimization={'study_name': study.study_name, 'trial_number': trial.number})
                
            return total_raso_loss

        self.logger.info('Starting hyperparameter optimization with Optuna...')

        study.optimize(objective, n_trials=n_trials, callbacks=[self._external_stop_callback])

        self.logger.info('Optimization completed.')
        self.logger.info(f'Best study params: {study.best_params}')
        return study


def separate_training_data(site, source, project_dir, simulation_dataset, dataset='training',
                           *, test_days_of_month, validation_days_of_month,
                           start_date=None, end_date=None, days_of_month=None):
    """Split selected simulations by inclusive day-of-month ranges.

    simulation_dataset explicitly names the input simulations. All timestamps
    on a calendar day have the same assignment across sources.
    Training receives days outside the test and validation ranges. Changed
    inputs regenerate the three files; no random assignments or old memberships
    are needed. The dataset workflow skips unchanged inputs.
    """
    selection = dict(start_date=start_date, end_date=end_date, days_of_month=days_of_month)
    check_time_selection(selection, 'split')
    check_split_days(test_days_of_month, validation_days_of_month)
    sources = [source] if isinstance(source, str) else list(source)
    for source_id in sources:
        input_file = dataset_file(project_dir, site, simulation_dataset, source_id)
        with xr.open_dataset(input_file) as original:
            if not original.indexes['time'].is_unique:
                raise ValueError(f'Duplicate timestamps in simulation {input_file}.')
            ds = select_time(original, **selection).load()
        if not ds.time.size:
            raise ValueError(f'No profiles remain after time selection for {input_file}.')
        paths = {split: dataset_file(project_dir, site, dataset, source_id, split=split)
                 for split in ('training', 'validation', 'test')}
        day = ds.time.dt.day
        test = (day >= test_days_of_month['first']) & (day <= test_days_of_month['last'])
        validation = (day >= validation_days_of_month['first']) & (day <= validation_days_of_month['last'])
        assignments = {'training': ~(test | validation), 'validation': validation, 'test': test}
        for split, mask in assignments.items():
            if not mask.any():
                raise ValueError(f'No {split} profiles in the selected period of {input_file}; adjust the period or split day ranges.')
        for split, mask in assignments.items():
            part = ds.isel(time=mask).sortby('time')
            paths[split].parent.mkdir(parents=True, exist_ok=True)
            part.to_netcdf(paths[split])
            logger.info('Saved %s: %s samples.', paths[split], part.time.size)

class MAEPlusBiasLoss(nn.Module):
    """
    Mean absolute error loss with an added bias penalty.

    Parameters
    ----------
    bias_weight : float
        Scaling factor for the mean absolute bias term across the batch.
    """
    def __init__(self, bias_weight):
        super().__init__()
        self.bias_weight = bias_weight
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, pred, target):
        """
        Compute MAE plus a bias term averaged over the batch.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground-truth targets.

        Returns
        -------
        torch.Tensor
            Scalar loss combining MAE and weighted bias.
        """
        # 1) Compute MAE (element-wise because reduction='none')
        mae_loss = self.mae(pred, target)  # Shape: [B, V] or multi-dimensional
        
        # 2) Average MAE (across all dimensions except batch)
        mae_loss_mean = mae_loss.mean()

        # 3) Compute per-variable bias across batch
        bias_per_var = torch.mean(pred - target, dim=0)  # averaged over batch only
        bias_loss = torch.mean(torch.abs(bias_per_var))  # average over variables

        # 4) Combine
        total_loss = mae_loss_mean + self.bias_weight * bias_loss
        return total_loss
    
class MSEPlusBiasLoss(nn.Module):
    """
    Mean squared error loss with an added bias penalty.

    Parameters
    ----------
    bias_weight : float
        Scaling factor for the mean absolute bias term across the batch.
    """
    def __init__(self, bias_weight):
        super().__init__()
        self.bias_weight = bias_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        """
        Compute MSE plus a bias term averaged over the batch.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground-truth targets.

        Returns
        -------
        torch.Tensor
            Scalar loss combining MSE and weighted bias.
        """
        # 1) Compute MSE (element-wise because reduction='none')
        mse_loss = self.mse(pred, target)  # Shape: [B, V] or multi-dimensional

        # 2) Average MSE (across all dimensions except batch)
        mse_loss_mean = mse_loss.mean()

        # 3) Compute per-variable bias across batch
        bias_per_var = torch.mean(pred - target, dim=0)  # averaged over batch only
        bias_loss = torch.mean(torch.abs(bias_per_var))  # average over variables

        # 4) Combine
        total_loss = mse_loss_mean + self.bias_weight * bias_loss
        return total_loss
