"""Validated, editable recipes for named retrievals in a working project."""

from dataclasses import dataclass
from pathlib import Path
from importlib.resources import files
import tomllib

from openMWR.config import (TRAINING_SETTINGS, training_parameters, check_keys,
                            check_number, check_name, check_strings,
                            validate_parameter_usage, resolve_parameter, input_noise_parameters,
                            require_keys, load_grid, apply_seed, check_seed, check_time_selection, TIME_SELECTION_KEYS,
                            check_score_weights, check_output_vars, check_learning_rates,
                            resolve_coordinate_table)
from openMWR.paths import project_root, retrieval_dir, site_config_file
from openMWR.site import SiteConfig, load_site_config

_NOISE_CONTROLS = {'method', 'angle_dependence', 'angle_factors', 'channel_noise'}
_REFERENCE_KEYS = {'data_source', 'dataset', 'sources', 'factor', 'statistic', 'frequency_bands', 'max_lwp', *TIME_SELECTION_KEYS}


def load_retrieval_template(name):
    """Read bundled recipe settings without a working project or prepared data."""
    check_name(name, 'template')
    path = files('openMWR').joinpath('templates', 'site', 'retrieval', name, 'retrieval.toml')
    return tomllib.loads(path.read_text(encoding='utf-8'))


def _check_reference(settings, section):
    """Validate an explicitly supplied analysis source and dataset name."""
    if 'data_source' in settings and settings['data_source'] not in ('radiosonde', 'era5'):
        raise ValueError(f"{section}.data_source must be 'radiosonde' or 'era5'.")
    if 'dataset' in settings:
        check_name(settings['dataset'], f'{section}.dataset')
    if 'sources' in settings:
        check_strings(settings['sources'], f'{section}.sources')
        if not settings['sources']:
            raise ValueError(f'{section}.sources must not be empty.')
        for source in settings['sources']:
            check_name(source, f'{section}.sources')


def noise_parameters(settings):
    """Lower recipe noise tables to distinct checkpoint/study parameter names."""
    metadata = _NOISE_CONTROLS | (_REFERENCE_KEYS if settings.get('method') == 'omb' else set())
    values = {key: value for key, value in settings.items() if key not in metadata}
    for key in values:
        if key.startswith('TB_angle_'):
            raise ValueError(f'Unknown input_noise setting {key!r}. Use angle_factor or the angle_factors table.')
        if key.startswith('TB_'):
            try:
                float(key.removeprefix('TB_'))
            except ValueError:
                continue
            raise ValueError(f'Unknown input_noise setting {key!r}. Put frequency-specific values in input_noise.channel_noise.')
    if 'angle_factor' in values:
        values['TB_angle_factor'] = values.pop('angle_factor')
    for field, prefix in (('channel_noise', 'TB_'), ('angle_factors', 'TB_angle_')):
        if field not in settings:
            continue
        label = f'input_noise.{field}'
        for coordinate, value in resolve_coordinate_table(settings[field], label).items():
            if field == 'angle_factors' and not 0 < coordinate <= 90:
                raise ValueError(f'{label} angles must be above 0 and at most 90 degrees.')
            key = f'{prefix}{coordinate}'
            values[key] = value
    return values


@dataclass(frozen=True)
class RetrievalRecipe:
    """One TOML recipe with explicit scientific settings and optional features."""

    project_dir: Path
    site: str
    name: str
    values: dict
    site_config: SiteConfig | None = None

    @classmethod
    def load(cls, project_dir, site, retrieval):
        check_name(site, "site")
        check_name(retrieval, "retrieval")
        root = project_root(project_dir)
        config = site_config_file(root, site)
        if not config.is_file():
            raise FileNotFoundError(f"Missing site configuration: {config}")
        site_config = load_site_config(site, root)
        path = retrieval_dir(root, site, retrieval) / "retrieval.toml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing retrieval recipe: {path}")
        with path.open("rb") as stream:
            recipe = cls(root, site, retrieval, tomllib.load(stream), site_config=site_config)
        recipe._validate()
        return recipe

    def section(self, name):
        return self.values.get(name, {})

    @property
    def requires_optimization(self):
        """Whether the recipe defines searches in its training or input noise."""
        return any(isinstance(value, dict) for value in self.section('training').values()) or any(
            isinstance(value, dict) for value in noise_parameters(self.section('input_noise')).values())

    def _validate(self):
        check_keys(self.values, ("schema_version", "model", "data", "training", "input_noise", "bias_correction", "optimization"), "recipe")
        if type(self.values.get("schema_version")) is not int or self.values["schema_version"] != 1:
            raise ValueError("Recipe schema_version must be 1.")
        for section in ("model", "data", "bias_correction", "optimization"):
            table = self.section(section)
            if isinstance(table, dict):
                for key, value in table.items():
                    if isinstance(value, dict) and key not in ('days_of_month', 'score_weights') and (section, key) not in (("model", "normalization"), ("model", "loss_weights")):
                        raise ValueError(f"Search definitions are supported only in training and input_noise; {section}.{key} does not support a search.")
        model = self.section("model")
        check_keys(model, ("kind", "mode", "frequencies", "heights", "angles", "hidden_layers", "hidden_layer_size", "dropout_rate", "additional_input_vars", "output_vars", "normalization", "loss_weights"), "model")
        require_keys(model, ('kind', 'mode', 'frequencies', 'normalization'), 'model')
        if model["kind"] not in ("nn", "mlr", "spectral"):
            raise ValueError("model.kind must be 'nn', 'mlr', or 'spectral'.")
        if model['mode'] not in ("zenith", "bls"):
            raise ValueError("model.mode must be 'zenith' or 'bls'.")
        settings = dict(model)
        for key in ("frequencies", "heights", "angles"):
            if key in model and model[key] != 'dataset':
                settings[key] = load_grid(self.project_dir, key, model[key], label=f'model.{key}')
        if "hidden_layers" in model:
            if not isinstance(model["hidden_layers"], list) or not model["hidden_layers"]:
                raise ValueError("model.hidden_layers must be a nonempty list.")
            for width in model["hidden_layers"]:
                check_number(width, "model.hidden_layers", 1, integer=True)
        if "hidden_layer_size" in model:
            check_number(model["hidden_layer_size"], "model.hidden_layer_size", 1, integer=True)
        if "dropout_rate" in model:
            check_number(model["dropout_rate"], "model.dropout_rate", 0)
            if model["dropout_rate"] >= 1:
                raise ValueError("model.dropout_rate must be below 1.")
        if model['mode'] != 'bls' and 'angles' in model:
            raise ValueError("Angles require model.mode = 'bls'.")
        if model["kind"] == "spectral":
            check_keys(model, ("kind", "mode", "frequencies", "hidden_layer_size", "normalization"), "spectral model")
            if model['mode'] != "zenith":
                raise ValueError("Spectral models use zenith observations.")
            require_keys(model, ('hidden_layer_size',), 'model')
            if model['frequencies'] != 'dataset' and len(settings['frequencies']) < 2:
                raise ValueError("Spectral models require at least two frequencies.")
        elif "hidden_layer_size" in model:
            raise ValueError("model.hidden_layer_size is only used by spectral models.")
        if model['kind'] in ('nn', 'mlr'):
            require_keys(model, ('output_vars', 'additional_input_vars'), 'model')
            check_output_vars(model['output_vars'])
            if set(model['output_vars']) & {'T', 'rh', 'lwc'}:
                require_keys(model, ('heights',), 'model')
            elif 'heights' in model:
                raise ValueError('model.heights applies only to profile outputs.')
        if model['kind'] == 'nn':
            require_keys(model, ('hidden_layers', 'loss_weights'), 'model')
        if model['mode'] == 'bls':
            require_keys(model, ('angles',), 'model')
        if model["kind"] == "mlr":
            for key in ("hidden_layers", "dropout_rate", "loss_weights"):
                if key in model:
                    raise ValueError(f"model.{key} is not used by mlr models.")
        if "additional_input_vars" in model:
            check_strings(model["additional_input_vars"], "model.additional_input_vars")
        if not isinstance(model["normalization"], dict) or any(v not in ("z_norm", "std_norm", "log_norm", "max_norm") for v in model["normalization"].values()):
            raise ValueError("model.normalization must map variables to supported normalization functions.")
        variables = {'TB', *model.get('additional_input_vars', []), *model.get('output_vars', [])}
        check_keys(model['normalization'], variables, 'model.normalization')
        require_keys(model['normalization'], variables, 'model.normalization')

        data = self.section("data")
        check_keys(data, ('dataset', 'sources'), 'data')
        check_name(data.get('dataset'), 'data.dataset')
        if 'sources' in data:
            check_strings(data['sources'], 'data.sources')
            if not data['sources']:
                raise ValueError('data.sources must not be empty.')
            for source in data['sources']:
                check_name(source, 'data.sources')
        training = self.section('training')
        check_keys(training, TRAINING_SETTINGS, 'training')
        parameters = {TRAINING_SETTINGS[key]: value for key, value in training.items()}
        validate_parameter_usage(model['kind'] == 'mlr', parameters)
        required = training_parameters(model['kind'] == 'mlr', parameters.get('loss_fn'))
        require_keys(training, {key for key, parameter in TRAINING_SETTINGS.items() if parameter in required}, 'training')
        for key, value in training.items():
            resolve_parameter(value, TRAINING_SETTINGS[key], f'training.{key}')
        check_learning_rates(parameters)
        noise = self.section("input_noise")
        if not isinstance(noise, dict):
            raise ValueError("input_noise must be a TOML table.")
        method = noise.get("method")
        if method not in ('specified', 'omb'):
            raise ValueError("input_noise.method must be 'specified' or 'omb'.")
        if "search" in noise:
            raise ValueError("Unknown input_noise setting 'search'. Put search definitions directly in the parameter values.")
        angle_method = noise.get('angle_dependence', 'none')
        if angle_method not in ('none', 'linear', 'per_angle'):
            raise ValueError('input_noise.angle_dependence must be none, linear or per_angle.')
        if model['mode'] != 'bls' and any(key in noise for key in ('angle_dependence', 'angle_factor', 'angle_factors')):
            raise ValueError('Angle-dependent input noise requires a BLS model.')
        for key, used in (('angle_factor', angle_method == 'linear'), ('angle_factors', angle_method == 'per_angle')):
            if used:
                require_keys(noise, (key,), 'input_noise')
            elif key in noise:
                raise ValueError(f'input_noise.{key} is not used by angle_dependence = {angle_method}.')
        for key, value in noise_parameters(noise).items():
            resolve_parameter(value, f'input_noise_{key}', f'input_noise.{key}')
        if method == "omb":
            require_keys(noise, ('dataset', 'factor', 'statistic'), 'input_noise')
            check_time_selection(noise, 'input_noise')
            _check_reference(noise, 'input_noise')
            if isinstance(noise["factor"], dict):
                raise ValueError("input_noise.factor does not support a search; use a fixed OMB scaling factor.")
            check_number(noise["factor"], "input_noise.factor", 0)
            if noise['statistic'] not in ('std', 'residual_std'):
                raise ValueError("input_noise.statistic must be 'std' or 'residual_std'.")
            if noise['statistic'] == 'residual_std':
                from openMWR.omb import check_frequency_bands
                require_keys(noise, ('frequency_bands',), 'input_noise')
                check_frequency_bands(noise['frequency_bands'])
            elif 'frequency_bands' in noise:
                raise ValueError('input_noise.frequency_bands is used only by residual_std.')
            if 'max_lwp' in noise:
                check_number(noise['max_lwp'], 'input_noise.max_lwp', 0)
        bias = self.section("bias_correction")
        check_keys(bias, ("data_source", "dataset", "sources", "factor", "frequencies", "max_lwp", *TIME_SELECTION_KEYS), "bias_correction")
        if 'bias_correction' in self.values:
            require_keys(bias, ('dataset', 'factor'), 'bias_correction')
            check_time_selection(bias, 'bias_correction')
            _check_reference(bias, 'bias_correction')
            check_number(bias["factor"], "bias_correction.factor", 0)
            if 'max_lwp' in bias:
                check_number(bias['max_lwp'], 'bias_correction.max_lwp', 0)
            if "frequencies" in bias:
                load_grid(self.project_dir, 'frequencies', bias['frequencies'], label='bias_correction.frequencies')
        optimization = self.section("optimization")
        check_keys(optimization, ("n_trials", "data_source", "dataset", "sources", "seed", "score_weights", *TIME_SELECTION_KEYS), "optimization")
        if self.requires_optimization:
            require_keys(optimization, ('n_trials', 'dataset', 'seed', 'score_weights'), 'optimization')
        if "n_trials" in optimization:
            check_number(optimization["n_trials"], "optimization.n_trials", 1, integer=True)
        _check_reference(optimization, 'optimization')
        if 'seed' in optimization:
            check_seed(optimization['seed'], 'optimization.seed')
        check_time_selection(optimization, 'optimization')
        if 'score_weights' in optimization:
            check_score_weights(optimization['score_weights'], ('TB',) if model['kind'] == 'spectral' else model['output_vars'])
        return settings

    def create_model(self):
        import numpy as np
        from openMWR.models import Model, SpectralConsistencyModel

        settings = self._validate()
        apply_seed(self.section('training')['seed'])
        kind = settings.pop("kind")
        mode = settings.pop('mode')
        inherited = {}
        if any(settings.get(key) == 'dataset' for key in ('frequencies', 'heights', 'angles')):
            from openMWR.datasets import DatasetWorkflow
            inherited = DatasetWorkflow(self.project_dir, self.site, site_config=self.site_config).coordinates(self.section('data')['dataset'])
        for key in ('frequencies', 'heights', 'angles'):
            if key in settings:
                value = inherited[key] if settings[key] == 'dataset' else settings[key]
                settings[key] = np.asarray(value, dtype=float)
        settings['freqs'] = settings.pop('frequencies')
        settings['norm_functions'] = settings.pop('normalization')
        if kind == "spectral":
            if len(settings['freqs']) < 2:
                raise ValueError('Spectral models require at least two frequencies.')
            model = SpectralConsistencyModel(self.name, **settings)
        else:
            if mode == 'bls':
                method = self.section('input_noise').get('angle_dependence', 'none')
                settings['vary_input_noise_with_angle'] = method == 'per_angle'
                settings['vary_input_noise_with_angle_linear'] = method == 'linear'
            if kind == 'nn':
                settings.setdefault('dropout_rate', 0.0)
            settings.setdefault('heights', None)
            model = Model(self.name, is_MLR=kind == 'mlr', is_bls_model=mode == 'bls', **settings)
        return model

    def training_params(self, workflow):
        """Resolve fixed values and searches where training parameters are defined."""
        return {TRAINING_SETTINGS[key]: resolve_parameter(value, TRAINING_SETTINGS[key], f"training.{key}", workflow)
                for key, value in self.section("training").items()}

    def input_noise_params(self, workflow):
        """Resolve complete fixed/search noise settings; OMB supplies TB noise."""
        settings = self.section("input_noise")
        if settings.get('angle_dependence') == 'per_angle':
            required_angles = {float(angle) for angle in workflow.model.angles}
            if {float(angle) for angle in settings['angle_factors']} != required_angles:
                raise ValueError(f'input_noise.angle_factors must specify exactly the selected angles {sorted(required_angles)}.')
        noise = noise_parameters(settings)
        if "TB" in noise:
            tb = noise.pop("TB")
            noise = {**{f"TB_{f}": tb for f in workflow.model.freqs}, **noise}
        allowed = input_noise_parameters(workflow.model)
        required = allowed.copy()
        if settings['method'] == 'omb':
            required.difference_update(f'TB_{frequency}' for frequency in workflow.model.freqs)
        check_keys(noise, allowed, 'input_noise')
        require_keys(noise, required, 'input_noise')
        return {key: resolve_parameter(value, f"input_noise_{key}", f"input_noise.{key}", workflow)
                for key, value in noise.items()}
