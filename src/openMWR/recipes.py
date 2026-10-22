"""Validated, editable recipes for named retrievals in a working project."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from importlib.resources import files
import tomllib

from openMWR.config import (TRAINING_PARAMETERS, check_keys,
                            check_number, check_name, check_strings, check_coordinates,
                            validate_parameter_usage, resolve_parameter, input_noise_parameters,
                            require_keys, load_grid)
from openMWR.paths import project_root, retrieval_dir, site_config_file


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


@dataclass(frozen=True)
class RetrievalRecipe:
    """One TOML recipe with explicit scientific settings and optional features."""

    project_dir: Path
    site: str
    name: str
    values: dict

    @classmethod
    def load(cls, project_dir, site, retrieval):
        check_name(site, "site")
        check_name(retrieval, "retrieval")
        root = project_root(project_dir)
        config = site_config_file(root, site)
        if not config.is_file():
            raise FileNotFoundError(f"Missing site configuration: {config}")
        with config.open("rb") as stream:
            tomllib.load(stream)
        path = retrieval_dir(root, site, retrieval) / "retrieval.toml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing retrieval recipe: {path}")
        with path.open("rb") as stream:
            recipe = cls(root, site, retrieval, tomllib.load(stream))
        recipe._validate()
        return recipe

    def section(self, name):
        return self.values.get(name, {})

    def _validate(self):
        check_keys(self.values, ("schema_version", "model", "data", "training", "input_noise", "bias_correction", "optimization", "evaluation"), "recipe")
        if type(self.values.get("schema_version")) is not int or self.values["schema_version"] != 1:
            raise ValueError("Recipe schema_version must be 1.")
        for section in ("model", "data", "bias_correction", "optimization", "evaluation"):
            table = self.section(section)
            if isinstance(table, dict):
                for key, value in table.items():
                    if isinstance(value, dict) and (section, key) not in (("model", "norm_functions"), ("model", "loss_weights")):
                        raise ValueError(f"Search definitions are supported only in training and input_noise; {section}.{key} does not support a search.")
        model = self.section("model")
        check_keys(model, ("kind", "mode", "freqs", "heights", "angles", "hidden_layers", "hidden_layer_size", "dropout_rate", "additional_input_vars", "norm_functions", "loss_weights", "vary_input_noise_with_angle", "vary_input_noise_with_angle_linear"), "model")
        require_keys(model, ('kind', 'mode', 'freqs', 'norm_functions'), 'model')
        if model["kind"] not in ("nn", "mlr", "spectral"):
            raise ValueError("model.kind must be 'nn', 'mlr', or 'spectral'.")
        if model['mode'] not in ("zenith", "bls"):
            raise ValueError("model.mode must be 'zenith' or 'bls'.")
        settings = dict(model)
        for key in ("freqs", "heights", "angles"):
            if key in model and model[key] != 'dataset':
                settings[key] = load_grid(self.project_dir, 'frequencies' if key == 'freqs' else key,
                                          model[key], label=f'model.{key}')
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
        for key in ("vary_input_noise_with_angle", "vary_input_noise_with_angle_linear"):
            if key in model and type(model[key]) is not bool:
                raise ValueError(f"model.{key} must be true or false.")
        if model.get("vary_input_noise_with_angle") and model.get("vary_input_noise_with_angle_linear"):
            raise ValueError("Select only one angle-dependent input noise scheme.")
        if model['mode'] != "bls" and any(k in model for k in ("angles", "vary_input_noise_with_angle", "vary_input_noise_with_angle_linear")):
            raise ValueError("Angles and angle-dependent noise require model.mode = 'bls'.")
        if model["kind"] == "spectral":
            check_keys(model, ("kind", "mode", "freqs", "hidden_layer_size", "norm_functions"), "spectral model")
            if model['mode'] != "zenith":
                raise ValueError("Spectral models use zenith observations.")
            require_keys(model, ('hidden_layer_size',), 'model')
            if model['freqs'] != 'dataset' and len(settings['freqs']) < 2:
                raise ValueError("Spectral models require at least two frequencies.")
        elif "hidden_layer_size" in model:
            raise ValueError("model.hidden_layer_size is only used by spectral models.")
        if model['kind'] in ('nn', 'mlr'):
            require_keys(model, ('heights', 'additional_input_vars'), 'model')
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
        if not isinstance(model["norm_functions"], dict) or any(v not in ("z_norm", "std_norm", "log_norm", "max_norm") for v in model["norm_functions"].values()):
            raise ValueError("model.norm_functions must map variables to supported normalization functions.")

        data = self.section("data")
        check_keys(data, ('dataset', 'sources', 'end_date_training'), 'data')
        check_name(data.get('dataset'), 'data.dataset')
        if 'sources' in data:
            check_strings(data['sources'], 'data.sources')
            if not data['sources']:
                raise ValueError('data.sources must not be empty.')
            for source in data['sources']:
                check_name(source, 'data.sources')
        if 'end_date_training' in data:
            if not isinstance(data['end_date_training'], str):
                raise ValueError('data.end_date_training must be a quoted ISO date.')
            date.fromisoformat(data['end_date_training'])
        check_keys(self.section("training"), TRAINING_PARAMETERS, "training")
        validate_parameter_usage(model["kind"] == "mlr", self.section("training"), complete=True)
        for key, value in self.section("training").items():
            resolve_parameter(value, key, f"training.{key}")
        noise = self.section("input_noise")
        if not isinstance(noise, dict):
            raise ValueError("input_noise must be a TOML table.")
        method = noise.get("method")
        if method not in ("standard", "omb"):
            raise ValueError("input_noise.method must be 'standard' or 'omb'.")
        if "search" in noise:
            raise ValueError("Unknown input_noise setting 'search'. Put search definitions directly in the parameter values.")
        for key, value in noise.items():
            if key == "method":
                continue
            if method != "omb" or key not in ("data_source", "dataset", "sources", "factor"):
                resolve_parameter(value, f"input_noise_{key}", f"input_noise.{key}")
        if method == "omb":
            require_keys(noise, ('data_source', 'dataset', 'factor'), 'input_noise')
            _check_reference(noise, 'input_noise')
            if "sources" in noise:
                check_strings(noise["sources"], "input_noise.sources")
            if isinstance(noise["factor"], dict):
                raise ValueError("input_noise.factor does not support a search; use a fixed OMB scaling factor.")
            check_number(noise["factor"], "input_noise.factor", 0)
        bias = self.section("bias_correction")
        check_keys(bias, ("data_source", "dataset", "sources", "factor", "specific_freqs"), "bias_correction")
        if 'bias_correction' in self.values:
            require_keys(bias, ('data_source', 'dataset', 'factor'), 'bias_correction')
            _check_reference(bias, 'bias_correction')
            check_number(bias["factor"], "bias_correction.factor")
            if "sources" in bias:
                check_strings(bias["sources"], "bias_correction.sources")
            if "specific_freqs" in bias:
                check_coordinates(bias["specific_freqs"], "bias_correction.specific_freqs")
        optimization = self.section("optimization")
        check_keys(optimization, ("n_trials", "data_source", "dataset"), "optimization")
        if any(isinstance(value, dict) for value in (*self.section('training').values(), *noise.values())):
            require_keys(optimization, ('n_trials', 'data_source', 'dataset'), 'optimization')
        if "n_trials" in optimization:
            check_number(optimization["n_trials"], "optimization.n_trials", 1, integer=True)
        _check_reference(optimization, 'optimization')
        evaluation = self.section("evaluation")
        check_keys(evaluation, ("data_source", "dataset", "days_gt_20", "station_id", "observations", "test_data"), "evaluation")
        if 'evaluation' in self.values:
            require_keys(evaluation, ('observations', 'test_data'), 'evaluation')
            if evaluation['observations']:
                require_keys(evaluation, ('data_source', 'dataset', 'days_gt_20'), 'evaluation')
        _check_reference(evaluation, 'evaluation')
        for key in ("days_gt_20", "observations", "test_data"):
            if key in evaluation and type(evaluation[key]) is not bool:
                raise ValueError(f"evaluation.{key} must be true or false.")
        if "station_id" in evaluation:
            check_name(evaluation["station_id"], "evaluation.station_id")
        return settings

    def create_model(self):
        import numpy as np
        from openMWR.models import Model, SpectralConsistencyModel

        settings = self._validate()
        kind = settings.pop("kind")
        mode = settings.pop('mode')
        inherited = {}
        if any(settings.get(key) == 'dataset' for key in ('freqs', 'heights', 'angles')):
            from openMWR.datasets import DatasetWorkflow
            inherited = DatasetWorkflow(self.project_dir, self.site).coordinates(self.section('data')['dataset'])
        for key in ('freqs', 'heights', 'angles'):
            if key in settings:
                coordinate_kind = 'frequencies' if key == 'freqs' else key
                value = inherited[coordinate_kind] if settings[key] == 'dataset' else settings[key]
                settings[key] = np.asarray(value, dtype=float)
        if kind == "spectral":
            if len(settings['freqs']) < 2:
                raise ValueError('Spectral models require at least two frequencies.')
            model = SpectralConsistencyModel(self.name, **settings)
        else:
            if kind == 'nn':
                settings.setdefault('dropout_rate', 0.0)
            model = Model(self.name, is_MLR=kind == 'mlr', is_bls_model=mode == 'bls', **settings)
        require_keys(model.norm_functions, set(model.input.data_vars) | set(model.output.data_vars), 'model.norm_functions')
        return model

    def training_params(self, workflow):
        """Resolve fixed values and searches where training parameters are defined."""
        return {key: resolve_parameter(value, key, f"training.{key}", workflow)
                for key, value in self.section("training").items()}

    def input_noise_params(self, workflow):
        """Resolve complete fixed/search noise settings; OMB supplies TB noise."""
        settings = self.section("input_noise")
        metadata = {"method", "data_source", "dataset", "sources", "factor"} if settings["method"] == "omb" else {"method"}
        noise = {key: value for key, value in settings.items() if key not in metadata}
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
