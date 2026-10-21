"""Validated, editable recipes for named retrievals in a working project."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import tomllib

from openMWR.config import (INTEGER_TRAINING, TRAINING_PARAMETERS, check_keys,
                            check_number, check_name, check_strings, check_coordinates,
                            validate_training_settings, validate_parameter_usage)
from openMWR.paths import project_root, retrieval_dir, site_config_file


def _parameter(value, key, label, workflow=None):
    """Validate a parameter and lower a search to the existing workflow API."""
    if not isinstance(value, dict):
        validate_training_settings({key: value})
        return value
    if "optimize" in value:
        check_keys(value, ("optimize",), label)
        if value["optimize"] is not True:
            raise ValueError(f"{label}.optimize must be true; use a fixed value to disable optimization.")
        if workflow is None:
            return value
        spaces = workflow.default_search_spaces()
        if key not in spaces:
            raise ValueError(f"No default search space for {label}. Specify low/high or choices explicitly.")
        return spaces[key]
    if "choices" in value:
        check_keys(value, ("choices",), label)
        choices = value["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{label}.choices must be a nonempty list.")
        candidates = choices
        kind, args = "categorical", (choices,)
    else:
        check_keys(value, ("low", "high", "step", "log"), label)
        if key == "loss_fn":
            raise ValueError(f"{label} requires choices rather than numeric bounds.")
        integer = key in INTEGER_TRAINING
        low, high = value.get("low"), value.get("high")
        check_number(low, f"{label}.low", integer=integer)
        check_number(high, f"{label}.high", integer=integer)
        if high < low:
            raise ValueError(f"{label}.high must be at least low.")
        log = value.get("log", False)
        if type(log) is not bool or (log and low <= 0):
            raise ValueError(f"{label}: log must be boolean and requires positive bounds.")
        step = value.get("step", 1 if integer else None)
        if step is not None:
            check_number(step, f"{label}.step", integer=integer)
            if step <= 0 or (log and (not integer or step != 1)):
                raise ValueError(f"{label}: invalid step for this distribution.")
        candidates = (low, high)
        kind, args = ("int" if integer else "float"), (low, high, step, log)
    for candidate in candidates:
        validate_training_settings({key: candidate})
    return getattr(workflow, f"suggest_{kind}")(*args) if workflow is not None else value


@dataclass(frozen=True)
class RetrievalRecipe:
    """One TOML recipe; scientific defaults remain in the model/workflow API."""

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
                    if isinstance(value, dict) and (section, key) != ("model", "norm_functions"):
                        raise ValueError(f"Search definitions are supported only in training and input_noise; {section}.{key} does not support a search.")
        model = self.section("model")
        check_keys(model, ("kind", "mode", "freqs", "heights", "angles", "hidden_layers", "hidden_layer_size", "dropout_rate", "additional_input_vars", "norm_functions", "vary_input_noise_with_angle", "vary_input_noise_with_angle_linear"), "model")
        if model.get("kind") not in ("nn", "mlr", "spectral"):
            raise ValueError("model.kind must be 'nn', 'mlr', or 'spectral'.")
        if model.get("mode", "zenith") not in ("zenith", "bls"):
            raise ValueError("model.mode must be 'zenith' or 'bls'.")
        for key in ("freqs", "heights", "angles"):
            if key in model:
                check_coordinates(model[key], f"model.{key}")
        if "heights" in model and model["heights"] != sorted(model["heights"]):
            raise ValueError("model.heights must be increasing (metres).")
        if "angles" in model and any(v > 90 for v in model["angles"]):
            raise ValueError("model.angles must be between 0 and 90 degrees.")
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
        if model.get("mode", "zenith") != "bls" and any(k in model for k in ("angles", "vary_input_noise_with_angle", "vary_input_noise_with_angle_linear")):
            raise ValueError("Angles and angle-dependent noise require model.mode = 'bls'.")
        if model["kind"] == "spectral":
            check_keys(model, ("kind", "mode", "freqs", "hidden_layer_size", "norm_functions"), "spectral model")
            if model.get("mode", "zenith") != "zenith":
                raise ValueError("Spectral models use zenith observations.")
            if "freqs" in model and len(model["freqs"]) < 2:
                raise ValueError("Spectral models require at least two frequencies.")
        elif "hidden_layer_size" in model:
            raise ValueError("model.hidden_layer_size is only used by spectral models.")
        if model["kind"] == "mlr":
            for key in ("hidden_layers", "dropout_rate"):
                if key in model:
                    raise ValueError(f"model.{key} is not used by mlr models.")
        if "additional_input_vars" in model:
            check_strings(model["additional_input_vars"], "model.additional_input_vars")
        if "norm_functions" in model:
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
        validate_parameter_usage(model["kind"] == "mlr", self.section("training"))
        for key, value in self.section("training").items():
            _parameter(value, key, f"training.{key}")
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
                _parameter(value, f"input_noise_{key}", f"input_noise.{key}")
        if method == "omb":
            if noise.get("data_source", "radiosonde") not in ("radiosonde", "era5"):
                raise ValueError("input_noise.data_source must be 'radiosonde' or 'era5'.")
            if "sources" in noise:
                check_strings(noise["sources"], "input_noise.sources")
            if "factor" in noise:
                if isinstance(noise["factor"], dict):
                    raise ValueError("input_noise.factor does not support a search; use a fixed OMB scaling factor.")
                check_number(noise["factor"], "input_noise.factor", 0)
        bias = self.section("bias_correction")
        check_keys(bias, ("data_source", "dataset", "sources", "factor", "specific_freqs"), "bias_correction")
        if bias:
            if bias.get("data_source", "radiosonde") not in ("radiosonde", "era5"):
                raise ValueError("bias_correction.data_source must be 'radiosonde' or 'era5'.")
            if "factor" in bias:
                check_number(bias["factor"], "bias_correction.factor")
            if "sources" in bias:
                check_strings(bias["sources"], "bias_correction.sources")
            if "specific_freqs" in bias:
                check_coordinates(bias["specific_freqs"], "bias_correction.specific_freqs")
        optimization = self.section("optimization")
        check_keys(optimization, ("n_trials", "data_source", "dataset"), "optimization")
        if "n_trials" in optimization:
            check_number(optimization["n_trials"], "optimization.n_trials", 1, integer=True)
        if optimization.get("data_source", "radiosonde") not in ("radiosonde", "era5"):
            raise ValueError("optimization.data_source must be 'radiosonde' or 'era5'.")
        evaluation = self.section("evaluation")
        check_keys(evaluation, ("data_source", "dataset", "days_gt_20", "station_id", "observations", "test_data"), "evaluation")
        if evaluation.get("data_source", "radiosonde") not in ("radiosonde", "era5"):
            raise ValueError("evaluation.data_source must be 'radiosonde' or 'era5'.")
        for key in ("days_gt_20", "observations", "test_data"):
            if key in evaluation and type(evaluation[key]) is not bool:
                raise ValueError(f"evaluation.{key} must be true or false.")
        if "station_id" in evaluation:
            check_name(evaluation["station_id"], "evaluation.station_id")
        for section in ('input_noise', 'bias_correction', 'optimization', 'evaluation'):
            if 'dataset' in self.section(section):
                check_name(self.section(section)['dataset'], f'{section}.dataset')


    def create_model(self):
        import numpy as np
        from openMWR.consts import std_freqs, std_heights
        from openMWR.models import Model, SpectralConsistencyModel

        settings = dict(self.section("model"))
        kind = settings.pop("kind")
        mode = settings.pop("mode", "zenith")
        settings["freqs"] = np.asarray(settings.get("freqs", std_freqs), dtype=float)
        if kind == "spectral":
            return SpectralConsistencyModel(self.name, **settings)
        settings["heights"] = np.asarray(settings.get("heights", std_heights), dtype=float)
        return Model(self.name, is_MLR=kind == "mlr", is_bls_model=mode == "bls", **settings)

    def training_params(self, workflow):
        """Resolve fixed values and searches where training parameters are defined."""
        return {key: _parameter(value, key, f"training.{key}", workflow)
                for key, value in self.section("training").items()}

    def input_noise_params(self, workflow):
        """Resolve explicit noise overrides; omitted values use method defaults."""
        settings = self.section("input_noise")
        metadata = {"method", "data_source", "dataset", "sources", "factor"} if settings["method"] == "omb" else {"method"}
        noise = {key: value for key, value in settings.items() if key not in metadata}
        if "TB" in noise:
            tb = noise.pop("TB")
            noise = {**{f"TB_{f}": tb for f in workflow.model.freqs}, **noise}
        return {key: _parameter(value, f"input_noise_{key}", f"input_noise.{key}", workflow)
                for key, value in noise.items()}
