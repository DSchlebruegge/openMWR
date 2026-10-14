"""Validated, editable recipes for named retrievals in a working project."""

from dataclasses import dataclass
from datetime import date
from difflib import get_close_matches
import inspect
import math
from numbers import Integral, Real
from pathlib import Path
import tomllib

from openMWR.paths import project_root, retrieval_dir, site_config_file, site_data_dir

_INTEGER_TRAINING = ("epochs", "batch_size", "repeats_for_linear_regression", "epoch_max_lr")
_NN_TRAINING = {"epochs", "batch_size", "lr_height", "lr_low", "epoch_max_lr", "loss_fn", "weight_decay"}


def _known(table, allowed, section):
    if not isinstance(table, dict):
        raise ValueError(f"{section} must be a TOML table.")
    for key in table.keys() - set(allowed):
        matches = get_close_matches(key, allowed, n=1)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        raise ValueError(f"Unknown {section} setting '{key}'.{hint}")


def _number(value, label, minimum=None, integer=False):
    expected = Integral if integer else Real
    if isinstance(value, bool) or not isinstance(value, expected) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite {'integer' if integer else 'number'}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")


def _component(value, label):
    if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a nonempty name without directory separators.")


def _strings(values, label):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label} must be a list of nonempty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates.")


def _coordinates(values, label, minimum=0):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty list.")
    for value in values:
        _number(value, label, minimum)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates.")


def validate_training_settings(settings):
    """Check fixed training values, also used for resolved optimization trials."""
    for key in _INTEGER_TRAINING:
        if key in settings:
            _number(settings[key], f"training.{key}", 0 if key == "epoch_max_lr" else 1, integer=True)
    for key in ("lr_height", "lr_low", "weight_decay"):
        if key in settings:
            _number(settings[key], f"training.{key}", 0)
    if "loss_fn" in settings and settings["loss_fn"] not in ("MAE", "MSE", "MAEPlusBiasLoss", "MSEPlusBiasLoss"):
        raise ValueError(f"Unknown training.loss_fn: {settings['loss_fn']!r}.")
    for key, value in settings.items():
        if key.startswith("input_noise_"):
            _number(value, key, 0)


def validate_parameter_usage(is_mlr, settings):
    """Reject explicit parameters that the chosen fitting method ignores."""
    unused = _NN_TRAINING if is_mlr else {"repeats_for_linear_regression"}
    for key in sorted(unused.intersection(settings)):
        model = "mlr" if is_mlr else "neural"
        raise ValueError(f"training.{key} is not used by {model} models.")


def _parameter(value, key, label, workflow=None):
    """Validate a parameter and lower a search to the existing workflow API."""
    if not isinstance(value, dict):
        validate_training_settings({key: value})
        return value
    if "optimize" in value:
        _known(value, ("optimize",), label)
        if value["optimize"] is not True:
            raise ValueError(f"{label}.optimize must be true; use a fixed value to disable optimization.")
        if workflow is None:
            return value
        spaces = workflow.default_search_spaces()
        if key not in spaces:
            raise ValueError(f"No default search space for {label}. Specify low/high or choices explicitly.")
        return spaces[key]
    if "choices" in value:
        _known(value, ("choices",), label)
        choices = value["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{label}.choices must be a nonempty list.")
        candidates = choices
        kind, args = "categorical", (choices,)
    else:
        _known(value, ("low", "high", "step", "log"), label)
        if key == "loss_fn":
            raise ValueError(f"{label} requires choices rather than numeric bounds.")
        integer = key in _INTEGER_TRAINING
        low, high = value.get("low"), value.get("high")
        _number(low, f"{label}.low", integer=integer)
        _number(high, f"{label}.high", integer=integer)
        if high < low:
            raise ValueError(f"{label}.high must be at least low.")
        log = value.get("log", False)
        if type(log) is not bool or (log and low <= 0):
            raise ValueError(f"{label}: log must be boolean and requires positive bounds.")
        step = value.get("step", 1 if integer else None)
        if step is not None:
            _number(step, f"{label}.step", integer=integer)
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
        _component(site, "site")
        _component(retrieval, "retrieval")
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
        from openMWR.train import TrainingWorkflow

        _known(self.values, ("schema_version", "model", "data", "training", "input_noise", "bias_correction", "optimization", "evaluation"), "recipe")
        if type(self.values.get("schema_version")) is not int or self.values["schema_version"] != 1:
            raise ValueError("Recipe schema_version must be 1.")
        for section in ("model", "data", "bias_correction", "optimization", "evaluation"):
            table = self.section(section)
            if isinstance(table, dict):
                for key, value in table.items():
                    if isinstance(value, dict) and (section, key) != ("model", "norm_functions"):
                        raise ValueError(f"Search definitions are supported only in training and input_noise; {section}.{key} does not support a search.")
        model = self.section("model")
        _known(model, ("kind", "mode", "freqs", "heights", "angles", "hidden_layers", "hidden_layer_size", "dropout_rate", "additional_input_vars", "norm_functions", "vary_input_noise_with_angle", "vary_input_noise_with_angle_linear"), "model")
        if model.get("kind") not in ("nn", "mlr", "spectral"):
            raise ValueError("model.kind must be 'nn', 'mlr', or 'spectral'.")
        if model.get("mode", "zenith") not in ("zenith", "bls"):
            raise ValueError("model.mode must be 'zenith' or 'bls'.")
        for key in ("freqs", "heights", "angles"):
            if key in model:
                _coordinates(model[key], f"model.{key}")
        if "heights" in model and model["heights"] != sorted(model["heights"]):
            raise ValueError("model.heights must be increasing (metres).")
        if "angles" in model and any(v > 90 for v in model["angles"]):
            raise ValueError("model.angles must be between 0 and 90 degrees.")
        if "hidden_layers" in model:
            if not isinstance(model["hidden_layers"], list) or not model["hidden_layers"]:
                raise ValueError("model.hidden_layers must be a nonempty list.")
            for width in model["hidden_layers"]:
                _number(width, "model.hidden_layers", 1, integer=True)
        if "hidden_layer_size" in model:
            _number(model["hidden_layer_size"], "model.hidden_layer_size", 1, integer=True)
        if "dropout_rate" in model:
            _number(model["dropout_rate"], "model.dropout_rate", 0)
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
            _known(model, ("kind", "mode", "freqs", "hidden_layer_size", "norm_functions"), "spectral model")
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
            _strings(model["additional_input_vars"], "model.additional_input_vars")
        if "norm_functions" in model:
            if not isinstance(model["norm_functions"], dict) or any(v not in ("z_norm", "std_norm", "log_norm", "max_norm") for v in model["norm_functions"].values()):
                raise ValueError("model.norm_functions must map variables to supported normalization functions.")

        data = self.section("data")
        _known(data, ("sources", "end_date_training", "variant"), "data")
        _strings(data.get("sources"), "data.sources")
        if not data["sources"]:
            raise ValueError("data.sources must not be empty.")
        for source in data["sources"]:
            _component(source, "data.sources")
        if "end_date_training" in data:
            if not isinstance(data["end_date_training"], str):
                raise ValueError('data.end_date_training must be a quoted ISO date, e.g. "2024-10-31".')
            date.fromisoformat(data["end_date_training"])
        suffix = data.get("variant", "")
        if not isinstance(suffix, str) or "/" in suffix or "\\" in suffix:
            raise ValueError("data.variant must be a string without directory separators.")
        allowed_training = set(inspect.signature(TrainingWorkflow.add_standard_training_hyper_params).parameters) - {"self"}
        _known(self.section("training"), allowed_training, "training")
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
            if method != "omb" or key not in ("data_source", "sources", "factor"):
                _parameter(value, f"input_noise_{key}", f"input_noise.{key}")
        if method == "omb":
            if noise.get("data_source", "radiosonde") not in ("radiosonde", "era5"):
                raise ValueError("input_noise.data_source must be 'radiosonde' or 'era5'.")
            if "sources" in noise:
                _strings(noise["sources"], "input_noise.sources")
            if "factor" in noise:
                if isinstance(noise["factor"], dict):
                    raise ValueError("input_noise.factor does not support a search; use a fixed OMB scaling factor.")
                _number(noise["factor"], "input_noise.factor", 0)
        bias = self.section("bias_correction")
        _known(bias, ("data_source", "sources", "factor", "specific_freqs"), "bias_correction")
        if bias:
            if bias.get("data_source", "radiosonde") not in ("radiosonde", "era5"):
                raise ValueError("bias_correction.data_source must be 'radiosonde' or 'era5'.")
            if "factor" in bias:
                _number(bias["factor"], "bias_correction.factor")
            if "sources" in bias:
                _strings(bias["sources"], "bias_correction.sources")
            if "specific_freqs" in bias:
                _coordinates(bias["specific_freqs"], "bias_correction.specific_freqs")
        optimization = self.section("optimization")
        _known(optimization, ("n_trials", "data_source"), "optimization")
        if "n_trials" in optimization:
            _number(optimization["n_trials"], "optimization.n_trials", 1, integer=True)
        if optimization.get("data_source", "radiosonde") not in ("radiosonde", "era5"):
            raise ValueError("optimization.data_source must be 'radiosonde' or 'era5'.")
        evaluation = self.section("evaluation")
        _known(evaluation, ("data_source", "days_gt_20", "station_id", "observations", "test_data"), "evaluation")
        if evaluation.get("data_source", "radiosonde") not in ("radiosonde", "era5"):
            raise ValueError("evaluation.data_source must be 'radiosonde' or 'era5'.")
        for key in ("days_gt_20", "observations", "test_data"):
            if key in evaluation and type(evaluation[key]) is not bool:
                raise ValueError(f"evaluation.{key} must be true or false.")
        if "station_id" in evaluation:
            _component(evaluation["station_id"], "evaluation.station_id")

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

    def require_data(self, splits):
        data = self.section("data")
        suffix = data.get("variant", "")
        suffix = f"_{suffix}" if suffix else ""
        for split in splits:
            for source in data["sources"]:
                path = site_data_dir(self.project_dir, self.site, "training") / f"{split}_{source}{suffix}.nc"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing prepared dataset: {path}. Prepare this dataset before running the selected action.")

    def training_params(self, workflow):
        """Resolve fixed values and searches where training parameters are defined."""
        return {key: _parameter(value, key, f"training.{key}", workflow)
                for key, value in self.section("training").items()}

    def input_noise_params(self, workflow):
        """Resolve explicit noise overrides; omitted values use method defaults."""
        settings = self.section("input_noise")
        metadata = {"method", "data_source", "sources", "factor"} if settings["method"] == "omb" else {"method"}
        noise = {key: value for key, value in settings.items() if key not in metadata}
        if "TB" in noise:
            tb = noise.pop("TB")
            noise = {**{f"TB_{f}": tb for f in workflow.model.freqs}, **noise}
        return {key: _parameter(value, f"input_noise_{key}", f"input_noise.{key}", workflow)
                for key, value in noise.items()}
