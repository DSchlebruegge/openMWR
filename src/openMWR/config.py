"""Shared configuration checks, independent of recipes and workflow execution."""

from difflib import get_close_matches
import math
from numbers import Integral, Real

INTEGER_TRAINING = ("epochs", "batch_size", "repeats_for_linear_regression", "epoch_max_lr")
_NN_TRAINING = {"epochs", "batch_size", "lr_height", "lr_low", "epoch_max_lr", "loss_fn", "weight_decay"}
TRAINING_PARAMETERS = _NN_TRAINING | {"repeats_for_linear_regression"}


def check_keys(table, allowed, section):
    if not isinstance(table, dict):
        raise ValueError(f"{section} must be a TOML table.")
    for key in table.keys() - set(allowed):
        matches = get_close_matches(key, allowed, n=1)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        raise ValueError(f"Unknown {section} setting '{key}'.{hint}")


def check_number(value, label, minimum=None, integer=False):
    expected = Integral if integer else Real
    if isinstance(value, bool) or not isinstance(value, expected) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite {'integer' if integer else 'number'}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")


def check_name(value, label):
    if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a nonempty name without directory separators.")


def check_strings(values, label):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label} must be a list of nonempty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates.")


def check_coordinates(values, label, minimum=0):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty list.")
    for value in values:
        check_number(value, label, minimum)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates.")


def validate_training_settings(settings):
    """Check fixed training values, also used for resolved optimization trials."""
    for key in INTEGER_TRAINING:
        if key in settings:
            check_number(settings[key], f"training.{key}", 0 if key == "epoch_max_lr" else 1, integer=True)
    for key in ("lr_height", "lr_low", "weight_decay"):
        if key in settings:
            check_number(settings[key], f"training.{key}", 0)
    if "loss_fn" in settings and settings["loss_fn"] not in ("MAE", "MSE", "MAEPlusBiasLoss", "MSEPlusBiasLoss"):
        raise ValueError(f"Unknown training.loss_fn: {settings['loss_fn']!r}.")
    for key, value in settings.items():
        if key.startswith("input_noise_"):
            check_number(value, key, 0)


def validate_parameter_usage(is_mlr, settings):
    """Reject explicit parameters that the chosen fitting method ignores."""
    unused = _NN_TRAINING if is_mlr else {"repeats_for_linear_regression"}
    for key in sorted(unused.intersection(settings)):
        model = "mlr" if is_mlr else "neural"
        raise ValueError(f"training.{key} is not used by {model} models.")
