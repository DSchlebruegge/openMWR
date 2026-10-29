"""Shared configuration checks, independent of recipes and workflow execution."""

from difflib import get_close_matches
from datetime import date, datetime
import math
from numbers import Integral, Real
import re
import tomllib

from openMWR.paths import project_root

INTEGER_TRAINING = ("epochs", "batch_size", "epoch_max_lr")
_NN_TRAINING = {"epochs", "batch_size", "lr_height", "lr_low", "epoch_max_lr", "loss_fn", "weight_decay"}
TRAINING_PARAMETERS = _NN_TRAINING | {"seed", "bias_weight"}
TIME_SELECTION_KEYS = ('start_date', 'end_date', 'days_of_month')
PROFILE_OUTPUTS = ('T', 'rh', 'lwc', 'lwp', 'iwv')


def check_output_vars(values):
    """Select an ordered, nonempty subset of the profile retrieval outputs."""
    check_strings(values, 'model.output_vars')
    if not values or set(values) - set(PROFILE_OUTPUTS):
        raise ValueError(f'model.output_vars must select from {list(PROFILE_OUTPUTS)}.')


def training_parameters(is_mlr, loss_fn=None):
    """Training settings used by the selected fitting method."""
    if is_mlr:
        return {"seed"}
    choices = (loss_fn.get('choices', []) if isinstance(loss_fn, dict)
               else getattr(loss_fn, 'choices', [loss_fn]))
    return _NN_TRAINING | {'seed'} | ({'bias_weight'} if any(v in ('MAEPlusBiasLoss', 'MSEPlusBiasLoss') for v in choices) else set())


def check_seed(value, label='seed'):
    """An explicit random run or a seed shared by Python, NumPy and PyTorch."""
    if value == 'random':
        return
    check_number(value, label, 0, integer=True)
    if value >= 2**32:
        raise ValueError(f'{label} must be smaller than 2**32.')


def apply_seed(seed):
    """Seed initialization and training; 'random' leaves the generators alone."""
    check_seed(seed)
    if seed != 'random':
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def check_days(days, label='days_of_month'):
    """Validate an inclusive day-of-month range, also used by Python callers."""
    check_keys(days, ('first', 'last'), label)
    require_keys(days, ('first', 'last'), label)
    for key in ('first', 'last'):
        check_number(days[key], f'{label}.{key}', 1, integer=True)
    if not days['first'] <= days['last'] <= 31:
        raise ValueError(f'{label} requires 1 <= first <= last <= 31.')


def check_split_days(test_days_of_month, validation_days_of_month):
    """Require distinct monthly test/validation ranges, leaving training days."""
    for key, days in (('test_days_of_month', test_days_of_month),
                      ('validation_days_of_month', validation_days_of_month)):
        check_days(days, f'split.{key}')
    if max(test_days_of_month['first'], validation_days_of_month['first']) <= min(test_days_of_month['last'], validation_days_of_month['last']):
        raise ValueError('split.test_days_of_month and validation_days_of_month must not overlap.')
    if sum(days['last'] - days['first'] + 1 for days in (test_days_of_month, validation_days_of_month)) == 31:
        raise ValueError('Split day ranges must leave days for training.')


def check_time_selection(settings, label='selection'):
    """Check optional ISO date bounds and an inclusive day-of-month range."""
    for key in ('start_date', 'end_date'):
        value = settings.get(key)
        if value is not None:
            try:
                if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                    raise ValueError
            except ValueError:
                raise ValueError(f'{label}.{key} must be a quoted ISO date (YYYY-MM-DD).') from None
    if settings.get('start_date') is not None and settings.get('end_date') is not None:
        if settings['start_date'] > settings['end_date']:
            raise ValueError(f'{label}.start_date must not be after end_date.')
    if settings.get('days_of_month') is not None:
        check_days(settings['days_of_month'], f'{label}.days_of_month')


def select_time(data, *, start_date=None, end_date=None, days_of_month=None):
    """Intersect date and monthly selections, including the whole end day."""
    import numpy as np

    check_time_selection(dict(start_date=start_date, end_date=end_date, days_of_month=days_of_month))
    selected = data.time.notnull()
    if start_date is not None:
        selected = selected & (data.time >= np.datetime64(start_date))
    if end_date is not None:
        selected = selected & (data.time < np.datetime64(end_date) + np.timedelta64(1, 'D'))
    if days_of_month is not None:
        selected = selected & (data.time.dt.day >= days_of_month['first']) & (data.time.dt.day <= days_of_month['last'])
    return data.isel(time=selected)


def check_score_weights(weights, variables):
    check_keys(weights, variables, 'optimization.score_weights')
    for key, value in weights.items():
        check_number(value, f'optimization.score_weights.{key}', 0)
    if not any(value > 0 for value in weights.values()):
        raise ValueError('optimization.score_weights needs at least one positive weight.')


def check_cloud(cloud):
    """Validate the cloud choices shared by profile recipes and Python calls."""
    label = 'preparation.cloud'
    check_keys(cloud, ('rh_threshold', 'min_temperature', 'adiabatic_fraction', 'a', 'b'), label)
    require_keys(cloud, ('rh_threshold', 'min_temperature', 'adiabatic_fraction'), label)
    check_number(cloud['min_temperature'], f'{label}.min_temperature', 0)
    check_number(cloud['rh_threshold'], f'{label}.rh_threshold', 0)
    if not 0 < cloud['rh_threshold'] <= 1:
        raise ValueError(f'{label}.rh_threshold must be greater than zero and at most one.')
    method = cloud['adiabatic_fraction']
    if method == 'power_law':
        require_keys(cloud, ('a', 'b'), label)
        for key in ('a', 'b'):
            check_number(cloud[key], f'{label}.{key}', 0)
        if cloud['a'] > 1 or cloud['b'] <= 0:
            raise ValueError(f'{label} requires 0 <= a <= 1 and b > 0.')
    elif method in ('karstens', 'none'):
        if {'a', 'b'} & cloud.keys():
            raise ValueError(f'{label}: a and b apply only to power_law; remove them for {method}.')
    else:
        raise ValueError(f"{label}.adiabatic_fraction must be 'power_law', 'karstens' or 'none'.")


def check_profile_quality(quality):
    label = 'preparation.quality'
    check_keys(quality, ('min_top_height', 'min_levels'), label)
    require_keys(quality, ('min_top_height', 'min_levels'), label)
    check_number(quality['min_top_height'], f'{label}.min_top_height', 0)
    check_number(quality['min_levels'], f'{label}.min_levels', 1, integer=True)


def check_height_range(value, label='height_range', *, allow_model=False):
    if allow_model and value == 'model':
        return
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f'{label} must be [minimum, maximum] in metres' + (' or "model".' if allow_model else '.'))
    for bound in value:
        check_number(bound, label, 0)
    if value[0] >= value[1]:
        raise ValueError(f'{label} requires minimum < maximum.')


def select_height_range(data, height_range):
    check_height_range(height_range, allow_model=True)
    if height_range == 'model' or 'height' not in data.dims:
        return data
    selected = data.sel(height=slice(*height_range))
    if not selected.sizes['height']:
        raise ValueError(f'No heights in the selected height_range {height_range}.')
    return selected


def check_hatpro_format(format, time_offset_hours=None):
    """Validate both site settings and standalone HATPRO file reads."""
    if format not in ('auto', 'netcdf', 'binary'):
        raise ValueError('hatpro_format must be "auto", "netcdf" or "binary".')
    if time_offset_hours is not None:
        check_number(time_offset_hours, 'hatpro_time_offset_hours')
        if abs(time_offset_hours) > 24:
            raise ValueError('hatpro_time_offset_hours must be between -24 and 24.')


def check_analysis(settings):
    check_keys(settings, ('max_time_difference_minutes', 'exclude_rain', 'height_range', 'required_observations'), 'analysis')
    require_keys(settings, ('max_time_difference_minutes', 'exclude_rain', 'height_range', 'required_observations'), 'analysis')
    check_number(settings['max_time_difference_minutes'], 'analysis.max_time_difference_minutes', 0, integer=True)
    if type(settings['exclude_rain']) is not bool:
        raise ValueError('analysis.exclude_rain must be true or false.')
    check_height_range(settings['height_range'], 'analysis.height_range')
    check_strings(settings['required_observations'], 'analysis.required_observations')
    if not settings['required_observations']:
        raise ValueError('analysis.required_observations must not be empty.')


def check_invalid_measurements(rules):
    """Validate variable-specific rules containing UTC dates or exact intervals."""
    if not isinstance(rules, list):
        raise ValueError('invalid_measurements must be an array of tables.')
    for rule in rules:
        label = 'invalid_measurements'
        check_keys(rule, ('variables', 'periods', 'reason'), label)
        require_keys(rule, ('variables', 'periods'), label)
        check_strings(rule['variables'], f'{label}.variables')
        if not rule['variables']:
            raise ValueError(f'{label}.variables must not be empty.')
        if 'reason' in rule and not isinstance(rule['reason'], str):
            raise ValueError(f'{label}.reason must be text.')
        if not isinstance(rule['periods'], list) or not rule['periods']:
            raise ValueError(f'{label}.periods must be a non-empty array of tables.')
        label += '.periods'
        for period in rule['periods']:
            check_keys(period, ('start_date', 'end_date', 'start_time', 'end_time'), label)
            if any(key in period for key in ('start_time', 'end_time')):
                if any(key in period for key in ('start_date', 'end_date')):
                    raise ValueError(f'{label}: use either date bounds or time bounds, without mixing them.')
                require_keys(period, ('start_time',), label)
                bounds = {}
                for key in ('start_time', 'end_time'):
                    if key not in period:
                        continue
                    value = period[key]
                    try:
                        if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z', value):
                            raise ValueError
                        bounds[key] = datetime.fromisoformat(value)
                    except ValueError:
                        raise ValueError(f'{label}.{key} must be a quoted UTC timestamp (YYYY-MM-DDTHH:MM:SS[.ffffff]Z).') from None
                if 'end_time' in bounds and bounds['start_time'] >= bounds['end_time']:
                    raise ValueError(f'{label}.start_time must be before the exclusive end_time.')
            else:
                require_keys(period, ('start_date',), label)
                if any(period[key] is None for key in ('start_date', 'end_date') if key in period):
                    raise ValueError(f'{label}: date bounds must be quoted ISO dates.')
                check_time_selection(period, label)


def check_infrared(settings):
    label = 'radiative_transfer.infrared'
    check_keys(settings, ('droplet_number_density', 'min_effective_radius'), label)
    require_keys(settings, ('droplet_number_density', 'min_effective_radius'), label)
    for key, value in settings.items():
        check_number(value, f'{label}.{key}', 0)
        if value == 0:
            raise ValueError(f'{label}.{key} must be positive.')
    if not 2.5 <= settings['min_effective_radius'] < 60:
        raise ValueError(f"{label}.min_effective_radius must be at least 2.5 and below 60 micrometres for libRadtran's Hu-Stamnes cloud optics.")


def input_noise_parameters(model):
    """Names of the noise parameters used by the model's active inputs."""
    names = {f'TB_{frequency}' for frequency in model.freqs}
    names.update(var for var in model.input_vars if var != 'TB')
    if model.is_bls_model:
        if model.vary_input_noise_with_angle:
            names.update(f'TB_{angle}' for angle in model.angles)
        elif model.vary_input_noise_with_angle_linear:
            names.add('TB_angle_factor')
    return names


def require_keys(table, required, section):
    missing = set(required) - table.keys()
    if missing:
        raise ValueError(f"Missing {section} settings: {', '.join(sorted(missing))}. Specify them explicitly.")


def load_grid(project_dir, kind, values, *, label=None):
    """Load a named project grid or validate an explicit numeric list.

    Names come only from the project's grids.toml, in heights (metres),
    frequencies (GHz), or angles (degrees). Reads observe edits on later calls.
    """
    kinds = ('heights', 'frequencies', 'angles')
    if kind not in kinds:
        raise ValueError(f'Grid kind must be one of {", ".join(kinds)}.')
    label = label or kind
    if isinstance(values, str):
        path = project_root(project_dir) / 'grids.toml'
        if not path.is_file():
            raise FileNotFoundError(f'Missing project grid definitions: {path}. Define [{kind}] grids there or provide a numeric list.')
        with path.open('rb') as stream:
            definitions = tomllib.load(stream)
        check_keys(definitions, kinds, str(path))
        grids = definitions.get(kind, {})
        if not isinstance(grids, dict):
            raise ValueError(f'{path}: {kind} must be a TOML table of named numeric lists.')
        if values not in grids:
            raise ValueError(f"Unknown {label} grid {values!r} in {path}; choose {', '.join(grids) or 'a defined name'} or provide a numeric list.")
        values = grids[values]
    check_coordinates(values, label)
    if kind == 'heights' and values != sorted(values):
        raise ValueError(f'{label} must be increasing (metres).')
    if kind == 'angles' and any(value <= 0 or value > 90 for value in values):
        raise ValueError(f'{label} must be between 0 and 90 degrees (excluding zero).')
    return list(values)


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
    for key in ("lr_height", "lr_low", "weight_decay", "bias_weight"):
        if key in settings:
            check_number(settings[key], f"training.{key}", 0)
    if 'seed' in settings:
        check_seed(settings['seed'], 'training.seed')
    if "loss_fn" in settings and settings["loss_fn"] not in ("MAE", "MSE", "MAEPlusBiasLoss", "MSEPlusBiasLoss"):
        raise ValueError(f"Unknown training.loss_fn: {settings['loss_fn']!r}.")
    for key, value in settings.items():
        if key.startswith("input_noise_"):
            check_number(value, key, 0)


def validate_parameter_usage(is_mlr, settings, *, complete=False):
    """Reject explicit parameters that the chosen fitting method ignores."""
    required = training_parameters(is_mlr, settings.get('loss_fn'))
    unused = TRAINING_PARAMETERS - required
    for key in sorted(unused.intersection(settings)):
        model = "mlr" if is_mlr else "neural"
        raise ValueError(f"training.{key} is not used by {model} models.")
    if complete:
        require_keys(settings, required, 'training')


def resolve_parameter(value, key, label, workflow=None):
    """Validate a parameter and lower a search to the existing workflow API."""
    if key == 'seed':
        check_seed(value, label)
        return value
    if workflow is not None and isinstance(value, (workflow.suggest_int, workflow.suggest_float, workflow.suggest_categorical)):
        return value
    if not isinstance(value, dict):
        validate_training_settings({key: value})
        return value
    if "optimize" in value:
        raise ValueError(f"{label}: specify low/high or choices explicitly instead of optimize = true.")
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
