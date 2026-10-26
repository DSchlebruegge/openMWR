"""Reusable retrieval evaluation and portable HTML reports."""

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import json
import logging
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib import colormaps

from openMWR.observations import load_analysis_dataset
from openMWR.config import check_time_selection, select_time, TIME_SELECTION_KEYS, check_height_range, select_height_range
from openMWR.calc_errors import cal_RMSE, calculate_errors
from openMWR.hatpro_data import calc_derived_quantities
from openMWR.models import Model, load_model
from openMWR.plot import plot_err, plot_scatter_lwp
from openMWR.predict import run_model_list, validate_model_data
from openMWR.devices import resolve_device, device_name
from openMWR.xr_utils import load_combined_dataset
from openMWR.paths import project_root, site_root, dataset_file, checkpoint_file, retrieval_dir
from openMWR.provenance import checkpoint_id, git_identity, software_identity, retrieval_log, utc_now


logger = logging.getLogger(__name__)
VARIABLE_LABELS = {
    "T": "Temperature [K]",
    "rh": "Relative humidity [%]",
    "lwc": "Liquid water content [g/m³]",
    "lwp": "Liquid water path [g/m²]",
}


@dataclass(frozen=True)
class SiteSettings:
    models: tuple[str, ...]
    main_model: str
    bls_models: tuple[str, ...]
    include_rpg: bool
    days_of_month: dict[str, int] | None
    observation_height_range: list[float] | str
    test_height_range: list[float] | str
    station_id: str | None = None
    training_dataset: str = 'training'
    analysis_dataset: str | None = None
    start_date: str | None = None
    end_date: str | None = None


@dataclass
class Evaluation:
    key: str
    title: str
    error_types: tuple[str, ...]
    samples: dict
    metrics: dict[str, xr.Dataset]
    note: str = ""


@dataclass
class Analysis:
    metadata: dict
    evaluations: list[Evaluation]


def _sample_summary(ds):
    variable = 'T' if 'T' in ds else next(iter(ds.data_vars))
    finite = np.isfinite(ds[variable])
    if "height" in finite.dims:
        finite = finite.any("height")
    if 'frq' in finite.dims:
        finite = finite.any('frq')
    return {
        "time_coordinates": ds.sizes["time"],
        "reference_temperature_profiles": int(finite.sum()),
        "stations": ds.station.values.tolist()
        if "station" in ds.coords
        else ds.attrs.get("analysis_stations", []),
        "start": str(ds.time.min().values),
        "end": str(ds.time.max().values),
    }


def _paired_profiles(pred, reference, variables):
    """Count paired profiles, rather than padded station/time coordinates."""
    counts = {}
    for var in variables:
        predicted, truth = xr.align(pred[var], reference[var], join="inner")
        valid = np.isfinite(predicted) & np.isfinite(truth)
        if "height" in valid.dims:
            valid = valid.any("height")
        counts[var] = valid.sum(
            [dim for dim in ("time", "station") if dim in valid.dims]
        )
    return xr.Dataset(counts)


def _evaluate(key, title, pred, reference, variables, error_types):
    errors, profiles = calculate_errors(
        pred[variables], reference[variables], error_types=list(error_types), variables=variables
    )
    return Evaluation(key, title, error_types, _sample_summary(reference[variables]),
                      {'errors': errors[variables], 'error_profiles': profiles,
                       'paired_profiles': _paired_profiles(pred, reference, variables)})


def _select_station(data, station_id, path):
    """Select a named station; an unambiguous single station needs no override."""
    if 'station' not in data.dims:
        raise ValueError(f'Missing radiosonde station dimension in {path}.')
    available = [str(value) for value in data.station.values]
    if len(set(available)) != len(available):
        raise ValueError(f'Duplicate station IDs in {path}.')
    if station_id is None:
        if len(available) != 1:
            raise ValueError(f'Select evaluation.station_id or --station-id for {path}; available stations: {available}.')
        station_id = available[0]
    if not isinstance(station_id, str) or station_id not in available:
        raise ValueError(f'Station ID {station_id!r} is unavailable in {path}; available stations: {available}.')
    return data.assign_coords(station=available).sel(station=station_id, drop=True), station_id


def _observation_note(days_of_month=None, start_date=None, end_date=None):
    check_time_selection(dict(start_date=start_date, end_date=end_date, days_of_month=days_of_month))
    note = (f"Observation days {days_of_month['first']}-{days_of_month['last']} of each month (inclusive)."
            if days_of_month is not None else 'All observation days.')
    if start_date is not None or end_date is not None:
        note += f" Date range: {start_date or 'first available'} through {end_date or 'last available'} (inclusive)."
    return note


def _evaluation_inputs(project_dir, site, retrieval, data_source=None, days_of_month=None, station_id=None, dataset=None, observation_height_range=None, test_height_range=None, start_date=None, end_date=None):
    from openMWR.config import check_name
    from openMWR.recipes import RetrievalRecipe

    check_name(site, 'site')
    check_name(retrieval, 'retrieval')
    root = project_root(project_dir)
    path = checkpoint_file(root, site, retrieval)
    if not path.is_file():
        raise FileNotFoundError(f'Missing trained checkpoint: {path}. Train this retrieval before evaluation.')
    model = load_model(retrieval, site, root)
    recipe_file = retrieval_dir(root, site, retrieval) / 'retrieval.toml'
    recipe = RetrievalRecipe.load(root, site, retrieval) if recipe_file.is_file() else None
    settings = dict(recipe.section('evaluation')) if recipe is not None else {}
    if recipe is not None and not settings:
        raise ValueError('Configure an explicit [evaluation] section before evaluating this retrieval.')
    for key, value in (('data_source', data_source), ('days_of_month', days_of_month), ('station_id', station_id), ('dataset', dataset), ('observation_height_range', observation_height_range), ('test_height_range', test_height_range), ('start_date', start_date), ('end_date', end_date)):
        if value is not None:
            settings[key] = value
    source = settings.get('data_source', 'radiosonde')
    if source not in ('radiosonde', 'era5'):
        raise ValueError("Evaluation data_source must be 'radiosonde' or 'era5'.")
    observed = settings.get('observations', True)
    test = settings.get('test_data', True)
    if not observed and not test:
        raise ValueError('Enable evaluation.observations or evaluation.test_data.')
    if observed:
        check_time_selection(settings, 'evaluation')
    if 'height' in model.output.dims:
        for enabled, key in ((observed, 'observation_height_range'), (test, 'test_height_range')):
            if enabled:
                if key not in settings:
                    raise ValueError(f'Specify evaluation.{key} explicitly ([minimum, maximum] in metres or "model").')
                check_height_range(settings[key], f'evaluation.{key}', allow_model=True)
    inputs = []
    paths = []
    if observed:
        analysis_path = dataset_file(root, site, settings.get('dataset', f'analysis_{source}'), bls=model.is_bls_model)
        if not analysis_path.is_file():
            raise FileNotFoundError(f'Missing evaluation analysis dataset: {analysis_path}. Prepare it before evaluation.')
        data = load_analysis_dataset(root, site, settings.get('dataset', f'analysis_{source}'), bls=model.is_bls_model)
        data = select_time(data, **{key: settings[key] for key in TIME_SELECTION_KEYS if key in settings})
        stations = []
        if source == 'radiosonde':
            data, selected_station = _select_station(data, settings.get('station_id'), analysis_path)
            settings['station_id'] = selected_station
            stations = [selected_station]
        else:
            settings.pop('station_id', None)
        reference = data.sel(data_source=source, drop=True).assign_attrs(analysis_stations=stations)
        measurements = data.sel(data_source='hatpro', drop=True)
        variables = ['T', 'rh'] if 'T' in model.output else ['TB']
        inputs.append(('observations', measurements, reference, variables))
        paths.append(analysis_path)
    if test:
        # Saved training inputs describe the trained model, even after recipe edits.
        sources = model.hyper_params.get('sources', model.hyper_params.get('station_ids'))
        if not sources:
            raise ValueError('The checkpoint does not record training sources; set evaluation.test_data = false to evaluate observations only.')
        for source_id in sources:
            path = dataset_file(root, site, model.training_dataset, source_id, split='test')
            if not path.is_file():
                raise FileNotFoundError(f'Missing test dataset: {path}. Prepare it before evaluation.')
            paths.append(path)
        data = load_combined_dataset(site, root, 'test', sources, dataset=model.training_dataset)
        variables = [v for v in ('T', 'rh', 'lwc', 'lwp') if v in model.output] or ['TB']
        inputs.append(('test', data, data, variables))
    prepared = []
    for label, data, reference, variables in inputs:
        if not model.is_bls_model and 'ang' in data.dims:
            data = data.sel(ang=90, drop=True)
        validate_model_data(model.input, data, f'{label} inputs')
        selected = model.output[variables]
        height_range = settings.get('observation_height_range' if label == 'observations' else 'test_height_range', 'model')
        selected = select_height_range(selected, height_range)
        reference = select_height_range(reference, height_range)
        if not model.is_bls_model and 'ang' in reference.dims:
            reference = reference.sel(ang=90, drop=True)
        validate_model_data(selected, reference, f'{label} reference')
        prepared.append((label, data, reference, variables))
    return root, model, settings, prepared, paths


def validate_evaluation(project_dir, site, retrieval, device='auto', **options):
    """Check saved-model evaluation inputs without prediction or writing artifacts."""
    resolve_device(device)
    root, model, settings, inputs, paths = _evaluation_inputs(project_dir, site, retrieval, **options)
    return [str(path) for path in paths]


def evaluate_retrieval(project_dir, site, retrieval, device='auto', **options):
    """Evaluate one saved retrieval, returning numerical results for Python users.

    Recipe evaluation options can be overridden by data_source, start_date,
    end_date, days_of_month and station_id. Time filters select observations;
    the test split already owns its period. Saving a report is a separate operation.
    """
    with retrieval_log(project_dir, site, retrieval, 'evaluate'):
        device = resolve_device(device)
        started = utc_now()
        root, model, settings, inputs, paths = _evaluation_inputs(project_dir, site, retrieval, **options)
        logger.info('Evaluation reference: %s; radiosonde station: %s.', settings.get('data_source', 'radiosonde'), settings.get('station_id', 'not applicable'))
        if any(label == 'observations' for label, *_ in inputs):
            logger.info('Observation selection: %s', _observation_note(**{key: settings[key] for key in TIME_SELECTION_KEYS if key in settings}))
        logger.info('Test data sources from checkpoint: %s.', model.hyper_params.get('sources', model.hyper_params.get('station_ids', [])))
        evaluations = []
        for label, data, reference, variables in inputs:
            pred = run_model_list([model], data, device)
            height_range = settings.get('observation_height_range' if label == 'observations' else 'test_height_range', 'model')
            pred = select_height_range(pred, height_range)
            evaluation = _evaluate(label, f'{"Observations" if label == "observations" else "Test data"}', pred, reference, variables,
                                   ('RMSE', 'MAE', 'ME') if label == 'observations' else ('RMSE', 'ME'))
            if label == 'observations':
                evaluation.note = _observation_note(**{key: settings[key] for key in TIME_SELECTION_KEYS if key in settings})
                setup = (model.training_metadata or {}).get('setup', {})
                optimization = setup.get('optimization')
                if optimization:
                    selection = {key: optimization[key] for key in TIME_SELECTION_KEYS if key in optimization}
                    # Older saved checkpoints recorded the two bounds as a list.
                    if isinstance(selection.get('days_of_month'), list):
                        selection['days_of_month'] = dict(zip(('first', 'last'), selection['days_of_month']))
                    overlap = select_time(data, **selection).time.size > 0
                    evaluation.note += f" This checkpoint was optimized against {optimization['data_source']}, using {optimization['score']}. Optimization selection: {_observation_note(**selection)}"
                    evaluation.note += ' The selected evaluation dates overlap the optimization selection.' if overlap else ' The selected evaluation dates are outside the optimization selection.'
            if 'lwp' in variables:
                evaluation.metrics['lwp_scatter'] = xr.Dataset({'predicted_lwp': pred.lwp, 'reference_lwp': reference.lwp})
            evaluations.append(evaluation)
        metadata = {'site': site, 'retrieval': retrieval, 'project_dir': str(root),
                    'data_source': settings.get('data_source', 'radiosonde'),
                    'started_utc': started, 'finished_utc': utc_now(), 'device': device,
                    'device_name': device_name(device),
                    'filters': {'observation_days_of_month': settings.get('days_of_month'),
                                'observation_start_date': settings.get('start_date'),
                                'observation_end_date': settings.get('end_date'),
                                'radiosonde_station_id': settings.get('station_id'),
                                'observation_height_range': settings.get('observation_height_range'),
                                'test_height_range': settings.get('test_height_range')},
                    'test_station_ids': model.hyper_params.get('sources', model.hyper_params.get('station_ids', [])),
                    'inputs': [str(path.relative_to(root)) for path in paths],
                    'code': software_identity(),
                    'models': [{'name': model.name, 'mode': 'Boundary layer scan' if model.is_bls_model else 'Zenith',
                                'checkpoint_sha256': checkpoint_id(checkpoint_file(root, site, retrieval)),
                                'training_metadata': model.training_metadata}]}
        return Analysis(metadata, evaluations)
def _load_observations(root, site, source, settings, *, bls=False):
    dataset = settings.analysis_dataset or f'analysis_{source}'
    path = dataset_file(root, site, dataset, bls=bls)
    ds = load_analysis_dataset(root, site, dataset, bls=bls)
    stations = []
    ds = select_height_range(select_time(ds, start_date=settings.start_date, end_date=settings.end_date,
                                        days_of_month=settings.days_of_month), settings.observation_height_range)
    if source == "radiosonde":
        ds, station = _select_station(ds, settings.station_id, path)
        stations = [station]
    if not ds.sizes.get("time", 0):
        raise ValueError(f"No observations remain after filtering {path}.")
    reference = ds.sel(data_source=source, drop=True).assign_attrs(
        analysis_stations=stations
    )
    return ds.sel(data_source="hatpro", drop=True), reference


def _predict(models, ds, device, height_range):
    # The notebook passed the site string as the fourth argument, making this
    # flag truthy. State the equivalent behavior explicitly.
    pred = run_model_list(models, ds, device, add_not_predicted_vars=True)
    return calc_derived_quantities(select_height_range(pred, height_range))


def _add_relative_rmse(evaluation):
    profiles = evaluation.metrics["error_profiles"]
    rpg = profiles.sel(model="RPG")
    relative = ((profiles - rpg) / rpg * 100).sel(error="RMSE", drop=True)
    evaluation.metrics["relative_rmse_profiles"] = relative
    evaluation.metrics["relative_rmse"] = relative.mean("height")


def _add_comparators(evaluation, zenith, models):
    for key in ("errors", "error_profiles", "paired_profiles"):
        evaluation.metrics[key] = xr.concat(
            [zenith.metrics[key].sel(model=models), evaluation.metrics[key]],
            dim="model",
            join="outer",
        )


def analyze_models(site, settings, data_source="radiosonde", project_dir=".", device="auto"):
    """Run the four notebook evaluations without displaying or writing output."""
    device = resolve_device(device)
    if data_source not in ("radiosonde", "era5"):
        raise ValueError("data_source must be 'radiosonde' or 'era5'.")
    check_height_range(settings.observation_height_range, allow_model=True)
    check_height_range(settings.test_height_range, allow_model=True)
    project_dir = project_root(project_dir)
    started = datetime.now(timezone.utc).isoformat()
    models = [Model.load(name, site, str(project_dir)) for name in settings.models]
    bls_models = [Model.load(name, site, str(project_dir)) for name in settings.bls_models]
    station_ids = models[0].hyper_params["station_ids"]
    paths = [
        dataset_file(project_dir, site, settings.analysis_dataset or f'analysis_{data_source}', bls=bls)
        for bls in (False, True)
    ]
    paths += [
        dataset_file(project_dir, site, settings.training_dataset, station, split='test') for station in station_ids
    ]
    for path in paths:
        if not path.is_file():
            if path.parent.name == "training":
                raise FileNotFoundError(
                    f"Missing test dataset: {path}. The report needs the existing "
                    "test-data split for the saved retrievals."
                )
            raise FileNotFoundError(
                f"Missing analysis input: {path}. Run bin/update.py to prepare data."
            )
    metadata = {
        "site": site,
        "data_source": data_source,
        "project_dir": str(project_dir),
        "started_utc": started,
        "device": device,
        "device_name": device_name(device),
        "main_model": settings.main_model,
        "filters": {
            "observation_days_of_month": settings.days_of_month,
            "observation_start_date": settings.start_date,
            "observation_end_date": settings.end_date,
            "radiosonde_station_id": settings.station_id if data_source == "radiosonde" else None,
            "observation_height_range": settings.observation_height_range,
            "test_height_range": settings.test_height_range,
        },
        "test_station_ids": station_ids,
        "inputs": [str(path.relative_to(project_dir)) for path in paths],
        "code": {
            "project": git_identity(project_dir),
            "software": software_identity(),
        },
        "models": [
            dict(
                name=model.name,
                mode=mode,
                checkpoint_sha256=checkpoint_id(
                    checkpoint_file(project_dir, site, model.name)
                ),
            )
            for mode, group in (("Zenith", models), ("Boundary layer scan", bls_models))
            for model in group
        ],
    }
    variables = ["T", "rh"]
    test_variables = ["T", "rh", "lwc", "lwp"]

    logger.info("Evaluating zenith retrievals on %s observations", data_source)
    hatpro, reference = _load_observations(project_dir, site, data_source, settings)
    if data_source == 'radiosonde':
        metadata['filters']['radiosonde_station_id'] = reference.attrs['analysis_stations'][0]
    logger.info('Evaluation reference: %s; radiosonde station: %s; %s', data_source, metadata['filters']['radiosonde_station_id'], _observation_note(settings.days_of_month, settings.start_date, settings.end_date))
    pred = _predict(models, hatpro, device, settings.observation_height_range)
    if settings.include_rpg:
        rpg = hatpro[["T", "rh", "ah", "TD", "iwv", "lwc", "lwp", "p"]].expand_dims(
            model=["RPG"]
        )
        pred = xr.concat([rpg, pred], dim="model", data_vars="minimal")
    observed = _evaluate(
        "zenith_observations",
        "Zenith retrieval · observations",
        pred,
        reference,
        variables,
        ("RMSE", "MAE", "ME"),
    )
    observed.note = _observation_note(settings.days_of_month, settings.start_date, settings.end_date)
    observed.metrics["mean_lwp"] = pred[["lwp"]].mean("time")
    observed.metrics["mame"] = np.abs(
        observed.metrics["error_profiles"].sel(error="ME", drop=True)
    ).mean("height")
    time_rmse = cal_RMSE(pred[variables], reference[variables], dims_to_mean="height")
    observed.metrics["rmse_time"] = time_rmse
    observed.metrics["monthly_temperature_rmse"] = (
        time_rmse.sel(model=settings.main_model, drop=True)[["T"]]
        .resample(time="1MS")
        .mean()
    )
    if settings.include_rpg:
        _add_relative_rmse(observed)
    del pred, hatpro, reference

    logger.info("Evaluating zenith retrievals on test data")
    test_data = load_combined_dataset(site, str(project_dir), 'test', station_ids, dataset=settings.training_dataset)
    if not test_data.sizes.get("time", 0):
        raise ValueError("The test data contain no time samples.")
    pred_test = _predict(models, test_data, device, settings.test_height_range)
    test = _evaluate(
        "zenith_test",
        "Zenith retrieval · test data",
        pred_test,
        select_height_range(test_data, settings.test_height_range),
        test_variables,
        ("RMSE", "ME"),
    )
    test.metrics["lwp_scatter"] = xr.Dataset(
        {"predicted_lwp": pred_test.lwp, "reference_lwp": test_data.lwp}
    )
    del pred_test

    logger.info("Evaluating boundary layer scans on %s observations", data_source)
    hatpro, reference = _load_observations(project_dir, site, data_source, settings, bls=True)
    pred = _predict(bls_models, hatpro, device, settings.observation_height_range)
    if settings.include_rpg:
        rpg = hatpro[["T", "p"]].expand_dims(model=["RPG_BLS"])
        pred = xr.concat([rpg, pred], dim="model", data_vars="minimal")
    bls = _evaluate(
        "bls_observations",
        "Boundary layer scan · observations",
        pred,
        reference,
        variables,
        ("RMSE", "ME"),
    )
    comparisons = (
        ["RPG", settings.main_model] if settings.include_rpg else [settings.main_model]
    )
    _add_comparators(bls, observed, comparisons)
    bls.note = (
        _observation_note(settings.days_of_month, settings.start_date, settings.end_date) + " Zenith comparator metrics come from the zenith observation section. "
        "The zenith and scan evaluations retain their own observation times; "
        "they are not restricted to common timestamps."
    )
    if settings.include_rpg:
        _add_relative_rmse(bls)
    del pred, hatpro, reference

    logger.info("Evaluating boundary layer scans on test data")
    pred_test = _predict(bls_models, test_data, device, settings.test_height_range)
    bls_test = _evaluate(
        "bls_test",
        "Boundary layer scan · test data",
        pred_test,
        select_height_range(test_data, settings.test_height_range),
        test_variables,
        ("RMSE", "ME"),
    )
    # Keep the notebook's order: scan retrieval, then the main zenith retrieval.
    for key in ("errors", "error_profiles", "paired_profiles"):
        bls_test.metrics[key] = xr.concat(
            [bls_test.metrics[key], test.metrics[key].sel(model=[settings.main_model])],
            dim="model",
            join="outer",
        )
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return Analysis(metadata, [observed, test, bls, bls_test])


def _table(ds, labels=VARIABLE_LABELS):
    frame = ds.to_dataframe()[list(ds.data_vars)].rename(columns=labels)
    return frame.to_html(
        float_format=lambda value: f"{value:.4g}", na_rep="—", border=0
    )


def _profile_figure(ds, metric, colors):
    if not {'T', 'rh'} <= set(ds.data_vars):
        figure = Figure(figsize=(10, 5))
        ax = figure.subplots()
        for variable, values in ds.items():
            for model in values.model.values:
                selected = values.sel(model=model)
                ax.plot(selected.frq, selected, label=str(model), color=colors[str(model)])
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel(f'{metric} of brightness temperature [K]')
        ax.legend()
        ax.grid()
        return figure
    figure = Figure(figsize=(12, 8))
    labels = {}
    if metric == "Relative RMSE":
        labels = {
            "xlabel_T": "Relative RMSE of temperature [%]",
            "xlabel_RH": "Relative RMSE of relative humidity [%]",
        }
    plot_err(ds, err=metric, colors=colors, figure=figure, show=False, **labels)
    return figure


def _scatter_figure(ds, model):
    figure = Figure(figsize=(10, 5))
    predicted = ds[["predicted_lwp"]].rename(predicted_lwp="lwp").sel(model=model)
    reference = ds[["reference_lwp"]].rename(reference_lwp="lwp")
    plot_scatter_lwp(predicted, reference, figure=figure, show=False)
    return figure


def _figure_html(figure, caption, fmt="svg"):
    stream = BytesIO()
    figure.savefig(stream, format=fmt, dpi=150, bbox_inches="tight")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    mime = "image/svg+xml" if fmt == "svg" else "image/png"
    return (
        f'<figure><img alt="{escape(caption, quote=True)}" src="data:{mime};base64,{encoded}">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def _samples_html(samples):
    stations = ", ".join(map(str, samples["stations"])) or "Single site"
    start, end = [
        np.datetime_as_string(np.datetime64(samples[key]), unit="m").replace("T", " ")
        for key in ("start", "end")
    ]
    return (
        f'<p class="samples">Period: {escape(start)} to {escape(end)}. '
        f"Time coordinates: {samples['time_coordinates']}; profiles with finite reference values: "
        f"{samples['reference_temperature_profiles']}. Stations: {escape(stations)}.</p>"
    )


def render_report(analysis):
    """Build the complete HTML in memory, including its figures."""
    meta = analysis.metadata
    # Use the same model colors throughout comparisons and individual sections.
    model_names = list(
        dict.fromkeys(
            str(model)
            for evaluation in analysis.evaluations
            for model in evaluation.metrics["errors"].model.values
        )
    )
    palette = colormaps["tab10"].colors
    colors = {name: palette[i % len(palette)] for i, name in enumerate(model_names)}
    toc = [(evaluation.key, evaluation.title) for evaluation in analysis.evaluations]
    toc += [(f"retrieval-{i}", model["name"]) for i, model in enumerate(meta["models"])]
    parts = [
        '<nav aria-label="Contents"><h2>Contents</h2><ul>'
        + "".join(
            f'<li><a href="#{escape(key, quote=True)}">{escape(title)}</a></li>'
            for key, title in toc
        )
        + '<li><a href="#provenance">Run details</a></li></ul></nav>'
    ]
    parts.append(
        "<p>ME is prediction minus reference. MAME is the mean absolute vertical "
        "mean error. Summary errors for profile variables aggregate over time, stations "
        "(where present) and height; vertical error profiles aggregate over time and "
        "stations. Missing values are excluded.</p>"
    )
    for evaluation in analysis.evaluations:
        parts.append(
            f'<section id="{evaluation.key}"><h2>{escape(evaluation.title)}</h2>'
        )
        parts.append(_samples_html(evaluation.samples))
        if evaluation.note:
            parts.append(f"<p>{escape(evaluation.note)}</p>")
        parts.append(
            "<details><summary>Paired profiles per retrieval and variable</summary>"
            "<p>A profile counts when prediction and reference have at least one finite paired "
            "height. LWP counts finite paired column values. Missing results appear as —.</p>"
            + _table(evaluation.metrics["paired_profiles"], labels={})
            + "</details>"
        )
        if "mean_lwp" in evaluation.metrics:
            parts.append(
                "<h3>Mean retrieved LWP</h3>" + _table(evaluation.metrics["mean_lwp"])
            )
        for metric in evaluation.error_types:
            parts.append(
                f"<h3>{metric}</h3>"
                + _table(evaluation.metrics["errors"].sel(error=metric, drop=True))
            )
            profile_errors = evaluation.metrics["error_profiles"]
            if not profile_errors.data_vars:
                profile_errors = evaluation.metrics['errors']
            profiles = profile_errors.sel(error=metric, drop=True)
            parts.append(
                _figure_html(
                    _profile_figure(profiles, metric, colors),
                    f"{evaluation.title}: {metric}",
                )
            )
            if metric == "ME" and "mame" in evaluation.metrics:
                parts.append("<h4>MAME</h4>" + _table(evaluation.metrics["mame"]))
        if "relative_rmse" in evaluation.metrics:
            parts.append(
                "<h3>Relative RMSE to RPG [%]</h3><p>Differences are calculated at each height, "
                "then averaged over height. Negative values indicate a smaller RMSE than RPG. "
                "A zero RPG RMSE gives an undefined relative difference.</p>"
                + _table(evaluation.metrics["relative_rmse"], labels={})
            )
            parts.append(
                _figure_html(
                    _profile_figure(
                        evaluation.metrics["relative_rmse_profiles"],
                        "Relative RMSE",
                        colors,
                    ),
                    "Relative RMSE to RPG [%]",
                )
            )
        if "monthly_temperature_rmse" in evaluation.metrics:
            monthly = evaluation.metrics["monthly_temperature_rmse"]
            fig = Figure(figsize=(10, 4))
            ax = fig.subplots()
            ax.plot(
                monthly.time.values, monthly.T.values, color=colors[meta["main_model"]]
            )
            ax.set_ylabel("Temperature RMSE [K]")
            ax.set_xlabel("Month")
            ax.grid()
            fig.autofmt_xdate()
            parts.append(
                "<h3>Monthly temperature RMSE</h3>"
                + _figure_html(
                    fig,
                    meta["main_model"] + ": mean of per-profile RMSE",
                )
            )
        parts.append("</section>")

    parts.append("<h2>Individual retrievals</h2>")
    scatter = next((e.metrics['lwp_scatter'] for e in analysis.evaluations if 'lwp_scatter' in e.metrics), None)
    for i, model in enumerate(meta["models"]):
        name = model["name"]
        parts.append(
            f'<section id="retrieval-{i}"><h3>{escape(name)}</h3><p>{escape(model["mode"])} retrieval.</p>'
        )
        frames = {}
        for evaluation in analysis.evaluations:
            errors = evaluation.metrics["errors"]
            if name in errors.model.values:
                ds = errors.sel(model=name, drop=True).sel(
                    error=list(evaluation.error_types)
                )
                frames[evaluation.title] = ds.to_dataframe()[list(ds.data_vars)]
        frame = pd.concat(frames, names=["Evaluation"]).rename(columns=VARIABLE_LABELS)
        parts.append(
            frame.to_html(
                float_format=lambda value: f"{value:.4g}", na_rep="—", border=0
            )
        )
        if scatter is not None and name in scatter.model.values:
            parts.append(
                _figure_html(
                    _scatter_figure(scatter, name),
                    "Test data: full range and LWP zoom",
                    fmt="png",
                )
            )
        details = json.dumps(model, indent=2, ensure_ascii=False)
        parts.append(
            f"<details><summary>Checkpoint identifier</summary><pre>{escape(details)}</pre></details></section>"
        )

    provenance = json.dumps(meta, indent=2, ensure_ascii=False)
    parts.append(
        f'<section id="provenance"><h2>Run details</h2><details><summary>Inputs, filters, checkpoints and code versions</summary>'
        f"<pre>{escape(provenance)}</pre></details></section>"
    )
    filters = meta["filters"]
    days = filters['observation_days_of_month']
    filter_text = (_observation_note(days, filters.get('observation_start_date'), filters.get('observation_end_date'))
                   if any(evaluation.key == 'observations' or evaluation.key == 'zenith_observations' for evaluation in analysis.evaluations)
                   else 'No observation evaluation')
    for label, key in (('Observation', 'observation_height_range'), ('Test', 'test_height_range')):
        heights = filters.get(key)
        if heights is not None:
            selection = 'full model grid' if heights == 'model' else f'{heights[0]}-{heights[1]} m'
            filter_text += f' {label} profile heights: {selection}.'
    title = f"Retrieval analysis · {meta['site']}"
    template = Template(Path(__file__).with_name("evaluation.html").read_text(encoding="utf-8"))
    return template.substitute(
        title=escape(title),
        source=escape(meta["data_source"]),
        started=escape(meta["started_utc"]),
        filters=filter_text,
        stations=escape(", ".join(map(str, meta["test_station_ids"]))),
        prediction_scope=('Observation profiles: 0–10 km above ground.'
                          if any('T' in e.metrics['errors'] for e in analysis.evaluations)
                          else 'Spectral brightness-temperature predictions.'),
        content="".join(parts),
    )


def write_report(analysis, output_path=None):
    """Write one HTML file; never overwrite an earlier report."""
    meta = analysis.metadata
    if output_path is None:
        stamp = datetime.fromisoformat(meta["started_utc"]).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        output_path = (
            (retrieval_dir(meta['project_dir'], meta['site'], meta['retrieval']) / 'evaluation'
             if 'retrieval' in meta else site_root(meta["project_dir"], meta["site"]) / "analysis" / "reports")
            / f"{stamp}_{meta['data_source']}.html"
        )
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Report already exists: {output_path}. Choose a new filename."
        )
    html = render_report(analysis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(html)
    logger.info("Report saved to %s", output_path)
    return output_path
