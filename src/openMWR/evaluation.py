"""Named evaluation recipes, common-sample comparisons and portable results."""

import base64
from dataclasses import dataclass
from datetime import datetime
from html import escape
from io import BytesIO
import json
import logging
from pathlib import Path
import shutil
from string import Template
import tempfile
import tomllib

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib import colormaps

from openMWR.config import (check_keys, check_name, check_strings, require_keys,
                            check_time_selection, select_time, TIME_SELECTION_KEYS,
                            check_height_range, select_height_range)
from openMWR.calc_errors import cal_RMSE, calculate_errors
from openMWR.devices import resolve_device, device_name
from openMWR.models import load_model
from openMWR.observations import load_analysis_dataset
from openMWR.paths import (project_root, site_config_file, dataset_file, checkpoint_file,
                           dataset_recipe_file, evaluation_dir, evaluation_recipe_file)
from openMWR.plot import plot_scatter_lwp
from openMWR.predict import run_model_list, validate_model_data
from openMWR.provenance import checkpoint_id, plain_value, software_identity, task_log, utc_now
from openMWR.xr_utils import load_combined_dataset

logger = logging.getLogger(__name__)
VARIABLE_LABELS = {
    'T': 'Temperature [K]', 'rh': 'Relative humidity [%]',
    'lwc': 'Liquid water content [g/m³]', 'lwp': 'Liquid water path [g/m²]',
    'iwv': 'Integrated water vapour [kg/m²]', 'TB': 'Brightness temperature [K]',
}


@dataclass
class EvaluationRecipe:
    """Editable comparison settings. Both the command and Python use this object."""

    project_dir: Path
    site: str
    name: str
    values: dict

    @classmethod
    def load(cls, project_dir, site, name):
        check_name(site, 'site')
        check_name(name, 'evaluation')
        root = project_root(project_dir)
        if not site_config_file(root, site).is_file():
            raise FileNotFoundError(f'Missing site configuration: {site_config_file(root, site)}')
        path = evaluation_recipe_file(root, site, name)
        with path.open('rb') as stream:
            recipe = cls(root, site, name, tomllib.load(stream))
        recipe.validate()
        return recipe

    def validate(self):
        check_name(self.site, 'site')
        check_name(self.name, 'evaluation')
        check_keys(self.values, ('schema_version', 'retrievals', 'observations', 'test'), 'evaluation recipe')
        if type(self.values.get('schema_version')) is not int or self.values['schema_version'] != 1:
            raise ValueError('Evaluation schema_version must be 1.')
        check_strings(self.values.get('retrievals'), 'retrievals')
        if not self.values['retrievals']:
            raise ValueError('Select at least one retrieval.')
        for name in self.values['retrievals']:
            check_name(name, 'retrieval')
        if not {'observations', 'test'} & self.values.keys():
            raise ValueError('Configure [observations], [test], or both.')
        for section in ('observations', 'test'):
            if section not in self.values:
                continue
            settings = self.values[section]
            allowed = ('data_source', 'station_id', 'include_rpg', *TIME_SELECTION_KEYS) if section == 'observations' else ('sources',)
            check_keys(settings, ('dataset', 'variables', 'height_range', *allowed), section)
            require_keys(settings, ('dataset', 'variables'), section)
            check_name(settings['dataset'], f'{section}.dataset')
            check_strings(settings['variables'], f'{section}.variables')
            if not settings['variables'] or set(settings['variables']) - VARIABLE_LABELS.keys():
                raise ValueError(f'{section}.variables must select from {list(VARIABLE_LABELS)}.')
            if 'TB' in settings['variables'] and len(settings['variables']) != 1:
                raise ValueError('Evaluate TB separately from profile and column variables.')
            if set(settings['variables']) & {'T', 'rh', 'lwc'}:
                require_keys(settings, ('height_range',), section)
                check_height_range(settings['height_range'], f'{section}.height_range', allow_model=True)
            elif 'height_range' in settings:
                raise ValueError(f'{section}.height_range applies only to profile variables.')
            if section == 'observations':
                require_keys(settings, ('data_source',), section)
                if settings['data_source'] not in ('radiosonde', 'era5'):
                    raise ValueError('observations.data_source must be radiosonde or era5.')
                check_time_selection(settings, section)
                if 'station_id' in settings:
                    check_name(settings['station_id'], 'observations.station_id')
                    if settings['data_source'] != 'radiosonde':
                        raise ValueError('observations.station_id applies only to radiosondes.')
                if 'include_rpg' in settings and type(settings['include_rpg']) is not bool:
                    raise ValueError('observations.include_rpg must be true or false.')
                if settings.get('include_rpg') and 'TB' in settings['variables']:
                    raise ValueError('RPG comparison applies to profile retrievals.')
            else:
                check_strings(settings.get('sources'), 'test.sources')
                if not settings['sources']:
                    raise ValueError('Select at least one test source.')
                for source in settings['sources']:
                    check_name(source, 'test.sources')
        return self


@dataclass
class Evaluation:
    key: str
    title: str
    error_types: tuple[str, ...]
    samples: dict
    metrics: dict[str, xr.Dataset]
    note: str = ''


@dataclass
class Analysis:
    metadata: dict
    evaluations: list[Evaluation]


def _sample_summary(ds):
    variable = 'T' if 'T' in ds else next(iter(ds.data_vars))
    finite = np.isfinite(ds[variable])
    for dim in ('height', 'frq'):
        if dim in finite.dims:
            finite = finite.any(dim)
    return dict(time_coordinates=ds.sizes['time'], reference_profiles=int(finite.sum()),
                stations=ds.station.values.tolist() if 'station' in ds.coords else ds.attrs.get('analysis_stations', []),
                start=str(ds.time.min().values), end=str(ds.time.max().values))


def _paired_profiles(pred, reference, variables):
    counts = {}
    for var in variables:
        valid = np.isfinite(pred[var]) & np.isfinite(reference[var])
        for dim in ('height', 'frq'):
            if dim in valid.dims:
                valid = valid.any(dim)
        counts[var] = valid.sum([dim for dim in ('time', 'station') if dim in valid.dims])
        counts[var].attrs = {'units': '1'}
    return xr.Dataset(counts)


def _select_station(data, station_id, path):
    if 'station' not in data.dims:
        raise ValueError(f'Missing radiosonde station dimension in {path}.')
    available = [str(value) for value in data.station.values]
    if len(set(available)) != len(available):
        raise ValueError(f'Duplicate station IDs in {path}.')
    if station_id is None:
        if len(available) != 1:
            raise ValueError(f'Select observations.station_id. Available stations in {path}: {available}.')
        station_id = available[0]
    if station_id not in available:
        raise ValueError(f'Station ID {station_id!r} is unavailable in {path}. Available stations: {available}.')
    selected = data.assign_coords(station=available).sel(station=station_id, drop=True)
    selected.attrs['analysis_stations'] = [station_id]
    return selected


def _observation_note(settings):
    days = settings.get('days_of_month')
    note = (f"Observation days {days['first']}-{days['last']} of each month (inclusive)."
            if days else 'All observation days.')
    if 'start_date' in settings or 'end_date' in settings:
        note += f" Date range: {settings.get('start_date', 'first available')} through {settings.get('end_date', 'last available')} (inclusive)."
    return note


def _reference_grid(models, reference, variables, settings):
    """Select one explicit grid. Different model grids require another height range."""
    grid = None
    for model in models:
        missing = set(variables) - set(model.output)
        if missing:
            raise ValueError(f'{model.name} does not predict requested variables {sorted(missing)}.')
        output = select_height_range(model.output[variables], settings.get('height_range', 'model'))
        coordinates = {name: value for name, value in output.coords.items() if name in output.dims}
        if grid is None:
            grid = coordinates
        elif grid.keys() != coordinates.keys() or any(not np.array_equal(grid[key], coordinates[key]) for key in grid):
            raise ValueError('Retrieval output grids differ. Choose a shared height range or evaluate the grids separately.')
    reference = select_height_range(reference[variables], settings.get('height_range', 'model'))
    try:
        reference = reference.sel(grid)
    except KeyError as exc:
        raise ValueError('Reference coordinates do not cover the selected model grid. Prepare matching reference data.') from exc
    return reference.reset_coords(drop=True)


def _evaluate_case(recipe, section, settings, models, device, inputs, notes):
    variables = settings['variables']
    logger.info('%s: dataset %s, variables %s.', section, settings['dataset'], ', '.join(variables))
    predictions, references, supported = [], [], {}
    test_data = None
    if section == 'test':
        for source in settings['sources']:
            path = dataset_file(recipe.project_dir, recipe.site, settings['dataset'], source, split='test')
            inputs[path] = checkpoint_id(path)
        test_data = load_combined_dataset(recipe.site, recipe.project_dir, 'test', settings['sources'], dataset=settings['dataset'])
    for bls in (False, True):
        group = [model for model in models if model.is_bls_model == bls]
        if not group:
            continue
        if section == 'observations':
            path = dataset_file(recipe.project_dir, recipe.site, settings['dataset'], bls=bls)
            inputs[path] = checkpoint_id(path)
            data = load_analysis_dataset(recipe.project_dir, recipe.site, settings['dataset'], bls=bls)
            data = select_time(data, **{key: settings[key] for key in TIME_SELECTION_KEYS if key in settings})
            if settings['data_source'] == 'radiosonde':
                data = _select_station(data, settings.get('station_id'), path)
            measured = data.sel(data_source='hatpro', drop=True)
            reference = data.sel(data_source=settings['data_source'], drop=True)
        else:
            measured = reference = test_data
        if not bls and 'ang' in measured.dims:
            measured = measured.sel(ang=90, drop=True)
        if not bls and 'ang' in reference.dims:
            reference = reference.sel(ang=90, drop=True)
        if not reference.sizes.get('time', 0):
            raise ValueError(f'No {section} samples remain after selection.')
        reference = _reference_grid(models, reference, variables, settings)
        references.append(reference)
        for model in group:
            logger.info('Predicting %s for %s.', model.name, section)
            optimization = (model.training_metadata or {}).get('setup', {}).get('optimization')
            if section == 'observations' and optimization:
                selection = {key: optimization[key] for key in TIME_SELECTION_KEYS if key in optimization}
                if isinstance(selection.get('days_of_month'), list):
                    selection['days_of_month'] = dict(zip(('first', 'last'), selection['days_of_month']))
                overlap = bool(select_time(reference, **selection).sizes['time'])
                notes[model.name].append(
                    f"Optimized against {optimization.get('data_source', 'an unrecorded reference')}. "
                    f"Optimization selection: {_observation_note(selection)} "
                    + ('Selected evaluation dates overlap this selection.' if overlap else 'Selected evaluation dates are outside this selection.'))
            valid = validate_model_data(model.input, measured, f'{section} inputs for {model.name}', require_samples=False)
            if bool(valid.any()):
                pred = run_model_list([model], measured, device, include_input_vars=False)
                pred = pred[variables].sel({dim: reference[dim] for dim in ('height', 'frq') if dim in reference.dims})
            else:
                pred = xr.full_like(reference, np.nan).expand_dims(model=[model.name])
            predictions.append(pred.reset_coords(drop=True))
            supported[model.name] = set(variables)
        if section == 'observations' and settings.get('include_rpg'):
            name = 'RPG_BLS' if bls else 'RPG'
            available = (set(variables) & set(measured)) - ({'rh', 'lwc', 'lwp', 'iwv'} if bls else set())
            if not available:
                raise ValueError(f'No requested RPG outputs in {settings["dataset"]}. Disable observations.include_rpg.')
            pred = measured[sorted(available)].sel({dim: reference[dim] for dim in ('height', 'frq') if dim in reference.dims}).reset_coords(drop=True)
            predictions.append(pred.expand_dims(model=[name]))
            supported[name] = available
    try:
        reference = xr.merge(references, join='outer', compat='no_conflicts')
    except xr.MergeError as exc:
        raise ValueError('Zenith and scan analysis files disagree on reference values. Prepare consistent analysis datasets.') from exc
    pred = xr.concat(predictions, dim='model', join='outer')
    # Retain the full reference cohort, including cases a model could not process.
    pred = pred.reindex({dim: reference[dim] for dim in ('time', 'station') if dim in reference.dims})
    available = _paired_profiles(pred, reference, variables)
    scored = pred.copy()
    truth = reference.copy()
    for variable in variables:
        participating = [name for name, outputs in supported.items() if variable in outputs]
        common = np.isfinite(reference[variable]) & np.isfinite(pred[variable].sel(model=participating)).all('model')
        scored[variable] = pred[variable].where(common)
        truth[variable] = reference[variable].where(common)
    error_types = ('RMSE', 'MAE', 'ME') if section == 'observations' else ('RMSE', 'ME')
    errors, profiles = calculate_errors(scored, truth, error_types=error_types, variables=variables)
    metrics = dict(errors=errors, error_profiles=profiles, available_profiles=available,
                   paired_profiles=_paired_profiles(scored, truth, variables))
    if 'T' in variables:
        metrics['rmse_time'] = cal_RMSE(scored[['T']], truth[['T']], dims_to_mean='height')
        metrics['monthly_temperature_rmse'] = metrics['rmse_time'].resample(time='1MS').mean()
    if 'height' in profiles.dims:
        metrics['mame'] = np.abs(profiles.sel(error='ME', drop=True)).mean('height')
    if 'lwp' in variables:
        metrics['lwp_scatter'] = xr.Dataset({'predicted_lwp': scored.lwp, 'reference_lwp': truth.lwp})
    if section == 'observations' and 'RPG' in pred.model.values and profiles.data_vars:
        rpg = profiles.sel(model='RPG')
        relative = ((profiles - rpg) / rpg.where(rpg != 0) * 100).sel(error='RMSE', drop=True)
        for values in relative.values():
            values.attrs = {'units': '%'}
        metrics.update(relative_rmse_profiles=relative, relative_rmse=relative.mean('height'))
    # Dataset attributes from source files belong in run metadata, not metrics.
    for values in metrics.values():
        values.attrs = {}
    note = (_observation_note(settings) if section == 'observations' else f"Explicit test dataset: {settings['dataset']}. Sources: {', '.join(settings['sources'])}.")
    note += ' Scores use common finite reference/prediction pairs across participating retrievals, separately for each variable and level. No reference interpolation is performed.'
    return Evaluation(section, 'Observations' if section == 'observations' else 'Simulated test data', error_types, _sample_summary(reference), metrics, note)


def _model_notes(model, recipe):
    metadata = model.training_metadata or {}
    setup = metadata.get('setup', {})
    notes = []
    if 'test' in recipe.values:
        notes.append(f"Checkpoint training dataset: {model.training_dataset}. Evaluation test dataset: {recipe.values['test']['dataset']}.")
        if not setup.get('training_selection', model.hyper_params.get('training_selection')):
            notes.append('The checkpoint has no recorded calendar training selection. Independence of these test data is not established.')
    return notes


def evaluate(recipe: EvaluationRecipe, *, retrievals=None, device='auto'):
    """Evaluate a named comparison without training, downloading or saving results."""
    settings = {**recipe.values, 'retrievals': list(retrievals) if retrievals is not None else recipe.values['retrievals']}
    recipe = EvaluationRecipe(project_root(recipe.project_dir), recipe.site, recipe.name, settings).validate()
    with task_log(evaluation_dir(recipe.project_dir, recipe.site, recipe.name) / 'evaluation.log'):
        started = utc_now()
        device = resolve_device(device)
        hashes = {name: checkpoint_id(checkpoint_file(recipe.project_dir, recipe.site, name)) for name in settings['retrievals']}
        models = [load_model(name, recipe.site, recipe.project_dir) for name in settings['retrievals']]
        if len({'TB' in model.output for model in models}) > 1:
            raise ValueError('Evaluate spectral and profile retrievals in separate recipes.')
        configuration = [site_config_file(recipe.project_dir, recipe.site)]
        configuration += [dataset_recipe_file(recipe.project_dir, recipe.site, settings[key]['dataset'])
                          for key in ('observations', 'test') if key in settings]
        inputs = {path: checkpoint_id(path) for path in configuration if path.is_file()}
        notes = {model.name: _model_notes(model, recipe) for model in models}
        logger.info('Evaluating %s with retrievals %s.', recipe.name, ', '.join(settings['retrievals']))
        evaluations = [_evaluate_case(recipe, key, settings[key], models, device, inputs, notes)
                       for key in ('observations', 'test') if key in settings]
        metadata = dict(site=recipe.site, evaluation=recipe.name, project_dir=str(recipe.project_dir),
                        started_utc=started, finished_utc=utc_now(), settings=plain_value(settings),
                        device=device, device_name=device_name(device), code=software_identity(),
                        inputs=[dict(path=str(path.relative_to(recipe.project_dir)), sha256=digest) for path, digest in sorted(inputs.items())],
                        models=[dict(name=model.name, mode='Boundary layer scan' if model.is_bls_model else 'Zenith',
                                     checkpoint_sha256=hashes[model.name],
                                     training_metadata=model.training_metadata, notes=notes[model.name]) for model in models])
        for evaluation in evaluations:
            for name in evaluation.metrics['errors'].model.values:
                if name not in {model['name'] for model in metadata['models']}:
                    metadata['models'].append(dict(name=str(name), mode='RPG product', notes=[]))
        logger.info('Finished evaluation %s.', recipe.name)
        return Analysis(metadata, evaluations)


def save_evaluation(analysis, output_dir=None):
    """Atomically publish report.html and grouped metrics.nc in a new run directory."""
    meta = analysis.metadata
    if output_dir is None:
        stamp = datetime.fromisoformat(meta['started_utc']).strftime('%Y%m%dT%H%M%S.%fZ')
        output_dir = evaluation_dir(meta['project_dir'], meta['site'], meta['evaluation']) / stamp
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Evaluation output already exists: {output_dir}. Choose a new directory.')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.evaluation-', dir=output_dir.parent))
    try:
        sections = [dict(key=e.key, title=e.title, error_types=e.error_types, samples=e.samples,
                         note=e.note, metrics=list(e.metrics)) for e in analysis.evaluations]
        xr.Dataset(attrs={'format_version': 1, 'metadata': json.dumps(plain_value(meta)),
                          'sections': json.dumps(plain_value(sections))}).to_netcdf(temporary / 'metrics.nc')
        for evaluation in analysis.evaluations:
            for name, values in evaluation.metrics.items():
                values.to_netcdf(temporary / 'metrics.nc', mode='a', group=f'{evaluation.key}/{name}')
        (temporary / 'report.html').write_text(render_report(analysis), encoding='utf-8')
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    logger.info('Evaluation saved to %s', output_dir)
    return output_dir


def load_evaluation(directory):
    """Reload numerical results and metadata for Python analysis or report rendering."""
    path = Path(directory) / 'metrics.nc'
    with xr.open_dataset(path) as root:
        if root.attrs.get('format_version') != 1:
            raise ValueError(f'Unsupported evaluation result format: {path}.')
        metadata = json.loads(root.attrs['metadata'])
        sections = json.loads(root.attrs['sections'])
    evaluations = []
    for section in sections:
        metrics = {name: xr.load_dataset(path, group=f'{section["key"]}/{name}') for name in section.pop('metrics')}
        section['error_types'] = tuple(section['error_types'])
        evaluations.append(Evaluation(**section, metrics=metrics))
    return Analysis(metadata, evaluations)


def _table(ds, labels=VARIABLE_LABELS):
    frame = ds.to_dataframe()[list(ds.data_vars)].rename(columns=labels)
    return frame.to_html(
        float_format=lambda value: f"{value:.4g}", na_rep="—", border=0
    )


def _profile_figure(ds, metric, colors):
    variables = [name for name, values in ds.items() if {'height', 'frq'} & set(values.dims)]
    if not variables:
        return None
    figure = Figure(figsize=(6 * len(variables), 5))
    axes = np.atleast_1d(figure.subplots(1, len(variables)))
    for ax, variable in zip(axes, variables):
        values = ds[variable]
        for model in values.model.values:
            selected = values.sel(model=model)
            if 'height' in selected.dims:
                ax.plot(selected, selected.height / 1000, label=str(model), color=colors[str(model)])
            else:
                ax.plot(selected.frq, selected, label=str(model), color=colors[str(model)])
        if 'height' in values.dims:
            ax.set_xlabel(f'{metric} [%]' if metric == 'Relative RMSE' else f'{metric}: {VARIABLE_LABELS.get(variable, variable)}')
            ax.set_ylabel('Height above instrument [km]')
            if metric in ('RMSE', 'MAE'):
                ax.set_xlim(left=0)
        else:
            ax.set_xlabel('Frequency [GHz]')
            ax.set_ylabel(f'{metric}: {VARIABLE_LABELS.get(variable, variable)}')
            if metric in ('RMSE', 'MAE'):
                ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid()
    return figure


def _scatter_figure(ds, model):
    figure = Figure(figsize=(10, 5))
    predicted = ds[["predicted_lwp"]].rename(predicted_lwp="lwp").sel(model=model)
    reference = ds[["reference_lwp"]].rename(reference_lwp="lwp")
    if 'station' not in ds.dims:
        predicted = predicted.expand_dims(station=['site'])
        reference = reference.expand_dims(station=['site'])
    plot_scatter_lwp(predicted, reference, figure=figure, show=False)
    for ax in figure.axes:
        ax.set_xlabel('Reference LWP [g/m²]')
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
        f"Time coordinates: {samples['time_coordinates']}. Reference profiles with finite values: "
        f"{samples['reference_profiles']}. Stations: {escape(stations)}.</p>"
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
        "stations. Scores use the same finite pairs for all participating retrievals in each variable and level. No common pairs means an unavailable score.</p>"
    )
    for evaluation in analysis.evaluations:
        parts.append(
            f'<section id="{evaluation.key}"><h2>{escape(evaluation.title)}</h2>'
        )
        parts.append(_samples_html(evaluation.samples))
        if evaluation.note:
            parts.append(f"<p>{escape(evaluation.note)}</p>")
        parts.append(
            "<details><summary>Coverage and common profiles per retrieval and variable</summary>"
            "<p>A profile counts when prediction and reference have at least one finite paired "
            "height. LWP counts finite paired column values. Missing results appear as —.</p>"
            + "<h4>Available before common-sample selection</h4>"
            + _table(evaluation.metrics["available_profiles"], labels={})
            + "<h4>Used for scores</h4>"
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
            figure = _profile_figure(profiles, metric, colors)
            if figure is not None:
                parts.append(_figure_html(figure, f'{evaluation.title}: {metric}'))
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
            for name in monthly.model.values:
                values = monthly.T.sel(model=name)
                if 'station' in values.dims:
                    values = values.mean('station')
                ax.plot(values.time.values, values.values, label=str(name), color=colors[str(name)])
            ax.legend()
            ax.set_ylabel("Temperature RMSE [K]")
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Month")
            ax.grid()
            fig.autofmt_xdate()
            parts.append(
                "<h3>Monthly temperature RMSE</h3>"
                + _figure_html(
                    fig,
                    "Mean of per-profile temperature RMSE",
                )
            )
        parts.append("</section>")

    parts.append("<h2>Individual retrievals</h2>")
    scatter = next((e for e in analysis.evaluations if 'lwp_scatter' in e.metrics), None)
    for i, model in enumerate(meta["models"]):
        name = model["name"]
        parts.append(
            f'<section id="retrieval-{i}"><h3>{escape(name)}</h3><p>{escape(model["mode"])} retrieval.</p>'
        )
        for note in model.get('notes', []):
            parts.append(f'<p>{escape(note)}</p>')
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
        if scatter is not None and name in scatter.metrics['lwp_scatter'].model.values:
            parts.append(
                _figure_html(
                    _scatter_figure(scatter.metrics['lwp_scatter'], name),
                    f"{scatter.title}: full range and LWP zoom",
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
    observations = meta['settings'].get('observations')
    test = meta['settings'].get('test')
    filter_text = _observation_note(observations) if observations else 'No observation evaluation.'
    for label, settings in (('Observation', observations), ('Test', test)):
        if settings and 'height_range' in settings:
            heights = settings['height_range']
            selection = 'full model grid' if heights == 'model' else f'{heights[0]}-{heights[1]} m'
            filter_text += f' {label} profile heights: {selection}.'
    title = f"Retrieval evaluation · {meta['site']} · {meta['evaluation']}"
    template = Template(Path(__file__).with_name("evaluation.html").read_text(encoding="utf-8"))
    return template.substitute(
        title=escape(title),
        source=escape(observations['data_source'] if observations else 'Simulated test data'),
        started=escape(meta["started_utc"]),
        filters=filter_text,
        stations=escape(', '.join(test['sources']) if test else 'No test evaluation'),
        content="".join(parts),
    )
