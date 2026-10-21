"""Named dataset recipes and optional dependency updates.

The workflow owns identities, settings and ordering. Scientific processing stays
in radiosonde, era5, hatpro_data and train, which are also usable from Python.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path
import shutil
import tomllib
from uuid import uuid4

import numpy as np
import pandas as pd

from openMWR.consts import std_angles, std_freqs, std_heights
from openMWR.paths import (dataset_dir, dataset_file, dataset_recipe_file,
                           era5_cache_file, project_root, radiosonde_station_dir,
                           radiosonde_year_file, site_config_file)
from openMWR.config import check_name, check_coordinates, check_keys, check_number, check_strings

logger = logging.getLogger(__name__)

_INPUTS = {
    'radiosonde': {}, 'era5': {}, 'hatpro': {},
    'profiles': {'source': ('radiosonde', 'era5')},
    'simulation': {'profiles': ('profiles',)},
    'split': {'simulation': ('simulation',)},
    'analysis': {'observations': ('hatpro',), 'reference': ('simulation',)},
}
_OPTIONS = {
    'radiosonde': ('stations', 'start_year', 'end_year', 'refresh_years'),
    'era5': ('start_date', 'end_date', 'hour_interval', 'delay_days'),
    'hatpro': ('start_date', 'end_date', 'observations', 'execution'),
    'profiles': ('stations', 'preparation', 'execution'),
    'simulation': ('stations', 'radiative_transfer', 'execution'),
    'split': ('stations', 'split'),
    'analysis': ('stations',),
}


def _coverage_shrank(previous, current):
    old_range, new_range = previous.get('range', {}), current.get('range', {})
    for key in old_range.keys() & new_range.keys():
        if ((key.startswith('start_') and new_range[key] > old_range[key])
                or (key.startswith('end_') and new_range[key] < old_range[key])):
            return True
    return any(_coverage_shrank(old, current.get('inputs', {}).get(role, {}))
               for role, old in previous.get('inputs', {}).items())


@dataclass
class DatasetRecipe:
    project_dir: Path
    site: str
    name: str
    values: dict

    @classmethod
    def load(cls, project_dir, site, name):
        path = dataset_recipe_file(project_dir, site, name)
        if not path.is_file():
            raise FileNotFoundError(f'Missing dataset recipe: {path}')
        with path.open('rb') as stream:
            recipe = cls(project_root(project_dir), site, name, tomllib.load(stream))
        recipe.validate()
        return recipe

    @property
    def kind(self):
        return self.values['kind']

    @property
    def inputs(self):
        return self.values.get('inputs', {})

    @property
    def directory(self):
        return dataset_dir(self.project_dir, self.site, self.name)

    def validate(self):
        kind = self.values.get('kind')
        if kind not in _INPUTS:
            raise ValueError(f'{self.name}: kind must be one of {", ".join(_INPUTS)}.')
        check_keys(self.values, ('kind', 'inputs', *_OPTIONS[kind]), self.name)
        check_keys(self.inputs, _INPUTS[kind], f'{self.name}.inputs')
        if set(self.inputs) != set(_INPUTS[kind]):
            raise ValueError(f'{self.name}: inputs must name {", ".join(_INPUTS[kind])}.')
        for name in self.inputs.values():
            check_name(name, f'{self.name}.inputs')
        if 'stations' in self.values:
            check_strings(self.values['stations'], f'{self.name}.stations')
            if not self.values['stations']:
                raise ValueError(f'{self.name}.stations must not be empty.')
            for station in self.values['stations']:
                check_name(station, 'station')
        if kind == 'radiosonde' and 'stations' not in self.values:
            raise ValueError(f'{self.name}: radiosonde sources require stations.')
        for key in ('start_year', 'end_year', 'refresh_years', 'hour_interval', 'delay_days'):
            if key in self.values:
                check_number(self.values[key], f'{self.name}.{key}', 0 if key == 'delay_days' else 1, integer=True)
        if kind == 'era5' and self.values.get('hour_interval', 6) not in (1, 2, 3, 4, 6, 8, 12, 24):
            raise ValueError('hour_interval must divide 24.')
        for key in ('start_date', 'end_date'):
            if key in self.values:
                if not isinstance(self.values[key], str):
                    raise ValueError(f'{key} must be a quoted ISO date.')
                date.fromisoformat(self.values[key])
        execution = self.values.get('execution', {})
        check_keys(execution, ('processes',), f'{self.name}.execution')
        if 'processes' in execution:
            check_number(execution['processes'], 'execution.processes', 1, integer=True)
        preparation = self.values.get('preparation', {})
        check_keys(preparation, ('heights', 'cut_off_at_mwr_height', 'scale_rh_clear_sky',
                             'scale_rh_clear_sky_seed', 'scale_rh_clear_sky_fraction',
                             'scale_rh_clear_sky_factor'), f'{self.name}.preparation')
        if 'heights' in preparation:
            check_coordinates(preparation['heights'], 'preparation.heights')
            if preparation['heights'][0] != 0 or preparation['heights'] != sorted(preparation['heights']):
                raise ValueError('preparation.heights must increase from zero (metres above the profile origin).')
        for key in ('cut_off_at_mwr_height', 'scale_rh_clear_sky'):
            if key in preparation and type(preparation[key]) is not bool:
                raise ValueError(f'preparation.{key} must be true or false.')
        if 'scale_rh_clear_sky_seed' in preparation:
            check_number(preparation['scale_rh_clear_sky_seed'], 'scale_rh_clear_sky_seed', 0, integer=True)
        for key in ('scale_rh_clear_sky_fraction', 'scale_rh_clear_sky_factor'):
            if key in preparation:
                check_number(preparation[key], key, 0)
                if preparation[key] <= 0 or (key.endswith('fraction') and preparation[key] > 1):
                    raise ValueError(f'{key} must be positive; fractions cannot exceed one.')
        rt = self.values.get('radiative_transfer', {})
        check_keys(rt, ('backend', 'absorption_model', 'frequencies', 'angles', 'frequency_shift'), f'{self.name}.radiative_transfer')
        if rt.get('backend', 'torchMWRT') not in ('torchMWRT', 'pyrtlib'):
            raise ValueError('radiative_transfer.backend must be torchMWRT or pyrtlib.')
        if 'absorption_model' in rt:
            check_name(rt['absorption_model'], 'radiative_transfer.absorption_model')
        for key in ('frequencies', 'angles'):
            if key in rt:
                check_coordinates(rt[key], f'radiative_transfer.{key}')
        if 'angles' in rt and (min(rt['angles']) <= 0 or max(rt['angles']) > 90):
            raise ValueError('RT angles must be between zero (exclusive) and 90 degrees.')
        if 'frequency_shift' in rt:
            shifts = rt['frequency_shift']
            if not isinstance(shifts, list) or len(shifts) != len(rt.get('frequencies', std_freqs)):
                raise ValueError('RT frequency_shift must contain one offset per frequency.')
            for value in shifts:
                check_number(value, 'radiative_transfer.frequency_shift')
        split = self.values.get('split', {})
        check_keys(split, ('test_fraction', 'validation_fraction', 'seed'), f'{self.name}.split')
        for key in ('test_fraction', 'validation_fraction'):
            if key in split:
                check_number(split[key], f'split.{key}', 0)
                if split[key] >= 1:
                    raise ValueError(f'split.{key} must be smaller than one.')
        if 'seed' in split:
            check_number(split['seed'], 'split.seed', 0, integer=True)
        observations = self.values.get('observations', {})
        check_keys(observations, ('import_retrieval_data',), f'{self.name}.observations')
        if 'import_retrieval_data' in observations and type(observations['import_retrieval_data']) is not bool:
            raise ValueError('observations.import_retrieval_data must be true or false.')
        return self


class DatasetWorkflow:
    """Build selected named datasets, optionally updating their dependencies.

    ``build`` accepts one name or several names, returns their processing order,
    and performs no writes or downloads with ``dry_run=True``. Recipe objects
    remain accessible through ``recipe(name)`` for Python inspection/custom work.
    """

    def __init__(self, project_dir, site):
        check_name(site, 'site')
        self.project_dir, self.site = project_root(project_dir), site
        with site_config_file(self.project_dir, site).open('rb') as stream:
            self.site_settings = tomllib.load(stream)
        self._recipes = {}
        self._settings_cache = None
        self._settings_date = None

    @contextmanager
    def _resolved_settings(self):
        """Share one settings/date snapshot throughout an operation."""
        if self._settings_cache is not None:
            yield
            return
        self._settings_cache = {}
        self._settings_date = date.today()
        try:
            yield
        finally:
            self._settings_cache = None
            self._settings_date = None

    def recipe(self, name):
        if name not in self._recipes:
            self._recipes[name] = DatasetRecipe.load(self.project_dir, self.site, name)
        return self._recipes[name]

    def _graph(self, names):
        ordered, visiting, visited = [], [], set()

        def visit(name):
            if name in visiting:
                raise ValueError('Circular dataset dependency: ' + ' -> '.join([*visiting, name]))
            if name in visited:
                return
            recipe = self.recipe(name).validate()
            visiting.append(name)
            for role, input_name in recipe.inputs.items():
                visit(input_name)
                if self.recipe(input_name).kind not in _INPUTS[recipe.kind][role]:
                    raise ValueError(f'{name}.inputs.{role}: {input_name} has incompatible kind {self.recipe(input_name).kind}.')
            self.settings(name)  # Validate inherited station selections and source-specific options.
            visiting.pop()
            visited.add(name)
            ordered.append(name)

        for name in names:
            visit(name)
        return ordered

    def sources(self, name):
        recipe = self.recipe(name)
        if recipe.kind == 'radiosonde':
            return recipe.values['stations']
        if recipe.kind == 'era5':
            return ['era5']
        if recipe.kind == 'hatpro':
            return []
        role = 'reference' if recipe.kind == 'analysis' else next(iter(recipe.inputs))
        available = self.sources(recipe.inputs[role])
        selected = recipe.values.get('stations', available)
        if set(selected) - set(available):
            raise ValueError(f'{name}: stations must be a subset of its input sources {available}.')
        return selected

    def source_kind(self, name):
        recipe = self.recipe(name)
        if recipe.kind in ('radiosonde', 'era5', 'hatpro'):
            return recipe.kind
        role = 'reference' if recipe.kind == 'analysis' else next(iter(recipe.inputs))
        return self.source_kind(recipe.inputs[role])

    def settings(self, name):
        """Resolve defaults; a build shares its snapshot with processing records."""
        if self._settings_cache is not None and name in self._settings_cache:
            return self._settings_cache[name]
        today = self._settings_date or date.today()
        recipe = self.recipe(name)
        values, kind = recipe.values, recipe.kind
        settings = {'kind': kind, 'inputs': recipe.inputs, 'sources': self.sources(name)}
        if kind == 'radiosonde':
            settings.update(start_year=values.get('start_year', 1990), end_year=values.get('end_year', today.year),
                            refresh_years=values.get('refresh_years', 2))
            if settings['start_year'] > settings['end_year']:
                raise ValueError(f'{name}: start_year is after end_year.')
        elif kind in ('era5', 'hatpro'):
            delay = values.get('delay_days', 5) if kind == 'era5' else 0
            settings.update(start_date=values.get('start_date', self.site_settings.get('mwr_measurement_start_date')),
                            end_date=values.get('end_date', str(today - pd.Timedelta(days=delay))))
            if not settings['start_date']:
                raise ValueError(f'{name}: provide start_date or the site mwr_measurement_start_date.')
            if pd.Timestamp(settings['start_date']) > pd.Timestamp(settings['end_date']):
                raise ValueError(f'{name}: start_date is after end_date.')
            if kind == 'era5':
                settings.update(hour_interval=values.get('hour_interval', 6),
                                latitude=round(self.site_settings['latitude'] * 4) / 4,
                                longitude=round(self.site_settings['longitude'] * 4) / 4)
            else:
                settings.update(import_retrieval_data=values.get('observations', {}).get('import_retrieval_data', self.site_settings.get('rpg_retrieval_exists', False)),
                                hatpro_data_dir=self.site_settings.get('hatpro_data_dir'), gen=self.site_settings.get('gen'))
        elif kind == 'profiles':
            preparation = values.get('preparation', {})
            settings['heights'] = preparation.get('heights', std_heights.tolist())
            settings['mwr_height'] = self.site_settings['mwr_height']
            if self.source_kind(name) == 'radiosonde':
                if values.get('execution', {}).get('processes', 1) != 1:
                    raise ValueError(f'{name}: radiosonde preparation is serial; execution.processes applies to ERA5 profiles.')
                settings.update(cut_off_at_mwr_height=False, scale_rh_clear_sky=False,
                                scale_rh_clear_sky_fraction=0.1, scale_rh_clear_sky_factor=0.2,
                                scale_rh_clear_sky_seed=None)
                settings.update({k: v for k, v in preparation.items() if k != 'heights'})
            elif set(preparation) - {'heights'}:
                raise ValueError(f'{name}: ERA5 profile preparation accepts heights; radiosonde preparation options do not apply.')
        elif kind == 'simulation':
            rt = values.get('radiative_transfer', {})
            settings.update(backend=rt.get('backend', 'torchMWRT'),
                            absorption_model=rt.get('absorption_model', 'R17' if self.source_kind(name) == 'era5' else 'R20'),
                            frequencies=rt.get('frequencies', std_freqs.tolist()),
                            angles=rt.get('angles', std_angles.tolist()),
                            frequency_shift=rt.get('frequency_shift'), gen=self.site_settings.get('gen'))
        elif kind == 'split':
            settings.update(test_fraction=0.1, validation_fraction=0.1, seed=42)
            settings.update(values.get('split', {}))
        if self._settings_cache is not None:
            self._settings_cache[name] = settings
        return settings

    def _definition(self, name):
        """Scientific lineage, excluding expanding source coverage and worker count."""
        settings = self.settings(name)
        if settings['kind'] in ('radiosonde', 'era5', 'hatpro'):
            settings = {k: v for k, v in settings.items() if k not in ('start_year', 'end_year', 'refresh_years', 'start_date', 'end_date')}
        return {'settings': settings, 'inputs': {role: self._definition(input_name)
                for role, input_name in self.recipe(name).inputs.items()}}

    def files(self, name, *, sources=None, splits=None):
        recipe = self.recipe(name)
        if recipe.kind == 'radiosonde':
            settings = self.settings(name)
            return [p for source in self.sources(name)
                    for p in sorted(radiosonde_station_dir(self.project_dir, source).glob('raw_data_*.nc'))
                    if settings['start_year'] <= int(p.stem.rsplit('_', 1)[1]) <= settings['end_year']]
        if recipe.kind == 'era5':
            settings = self.settings(name)
            return [era5_cache_file(self.project_dir, settings['latitude'], settings['longitude'], settings['hour_interval'])]
        if recipe.kind in ('hatpro', 'analysis'):
            return [dataset_file(self.project_dir, self.site, name, bls=bls) for bls in (False, True)]
        if splits is None:
            splits = ('training', 'validation', 'test') if recipe.kind == 'split' else (None,)
        if sources is None:
            sources = self.sources(name)
        return [dataset_file(self.project_dir, self.site, name, source, split=split)
                for source in sources for split in splits]

    def require(self, name, *, sources=None, splits=None):
        """Check a prepared dataset's definition, state and selected files.

        Selecting splits/sources requires a split dataset. Unselected split files
        are not needed, so training need not load or require held-out test data.
        This performs no processing or downloads.
        """
        with self._resolved_settings():
            self._graph([name])
            if sources is not None or splits is not None:
                if self.recipe(name).kind != 'split':
                    raise ValueError(f'{name}: select a split dataset when requesting sources or splits.')
                if sources is not None:
                    check_strings(sources, 'sources')
                    available = self.sources(name)
                    if set(sources) - set(available):
                        raise ValueError(f'sources must be a subset of {available}.')
                if splits is not None:
                    splits = list(splits)
                    check_strings(splits, 'splits')
                    if not splits or set(splits) - {'training', 'validation', 'test'}:
                        raise ValueError('splits must select training, validation or test.')
            self._check_state(name)
            self._require_files(name, sources=sources, splits=splits)

    def _require_files(self, name, *, sources=None, splits=None):
        paths = self.files(name, sources=sources, splits=splits)
        if self.recipe(name).kind == 'radiosonde':
            for source in self.sources(name):
                if not any(p.parent == radiosonde_station_dir(self.project_dir, source) for p in paths):
                    raise FileNotFoundError(f'Missing raw data for station {source}; build dataset {name} first (or use --with-dependencies).')
        if not paths or any(not p.is_file() for p in paths):
            missing = next((p for p in paths if not p.is_file()), self.recipe(name).directory)
            raise FileNotFoundError(f'Missing input dataset {name}: {missing}. Build it first or use --with-dependencies.')

    def _state(self, name):
        path = self.recipe(name).directory / 'state.json'
        return json.loads(path.read_text()) if path.is_file() else {}

    def _check_state(self, name, rebuilding=False):
        if rebuilding:
            return
        state = self._state(name)
        has_output = any(p.is_file() for p in self.files(name))
        if not has_output:
            return
        if self.recipe(name).kind in ('radiosonde', 'era5') and not state:
            return  # A matching shared raw cache may predate this site recipe.
        reason = None
        if not state:
            reason = 'output has no processing record'
        elif state.get('requires_rebuild'):
            reason = state.get('provenance', 'historical processing settings are unknown')
        elif state.get('definition') != self._definition(name):
            reason = 'scientific settings or input definitions changed'
        elif (self.recipe(name).kind not in ('radiosonde', 'era5')
              and _coverage_shrank(state.get('coverage', {}), self._coverage(name))):
            reason = 'requested date coverage narrowed'
        elif (self.recipe(name).kind == 'hatpro'
              and state.get('effective', {}).get('start_date') != self.settings(name)['start_date']):
            reason = 'changing historical HATPRO coverage requires regeneration'
        else:
            for input_name, generation in state.get('input_generations', {}).items():
                if generation != self._state(input_name).get('generation'):
                    reason = f'input {input_name} was rebuilt'
        if reason:
            raise ValueError(f'{name}: {reason}. Rebuild this dataset explicitly with --rebuild, or use a new dataset name.')

    def _input_stats(self, name):
        return {str(p.relative_to(self.project_dir)): [p.stat().st_size, p.stat().st_mtime_ns]
                for input_name in self.recipe(name).inputs.values() for p in self.files(input_name) if p.is_file()}

    def _coverage(self, name):
        settings = self.settings(name)
        return {'range': {k: v for k, v in settings.items() if k in ('start_year', 'end_year', 'start_date', 'end_date')},
                'inputs': {role: self._coverage(input_name) for role, input_name in self.recipe(name).inputs.items()}}

    def record(self, name, *, rebuilt=False, provenance=None, requires_rebuild=False, source_years_checked=None):
        """Record processing settings; migration may explicitly describe inferred provenance."""
        with self._resolved_settings():
            recipe = self.recipe(name)
            old = self._state(name)
            state = {'definition': self._definition(name), 'effective': self.settings(name),
                     'generation': uuid4().hex if rebuilt or not old else old['generation'],
                     'input_generations': {n: self._state(n).get('generation') for n in recipe.inputs.values()},
                     'input_files': self._input_stats(name),
                     'coverage': self._coverage(name),
                     'updated_utc': pd.Timestamp.now(tz='UTC').isoformat()}
            if provenance is not None:
                state['provenance'] = provenance
            elif old.get('provenance'):
                state['provenance'] = old['provenance']
            if requires_rebuild:
                state['requires_rebuild'] = True
            if source_years_checked is not None:
                state['source_years_checked'] = source_years_checked
            recipe.directory.mkdir(parents=True, exist_ok=True)
            path = recipe.directory / 'state.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(path)
            return state

    def build(self, datasets, *, with_dependencies=False, rebuild=False, dry_run=False):
        with self._resolved_settings():
            names = [datasets] if isinstance(datasets, str) else list(datasets)
            if not names:
                raise ValueError('Select at least one dataset.')
            graph = self._graph(names)
            selected = set(names)
            order = graph if with_dependencies else [n for n in graph if n in selected]
            logger.info('Dataset order: %s', ' -> '.join(order))
            # Validate the whole selected operation before deleting any output or downloading.
            for name in order:
                self._check_state(name, rebuilding=rebuild and name in selected)
                if rebuild and name not in selected and any(p.is_file() for p in self.files(name)):
                    ancestors = set(self._graph([name])) - {name}
                    if ancestors & selected:
                        raise ValueError(f'{name}: a selected input will be rebuilt. Select {name} for rebuild too, or rebuild the input separately first.')
                for input_name in self.recipe(name).inputs.values():
                    if input_name not in order:
                        self._require_files(input_name)
                        self._check_state(input_name)
            for name in order:
                rebuilding = rebuild and name in selected
                recipe = self.recipe(name)
                logger.info('%s %s (%s)%s', 'Rebuild' if rebuilding else 'Build/update', name, recipe.kind,
                            ' [download/check source]' if recipe.kind in ('radiosonde', 'era5') else '')
                if dry_run:
                    continue
                for input_name in recipe.inputs.values():
                    self._require_files(input_name)
                    self._check_state(input_name)
                self._check_state(name, rebuilding=rebuilding)
                if rebuilding and recipe.directory.exists():
                    shutil.rmtree(recipe.directory)
                recipe.directory.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(recipe.directory / 'build.log')
                handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
                logging.getLogger().addHandler(handler)
                try:
                    details = self._build(recipe, rebuilding) or {}
                    self._require_files(name)
                    self.record(name, rebuilt=rebuilding, **details)
                finally:
                    logging.getLogger().removeHandler(handler)
                    handler.close()
            return order

    def _build(self, recipe, rebuilding):
        name, kind, settings = recipe.name, recipe.kind, self.settings(recipe.name)
        processes = recipe.values.get('execution', {}).get('processes', 12 if kind == 'simulation' else 1)
        state = self._state(name)
        if (recipe.inputs and state.get('input_files') == self._input_stats(name)
                and (kind != 'profiles' or state.get('coverage') == self._coverage(name))
                and all(p.is_file() for p in self.files(name))):
            logger.info('%s is up to date.', name)
            return
        if kind == 'radiosonde':
            from openMWR.dwd_opendata import update_radiosonde_data
            recent = self._settings_date.year - settings['refresh_years'] + 1
            checked = set(self._state(name).get('source_years_checked', []))
            for station in self.sources(name):
                for year in range(settings['start_year'], settings['end_year'] + 1):
                    path = radiosonde_year_file(self.project_dir, station, year)
                    key = f'{station}/{year}'
                    if rebuilding or (not path.is_file() and key not in checked) or year >= recent:
                        update_radiosonde_data(station, year, self.project_dir)
                    checked.add(key)
            return {'source_years_checked': sorted(checked)}
        elif kind == 'era5':
            from openMWR.era5 import ERA5Processor
            processor = ERA5Processor(self.site, self.project_dir, start_date=settings['start_date'],
                                      end_date=settings['end_date'], update_only=not rebuilding,
                                      hour_interval=settings['hour_interval'])
            processor.create_original_dataset()
        elif kind == 'hatpro':
            from openMWR.hatpro_data import create_hatpro_dataset
            create_hatpro_dataset(self.site, self.project_dir, dataset=name, update_only=True,
                                  import_retrieval_data=settings['import_retrieval_data'],
                                  start_date=settings['start_date'], end_date=settings['end_date'], num_of_processes=processes)
        elif kind == 'profiles':
            source = self.recipe(recipe.inputs['source'])
            source_settings = self.settings(source.name)
            if source.kind == 'radiosonde':
                from openMWR.radiosonde import create_radiosonde_dataset_for_RT
                preparation = {key: settings[key] for key in ('cut_off_at_mwr_height', 'scale_rh_clear_sky',
                               'scale_rh_clear_sky_seed', 'scale_rh_clear_sky_fraction', 'scale_rh_clear_sky_factor')}
                for station in self.sources(name):
                    create_radiosonde_dataset_for_RT(self.site, station, self.project_dir, np.asarray(settings['heights']),
                        dataset=name, update_only=True, first_year=source_settings['start_year'],
                        last_year=source_settings['end_year'], **preparation)
            else:
                from openMWR.era5 import ERA5Processor
                processor = ERA5Processor(self.site, self.project_dir, start_date=source_settings['start_date'],
                    end_date=source_settings['end_date'], hour_interval=source_settings['hour_interval'], profiles_dataset=name)
                processor.create_profile_dataset(np.asarray(settings['heights']), num_of_processes=processes)
        elif kind == 'simulation':
            profiles = recipe.inputs['profiles']
            options = dict(freqs=np.asarray(settings['frequencies']), angles=np.asarray(settings['angles']),
                           absmdl=settings['absorption_model'], RT_model=settings['backend'],
                           freq_shift=None if settings['frequency_shift'] is None else np.asarray(settings['frequency_shift']),
                           num_of_processes=processes)
            if self.source_kind(name) == 'radiosonde':
                from openMWR.radiosonde import create_radiosonde_dataset_with_RT
                for station in self.sources(name):
                    create_radiosonde_dataset_with_RT(station, self.site, self.project_dir,
                        profiles_dataset=profiles, dataset=name, update_only=True, **options)
            else:
                from openMWR.era5 import ERA5Processor
                processor = ERA5Processor(self.site, self.project_dir, profiles_dataset=profiles, simulation_dataset=name)
                processor.create_forward_calc_dataset(**options)
        elif kind == 'split':
            from openMWR.train import separate_training_data
            separate_training_data(self.site, self.sources(name), self.project_dir,
                simulation_dataset=recipe.inputs['simulation'], dataset=name, update_only=True,
                test_fraction=settings['test_fraction'], val_fraction=settings['validation_fraction'], seed=settings['seed'])
        elif kind == 'analysis':
            # Matching uses both inputs, so regenerate when either input changes.
            observations, reference = recipe.inputs['observations'], recipe.inputs['reference']
            if self.source_kind(name) == 'radiosonde':
                from openMWR.radiosonde import create_analysis_dataset
                create_analysis_dataset(self.site, self.sources(name), self.project_dir,
                    observations_dataset=observations, reference_dataset=reference, dataset=name)
            else:
                from openMWR.era5 import ERA5Processor
                processor = ERA5Processor(self.site, self.project_dir, simulation_dataset=reference)
                processor.create_analysis_dataset_from_forward_calc(observations_dataset=observations, dataset=name)
