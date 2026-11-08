"""One-time conversion of old site settings and checkpoint formats."""

from importlib.resources import files
import json
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import tomllib

import tomli_w

from openMWR.config import check_name
from openMWR.site import SITE_KEYS, _validate_site
from openMWR.paths import checkpoint_file, project_root, site_config_file, site_root

logger = logging.getLogger(__name__)


def _convert_angle_noise(checkpoint):
    """Separate historical per-angle factors from channel noise parameter names."""
    structure, parameters = checkpoint['structure'], checkpoint['hyper_params']
    if not structure.get('is_bls_model') or not structure.get('vary_input_noise_with_angle'):
        return False
    channels = {f'input_noise_TB_{frequency}' for frequency in structure['freqs']}
    changed = False
    for angle in structure['angles']:
        old, new = f'input_noise_TB_{angle}', f'input_noise_TB_angle_{angle}'
        if new not in parameters and old in parameters:
            parameters[new] = parameters[old]
            if old not in channels:
                del parameters[old]
            changed = True
    return changed


def _convert_spectral_normalization(checkpoint):
    """Fold historical double linear normalization into effective statistics."""
    import xarray as xr

    if checkpoint.get('model_class') != 'SpectralConsistencyModel' or checkpoint.get('normalization_passes') == 1:
        return False
    method = (checkpoint['structure'].get('norm_functions') or {'TB': 'z_norm'})['TB']
    if method == 'log_norm':
        raise ValueError('Historical spectral log_norm cannot be converted to single-pass normalization. '
                         'Retrain this spectral model. Its checkpoint is left unchanged.')
    normalization = xr.Dataset.from_dict(checkpoint['normalization'])
    if method in ('z_norm', 'std_norm', 'max_norm'):
        normalization['TB'] = normalization.TB.astype(float)
        stat = 'max' if method == 'max_norm' else 'std'
        scale = normalization.TB.sel(stat=stat, drop=True).copy()
        if method == 'z_norm':
            normalization.TB.loc[{'stat': 'mean'}] *= 1 + scale
        normalization.TB.loc[{'stat': stat}] = scale ** 2
    checkpoint['normalization'] = normalization.to_dict()
    checkpoint['normalization_passes'] = 1
    checkpoint['training_metadata'] = {**(checkpoint.get('training_metadata') or {}),
                                       'spectral_normalization_conversion': 'effective_linear_statistics'}
    return True


def _convert_checkpoint(path, name):
    import torch
    import xarray as xr
    from openMWR.models import CHECKPOINT_VERSION, Model, SpectralConsistencyModel, _checkpoint_payload, _read_checkpoint
    from openMWR.config import PROFILE_OUTPUTS

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    version = checkpoint.get('format_version')
    if version == CHECKPOINT_VERSION:
        _read_checkpoint(path)
        changed = _convert_angle_noise(checkpoint)
        changed = _convert_spectral_normalization(checkpoint) or changed
        return _checkpoint_payload(checkpoint) if changed else None
    if version not in (None, 1):
        raise ValueError(f'Unsupported checkpoint format {version!r}.')
    structure = checkpoint['structure']
    kind = checkpoint.get('model_class')
    if kind is None:
        kind = 'Model' if 'heights' in structure else 'SpectralConsistencyModel'
    classes = {'Model': Model, 'SpectralConsistencyModel': SpectralConsistencyModel}
    if kind not in classes:
        raise ValueError(f'Cannot convert custom checkpoint model class {kind!r}.')
    if kind == 'Model':
        # Both older formats always predicted these five variables in this order.
        structure['output_vars'] = list(PROFILE_OUTPUTS)
    if structure.get('is_bls_model') and structure.get('angles') is None:
        normalization = xr.Dataset.from_dict(checkpoint['normalization'])
        if 'ang' not in normalization.coords:
            raise ValueError('Saved BLS model has no angle coordinates. Its original grid is required.')
        structure['angles'] = normalization.ang.values.tolist()
    if kind == 'Model' and not structure.get('is_MLR') and structure.get('loss_weights') is None:
        # These are the actual implicit weights of the 0.1 algorithms.
        bls = structure.get('is_bls_model', False)
        profile = {'upper_bounds': [2000, 10000] if bls else [10000],
                   'values': [1., .5, .25] if bls else [1., .5]}
        structure['loss_weights'] = {
            **{key: profile for key in ('T', 'rh', 'lwc')}, 'lwp': 1., 'iwv': 1.}
    parameters = checkpoint['hyper_params']
    suffix = parameters.pop('file_extension', '')
    parameters.setdefault('training_dataset', f'training_{suffix.lower()}' if suffix else 'training')
    checkpoint['model_class'] = kind
    _convert_spectral_normalization(checkpoint)
    model = classes[kind]._from_checkpoint(checkpoint, name)
    checkpoint['structure'] = model.checkpoint_structure()
    checkpoint['model_class'] = kind
    checkpoint['format_version'] = CHECKPOINT_VERSION
    _convert_angle_noise(checkpoint)
    return _checkpoint_payload(checkpoint)


def _write_checkpoint(source, target, checkpoint, modified, backup):
    import torch

    if source.stat().st_mtime_ns != modified:
        raise RuntimeError(f'Checkpoint changed during migration: {source}. Retry after its writer has stopped.')
    target.parent.mkdir(parents=True, exist_ok=True)
    saved = backup.open('xb')
    try:
        with saved, source.open('rb') as original:
            shutil.copyfileobj(original, saved)
        shutil.copystat(source, backup)
    except BaseException:
        backup.unlink()
        raise
    temporary = None
    try:
        with NamedTemporaryFile(dir=target.parent, prefix='.model-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(checkpoint, stream)
        shutil.copystat(backup, temporary)
        if source.stat().st_mtime_ns != modified:
            raise RuntimeError(f'Checkpoint changed during migration: {source}. Retry after its writer has stopped.')
        if source == target:
            temporary.replace(target)
        else:
            # Both paths are in the same directory. Linking publishes the
            # completed file exclusively, including after a late conflict.
            target.hardlink_to(temporary)
    except BaseException:
        backup.unlink()
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if source != target:
        source.unlink()


def migrate_project(project_dir, *, dry_run=False):
    """Convert old sites and checkpoints, retaining original checkpoint backups.

    Handles the 0.1 layout and older checkpoints already in the current layout.
    All configurations, conversions and destinations are checked before writing.
    Current checkpoints and matching site TOML files are kept on repeated runs.
    Datasets, scripts and studies are untouched. Returns the site names checked.
    """
    root = project_root(project_dir)
    legacy = root / 'data/sites'
    current = root / 'sites'
    if not legacy.is_dir() and not current.is_dir():
        raise FileNotFoundError(f'No site directory in {root}. Run from the project containing data/ and sites/.')
    names = {path.parent.name for path in current.glob('*/site.toml')}
    if legacy.is_dir():
        names.update(path.name for path in legacy.iterdir() if path.is_dir()
                     and ((path / 'config.json').exists() or any((path / 'retrieval').glob('*.pth'))))
    sites, documents, conversions = [], {}, []
    for site in sorted(names):
        check_name(site, 'site')
        source = legacy / site / 'config.json'
        destination = site_config_file(root, site)
        old_checkpoints = sorted((legacy / site / 'retrieval').glob('*.pth'))
        if source.exists() or old_checkpoints:
            try:
                settings = json.loads(source.read_text(encoding='utf-8'))
                if not isinstance(settings, dict):
                    raise ValueError('site configuration must be a JSON object.')
                settings = {key: value for key, value in settings.items() if value is not None}
                settings.setdefault('hatpro_format', 'netcdf')
                if 'gen' in settings:
                    settings['instrument_generation'] = settings.pop('gen')
                if 'retrieval_output_dir' in settings:
                    settings.setdefault('prediction', {})['output_dir'] = settings.pop('retrieval_output_dir')
                for old_key, key in (('plot_dir', 'plot_dir'), ('rpg_retrieval_exists', 'include_rpg')):
                    if old_key in settings:
                        settings.setdefault('custom', {})[key] = settings.pop(old_key)
                extra = {key: settings.pop(key) for key in list(settings) if key not in SITE_KEYS}
                if extra:
                    settings['custom'] = {**extra, **settings.get('custom', {})}
                _validate_site(settings)
                text = tomli_w.dumps(settings)
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError(f'Cannot convert {source}: {exc}') from exc
            if destination.exists() or destination.is_symlink():
                if not destination.is_file() or tomllib.loads(destination.read_text(encoding='utf-8')) != settings:
                    raise FileExistsError(f'Conflicting site settings: {destination}. Existing settings are never overwritten.')
            else:
                documents[destination] = text
        paths = [(path, checkpoint_file(root, site, path.stem)) for path in old_checkpoints]
        paths.extend((path, path) for path in sorted((site_root(root, site) / 'retrieval').glob('*/model.pth')))
        for path, target in paths:
            check_name(target.parent.name, f'{site} retrieval')
            if path != target and (target.exists() or target.is_symlink()):
                raise FileExistsError(f'Checkpoint destination already exists: {target}. Existing files are never overwritten.')
            modified = path.stat().st_mtime_ns
            try:
                checkpoint = _convert_checkpoint(path, target.parent.name)
                if path != target and checkpoint is None:
                    import torch
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            except Exception as exc:
                raise ValueError(f'Cannot convert checkpoint {path}: {exc}') from exc
            if checkpoint is None:
                continue
            backup = target.with_name('model.pth.bak')
            if path == target:
                index = 1
                while backup.exists() or backup.is_symlink():
                    backup = target.with_name(f'model.pth.bak.{index}')
                    index += 1
            if backup.exists() or backup.is_symlink():
                raise FileExistsError(f'Checkpoint backup already exists: {backup}. Keep it and choose another name before retrying.')
            conversions.append((path, target, checkpoint, modified, backup))
        sites.append(site)

    if not sites:
        logger.info('No site configurations or checkpoints found. Nothing to migrate.')
        return []
    grids = root / 'grids.toml'
    if not grids.exists() and not grids.is_symlink():
        documents[grids] = files('openMWR').joinpath('templates', 'grids.toml').read_text(encoding='utf-8')
    elif not grids.is_file():
        raise FileExistsError(f'Not a grids file: {grids}. Existing paths are never overwritten.')
    directories = [site_root(root, site) / name for site in sites for name in ('datasets', 'retrieval')]
    for path in [*documents, *(target for _, target, _, _, _ in conversions), *directories]:
        parent = path if path in directories else path.parent
        while parent != root:
            if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
                raise FileExistsError(f'Migration needs a directory: {parent}. Existing paths are never overwritten.')
            parent = parent.parent
    for destination in documents:
        logger.info('%s %s', 'Would create' if dry_run else 'Create', destination.relative_to(root))
    for source, target, _, _, backup in conversions:
        logger.info('%s %s -> %s (original: %s)', 'Would convert' if dry_run else 'Convert',
                    source.relative_to(root), target.relative_to(root), backup.relative_to(root))
    if not dry_run:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        for destination, text in documents.items():
            with destination.open('x', encoding='utf-8') as stream:
                stream.write(text)
        for source, target, checkpoint, modified, backup in conversions:
            _write_checkpoint(source, target, checkpoint, modified, backup)
    configurations = sum(path.name == 'site.toml' for path in documents)
    logger.info('%s: %d sites checked, %d site configurations %s, %d checkpoints %s.',
                'Preview' if dry_run else 'Migration complete', len(sites), configurations,
                'to create' if dry_run else 'created', len(conversions), 'to convert' if dry_run else 'converted')
    return sites
