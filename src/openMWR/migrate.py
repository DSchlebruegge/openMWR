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
from openMWR.paths import checkpoint_file, project_root, site_config_file, site_root

logger = logging.getLogger(__name__)


def _convert_checkpoint(path, name):
    import torch
    import xarray as xr
    from openMWR.models import CHECKPOINT_VERSION, Model, SpectralConsistencyModel

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    version = checkpoint.get('format_version')
    if version == CHECKPOINT_VERSION:
        return None
    if version is not None:
        raise ValueError(f'Unsupported checkpoint format {version!r}.')
    structure = checkpoint['structure']
    kind = checkpoint.get('model_class')
    if kind is None:
        kind = 'Model' if 'heights' in structure else 'SpectralConsistencyModel'
    classes = {'Model': Model, 'SpectralConsistencyModel': SpectralConsistencyModel}
    if kind not in classes:
        raise ValueError(f'Cannot convert custom checkpoint model class {kind!r}.')
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
    model = classes[kind]._from_checkpoint(checkpoint, name)
    checkpoint['structure'] = model.checkpoint_structure()
    checkpoint['model_class'] = kind
    checkpoint['format_version'] = CHECKPOINT_VERSION
    return checkpoint


def _write_checkpoint(source, target, checkpoint, modified):
    import torch

    if source.stat().st_mtime_ns != modified:
        raise RuntimeError(f'Checkpoint changed during migration: {source}. Retry after its writer has stopped.')
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name('model.pth.bak')
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
            if backup.exists() or backup.is_symlink():
                raise FileExistsError(f'Checkpoint backup already exists: {backup}. Keep it and choose another name before retrying.')
            conversions.append((path, target, checkpoint, modified))
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
    for path in [*documents, *(target for _, target, _, _ in conversions), *directories]:
        parent = path if path in directories else path.parent
        while parent != root:
            if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
                raise FileExistsError(f'Migration needs a directory: {parent}. Existing paths are never overwritten.')
            parent = parent.parent
    for destination in documents:
        logger.info('%s %s', 'Would create' if dry_run else 'Create', destination.relative_to(root))
    for source, target, _, _ in conversions:
        logger.info('%s %s -> %s (original: %s)', 'Would convert' if dry_run else 'Convert',
                    source.relative_to(root), target.relative_to(root), target.with_name('model.pth.bak').relative_to(root))
    if not dry_run:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        for destination, text in documents.items():
            with destination.open('x', encoding='utf-8') as stream:
                stream.write(text)
        for source, target, checkpoint, modified in conversions:
            _write_checkpoint(source, target, checkpoint, modified)
    configurations = sum(path.name == 'site.toml' for path in documents)
    logger.info('%s: %d sites checked, %d site configurations %s, %d checkpoints %s.',
                'Preview' if dry_run else 'Migration complete', len(sites), configurations,
                'to create' if dry_run else 'created', len(conversions), 'to convert' if dry_run else 'converted')
    return sites
