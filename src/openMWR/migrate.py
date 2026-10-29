"""One-time conversion of 0.1 site settings and saved retrieval paths."""

from importlib.resources import files
import json
import logging
import shutil
import tomllib

import tomli_w

from openMWR.config import check_name
from openMWR.paths import checkpoint_file, project_root, site_config_file, site_root

logger = logging.getLogger(__name__)


def migrate_project(project_dir, *, dry_run=False):
    """Migrate all 0.1 sites without loading models or adopting scientific data.

    Original JSON files stay in place. Checkpoints move without being rewritten.
    All configurations and destinations are checked before the first write.
    An existing site TOML must match the converted settings and is kept when
    repeating the command. Returns the names of the legacy sites found.
    """
    root = project_root(project_dir)
    legacy = root / 'data/sites'
    if not legacy.is_dir():
        raise FileNotFoundError(f'No legacy site directory: {legacy}. Run from the project containing data/.')

    sites, documents, moves = [], {}, []
    for directory in sorted(legacy.iterdir()):
        if not directory.is_dir():
            continue
        source = directory / 'config.json'
        checkpoints = sorted((directory / 'retrieval').glob('*.pth'))
        if not source.exists() and not checkpoints:
            continue
        site = directory.name
        check_name(site, 'site')
        try:
            settings = json.loads(source.read_text(encoding='utf-8'))
            if not isinstance(settings, dict):
                raise ValueError('site configuration must be a JSON object.')
            settings = {key: value for key, value in settings.items() if value is not None}
            # Version 0.1 read NetCDF only. Do not enable a different backend.
            settings.setdefault('hatpro_format', 'netcdf')
            text = tomli_w.dumps(settings)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f'Cannot convert {source}: {exc}') from exc
        destination = site_config_file(root, site)
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or tomllib.loads(destination.read_text(encoding='utf-8')) != settings:
                raise FileExistsError(f'Conflicting site settings: {destination}. Existing files are never overwritten.')
        else:
            documents[destination] = text
        for checkpoint in checkpoints:
            check_name(checkpoint.stem, 'retrieval')
            if not checkpoint.is_file():
                raise ValueError(f'Not a checkpoint file: {checkpoint}.')
            target = checkpoint_file(root, site, checkpoint.stem)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f'Checkpoint destination already exists: {target}. Existing files are never overwritten.')
            moves.append((checkpoint, target))
        sites.append(site)

    if not sites:
        logger.info('No 0.1 site configurations or checkpoints found in %s. Nothing to migrate.', legacy)
        return []

    grids = root / 'grids.toml'
    if not grids.exists() and not grids.is_symlink():
        documents[grids] = files('openMWR').joinpath('templates', 'grids.toml').read_text(encoding='utf-8')
    elif not grids.is_file():
        raise FileExistsError(f'Not a grids file: {grids}. Existing paths are never overwritten.')

    # Include empty dataset/retrieval directories in preflight, even for sites
    # that have no saved models. A conflict at a later site must stop all writes.
    directories = [site_root(root, site) / name for site in sites for name in ('datasets', 'retrieval')]
    for path in [*documents, *(target for _, target in moves), *directories]:
        parent = path if path in directories else path.parent
        while parent != root:
            if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
                raise FileExistsError(f'Migration needs a directory: {parent}. Existing paths are never overwritten.')
            parent = parent.parent

    for destination in documents:
        logger.info('%s %s', 'Would create' if dry_run else 'Create', destination.relative_to(root))
    for source, target in moves:
        logger.info('%s %s -> %s', 'Would move' if dry_run else 'Move', source.relative_to(root), target.relative_to(root))
    if not dry_run:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        for destination, text in documents.items():
            with destination.open('x', encoding='utf-8') as stream:
                stream.write(text)
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects a destination created after
            # preflight. Copy before unlinking so data/ may be on another disk.
            migrated = target.open('xb')
            try:
                with migrated, source.open('rb') as original:
                    shutil.copyfileobj(original, migrated)
            except BaseException:
                target.unlink()
                raise
            source.unlink()

    configurations = sum(path.name == 'site.toml' for path in documents)
    logger.info('%s: %d sites checked, %d site configurations %s, %d checkpoints %s.',
                'Preview' if dry_run else 'Migration complete', len(sites), configurations,
                'to create' if dry_run else 'created', len(moves), 'to move' if dry_run else 'moved')
    return sites
