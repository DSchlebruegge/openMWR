"""Detached command launch and console capture in the existing task logs."""

import logging
import os
from pathlib import Path
import subprocess
import sys

from openMWR.config import check_name
from openMWR.paths import dataset_dir, retrieval_dir, evaluation_log_file


LOG_ENV = '_OPENMWR_BACKGROUND_LOG'
_console_path = None


def launch(args, argv):
    """Start the same CLI in a new session and report its PID and console log."""
    if os.name != 'posix':
        raise ValueError('--background requires a POSIX system, such as Linux or macOS.')
    check_name(args.site, 'site')
    if args.command == 'data':
        for name in args.dataset:
            check_name(name, 'dataset')
        path = dataset_dir(args.project_dir, args.site, args.dataset[0]) / 'build.log'
    elif args.command == 'predict':
        from openMWR.site import prediction_output_dir

        for name in args.retrieval:
            check_name(name, 'retrieval')
        path = prediction_output_dir(args.project_dir, args.site, args.output_dir) / 'prediction.log'
    elif args.command == 'evaluate':
        from openMWR.evaluation import EvaluationRecipe

        EvaluationRecipe.load(args.project_dir, args.site, args.evaluation)
        path = evaluation_log_file(args.project_dir, args.site, args.evaluation)
    elif args.command == 'omb':
        check_name(args.dataset, 'dataset')
        path = dataset_dir(args.project_dir, args.site, args.dataset) / 'omb.log'
    else:
        check_name(args.retrieval, 'retrieval')
        from openMWR.recipes import RetrievalRecipe

        recipe = RetrievalRecipe.load(args.project_dir, args.site, args.retrieval)
        action = 'optimization' if recipe.requires_optimization else 'training'
        path = retrieval_dir(args.project_dir, args.site, args.retrieval) / f'{action}.log'
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment[LOG_ENV] = str(path)
    with path.open('a', encoding='utf-8') as output:
        process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'openMWR', *argv],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment,
        )
    print(f'Started {args.command} in background (PID {process.pid}).')
    print(f'Console log: {path}')


def capture_console(path):
    """Capture logging once, including libraries with their own console handler."""
    global _console_path
    _console_path = Path(path).resolve()
    root = logging.getLogger()

    def not_already_logged(record):
        logger = logging.getLogger(record.name)
        while logger is not None and logger is not root:
            for handler in logger.handlers:
                if (isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
                        and handler.stream in (sys.stdout, sys.stderr)
                        and handler.level <= record.levelno and not handler.filters):
                    return False
            if not logger.propagate:
                break
            logger = logger.parent
        return True

    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.addFilter(not_already_logged)


def task_log_handler(path, **kwargs):
    """Keep normal task logs, leaving the captured file to standard output."""
    handler = logging.FileHandler(path, **kwargs)
    handler.addFilter(lambda record: Path(handler.baseFilename).resolve() != _console_path)
    return handler
