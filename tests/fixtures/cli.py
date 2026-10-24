"""Run the real command parser without importing the package in a subprocess each time."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import logging
from subprocess import CompletedProcess

from openMWR.cli import main


def run_cli(*args):
    stdout, stderr = StringIO(), StringIO()
    handler = logging.StreamHandler(stderr)
    logger = logging.getLogger()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                main(list(args))
                status = 0
            except SystemExit as exc:
                status = exc.code
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return CompletedProcess(args, status, stdout.getvalue(), stderr.getvalue())
