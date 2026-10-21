"""Example: update selected datasets from the current working project."""
from pathlib import Path

from openMWR.datasets import DatasetWorkflow
from openMWR.utils import setup_logging


def main():
    setup_logging()
    DatasetWorkflow(Path.cwd(), 'munich_G5').build(
        ['analysis_radiosonde', 'analysis_era5'], with_dependencies=True)


if __name__ == '__main__':
    main()
