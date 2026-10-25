# Contributing Guide

Any contributions to the openMWR package are very welcome!

## How to Contribute

You can contribute by:
- Reporting bugs
- Suggesting improvements
- Adding new features
- Improving documentation

To contribute, please create a new issue or create a pull request with your changes.

## Build the documentation

With your Python environment activated, run from the openMWR checkout:

```bash
python -m pip install --editable ".[docs]"
python -m sphinx -b html docs/source docs/_build/html
```

For a uv-managed environment, use `uv pip install --editable ".[docs]"` for the
installation step. Open `docs/_build/html/index.html` to preview the site.
The build renders saved notebook outputs. It does not execute the notebooks.

## Possible features we want to add in the future

- support for more ways to download radiosonde data (e.g., from the University of Wyoming)
- compatibility with [MWRpy](https://github.com/actris-cloudnet/mwrpy) (For example, by outputting Level 2 data in the correct format)
- support hatpro binary files as input.
