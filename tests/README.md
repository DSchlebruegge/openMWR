# Tests

Install the package with `python -m pip install -e '.[test]'`, then run:

```sh
python -m pytest
python -m pytest --cov --cov-report=term-missing --durations=10
```

The suite is offline, uses a headless plot backend and temporary projects, and
needs neither credentials nor a libRadtran installation. It exercises real
NetCDF files, microwave RT, training, SQLite studies, checkpoints and reports.
Numerical workflows use CPU. Device-selection tests mock CUDA/MPS availability.
External acquisition and the libRadtran executable are replaced at their I/O
boundaries. The libRadtran tests check discovery, arguments and output parsing;
they do not establish infrared numerical accuracy.

## Where a test belongs

| Area | Owns the detailed proof |
| --- | --- |
| `workflows/test_setup.py` | Creating a usable starter and preserving user files |
| `workflows/test_background.py` | Detachment, completed commands, preserved logs and startup failures |
| `workflows/test_datasets.py` | Build/update/rebuild, dependency sharing, calendar membership, readiness and recovery |
| `workflows/test_training.py` | Fixed training, Python/command equivalence, held-out fit, reproducibility and saved settings |
| `workflows/test_optimization.py` | Searches, scoring, continuation, reproducibility and safe replacement |
| `workflows/test_evaluation.py` | Observation selection, OMB preparation and portable reports |
| `workflows/test_prediction.py` | Applying saved models, smoothing, grouped daily outputs and missing observations |
| `science/test_observations.py` | Variable validity, UTC period bounds, independent station samples and matching across outages |
| `interfaces/test_devices.py` | Automatic device priority, explicit overrides and hardware availability |
| `science/` | Numerical values, units, coordinates, cloud/profile rules, loss weights and derivatives |
| `interfaces/` | Parser/dispatch, invalid recipes, editable grids, recommendations, storage, acquisition, plotting and external binaries |

Give each behavior one owner. Other tests may execute the same code to verify a
different contract. Prefer one representative real workflow over the same
scenario through every model/API combination. Keep separate cases for distinct
failure or data-preservation rules. Small parameter tables for genuinely
different invalid inputs or physical models are useful; test count is not a goal.
Assert outcomes and meaningful diagnostics rather than internal call order or
incidental log text. Fixtures in `fixtures/` are explicit small inputs, not a
second implementation of the algorithms. Use independent expected numbers or
reference implementations for numerical assertions. Branch coverage and durations
help find gaps and repetition; there is no percentage target or production-code
exclusion list.

## Microwave reference values

`data/rt_reference.json` contains fixed PyRTlib results for the three tracked
NetCDF inputs, including supported absorption models, clear/cloudy profiles,
angles, curved rays and satellite viewing. It records input hashes, units and
generator versions. Ordinary tests only read it. They fail if inputs change.
Most comparisons retain `rtol=5e-5, atol=1e-6 K`; the radiosonde cases use
`rtol=0, atol=1e-8 K`. Gradient checks compare autograd with finite differences.

For an explicit check against the installed PyRTlib version:

```sh
python -m pytest -m reference
```

To deliberately regenerate reference values, then review all numerical changes:

```sh
python -m tests.tools.generate_rt_reference
git diff -- tests/data/rt_reference.json
```

Never regenerate reference values just to make a failing numerical test pass.
CI runs the default suite with branch coverage, then the optional reference checks.
