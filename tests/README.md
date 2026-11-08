# Tests

From the repository root, install both packages with
`python -m pip install -e ./torchMWRT -e '.[test]'`, then run:

```sh
python -m pytest
python -m pytest --cov --cov-report=term-missing --durations=10
```

The suite is offline, uses a headless plot backend and temporary projects, and
needs neither credentials nor a libRadtran installation. It exercises real
NetCDF files, microwave RT, training, SQLite studies, checkpoints and reports.
Numerical workflows use CPU. Device-selection tests mock CUDA/MPS availability.
External acquisition and the libRadtran executable are replaced at their I/O
boundaries. The libRadtran tests check discovery, profile-base altitude,
absolute cloud coordinates, scratch-file ownership and output parsing;
they do not establish infrared numerical accuracy.

CI uses Python 3.11, 3.14 and 3.15. Each job builds a source archive and a wheel
from that archive for both distributions and checks their metadata. It installs
torchMWRT first and calculates one R24 profile without openMWR installed, then
installs openMWR and runs this shared suite once against both wheels. This also
checks that packaged templates, grids and PyRTlib coefficients are available
without an editable checkout.

## Where a test belongs

| Area | Owns the detailed proof |
| --- | --- |
| `workflows/test_setup.py` | Creating a minimal NN starter, adding other example recipes and preserving user files |
| `workflows/test_migration.py` | Previewing/converting old layouts and checkpoints, original backups, all-site preflight and safe retry/repeat |
| `workflows/test_background.py` | Detachment, completed commands, preserved logs and startup failures |
| `workflows/test_datasets.py` | Build/update/rebuild, prediction-directory ownership, dependency sharing, calendar membership, readiness and recovery |
| `workflows/test_training.py` | Fixed training, Python/command equivalence, held-out fit, selected outputs, reproducibility and saved settings |
| `workflows/test_optimization.py` | Searches, scoring, recipe overrides, continuation across directory renames, reproducibility and safe replacement |
| `workflows/test_omb.py` | Paired selection and plots, station/angle reports, calibration markers, checkpoint noise, vector output, fitted shifts and saved OMB settings |
| `science/test_omb.py` | Physical moments, common-sample covariance and regression invariants |
| `workflows/test_evaluation.py` | Separate zenith/BLS groups, rescored baselines, coverage, explicit selections and standalone vector HTML reports |
| `workflows/test_prediction.py` | Applying saved models, smoothing, mixed profile/column outputs and missing observations |
| `science/test_observations.py` | Variable validity, UTC period bounds, independent station samples and matching across outages |
| `interfaces/test_devices.py` | Automatic device priority, explicit overrides and hardware availability |
| `interfaces/test_hatpro_backends.py` | Paired manufacturer exports and binary measurements/retrieval products, format selection, local timestamps, malformed files and integration with datasets/prediction |
| `interfaces/test_lineshape_data.py` | Direct coefficient loading from read-only installed packages, one table read per model, configuration errors and missing-table diagnostics |
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
NetCDF inputs, focusing on R24 for clear/cloudy profiles, angles, curved rays,
satellite viewing and the isolated 118 GHz oxygen and 183 GHz water-vapour lines.
Other supported absorption models retain representative
checks. It records input hashes, units and
generator versions. Ordinary tests only read it. They fail if inputs change.
Most comparisons retain `rtol=5e-5, atol=1e-6 K`; the radiosonde cases use
`rtol=0, atol=1e-8 K`. Gradient checks compare autograd with finite differences
for temperature, pressure, humidity and cloud water in R24, including batched
curved paths. A separate R17 case covers the older absorption formulation.
The profile/RT integration checks both geometry choices and rejects invalid rays
without silently dropping failed profiles from the saved simulation.
Water-vapour input checks compare scalar, profile and batched calculations,
including different frequency-grid ranks and zero humidity.

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

## Historical checkpoints

`interfaces/test_storage.py` checks explicit conversion and current-format loading using small checkpoints and expected
predictions generated by the actual 0.1 code. [checkpoint_01.md](data/checkpoint_01.md)
records their provenance and deliberate regeneration procedure. These fixtures
cover zenith NN, BLS NN, MLR and spectral loading without a recipe or retraining,
unchanged weights/normalization and byte-identical original backups.
