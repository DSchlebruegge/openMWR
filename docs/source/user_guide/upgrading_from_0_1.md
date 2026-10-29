# Upgrading from 0.1

Version 0.2 changes the project layout, checkpoint format and Python interface.
Existing trained retrievals can be converted without retraining.
Keep a project backup before migrating. The `0.1` Git tag identifies the previous
interface.

Install the new version using the [installation instructions](installation.md).
Python 3.11 or newer is required. An editable installation must point to the
updated openMWR checkout.

## Keep using a trained retrieval

Run the following examples from your working project, the directory containing
`data/`. Replace `my_site` and `NN` with your own names.

Preview the migration, then apply it:

```bash
openMWR migrate --dry-run
openMWR migrate
```

The command converts every `data/sites/<site>/config.json` to
`sites/<site>/site.toml` and moves each `retrieval/<name>.pth` into
`sites/<site>/retrieval/<name>/model.pth`. It makes the saved model class, grids,
loss weights and dataset name explicit. Model weights, normalization and names
stay the same. Each original checkpoint is retained as `model.pth.bak` beside
the converted file. Further upgrades of that file use numbered backups such as
`model.pth.bak.1`, preserving earlier originals. Original JSON files, datasets, scripts and optimization
databases stay in place. A shared `grids.toml` is added if missing. No dataset or
training recipes are generated. Extra historical site fields are retained under
`[custom]`. Scientific processing choices now belong in dataset recipes.

The command also converts older checkpoints already in the new directories.
Run it again if you used the earlier layout-only migration. Current-format
checkpoints and matching site TOML files are kept on repeated runs.

All conversions and destinations are checked before any changes. Conflicting
site settings, occupied destinations for old-layout files and occupied backups
for those files stop conversion. Original backups are never overwritten.

Current checkpoints contain tensors and plain metadata and use restricted
loading. Older formats require this explicit conversion. Migration reads older
Python objects, so migrate only checkpoints you trust.

Check paths in the converted site TOML files. Relative paths now start at the
working-project root. `hatpro_format = "netcdf"` keeps the reader choice of 0.1.
Choose a saved retrieval explicitly when predicting:

```bash
openMWR predict my_site NN --date 2025-01-12 --output-dir data/migration_prediction
```

Prediction reads the saved weights, grids and normalization. It does not need a
`retrieval.toml`. The example saves to a separate comparison directory.
New zenith and BLS prediction files both retain a `model` dimension, including
single-model results. Previously saved scalar-model BLS files remain readable.

Zenith predictions use a centred five-sample mean, as the old daily workflow did.
In Python, `PredictionWorkflow(..., smooth=False)` returns unsmoothed predictions.
Compare versions on identical prepared observations first. Reader fixes can
change which raw measurements are accepted.

BLS prediction requires the model's elevation angles in the measured scans.
A previously interpolated cache does not establish that the raw scans contain
those angles. Use compatible measurements or a retrieval trained for the measured grid.

## Update Python scripts

Pass the working-project root as `project_dir`, replacing the old `data_dir`
argument. For example, replace `data_dir="data"` with `project_dir="."`.

| Old use | New use |
| --- | --- |
| `from openMWR.run import run_model, run_model_list` | Import the same functions from `openMWR.predict` |
| `Model.load("NN", site, "data")` | `Model.load("NN", site, ".")` |
| `import_data_and_run_retrieval(...)` | `PredictionWorkflow(...).predict(day)` |
| `std_heights`, `std_freqs`, `std_angles` | `load_grid(project_dir, kind, name)` or explicit numeric arrays |
| Processing selected by `file_extension` | Named dataset recipes |

[Using Python](python_usage.md) shows prediction, plotting and custom training
with the new workflows. Direct neural profile construction now requires explicit
`loss_weights` and `output_vars`. Existing models retain all five profile/column
outputs during conversion. Existing profile recipes need an explicit
`output_vars = ["T", "rh", "lwc", "lwp", "iwv"]` entry in `[model]`.
BLS models also require explicit `angles`. Direct RT calls
require an explicit absorption model, backend and angle grid.

Site settings are validated when loaded. Put project-specific fields in
`[custom]`. Unknown top-level fields are rejected, with suggestions for misspelled
settings.

## Keep or rebuild processed datasets

Dataset recipes live in `sites/my_site/datasets/`. Set their scientific choices
to match the code that generated your files. The starter recommends R24, which
does not describe an existing R17 or R20 simulation. Check frequency offsets,
grids and profile preparation too. A setting in the old site JSON does not
necessarily mean it was passed to the calculation.

These are example names for the new directories:

| Old file under `data/sites/my_site/` | New file under the same directory |
| --- | --- |
| `radiosonde/radiosonde_data_03715.nc` | `profiles/03715.nc` |
| `radiosonde/radiosonde_data_with_RT_03715.nc` | `simulation/03715.nc` |
| `hatpro/hatpro_data.nc` | `hatpro/zenith.nc` |
| `hatpro/hatpro_data_bls.nc` | `hatpro/bls.nc` |
| `analysis/analysis_data_radiosonde.nc` | `analysis_radiosonde/zenith.nc` |
| `analysis/analysis_data_radiosonde_bls.nc` | `analysis_radiosonde/bls.nc` |
| `era5/forward_calc_era5.nc` | `simulation_era5/era5.nc` |

Separate settings need separate recipes and directories. Raw radiosonde caches
keep their paths. Raw ERA5 cache names now include the requested hour interval,
for example `data/era5/era5_48.25_11.5_2h.nc`. Verify the old sampling before
renaming. Existing caches can contain mixed historical intervals.

Moved derived files also need processing records. After checking their settings
and inputs, record each adopted dataset in dependency order:

```python
from openMWR.datasets import DatasetWorkflow

datasets = DatasetWorkflow(".", "my_site")
datasets.record("profiles", provenance="Imported from 0.1. Preparation checked against the original script.")
datasets.record("simulation", provenance="Imported from 0.1. RT settings checked against the original script.")
```

`record()` records your declaration. It cannot recover or verify historical
settings. If those settings are unknown, retain the files for reference and
rebuild that dataset. Do not create records that claim guessed settings are known.
See [recipe settings](recipe_settings.md) for the required choices.

## Training and optimization

Keep the old random training, validation and test files as historical data.
Version 0.2 assigns whole days by their day of the month. Create a split recipe
with explicit test and validation days and the old training-period cutoff, then
build it from the adopted simulation:

```bash
openMWR data my_site training --rebuild
```

Without `--with-dependencies`, this uses the existing simulation without
downloading data or repeating RT. An older model may have trained on dates now
assigned to test. Its scores on the new split are not independent test scores.

For future fitting, create a retrieval recipe using the examples in
[Create a Retrieval](create_retrieval.md). A checkpoint often preserves fitted
parameter values, but may omit other training choices, search ranges or bias/OMB selection.
Keep the original scripts and Optuna database as reference. Version 0.1 studies
cannot be continued through the new optimization workflow.

MLR studies fitted with random input-noise copies require `--fresh` or a new
retrieval name. They cannot continue with the expected-noise fitting method.
Saved models remain usable for prediction and evaluation.

OMB experiments also need a fresh fit. Bias adjustment now adds the mean
observed-minus-simulated TB to synthetic training inputs. Regression residuals
are calculated in kelvin. OMB noise recipes require an explicit `statistic`
and regression requires `frequency_bands`. See the
[recipe settings](recipe_settings.md#input-noise-and-bias-correction).
Saved models retain their fitted behavior after migration.

Use a new retrieval name for a new experiment. `--fresh` on an existing name
deletes its saved model and optimization database.
