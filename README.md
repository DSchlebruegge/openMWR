# openMWR

openMWR is a Python package for training a neural network-based retrieval for a Hatpro microwave radiometer.

It includes tools for generating training datasets, training retrieval algorithms,
analyzing the performance of retrieval algorithms, and applying the trained
algorithms to measurements operationally.

## Usage

For a full user guide, please refer to the [Documentation](https://dschlebruegge.github.io/openMWR/).

Working projects contain `data/` for datasets and `sites/` for site TOML settings
and named retrievals. Affected Python APIs take this root as `project_dir`.
The scripts under `bin/` are Munich-specific examples: run them with your working
project as the current directory and adapt their site/model choices as needed.

Installing the package provides the `openMWR` command. With your environment
activated, use `openMWR --help` or, for example,
`openMWR train munich_G5 NN_example`.
You can also use `python -m openMWR ...`; both forms call the same implementation
and work with pip or uv installations.

Common options also have short aliases, shown by `openMWR COMMAND -h`:

| Option | Alias |
| --- | --- |
| `--project-dir` | `-p` |
| `--device` | `-D` |
| `--with-dependencies` | `-w` |
| `--dry-run` | `-n` |
| `--rebuild` | `-r` |
| `--n-trials` | `-t` |
| `--fresh` | `-f` |
| `--continue` | `-c` |
| `--output` / `--output-dir` | `-o` |
| `--date` | `-d` |
| `--start-date` / `--end-date` | `-s` / `-e` |

For example:

```bash
openMWR data munich_G5 analysis_era5 -w -n
openMWR train munich_G5 NN_opt -c -t 100
```

### Create a site

From your working-project directory:

```bash
openMWR create-site my_site
```

This writes `sites/my_site/site.toml`, six dataset recipes for a radiosonde
training chain and HATPRO/radiosonde comparisons, and eight retrieval recipes under
`sites/my_site/retrieval/`: `NN_example` (zenith NN), `MLR_example` (zenith MLR),
`NN_BLS_example` (boundary-layer-scan NN), and `SPC_example` (spectral consistency),
each with a matching `_opt_example` recipe for optimization. It creates empty data
directories and leaves existing sites untouched by refusing to overwrite them.
It also creates a shared `grids.toml` in the project directory if missing,
containing the initial height, frequency and elevation-angle grids. Existing
grid definitions are preserved when adding another site.
`--project-dir /path/to/project` selects another project.

Fill in the commented site settings and the empty `stations` list in
`datasets/radiosondes.toml`. The source list is inherited throughout the chain.
The examples write recommended model, normalization, training and noise
settings explicitly and share the same training dataset; the BLS example also
inherits its elevation angles. `NN_example` is the site's default prediction selection.
Missing scientific choices produce an error; optional
processing is off unless enabled. Operational options keep convenient defaults.
The optimization examples write explicit search ranges and trial budgets.
Creation performs no downloads or calculations; RT additionally needs libRadtran.

After configuration, check the chain before processing:

```bash
openMWR data my_site training --with-dependencies --dry-run
openMWR data my_site training --with-dependencies
openMWR train my_site NN_example
```

`analysis_radiosonde` prepares observations for evaluation when built with
dependencies. Copy and edit recipes to create further experiments.
Python callers use `create_site("my_site", project_dir=".", with_examples=True)`
from `openMWR.site`; omitting `with_examples` creates directories, the shared
grid file if missing, and the explicitly supplied site settings.

### Dataset recipes and commands

Each dataset has a named TOML file under `sites/<site>/datasets/`. Site settings
describe the instrument/location; dataset recipes own preparation and RT choices.
Generated files and processing records live under `data/sites/<site>/<name>/`.
Raw downloads remain shared under `data/radiosondes/` and `data/era5/`.

For example, these three recipe files describe one processing chain:

```toml
# radiosondes.toml
kind = "radiosonde"
stations = ["03715"]
start_year = 1990
end_year = "current_year"
refresh_years = 2
```

```toml
# profiles.toml
kind = "profiles"

[inputs]
source = "radiosondes"

[preparation]
incomplete_profiles = "monthly_mean"  # or "filter" to reject incomplete profiles
heights = "levels_208_30km"
cut_off_at_mwr_height = false
# heights = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 30000]

[preparation.cloud]
rh_threshold = 0.96  # fraction (96% RH)
min_temperature = 253.15  # liquid cloud temperature threshold [K]
# LWC = f_ad * LWC_adiabatic; h is height above cloud base [m], H is cloud depth [m].
# power_law: f_ad = a * (1 - (h / H)^b)
# karstens: f_ad = max(0, 1.239 - 0.145 * log(h + 1)); log is the natural logarithm.
adiabatic_fraction = "power_law"
# "none" keeps fully adiabatic LWC (f_ad = 1).
# a and b apply only to power_law; remove both for karstens or none.
a = 0.7
b = 2.3

[preparation.quality]
min_top_height = 15000  # metres above sea level, before rebasing/interpolation
min_levels = 500        # distinct valid source levels, before interpolation
```

```toml
# simulation_r17.toml
kind = "simulation"

[inputs]
profiles = "profiles"

[radiative_transfer]
backend = "torchMWRT"
absorption_model = "R17"
frequencies = "hatpro"
angles = "hatpro_bls"

[radiative_transfer.infrared]
droplet_number_density = 200000000.0  # m^-3, equivalent to 200 cm^-3
min_effective_radius = 2.5           # micrometres

[execution]
processes = 12
```

Copying a recipe to a new filename creates a separate dataset identity. Its name
does not imply settings. Several simulations can share profiles; several profile
preparations can share the same raw source.

```bash
openMWR data munich_G5 simulation_r17
openMWR data munich_G5 simulation_r17 --with-dependencies
openMWR data munich_G5 simulation_r17 --with-dependencies --dry-run
openMWR data munich_G5 simulation_r17 --rebuild
```

`data` creates missing output or updates existing output. Without
`--with-dependencies`, inputs must already exist. With it, prerequisites run once
in dependency order. Site and dataset names are required positional arguments:
`openMWR data SITE DATASET [DATASET ...]`. List several dataset names to select
multiple targets.
`--dry-run` checks recipes and shows the order without writing or contacting
sources. `--rebuild` regenerates only explicitly selected outputs; dependencies
receive ordinary updates. Rebuilding a source refreshes its configured raw
coverage while preserving caches shared with other recipes.

Supported recipe kinds and input roles are:

| Kind | Inputs | Settings |
| --- | --- | --- |
| `radiosonde` | None | Required `stations`, `start_year`, `end_year`; optional `refresh_years` (2) |
| `era5` | None | Required `start_date`, `end_date`, `hour_interval`; `delay_days` required with `end_date = "today"` |
| `hatpro` | None | Required dates; `[observations] import_retrieval_data` inherits the site setting (off when absent) |
| `profiles` | `source`: radiosonde or ERA5 | `[preparation] heights`; radiosondes also require `incomplete_profiles`, `cloud` and `quality`, with optional cutoff |
| `simulation` | `profiles` | `[radiative_transfer] backend`, `absorption_model`, `frequencies`, `angles`, required `infrared` microphysics, optional `frequency_shift` |
| `split` | `simulation` | Required `[split] test_days_of_month`, `validation_days_of_month`; optional `start_date`, `end_date`, `days_of_month` |
| `analysis` | `observations`: HATPRO; `reference`: simulation | Required `[analysis] max_time_difference_minutes`, `exclude_rain`, `height_range`; optional `stations` |

Derived recipes inherit their input stations unless selecting a subset.
Split recipes select their profile pool, then assign whole days to test and
validation using inclusive monthly ranges. All other days go to training:

```toml
[split]
start_date = "1990-01-01"       # optional pool bounds
end_date = "2024-10-31"         # includes the entire day
test_days_of_month = { first = 1, last = 3 }
validation_days_of_month = { first = 15, last = 17 }
```

The day-of-month comes from the dataset's UTC time coordinate. The same rule
applies to every source/station and all timestamps within each day.
Ranges must not overlap and the selected pool must contain profiles for all
three splits. Missing profiles and month lengths determine the resulting sizes;
there are no split fractions or random seed. Changing the selection requires
an explicit rebuild or a new dataset name. Unchanged inputs skip processing;
changed inputs regenerate the three files with the same calendar rule.
Scientific settings, including the RT backend/model and grids, must be explicit.
Grids accept numeric lists or names from the project's `grids.toml`. Its starter contains
`levels_208_30km` for 208 heights from 0 to 30,000 m, `hatpro` for frequencies (GHz), and
`hatpro_bls` for elevation angles (degrees). Add or edit definitions there;
all sites share this file. For example, additional grids can be defined as:

```toml
# grids.toml: entries within the respective existing tables
[heights]
custom = [0, 100, 500, 1000, 2000, 5000, 10000, 30000]

[frequencies]
custom = [22.24, 31.4]

[angles]
custom = [90.0, 30.0, 10.0]
```

Recipes select them with `heights = "custom"`, `frequencies = "custom"`, or
`angles = "custom"` in the corresponding settings sections. Numeric values are
recorded in processing state and checkpoints; changes to grids used by existing
datasets require an explicit rebuild. Named grids require the project's file.
Python uses the same loader and definitions:

```python
from openMWR.config import load_grid
from openMWR.models import Model

profile_weights = {"upper_bounds": [2000, 10000], "values": [1.0, 0.5, 0.25]}
model = Model(
    "my_retrieval",
    heights=load_grid(".", "heights", "levels_208_30km"),
    freqs=load_grid(".", "frequencies", "hatpro"),
    is_bls_model=True,
    angles=load_grid(".", "angles", "hatpro_bls"),
    loss_weights={
        "T": profile_weights, "rh": profile_weights, "lwc": profile_weights,
        "lwp": 1.0, "iwv": 1.0,
    },
)
```

BLS models require explicit angles. The former `std_heights`, `std_freqs` and
`std_angles` arrays are removed; numerical functions and saved models use the
resolved coordinates.

Radiosonde profiles explicitly select the cloud model and source quality limits
shown above. Choose `adiabatic_fraction = "power_law"` with explicit `a`/`b`, or
`"karstens"` or `"none"` without those coefficients. `"none"` keeps the unreduced
adiabatic liquid water content. The temperature threshold and formulas/units
are stated in recipe comments. ERA5 preparation uses its source cloud water and accepts
only `heights`; these radiosonde settings do not apply.

`create-site` recommends R24; existing recipes retain their explicit model choice.
`execution.processes` controls workers for ERA5 preparation, RT or HATPRO;
radiosonde preparation is serial.

Source dates are inclusive. Set `start_date = "site"` to inherit the site's
measurement start and `end_date = "today"` to follow current data. ERA5 subtracts
its explicit `delay_days` from today. Radiosondes accept an integer end year or
`end_year = "current_year"` for ongoing updates. ERA5's cache key
includes location and sampling. Routine updates append days after the latest cached
day, leaving historical gaps and earlier sampling unchanged. Radiosondes
download missing historical years and refresh the recent years. Profiles and RT
append missing timestamps. Corrected values at already processed timestamps
require explicitly rebuilding the affected derived datasets. Split updates keep
all existing membership and assign new samples to maintain the requested overall
fractions; updates never reshuffle old samples. Analysis regenerates when its
inputs change. Unchanged inputs skip derived processing.

Narrowing date coverage requires rebuilding the affected derived datasets.
Extending historical HATPRO coverage also requires a rebuild; regular HATPRO
updates refresh from the last available day.

Radiosonde `incomplete_profiles = "monthly_mean"` fills missing upper levels
using monthly means. `"filter"` rejects profiles that do not cover the required
grid. Both modes reject any profiles still missing T, p, RH or LWC; RT checks
these fields before calculation. Monthly completion needs other profiles in
the same month that provide the missing levels.

Analysis recipes explicitly select matching, rain filtering and their profile range:

```toml
[analysis]
max_time_difference_minutes = 10
exclude_rain = true
height_range = [0, 10000]  # metres above the instrument
```

These settings apply to both radiosonde and ERA5 analysis, for zenith and scan
data. An evaluation range must be covered by the prepared analysis dataset.
IR droplet concentration and minimum effective radius belong in the simulation's
`[radiative_transfer.infrared]` table; they affect infrared simulation only.
The minimum radius must be at least 2.5 and below 60 micrometres. Calculated
radii of 60 micrometres or more are rejected before RT starts, matching
libRadtran's Hu-Stamnes cloud optics.

Effective settings and input identities are recorded in `state.json`, and each
dataset has `build.log`. Changing scientific settings or rebuilding an input
requires an explicit downstream rebuild, preventing mixed processing settings.
Rebuild errors identify changed fields and their old/new values, including
the responsible input dataset. They print a command for the affected chain,
with an explicit project directory so it also works outside the project folder.
The hint selects the necessary datasets directly; healthy source caches are
reused without a routine dependency refresh. Missing or unreadable processing
records show their path and explain that the existing processing history cannot
be verified. An explicit rebuild regenerates both data and its processing record.
Moving a recipe alone does not move its generated data; keep the dataset name
stable when updating it.

Python uses exactly the same workflow:

```python
from openMWR.datasets import DatasetWorkflow

data = DatasetWorkflow(project_dir=".", site="munich_G5")
data.build("simulation_r17")
data.build(["analysis_radiosonde", "analysis_era5"], with_dependencies=True)
data.require("training", sources=["03715"], splits=("training", "validation"))
```

`require` checks recipe lineage, rebuild requirements and the selected files
without processing anything. Each build resolves settings and today's date once,
so execution and its records use the same resolved choices throughout the run.
`coordinates(name)` returns grids inherited through a prepared dataset's inputs.

Direct functions remain available with named `dataset`, `profiles_dataset` or
`simulation_dataset` arguments. ERA5 uses separate `create_profile_dataset` and
`create_forward_calc_dataset` methods so simulations can reuse prepared profiles.

### Retrieval recipes and commands

Run these commands from your working project, using the Python environment
where openMWR is installed. pip, editable installations, and uv all work:

```bash
openMWR train munich_G5 NN_example --device cuda
openMWR evaluate munich_G5 NN_example
```

Site and retrieval names are required positional arguments. Each command
performs only the selected action using prepared datasets. Pass
`--project-dir /path/to/project` to work from another directory. The default
device is `cpu`. Each task checks its required settings and inputs before starting
the expensive computation. `train` runs optimization when any resolved training
or noise parameter contains a search definition; otherwise it trains one model.
It announces the selected mode, optimization trial budget, and resolved search
ranges. `--n-trials`, `--data-source` and `--dataset` can override optimization
settings; evaluation also accepts `--data-source` and `--dataset`.
An `optimization` section containing only a budget or reference data does not
trigger optimization.

The first run needs no mode flag. When a recipe already has a model or study,
choose explicitly:

```bash
openMWR train munich_G5 NN_example --continue
openMWR train munich_G5 NN_example --fresh
```

`--continue` adds trials to a matching optimization study. `--fresh` deletes only
this retrieval's checkpoint, progress PDF, and study before starting again;
its recipe, logs, reports, datasets, and other retrievals remain available.
Settings and required data are checked before a fresh run removes results.
Single-model training supports fresh runs; continuing neural-network weights
is not implemented.

Create `sites/<site>/retrieval/<name>/retrieval.toml`, for example:

```toml
schema_version = 1

[model]
kind = "nn"                 # nn, mlr, or spectral
mode = "zenith"             # zenith or bls (spectral uses zenith)
freqs = "dataset"           # inherit from the named training dataset
heights = "dataset"
hidden_layers = [1024, 64, 1024]
dropout_rate = 0.1
additional_input_vars = ["surface_T"]

[model.loss_weights]
# Height bounds in metres, inclusive.
T = { upper_bounds = [10000], values = [1.0, 0.5] }
rh = { upper_bounds = [10000], values = [1.0, 0.5] }
lwc = { upper_bounds = [10000], values = [1.0, 0.5] }
lwp = 1.0
iwv = 1.0

[model.norm_functions]
TB = "z_norm"
surface_T = "z_norm"
T = "z_norm"
rh = "z_norm"
lwc = "log_norm"
lwp = "log_norm"
iwv = "z_norm"

[data]
dataset = "training"        # named split dataset
sources = ["03715"]          # station IDs and/or "era5"

[training]
seed = 42                  # integer, or "random" for an unseeded run
epochs = 500
batch_size = 512
loss_fn = "MAE"
lr_height = { low = 0.0001, high = 0.01, log = true }
lr_low = 1e-7
epoch_max_lr = 50
weight_decay = 1e-5

[input_noise]
method = "standard"         # explicit noise values; or omb for derived TB noise
TB = 0.5                    # all frequencies; per-channel overrides also work
surface_T = 0.5
# "TB_22.24" = 0.4

[optimization]
seed = 42                  # Optuna sampling seed, independent of training
days_of_month = { first = 1, last = 20 }
score_weights = { T = 1.0, rh = 0.1 }
n_trials = 100
data_source = "radiosonde"
dataset = "analysis_radiosonde"

[evaluation]
observation_height_range = [0, 10000]  # profile bounds in metres
test_height_range = "model"           # full model grid, or explicit bounds
observations = true
test_data = true
data_source = "radiosonde"
dataset = "analysis_radiosonde"
days_of_month = { first = 21, last = 31 }  # inclusive; 1 to 31 selects all days
station_id = "03715"        # explicit radiosonde station identifier
```

The retrieval folder supplies the model name. `data.dataset` selects a named
split dataset, which owns the fitting and test period; omit `data.sources` to
use all its sources. There is no separate retrieval training cutoff. For example,
`training_r17.toml` can use `kind = "split"` and
`[inputs] simulation = "simulation_r17"` to prepare inputs for an R17 retrieval.
Optimization/evaluation/OMB/bias sections explicitly select their reference
`data_source` and named analysis `dataset`.

Replace the example's `lr_height` range with a number to train one model.
Model recipes specify `freqs` (GHz), `heights` (m above the instrument), `angles`
(degrees, BLS only), `additional_input_vars` (which may be empty), and
`norm_functions` for every active input/output variable. Grids accept numeric
lists, named presets, or `"dataset"` to inherit the training dataset's grid.
NN models specify `hidden_layers`; spectral models specify `hidden_layer_size`.
Dropout is off when omitted.
NN profile models require explicit `loss_weights` in both recipes and Python.
Each output has a scalar weight, a per-height list for profiles, or height bands
as in the example above. `upper_bounds` are increasing heights in metres and
inclusive: `[10000]` with values `[1.0, 0.5]` gives weight 1 through 10 km and 0.5
above. There is one more value than upper bounds. `[2000, 10000]` with values
`[1.0, 0.5, 0.25]` reproduces the former BLS weighting. For uniform or independent
variable weights, use scalars:

```toml
[model.loss_weights]
T = 1.0
rh = 0.5
lwc = 0.2
lwp = 1.0
iwv = 1.0
```

Python passes the same weight dictionary. There are no named presets or implicit
mode-based weights for new models. Weights are finite, nonnegative
and include every output; at least one must be positive. They apply to fitting;
optimization applies the same height weights before its separate variable weights. New checkpoints store their resolved
numeric values. Historical checkpoints retain their original weighting when
loaded, without rewriting their files. MLR and spectral models do not accept
profile loss weights.
BLS noise can use `vary_input_noise_with_angle` or
`vary_input_noise_with_angle_linear`; their angle/factor settings belong in
`input_noise`. The optional `bias_correction` table requires `data_source`,
`dataset`, `factor`, and `days_of_month`; optional `sources` and `specific_freqs` restrict it.
Training settings use the arguments of `add_standard_training_hyper_params`.
MLR recipes require `seed` and `repeats_for_linear_regression` in `training`, alongside
their input-noise settings; NN-only settings such as weight decay/learning rate
and model hidden layers/dropout are rejected. NN/spectral recipes reject the
MLR repeat setting. Explicit Python searches are checked for model applicability
as well; ordinary Python default hyperparameter dictionaries remain supported.
Training and noise parameters accept fixed values or searches in the same
place; seeds are fixed choices. Each active parameter needs a fixed value, or explicit `low`/`high`,
optional `step`, and optional `log = true` for a search.
Integer training parameters use integer ranges; learning rates, weight decay,
bias coefficients, and noise use floating-point ranges. Discrete searches use
`batch_size = { choices = [256, 512, 1024] }`; loss functions can also use choices.
Searches in model structure, loss weights, bias correction, or study controls are unsupported
and produce an error.

For example, `weight_decay = { low = 1e-8, high = 1e-4, log = true }`
spells out a logarithmic search. `{ optimize = true }` is rejected.
Hand-written Python workflows can still opt into recommended settings with
`add_standard_training_hyper_params()`, `add_input_noise_params()` and
`default_search_spaces()`. The first two read the matching bundled fixed-value
example; `add_input_noise_params(optimization=True)` and `default_search_spaces()`
read its optimization example. Explicit arguments override those recommendations;
`TB` applies to all selected frequencies, with per-frequency entries taking precedence.
The helpers need no site configuration or prepared datasets and accept only active
parameters. Inputs absent from the example, such as `years_since_1970` or optional
angle-dependent noise, require explicit values or ranges. Custom model types without
a bundled example can use `change_hyper_params()` to supply settings explicitly.
Project recipes still require their scientific settings and do not fill omissions
from bundled recommendations.
Python workflows can also set just the training parameters their model uses:
MLR requires `seed` and `repeats_for_linear_regression`; NN and spectral models require the
neural training settings shown above.
For `MAEPlusBiasLoss` or `MSEPlusBiasLoss`, also specify `training.bias_weight`
(a number or search). Other loss functions omit it.

A fixed training seed controls recipe model initialization, shuffling and added
noise. `"random"` leaves the random generators unseeded. For fully manual Python
model construction, call `openMWR.config.apply_seed(seed)` before constructing the
model as well as setting `training.seed`. Optuna sampling and trial
initialization/noise each use their own seed plus the stored trial number
(modulo 2**32). With fixed seeds and the same trial history, continued sequential
studies match uninterrupted runs. Exact numerical reproducibility also depends
on the backend.

Every recipe selects its noise method: `standard` requires explicit values or
searches for every active input; `omb` derives TB noise from observations and
requires explicit noise for auxiliary inputs. To search noise values, replace the example's
`input_noise` section with:

```toml
[input_noise]
method = "standard"

TB = { low = 0.0, high = 1.5 }  # independent search for each frequency
surface_T = 0.5
```

The `train` command automatically optimizes this recipe. Trial counts and
reference data stay in `optimization`. Alternatively, estimate TB noise from current
prepared observations using the existing OMB method:

```toml
[input_noise]
method = "omb"
data_source = "era5"         # radiosonde or era5
dataset = "analysis_era5"
days_of_month = { first = 1, last = 20 }
factor = 2.0
surface_T = 0.5
# sources = ["03715"]       # optional; defaults to the training stations
```

Explicit TB noise entries can override OMB estimates with numbers or searches;
the OMB scaling factor is fixed.

Training with `--fresh` replaces the named `model.pth`, appends to `training.log`, and
records effective settings and available software identity in the checkpoint.
Optimization uses `optimization.log` and `optimization.db` beside the named
retrieval's `retrieval.toml` and `model.pth`. Each retrieval owns its own SQLite
database containing its study; copying or deleting the folder includes its history.
Studies resume only with matching experiment definitions, including model settings,
active training/noise parameters and applied bias corrections; use a
new retrieval name or `--fresh` when changing an experiment.
The saved best checkpoint must also remain present. Study compatibility covers the settings, not changes to dataset
contents. Studies without recorded experiment settings require a fresh run or a
new retrieval name before optimization can continue; their histories remain readable.

Evaluation writes dated portable HTML files under the retrieval's `evaluation/`
directory. Set `evaluation.observations = false` or `evaluation.test_data = false`
to select its inputs. Optimization, OMB noise, bias correction and evaluation
all accept the same optional `start_date`, `end_date` and
`days_of_month = { first = 1, last = 20 }` settings in their respective sections.
Filters intersect: date bounds include the entire start/end day, and the monthly
range is inclusive and repeats in every selected month. Omitted bounds impose
no restriction; omit `days_of_month` or set first/last to 1/31 for all days.
For example, optimize on one year and evaluate on the next:

```toml
[optimization]
# Other optimization settings remain as configured.
start_date = "2024-01-01"
end_date = "2024-12-31"

[evaluation]
# Other evaluation settings remain as configured.
start_date = "2025-01-01"
end_date = "2025-12-31"
```

Analysis datasets retain all matched dates; these selections apply when using
them. `train` (for optimization) and `evaluate` can override the recipe selection
with `--start-date` / `-s`, `--end-date` / `-e`, and `--days-of-month FIRST LAST`.
These flags do not change the fitting or simulated test period. Python
`optimize()`, `make_omb_analysis()` and `evaluate_retrieval()` take the same
selection keywords. Profile evaluations require explicit
`observation_height_range` and `test_height_range` for their enabled inputs.
Each accepts `[minimum, maximum]` in metres or `"model"` for the whole model grid.
CLI overrides are `--observation-height-range MIN MAX` and
`--test-height-range MIN MAX`; either also accepts `model`. Checkpoints without
recipes need their day selection and both enabled profile ranges supplied
explicitly. Spectral evaluations have no profile height range.

Optimization requires explicit `score_weights`, separate from fitting loss
weights: the score sums each selected variable's weight times its mean RMSE
(model height weights apply where present). Omitted or zero-weight variables do
not contribute. Starter profile recipes use `{ T = 1.0, rh = 0.1 }`; spectral
recipes use `{ TB = 1.0 }`. Logs and saved checkpoints record the reference, day
selection and score. Reports describe actual selected days and their overlap
with the checkpoint's recorded optimization days.
Select radiosonde observations by `station_id` or `--station-id`. A single station
can be inferred; multiple stations require an explicit ID.
Inference and test-data selection use saved checkpoint settings even after
recipe edits.

### Daily prediction

Apply saved checkpoints directly to local daily HATPRO measurement files:

```bash
openMWR predict munich_G5 NN_opt MLR_opt --date 2026-10-20
openMWR predict munich_G5 --start-date 2026-10-01 --end-date 2026-10-20
```

Site is required; retrieval names are optional positional arguments. Omitting
retrieval names uses the optional site.toml selection:

```toml
[prediction]
retrievals = ["NN_opt", "MLR_opt", "NN_BLS_opt_var_ang_lin_3"]
```

Explicit names replace this selection. Models determine whether zenith or BLS
input is needed; training recipes and prepared training datasets are not read.
Dates are UTC and ranges include both endpoints. Prediction does not download
data or build dataset dependencies. `--project-dir` defaults to the current
directory and `--device` to cpu.

The output root is `retrieval_output_dir` from site.toml, or
`data/sites/<site>/predictions/` when omitted. `--output-dir` overrides it;
relative paths resolve from the project root. Files are
`YYYY/MM/YYYYMMDD_retrieval.nc` and `YYYY/MM/YYYYMMDD_retrieval_bls.nc`.
Reprocessing replaces the selected day's files. Zenith files retain the model
dimension; a single BLS retrieval retains the historical scalar model coordinate.

Zenith input uses one-minute averages and results use the existing centered
five-sample smoothing. BLS results stay unsmoothed. Profile outputs include the
existing derived quantities; spectral models produce TB predictions. Missing
measurements or input validation failures are reported per date/group. Other
requested dates/groups still run and usable output is saved; the command exits
with status 1 if any failed.
Previously saved files for failed/unselected groups are left in place.
Configuration, computation and output-write errors stop the command.

The command uses the same Python API:

```python
from openMWR.predict import PredictionWorkflow, load_predictions

workflow = PredictionWorkflow(".", "munich_G5", ["NN_opt", "MLR_opt"])
result = workflow.predict("2026-10-20")  # No writes; models are loaded once.
paths = workflow.save(result)
if result.zenith is not None:
    pred = result.zenith.sel(model="NN_opt")

# Read outputs for plotting without loading checkpoints or rerunning inference.
saved = load_predictions(".", "munich_G5", "2026-10-20")
```

Results expose `zenith`, `bls`, `observations`, `observations_bls` and `errors`.
Unrequested/unavailable groups are None. Missing-input days always return a
result; check `result.errors`, including when both groups are unavailable.
Configuration and computation errors still raise exceptions.
Saved files contain predictions, not RPG comparison observations. Python can
select `smooth=False` or `import_retrieval_data=True`; the latter includes local
RPG products in returned observations. Plot composition remains Python using
`TimeSeriesCreator`/`SkewT`. Low-level `run_model` and `run_model_list` retain their
names and now live in `openMWR.predict`, together with normalization helpers.
`load_radiosonde_profile` in `openMWR.plot` selects one launch from its yearly
source; pass it as `SkewT.plot_radiosonde(station_id, data=profile)` to reuse it
across figures without repeated reads.

### Custom Python work

Recipes and commands use the same `TrainingWorkflow` as ordinary scripts:

```python
from openMWR.train import TrainingWorkflow

workflow = TrainingWorkflow.from_recipe(
    project_dir=".", site="munich_G5", retrieval="NN_example", device="cuda"
)
# Inspect/change workflow.training_data or workflow.validation_data here.
# Resolve the example's learning-rate search to a fixed value for one model.
workflow.change_hyper_params(epochs=1000, lr_height=0.002)
model = workflow.train()
```

Python scripts explicitly choose `workflow.train()` for one model or
`workflow.optimize()` for a study, independently of the command's automatic
selection. `train()` requires fixed values; replace any search definitions with
fixed values before calling it. `workflow.requires_optimization` reports whether
the current parameters still contain searches. All workflows use
`train(run_mode="fresh")`, `optimize(run_mode="fresh")`, or
`optimize(run_mode="continue")` when results already exist. A first run needs no
run mode. Optimization requires a search definition and applies the same study
and saved-model checks to both setup paths. Direct model/workflow construction and the existing preparation
functions remain available. `openMWR.evaluation.evaluate_retrieval` returns numerical results;
`write_report` saves them separately. Shared comparison/report functionality is
also available through `analyze_models` with explicit `SiteSettings`.


## Installation

Clone the repository:

```bash
git clone https://github.com/DSchlebruegge/openMWR.git
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the openMWR package as editable:

```bash
cd openMWR
pip install -e .
```

For more details on installation, please refer to the [Installation Guide](https://dschlebruegge.github.io/openMWR/user_guide/installation.html).

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the [LICENSE](LICENSE) file for details.

## Attribution

This project contains a PyTorch port/translation of code derived from SatCloP/pyrtlib (GPL-3.0).
Upstream: https://github.com/SatCloP/pyrtlib
