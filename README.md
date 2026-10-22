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

### Create a site

From your working-project directory:

```bash
openMWR create-site my_site
```

This writes `sites/my_site/site.toml`, six dataset recipes for a radiosonde
training chain and HATPRO/radiosonde comparisons, and three retrieval recipes under
`sites/my_site/retrieval/`: `NN_example` (zenith NN), `MLR_example` (zenith MLR),
and `NN_BLS_example` (boundary-layer-scan NN). It creates empty data
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
The generated files include no ERA5 recipes or optimization searches.
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
heights = "levels_208_30km"
cut_off_at_mwr_height = false
# heights = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 30000]
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
| `profiles` | `source`: radiosonde or ERA5 | `[preparation] heights`; radiosondes also support cutoff |
| `simulation` | `profiles` | `[radiative_transfer] backend`, `absorption_model`, `frequencies`, `angles`, optional `frequency_shift` |
| `split` | `simulation` | Required `[split] test_fraction`, `validation_fraction` (fraction of the remainder), `seed` |
| `analysis` | `observations`: HATPRO; `reference`: simulation | Optional `stations` selection |

Derived recipes inherit their input stations unless selecting a subset.
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

Effective settings and input identities are recorded in `state.json`, and each
dataset has `build.log`. Changing scientific settings or rebuilding an input
requires an explicit downstream rebuild, preventing mixed processing settings.
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
end_date_training = "2024-10-31"

[training]
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
n_trials = 100
data_source = "radiosonde"
dataset = "analysis_radiosonde"

[evaluation]
observations = true
test_data = true
data_source = "radiosonde"
dataset = "analysis_radiosonde"
days_gt_20 = true           # explicit choice; false evaluates all days
station_id = "03715"        # explicit radiosonde station identifier
```

The retrieval folder supplies the model name. `data.dataset` selects a named
split dataset; omit `data.sources` to use all its sources. For example,
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
profile optimization uses the T/RH weights in its existing score. New checkpoints store their resolved
numeric values. Historical checkpoints retain their original weighting when
loaded, without rewriting their files. MLR and spectral models do not accept
profile loss weights.
BLS noise can use `vary_input_noise_with_angle` or
`vary_input_noise_with_angle_linear`; their angle/factor settings belong in
`input_noise`. The optional `bias_correction` table requires `data_source`,
`dataset`, and `factor`; optional `sources` and `specific_freqs` restrict it.
Training settings use the arguments of `add_standard_training_hyper_params`.
MLR recipes accept only `repeats_for_linear_regression` in `training`, alongside
their input-noise settings; NN-only settings such as weight decay/learning rate
and model hidden layers/dropout are rejected. NN/spectral recipes reject the
MLR repeat setting. Explicit Python searches are checked for model applicability
as well; ordinary Python default hyperparameter dictionaries remain supported.
Each training or noise parameter accepts a fixed value or a search in the same
place. Each active parameter needs a fixed value, or explicit `low`/`high`,
optional `step`, and optional `log = true` for a search.
Integer training parameters use integer ranges; learning rates, weight decay,
and noise use floating-point ranges. Discrete searches use
`batch_size = { choices = [256, 512, 1024] }`; loss functions can also use choices.
Searches in model structure, loss weights, bias correction, or study controls are unsupported
and produce an error.

For example, `weight_decay = { low = 1e-8, high = 1e-4, log = true }`
spells out a logarithmic search. `{ optimize = true }` is rejected.
Hand-written Python workflows can still opt into recommended settings with
`add_standard_training_hyper_params()`, `add_input_noise_params()` and
`default_search_spaces()`.
Python workflows can also set just the training parameters their model uses:
MLR requires `repeats_for_linear_regression`; NN and spectral models require the
neural training settings shown above.

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
Recipe-driven studies resume only with matching experiment definitions; use a
new retrieval name or `--fresh` when changing an experiment.
The saved best checkpoint must also remain present. Study compatibility covers the settings, not changes to dataset
contents. Existing studies remain accessible through the manual Python API.

Evaluation writes dated portable HTML files under the retrieval's `evaluation/`
directory. Set `evaluation.observations = false` or `evaluation.test_data = false`
to select its inputs. Optimization retains the existing observation filter
(days ≤ 20); evaluation requires its own explicit filter. Logs show training
sources, the training cutoff and named dataset, the fitting loss, and the
separate optimization reference/score. Profile trials use the existing heightwise
temperature RMSE plus RH RMSE/10 (model height weights where present); spectral
trials use mean channel TB RMSE. These setup details are saved with new checkpoints.
Reports label all-days observations as diagnostic, including the standard
optimization/OMB day range. Days after the 20th are outside that selection;
this describes the day split, without claiming broader data independence.
Select radiosonde observations by `station_id` or `--station-id`. A single station
can be inferred; multiple stations require an explicit ID. Existing checkpoints
can be evaluated without recipes by passing `--days-gt-20` or `--all-days`.
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
the current parameters still contain searches. Recipe workflows use
`train(run_mode="fresh")`, `optimize(run_mode="fresh")`, or
`optimize(run_mode="continue")` when results already exist. Direct manual Python
workflows retain their existing default overwrite/resume behavior and also
accept explicit run modes. Direct model/workflow construction and the existing preparation
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
