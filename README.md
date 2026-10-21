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
refresh_years = 2
```

```toml
# profiles_standard.toml
kind = "profiles"

[inputs]
source = "radiosondes"

[preparation]
cut_off_at_mwr_height = false
# heights = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 30000]
# scale_rh_clear_sky = true
# scale_rh_clear_sky_seed = 42
```

```toml
# rt_r17.toml
kind = "simulation"

[inputs]
profiles = "profiles_standard"

[radiative_transfer]
backend = "torchMWRT"
absorption_model = "R17"

[execution]
processes = 12
```

Copying a recipe to a new filename creates a separate dataset identity. Its name
does not imply settings. Several simulations can share profiles; several profile
preparations can share the same raw source.

```bash
python -m openMWR data build --site munich_G5 --dataset rt_r17
python -m openMWR data build --site munich_G5 --dataset rt_r17 --with-dependencies
python -m openMWR data build --site munich_G5 --dataset rt_r17 --with-dependencies --dry-run
python -m openMWR data build --site munich_G5 --dataset rt_r17 --rebuild
```

`build` creates missing output or updates existing output. Without
`--with-dependencies`, inputs must already exist. With it, prerequisites run once
in dependency order. Select several targets with `--dataset NAME NAME`.
`--dry-run` checks recipes and shows the order without writing or contacting
sources. `--rebuild` regenerates only explicitly selected outputs; dependencies
receive ordinary updates. Rebuilding a source refreshes its configured raw
coverage while preserving caches shared with other recipes.

Supported recipe kinds and input roles are:

| Kind | Inputs | Settings |
| --- | --- | --- |
| `radiosonde` | None | `stations`, `start_year` (1990), optional `end_year`, `refresh_years` (2) |
| `era5` | None | Optional `start_date`/`end_date`, `hour_interval` (6), `delay_days` (5) |
| `hatpro` | None | Optional dates; `[observations] import_retrieval_data` defaults to the site setting |
| `profiles` | `source`: radiosonde or ERA5 | `[preparation] heights`; radiosondes also support cutoff and RH scaling |
| `simulation` | `profiles` | `[radiative_transfer] backend`, `absorption_model`, `frequencies`, `angles`, optional `frequency_shift` |
| `split` | `simulation` | `[split] test_fraction` (0.1), `validation_fraction` (0.1 of the remainder), `seed` (42) |
| `analysis` | `observations`: HATPRO; `reference`: simulation | Optional `stations` selection |

Derived recipes can select a subset of input `stations`. Grids default to the
existing standard heights (m), frequencies (GHz) and elevation angles (degrees).
RT defaults remain R20 for radiosondes and R17 for ERA5. RH scaling defaults to
10% of clear profiles multiplied by 0.2; optional `scale_rh_clear_sky_fraction`
and `scale_rh_clear_sky_factor` override those values. Each profile recipe owns
its choices. `execution.processes` controls workers for ERA5 preparation, RT or
HATPRO; radiosonde preparation is serial.

Source dates are inclusive. ERA5/HATPRO start at the site's measurement start
unless overridden; ERA5 ends five days behind today by default. Its cache key
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
data.build("rt_r17")
data.build(["analysis_radiosonde", "analysis_era5"], with_dependencies=True)
```

Direct functions remain available with named `dataset`, `profiles_dataset` or
`simulation_dataset` arguments. ERA5 uses separate `create_profile_dataset` and
`create_forward_calc_dataset` methods so simulations can reuse prepared profiles.

### Retrieval recipes and commands

Run these commands from your working project, using the Python environment
where openMWR is installed. pip, editable installations, and uv all work:

```bash
python -m openMWR train --site munich_G5 --retrieval NN_example --device cuda
python -m openMWR evaluate --site munich_G5 --retrieval NN_example
```

Each command performs only the selected action using prepared datasets. Pass
`--project-dir /path/to/project` to work from another directory. The default
device is `cpu`. Each task checks its required settings and inputs before starting
the expensive computation. `train` runs optimization when any resolved training
or noise parameter contains a search definition; otherwise it trains one model.
It announces the selected mode, optimization trial budget, and resolved search
ranges. `--n-trials` and `--data-source` can override optimization settings.
An `optimization` section containing only a budget or reference data does not
trigger optimization.

The first run needs no mode flag. When a recipe already has a model or study,
choose explicitly:

```bash
python -m openMWR train --site munich_G5 --retrieval NN_example --continue
python -m openMWR train --site munich_G5 --retrieval NN_example --fresh
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
hidden_layers = [1024, 64, 1024]
dropout_rate = 0.1

[data]
dataset = "training"        # named split dataset
sources = ["03715"]          # station IDs and/or "era5"
end_date_training = "2024-10-31"

[training]
epochs = 500
batch_size = 512
loss_fn = "MAE"
lr_height = { low = 0.0001, high = 0.01, log = true }
# weight_decay = { optimize = true }  # use the default search space

[input_noise]
method = "standard"         # standard or omb; omitted noise values stay fixed
TB = 0.5                    # all frequencies; per-channel overrides also work
# "TB_22.24" = 0.4

[optimization]
n_trials = 100
data_source = "radiosonde"

[evaluation]
data_source = "radiosonde"
days_gt_20 = true           # explicit choice; false evaluates all days
station_id = "03715"        # explicit radiosonde station identifier
```

The retrieval folder supplies the model name. `data.dataset` selects a named
split dataset; omit `data.sources` to use all its sources. For example,
`training_r17.toml` can use `kind = "split"` and
`[inputs] simulation = "rt_r17"` to prepare inputs for an R17 retrieval.
Optimization/evaluation/OMB/bias sections accept `dataset` to select a named
analysis dataset; the default is `analysis_radiosonde` or `analysis_era5`,
according to `data_source`.

Replace the example's `lr_height` range with a number to train one model.
Unspecified scientific settings use the existing Python defaults. Optional model
settings include `freqs` (GHz), `heights` (m above the instrument), `angles`
(degrees, BLS only), `additional_input_vars`, and `norm_functions` (a complete
variable-to-normalization mapping). Spectral models use `hidden_layer_size`.
BLS noise can use `vary_input_noise_with_angle` or
`vary_input_noise_with_angle_linear`; their angle/factor settings belong in
`input_noise`. The optional `bias_correction` table accepts `data_source`,
`sources`, `specific_freqs`, and `factor`, reusing the existing OMB correction.
Training settings use the arguments of `add_standard_training_hyper_params`.
MLR recipes accept only `repeats_for_linear_regression` in `training`, alongside
their input-noise settings; NN-only settings such as weight decay/learning rate
and model hidden layers/dropout are rejected. NN/spectral recipes reject the
MLR repeat setting. Explicit Python searches are checked for model applicability
as well; ordinary Python default hyperparameter dictionaries remain supported.
Each training or noise parameter accepts a fixed value or a search in the same
place. Omission uses its fixed default; a number fixes its value. Use
`{ optimize = true }` to request the parameter's default search space, or give
`low`/`high`, optional `step`, and optional `log = true` for a custom range.
Integer training parameters use integer ranges; learning rates, weight decay,
and noise use floating-point ranges. Discrete searches use
`batch_size = { choices = [256, 512, 1024] }`; loss functions can also use choices.
Searches in model structure, bias correction, or study controls are unsupported
and produce an error.

Weight decay's default search space is `1e-8`–`1e-4`, logarithmic; its omitted
fixed default remains `1e-5`. Noise uses the existing input-specific search
ranges. Other training parameters require explicit ranges or choices when no
default search space is defined; requesting a missing default produces an error.
Python workflows use the same definitions through `workflow.default_search_spaces()`.

Every recipe selects how unspecified noise values are obtained: `standard` uses
standard fixed noise and `omb` estimates TB noise from observations. Optimization
is always explicit on individual parameters. To search noise values, replace the example's
`input_noise` section with:

```toml
[input_noise]
method = "standard"

# Optional overrides use noise names without the input_noise_ prefix.
TB = { optimize = true }     # default range, independent search for each frequency
surface_T = 0.5              # fixed; other omitted inputs also keep fixed defaults
# surface_p = { low = 0.2, high = 1.0 }  # optional custom search
```

The `train` command automatically optimizes this recipe. Trial counts and
reference data stay in `optimization`. Alternatively, estimate TB noise from current
prepared observations using the existing OMB method:

```toml
[input_noise]
method = "omb"
data_source = "era5"         # radiosonde or era5
factor = 2.0                # optional scaling, default 1
# sources = ["03715"]       # optional; defaults to the training stations
```

Other input variables retain standard noise. Explicit noise entries override the
OMB estimates/defaults with numbers or searches; the OMB scaling factor is fixed.

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
python -m openMWR predict --site munich_G5 --retrieval NN_opt MLR_opt --date 2026-10-20
python -m openMWR predict --site munich_G5 --start-date 2026-10-01 --end-date 2026-10-20
```

Omitting `--retrieval` uses the optional site.toml selection:

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
