# Recipe Settings

Start with [Create a Retrieval](create_retrieval.md) for the workflow and generated
examples. This page describes further choices when editing those recipes.
Scientific settings must be explicit. Optional processing is off unless enabled.
Use `openMWR COMMAND --help` for command options and their short aliases.

## Site settings

`site.toml` contains instrument and location settings. openMWR checks their names,
types and ranges when loading a workflow. Required settings depend on the task.
For example, downloading radiosondes needs no instrument coordinates, while ERA5
needs latitude and longitude. Fill in the commented settings created by
`create-site` before the relevant step.

A workflow keeps one validated copy of these settings throughout processing,
including worker processes. Create a new workflow to use edits to `site.toml`.
Project-specific settings belong in `[custom]`. They do not control openMWR's
scientific processing.

Optional calibration dates mark the time plots in OMB reports:

```toml
calibration_dates = ["2025-03-12", "2025-09-04"]
```

Put this setting at the top level of `site.toml`, before any tables. Dates are
quoted UTC dates. They are information only and do not filter data or change
training. Omit the setting if the dates are unknown.

## Grids

Height, frequency and elevation-angle grids accept numeric lists or names from
the project's `grids.toml`. Its tables are `heights` in metres, `frequencies` in
GHz and `angles` in degrees. For example, add entries to the existing tables:

```toml
[heights]
custom = [0, 100, 500, 1000, 2000, 5000, 10000, 30000]

[frequencies]
custom = [22.24, 31.4]

[angles]
custom = [90.0, 30.0, 10.0]
```

Select these with `heights = "custom"`, `frequencies = "custom"` or
`angles = "custom"` in a dataset recipe. Retrieval recipes use `freqs` instead
of `frequencies`. They also accept `"dataset"` to inherit the training dataset's
grid. All sites share the project grid file. Processing records and checkpoints
store resolved numeric values. Changing a grid used by existing datasets requires
rebuilding them.

## Dataset settings

Recipes live under `sites/<site>/datasets/`. Their filenames supply the dataset
names and `[inputs]` supplies the dependency edges.

| `kind` | Input roles | Scientific settings |
| --- | --- | --- |
| `radiosonde` | None | `stations`, `start_year`, `end_year` |
| `era5` | None | `start_date`, `end_date`, `hour_interval`, `delay_days` when using `"today"` |
| `hatpro` | None | `start_date`, `end_date`. `[observations].import_retrieval_data` inherits the site choice when omitted |
| `profiles` | `source`: radiosonde or ERA5 | `[preparation].heights`. Radiosondes also require `incomplete_profiles`, `cloud` and `quality` settings |
| `simulation` | `profiles` | `[radiative_transfer]` backend, absorption model, frequencies, angles and infrared microphysics |
| `split` | `simulation` | `[split]` test and validation day ranges, with optional pool filters |
| `analysis` | `observations`: HATPRO, `reference`: simulation | `[analysis]` matching tolerance, rain exclusion, height range and required observations |

Derived recipes inherit their source stations unless selecting a subset with
`stations`. Generated files, `state.json` and `build.log` live under
`data/sites/<site>/<name>/`. Processing records track effective settings and input
identities. Changed settings or rebuilt inputs require a downstream rebuild.

### Source coverage and updates

Source date bounds include the whole end day. `start_date = "site"` inherits
the site's measurement start. `end_date = "today"` follows the current date,
subtracting `delay_days` for ERA5. Radiosondes also accept
`end_year = "current_year"`.

Radiosondes download missing historical years and refresh recent years.
`refresh_years` defaults to 2. ERA5's raw cache is shared by location and sampling
interval. Routine ERA5 updates append after the latest cached day and leave
historical gaps and earlier sampling unchanged.

Profiles and simulations append missing timestamps. Corrections to already
processed timestamps need an explicit rebuild. Split datasets regenerate from
changed simulation inputs using their calendar rule. Analysis regenerates when
either input changes. Unchanged inputs skip derived processing.

`execution.processes` controls ERA5 preparation, RT and HATPRO workers.
Radiosonde preparation is serial. Narrowing date coverage or changing historical
HATPRO start coverage requires rebuilding. Routine HATPRO updates refresh from
the last available day.

### Profile preparation and simulation

Radiosonde `[preparation.cloud]` specifies `rh_threshold` as a fraction,
`min_temperature` in K and `adiabatic_fraction`. `"power_law"` requires `a` and `b`.
`"karstens"` and `"none"` omit both. The formulas are given in the generated recipe
comments. `"none"` keeps the fully adiabatic liquid water content.
`[preparation.quality]` specifies `min_top_height` in metres above sea level before
rebasing, and `min_levels` as distinct valid source levels before interpolation.
ERA5 uses its native cloud water and does not accept these radiosonde settings.

`[radiative_transfer.infrared]` specifies `droplet_number_density` in m^-3 and
`min_effective_radius` in micrometres. The minimum radius must be at least 2.5 and
below 60. Calculated radii at or above 60 are rejected before RT, matching
libRadtran's Hu-Stamnes cloud optics.

Analysis `height_range` is in metres above the instrument. It must cover the
heights later selected for evaluation or optimization. `required_observations`
selects the variables that must be finite for a matched sample to be retained.
See [Invalid measurements](run_operational.md#invalid-measurements) for sensor
outage declarations.

### HATPRO input formats

`site.toml` selects `hatpro_format = "auto"`, `"netcdf"` or `"binary"` for both
dataset generation and prediction. `auto` prefers NetCDF for each product and
uses binary only when that NetCDF file is absent. A corrupt selected file does
not trigger a format switch. Missing or unreadable optional MET/IRT inputs are
reported while other observations remain available.

Binary BRT, BLB, MET and IRT are decoded by MWRpy. Masking, time matching and
averaging use the same code as NetCDF. BLS uses integrated BLB scans. Binary BLB
has no explicit azimuth field. The NetCDF reader removes its extra 0-degree
surface-temperature channel. RPG retrieval products require NetCDF in every
mode when `import_retrieval_data` is enabled. Missing or unreadable comparison
products are logged and omitted independently. Their absence does not discard
observations or other available products. After daily files are combined,
unavailable comparison values remain missing, without interpolation.

Daily import skips empty zenith selections and BLS scans missing standard HATPRO
angles. `build.log` reports the reason, including missing and measured scan
angles. Measured angle labels are preserved, and extra angles are retained.
Zenith and BLS are processed independently.

Daily imports discard missing timestamps and timestamps more than 24 hours
outside the file's nominal day. This tolerance preserves midnight overlaps and
local-to-UTC shifts. The log names the file and reports the discarded count.

Binary files marked as local time require `hatpro_time_offset_hours`, a fixed
local time minus UTC offset. For example, `1` converts UTC+01:00 to UTC. UTC files
ignore it. The reader does not infer time zones or daylight-saving changes.
Changing the format or offset requires rebuilding prepared HATPRO datasets.

## Retrieval settings

`sites/<site>/retrieval/<name>/retrieval.toml` uses `schema_version = 1`.
The folder supplies the model name. `[data].dataset` selects a named split
dataset. Omit `data.sources` to use all its sources. The split recipe owns the
fitting and test period.

| `[model].kind` | Mode | Model and training choices |
| --- | --- | --- |
| `nn` | `zenith` or `bls` | `hidden_layers`, `loss_weights`, optional `dropout_rate` and neural training settings |
| `mlr` | `zenith` or `bls` | `training.seed` and explicit input noise. Neural settings and profile loss weights are not used |
| `spectral` | `zenith` | `hidden_layer_size`, at least two frequencies and neural training settings. No profile settings |

Profile retrievals specify an ordered `output_vars` list selecting from `T`, `rh`,
`lwc`, `lwp` and `iwv`. For example, edit a recipe to use
`output_vars = ["T", "rh"]` and keep only those entries in `[model.loss_weights]`
for a neural model. Only the selected targets are required in training data.

`heights` is required for T, rh or lwc profiles. Omit it for a retrieval predicting
only lwp or iwv. `additional_input_vars` lists auxiliary inputs and may be empty.
BLS models also specify `angles`. All models require `norm_functions` for every
active input and output. Supported names are `z_norm`, `std_norm`, `log_norm` and
`max_norm`. Dropout is off when omitted. Optimization score weights and evaluation
variables must refer to outputs the retrieval predicts. Derived quantities are
calculated when their required inputs are available.

MLR fits the expected squared error under independent, zero-mean input noise
directly. It uses the configured noise levels without generating random noisy
copies. In normalized coordinates, each input's noise variance penalizes its
squared regression coefficients. The intercept is unpenalized. This is a form
of [Tikhonov regularization](https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/).
The fit is deterministic for fixed data and noise levels. Noise searches
use the separate observation score below. Values selected this way are tuning
parameters and need not represent instrument noise alone.

### Loss weights

Neural profile models require weights for every output. A weight can be a scalar,
a per-height list for a profile, or height bands:

```toml
[model.loss_weights]
T = { upper_bounds = [2000, 10000], values = [1.0, 0.5, 0.25] }
rh = 1.0
lwc = 1.0
lwp = 1.0
iwv = 1.0
```

Bounds are increasing heights in metres and include their upper endpoint.
The example weights temperature by 1 through 2 km, 0.5 above 2 through 10 km,
and 0.25 above 10 km. There is one more value than upper bounds. Weights must be
finite and nonnegative, with at least one positive value. They apply to fitting.
Optimization uses the height weights before its separate variable score weights.
MLR and spectral models do not accept profile loss weights.

### Fixed values and searches

Training and input-noise parameters accept a number or a search in the same
location. For example:

```toml
[training]
weight_decay = { low = 1e-8, high = 1e-4, log = true }
batch_size = { choices = [256, 512, 1024] }
```

These are entries to edit in an existing recipe. Other required settings remain.
Ranges accept `low`, `high`, optional `step` and optional `log = true`.
Integer parameters use integer bounds. Learning rates, weight decay, bias
coefficients and noise use floating-point bounds. Loss functions can use
`choices`. Search limits must be explicit. `{ optimize = true }` is rejected.
Searches are supported only in training and input-noise parameters. Seeds are
fixed choices. Model structure, loss weights, bias correction and study controls
do not accept searches.

`MAEPlusBiasLoss` and `MSEPlusBiasLoss` also require `training.bias_weight`, as a
number or search. Other losses omit it. A search starts optimization through
`train`. An `[optimization]` section alone does not.

### Input noise and bias correction

`input_noise.method = "standard"` requires explicit values or searches for every
active input. `TB` applies to all selected frequencies, with entries such as
`"TB_22.24"` taking precedence. A `TB` search samples each channel independently.

The alternative `omb` method uses observed-minus-simulated TB scatter as training
noise. It includes background, RT and sampling errors. It does not measure
instrument noise. Replace `[input_noise]` with, for example:

```toml
[input_noise]
method = "omb"
statistic = "std"
data_source = "era5"
dataset = "analysis_era5"
days_of_month = { first = 1, last = 20 }
factor = 2.0
surface_T = 0.5
```

Specify noise for every auxiliary input used by the model. Explicit TB entries
can override the OMB estimates with numbers or searches. The OMB scaling factor
is fixed. Optional `sources` selects the reference stations. Their finite pairs
are pooled for training noise. OMB reports keep stations separate.

`statistic` is required. `std` uses total OMB standard deviation. Experimental
`residual_std` uses the scatter left after regression on the nearest lower and
upper frequency in each band. It additionally requires explicit bands in GHz:

```toml
# Entries within [input_noise].
statistic = "residual_std"
frequency_bands = [[20.0, 40.0], [50.0, 60.0]]
```

OMB reports include residual diagnostics with HATPRO groups by default. Training
requires the explicit `statistic` and `frequency_bands` choices above.

This calculation stays in kelvin and fits only finite channel pairs. Its
in-sample residual scatter can include background errors and remove correlated
measurement errors. It is an experimental augmentation choice.
Optional `max_lwp` selects pairs below that LWP in both sources, in g/m².

BLS models can enable `model.vary_input_noise_with_angle` or
`model.vary_input_noise_with_angle_linear`, but not both. Their noise values or
factor belong in `[input_noise]`. OMB-derived noise uses the zenith estimate and
these configured angle factors. A zenith comparison must be available.

Optional `[bias_correction]` requires `data_source`, `dataset` and `factor`.
`sources` and `specific_freqs` restrict it. It shares the observation date filters
below and accepts `max_lwp`. The fitted mean of observed minus simulated TB is
added to synthetic training and validation inputs. Measured prediction inputs
are used directly. BLS corrections are estimated separately by angle.
The fitted vector, coordinates, counts, selected data identity and dataset
recipes are recorded in the checkpoint. Use fitting dates separate from final
evaluation days. Review the reference and calibration period before applying
an offset. Low LWP alone does not independently establish cloud-free conditions.

### Optimization and observation selection

Search recipes require `[optimization]` with `n_trials`, `data_source`, `dataset`,
`seed` and `score_weights`. The reference is a named analysis dataset.
The score sums each selected variable's weight times its mean RMSE, using the
model's height weights where present. Omitted or zero-weight variables do not
contribute. Generated profile examples use `{ T = 1.0, rh = 0.1 }`. Spectral
examples use `{ TB = 1.0 }`. Review these choices for the intended application.

Optimization, OMB noise, bias correction and evaluation observations accept
`start_date`, `end_date` and `days_of_month = { first = 1, last = 20 }`.
Filters intersect. Date bounds include the whole end day and omitted filters
impose no restriction. Analysis datasets themselves retain all matched dates.
Keep final evaluation data outside fitting and tuning choices.

`training.seed` controls initialization and, for neural models, shuffling and
added noise. Use an integer or `"random"` for an unseeded run. MLR fitting uses
no random draws. `optimization.seed` controls Optuna sampling
separately. With fixed seeds and the same trial history, sequential continuation
matches an uninterrupted study, subject to the numerical backend.

`--continue` requires matching experiment settings, the saved best model, and the
same selected fitting, validation, observation and reference values. The check
includes custom Python preprocessing. File attributes, file locations and data
outside the selected period do not affect it. Changed data or studies without
recorded data identities require `--fresh` or a new retrieval name. `--fresh` replaces this retrieval's model, study and progress PDF
after input checks. It retains recipes, logs and datasets. It does not resume
neural network weights.

## Evaluation settings

Evaluation recipes use `schema_version = 1` and select saved models independently
of training recipes. `[observations]`, `[test]` or both may be present.
Profile `height_range` is in metres above the instrument or `"model"` for the
whole model grid. Spectral comparisons select `variables = ["TB"]` without a
height range. With several radiosonde stations, select `observations.station_id`.

Zenith and BLS models form separate groups, each scored on common finite pairs
per variable and height. A `bls_baseline` must name a selected zenith profile
model. It is rescored on the BLS group's common cases. A `--retrieval` override
must include its name to retain it.

`observations.include_rpg = true` adds the instrument's retrieval products.
The BLS product contributes temperature only. Both groups use zenith RPG for
relative RMSE, recalculated on each group's common pairs. BLS-only comparisons
therefore also need the zenith analysis file. Mean retrieved LWP compares shared
finite predictions without an LWP reference. No common pairs gives an unavailable
score.

Each run saves `report.html` and `metrics.nc` together in a new directory. Results
record effective settings, input and checkpoint hashes and software identity.
All report figures are embedded SVG vector graphics.
Changing a retrieval recipe does not change a saved model's inference settings.
