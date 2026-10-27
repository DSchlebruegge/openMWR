# Create a Retrieval

This section describes how to create a retrieval using radiosonde profiles and
simulated brightness temperatures. The main steps are:

1. Create the site configuration
2. Download and prepare the training data
3. Run the forward calculation
4. Separate training, validation and test datasets
5. Train the retrieval model
6. Compare the retrieval with real measurements

Follow [Installation](installation.md) first. Run the commands from your working
project, with its virtual environment activated. The example uses `my_site` as
the site name and Munich settings. Replace these with your own.

The commands read TOML files, called *recipes*, which describe datasets and
retrievals. The first step creates these files.

## Create the Site Configuration

```bash
openMWR create-site my_site
```

This creates site settings, dataset recipes and retrieval examples. They contain
example settings to review for your instrument and application:

```text
grids.toml
sites/my_site/
    site.toml
    datasets/
        radiosondes.toml
        profiles.toml
        simulation.toml
        training.toml
        hatpro.toml
        analysis_radiosonde.toml
    retrieval/
        NN_example/retrieval.toml
        ...
    evaluations/
        radiosonde.toml
        spectral.toml
data/sites/my_site/
```

The TOML snippets below show sections to edit in these generated files.
Keep the other settings unless instructed otherwise.

`sites/` contains configuration and trained retrievals. `data/` contains
downloaded and generated datasets. `grids.toml` defines named height, frequency
and elevation-angle grids shared by the sites in this project.
Creating a site does not download or process data and refuses to overwrite an
existing site.

Uncomment and fill in the instrument settings at the top of
`sites/my_site/site.toml`, before `[prediction]`. For the G5 HATPRO in Munich:

```toml
gen = "G5"
mwr_height = 550
latitude = 48.148
longitude = 11.573
```

`gen` is the instrument generation (`G4` or `G5`). `mwr_height` is its height in
metres above sea level. Coordinates are in decimal degrees. Local measurement
paths are needed later for evaluation and prediction.

## How the datasets fit together

The diagram shows the available dataset types and how their inputs connect.
The names below each type correspond to recipe filenames without `.toml`.
`create-site` supplies the radiosonde path. [ERA5](#using-era5) is an alternative
that you can add later.

```{figure} ../../_static/dataset_workflow.svg
:alt: Radiosondes or ERA5 feed prepared profiles, then simulations. Simulations feed training splits and analysis datasets. Analysis also takes HATPRO measurements.
:width: 780px
:align: center

Dataset dependencies. Each prepared-profile recipe selects one source.
```

A recipe's `[inputs]` section names the datasets it uses. For example,
`simulation.toml` selects `profiles`, and `training.toml` selects `simulation`.
An analysis dataset needs two inputs: HATPRO observations and a simulation dataset
with matching reference profiles. It does not use the training split.

Several recipes can share an input. To compare two absorption models, keep one
prepared-profile dataset and create two simulation recipes with different names.
You can also prepare the same source in different ways. Each derived dataset is
saved under `data/sites/my_site/<recipe-name>/`.

`openMWR data` builds or updates the selected datasets. Adding
`--with-dependencies` follows the arrows from their sources first. Without it,
the inputs must already exist. The steps below run separately so you can inspect
each result.

## Download and Prepare the Training Data

The tutorial uses radiosondes from the German weather service (DWD). ERA5 can
also supply profiles. See [Using ERA5](#using-era5) below.

### Download radiosonde data

Choose stations that represent the atmospheric conditions at the MWR site.
The example uses Oberschleißheim near Munich. The built-in downloader reads
DWD high-resolution radiosonde data. Other providers need a separate importer.

```{figure} ../../_static/radiosonde.jpg
:alt: Radiosonde stations in Germany
:width: 400px
:align: center

Radiosonde stations in Germany (source: [DWD](https://www.dwd.de/DE/derdwd/messnetz/atmosphaerenbeobachtung/_functions/Teasergroup/radiosonden_teaser2.html))
```

|   Station Id | Name                       |   Height (m) |   Latitude |   Longitude |
|-------------:|:---------------------------|-------------:|-----------:|------------:|
|        00125 | Altenstadt                 |          760 |    47.8342 |     10.8667 |
|        00368 | Bergen                     |           69 |    52.8152 |      9.9248 |
|        01303 | Essen-Bredeney             |          150 |    51.4041 |      6.9677 |
|        01757 | Greifswald                 |            2 |    54.0967 |     13.4056 |
|        02290 | Hohenpeißenberg            |          977 |    47.8009 |     11.0108 |
|        02385 | Idar-Oberstein             |          376 |    49.6927 |      7.3263 |
|        02773 | Kümmersbruck               |          416 |    49.4283 |     11.9016 |
|        03015 | Lindenberg                 |           98 |    52.2085 |     14.118  |
|        03231 | Meiningen                  |          450 |    50.5611 |     10.3771 |
|        03254 | Meppen                     |           19 |    52.7155 |      7.3176 |
|        03631 | Norderney                  |           12 |    53.7123 |      7.1519 |
|        03715 | Oberschleißheim (Lustheim) |          484 |    48.2446 |     11.5525 |
|        04466 | Schleswig                  |           43 |    54.5275 |      9.5487 |
|        04928 | Stuttgart (Schnarrenberg)  |          314 |    48.8281 |      9.2    |
|        06202 | Oppin                      |          106 |    51.5467 |     12.0608 |

Source: [DWD](https://opendata.dwd.de/climate_environment/CDC/observations_germany/radiosondes/high_resolution/recent/sec_aero_Beschreibung_Stationen.txt)


Edit `sites/my_site/datasets/radiosondes.toml`:

```toml
stations = ["03715"]
start_year = 1990
end_year = "current_year"
```

Quote station IDs to preserve leading zeros. These years select the source
coverage, including profiles for later measurement comparisons. The training
period is selected separately below. For a first run, a shorter source period
reduces download and processing time.

```bash
openMWR data my_site radiosondes
```

Raw profiles are saved under `data/radiosondes/station_03715/` and can be reused
by other sites and recipes. The station list is inherited by the following steps.

### Prepare profiles

`sites/my_site/datasets/profiles.toml` selects the radiosonde source:

```toml
[inputs]
source = "radiosondes"

[preparation]
incomplete_profiles = "monthly_mean"
heights = "levels_208_30km"
```

With the generated settings, preparation:

- Rejects profiles with a top below 15 km above sea level or fewer than 500 valid levels.
- Estimates liquid water content (LWC) from humidity and a scaled adiabatic cloud model.
- Changes the height coordinate to height above the radiosonde station.
- Interpolates to the selected height grid.
- Completes missing upper levels using monthly means of the available profiles.
- Adds derived quantities, including liquid water path (LWP) and integrated water vapour (IWV).

Cloud estimates and monthly completion are modelling assumptions. The recipe
states the cloud thresholds and adiabatic fraction. Its comments describe the
available formulas. `incomplete_profiles = "filter"` instead rejects profiles
that do not cover the required grid. Both choices reject profiles that still
lack the fields needed for radiative transfer.

`levels_208_30km` names 208 heights up to 30 km in `grids.toml`. You can add another
named grid there or use a numeric list in the recipe. Heights are in metres.
The retrieval can use this grid or a subset of it.

To use the MWR height as the profile base instead of the radiosonde station
height, set `cut_off_at_mwr_height = true` under `[preparation]`. This cuts off
the lower profile when the instrument is above the station.

```bash
openMWR data my_site profiles
```

Prepared profiles are saved as `data/sites/my_site/profiles/03715.nc`.

## Run the forward calculation

The forward calculation simulates the microwave and infrared brightness
temperatures for each prepared profile. Review
`sites/my_site/datasets/simulation.toml`:

```toml
[inputs]
profiles = "profiles"

[radiative_transfer]
backend = "torchMWRT"
absorption_model = "R24"
frequencies = "hatpro"
angles = "hatpro_bls"
```

`torchMWRT` is the PyTorch implementation of the microwave calculation. `pyrtlib`
is an alternative backend. R24 selects the absorption model. Frequencies (GHz)
and elevation angles (degrees) refer to grids in `grids.toml`.
The supplied scan angles include zenith, so this dataset supports both zenith
and boundary layer scan retrievals. For zenith only, use `angles = [90.0]`.

The infrared calculation uses [libRadtran](installation.md#install-libradtran).
Its cloud droplet concentration and minimum effective radius are set in
`[radiative_transfer.infrared]`.

```bash
openMWR data my_site simulation
```

Results are saved as `data/sites/my_site/simulation/03715.nc`.
To control parallel processing, add `[execution]` with `processes = 4`, for
example. With torchMWRT, this controls the libRadtran workers. The microwave
calculation processes profiles in batches. Choose a worker count that fits your
available CPU cores and memory.

For channel-frequency experiments, `frequency_shift` accepts one offset per
channel under `[radiative_transfer]`. Offsets are in GHz and are subtracted from
the nominal frequencies. Copy the recipe to a new name before changing the
simulation if you want to keep both datasets.

## Separate training, validation and test datasets

`sites/my_site/datasets/training.toml` selects the profile period and assigns
whole UTC days to the three datasets. Edit its `[split]` section:

```toml
[split]
end_date = "2024-10-31"
test_days_of_month = { first = 1, last = 3 }
validation_days_of_month = { first = 15, last = 17 }
```

Here, days 1–3 of every month are used for testing, days 15–17 for validation,
and the remaining days for training. Bounds are inclusive. The same assignment
applies across stations. These are calendar groups, not fixed percentages or a
guarantee of statistical independence.

Training adjusts the model using the training group and selects the saved neural
network using validation loss. The test group is reserved for evaluation.
The `end_date` above applies to all three groups. Choose it before the observations
used for optimization and final comparison. The example reserves observations
from November 2024 onwards. Optional `start_date` limits the beginning of the pool.

```bash
openMWR data my_site training
```

The three files are stored under `data/sites/my_site/training/`, for example
`training_03715.nc`, `validation_03715.nc` and `test_03715.nc`.

### Build or update the whole chain

The preceding commands run one step at a time, using existing inputs. To run
the prerequisites automatically, use:

```bash
openMWR data my_site training --with-dependencies --dry-run
openMWR data my_site training --with-dependencies
```

The first command checks the configuration and prints the processing order
without writing or downloading. The second creates missing datasets or updates
existing ones, following `radiosondes → profiles → simulation → training`.

Running `data` again updates the selected datasets. Without
`--with-dependencies`, it uses the inputs already on disk. With that option,
source steps also check for new data. Changed processing settings require an
explicit rebuild. The error gives the command for the affected datasets.
`--rebuild` replaces only the selected dataset outputs. Raw source caches are
reused unless a source dataset itself is selected for rebuilding.

## Train the retrieval model

### Configure the model

Open `sites/my_site/retrieval/NN_example/retrieval.toml`. Its `[model]` section
selects a zenith neural network with hidden layers of 1024, 64 and 1024 neurons.
`freqs = "dataset"` and `heights = "dataset"` inherit the grids from the
`training` dataset selected under `[data]`.

Inputs are microwave and infrared brightness temperatures, surface temperature,
pressure and relative humidity, and the sine and cosine of the day of year.
Outputs are temperature, relative humidity and LWC profiles, plus LWP and IWV.
Normalization and loss weights are written explicitly in the recipe.
The network and its training use PyTorch.

### Train the model

`[training]` specifies the epochs, batch size, learning-rate schedule, loss,
weight decay and random seed. `[input_noise]` specifies the standard deviations
of noise added to the inputs during training, in each input's units. For example,
`TB = 0.5` adds noise with a standard deviation of 0.5 K to each microwave channel.
These values describe the training perturbations, not measured instrument errors.

```bash
openMWR train my_site NN_example
```

The default `--device auto` selects a usable CUDA GPU, then MPS, then CPU.
CUDA devices unsupported by the installed PyTorch build are skipped.
The selected hardware is logged at startup. Use `--device cpu`
to select the CPU explicitly.
The trained model and `training.log` are saved beside `retrieval.toml` as
`sites/my_site/retrieval/NN_example/model.pth` and `training.log`.

To try different settings while keeping a model, copy its recipe into a new
retrieval directory and use that name in the command. To replace the current
model deliberately, run `train` with `--fresh`.

## Compare the retrieval with real measurements

An analysis dataset pairs MWR measurements with reference profiles and their
simulated brightness temperatures. It is used for measurement comparisons and
can also be used for optimization. It is not needed for the fixed-value training
above.

### Prepare measurements and matching profiles

Fill in the local input path and measurement start at the top of `site.toml`,
before `[prediction]`:

```toml
hatpro_data_dir = "/path/to/hatpro-data"
mwr_measurement_start_date = "2024-11-01"
```

The input root must contain daily RPG files in folders such as
`Y2024/M11/D21/`, named `241121.BRT.NC` or `241121.BRT`.
By default, openMWR prefers NetCDF for each observation product and uses binary
when that NetCDF file is absent. Set `hatpro_format = "netcdf"` or `"binary"`
in `site.toml` to require one observation format for both datasets and prediction.
Binary input supports BRT, BLB, MET and IRT. Files marked as local time also need
`hatpro_time_offset_hours`, the fixed local time minus UTC in hours.

Set `rpg_retrieval_exists = true` if the instrument's own NetCDF retrieval
products are also present and should be included for comparison. These products
require NetCDF even when observations use binary. Missing or unreadable products
are logged and omitted. Available measurements and other comparison products
are retained.
`datasets/hatpro.toml` uses the site start and `end_date = "today"`. Set explicit
dates there to limit the measurement period.

`datasets/analysis_radiosonde.toml` pairs `hatpro` with `simulation`.
The example allows a time difference of at most 10 minutes and excludes rain.
To keep the full model grid available for optimization, change its `[analysis]`
section to:

```toml
[analysis]
max_time_difference_minutes = 10
exclude_rain = true
height_range = [0, 30000]
required_observations = ["TB", "surface_T", "surface_p", "surface_rh"]
```

This range must cover the model's output heights for profile optimization.
Evaluation selects a smaller range separately. Build the dataset:

```bash
openMWR data my_site analysis_radiosonde --with-dependencies
```

This also updates its source profiles and simulations. Matched datasets are
saved as `data/sites/my_site/analysis_radiosonde/zenith.nc` and `bls.nc`.
The reference profiles must cover the selected observation period.

### Evaluate the model

Open `sites/my_site/evaluations/radiosonde.toml`. This comparison recipe selects
the saved models and reference data independently of their training recipes.
Set `station_id` and `test.sources` to your radiosonde station IDs. Add the
observation start date under `[observations]`:

```toml
start_date = "2024-11-01"
```

The example compares `NN_example` with observations on days 21–31 up to 10 km,
and with simulated test data up to 30 km. To compare more trained models, edit
the list at the top of the file:

```toml
retrievals = ["NN_example", "MLR_example"]
```

```bash
openMWR evaluate my_site radiosonde
```

The command prints a new directory under `sites/my_site/evaluations/radiosonde/`.
Open its `report.html` in a browser. `metrics.nc` contains the same numerical
results for [Python analysis](python_usage.md#evaluate-retrievals).

All selected models use the explicitly chosen test dataset. Zenith and BLS
models have separate report sections. Scores compare common finite samples
within each group for each variable and height. Coverage tables show how many
profiles each model could process and how many remain for scoring. Model grids
must agree within each group over the selected evaluation height range.

After training a BLS model, add it to `retrievals` to include its comparison.
Optionally add `bls_baseline = "NN_example"` at the top of the evaluation recipe
to show that selected zenith model alongside the BLS models. Its scores are
recalculated on the BLS comparison's common cases.

Measurement comparison and simulated test performance answer different questions.
The latter does not measure performance on the instrument's observations.
Changing the test dataset does not make it independent of training or tuning.
Keep evaluation data outside those choices. The report shows recorded training
settings and flags overlap with recorded optimization dates.

Without local observations, remove `[observations]` and its settings to evaluate
only `[test]`. Add `--retrieval NN_example` to evaluate one selected model.
See [Retrieval Analysis](retrieval_analysis.ipynb) for custom plots.

## Optimize hyperparameters and input noise

The generated `NN_opt_example` recipe demonstrates parameter searches. A fixed
value is replaced by an explicit range in the same place, for example under
`[training]`:

```toml
weight_decay = { low = 1e-8, high = 1e-4, log = true }
```

Its `[input_noise]` section also contains search ranges. `[optimization]`
sets the trial count, sampling seed, analysis dataset and scoring weights.
The example score weights temperature RMSE by 1.0 and relative-humidity RMSE by
0.1, with the model's height weights applied across its full output grid.
Here that includes monthly-completed upper levels. Review this objective and
the output grid for your use.

Add `start_date = "2024-11-01"` under `[optimization]` in the retrieval recipe.
The example uses observation days 1–20 for optimization. The separate
`evaluations/radiosonde.toml` selects days 21–31 for evaluation.
Keep this selection out of parameter tuning.

```bash
openMWR train my_site NN_opt_example
openMWR evaluate my_site radiosonde --retrieval NN_opt_example
```

`train` starts an Optuna study whenever the recipe contains a search. It saves
the best model as `model.pth`, with `optimization.db` and `optimization.log` in
the same retrieval folder. To add 20 trials to a compatible existing study:

```bash
openMWR train my_site NN_opt_example --continue --n-trials 20
```

Use a new retrieval name for a changed experiment, or `--fresh` to replace its
model and study. Continuation checks the experiment settings. It does not freeze
the underlying datasets. Keep those unchanged when continuing a study.

## Other retrievals

The generated examples also include `MLR_example` for multilinear regression,
`NN_BLS_example` for boundary layer scans, and `SPC_example` for spectral
consistency. Each has an `_opt_example` counterpart. They use the same commands
with their own retrieval name. Scan models need simulations at their input angles.

Next, [run and plot a retrieval](run_retrieval.ipynb), set up
[regular prediction and updates](run_operational.md), or use the same workflows
[from Python](python_usage.md).

See [Recipe Settings](recipe_settings.md) for further configuration choices.

## Using ERA5

ERA5 can provide profiles for training or measurement comparison. It requires
CDS access configured for `cdsapi`. Follow the
[CDS API setup instructions](https://cds.climate.copernicus.eu/how-to-api).

Create `sites/my_site/datasets/era5.toml` with this complete source recipe:

```toml
kind = "era5"
start_date = "2024-01-01"
end_date = "today"
delay_days = 5
hour_interval = 6
```

Create `datasets/profiles_era5.toml` with:

```toml
kind = "profiles"

[inputs]
source = "era5"

[preparation]
heights = "levels_208_30km"
```

ERA5 preparation uses its atmospheric fields. The radiosonde cloud and completion
settings do not apply. Copy `simulation.toml` to `simulation_era5.toml` and
change `[inputs].profiles` to `"profiles_era5"`. Copy `training.toml` to
`training_era5.toml` and change `[inputs].simulation` to `"simulation_era5"`.
Choose source and split dates that cover the intended training period.

```bash
openMWR data my_site training_era5 --with-dependencies
```

Copy a retrieval recipe into a new retrieval folder and set its `[data].dataset`
to `"training_era5"`. For measurement comparison, copy
`analysis_radiosonde.toml` to `analysis_era5.toml`, set `[inputs].reference` to
`"simulation_era5"`, and build it with dependencies. Copy the evaluation recipe
to `evaluations/era5.toml`. Under `[observations]`, remove `station_id` and select
`data_source = "era5"` and `dataset = "analysis_era5"`. Optimization selects these
same settings in the retrieval's `[optimization]` section. ERA5 comparisons are
comparisons with a reanalysis.
