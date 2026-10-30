# Using Python

The commands and Python interface use the same dataset, training, evaluation
and prediction workflows. Start from the recipes created in
[Create a Retrieval](create_retrieval.md), then use Python to inspect data or
customize a calculation. Run scripts and notebooks from the
working project with its environment.

## Build datasets

```python
from openMWR.datasets import DatasetWorkflow

project_dir = "."
site = "my_site"

datasets = DatasetWorkflow(project_dir, site)
datasets.build("training", with_dependencies=True)
```

This is the Python equivalent of `openMWR data my_site training --with-dependencies`.
`dry_run=True` checks the operation without processing. `rebuild=True` rebuilds
the selected outputs. `datasets.files("profiles")` returns the prepared file paths
for inspection with xarray.

Direct RT calls also require explicit `angles`, `absmdl` and `RT_model`.
For example, choose `absmdl="R24"` and `RT_model="torchMWRT"`. The named dataset
workflow reads these choices from the simulation recipe. `torchMWRT.RTModel`
requires `angles` and `absmdl` too.

## Train from a recipe

Copy the tutorial recipe to a new retrieval name to keep the existing model:

```bash
mkdir sites/my_site/retrieval/NN_python
cp sites/my_site/retrieval/NN_example/retrieval.toml sites/my_site/retrieval/NN_python/retrieval.toml
```

```python
from openMWR.train import TrainingWorkflow

workflow = TrainingWorkflow.from_recipe(
    project_dir=".", site="my_site", retrieval="NN_python"
)

# Inspect or modify workflow.training_data and workflow.validation_data here.
workflow.change_hyper_params(epochs=100, epoch_max_lr=10)
model = workflow.train()
```

`from_recipe` loads the prepared training and validation data. It does not build
them. The default `device="auto"` selects CUDA, then MPS, then CPU. Pass an
explicit device to override it. Changes to the workflow do not edit the TOML
file. The checkpoint records
the effective settings. Use another retrieval name to keep experiments separate,
or `train(run_mode="fresh")` to replace an existing result.

Python selects training or optimization explicitly. Create a search recipe with
`openMWR create-retrieval my_site NN_opt_example --example NN_opt`, then run:

```python
workflow = TrainingWorkflow.from_recipe(".", "my_site", "NN_opt_example")
study = workflow.optimize(n_trials=20)
```

The observation selection and score come from the recipe. Use
`optimize(run_mode="continue", n_trials=20)` to add trials to a compatible study.
`workflow.requires_optimization` indicates whether searches remain in the current
parameters. `train()` needs fixed values. `optimize()` needs at least one search.

## Construct a model directly

For custom model construction, load the same project grids by name. This example
creates a multilinear regression model and uses the bundled MLR recommendations
for training and input noise:

```python
from openMWR.config import apply_seed, load_grid
from openMWR.models import Model
from openMWR.train import TrainingWorkflow

apply_seed(42)
model = Model(
    "MLR_python",
    output_vars=["T", "rh", "lwc", "lwp", "iwv"],
    heights=load_grid(".", "heights", "levels_208_30km"),
    freqs=load_grid(".", "frequencies", "hatpro"),
    is_MLR=True,
)
workflow = TrainingWorkflow(model, "my_site", project_dir=".")
workflow.load_training_data(dataset="training")
workflow.add_standard_training_hyper_params(seed=42)
workflow.add_input_noise_params()
model = workflow.train()
```

`output_vars` is explicit in Python too. Use `output_vars=["T", "rh"]` to select
temperature and humidity. A column-only model uses `heights=None`. Neural models
supply loss weights for exactly their selected outputs. Custom optimization
scores must select predicted variables.

To use prepared xarray datasets without dataset recipes, replace
`load_training_data()` with:

```python
workflow.set_training_data(training, validation)
```

Supply unnormalized data with a `time` dimension and optional `station` dimension.
The method keeps model variables, selects 90° for zenith models and checks both
datasets before replacing the current data. A missing station dimension becomes
one source. It clears previous dataset-selection and applied bias-correction
records. Apply bias correction after setting the data.

The recommendation helpers read the bundled example recipes. Extra inputs absent
from those examples need explicit noise values or search ranges. Neural profile
models also require loss weights. See {py:class}`openMWR.models.Model` and
{py:class}`openMWR.train.TrainingWorkflow` for direct construction and custom
hyperparameters. Recipe-based construction keeps more choices together in one file.

For custom architectures, subclass {py:class}`openMWR.models.BaseModel` in Python.
Use the built-in models as a reference for the training and prediction interface.
Load custom checkpoints through the custom class's `load()` method. Recipe model
kinds and command checkpoint loading use the built-in model classes.

For custom models without bundled recommendations, supply training parameters
through `change_hyper_params()` and noise values or search ranges through
`add_input_noise_params()`. Specify noise for every active input. `TB` applies to
all microwave channels. Complete explicit noise settings do not read an example.
Optimization also requires the trial count, analysis dataset, sampler seed and
score weights:

```python
study = workflow.optimize(
    n_trials=20, dataset="analysis_radiosonde", seed=42,
    score_weights={"T": 1.0},
)
```

Time and source filters are optional. Use `run_mode="continue"` to add trials to
the same experiment.

## Evaluate retrievals

```python
from openMWR.evaluation import (
    EvaluationRecipe, evaluate, save_evaluation,
)

recipe = EvaluationRecipe.load(".", "my_site", "radiosonde")
recipe.values["observations"]["start_date"] = "2025-01-01"
analysis = evaluate(recipe, retrievals=["NN_example", "MLR_example"])
report = save_evaluation(analysis)

zenith = next(case for case in analysis.evaluations if case.key == "zenith_observations")
errors = zenith.metrics["errors"]
temperature_rmse = errors.T.sel(error="RMSE")
```

The command uses this same engine and recipe. `evaluate()` returns xarray
metrics without saving a report. `save_evaluation()` writes one standalone HTML
file beside the evaluation recipe. Use the returned analysis for custom plots
or explicitly export selected metrics if needed. The metadata records effective settings and
input identities. Result keys are `zenith_observations`, `zenith_test`,
`bls_observations` and `bls_test` for the groups and inputs present in the recipe.
A configured `bls_baseline` has separate scores in each group. See
[Retrieval Analysis](retrieval_analysis.ipynb) for plotting examples.

## Predict and plot

```python
from openMWR.predict import PredictionWorkflow

workflow = PredictionWorkflow(".", "my_site", ["NN_example"])
result = workflow.predict("2025-01-12")
if result.errors:
    raise RuntimeError(result.errors)
pred = result.zenith.sel(model="NN_example")
paths = workflow.save(result)
```

`predict` returns results without writing files. Reuse the workflow for several
days to avoid reloading the models. Results expose `zenith`, `bls`, `observations`
and `observations_bls`. Unavailable groups are `None`. `result.errors` reports
missing or invalid measurement inputs.

To select a retrieval after inspecting its inputs, load observations with
`openMWR.hatpro_data.import_hatpro_data` and check variables with
`openMWR.observations.valid_observation_times`. Pass the same standardized dataset
to `workflow.predict(day, observations=observations)`. For scans, use
`observations_bls`. Supplied observations must belong to the requested UTC day.
Site validity declarations apply to both supplied and internally loaded data.

To read files already written by `openMWR predict`:

```python
from openMWR.predict import load_predictions

saved = load_predictions(".", "my_site", "2025-01-12")
pred = saved.zenith.sel(model="NN_example")
```

Use xarray and the plotting functions demonstrated in
[Run and Plot a Retrieval](run_retrieval.ipynb) for custom figures.
