# Retrieval Examples

`create-site` supplies one zenith neural network recipe, `NN_example`.
To add another retrieval, run this from the project directory.
The site must already exist. See [Create a Retrieval](create_retrieval.md#create-the-site-configuration)
for site setup.

```bash
openMWR create-retrieval my_site MLR_trial --example MLR
```

This creates `sites/my_site/retrieval/MLR_trial/retrieval.toml`.
`MLR_trial` is the new retrieval name. `--example MLR` selects the bundled
recipe to copy. The command copies it unchanged and refuses an existing retrieval
directory. It does not adapt settings to the site, train or build datasets.
Without `--example`, it copies the NN recipe.

Open the new file and review the dataset and station selection, inputs, outputs,
training settings and any optimization settings. With the input datasets ready,
train using the retrieval name:

```bash
openMWR train my_site MLR_trial
```

`train` runs optimization when the recipe contains search ranges. The retrieval
name does not control this. These recipes are starting points for an experiment.
Retrieval performance needs independent evaluation.

| Example | Model | Download recipe |
| --- | --- | --- |
| `NN` | Zenith neural network | {download}`TOML <../../../src/openMWR/templates/site/retrieval/NN_example/retrieval.toml>` |
| `NN_opt` | Zenith neural network with parameter searches | {download}`TOML <../../../src/openMWR/templates/site/retrieval/NN_opt_example/retrieval.toml>` |
| `MLR` | Zenith multilinear regression | {download}`TOML <../../../src/openMWR/templates/site/retrieval/MLR_example/retrieval.toml>` |
| `MLR_opt` | Zenith multilinear regression with input-noise searches | {download}`TOML <../../../src/openMWR/templates/site/retrieval/MLR_opt_example/retrieval.toml>` |
| `NN_BLS` | Boundary layer scan neural network | {download}`TOML <../../../src/openMWR/templates/site/retrieval/NN_BLS_example/retrieval.toml>` |
| `NN_BLS_opt` | Boundary layer scan neural network with parameter searches | {download}`TOML <../../../src/openMWR/templates/site/retrieval/NN_BLS_opt_example/retrieval.toml>` |
| `SPC` | Spectral-consistency neural network | {download}`TOML <../../../src/openMWR/templates/site/retrieval/SPC_example/retrieval.toml>` |
| `SPC_opt` | Spectral-consistency neural network with parameter searches | {download}`TOML <../../../src/openMWR/templates/site/retrieval/SPC_opt_example/retrieval.toml>` |

The downloads contain the same recipes as the command. To use a download,
save it as `sites/<site>/retrieval/<name>/retrieval.toml`.
The settings are explained in [Recipe Settings](recipe_settings.md).

After training, add the retrieval name to an evaluation recipe's `retrievals`
list. Profile retrievals can use `evaluations/radiosonde.toml`.
For spectral comparisons, save this example as `evaluations/spectral.toml` and
adjust its retrieval name and reference selections:

```{literalinclude} ../../../src/openMWR/templates/site/evaluations/spectral.toml
:language: toml
```

For Python workflows, use
`openMWR.site.create_retrieval(site, name, project_dir, example="MLR")`.
