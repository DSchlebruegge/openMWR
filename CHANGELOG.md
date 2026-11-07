# Release notes

## 0.2.1

- Publish openMWR and torchMWRT as separate packages from the shared repository.
  Installing openMWR also installs torchMWRT. Package metadata links to the
  documentation, source and issue tracker.
- Read RT coefficients directly from the installed PyRTlib package, including
  read-only installations. Load and validate each gas's table in one operation.
- Use R24 in the RT example and primary numerical checks. Evaluate complex
  speed-dependent line profiles only where they contribute, reducing calculation
  time and memory use and avoiding NaN gradients from inactive lines.
- Read binary RPG retrieval products alongside binary HATPRO measurements.
- Restrict daily HATPRO observations to the requested UTC day and preserve
  missing predictions during smoothing. Time-series plots can show long gaps.
- Support optimization of custom Python models and direct Python configuration
  of training data and input noise.
- Allow retrieval directories to be renamed without editing saved artifacts.
  Evaluation and OMB logs are saved only for background commands.

## 0.2

Version 0.2 changes the project layout, configuration and Python interface.

- TOML recipes and commands cover dataset generation, training, optimization,
  prediction, evaluation and observed-minus-simulated brightness temperatures.
  Python workflows use the same implementation.
- Named datasets support optional dependency updates. Recipes record processing
  and training choices, with editable height, frequency and elevation grids in
  `grids.toml`.
- Retrieval directories hold their recipe, saved model and optimization study.
  Evaluation and OMB reports are standalone HTML files with embedded vector plots.
- Calendar-based day selections replace random daily splits. The starter reserves
  days 1–3 for testing and days 15–17 for validation. Existing retrievals may have
  trained on these dates.
- HATPRO input supports manufacturer NetCDF and binary files. Observation
  validity rules and scan-angle checks control which samples can be used.
- Daily prediction files are replaced only after successful writes. Skew-T plots
  use a fixed layout and local pressure profiles for height labels.

### Upgrading from 0.1

Back up the working project, then preview conversion with
`openMWR migrate --dry-run` and apply it with `openMWR migrate`.
Conversion preserves saved weights and normalization and keeps the original
checkpoints as backups. Only migrate trusted checkpoints.
Scripts and processed datasets need separate review.

See [Upgrading from 0.1](https://dschlebruegge.github.io/openMWR/user_guide/upgrading_from_0_1.html)
for the layout, Python interface and dataset changes. The `0.1` Git tag retains
the previous interface.
