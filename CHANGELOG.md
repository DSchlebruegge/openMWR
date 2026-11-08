# Release notes

## Unreleased

- Apply normalization once per variable, including spectral-consistency models.
  Align saved statistics by feature coordinates and select model features before
  filtering training samples. Migrate historical spectral checkpoints to preserve
  predictions with linear normalization. Spectral optimization histories require
  a fresh start. Historical spectral log normalization requires retraining.
- Include clouds reaching the highest profile level in the PyRTlib backend.
- Match weather timestamps without a dense distance matrix. Include samples
  exactly at the maximum time difference and retain the earlier source on ties.
- Require an explicit `ray_tracing` choice in simulation recipes and direct
  dataset-generation calls. Starter recipes and the torchMWRT example select
  spherical refractive paths. Rebuild simulations, training splits and paired
  analyses when enabling this setting. Prepared profiles and raw caches can be
  reused. Reject ducting and invalid ray paths before calculating absorption,
  reporting time, altitude and angles instead of returning partial atmospheres.
  Microwave ray tracing uses the actual profile-base altitude. Custom RT inputs
  require scalar or per-time profile-base altitude.
- Anchor libRadtran atmospheres and cloud heights at each profile's actual
  altitude above sea level. Prepared profiles record their base altitude,
  including uncut radiosondes whose base differs from the instrument.
  Existing profiles, simulations, training splits and paired analyses require
  rebuilding from local raw caches. Saved models are unchanged.
  Concurrent RT builds keep separate temporary libRadtran inputs.
- Convert ERA5 mixed-phase relative humidity to water-relative humidity and
  specific cloud liquid water to g/m³ before profile interpolation. Existing
  ERA5 profiles, simulations and paired analyses require rebuilding. Raw
  downloads can be reused.

## 0.2.1

- Publish openMWR and torchMWRT as separate packages from the shared repository.
  Installing openMWR also installs torchMWRT. Package metadata links to the
  documentation, source and issue tracker.
- Read RT coefficients directly from the installed PyRTlib package, including
  read-only installations. Load and validate each gas's table in one operation.
- Use R24 in the RT example and primary numerical checks. Evaluate complex
  speed-dependent line profiles only where they contribute, reducing calculation
  time and memory use and avoiding NaN gradients from inactive lines.
- Compute frequency-independent water-vapour quantities once per atmospheric
  state and skip unused generic continuum calculations for R23SD/R24, reducing
  repeated work and memory use across channels. Factor common frequency terms
  out of spectral-line sums to reduce repeated calculations in other models too.
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
