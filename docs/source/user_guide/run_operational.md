# Run Retrieval Operational

## Predict daily measurements

After training and evaluation, apply a saved retrieval to local HATPRO files:

```bash
openMWR predict my_site NN_example --date 2025-01-12
```

For an inclusive range of UTC dates:

```bash
openMWR predict my_site NN_example --start-date 2025-01-01 --end-date 2025-01-31
```

Prediction reads the local measurement path from `site.toml`. It does not download
data or update prepared datasets. The saved model supplies its input grids and
normalization. Training data is not needed.

Files are saved under `data/sites/my_site/predictions/YYYY/MM/` as
`YYYYMMDD_retrieval.nc` and, for scans, `YYYYMMDD_retrieval_bls.nc`.
Set `retrieval_output_dir` in `site.toml` or pass `--output-dir` to choose another
root. Relative paths resolve from the working project.

You can list several retrieval names. Omitting them uses `[prediction].retrievals`
in `site.toml`. Reprocessing replaces the selected day's zenith or scan file,
including its model selection. Zenith measurements are averaged to one minute.
Zenith predictions receive a centred five-sample running mean. Scan predictions
are not smoothed.

Missing or invalid measurement inputs are reported per day and group. Other
requested days and groups still run, but the command exits with status 1 if any
failed. Files from previous runs remain for failed groups. Check the log before
using them as current results.

For regular prediction, put this command in a cron job or another scheduler.
Use the environment's full `openMWR` path and set `--project-dir` to the working
project. Rerun recent days to include late-arriving measurements. Plotting can
follow in a Python script, as shown in [Run and Plot a Retrieval](run_retrieval.ipynb).

## Run a command in the background

On Linux and macOS, add `--background` or `-b` to `data`, `train`, `evaluate` or
`predict` to keep it running after the terminal closes:

```bash
openMWR data my_site training --with-dependencies --background
```

The command prints a process ID and console log path, then returns the prompt.
It uses the same Python environment. Check the log for completion or errors.
Successful launching does not mean that processing has finished.

Data commands append console output to the first selected dataset's `build.log`.
Other datasets keep their own build logs. Rebuilds preserve `build.log` while
replacing the generated data. Training uses `training.log` or `optimization.log`
beside the retrieval. Evaluation uses `sites/<site>/evaluations/<name>/evaluation.log`. Prediction appends
to `prediction.log` in the prediction output directory.

Use `--dry-run` before launching a background data build. The two options cannot
be combined.

## Update datasets

Use the same `data` command for initial creation and later updates:

```bash
openMWR data my_site analysis_radiosonde --with-dependencies
```

This updates the measurement comparison and its prerequisites, including source
checks. Source recipes with `end_year = "current_year"` or `end_date = "today"`
continue into new years or days. Explicit end dates stay fixed.

ERA5 saves each completed day. If an update is interrupted, repeat the same
command to continue after the last saved day. A failed download or file write
leaves the previously saved cache intact.

To update only the selected dataset using existing inputs, omit
`--with-dependencies`. Use `--dry-run` to check the processing order first.
If processing settings changed, rebuild the affected datasets using the command
reported by openMWR. See [Create a Retrieval](create_retrieval.md#build-or-update-the-whole-chain).

Dataset updates do not retrain saved models. Retraining and changing the models
used for prediction remain separate decisions.

## Invalid measurements

Record known sensor failures in `site.toml`, using openMWR variable names:

```toml
[[invalid_measurements]]
variables = ["surface_T", "surface_p", "surface_rh"]
reason = "Weather station unavailable"
periods = [
    { start_date = "2025-01-12", end_date = "2025-01-14" },
]
```

Dates are inclusive UTC days. Omit `end_date` for an ongoing failure.
For part of a day, use quoted UTC timestamps instead:

```toml
[[invalid_measurements]]
variables = ["TB_IR"]
reason = "IR sensor unavailable"
periods = [
    { start_time = "2025-01-12T10:15:30Z", end_time = "2025-01-12T10:40:00Z" },
    { start_time = "2025-01-13T08:00:00Z", end_time = "2025-01-13T08:30:00Z" },
]
```

The start is included and the end is excluded. Timestamps require `Z` for UTC
and can include up to six fractional-second digits. Omit `end_time` for an
ongoing failure. Use either date bounds or timestamp bounds within each period.
List periods sharing the same variables and reason in one entry. Add another
entry for a different sensor or fault.

Loading masks these variables before averaging and retains other measurements.
Original files are preserved. Fault detection and the choice of intervals belong
in the working project.

Analysis recipes list `required_observations`. Matched samples need finite values
in each listed variable. Optimization, evaluation and OMB analysis also apply
current validity declarations when reading existing comparison files. Changed
declarations require rebuilding affected prepared datasets before updating them.

Prediction requires the inputs recorded in the selected checkpoint. MET and IRT
files can be absent when their variables are not required. Profile and column
predictions can still be saved without surface pressure. Pressure profiles and
pressure-based plots then remain unavailable. Choosing a different retrieval
based on input availability belongs in the project's Python workflow.
