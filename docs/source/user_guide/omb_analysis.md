# OMB Analysis

Observation minus background (OMB) is measured brightness temperature minus
brightness temperature simulated from reference profiles. It can be inspected
without a trained retrieval.

## Prepare the paired data

Configure the site, measurement input and analysis recipe as described in
[Create a Retrieval](create_retrieval.md#prepare-measurements-and-matching-profiles).
The recipe pairs HATPRO observations with simulations from radiosonde profiles.
It selects the time-matching tolerance, rain filter and required measurements.

From the working project, with its environment activated, run:

```bash
openMWR data my_site analysis_radiosonde --with-dependencies
```

This builds or updates the analysis dataset and its inputs. The paired data are
saved under `data/sites/my_site/analysis_radiosonde/` as `zenith.nc` and, for
scans, `bls.nc`. Check that the simulation recipe uses
the absorption model and other settings intended for this comparison.
[ERA5](create_retrieval.md#using-era5) can supply the reference profiles instead.

## Generate the report

```bash
openMWR omb my_site analysis_radiosonde
```

The command reads the existing paired data and prints the report path. It saves
one `omb_YYYY-MM-DD_HH-MM-SS.html` beside the analysis data, with a UTC timestamp.
Vector plots are embedded in the HTML, so the file can be shared on its own.
Use `--output report.html` to choose a filename. Existing files are preserved.

For scans and a selected period:

```bash
openMWR omb my_site analysis_radiosonde --bls --start-date 2025-01-01 --end-date 2025-12-31
```

Scan reports use the prefix `omb_bls_`. Stations and viewing angles are reported
separately. Use `--stations`, `--frequencies` and `--angles` to select coordinates.
`--days-of-month 21 31` selects those days in each month.
With observed LWP available, `--max-lwp 2` keeps pairs below 2 g/m² in both
sources. Low LWP alone does not establish clear skies. See `openMWR omb --help`
for the options.

Optional calibration dates mark the time plots. Put them at the top level of
`sites/my_site/site.toml`, before any tables:

```toml
calibration_dates = ["2025-03-12", "2025-09-04"]
```

Dates are UTC and do not filter data. Omit this setting if the dates are unknown.

To overlay a saved retrieval's TB training input noise:

```bash
openMWR omb my_site analysis_radiosonde --retrieval NN_example
```

Several retrieval names can be listed. The overlay reads checkpoint settings at
matching frequencies and viewing angles. Training noise is an augmentation
choice, not a measurement of instrument noise.

## Read the report

The report contains channel counts, mean OMB, standard deviations, RMSE,
correlations and monthly heatmaps. Expand a channel for individual OMB and
residual time series, observed-vs-simulated scatter, histograms and empirical CDFs.
Similar distributions do not imply agreement of individual pairs.

Channel statistics use finite observation/reference pairs. Correlations use
common finite pairs across the selected channels. Standard deviations use
`ddof=0`. Mean OMB, standard deviation and RMSE are in kelvin.

Residual diagnostics regress each channel's OMB on its immediate lower and upper
selected frequencies within the groups 20–40 and 50–60 GHz. Edge channels use
one neighbor. A channel alone in its group uses centered OMB. The report lists
the predictors. Use `--frequency-band LOW HIGH`, repeated for each group, to
override the groups for other channel arrangements.

OMB combines instrument, reference-profile, radiative-transfer and sampling
errors. Neither total OMB scatter nor the regression residuals isolate instrument
noise. The regression fit and its scatter use the same samples. For monitoring
examples and limitations, see
[De Angelis et al. (2017)](https://amt.copernicus.org/articles/10/3947/2017/) and
[Böck et al. (2025)](https://amt.copernicus.org/articles/18/6251/2025/).

## Work with the data in Python

Use the same paired dataset for custom analysis:

```python
from openMWR.omb import Omb_Analysis

omb = Omb_Analysis(
    "my_site",
    project_dir=".",
    dataset="analysis_radiosonde",
)

observed = omb.observation
simulated = omb.background
difference = omb.diff
statistics = omb.statistics
```

The observations and simulations are xarray datasets. `difference` contains TB
differences, and `statistics` contains their channel summaries. Stations and
scan angles remain separate dimensions where present. Pass `bls=True` to load
scans. Date and coordinate selections are constructor arguments too.

To calculate neighboring-channel residuals explicitly:

```python
residuals, residual_std = omb.remove_explained_variance(
    frequency_bands=[[20.0, 40.0], [50.0, 60.0]],
)
```

Both arrays are in kelvin. Use xarray selection and plotting methods for further
analysis. `difference.to_netcdf("omb.nc")` exports the differences if needed.
`omb.save()` generates the HTML report. Training noise and bias adjustment
require separate [recipe settings](recipe_settings.md#input-noise-and-bias-correction).
