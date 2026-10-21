import xarray as xr
import numpy as np
from typing import Optional, Sequence, Dict, Tuple, TYPE_CHECKING
from openMWR.paths import dataset_file

if TYPE_CHECKING:
    from openMWR.models import BaseModel


class Omb_Analysis:
    """
    Observation-minus-background (OmB) analysis helper.

    Loads or creates a paired dataset of Hatpro observations and a background
    source (radiosonde forward calc or ERA5), computes TB differences, and
    provides filters and noise estimation utilities used in OmB analysis.

    Parameters
    ----------
    site : str
        Site name used to locate analysis data under
        `data/sites/{site}/`.
    project_dir : str
        Working-project root containing data/ and sites/.
    data_source : str, optional
        Background source to compare against observations. Options are
        ``"radiosonde"`` and ``"era5"``.
    station_ids_for_analysis : Sequence[str] or None, optional
        Radiosonde station IDs to use. The first ID is selected for the
        analysis. Required when `data_source="radiosonde"`.
    load_existing_analysis_dataset : bool, optional
        If True, load the existing analysis dataset from disk. If False,
        create it on demand using the corresponding import workflow.

    Notes
    -----
    On initialization, the class populates `observation`, `background`,
    `diff`, `mean_diff`, and `std_diff`. Variables that are entirely NaN
    are dropped from both sources.
    """
    def __init__(
        self,
        site: str,
        project_dir: str,
        data_source: str = 'radiosonde',
        station_ids_for_analysis: Optional[Sequence[str]] = None,
        load_existing_analysis_dataset: bool = False,
        dataset: str = None
    ) -> None:
        self.site = site
        self.project_dir = project_dir
        self.data_source = data_source
        self.station_ids_for_analysis = station_ids_for_analysis
        if data_source not in ['radiosonde', 'era5']:
            raise ValueError(f"Data source '{data_source}' not supported. Choose either 'radiosonde' or 'era5'.")
        if data_source == 'radiosonde' and not station_ids_for_analysis:
            raise ValueError('Radiosonde OMB analysis requires at least one station ID.')
        # ERA5 analysis has no radiosonde station selection.
        self.station_id_for_analysis = station_ids_for_analysis[0] if station_ids_for_analysis else None

        self.dataset = dataset or f'analysis_{data_source}'
        if not load_existing_analysis_dataset:
            from openMWR.datasets import DatasetWorkflow
            DatasetWorkflow(project_dir, site).build(self.dataset)
        self.an_data = xr.load_dataset(dataset_file(project_dir, site, self.dataset))
        if data_source == 'radiosonde':
            self.an_data = self.an_data.sel(station=self.station_id_for_analysis, drop=True)

        self.observation = self.an_data.sel(data_source='hatpro', drop=True)
        self.background = self.an_data.sel(data_source=data_source, drop=True)

        # Drop variables that are always NaN
        self.observation = self.observation.drop_vars([var for var in self.observation.data_vars if self.observation[var].isnull().all()])
        self.background = self.background.drop_vars([var for var in self.background.data_vars if self.background[var].isnull().all()])

        self.calc_difference()

    def calc_difference(self) -> None:
        """
        Compute observation-minus-background TB differences and update stats.

        Returns
        -------
        None
        """

        self.diff = self.observation.TB - self.background.TB
        
        self.calc_mean_std()

    def calc_mean_std(self) -> None:
        """
        Compute mean and standard deviation of TB differences over time.

        Returns
        -------
        None
        """
        self.mean_diff = self.diff.mean(dim='time')
        self.std_diff = self.diff.std(dim='time')

    def filter_lwp(self, lwp_threshold: float = 15) -> None:
        """
        Filter scenes by liquid water path (LWP) threshold.

        Parameters
        ----------
        lwp_threshold : float, optional
            Maximum allowed LWP value for both observation and background.

        Returns
        -------
        None
        """

        mask = (self.observation.lwp < lwp_threshold) & (self.background.lwp < lwp_threshold)

        self.observation = self.observation.where(mask, drop=True)
        self.background = self.background.where(mask, drop=True)

        self.calc_difference()

    def filter_channel_6(self, channel_6_threshold: float = 20) -> None:
        """
        Filter scenes by a threshold on the 7th TB channel (index 6).

        Parameters
        ----------
        channel_6_threshold : float, optional
            Maximum allowed TB for channel index 6 in both observation and
            background datasets.

        Returns
        -------
        None
        """

        mask = (self.observation.TB.isel(frq=6, drop=True) < channel_6_threshold) & (self.background.TB.isel(frq=6, drop=True) < channel_6_threshold)

        self.observation = self.observation.where(mask, drop=True)
        self.background = self.background.where(mask, drop=True)

        self.calc_difference()

    def remove_explained_variance(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Remove variance explained by neighboring frequency channels.

        For each frequency channel, a linear regression is fit against its
        immediate neighbors to predict standardized and centered differences.
        The residuals represent the unexplained variance per channel.

        Returns
        -------
        resid_diff : xr.DataArray
            Residuals of the difference data after removing explained variance,
            with the same shape and coordinates as `self.diff`.
        resid_std : xr.DataArray
            Standard deviation of the residuals along the `time` dimension.

        Notes
        -----
        - Differences are standardized using `mean_diff` and `std_diff`.
        - Neighbor selection avoids crossing the K/V band split by excluding
          indices 6 and 7 as neighbors.
        - Results are stored in `resid_diff` and `resid_std`.
        """

        n_freq = self.diff.sizes['frq']

        resid_array = np.empty(self.diff.shape, dtype=float)

        diff_standardized = (self.diff - self.mean_diff) / self.std_diff

        diff_centered = diff_standardized - self.mean_diff

        for i in range(n_freq):
            y = diff_centered.isel(frq=i).values

            # Neighbor frequencies
            j_list = []
            if i > 0 and i != 7:
                j_list.append(i-1)
            if i < n_freq - 1 and i != 6:
                j_list.append(i+1)

            
            X = diff_centered.isel(frq=j_list).values  # shape: (time, n_freq-1)

            # Linear regression
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta

            # Calculate Residuals
            resid_i = y - y_pred
            
            resid_array[:, i] = resid_i 

        self.resid_diff = xr.DataArray(resid_array, coords=self.diff.coords, dims=self.diff.dims)

        self.resid_std = self.resid_diff.std(dim='time')

        return self.resid_diff, self.resid_std

    def create_input_noise_from_diff(
        self,
        model: "BaseModel",
        from_resid_std: bool = True,
        factor: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Build input-noise hyperparameters from OmB statistics.

        Parameters
        ----------
        model : BaseModel
            Model with a `freqs` attribute listing the TB frequencies.
        from_resid_std : bool, optional
            If True, use residual standard deviations from
            `remove_explained_variance`. If False, use `std_diff`.
        factor : float or None, optional
            Optional multiplier applied to all noise values.

        Returns
        -------
        dict[str, float]
            Mapping of `input_noise_TB_<freq>` to noise values.

        Notes
        -----
        Call `remove_explained_variance` before using `from_resid_std=True`.
        """

        input_noise_dict = {}

        if from_resid_std:
            std = self.resid_std
        else:
            std = self.std_diff

        for f in model.freqs:
            input_noise_dict[f"input_noise_TB_{f}"] = std.sel(frq=f).item()

        if factor is not None:
            for key in input_noise_dict.keys():
                input_noise_dict[key] *= factor

        return input_noise_dict


def get_input_noise(model):
    hyper_params = model.hyper_params

    frq = model.freqs

    # Create a dataset for input noise
    input_noise = xr.Dataset()

    if model.is_bls_model:
        ang = model.angles
        if model.vary_input_noise_with_angle:
            input_noise['TB'] = xr.DataArray([[hyper_params[f"input_noise_TB_{f}"] * hyper_params[f"input_noise_TB_{a}"] for a in ang] for f in frq],
                                                coords={'frq': frq, 'ang': ang}, dims=('frq', 'ang'))
        else:
            input_noise['TB'] = xr.DataArray([[hyper_params[f"input_noise_TB_{f}"] for a in ang] for f in frq],
                                                coords={'frq': frq, 'ang': ang}, dims=('frq', 'ang'))
    else:
        input_noise['TB'] = xr.DataArray([hyper_params[f"input_noise_TB_{f}"] for f in frq],
                                                coords={'frq': frq}, dims='frq')

    for var in model.input_vars:
        if var != 'TB':
            input_noise[var] = hyper_params[f'input_noise_{var}']
    return input_noise
