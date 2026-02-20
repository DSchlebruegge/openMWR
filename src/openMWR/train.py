import torch
from torch import nn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
import copy
import os
import optuna
import logging
from sklearn.linear_model import LinearRegression
from collections import namedtuple

from typing import Union, List

from openMWR.utils import progress_bar, is_called_from_notebook, progress_log
from openMWR.xr_utils import load_combined_dataset, to_1d_tensor
from openMWR.run import normalize, create_tensor_from_ds, create_ds_from_y, denormalize, create_stats
from openMWR.calc_errors import cal_RMSE
from openMWR.models import Model, BaseModel
from openMWR.paths import data_root, site_root, site_subdir

#__all__ = ["separate_training_data", "TrainingWorkflow"]

class TrainingWorkflow:
    """
    Training workflow for openMWR retrieval models.

    This class manages the full workflow for training a retrieval model,
    including loading and preprocessing training data, configuring
    hyperparameters, adding input noise, model training, and optionally
    performing Optuna-based hyperparameter optimization.

    Parameters
    ----------
    model : Model
        Model instance derived from `BaseModel` used for the retrieval.
    site : str
        Site identifier for which the model is trained.
    device : str
        Device identifier (e.g., "cpu" or "cuda") used for training.
    data_dir : str
        Base directory containing all openMWR-managed data.

    Attributes
    ----------
    model : Model
        The retrieval model to be trained.
    site : str
        Site for which the training data is loaded.
    device : str
        Device used for training computations.
    data_dir : str
        Base directory containing all openMWR-managed data.
    hyper_params : dict
        Dictionary of training and input-noise hyperparameters.
    training_data : xr.Dataset
        Preprocessed training dataset.
    validation_data : xr.Dataset
        Preprocessed validation dataset.
    omb : Omb_Analysis
        OMB (observation-minus-background) analysis instance, available
        after calling `make_omb_analysis()`.

    Examples
    --------

    Creating a model and a workflow for training:

    .. code-block:: python

        from openMWR.models import Model
        from openMWR.train import TrainingWorkflow
        model = Model('NN_1', std_heights, std_freqs)
        workflow = TrainingWorkflow(model, site='munich', device='cuda', data_dir='data')
        workflow.load_training_data(['03715'], end_date_training='2024-10-31')

    Training a model with default hyperparameters and input noise:

    .. code-block:: python

        workflow.add_standard_training_hyper_params()
        workflow.add_input_noise_params()
        workflow.train()

    Or optimizing input noise using Optuna:

    .. code-block:: python

        workflow.add_standard_training_hyper_params()
        workflow.add_input_noise_params(optimization=True)
        workflow.optimize(n_trials=50)

    Or use input noise and hyperparameters from an existing model:

    .. code-block:: python

        workflow.get_hyper_params_from_existing_model('NN_opt_era5')
        workflow.train()

    Or use input noise and hyperparameters from an existing model but optimize some parameters:

    .. code-block:: python

        workflow.get_hyper_params_from_existing_model('NN_opt_era5')
        workflow.change_hyper_params(
            batch_size=workflow.suggest_int(256, 2048, step=256),
            lr_height=workflow.suggest_float(0.0001, 0.01, log=True)
        )
        workflow.optimize(n_trials=20)

    Or create an OMB analysis and use it for input noise selection:

    .. code-block:: python

        workflow.make_omb_analysis(['03715'], data_source='radiosonde')
        workflow.select_omb_input_noise_hyper_params()
        workflow.train()

    """
    def __init__(self, model: BaseModel, site: str, device: str, data_dir: str):
        self.model = model
        self.site = site
        self.device = device
        self.data_dir = data_dir

        self.hyper_params = {}

        self.logger = logging.getLogger(__name__)

        self.suggest_int = namedtuple('suggest_int', ['low', 'high', 'step', 'log'], defaults=[None, None, 1, False])
        self.suggest_float = namedtuple('suggest_float', ['low', 'high', 'step', 'log'], defaults=[None, None, None, False])
        self.suggest_categorical = namedtuple('suggest_categorical', ['choices'])

    def load_training_data(self, sources: List[str], end_date_training: str = None, file_extension: str = ''):
        """
        Load and prepare training and validation data for the retrieval model.

        Parameters
        ----------
        sources : list of str
            List of data sources used for training (e.g., radiosonde station IDs
            and optionally 'era5').
        end_date_training : str, optional
            ISO-format date string (YYYY-MM-DD). If provided, data occurring
            after this date is excluded from both training and validation sets.
        file_extension : str, optional
            Optional suffix for the training and validation data files.

        Notes
        -----
        - Selects only model-relevant variables.
        - For non-BLS models, selects only 90° elevation angle if present.
        - Stores datasets in `training_data` and `validation_data`.
        """
        self.hyper_params.update({
            'end_date_training': end_date_training,
            'sources': sources,
            'station_ids': [s for s in sources if s != 'era5']
        })

        self.training_data = load_combined_dataset(self.site, self.data_dir, 'training_data', sources, file_extension=file_extension)
        self.validation_data = load_combined_dataset(self.site, self.data_dir, 'validation_data', sources, file_extension=file_extension)

        if end_date_training is not None:
            # Filter the training data to only include data before the end date
            self.training_data = self.training_data.sel(time=slice(None, np.datetime64(end_date_training)))
            self.validation_data = self.validation_data.sel(time=slice(None, np.datetime64(end_date_training)))

        #self.training_data = add_negativ_lwp(self.training_data)
        #self.validation_data = add_negativ_lwp(self.validation_data)

        # Select only the relevant variables
        vars = list(set(self.model.input.data_vars) | set(self.model.output.data_vars))
        self.training_data = self.training_data[vars]
        self.validation_data = self.validation_data[vars]

        #If model is not a BLS model, select only the 90 degree angle
        if not self.model.is_bls_model and 'ang' in self.training_data.dims:
            self.training_data = self.training_data.sel(ang = 90, drop=True)
            self.validation_data = self.validation_data.sel(ang = 90, drop=True)

    def add_standard_training_hyper_params(self, 
                        batch_size=512,
                        epochs=500,
                        lr_height=0.002,
                        lr_low=0.0000001,
                        epoch_max_lr=50,
                        loss_fn='MAE',
                        weight_decay=1e-5,
                        repeats_for_linear_regression=10):
        """
        Add standard training hyperparameters.

        Parameters
        ----------
        batch_size : int, default=512
            Batch size used during training.
        epochs : int, default=500
            Number of training epochs.
        lr_height : float, default=0.002
            Maximum learning rate for the triangular LR schedule.
        lr_low : float, default=1e-7
            Minimum learning rate for the LR schedule (start/end value).
        epoch_max_lr : int, default=50
            Epoch at which the learning rate reaches `lr_height`.
        loss_fn : {'MAE', 'MSE', 'MAEPlusBiasLoss', 'MSEPlusBiasLoss'}, default='MAE'
            Loss function used for training.
        weight_decay : float, default=1e−5
            L2 regularization weight.
        repeats_for_linear_regression : int, default=10
            Number of input-noise repetitions for linear regression (for MLR models).

        """

        self.hyper_params.update({
            'batch_size': batch_size,
            'epochs': epochs,
            'lr_height': lr_height,
            'lr_low': lr_low,
            'epoch_max_lr': epoch_max_lr,
            'loss_fn': loss_fn,
            'weight_decay': weight_decay,
            'repeats_for_linear_regression': repeats_for_linear_regression
        })

    def change_hyper_params(self, **kwargs):
        """
        Update existing hyperparameters with new values.

        Parameters
        ----------
        **kwargs
            Arbitrary hyperparameters to modify or add.

        Notes
        -----
        Only updates provided keys; others remain unchanged.
        """
        self.hyper_params.update(kwargs)

    def get_hyper_params_from_existing_model(self, old_model_name, site_old_model=None):
        """
        Load hyperparameters from an already trained model and merge them
        into the current `hyper_params`.

        Parameters
        ----------
        old_model_name : str
            Name of the existing trained model.
        site_old_model : str, optional
            Site for the old model. Defaults to `self.site`.

        Notes
        -----
        Uses `Model.load_hyper_params`.
        """
        if site_old_model is None:
            site_old_model = self.site

        hyper_params_old = Model.load_hyper_params(site=site_old_model, name=old_model_name, data_dir=self.data_dir)

        self.hyper_params.update(hyper_params_old)

    def get_hyper_params_from_existing_study(self, study_site, old_model_name):
        """
        Import hyperparameters from an existing Optuna study.

        Parameters
        ----------
        study_site : str
            Site where the Optuna study is stored.
        old_model_name : str
            Name of the study/model whose parameters should be loaded.

        Notes
        -----
        Loads best parameters from the Optuna study database.
        """
        study_file = site_root(self.data_dir, study_site) / "optimization_studies.db"
        study = optuna.load_study(study_name=old_model_name, storage=f"sqlite:///{study_file}")

        self.hyper_params.update(study.best_params)

    def add_input_noise_params(self, optimization=False, **kwargs):
        """
        Add input noise hyperparameters for all model input variables.

        Parameters
        ----------
        optimization : bool, default=False
            If True, uses parameter ranges (`suggest_float`, etc.)
            instead of fixed values to support Optuna optimization.
        **kwargs
            Custom noise parameters overriding defaults.

        Notes
        -----
        - Generates noise parameters for TB frequencies and additional inputs.
        - For BLS models, handles angle-dependent noise.
        - Populates `hyper_params` keys of the form `input_noise_*`.
        """

        # Define standard input noise parameters
        standard_input_noise_params = {f'TB_{f}': 0.5 for f in self.model.freqs}
        # For BLS models, add angle dependent noise parameters
        if self.model.is_bls_model:
            standard_input_noise_params.update({f"TB_{a}": 1.0 for a in self.model.angles})
            standard_input_noise_params[f"TB_angle_factor"] = 3.0

        standard_input_noise_params.update({
            'TB_IR': 0.5,
            'surface_T': 0.5,
            'surface_p': 0.5,
            'surface_rh': 0.5,
            'doy_cos': 0.2, 
            'doy_sin': 0.2,
            'years_since_1970': 1.5,
        })

        # Define bounds for optimization
        standard_input_noise_bounds = {f'TB_{f}': self.suggest_float(0., 1.5) for f in self.model.freqs}
        # For BLS models, add angle dependent noise parameters
        if self.model.is_bls_model:
            standard_input_noise_bounds.update({f"TB_{a}": self.suggest_float(0., 5.0) for a in self.model.angles})
            standard_input_noise_bounds[f"TB_angle_factor"] = self.suggest_float(1.0, 3.0)

        standard_input_noise_bounds.update({
            'TB_IR': self.suggest_float(0.0, 2.0),
            'surface_T': self.suggest_float(0.1, 1.5),
            'surface_p': self.suggest_float(0.1, 2.0),
            'surface_rh': self.suggest_float(0.1, 2.0),
            'doy_cos': self.suggest_float(0.05, 0.5), 
            'doy_sin': self.suggest_float(0.05, 0.5),
            'years_since_1970': self.suggest_float(0.5, 3),
        })

        def add_param(var):
            if var in kwargs:
                param = kwargs[var]
            elif var in standard_input_noise_params and not optimization:
                param = standard_input_noise_params[var]
            elif var in standard_input_noise_bounds and optimization:
                param = standard_input_noise_bounds[var]
            else:
                raise ValueError(f"No standard input noise parameter for variable {var} found. Please provide a value.")
            
            self.hyper_params[f'input_noise_{var}'] = param

        for var in self.model.input_vars:
            if var == 'TB':
                for f in self.model.freqs:
                    add_param(f'TB_{f}')
                if self.model.is_bls_model:
                    if self.model.vary_input_noise_with_angle:
                        for a in self.model.angles:
                            add_param(f'TB_{a}')
                    elif self.model.vary_input_noise_with_angle_linear:
                        add_param("TB_angle_factor")
            else:
                add_param(var)


    def make_omb_analysis(self, station_ids_for_analysis, data_source='radiosonde'):
        """
        Create and process an OMB (observation-minus-background) analysis.

        Parameters
        ----------
        station_ids_for_analysis : list of str
            Radiosonde station IDs used for the OMB analysis.
        data_source : str, default='radiosonde'
            Data source used for background profiles.

        Notes
        -----
        - Filters analysis dataset to days ≤ 20 for training/validation separation.
        - Computes mean and standard deviation of OMB differences.
        """
        from openMWR.omb import Omb_Analysis

        self.omb = Omb_Analysis(
            self.site,
            self.data_dir,
            data_source=data_source,
            station_ids_for_analysis=station_ids_for_analysis,
            load_existing_analysis_dataset=True,
        )

        # Only select days with day smaller than 20. The other days are used for model validation.
        self.omb.diff = self.omb.diff.where(self.omb.diff['time'].dt.day <= 20, drop=True)
        self.omb.calc_mean_std()

    def make_bias_correction(self, specific_freqs=None, factor=None):
        """
        Apply bias correction to TB inputs based on OMB analysis.

        Parameters
        ----------
        specific_freqs : list of float, optional
            Frequencies for which bias correction should be applied.
            All others are set to zero.
        factor : float, optional
            Scale factor applied to the bias values.

        Raises
        ------
        ValueError
            If OMB analysis has not been created via `make_omb_analysis`.

        Notes
        -----
        Subtracts the OMB mean bias from training and validation TBs.
        Stores the bias as a list under `hyper_params['mean_bias']`.
        """
        if not hasattr(self, 'omb'):
            raise ValueError("Omb analysis not made yet. Call make_omb_analysis() first.")
        
        mean_bias = self.omb.mean_diff

        if specific_freqs is not None:
            for freq in mean_bias.frq.values:
                if freq not in specific_freqs:
                    mean_bias.loc[freq] = 0.0

        if factor is not None:
            mean_bias = mean_bias * factor

        self.training_data['TB'] = self.training_data['TB'] - mean_bias
        self.validation_data['TB'] = self.validation_data['TB'] - mean_bias

        self.hyper_params['mean_bias'] = mean_bias.values.tolist()

    def select_omb_input_noise_hyper_params(self, factor=None):
        """
        Generate input-noise hyperparameters from OMB using explained variance removal.

        Parameters
        ----------
        factor : float, optional
            Scaling factor applied to derived noise values.

        Raises
        ------
        ValueError
            If OMB analysis has not been computed.

        Notes
        -----
        Updates `hyper_params` with OMB-based noise estimates.
        """
        if not hasattr(self, 'omb'):
            raise ValueError("Omb analysis not made yet. Call make_omb_analysis() first.")

        self.omb.remove_explained_variance()
        self.hyper_params.update(self.omb.create_input_noise_from_diff(self.model, factor=factor))

    def add_factor_to_input_noise_TB(self, factor):
        """
        Multiply all TB-related input noise parameters by a scaling factor.

        Parameters
        ----------
        factor : float
            Multiplicative factor applied to all `input_noise_TB_*` parameters.
        """
        for f in self.model.freqs:
            self.hyper_params[f"input_noise_TB_{f}"] *= factor

    def _create_training_tensors(self):
        """
        Prepare normalized and stacked training/validation tensors.

        Calculates statistics like mean and std from training data,
        normalizes both training and validation datasets, stacks them
        over station and time dimensions, and converts to PyTorch tensors.

        Returns
        -------
        x_training : torch.Tensor
        y_training : torch.Tensor
        x_val : torch.Tensor
        y_val : torch.Tensor
        """
        # Normalize the data
        self.model.nor = create_stats(self.model, self.training_data)
        self.logger.info('Created normalization dataset')

        self.training_data = normalize(self.training_data, self.model)
        self.validation_data = normalize(self.validation_data, self.model)
        
        # Stack the data to create a single dimension for station and time
        # This is necessary to create a 2D tensor for training
        self.training_data = self.training_data.stack(station_time=("station", "time")).dropna(dim='station_time')
        self.validation_data = self.validation_data.stack(station_time=("station", "time")).dropna(dim='station_time')

        # Create tensors for training and validation
        x_training = create_tensor_from_ds(self.model.input, self.training_data, self.device)
        y_training = create_tensor_from_ds(self.model.output, self.training_data, self.device)
        x_val = create_tensor_from_ds(self.model.input, self.validation_data, self.device)
        y_val = create_tensor_from_ds(self.model.output, self.validation_data, self.device)

        return x_training, y_training, x_val, y_val
    
    def _get_std_dev_array_from_input_noise(self, model, hyper_params):
        """
        Compute standard deviation array for input noise injection.

        Parameters
        ----------
        model : BaseModel
            Model with input definitions and normalization.
        hyper_params : dict
            Hyperparameters containing noise definitions.

        Returns
        -------
        np.ndarray
            Flattened array of standard deviations for all inputs.

        Notes
        -----
        - Three possibilitys for TB noise in BLS models:
            1) Angle-dependent noise. (Different noise for each angle) (`model.vary_input_noise_with_angle`).
            2) Linearly angle-dependent noise (`model.vary_input_noise_with_angle_linear`).
            3) Angle-independent noise.
        """
        frq = model.freqs

        # Create a dataset for input noise
        input_noise = xr.Dataset()

        if model.is_bls_model:
            ang = model.angles
            if model.vary_input_noise_with_angle:
                input_noise['TB'] = xr.DataArray([[hyper_params[f"input_noise_TB_{f}"] * hyper_params[f"input_noise_TB_{a}"] for a in ang] for f in frq],
                                                    coords={'frq': frq, 'ang': ang}, dims=('frq', 'ang'))
            elif model.vary_input_noise_with_angle_linear:
                input_noise['TB'] = xr.DataArray([[hyper_params[f"input_noise_TB_{f}"] * (1 + hyper_params[f"input_noise_TB_angle_factor"] * ((90 - a) / 90)) for a in ang] for f in frq],
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

        # Normalize the input noise
        input_noise_nor = input_noise / model.nor.sel(stat='std', drop=True)

        input_noise_nor = input_noise_nor.sel(frq=model.freqs)

        if model.is_bls_model:
            input_noise_nor = input_noise_nor.stack(ang_frq=("ang", "frq"))

        # Concatenate the standard deviations for each variable
        std_dev = np.concatenate([input_noise_nor['TB'].values, [input_noise_nor[var].values for var in model.input_vars if var != 'TB']])

        return std_dev
    
    def _adjust_lr(self, optimizer, epoch, epochs, lr_height, lr_low, epoch_max_lr):
        """
        Adjusts the learning rate of the optimizer according to a two-phase linear schedule.
        In the first phase (before `epoch_max_lr`), the learning rate increases linearly from `lr_low` to `lr_height`.
        In the second phase (after `epoch_max_lr`), the learning rate decreases linearly from `lr_height` back to `lr_low`.
        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
            epoch (int): The current epoch number.
            epochs (int): The total number of training epochs.
            lr_height (float): The maximum learning rate to reach at `epoch_max_lr`.
            lr_low (float): The minimum learning rate (starting and ending value).
            epoch_max_lr (int): The epoch at which the learning rate peaks at `lr_height`.
        Returns:
            None
        """
        
        if epoch < epoch_max_lr:
            lr = lr_low - (lr_low - lr_height) * (epoch / epoch_max_lr)
        else:
            lr = lr_height - (lr_height - lr_low) * ((epoch - epoch_max_lr) / (epochs - epoch_max_lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def _plot_training_progress(self, epochs, eval_losses, train_losses=None):
        """
        Plot training and validation loss curves.

        Parameters
        ----------
        epochs : int
            Total number of epochs.
        eval_losses : list of float
            Validation loss per epoch.
        train_losses : list of float, optional
            Training loss per epoch.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Plot for the first subplot
        if train_losses is not None:
            ax[0].plot(train_losses, label='training_data')
        ax[0].plot(eval_losses, label='validation_data')
        ax[0].legend()
        ax[0].grid()

        # Plot for the second subplot
        start = int(epochs / 3)
        if train_losses is not None:
            ax[1].plot(range(start, len(train_losses)), train_losses[start:], label='training_data')
        ax[1].plot(range(start, len(eval_losses)), eval_losses[start:], label='validation_data')
        ax[1].legend()
        ax[1].grid()

        if is_called_from_notebook():
            plt.show()

        else:
            plot_dir = data_root(self.data_dir) / 'plots'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            fig.savefig(plot_dir / 'training_progress.pdf')
            plt.close(fig)
    
    def _training_loop(self, model: BaseModel, x_training: torch.Tensor, y_training: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, hyper_params: dict, device: str, quiet: bool = False):
        """
        Internal training loop for neural network models.

        Parameters
        ----------
        model : BaseModel
            Model instance to train.
        x_training, y_training : torch.Tensor
            Training inputs and targets.
        x_val, y_val : torch.Tensor
            Validation inputs and targets.
        hyper_params : dict
            Current hyperparameter dictionary.
        device : str
            Target device for computation.
        quiet : bool, default=False
            If True, suppress logging and progress output.

        Returns
        -------
        BaseModel
            Best-performing model based on validation loss.

        Notes
        -----
        - Input noise is injected during training.
        - Learning rate follows a triangular schedule (see `_adjust_lr` method).
        - Evaluation is performed on a validation set each epoch.
        """
        model = model.to(device)

        logger = logging.getLogger(__name__)

        epochs = hyper_params['epochs']
        batch_size = hyper_params['batch_size']
        weight_decay = hyper_params['weight_decay']

        # Learning rate
        lr_height = hyper_params['lr_height']
        lr_low = hyper_params['lr_low']
        epoch_max_lr = hyper_params['epoch_max_lr']

        if hyper_params['loss_fn'] == 'MSE':
            loss_fn = nn.MSELoss(reduction='none')
        elif hyper_params['loss_fn'] == 'MAE':
            loss_fn = nn.L1Loss(reduction='none')
        elif hyper_params['loss_fn'] == 'MAEPlusBiasLoss':
            loss_fn = MAEPlusBiasLoss(bias_weight=0.2)
        elif hyper_params['loss_fn'] == 'MSEPlusBiasLoss':
            loss_fn = MSEPlusBiasLoss(bias_weight=0.2)
        else:
            raise ValueError(f"Unknown loss function: {hyper_params['loss_fn']}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_low, weight_decay=weight_decay)

        if hasattr(model, 'loss_weights'):
            loss_weight = to_1d_tensor(model.loss_weights, device)
        else:
            loss_weight = None

        training_size = x_training.shape[0]

        # Errors
        std_dev_np = self._get_std_dev_array_from_input_noise(model, hyper_params)
        std_dev = torch.tensor(std_dev_np, dtype=torch.float32).to(device)

        train_losses = []
        eval_losses = []


        idxs = np.arange(0, training_size)

        start_time = time.time()

        for epoch in range(epochs):

            # Compute linear learning rate
            self._adjust_lr(optimizer, epoch, epochs, lr_height, lr_low, epoch_max_lr)

            # Shuffle
            np.random.shuffle(idxs)

            # Training
            model.train()

            for i in range(0, training_size, batch_size):
                idx = idxs[i: i+batch_size]
                x = x_training[idx,:]
                y = y_training[idx,:]

                ### Add random noise
                x_noisy = x + torch.randn(x.size(), device=device) * std_dev

                # Compute prediction and loss
                pred = model(x_noisy)
                loss_per_output = loss_fn(pred, y)

                if loss_weight is None:
                    loss = loss_per_output.mean()
                else:
                    loss = (loss_per_output * loss_weight).mean()

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            #Testing
            model.eval()
            # pred = model(x_training)
            # loss_per_output = loss_fn(pred, y_training)
            # if loss_weight is None:
            #     loss = loss_per_output.mean()
            # else:
            #     loss = (loss_per_output * loss_weight).mean()
            # train_losses.append(loss.item())
            train_losses=None

            pred = model(x_val)
            loss_per_output = loss_fn(pred, y_val)
            if loss_weight is None:
                loss = loss_per_output.mean()
            else:
                loss = (loss_per_output * loss_weight).mean()
            eval_losses.append(loss.item())

            if epoch == 0:
                best_loss = loss.item()
            elif loss.item() < best_loss:
                best_model = copy.deepcopy(model)
                best_loss = loss.item()
            
            if not quiet:
                if epoch % int(epochs / 100) == 0:
                    if is_called_from_notebook():
                        progress_bar(epoch, epochs, start_time)
                    else:
                        logger.info(progress_log(epoch, epochs, start_time))
        
        #if not quiet:
        self._plot_training_progress(epochs, eval_losses, train_losses)

        logger.info(f'Finished training loop with best loss on validation dataset: {best_loss}')

        return best_model
    
    def _calc_linear_regression(self, model: BaseModel, x_training, y_training, hyper_params):
        """
        Fit a linear regression model for MLR-based architectures.

        Parameters
        ----------
        model : BaseModel
            Model with a linear (stack) structure.
        x_training : torch.Tensor
            Training inputs.
        y_training : torch.Tensor
            Training targets.
        hyper_params : dict
            Hyperparameters including noise and repeats.

        Returns
        -------
        BaseModel
            Model with fitted linear coefficients.

        Notes
        -----
        - Uses LinearRegression from scikit-learn.
        - repeats training data with input noise for robustness.
        """
        repeats = hyper_params['repeats_for_linear_regression']

        x_training_np = x_training.cpu().numpy()
        y_training_np = y_training.cpu().numpy()

        x_repeated = np.concatenate([x_training_np] * repeats, axis=0)
        y_repeated = np.concatenate([y_training_np] * repeats, axis=0)

        std_dev = self._get_std_dev_array_from_input_noise(model, hyper_params)

        x_noisy = x_repeated + np.random.randn(*x_repeated.shape) * std_dev

        LR = LinearRegression()
        LR.fit(x_noisy, y_repeated)

        with torch.no_grad():
            model.stack.weight.copy_(torch.tensor(LR.coef_))
            model.stack.bias.copy_(torch.tensor(LR.intercept_))   

        return model

    def train(self):
        """
        Run training using current hyperparameters and data.

        Notes
        -----
        - Prepares data tensors.
        - Runs either linear regression (MLR) or neural network training.
        - Saves the trained model to disk.
        """
        self.logger.info(f'Using hyper parameters: {self.hyper_params}')

        x_training, y_training, x_val, y_val = self._create_training_tensors()

        if self.model.is_MLR:
            self.model = self._calc_linear_regression(self.model, x_training, y_training, self.hyper_params)
        else:
            self.model = self._training_loop(self.model, x_training, y_training, x_val, y_val, self.hyper_params, device=self.device, quiet=False)

        # Save model
        self.model.save(self.site, self.data_dir, self.hyper_params)

    def _external_stop_callback(self, study, trial):
        """
        Optuna callback to externally stop optimization if 'stop.txt' exists.

        Used to gracefully terminate long-running optimization jobs. Optuna will
        check for the presence of 'stop.txt' in the current directory at the end
        of each trial and stop the study if found.

        """
        if os.path.exists("stop.txt"):
            self.logger.error("Stop-Datei gefunden. Optuna-Optimierung wird abgebrochen.")
            study.stop()

    def _suggest_values(self, trial: optuna.Trial, value: any, name: str = None):
        """
        Recursively convert dictionaries/lists/namedtuples of suggestion
        definitions into Optuna parameter suggestions.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object used to sample values.
        value : any
            Nested structure of suggest_* objects, lists, dicts, or fixed values.
        name : str, optional
            Parameter name (used for nested structures).

        Returns
        -------
        any
            Sampled value or structure of sampled values.

        Notes
        -----
        Supports:
        - suggest_int(low, high)
        - suggest_float(low, high)
        - suggest_categorical(choices)
        - nested lists and dictionaries.
        """
        if isinstance(value, dict):
            return {
                k: self._suggest_values(trial, v, name=k)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [
                self._suggest_values(trial, v, name=f'{name}_{i}')
                for i, v in enumerate(value)
            ]
        elif isinstance(value, self.suggest_int):
            return trial.suggest_int(name, value.low, value.high, step=value.step, log=value.log)
        elif isinstance(value, self.suggest_float):
            return trial.suggest_float(name, value.low, value.high, step=value.step, log=value.log)
        elif isinstance(value, self.suggest_categorical):
            return trial.suggest_categorical(name, value.choices)
        else:
            return value

    def optimize(self, n_trials=None, data_source='radiosonde'):
        """
        Perform hyperparameter optimization using Optuna.

        Parameters
        ----------
        n_trials : int, optional
            Number of optimization trials. Defaults to 300 for BLS models
            and 200 otherwise.
        data_source : str, default='radiosonde'
            Data source used for real-data evaluation during optimization.
            Options include 'radiosonde' or 'era5'.

        Notes
        -----
        - Each trial trains a model using suggested parameters.
        - Evaluation is performed on real atmospheric profiles.
        - Best models are saved automatically.
        - Study is stored under `data/sites/{site}/optimization_studies.db`.
        """
        if n_trials is None:
            n_trials = 300 if self.model.is_bls_model else 200

        self.logger.info(f'Using hyper parameters: {self.hyper_params}')

        x_training, y_training, x_val, y_val = self._create_training_tensors()

        #Real test data
        file_extension = '_bls' if self.model.is_bls_model else ''

        analysis_file = site_subdir(self.data_dir, self.site, "analysis") / f'analysis_data_{data_source}{file_extension}.nc'
        an_data = xr.load_dataset(analysis_file)

        # Only select days with day smaller than 20. The other days are used for model validation.
        an_data = an_data.where(an_data['time'].dt.day <= 20, drop=True)

        ds_hatpro = an_data.sel(data_source='hatpro', drop=True)
        truth = an_data.sel(data_source=data_source, drop=True)

        ds_hatpro = ds_hatpro[self.model.input_vars]

        ds_nor = normalize(ds_hatpro, self.model)

        use_station_time = True if 'station' in ds_nor.dims else False
        dim_0='station_time' if use_station_time else 'time'

        if use_station_time:
            ds_nor = ds_nor.stack(station_time=("station", "time")).dropna(dim='station_time')

        x_real = create_tensor_from_ds(self.model.input, ds_nor, self.device, dim_0=dim_0)

        def objective(trial):

            model_trial = type(self.model)(**self.model.structure)

            model_trial.nor = self.model.nor.copy(deep=True)

            hyper_params_trial = self._suggest_values(trial, self.hyper_params)

            if model_trial.is_MLR:
                model_trial = self._calc_linear_regression(model_trial, x_training, y_training, hyper_params_trial)
                model_trial = model_trial.to(self.device)
            else:
                model_trial = self._training_loop(model_trial, x_training, y_training, x_val, y_val, hyper_params_trial, device=self.device, quiet=True)

            # Run model
            model_trial.eval()
            with torch.no_grad():
                y_real = model_trial(x_real).detach().cpu().numpy()

            pred_nor = create_ds_from_y(y_real, model_trial, ds_nor, dim_0=dim_0)

            if use_station_time:
                pred_nor = pred_nor.unstack("station_time")

            pred = denormalize(pred_nor, model_trial)

            # Calc real loss
            total_raso_loss = 0
            for var in ['T', 'rh']:
                if var in pred.data_vars:
                    raso_loss_height = cal_RMSE(pred[var], truth[var], dims_to_mean=['time', 'station'] if use_station_time else ['time'])

                    raso_loss = (raso_loss_height * model_trial.loss_weights[var]).mean().item()

                    self.logger.info(f'{var}: {raso_loss}')

                    if var == 'rh':
                        raso_loss /= 10

                    total_raso_loss += raso_loss

            # save the model if it is better than the best one so far
            try:
                best_value = trial.study.best_value
            except ValueError:
                best_value = float('inf')   

            if total_raso_loss < best_value:
                model_trial.save(self.site, self.data_dir, hyper_params_trial)
                
            return total_raso_loss

        study_file = site_root(self.data_dir, self.site) / "optimization_studies.db"
        study = optuna.create_study(study_name=self.model.name, storage=f"sqlite:///{study_file}", load_if_exists=True, direction="minimize")

        self.logger.info('Starting hyperparameter optimization with Optuna...')

        study.optimize(objective, n_trials=n_trials, callbacks=[self._external_stop_callback])

        self.logger.info('Optimization completed.')
        self.logger.info(f'Best study params: {study.best_params}')


def separate_training_data(site: str, source: Union[str, List[str]], data_dir: str, test_fraction: float = 0.1, val_fraction: float = 0.1, file_extension: str = ''):
    """
    Split forward-calculation data into training, validation, and test sets.

    Parameters
    ----------
    site : str
        Site identifier used to locate data under ``data/sites/{site}``.
    source : str or list of str
        Either ``'era5'`` or a radiosonde station ID; lists are processed
        recursively per entry.
    data_dir : str
        Base directory containing all openMWR-managed data.
    test_fraction : float, default=0.1
        Fraction of time indices assigned to the test set.
    val_fraction : float, default=0.1
        Fraction of time indices assigned to the validation set.
    file_extension : str, default=''
        Optional suffix for the forward-calculation file name
        and output files.

    Raises
    ------
    FileNotFoundError
        If the required forward-calculation file does not exist.

    Notes
    -----
    - Resulting NetCDF files are saved in ``data/sites/{site}/training`` with suffixes
      ``training_data_{source}.nc``, ``validation_data_{source}.nc``, and ``test_data_{source}.nc``.
    - Time indices are selected randomly without replacement.

    """

    if isinstance(source, list):
        for s in source:
            separate_training_data(site, s, data_dir, test_fraction=test_fraction, val_fraction=val_fraction, file_extension=file_extension)

    # Load the dataset with RT
    if file_extension != '':
        file_extension = '_' + file_extension

    if source == 'era5':
        file = site_subdir(data_dir, site, "era5") / f"forward_calc_era5{file_extension}.nc"
    else:
        file = site_subdir(data_dir, site, "radiosonde") / f"radiosonde_data_with_RT_{source}{file_extension}.nc"

    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist. Please create it with mwr_retrieval.run_RT.run_RT")
    ds = xr.load_dataset(file)

    # Randomly select 10% of the time indices for the test dataset
    test_time_indices = np.random.choice(ds.time, size=int(test_fraction * len(ds.time)), replace=False)
    test_time_indices = np.sort(test_time_indices)

    # Remaining time indices after selecting test data
    remaining_time_indices = np.setdiff1d(ds.time, test_time_indices)

    # Randomly select 10% of the remaining time indices for the validation dataset
    val_time_indices = np.random.choice(remaining_time_indices, size=int(val_fraction * len(ds.time)), replace=False)
    val_time_indices = np.sort(val_time_indices)

    # Remaining time indices are used for the training dataset
    train_time_indices = np.setdiff1d(remaining_time_indices, val_time_indices)

    logger = logging.getLogger(__name__)

    logger.info(f'Total number of time indices: {len(ds.time)}')
    logger.info(f'Number of training time indices: {len(train_time_indices)}')
    logger.info(f'Number of test time indices: {len(test_time_indices)}')
    logger.info(f'Number of validation time indices: {len(val_time_indices)}')

    # Extract training, test, and validation datasets
    training_data = ds.sel(time=train_time_indices)
    test_data = ds.sel(time=test_time_indices)
    validation_data = ds.sel(time=val_time_indices)

    # Save the datasets to NetCDF files
    dir = site_subdir(data_dir, site, "training")

    os.makedirs(dir, exist_ok=True)

    file = dir / f"training_data_{source}{file_extension}.nc"
    if os.path.exists(file):
        os.remove(file)
    training_data.to_netcdf(file)
    logger.info(f'Saved training data to {file}')

    file = dir / f"test_data_{source}{file_extension}.nc"
    if os.path.exists(file):
        os.remove(file)
    test_data.to_netcdf(file)
    logger.info(f'Saved test data to {file}')

    file = dir / f"validation_data_{source}{file_extension}.nc"
    if os.path.exists(file):
        os.remove(file)
    validation_data.to_netcdf(file)
    logger.info(f'Saved validation data to {file}')

class MAEPlusBiasLoss(nn.Module):
    """
    Mean absolute error loss with an added bias penalty.

    Parameters
    ----------
    bias_weight : float, optional
        Scaling factor for the mean absolute bias term across the batch.
    """
    def __init__(self, bias_weight=0.1):
        super().__init__()
        self.bias_weight = bias_weight
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, pred, target):
        """
        Compute MAE plus a bias term averaged over the batch.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground-truth targets.

        Returns
        -------
        torch.Tensor
            Scalar loss combining MAE and weighted bias.
        """
        # 1) Compute MAE (element-wise because reduction='none')
        mae_loss = self.mae(pred, target)  # Shape: [B, V] or multi-dimensional
        
        # 2) Average MAE (across all dimensions except batch)
        mae_loss_mean = mae_loss.mean()

        # 3) Compute per-variable bias across batch
        bias_per_var = torch.mean(pred - target, dim=0)  # averaged over batch only
        bias_loss = torch.mean(torch.abs(bias_per_var))  # average over variables

        # 4) Combine
        total_loss = mae_loss_mean + self.bias_weight * bias_loss
        return total_loss
    
class MSEPlusBiasLoss(nn.Module):
    """
    Mean squared error loss with an added bias penalty.

    Parameters
    ----------
    bias_weight : float, optional
        Scaling factor for the mean absolute bias term across the batch.
    """
    def __init__(self, bias_weight=0.1):
        super().__init__()
        self.bias_weight = bias_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        """
        Compute MSE plus a bias term averaged over the batch.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground-truth targets.

        Returns
        -------
        torch.Tensor
            Scalar loss combining MSE and weighted bias.
        """
        # 1) Compute MSE (element-wise because reduction='none')
        mse_loss = self.mse(pred, target)  # Shape: [B, V] or multi-dimensional

        # 2) Average MSE (across all dimensions except batch)
        mse_loss_mean = mse_loss.mean()

        # 3) Compute per-variable bias across batch
        bias_per_var = torch.mean(pred - target, dim=0)  # averaged over batch only
        bias_loss = torch.mean(torch.abs(bias_per_var))  # average over variables

        # 4) Combine
        total_loss = mse_loss_mean + self.bias_weight * bias_loss
        return total_loss
