from __future__ import annotations

import inspect
import logging
import os
from functools import wraps
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import xarray as xr
from torch import nn
from openMWR.consts import std_angles
from openMWR.paths import site_subdir

logger = logging.getLogger(__name__)

def _capture_init_args(init):
    sig = inspect.signature(init)

    @wraps(init)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        ret = init(self, *args, **kwargs)

        self.structure = {
            k: v for k, v in bound.arguments.items()
            if k != "self"
        }

        return ret

    return wrapper


class BaseModel(nn.Module):
    """
    Base class for openMWR retrieval models.

    Handles common bookkeeping (model name, normalization, initialization
    arguments) and provides checkpoint helpers that are used by the training
    workflow to persist and restore models together with the hyperparameters
    and normalization statistics that were used during training.
    """
    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            Identifier for the model; also used as checkpoint file name.
        """
        super().__init__()

        self.name = name

        self.is_loaded = False

        self.relu = nn.ReLU()

    def __init_subclass__(cls, **kwargs):
        """
        Wrap subclass ``__init__`` to capture initialization arguments.

        The captured arguments are stored on each instance under ``structure``.
        This allows :meth:`BaseModel.save` and :meth:`BaseModel.load` to
        reconstruct a model with the exact architecture, inputs, and settings
        that were used when it was first created and trained.
        """
        super().__init_subclass__(**kwargs)

        # Nur wenn das Kind ein __init__ definiert
        if "__init__" in cls.__dict__:
            original_init = cls.__dict__["__init__"]
            cls.__init__ = _capture_init_args(original_init)

    def save(self, site: str, data_dir: str, hyper_params: dict):
        """
        Persist model weights and training metadata.

        Parameters
        ----------
        site : str
            Site identifier; determines the target checkpoint directory
            ``data/sites/{site}/retrieval/``.
        data_dir : str
            Base directory containing all openMWR-managed data.
        hyper_params : dict
            Hyperparameters used during training (e.g., learning rate, noise
            configuration). Stored so training settings can be inspected or
            reused later.

        Notes
        -----
        Saves the PyTorch ``state_dict``, the provided hyperparameters, the
        normalization dataset ``self.nor`` (created during training), and the
        captured ``structure`` of the model so it can be re-instantiated via
        :meth:`BaseModel.load`.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'hyper_params': hyper_params,
            'normalization': self.nor.to_dict(),
            'structure': self.structure,
        }

        file = site_subdir(data_dir, site, "retrieval") / f'{self.name}.pth'
        os.makedirs(file.parent, exist_ok=True)
        torch.save(checkpoint, file)
        logger.info(f'saved model under {file}')
    
    @classmethod
    def load(cls, name, site: str, data_dir: str, file_path=None):
        """
        Load a saved model checkpoint.

        Parameters
        ----------
        name : str
            Desired model name for the loaded instance. This can differ from
            the name stored in the checkpoint; the provided name is enforced.
        site : str
            Site identifier; used to resolve the default checkpoint path.
        data_dir : str
            Base directory containing all openMWR-managed data.
        file_path : str, optional
            Explicit path to a ``.pth`` checkpoint. If omitted, the default
            ``data/sites/{site}/retrieval/{name}.pth`` is used.

        Returns
        -------
        BaseModel
            Instantiated model with weights, normalization stats, and
            hyperparameters restored.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file cannot be located.
        """
        if file_path is None:
            file_path = site_subdir(data_dir, site, "retrieval") / f'{name}.pth'

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} does not exist.")

        checkpoint = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)

        if name in checkpoint['structure']:
            if name != checkpoint['structure']['name']:
                logger.warning(f"Model name '{name}' does not match the name stored in the checkpoint '{checkpoint['structure']['name']}'. Using the provided name.")
            
        checkpoint['structure']['name'] = name

        model = cls(**checkpoint['structure'])

        model.load_state_dict(checkpoint['model_state_dict'])
        model.nor = xr.Dataset.from_dict(checkpoint['normalization'])

        model.hyper_params = checkpoint['hyper_params']

        return model

    @classmethod
    def load_hyper_params(cls, site=None, name=None, data_dir=None, file_path=None):
        """
        Load only the hyperparameter dictionary from a checkpoint.

        Parameters
        ----------
        site : str, optional
            Site identifier used to resolve the default path.
        name : str, optional
            Model name used to resolve the default path.
        data_dir : str, optional
            Base directory containing all openMWR-managed data.
        file_path : str, optional
            Explicit checkpoint path. If provided, ``site`` and ``name`` are
            ignored.

        Returns
        -------
        dict
            Hyperparameters stored with the checkpoint.

        Raises
        ------
        ValueError
            If neither (``site`` and ``name``) nor ``file_path`` is provided.
        FileNotFoundError
            If the checkpoint file cannot be located.

        Notes
        -----
        This is used by :class:`~openMWR.train.TrainingWorkflow` to
        reuse training settings without loading the full model.
        """
        if (name is None or site is None or data_dir is None) and file_path is None:
            raise ValueError("Either name, site and data_dir or file_path must be provided.")
        if file_path is None:
            file_path = site_subdir(data_dir, site, "retrieval") / f'{name}.pth'

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} does not exist.")

        return torch.load(file_path, map_location='cpu', weights_only=False)['hyper_params']
    
    def create_slices(self):
        """
        Create slice indices for each output variable in the flattened output.

        The slices map variables in ``self.output`` (e.g., ``'T'``, ``'lwc'``)
        to their positions in the stacked network output. They are used in
        :meth:`Model.forward` to apply variable-specific activations and to
        reconstruct datasets after inference.
        """
        self.output_slices = {}
        idx = 0
        for var, da in self.output.items():
            self.output_slices[var] = slice(idx, idx + da.size)
            idx += da.size
    
class Model(BaseModel):
    """
    Default feed-forward retrieval architecture.

    Builds the xarray input/output templates, normalization setup, loss weights,
    and PyTorch network used during training with
    :class:`openMWR.train.TrainingWorkflow` and later during inference via
    :func:`openMWR.run.run_model`. Supports zenith-only and boundary-layer-scan
    (BLS) configurations as well as a multilinear regression fallback.
    """
    def __init__(
            self,
            name: str,
            heights: np.ndarray,
            freqs: np.ndarray,
            *, 
            is_bls_model: bool = False,
            is_MLR: bool = False,

            # only relevant when not is_MLR
            hidden_layers: Optional[Sequence[int]] = None,
            dropout_rate: float = 0.1,

            # only relevant when is_bls_model
            vary_input_noise_with_angle: bool = False,
            vary_input_noise_with_angle_linear: bool = False,
            angles: Optional[Sequence[float]] = None,

            # always relevant
            norm_functions: Optional[Mapping[str, str]] = None,
            additional_input_vars: Optional[Sequence[str]] = None,
        ):
        """
        Parameters
        ----------
        name : str
            Identifier for the model and its checkpoint file.
        heights : array-like
            Height grid (meters) for profile outputs (T, rh, lwc).
        freqs : array-like
            Microwave channel frequencies (GHz) for TB inputs.
        is_bls_model : bool, default=False
            If ``True``, creates a Boundary-Layer-Scan (BLS) model and
            creates TB inputs with ``ang`` and ``frq`` dimensions.
        is_MLR : bool, default=False
            If ``True``, use a single linear layer instead of an MLP and switch
            LWC/LWP normalization to ``max_norm``.
        hidden_layers : Sequence[int], optional
            Widths of hidden layers for the MLP (default ``[1024, 64, 1024]``).
        dropout_rate : float, default=0.1
            Dropout applied before the final layer when using the MLP.
        vary_input_noise_with_angle : bool, default=False
            Enable per-angle TB noise hyperparameters for BLS training.
        vary_input_noise_with_angle_linear : bool, default=False
            Apply linearly angle-dependent TB noise during training.
        angles : Sequence[float], optional
            Elevation angles for BLS inputs; defaults to ``std_angles``.
        norm_functions : Mapping[str, str], optional
            Override of variable -> normalization scheme; defaults are provided.
            Options are ``'z_norm'``, ``'std_norm'``, ``'log_norm'`` and ``'max_norm'``.
        additional_input_vars : Sequence[str], optional
            Extra inputs next to TBs. Defaults are for 
            Zenith: ``['TB_IR', 'surface_T', 'surface_p', 'surface_rh', 'doy_cos', 'doy_sin']``
            and for BLS: ``['surface_T', 'surface_p', 'surface_rh', 'doy_cos', 'doy_sin']``

        Notes
        -----
        - Defines ``self.input`` (TB + auxiliary inputs) and ``self.output``
          (T, rh, lwc profiles plus scalar lwp/iwv) datasets used by training and
          inference helpers in ``openMWR.run``.
        - Initializes height-dependent ``loss_weights`` for profile variables
          when using the MLP.
        - Computes ``n_input``, ``n_output``, and ``output_slices`` so tensors
          can be stacked/reshaped consistently during training and inference.
        """
        super().__init__(name)

        self.heights = heights
        self.freqs = freqs
        self.is_bls_model = is_bls_model
        self.is_MLR = is_MLR

        if not is_MLR:
            self.hidden_layers = list(hidden_layers) if hidden_layers is not None else [1024, 64, 1024]
            self.dropout_rate = dropout_rate

        if is_bls_model:
            self.vary_input_noise_with_angle = vary_input_noise_with_angle
            self.vary_input_noise_with_angle_linear = vary_input_noise_with_angle_linear
            self.angles = list(angles) if angles is not None else list(std_angles)

        standard_normfunctions = {
            'TB': 'z_norm',
            'TB_IR': 'z_norm',
            'surface_T': 'z_norm',
            'surface_p': 'z_norm',
            'surface_rh': 'z_norm',
            'doy_cos': 'std_norm',
            'doy_sin': 'std_norm',
            'T': 'z_norm',
            'rh': 'z_norm',
            'lwc': 'log_norm' if not is_MLR else 'max_norm',
            'lwp': 'log_norm' if not is_MLR else 'max_norm',
            'iwv': 'z_norm',
        }
        self.norm_functions = dict(norm_functions) if norm_functions is not None else standard_normfunctions

        standard_additional_input_vars = (
            ['surface_T', 'surface_p', 'surface_rh', 'doy_cos', 'doy_sin']
            if is_bls_model
            else ['TB_IR', 'surface_T', 'surface_p', 'surface_rh', 'doy_cos', 'doy_sin']
        )
        self.additional_input_vars = (
            list(additional_input_vars) if additional_input_vars is not None else standard_additional_input_vars
        )

        if is_bls_model:
            freq_da = xr.DataArray(np.zeros((len(self.angles), len(self.freqs))), 
                        dims=['ang', 'frq'], coords={'ang': self.angles, 'frq': self.freqs})
        else:
            freq_da = xr.DataArray(np.zeros(self.freqs.shape), dims='frq', coords={'frq': self.freqs})

        self.input = xr.Dataset({'TB': freq_da, **{var: 1 for var in self.additional_input_vars}})

        heights_da = xr.DataArray(np.zeros(heights.shape), dims='height', coords={'height': heights})
        self.output = xr.Dataset({'T': heights_da, 
                                  'rh': heights_da, 
                                  'lwc': heights_da, 
                                  'lwp': 1, 
                                  'iwv': 1})
        
        if not hasattr(self, "loss_weights") and not is_MLR:
            h = self.heights
            if self.is_bls_model:
                loss_weights_height_np = np.where(h <= 2000, 1.0, np.where(h <= 10000, 0.5, 0.25))
            else:
                loss_weights_height_np = np.where(h <= 10000, 1.0, 0.5)

            loss_weights_height = xr.DataArray(loss_weights_height_np, dims='height', coords={'height': h})
            self.loss_weights = xr.Dataset({'T': loss_weights_height, 
                                  'rh': loss_weights_height, 
                                  'lwc': loss_weights_height, 
                                  'lwp': 1, 
                                  'iwv': 1})

        self.input_vars = list(self.input.data_vars)
        self.output_vars = list(self.output.data_vars)

        self.n_input = sum(self.input[dim].size for dim in self.input_vars)
        self.n_output = sum(self.output[dim].size for dim in self.output_vars)

        self.create_slices()

        if is_MLR:
            self.stack = nn.Linear(self.n_input, self.n_output)
        else:
            layer_list = []
            layer_list.append(nn.Linear(self.n_input, self.hidden_layers[0]))
            layer_list.append(nn.Tanh())     

            for i in range(len(self.hidden_layers) - 1):
                layer_list.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
                layer_list.append(nn.Tanh())

            if self.dropout_rate > 0:
                layer_list.append(nn.Dropout(p=self.dropout_rate))

            layer_list.append(nn.Linear(self.hidden_layers[-1], self.n_output))

            self.stack = nn.Sequential(*layer_list)

        #self.structure_was_initialized = True

    def forward(self, x):
        """
        Forward pass through the retrieval network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_input)`` created by
            :func:`openMWR.run.create_tensor_from_ds`.

        Returns
        -------
        torch.Tensor
            Flattened predictions of shape ``(batch_size, n_output)`` ordered
            according to ``self.output_slices``.

        Notes
        -----
        Applies ``ReLU`` to ``lwc`` and ``lwp`` outputs (except during
        training of ``is_MLR`` models) to keep these quantities non-negative.
        """
        y = self.stack(x)

        if not self.is_MLR or not self.training:
            y[:, self.output_slices['lwc']] = self.relu(y[:, self.output_slices['lwc']])
            y[:, self.output_slices['lwp']] = self.relu(y[:, self.output_slices['lwp']])

        return y

class SpectralConsistencyModel(BaseModel):
    """
    Ensures each channel's brightness temperature stays consistent with the
    remaining spectrum by learning a dedicated stack for every frequency.

    Each stack observes the entire spectrum except the target frequency and
    predicts the brightness temperature at that frequency. The resulting
    spectrum can be used for example to detect drifting channels.

    Parameters
    ----------
    name : str
        Unique identifier for checkpoints and logging.
    freqs : np.ndarray
        Frequencies (or channels) modeled by the network.
    hidden_layer_size : int, optional
        Width of the intermediate linear layer in each per-frequency stack.
    norm_functions : Mapping[str, str], optional
        Normalization functions to apply to ``TB``. Defaults to ``{'TB':
        'z_norm'}``.

    Attributes
    ----------
    freqs : np.ndarray
        Frequency coordinates that define the input/output dimension.
    stacks : nn.ModuleList
        Per-frequency feed-forward blocks predicting each brightness temperature
        from the remaining spectrum.
    n_freqs : int
        Number of spectral channels, used to build the stacks and slicing
        metadata.
    """
    def __init__(self, 
                 name,
                 freqs: np.ndarray, 
                 hidden_layer_size: int = 64,
                 norm_functions: Optional[Mapping[str, str]] = None
        ):
        super().__init__(name)

        self.freqs = freqs
        self.hidden_layer_size = hidden_layer_size

        self.norm_functions = dict(norm_functions) if norm_functions is not None else {'TB': 'z_norm'}

        self.is_bls_model = False
        self.is_MLR = False

        freq_da = xr.DataArray(np.zeros(self.freqs.shape), dims='frq', coords={'frq': self.freqs})

        self.input = xr.Dataset({'TB': freq_da})
        self.output = xr.Dataset({'TB': freq_da})

        self.input_vars = list(self.input.data_vars)
        self.output_vars = list(self.output.data_vars)

        self.n_input = sum(self.input[dim].size for dim in self.input_vars)
        self.n_output = sum(self.output[dim].size for dim in self.output_vars)

        self.create_slices()

        self.n_freqs = len(self.freqs)

        self.stacks = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.n_freqs - 1, self.hidden_layer_size),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_size, 1),
            ) for _ in range(self.n_freqs)]
        )

    def forward(self, x):
        """
        Predicts brightness temperatures for each frequency while masking the
        target channel in the input.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch_size, n_freqs)`` containing brightness
            temperature inputs.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, n_freqs)`` containing the spectral
            reconstruction where each channel is predicted from the other
            frequencies.
        """
        y = torch.zeros_like(x)

        for i in range(self.n_freqs):
            x_without_i = torch.cat((x[:, :i], x[:, i+1:]), dim=1)  # Shape: (B, 13)
            y[:, i] = self.stacks[i](x_without_i).squeeze(-1)       # Shape: (B,)

        return y  # Shape: (B, 14)
