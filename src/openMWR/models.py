from __future__ import annotations

import inspect
import logging
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import xarray as xr
from torch import nn
from openMWR.config import check_coordinates, check_keys, check_number, require_keys, check_output_vars
from openMWR.paths import checkpoint_file

logger = logging.getLogger(__name__)
CHECKPOINT_VERSION = 2


def _plain_metadata(value, label):
    """Lower checkpoint metadata to ordinary Python values, rejecting objects."""
    if isinstance(value, np.ndarray):
        return _plain_metadata(value.tolist(), label)
    if isinstance(value, np.generic):
        return _plain_metadata(value.item(), label)
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f'{label} must have string keys.')
        return {str(key): _plain_metadata(item, f'{label}.{key}') for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item, f'{label}[{index}]') for index, item in enumerate(value)]
    raise ValueError(f'{label}: unsupported metadata type {type(value).__name__}. Use strings, numbers, lists or dictionaries.')


def _checkpoint_payload(checkpoint):
    """Validate the current envelope and keep Python objects out of metadata."""
    required = ('format_version', 'model_state_dict', 'hyper_params', 'normalization', 'structure', 'model_class')
    check_keys(checkpoint, (*required, 'training_metadata', 'normalization_passes'), 'checkpoint')
    require_keys(checkpoint, required, 'checkpoint')
    if type(checkpoint['format_version']) is not int or checkpoint['format_version'] != CHECKPOINT_VERSION:
        raise ValueError(f'Checkpoint format_version must be {CHECKPOINT_VERSION}.')
    for key in ('model_state_dict', 'hyper_params', 'normalization', 'structure'):
        if not isinstance(checkpoint[key], dict):
            raise ValueError(f'checkpoint.{key} must be a dictionary.')
    if not isinstance(checkpoint['model_class'], str) or not checkpoint['model_class']:
        raise ValueError('checkpoint.model_class must be a nonempty class name.')
    if 'training_metadata' in checkpoint and not isinstance(checkpoint['training_metadata'], dict):
        raise ValueError('checkpoint.training_metadata must be a dictionary.')
    if 'normalization_passes' in checkpoint and (type(checkpoint['normalization_passes']) is not int
                                               or checkpoint['normalization_passes'] != 1):
        raise ValueError('checkpoint.normalization_passes must be 1.')
    state = checkpoint['model_state_dict']
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError('checkpoint.model_state_dict must map parameter names to tensors.')
    return {key: state if key == 'model_state_dict' else _plain_metadata(value, f'checkpoint.{key}')
            for key, value in checkpoint.items()}


def _read_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(f'Cannot read checkpoint {path} in the current format. Run openMWR migrate --project-dir <project> for older checkpoints.') from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Checkpoint {path} must be a dictionary.')
    if checkpoint.get('format_version') != CHECKPOINT_VERSION:
        raise ValueError(f'Checkpoint {path} requires conversion. Run openMWR migrate --project-dir <project> for older checkpoints.')
    return _checkpoint_payload(checkpoint)

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
            Identifier for the model and its retrieval directory.
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

    def save(self, site: str, project_dir: str, hyper_params: dict, training_metadata=None):
        """
        Persist model weights and training metadata.

        Parameters
        ----------
        site : str
            Site identifier; determines the target checkpoint directory
            ``sites/{site}/retrieval/{name}/``.
        project_dir : str
            Working-project root containing data/ and sites/.
        hyper_params : dict
            Hyperparameters used during training (e.g., learning rate, noise
            configuration). Stored so training settings can be inspected or
            reused later.

        Notes
        -----
        Saves the PyTorch ``state_dict``, the provided hyperparameters, the
        normalization dataset ``self.nor`` (created during training), and the
        captured ``structure`` of the model so it can be re-instantiated via
        :meth:`BaseModel.load`. The name is supplied when loading and is not
        part of the saved model settings.
        """
        checkpoint = {
            'format_version': CHECKPOINT_VERSION,
            'model_state_dict': self.state_dict(),
            'hyper_params': hyper_params,
            'normalization': self.nor.to_dict(),
            'structure': self.checkpoint_structure(),
            'model_class': type(self).__name__,
        }
        if set(self.input_vars) & set(self.output_vars):
            checkpoint['normalization_passes'] = 1
        if training_metadata is not None:
            checkpoint['training_metadata'] = training_metadata

        file = checkpoint_file(project_dir, site, self.name)
        os.makedirs(file.parent, exist_ok=True)
        torch.save(_checkpoint_payload(checkpoint), file)
        logger.info(f'saved model under {file}')

    def checkpoint_structure(self):
        """Return model settings with resolved defaults, independent of its name."""
        structure = {key: value for key, value in self.structure.items() if key != 'name'}
        for key in ('hidden_layers', 'angles', 'additional_input_vars', 'norm_functions', 'output_vars'):
            if key in structure and hasattr(self, key):
                structure[key] = getattr(self, key)
        if 'loss_weights' in structure and hasattr(self, 'loss_weights'):
            structure['loss_weights'] = {name: data.values.tolist()
                                         for name, data in self.loss_weights.data_vars.items()}
        return structure

    @property
    def training_dataset(self):
        """Named training input, or None for data supplied directly in Python."""
        return self.hyper_params.get('training_dataset', 'training')
    
    @classmethod
    def load(cls, name, site: str, project_dir: str, file_path=None):
        """
        Load a saved model checkpoint.

        Parameters
        ----------
        name : str
            Desired model name for the loaded instance. This can differ from
            the name stored in the checkpoint; the provided name is enforced.
        site : str
            Site identifier; used to resolve the default checkpoint path.
        project_dir : str
            Working-project root containing data/ and sites/.
        file_path : str, optional
            Explicit path to a ``.pth`` checkpoint. If omitted, the default
            ``sites/{site}/retrieval/{name}/model.pth`` is used.

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
            file_path = checkpoint_file(project_dir, site, name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} does not exist.")

        checkpoint = _read_checkpoint(file_path)

        return cls._from_checkpoint(checkpoint, name)

    @classmethod
    def _from_checkpoint(cls, checkpoint, name):
        structure = {**checkpoint['structure'], 'name': name}
        normalization = xr.Dataset.from_dict(checkpoint['normalization'])

        model = cls(**structure)
        if set(model.input_vars) & set(model.output_vars) and checkpoint.get('normalization_passes') != 1:
            raise ValueError('Saved spectral normalization requires conversion. '
                             'Run openMWR migrate --project-dir <project>, or retrain the model.')

        model.load_state_dict(checkpoint['model_state_dict'])
        model.nor = normalization

        model.hyper_params = checkpoint['hyper_params']
        model.training_metadata = checkpoint.get('training_metadata')

        return model

    @classmethod
    def load_hyper_params(cls, site=None, name=None, project_dir=None, file_path=None):
        """
        Load only the hyperparameter dictionary from a checkpoint.

        Parameters
        ----------
        site : str, optional
            Site identifier used to resolve the default path.
        name : str, optional
            Model name used to resolve the default path.
        project_dir : str, optional
            Working-project root containing data/ and sites/.
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
        if (name is None or site is None or project_dir is None) and file_path is None:
            raise ValueError("Either name, site and project_dir or file_path must be provided.")
        if file_path is None:
            file_path = checkpoint_file(project_dir, site, name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} does not exist.")

        return _read_checkpoint(file_path)['hyper_params']
    
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
    
def load_model(name, site, project_dir, file_path=None):
    """Load a built-in model using its checkpoint, without consulting a recipe."""
    path = file_path if file_path is not None else checkpoint_file(project_dir, site, name)
    checkpoint = _read_checkpoint(path)
    kind = checkpoint['model_class']
    classes = {'Model': Model, 'SpectralConsistencyModel': SpectralConsistencyModel}
    if kind not in classes:
        raise ValueError(f"Unsupported checkpoint model class {kind!r}; load custom models with their Python class.")
    return classes[kind]._from_checkpoint(checkpoint, name)


class Model(BaseModel):
    """
    Default feed-forward retrieval architecture.

    Builds the xarray input/output templates, normalization setup, loss weights,
    and PyTorch network used during training with
    :class:`openMWR.train.TrainingWorkflow` and later during inference via
    :func:`openMWR.predict.run_model`. Supports zenith-only and boundary-layer-scan
    (BLS) configurations as well as a multilinear regression fallback.
    """
    def __init__(
            self,
            name: str,
            heights: np.ndarray | None,
            freqs: np.ndarray,
            *, 
            output_vars: list[str],
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
            loss_weights: Mapping[str, float | Sequence[float] | Mapping[str, Sequence[float]]] | None = None,
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
        output_vars : list of str
            Ordered, nonempty selection of T, rh, lwc, lwp and iwv.
            Supply heights=None for a retrieval with only column outputs.
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
            Required for BLS inputs. Supply numeric angles, for example from
            ``load_grid(project_dir, 'angles', 'hatpro_bls')``.
        norm_functions : Mapping[str, str], optional
            Override of variable -> normalization scheme; defaults are provided.
            Options are ``'z_norm'``, ``'std_norm'``, ``'log_norm'`` and ``'max_norm'``.
        additional_input_vars : Sequence[str], optional
            Extra inputs next to TBs. Defaults are for 
            Zenith: ``['TB_IR', 'surface_T', 'surface_p', 'surface_rh', 'doy_cos', 'doy_sin']``
            and for BLS: ``['surface_T', 'surface_p', 'surface_rh', 'doy_cos', 'doy_sin']``
        loss_weights : mapping
            Required for neural profile models. Specifies every output weight;
            profile weights accept scalars, one value per height, or a table
            with inclusive height ``upper_bounds`` and one more ``values`` entry.
            MLR models omit this setting.

        Notes
        -----
        - Defines ``self.input`` (TB + auxiliary inputs) and the selected
          ``self.output`` profile/column variables used by training and
          inference helpers in ``openMWR.predict``.
        - Initializes height-dependent ``loss_weights`` for profile variables
          when using the MLP.
        - Computes ``n_input``, ``n_output``, and ``output_slices`` so tensors
          can be stacked/reshaped consistently during training and inference.
        """
        super().__init__(name)

        check_output_vars(output_vars)
        profiles = bool(set(output_vars) & {'T', 'rh', 'lwc'})
        if profiles and heights is None:
            raise ValueError('Profile outputs require model.heights.')
        if not profiles and heights is not None:
            raise ValueError('model.heights applies only to profile outputs. Use heights=None for column outputs.')
        self.heights = np.asarray(heights) if profiles else None
        self.freqs = np.asarray(freqs)
        self.is_bls_model = is_bls_model
        self.is_MLR = is_MLR

        if not is_MLR:
            self.hidden_layers = list(hidden_layers) if hidden_layers is not None else [1024, 64, 1024]
            self.dropout_rate = dropout_rate

        if is_bls_model:
            if angles is None:
                raise ValueError('BLS models require explicit angles; load a named project grid or provide numeric angles.')
            self.vary_input_noise_with_angle = vary_input_noise_with_angle
            self.vary_input_noise_with_angle_linear = vary_input_noise_with_angle_linear
            self.angles = list(angles)

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

        heights_da = (xr.DataArray(np.zeros(self.heights.shape), dims='height', coords={'height': self.heights})
                      if profiles else None)
        self.output = xr.Dataset({name: heights_da if name in ('T', 'rh', 'lwc') else 1 for name in output_vars})
        
        if not is_MLR or loss_weights is not None:
            self.set_loss_weights(loss_weights)

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

    def set_loss_weights(self, weights):
        """Set and validate weights used by both fitting and optimization scores."""
        if self.is_MLR:
            raise ValueError('model.loss_weights is not used by mlr models.')
        if weights is None:
            raise ValueError('Neural profile models require explicit model.loss_weights for every output.')
        if not isinstance(weights, Mapping):
            raise ValueError('model.loss_weights must be a table with explicit weights for every output.')
        weights = dict(weights)
        check_keys(weights, self.output.data_vars, 'model.loss_weights')
        require_keys(weights, self.output.data_vars, 'model.loss_weights')
        resolved = {}
        for name, template in self.output.data_vars.items():
            label = f'model.loss_weights.{name}'
            definition = weights[name]
            banded = isinstance(definition, Mapping)
            if banded:
                if 'height' not in template.dims:
                    raise ValueError(f'{label}: height bands apply only to profile outputs.')
                definition = dict(definition)
                check_keys(definition, ('upper_bounds', 'values'), label)
                require_keys(definition, ('upper_bounds', 'values'), label)
                bounds = definition['upper_bounds']
                check_coordinates(bounds, f'{label}.upper_bounds')
                if bounds != sorted(bounds):
                    raise ValueError(f'{label}.upper_bounds must be increasing (metres).')
                values = np.asarray(definition['values'])
                if values.ndim != 1 or len(values) != len(bounds) + 1:
                    raise ValueError(f'{label}.values must contain one more weight than upper_bounds.')
            else:
                values = np.asarray(definition)
            value_label = f'{label}.values' if banded else label
            for value in values.ravel():
                check_number(value, value_label, 0)
            if banded:
                values = values[np.searchsorted(bounds, self.heights, side='left')]
            if values.ndim and values.shape != template.shape:
                raise ValueError(f'{label} must be a scalar or have shape {template.shape}.')
            resolved[name] = xr.DataArray(np.broadcast_to(values, template.shape).copy(),
                                          dims=template.dims, coords=template.coords)
        if not any(np.any(data.values > 0) for data in resolved.values()):
            raise ValueError('model.loss_weights must include at least one positive weight.')
        self.loss_weights = xr.Dataset(resolved)

    def forward(self, x):
        """
        Forward pass through the retrieval network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_input)`` created by
            :func:`openMWR.predict.create_tensor_from_ds`.

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
            for name in ('lwc', 'lwp'):
                if name in self.output_slices:
                    y[:, self.output_slices[name]] = self.relu(y[:, self.output_slices[name]])

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

        self.freqs = np.asarray(freqs)
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
