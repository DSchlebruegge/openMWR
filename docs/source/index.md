# The openMWR package

> **Note:** This documentation is still in development and may contain errors or incomplete sections.

`openMWR` is a Python package for training a neural network-based retrieval for a Hatpro microwave radiometer.

It includes tools for generating training datasets, training retrieval algorithms,
analyzing their performance, and applying the trained
algorithms to measurements operationally. It can be used to create multilinear
regression retrievals, but it is focused on more complex neural network retrievals.
It also supports the training of retrieval algorithms for elevation scans, to study
the boundary layer.

`OpenMWR` also includes the `torchMWRT` package, a PyTorch translation of the radiative-transfer 
calculation core of [PyRTlib](https://github.com/SatCloP/pyrtlib). It calculates brightness temperatures in batches and supports GPU acceleration
and automatic differentiation.

Start with [Installation](user_guide/installation.md), then
[Create a Retrieval](user_guide/create_retrieval.md). The tutorial uses TOML
recipes and the `openMWR` commands. The same workflows are available
[from Python](user_guide/python_usage.md).

For an existing project, see [Upgrading from 0.1](user_guide/upgrading_from_0_1.md).

## User Guide

```{toctree}
:maxdepth: 2

user_guide/index
```

## API

```{toctree}
:maxdepth: 2

api/index
```

## Contributing

```{toctree}
:maxdepth: 2

contributing
```
