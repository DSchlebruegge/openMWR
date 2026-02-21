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
calculation core of [PyRTlib](https://github.com/SatCloP/pyrtlib). It produces the same results as 
`PyRTlib` while being orders of magnitude faster. That is mainly achieved by 
heavy vectorizations. It also supports GPU acceleration and automatic differentiation, 
which is, however, not necessary for the classical retrieval algorithms.

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