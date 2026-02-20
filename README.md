# openMWR

openMWR is a Python package for training a neural network-based retrieval for a Hatpro microwave radiometer.

It includes tools for generating training datasets, training retrieval algorithms,
analyzing the performance of retrieval algorithms, and applying the trained
algorithms to measurements operationally.

## Usage

For a full user guide, please refer to the [Documentation](https://dschlebruegge.github.io/openMWR/).


## Installation

Clone the repository:

```bash
git clone https://github.com/DSchlebruegge/openMWR.git
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the openMWR package as editable:

```bash
cd openMWR
pip install -e .
```

For more details on installation, please refer to the [Installation Guide](https://dschlebruegge.github.io/openMWR/user_guide/installation.html).

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the [LICENSE](LICENSE) file for details.

## Attribution

This project contains a PyTorch port/translation of code derived from SatCloP/pyrtlib (GPL-3.0).
Upstream: https://github.com/SatCloP/pyrtlib
