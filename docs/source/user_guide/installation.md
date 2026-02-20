# Installation

## Install openMWR

Clone the repository:

```bash
git clone https://gitlab.physik.uni-muenchen.de/D.Schlebruegge/openMWR.git
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

If you want to build the documentation, you also need to install the dependencies for that:

```bash
pip install -e ".[docs]"
```

## Install libRadtran

libRadtran is a required dependency for the forward calculation of the infrared sensor measurements on
an Hatpro MWR. 

libRadtran is GPL-licensed third-party software. This project does not bundle or distribute libRadtran.
Users install libRadtran separately. See: <https://www.libradtran.org/doku.php?id=download>
To install libRadtran, follow these steps in the openMWR root directory:

Check for the latest version and download it:
```bash
libradtran_version=2.0.6
wget https://www.libradtran.org/download/libRadtran-${libradtran_version}.tar.gz
```

Unpack the distribution
```bash
gzip -d libRadtran-${libradtran_version}.tar.gz
tar -xvf libRadtran-${libradtran_version}.tar
rm libRadtran-${libradtran_version}.tar
```

Compile the distribution
```bash
cd libRadtran-${libradtran_version}
./configure
make
```

Test the program
```bash
make check
```

Download data for the REPTRAN absorption parameterization
```bash
wget http://www.meteo.physik.uni-muenchen.de/~libradtran/lib/exe/fetch.php?media=download:reptran_2024_all.tar.gz -O reptran_2024_all.tar.gz

gzip -d reptran_2024_all.tar.gz
tar -xvf reptran_2024_all.tar
rm reptran_2024_all.tar
```
