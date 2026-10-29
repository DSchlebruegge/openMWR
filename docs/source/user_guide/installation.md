# Installation

## Install openMWR

Use Python 3.11 or newer. Keep the package checkout separate from the working
project containing your site settings, datasets and retrievals:

```bash
git clone https://github.com/DSchlebruegge/openMWR.git
mkdir my_project
cd my_project
```

Create a virtual environment and install the neighbouring checkout with either
uv or pip.

### With uv

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --editable ../openMWR
```

### With pip

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --editable ../openMWR
```

Both install the `openMWR` command. The editable installation uses the source
from the checkout directly. Check the installation:

```bash
openMWR --help
```

The directory layout is:

```text
openMWR/            package checkout
my_project/        working project. Run tutorial commands here
    .venv/
```

Activate this environment again when opening a new shell. `python -m openMWR`
is equivalent to `openMWR`. The commands work with either installation method.
To work from another directory, pass `--project-dir /path/to/my_project` to a
command. `openMWR COMMAND --help` lists its options and short aliases.

## Install libRadtran

libRadtran is a required dependency for the forward calculation of the infrared sensor measurements on
a HATPRO MWR.

libRadtran is GPL-licensed third-party software. This project does not bundle or distribute libRadtran.
Users install libRadtran separately. See: <https://www.libradtran.org/doku.php?id=download>
The commands below install it inside the working project created above:

Download libRadtran 2.0.6:
```bash
libradtran_version=2.0.6
wget https://www.libradtran.org/download/libRadtran-${libradtran_version}.tar.gz
```

Unpack the distribution:
```bash
gzip -d libRadtran-${libradtran_version}.tar.gz
tar -xvf libRadtran-${libradtran_version}.tar
rm libRadtran-${libradtran_version}.tar
```

Compile the distribution:
```bash
cd libRadtran-${libradtran_version}
./configure
make
```

Test the program:
```bash
make check
```

From the libRadtran directory, download the REPTRAN absorption data:
```bash
wget 'https://www.libradtran.org/lib/exe/fetch.php?media=download:reptran_2024_all.tar.gz' -O reptran_2024_all.tar.gz

gzip -d reptran_2024_all.tar.gz
tar -xvf reptran_2024_all.tar
rm reptran_2024_all.tar
```

openMWR first looks for a `libRadtran-*` directory in the current working directory, then for `uvspec` on `PATH`.
For an installation elsewhere, add its `bin` directory to `PATH` once in your shell configuration (e.g. `~/.bashrc` or `~/.zshrc`):

```bash
export PATH="$HOME/opt/libRadtran-2.0.6/bin:$PATH"
```

The data directory is found automatically at `data/` or `share/libRadtran/data/` under the installation prefix.
If you keep the data elsewhere, set `LIBRADTRAN_DATA_FILES` to that directory.

Return to the working project:

```bash
cd ..
```

Continue with [Create a Retrieval](create_retrieval.md).
