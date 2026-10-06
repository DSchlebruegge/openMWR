import pandas as pd
import numpy as np
import subprocess
import shutil
import os
from pathlib import Path


def get_libradtran_paths(data_files_path: str | Path | None = None) -> tuple[Path, Path]:
    """Find uvspec and its data, preferring a local libRadtran-* installation.

    Otherwise, find uvspec on PATH and resolve symlinks to its installation.
    Explicit data_files_path and LIBRADTRAN_DATA_FILES take precedence over
    the installation's data/ or share/libRadtran/data/ directory.
    Automatic discovery checks for the standard atmosphere at atmmod/afglus.dat.
    Relative data paths are interpreted from uvspec's bin directory.
    """
    local_roots = sorted(p for p in Path.cwd().glob("libRadtran-*") if p.is_dir())
    if local_roots:
        uvspec = local_roots[0] / "bin" / "uvspec"
    else:
        executable = shutil.which("uvspec")
        if executable is None:
            raise FileNotFoundError(
                "uvspec not found. Place libRadtran-* in the working directory "
                "or add your libRadtran bin directory to PATH."
            )
        uvspec = Path(executable)

    uvspec = uvspec.resolve()
    if not uvspec.is_file():
        raise FileNotFoundError(f"uvspec not found at: {uvspec}")
    if not os.access(uvspec, os.X_OK):
        raise PermissionError(f"uvspec exists but is not executable: {uvspec}")

    data_override = data_files_path or os.environ.get("LIBRADTRAN_DATA_FILES")
    if data_override:
        data_dir = Path(data_override).expanduser()
        if not data_dir.is_absolute():
            data_dir = uvspec.parent / data_dir
        data_dir = data_dir.resolve()
    else:
        prefix = uvspec.parent.parent
        candidates = [prefix / "data", prefix / "share" / "libRadtran" / "data"]
        data_dir = next((p for p in candidates if (p / "atmmod" / "afglus.dat").is_file()), None)

    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(
            f"libRadtran data directory not found at {data_dir or uvspec.parent.parent}. "
            "Set LIBRADTRAN_DATA_FILES to the directory containing the libRadtran data."
        )
    return uvspec, data_dir.resolve()

def libRadtran(**input_dic: dict) -> 'tuple[pd.DataFrame, str]':
    """
    Run libRadtran's uvspec with the given input parameters.
    Remouves the need for an input file by passing parameters directly.

    Parameters
    ----------
    **input_dic : dict
        Key-value pairs corresponding to libRadtran input parameters.

    Returns
    -------
    tuple[pd.DataFrame, str]
        A tuple containing:
        - A pandas DataFrame with the output data.
        - A string with any stderr output from the uvspec command.
        
    Raises
    ------
    libRadtranError
        If there is an error during the execution of uvspec or in its output.
    """
    uvspec_exe, data_dir = get_libradtran_paths(input_dic.get("data_files_path"))
    input_dic["data_files_path"] = str(data_dir)

    input_text = ""
    parsed = {} 

    for key, value in input_dic.items():
        if isinstance(value, bool):
            if value:
                input_text += f"{key}\n"
            parsed[key] = value
        elif isinstance(value, (list, np.ndarray)):
            input_text += f"{key} " + " ".join(map(str, value)) + "\n"
            parsed[key] = list(value)
        else:
            input_text += f"{key} {value}\n"
            parsed[key] = value.split() if isinstance(value, str) else value
    
    #print(input_text)

    class libRadtranError(Exception):
        pass
    

    try:
        result = subprocess.run(
            [str(uvspec_exe)],
            input=input_text,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",
            cwd=uvspec_exe.parent
        )
    except subprocess.CalledProcessError as exc:
        raise libRadtranError(f'\n{exc.stderr}\nInput File:\n{input_text}') from None 

    if 'error' in result.stdout or 'Error' in result.stdout:
        raise libRadtranError(f'\nError in stdout:\n{result.stdout}\nstderr:\n{result.stderr}\nInput File:\n{input_text}')
    
    #print(result.stderr)
    #print(result.stdout)
        
    data = [line.split() for line in result.stdout.strip().split('\n')]

    columns = parsed.get("output_user", ['lambda', 'edir', 'edn', 'eup', 'uavgdir', 'uavgdn', 'uavgup'])
    if isinstance(columns, str):
        columns = columns.split()

    if "umu" in parsed and parsed["umu"]:
        new_columns = []
        for col in columns:
            if col == "uu":
                new_columns.extend([(col, umu) for umu in parsed["umu"]])
            else:
                new_columns.append((col, ""))
        columns = pd.MultiIndex.from_tuples(new_columns)

    if data and len(data[0]) > len(columns):
        columns = list(columns) + [f"-{i}-" for i in range(len(data[0]) - len(columns))]

    df = pd.DataFrame(data, columns=columns).apply(pd.to_numeric, errors="coerce")
    return df, result.stderr
