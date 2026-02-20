import pandas as pd
import numpy as np
import subprocess
import glob
import os

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
    libRadtran_root = next(p for p in glob.glob("./libRadtran-*") if os.path.isdir(p))
    libRadtran_bin = os.path.join(libRadtran_root, "bin")
    uvspec_exe = os.path.join(libRadtran_bin, "uvspec")

    if not os.path.isfile(uvspec_exe):
        raise FileNotFoundError(f"uvspec not found at: {uvspec_exe}")
    if not os.access(uvspec_exe, os.X_OK):
        raise PermissionError(f"uvspec exists but is not executable: {uvspec_exe}")

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
            ["./uvspec"],
            input=input_text,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",
            cwd=libRadtran_bin
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
