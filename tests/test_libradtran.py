import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openMWR.libRadtran import get_libradtran_paths, libRadtran
from openMWR.run_RT import calculate_IR_band_TB


@pytest.fixture(autouse=True)
def isolated_discovery(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("LIBRADTRAN_DATA_FILES", raising=False)


@pytest.fixture
def installation():
    def create(root, data_layout="data"):
        executable = root / "bin" / "uvspec"
        executable.parent.mkdir(parents=True)
        data = root / data_layout
        (data / "atmmod").mkdir(parents=True)
        (data / "atmmod" / "afglus.dat").write_text("test atmosphere\n")
        recorded_input = root / "input.txt"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import sys\nfrom pathlib import Path\n"
            "text = sys.stdin.read()\n"
            f"Path({str(recorded_input)!r}).write_text(text)\n"
            "umu = next((line.split()[1:] for line in text.splitlines() "
            "if line.startswith('umu ')), [0])\n"
            "print(' '.join(['199.0'] * len(umu)))\n"
        )
        executable.chmod(0o755)
        return executable, data, recorded_input
    return create


@pytest.mark.parametrize("discovery", ["local", "path", "symlink", "shared"])
def test_discover_installation(discovery, tmp_path, monkeypatch, installation):
    root = Path.cwd() / "libRadtran-2.0.6" if discovery == "local" else tmp_path / "install"
    layout = "share/libRadtran/data" if discovery == "shared" else "data"
    executable, data, _ = installation(root, layout)
    if discovery == "symlink":
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "uvspec").symlink_to(executable)
        monkeypatch.setenv("PATH", str(scripts))
    elif discovery != "local":
        monkeypatch.setenv("PATH", str(executable.parent))
    assert get_libradtran_paths() == (executable.resolve(), data.resolve())


def test_local_installation_precedes_path(tmp_path, monkeypatch, installation):
    local, data, _ = installation(Path.cwd() / "libRadtran-2.0.6")
    external, _, _ = installation(tmp_path / "external")
    monkeypatch.setenv("PATH", str(external.parent))
    assert get_libradtran_paths() == (local, data)


def test_empty_data_directory_does_not_hide_shared_data(tmp_path, monkeypatch, installation):
    root = tmp_path / "install"
    executable, data, _ = installation(root, "share/libRadtran/data")
    (root / "data").mkdir()
    monkeypatch.setenv("PATH", str(executable.parent))
    assert get_libradtran_paths() == (executable, data)


def test_data_overrides(tmp_path, monkeypatch, installation):
    executable, _, _ = installation(tmp_path / "install")
    monkeypatch.setenv("PATH", str(executable.parent))
    external_data = tmp_path / "external_data"
    external_data.mkdir()
    explicit_data = tmp_path / "explicit_data"
    explicit_data.mkdir()
    monkeypatch.setenv("LIBRADTRAN_DATA_FILES", str(external_data))
    assert get_libradtran_paths()[1] == external_data
    assert get_libradtran_paths(explicit_data)[1] == explicit_data


def test_missing_executable():
    with pytest.raises(FileNotFoundError, match="add your libRadtran bin directory to PATH"):
        get_libradtran_paths()


def test_incomplete_local_installation(tmp_path, monkeypatch, installation):
    (Path.cwd() / "libRadtran-2.0.6").mkdir()
    external, _, _ = installation(tmp_path / "external")
    monkeypatch.setenv("PATH", str(external.parent))
    with pytest.raises(FileNotFoundError, match="uvspec not found at"):
        get_libradtran_paths()


def test_non_executable_binary(installation):
    executable, _, _ = installation(Path.cwd() / "libRadtran-2.0.6")
    executable.chmod(0o644)
    with pytest.raises(PermissionError, match="not executable"):
        get_libradtran_paths()


@pytest.mark.parametrize("override", [False, True])
def test_missing_data(tmp_path, monkeypatch, installation, override):
    executable, data, _ = installation(tmp_path / "install")
    monkeypatch.setenv("PATH", str(executable.parent))
    if override:
        monkeypatch.setenv("LIBRADTRAN_DATA_FILES", str(tmp_path / "missing"))
    else:
        (data / "atmmod" / "afglus.dat").unlink()
    with pytest.raises(FileNotFoundError, match="Set LIBRADTRAN_DATA_FILES"):
        get_libradtran_paths()


def test_wrapper_uses_explicit_data_path(tmp_path, monkeypatch, installation):
    executable, _, recorded = installation(tmp_path / "install")
    monkeypatch.setenv("PATH", str(executable.parent))
    data = tmp_path / "separate_data"
    data.mkdir()
    result, stderr = libRadtran(data_files_path=data, output_user="uu", umu=[-1.0])
    assert result["uu"].iloc[0].item() == 199.0
    assert stderr == ""
    assert f"data_files_path {data}\n" in recorded.read_text()


def test_relative_data_path_keeps_uvspec_working_directory(tmp_path, monkeypatch, installation):
    executable, data, recorded = installation(tmp_path / "install")
    monkeypatch.setenv("PATH", str(executable.parent))
    monkeypatch.setenv("LIBRADTRAN_DATA_FILES", "../data")
    assert get_libradtran_paths()[1] == data
    result, _ = libRadtran(data_files_path="../data", output_user="uu", umu=[-1.0])
    assert result["uu"].iloc[0].item() == 199.0
    assert f"data_files_path {data}\n" in recorded.read_text()


@pytest.mark.parametrize("gen,band", [("G4", "9200 10600"), ("G5", "9600 11500")])
@pytest.mark.parametrize("angles", [[90.0], [30.0, 90.0]])
def test_ir_uses_discovered_atmosphere(tmp_path, monkeypatch, installation, gen, band, angles):
    executable, data, recorded = installation(tmp_path / "install", "share/libRadtran/data")
    monkeypatch.setenv("PATH", str(executable.parent))
    data_dir = tmp_path / "retrieval_data"
    config = data_dir / "sites" / "test" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"gen": gen}))
    result = calculate_IR_band_TB(
        "test", data_dir,
        z_km=[0.0, 1.0], T_K=[290.0, 280.0], rh_100=[50.0, 50.0],
        p_hPa=[1000.0, 900.0], lwc_gpm3=[0.0, 0.0],
        effective_droplet_radius_mum=[10.0, 10.0], ang=np.array(angles),
    )
    np.testing.assert_array_equal(result, 199.0 if len(angles) == 1 else [199.0, 199.0])
    text = recorded.read_text()
    assert f"atmosphere_file {data / 'atmmod/afglus.dat'}\n" in text
    assert f"data_files_path {data}\n" in text
    assert f"wavelength {band}\n" in text
    assert not list((data_dir / "libRadtran").iterdir())
