from pathlib import Path

import pytest

import torchMWRT.lineshape as lineshape


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_ensure_lineshape_data_copies_only_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lineshape,
        "REQUIRED_LINESHAPE_FILES",
        ("h2o_lineshape.nc", "o2_lineshape.nc", "o3_lineshape.nc"),
    )
    source_dir = tmp_path / "pyrtlib" / "_lineshape"
    dst_dir = tmp_path / "torchMWRT" / "_lineshape"

    _write(source_dir / "h2o_lineshape.nc", "src-h2o")
    _write(source_dir / "o2_lineshape.nc", "src-o2")
    _write(source_dir / "o3_lineshape.nc", "src-o3")
    _write(dst_dir / "h2o_lineshape.nc", "dst-h2o")

    monkeypatch.setattr(lineshape, "_discover_pyrtlib_lineshape_dir", lambda: source_dir)

    returned = lineshape.ensure_lineshape_data_available(dst_dir)

    assert returned == dst_dir
    assert (dst_dir / "h2o_lineshape.nc").read_text(encoding="utf-8") == "dst-h2o"
    assert (dst_dir / "o2_lineshape.nc").read_text(encoding="utf-8") == "src-o2"
    assert (dst_dir / "o3_lineshape.nc").read_text(encoding="utf-8") == "src-o3"


def test_ensure_lineshape_data_skips_lookup_when_all_files_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lineshape,
        "REQUIRED_LINESHAPE_FILES",
        ("h2o_lineshape.nc", "o2_lineshape.nc", "o3_lineshape.nc"),
    )
    dst_dir = tmp_path / "torchMWRT" / "_lineshape"
    for name in lineshape.REQUIRED_LINESHAPE_FILES:
        _write(dst_dir / name, "ok")

    def _should_not_be_called():
        raise AssertionError("pyrtlib lookup should not run when files are present")

    monkeypatch.setattr(lineshape, "_discover_pyrtlib_lineshape_dir", _should_not_be_called)

    returned = lineshape.ensure_lineshape_data_available(dst_dir)

    assert returned == dst_dir


def test_ensure_lineshape_data_raises_when_pyrtlib_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lineshape,
        "REQUIRED_LINESHAPE_FILES",
        ("h2o_lineshape.nc", "o2_lineshape.nc", "o3_lineshape.nc"),
    )
    dst_dir = tmp_path / "torchMWRT" / "_lineshape"
    monkeypatch.setattr(lineshape, "_discover_pyrtlib_lineshape_dir", lambda: None)

    with pytest.raises(FileNotFoundError, match="pyrtlib/_lineshape"):
        lineshape.ensure_lineshape_data_available(dst_dir)
