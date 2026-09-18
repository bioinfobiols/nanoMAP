import os
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import anndata
import pytest
from scipy.sparse import csr_matrix

from joint.backends.cardinal import run_cardinal
from joint.errors import InputFormatError, OptionalDependencyError


def _write_imzml_pair(directory: Path, name: str = "sample") -> Path:
    imzml = directory / f"{name}.imzML"
    imzml.write_text("xml")
    imzml.with_suffix(".ibd").write_bytes(b"ibd")
    return imzml


def _successful_cardinal_run(command, check, capture_output, text):
    _write_imzml_pair(Path(command[3]), "cardinal")
    return type("Completed", (), {"stdout": "ok", "stderr": ""})()


def _result() -> anndata.AnnData:
    return anndata.AnnData(csr_matrix([[1.0]]), var={"mz": [100.0]})


def test_missing_rscript_has_precise_optional_dependency_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: None)

    with pytest.raises(OptionalDependencyError, match="Rscript"):
        run_cardinal(tmp_path / "sample.imzML", tmp_path / "out")


def test_cardinal_runs_versioned_script_in_temporary_directory_and_promotes_output(
    monkeypatch, tmp_path: Path
):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    commands = []
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")

    def fake_run(command, check, capture_output, text):
        commands.append(command)
        assert command[-3:] == ["10.0", "ppm", "3.0"]
        _write_imzml_pair(Path(command[3]), "cardinal")
        return type("Completed", (), {"stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", fake_run)
    monkeypatch.setattr("joint.backends.cardinal.read_imzml", lambda path, **kwargs: _result())

    result = run_cardinal(input_path, output_dir)

    assert result.shape == (1, 1)
    assert Path(commands[0][3]) != output_dir.resolve()
    assert Path(commands[0][3]).parent == output_dir.resolve().parent
    assert (output_dir / "cardinal.imzML").is_file()
    assert (output_dir / "cardinal.ibd").is_file()
    assert result.uns["joint"]["source"] == str(output_dir / "cardinal.imzML")


def test_cardinal_passes_dataset_id_to_imzml_reader(monkeypatch, tmp_path: Path):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    reader_arguments = {}
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")
    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", _successful_cardinal_run)

    def fake_read(path, *, dataset_id=None):
        reader_arguments["path"] = path
        reader_arguments["dataset_id"] = dataset_id
        result = _result()
        result.obs["dataset"] = dataset_id
        return result

    monkeypatch.setattr("joint.backends.cardinal.read_imzml", fake_read)

    result = run_cardinal(input_path, output_dir, dataset_id="sample")

    assert reader_arguments["dataset_id"] == "sample"
    assert result.obs["dataset"].tolist() == ["sample"]


def test_cardinal_failure_retains_full_command_stdout_and_stderr(monkeypatch, tmp_path: Path):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")

    def fake_run(command, check, capture_output, text):
        raise subprocess.CalledProcessError(
            7, command, output="complete standard output", stderr="complete standard error"
        )

    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", fake_run)

    with pytest.raises(InputFormatError, match="Cardinal") as error:
        run_cardinal(input_path, output_dir)

    assert error.value.command[:3] == [
        "/usr/bin/Rscript",
        str(files("joint.backends").joinpath("cardinal_pipeline.R")),
        str(input_path.resolve()),
    ]
    assert Path(error.value.command[3]) != output_dir.resolve()
    assert error.value.stdout == "complete standard output"
    assert error.value.stderr == "complete standard error"
    assert "complete standard output" in str(error.value)
    assert "complete standard error" in str(error.value)
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
    assert not output_dir.exists()


def test_cardinal_wraps_oserror_with_empty_subprocess_diagnostics(monkeypatch, tmp_path: Path):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")
    monkeypatch.setattr(
        "joint.backends.cardinal.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("R launch failed")),
    )

    with pytest.raises(InputFormatError, match="Cardinal subprocess failed") as error:
        run_cardinal(input_path, output_dir)

    assert error.value.command[0] == "/usr/bin/Rscript"
    assert error.value.stdout == ""
    assert error.value.stderr == ""
    assert isinstance(error.value.__cause__, OSError)
    assert not output_dir.exists()


def test_cardinal_success_retains_command_stdout_and_stderr_diagnostics(monkeypatch, tmp_path: Path):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")

    def fake_run(command, check, capture_output, text):
        _write_imzml_pair(Path(command[3]), "cardinal")
        return type("Completed", (), {"stdout": "all output", "stderr": "all errors"})()

    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", fake_run)
    monkeypatch.setattr("joint.backends.cardinal.read_imzml", lambda path, **kwargs: _result())

    result = run_cardinal(input_path, output_dir)

    assert result.uns["joint"]["cardinal"] == {
        "command": [
            "/usr/bin/Rscript",
            str(files("joint.backends").joinpath("cardinal_pipeline.R")),
            str(input_path.resolve()),
            result.uns["joint"]["cardinal"]["command"][3],
            "10.0",
            "ppm",
            "3.0",
        ],
        "stdout": "all output",
        "stderr": "all errors",
    }
    assert Path(result.uns["joint"]["cardinal"]["command"][3]) != output_dir.resolve()


def test_cardinal_output_read_failure_retains_subprocess_diagnostics_for_any_exception(
    monkeypatch, tmp_path: Path
):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")

    def fake_run(command, check, capture_output, text):
        _write_imzml_pair(Path(command[3]), "cardinal")
        return type("Completed", (), {"stdout": "pipeline output", "stderr": "pipeline errors"})()

    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", fake_run)
    monkeypatch.setattr(
        "joint.backends.cardinal.read_imzml",
        lambda path, **kwargs: (_ for _ in ()).throw(ValueError("cannot parse output")),
    )

    with pytest.raises(InputFormatError, match="Cardinal output could not be read") as error:
        run_cardinal(input_path, output_dir)

    assert error.value.stdout == "pipeline output"
    assert error.value.stderr == "pipeline errors"
    assert isinstance(error.value.__cause__, ValueError)
    assert not output_dir.exists()


def test_cardinal_rejects_missing_output_ibd_pair_without_final_partial_output(
    monkeypatch, tmp_path: Path
):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")

    def fake_run(command, check, capture_output, text):
        (Path(command[3]) / "cardinal.imzML").write_text("xml")
        return type("Completed", (), {"stdout": "completed", "stderr": ""})()

    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", fake_run)

    with pytest.raises(InputFormatError, match="cardinal.ibd") as error:
        run_cardinal(input_path, output_dir)

    assert error.value.stdout == "completed"
    assert error.value.stderr == ""
    assert not output_dir.exists()


def test_cardinal_rejects_existing_nonempty_output_without_explicit_overwrite(
    monkeypatch, tmp_path: Path
):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old-result").write_text("keep")
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")
    monkeypatch.setattr(
        "joint.backends.cardinal.subprocess.run",
        lambda *args, **kwargs: pytest.fail("Cardinal must not run before overwrite is explicit"),
    )

    with pytest.raises(ValueError, match="overwrite=True"):
        run_cardinal(input_path, output_dir)

    assert (output_dir / "old-result").read_text() == "keep"


def test_cardinal_overwrite_restores_existing_output_if_promotion_fails(monkeypatch, tmp_path: Path):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old-result").write_text("keep")
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")
    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", _successful_cardinal_run)
    monkeypatch.setattr("joint.backends.cardinal.read_imzml", lambda path, **kwargs: _result())
    original_replace = os.replace

    def fail_new_promotion(source, destination):
        if Path(source).name.startswith(".out.cardinal-") and Path(destination) == output_dir:
            raise OSError("cannot promote new result")
        return original_replace(source, destination)

    monkeypatch.setattr("joint.backends.cardinal.os.replace", fail_new_promotion)

    with pytest.raises(InputFormatError, match="Cardinal output promotion failed") as error:
        run_cardinal(input_path, output_dir, overwrite=True)

    assert isinstance(error.value.__cause__, OSError)
    assert (output_dir / "old-result").read_text() == "keep"
    assert not list(tmp_path.glob(".out.cardinal-*"))
    assert not list(tmp_path.glob(".out.backup-*"))


def test_cardinal_returns_success_when_promoted_backup_cleanup_fails(monkeypatch, tmp_path: Path):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old-result").write_text("replace")
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")
    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", _successful_cardinal_run)
    monkeypatch.setattr("joint.backends.cardinal.read_imzml", lambda path, **kwargs: _result())
    original_rmtree = shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".out.backup-"):
            raise OSError("backup cleanup failed")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("joint.backends.cardinal.shutil.rmtree", fail_backup_cleanup)

    result = run_cardinal(input_path, output_dir, overwrite=True)

    assert (output_dir / "cardinal.imzML").is_file()
    assert result.uns["joint"]["cardinal"]["cleanup_warnings"] == [
        {
            "path": str(next(tmp_path.glob(".out.backup-*"))),
            "error": "backup cleanup failed",
        }
    ]


def test_cardinal_staging_cleanup_failure_does_not_mask_subprocess_error(monkeypatch, tmp_path: Path):
    input_path = _write_imzml_pair(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")

    def fail_run(command, check, capture_output, text):
        raise subprocess.CalledProcessError(
            7, command, output="complete standard output", stderr="complete standard error"
        )

    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", fail_run)
    monkeypatch.setattr(
        "joint.backends.cardinal.shutil.rmtree",
        lambda path, *args, **kwargs: (_ for _ in ()).throw(OSError("staging cleanup failed")),
    )

    with pytest.raises(InputFormatError, match="Cardinal subprocess failed") as error:
        run_cardinal(input_path, output_dir)

    assert error.value.command[0] == "/usr/bin/Rscript"
    assert error.value.stdout == "complete standard output"
    assert error.value.stderr == "complete standard error"
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
    assert any("staging cleanup failed" in note for note in error.value.__notes__)


def test_cardinal_pipeline_r_script_is_available_as_a_resource():
    script = files("joint.backends").joinpath("cardinal_pipeline.R")

    assert script.is_file()
