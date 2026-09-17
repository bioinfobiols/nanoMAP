"""Subprocess adapter for the optional Cardinal R preprocessing backend."""

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from importlib.resources import files
from pathlib import Path

from anndata import AnnData

from joint.errors import InputFormatError, OptionalDependencyError
from joint.io import read_imzml


def _diagnostic_error(
    message: str,
    *,
    command: list[str],
    stdout: str | None,
    stderr: str | None,
) -> InputFormatError:
    """Create an auditable JOINT error for a Cardinal subprocess result."""
    stdout = "" if stdout is None else stdout
    stderr = "" if stderr is None else stderr
    error = InputFormatError(
        f"{message}; command={command!r}; stdout={stdout!r}; stderr={stderr!r}"
    )
    error.command = command
    error.stdout = stdout
    error.stderr = stderr
    return error


def _validate_input_pair(source: Path) -> None:
    if source.suffix.lower() != ".imzml":
        raise InputFormatError(f"Cardinal input must be an imzML file: {source}")
    if not source.is_file():
        raise InputFormatError(f"Cardinal imzML file does not exist: {source}")
    ibd = source.with_suffix(".ibd")
    if not ibd.is_file():
        raise InputFormatError(f"Matching ibd file does not exist: {ibd}")


def _validate_destination(destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ValueError(f"Cardinal output path must be a directory: {destination}")
    if any(destination.iterdir()) and not overwrite:
        raise ValueError(
            f"Cardinal output directory already contains files: {destination}; "
            "pass overwrite=True to replace it"
        )


def _best_effort_cleanup(path: Path) -> dict[str, str] | None:
    """Remove a temporary directory without allowing cleanup to change the result."""
    try:
        shutil.rmtree(path)
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}
    return None


def _record_cleanup_warning(error: Exception, warning: dict[str, str]) -> None:
    """Attach cleanup diagnostics without replacing the exception in flight."""
    error.cleanup_warnings = [*getattr(error, "cleanup_warnings", []), warning]
    error.add_note(f"Cardinal cleanup warning: {warning['path']}: {warning['error']}")


def _promote_directory(staging: Path, destination: Path) -> list[dict[str, str]]:
    """Atomically promote staging, restoring a prior destination on failure."""
    if not destination.exists():
        os.replace(staging, destination)
        return []

    backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except OSError:
        os.replace(backup, destination)
        raise
    warning = _best_effort_cleanup(backup)
    return [] if warning is None else [warning]


def run_cardinal(
    input_imzml: str | Path,
    output_dir: str | Path,
    *,
    rscript: str = "Rscript",
    tolerance: float = 10.0,
    unit: str = "ppm",
    snr: float = 3.0,
    dataset_id: str | None = None,
    overwrite: bool = False,
) -> AnnData:
    """Run Cardinal transactionally and read its promoted imzML/ibd output pair."""
    executable = shutil.which(rscript)
    if executable is None:
        raise OptionalDependencyError(
            "Rscript is required for the Cardinal backend; install R and Cardinal"
        )

    source = Path(input_imzml).expanduser().resolve()
    _validate_input_pair(source)
    destination = Path(output_dir).expanduser().resolve()
    _validate_destination(destination, overwrite=overwrite)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.cardinal-", dir=destination.parent)
    )
    script = files("joint.backends").joinpath("cardinal_pipeline.R")
    command = [
        executable,
        str(script),
        str(source),
        str(staging),
        str(tolerance),
        unit,
        str(snr),
    ]
    try:
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise _diagnostic_error(
                "Cardinal subprocess failed",
                command=command,
                stdout=exc.stdout,
                stderr=exc.stderr,
            ) from exc
        except Exception as exc:
            raise _diagnostic_error(
                "Cardinal subprocess failed",
                command=command,
                stdout="",
                stderr="",
            ) from exc

        result_path = staging / "cardinal.imzML"
        result_ibd = result_path.with_suffix(".ibd")
        if not result_path.is_file() or not result_ibd.is_file():
            raise _diagnostic_error(
                f"Cardinal completed without output imzML/ibd pair: {result_path}, {result_ibd}",
                command=command,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        try:
            result = read_imzml(result_path, dataset_id=dataset_id)
        except Exception as exc:
            raise _diagnostic_error(
                "Cardinal output could not be read",
                command=command,
                stdout=completed.stdout,
                stderr=completed.stderr,
            ) from exc
        try:
            cleanup_warnings = _promote_directory(staging, destination)
        except OSError as exc:
            raise _diagnostic_error(
                "Cardinal output promotion failed",
                command=command,
                stdout=completed.stdout,
                stderr=completed.stderr,
            ) from exc

        result.uns.setdefault("joint", {})["source"] = str(destination / "cardinal.imzML")
        result.uns["joint"]["cardinal"] = {
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if cleanup_warnings:
            result.uns["joint"]["cardinal"]["cleanup_warnings"] = cleanup_warnings
        return result
    finally:
        if staging.exists():
            warning = _best_effort_cleanup(staging)
            if warning is not None:
                active_error = sys.exception()
                if isinstance(active_error, Exception):
                    _record_cleanup_warning(active_error, warning)
