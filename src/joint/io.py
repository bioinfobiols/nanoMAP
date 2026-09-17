"""Input and output helpers for JOINT AnnData datasets."""

import json
import os
import pickle
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import anndata
import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser

from joint._peaks import bin_profile_spectra, build_consensus_axis, project_spectra
from joint.errors import InputFormatError, PeakAlignmentError
from joint.models import QuantificationResult, RegistrationResult, SegmentationResult


@contextmanager
def _open_imzml_binary(path: Path) -> Iterator[BinaryIO]:
    handle = path.open("rb")
    try:
        yield handle
    finally:
        if not handle.closed:
            active_error = sys.exc_info()[1]
            try:
                handle.close()
            except Exception as close_error:
                if active_error is not None:
                    active_error.add_note(f"Could not close imzML binary handle: {close_error}")
                else:
                    raise InputFormatError(
                        f"Could not close imzML binary handle: {close_error}"
                    ) from close_error


def read_imzml(
    path: str | Path,
    *,
    dataset_id: str | None = None,
    tolerance: float = 10.0,
    unit: str = "ppm",
    profile_bin_size: float | None = None,
) -> anndata.AnnData:
    """Read an imzML/ibd pair into an AnnData object with (x, y) coordinates."""
    imzml = Path(path).expanduser().resolve()
    if not imzml.is_file():
        raise InputFormatError(f"imzML file does not exist: {imzml}")
    ibd = imzml.with_suffix(".ibd")
    if not ibd.is_file():
        raise InputFormatError(f"Matching ibd file does not exist: {ibd}")

    try:
        with _open_imzml_binary(ibd) as handle:
            try:
                parser = ImzMLParser(str(imzml), ibd_file=handle)
                if len(parser.coordinates) == 0:
                    raise InputFormatError("imzML contains no pixel spectra")
                spectra = [parser.getspectrum(index) for index in range(len(parser.coordinates))]
                coordinates = np.asarray([(x, y) for x, y, _ in parser.coordinates], dtype=float)
            except InputFormatError:
                raise
            except Exception as exc:
                raise InputFormatError(f"Could not parse imzML file {imzml}: {exc}") from exc
    except InputFormatError:
        raise
    except Exception as exc:
        raise InputFormatError(f"Could not parse imzML file {imzml}: {exc}") from exc

    try:
        mz_arrays = [np.asarray(mzs) for mzs, _ in spectra]
        shared_axis = bool(mz_arrays) and all(
            np.array_equal(mz_arrays[0], candidate) for candidate in mz_arrays[1:]
        )
        if profile_bin_size is not None:
            axis, matrix = bin_profile_spectra(spectra, bin_size=profile_bin_size)
        else:
            axis = (
                mz_arrays[0].copy()
                if shared_axis
                else build_consensus_axis(mz_arrays, tolerance=tolerance, unit=unit)
            )
            matrix = project_spectra(spectra, axis, tolerance=tolerance, unit=unit)
    except PeakAlignmentError as exc:
        raise InputFormatError(f"Could not parse imzML file {imzml}: {exc}") from exc

    sample = dataset_id or imzml.stem
    obs = pd.DataFrame(index=[f"p{index + 1}" for index in range(matrix.shape[0])])
    obs["dataset"] = sample
    var = pd.DataFrame(index=[f"m{index + 1}" for index in range(matrix.shape[1])])
    var["mz"] = axis.astype(float, copy=False)
    adata = anndata.AnnData(matrix, obs=obs, var=var, obsm={"spatial": coordinates})
    adata.uns["joint"] = {"source": str(imzml), "shared_mz_axis": shared_axis}
    return adata


def read_h5ad(path: str | Path) -> anndata.AnnData:
    """Read an h5ad file, exposing malformed inputs as a JOINT domain error."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputFormatError(f"h5ad file does not exist: {source}")
    try:
        return anndata.read_h5ad(source)
    except Exception as exc:
        raise InputFormatError(f"Could not read h5ad file {source}: {exc}") from exc


def read_segmentation(path: str | Path, *, trusted_pickle: bool = False) -> np.ndarray:
    """Read a two-dimensional integer segmentation label array.

    Pickle inputs require explicit caller authorization because deserialization can
    execute arbitrary code.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputFormatError(f"Segmentation file does not exist: {source}")
    pickle_input = False
    try:
        if source.suffix == ".npy":
            labels = np.load(source, allow_pickle=False)
        elif source.suffix in {".pkl", ".pickle"}:
            if not trusted_pickle:
                raise InputFormatError("Pickle segmentation requires trusted_pickle=True")
            pickle_input = True
            with source.open("rb") as handle:
                labels = pickle.load(handle)
        else:
            raise InputFormatError(f"Unsupported segmentation format: {source.suffix}")
    except InputFormatError:
        raise
    except Exception as exc:
        raise InputFormatError(f"Could not read segmentation file {source}: {exc}") from exc

    try:
        labels = np.asarray(labels)
        if pickle_input and labels.ndim == 4 and labels.shape[0] == labels.shape[-1] == 1:
            labels = labels[0, :, :, 0]
        if labels.ndim != 2 or labels.dtype.kind not in "iu":
            raise InputFormatError("Segmentation must be a two-dimensional integer label array")
        return labels
    except InputFormatError:
        raise
    except Exception as exc:
        raise InputFormatError(f"Could not validate segmentation file {source}: {exc}") from exc


def atomic_write_h5ad(adata: anndata.AnnData, path: str | Path) -> Path:
    """Atomically write an AnnData object and return its destination path."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.h5ad", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        adata.write_h5ad(temporary)
        os.replace(temporary, destination)
    except Exception as exc:
        raise InputFormatError(f"Could not write h5ad file {destination}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_save_npy(array: np.ndarray, destination: Path) -> Path:
    """Atomically write a NumPy array without overwriting an existing result early."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array)
        os.replace(temporary, destination)
    except Exception as exc:
        raise InputFormatError(f"Could not write NumPy file {destination}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_write_text(text: str, destination: Path) -> Path:
    """Atomically write a UTF-8 text artifact."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    except Exception as exc:
        raise InputFormatError(f"Could not write text file {destination}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_write_dataframe(table: pd.DataFrame, destination: Path) -> Path:
    """Atomically write a CSV table."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    except Exception as exc:
        raise InputFormatError(f"Could not write CSV file {destination}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_results(
    result: anndata.AnnData | SegmentationResult | RegistrationResult | QuantificationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a supported JOINT result bundle and return its named artifacts.

    Each artifact is atomically promoted after its own write completes. Bundle-wide
    promotion is owned by the pipeline stage store.
    """
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(result, anndata.AnnData):
        return {"adata": atomic_write_h5ad(result, destination / "result.h5ad")}
    if isinstance(result, SegmentationResult):
        labels_path = _atomic_save_npy(result.labels, destination / "labels.npy")
        regions_path = _atomic_write_dataframe(result.regions, destination / "regions.csv")
        serializable_diagnostics = {
            key: value
            for key, value in result.diagnostics.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        report_path = _atomic_write_text(
            json.dumps(serializable_diagnostics, indent=2, sort_keys=True),
            destination / "report.json",
        )
        return {"labels": labels_path, "regions": regions_path, "report": report_path}
    if isinstance(result, RegistrationResult):
        adata_path = atomic_write_h5ad(result.adata, destination / "laser.h5ad")
        mapping_path = _atomic_write_dataframe(result.mapping, destination / "mapping.csv")
        report_path = _atomic_write_text(
            json.dumps({**result.transform, **result.report}, indent=2, default=str),
            destination / "report.json",
        )
        return {"adata": adata_path, "mapping": mapping_path, "report": report_path}
    if isinstance(result, QuantificationResult):
        artifacts = {"adata": atomic_write_h5ad(result.adata, destination / "cells.h5ad")}
        for name, table in {
            "overlaps": result.overlaps,
            "accepted": result.accepted,
            "rejected": result.rejected,
        }.items():
            artifacts[name] = _atomic_write_dataframe(table, destination / f"{name}.csv")
        artifacts["report"] = _atomic_write_text(
            json.dumps(result.report, indent=2, default=str), destination / "report.json"
        )
        return artifacts
    raise TypeError(f"Unsupported JOINT result type: {type(result).__name__}")
