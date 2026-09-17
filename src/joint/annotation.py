"""Deterministic m/z reference annotation adapters."""

import json
from numbers import Real
from pathlib import Path
from typing import Final

import anndata
import numpy as np
import pandas as pd

from joint.errors import InputFormatError

_HMDB_COLUMNS: Final = frozenset({"accession", "name", "mz", "mode"})
_METABOSCAPE_COLUMNS: Final = frozenset({"Measured m/z", "Molecular Formula", "Name"})


def _load_table(reference: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Copy a reference table or load a comma- or tab-delimited file."""
    if isinstance(reference, pd.DataFrame):
        return reference.copy()
    if not isinstance(reference, (str, Path)):
        try:
            raise TypeError(f"received {type(reference).__name__}")
        except TypeError as exc:
            raise InputFormatError("reference must be a pandas DataFrame or a path") from exc

    try:
        source = Path(reference)
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() == ".csv":
            return pd.read_csv(source)
        return pd.read_table(source)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise InputFormatError(f"Could not read reference file {reference!s}: {exc}") from exc


def _require_exact_columns(table: pd.DataFrame, required: frozenset[str], label: str) -> None:
    if not table.columns.is_unique or set(table.columns) != required:
        raise InputFormatError(f"{label} reference must contain exactly columns: {sorted(required)}")


def _numeric_mz(values: pd.Series, label: str) -> np.ndarray:
    non_missing = values.dropna()
    raw_values = non_missing.to_numpy(dtype=object)
    if pd.api.types.is_bool_dtype(values) or any(
        isinstance(value, (bool, np.bool_)) for value in raw_values
    ):
        raise InputFormatError(f"{label} must be numeric")
    if pd.api.types.is_complex_dtype(values) or any(np.iscomplexobj(value) for value in raw_values):
        raise InputFormatError(f"{label} must be real numeric")
    try:
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputFormatError(f"{label} must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise InputFormatError(f"{label} must contain finite values")
    if not np.all(numeric > 0):
        raise InputFormatError(f"{label} must contain positive values")
    return numeric


def _validate_query(adata: object) -> tuple[anndata.AnnData, np.ndarray]:
    if not isinstance(adata, anndata.AnnData):
        raise InputFormatError("adata must be an AnnData object")
    if not adata.var_names.is_unique:
        raise InputFormatError("adata.var_names must be unique")
    if "mz" not in adata.var:
        raise InputFormatError('adata.var must contain numeric column "mz"')
    return adata, _numeric_mz(adata.var["mz"], 'adata.var["mz"]')


def _validate_ppm(ppm: object) -> float:
    if isinstance(ppm, (bool, np.bool_)) or not isinstance(ppm, Real):
        raise InputFormatError("ppm must be a finite non-negative number")
    value = float(ppm)
    if not np.isfinite(value) or value < 0:
        raise InputFormatError("ppm must be a finite non-negative number")
    return value


def _matches(query_mz: float, reference_mz: np.ndarray, ppm: float) -> np.ndarray:
    """Return matches within the inclusive ppm tolerance for one query m/z."""
    if ppm == 0:
        return reference_mz == query_mz
    query = np.longdouble(query_mz)
    references = reference_mz.astype(np.longdouble)
    tolerance = query * np.longdouble(ppm) / np.longdouble(1_000_000.0)
    reference_ulp = reference_mz - np.nextafter(reference_mz, 0.0)
    query_ulp = query_mz - np.nextafter(query_mz, 0.0)
    reference_ulp = reference_ulp.astype(np.longdouble)
    query_ulp = np.longdouble(query_ulp)
    allowance = np.longdouble(2.0) * (reference_ulp + query_ulp)
    return np.abs(references - query) <= tolerance + allowance


def _require_non_missing(table: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        if table[column].isna().any():
            raise InputFormatError(f"{label} reference column {column!r} must not contain missing values")


def _as_json(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _set_annotation_column(result: anndata.AnnData, name: str, rows: list[list[str]]) -> None:
    values = [_as_json(row) for row in rows]
    result.var[name] = pd.Series(values, index=result.var.index, dtype=object)


def annotate_hmdb(
    adata: anndata.AnnData,
    reference: pd.DataFrame | str | Path,
    *,
    mode: str,
    ppm: float = 5.0,
) -> anndata.AnnData:
    """Annotate features against an HMDB-formatted reference table.

    Every feature receives JSON list strings for all matching accessions and names,
    retaining the filtered reference table's original row order.
    """
    query, query_mz = _validate_query(adata)
    checked_ppm = _validate_ppm(ppm)
    if not isinstance(mode, str) or not mode.strip():
        raise InputFormatError("mode must be a non-empty string")

    table = _load_table(reference)
    _require_exact_columns(table, _HMDB_COLUMNS, "HMDB")
    _require_non_missing(table, ("accession", "name"), "HMDB")
    reference_mz = _numeric_mz(table["mz"], 'HMDB reference column "mz"')
    normalized_mode = mode.strip().casefold()
    mode_matches = (
        table["mode"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq(normalized_mode)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    selected_rows = table.loc[mode_matches]
    selected_mz = reference_mz[mode_matches]

    accessions: list[list[str]] = []
    names: list[list[str]] = []
    for value in query_mz:
        matched = selected_rows.loc[_matches(value, selected_mz, checked_ppm)]
        accessions.append(matched["accession"].astype(str).tolist())
        names.append(matched["name"].astype(str).tolist())

    result = query.copy()
    _set_annotation_column(result, "hmdb_accessions", accessions)
    _set_annotation_column(result, "hmdb_names", names)
    return result


def annotate_metaboscape(
    adata: anndata.AnnData,
    reference: pd.DataFrame | str | Path,
    *,
    ppm: float = 3.0,
) -> anndata.AnnData:
    """Annotate features against a MetaboScape-formatted reference table."""
    query, query_mz = _validate_query(adata)
    checked_ppm = _validate_ppm(ppm)

    table = _load_table(reference)
    _require_exact_columns(table, _METABOSCAPE_COLUMNS, "MetaboScape")
    _require_non_missing(table, ("Molecular Formula", "Name"), "MetaboScape")
    reference_mz = _numeric_mz(table["Measured m/z"], 'MetaboScape reference column "Measured m/z"')

    formulas: list[list[str]] = []
    names: list[list[str]] = []
    for value in query_mz:
        matched = table.loc[_matches(value, reference_mz, checked_ppm)]
        formulas.append(matched["Molecular Formula"].astype(str).tolist())
        names.append(matched["Name"].astype(str).tolist())

    result = query.copy()
    _set_annotation_column(result, "metaboscape_formulas", formulas)
    _set_annotation_column(result, "metaboscape_names", names)
    return result
