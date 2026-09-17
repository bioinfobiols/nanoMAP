"""Native sparse preprocessing helpers for MSI AnnData datasets."""

from copy import deepcopy
from pathlib import Path
from typing import Literal

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix

from joint._peaks import _build_consensus_axis_with_assignments
from joint.config import PreprocessingConfig
from joint.errors import InputFormatError
from joint.io import read_h5ad, read_imzml


def _mz_values(adata: anndata.AnnData) -> np.ndarray:
    if "mz" not in adata.var:
        raise InputFormatError('adata.var must contain numeric column "mz"')
    try:
        mz = adata.var["mz"].astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise InputFormatError('adata.var must contain numeric column "mz"') from exc
    if not np.all(np.isfinite(mz)):
        raise InputFormatError('adata.var must contain finite numeric column "mz"')
    if not np.all(mz > 0):
        raise InputFormatError('adata.var must contain positive numeric column "mz"')
    return mz


def align_peaks(
    adata: anndata.AnnData, *, tolerance: float, unit: Literal["ppm", "da"] = "ppm"
) -> anndata.AnnData:
    """Merge features that fall within a common m/z tolerance window."""
    mz = _mz_values(adata)
    axis, assignments = _build_consensus_axis_with_assignments(
        [mz], tolerance=tolerance, unit=unit
    )
    projector = coo_matrix(
        (np.ones(len(mz)), (np.arange(len(mz)), assignments[0])), shape=(len(mz), len(axis))
    ).tocsr()
    matrix = adata.X @ projector
    result = anndata.AnnData(
        sparse.csr_matrix(matrix),
        obs=adata.obs.copy(),
        var={"mz": axis},
        obsm={key: value.copy() for key, value in adata.obsm.items()},
    )
    result.var_names = [f"m{index + 1}" for index in range(result.n_vars)]
    result.uns = deepcopy(adata.uns)
    return result


def _row_factors(matrix: sparse.spmatrix | np.ndarray, method: str) -> np.ndarray:
    if method not in {"rms", "tic"}:
        raise ValueError(f"Unsupported normalization method: {method}")
    if matrix.shape[1] == 0:
        return np.ones(matrix.shape[0], dtype=float)
    values = matrix.astype(float, copy=False) if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
    squared = values.multiply(values) if sparse.issparse(values) else np.square(values)
    if method == "rms":
        factors = np.sqrt(np.asarray(squared.mean(axis=1)).ravel())
    else:
        factors = np.asarray(np.abs(values).sum(axis=1)).ravel()
    factors[factors == 0] = 1.0
    return factors


def filter_features(adata: anndata.AnnData, *, min_occurrence: float = 0.05) -> anndata.AnnData:
    """Return a copy containing features present above the given pixel fraction."""
    if not 0 <= min_occurrence <= 1:
        raise ValueError("min_occurrence must be between 0 and 1")
    counts = _occurrence_counts(adata.X)
    if adata.n_obs == 0:
        keep = np.zeros(adata.n_vars, dtype=bool)
    else:
        keep = np.asarray(counts).ravel() / adata.n_obs > min_occurrence
    result = adata[:, keep].copy()
    result.uns.setdefault("joint", {})["min_occurrence"] = min_occurrence
    return result


def normalize(
    adata: anndata.AnnData,
    *,
    method: Literal["rms", "tic", "none"] = "rms",
    layer: str = "normalized",
) -> anndata.AnnData:
    """Store row-normalized intensities in a layer without changing ``X``."""
    if not isinstance(layer, str) or not layer:
        raise ValueError("layer must be a non-empty string")
    result = adata.copy()
    if method == "none":
        result.layers[layer] = result.X.copy()
    else:
        factors = _row_factors(result.X, method)
        inverse = sparse.diags(1.0 / factors)
        normalized = inverse @ result.X if sparse.issparse(result.X) else result.X / factors[:, None]
        result.layers[layer] = sparse.csr_matrix(normalized)
    result.uns.setdefault("joint", {})["normalization"] = method
    return result


def _occurrence_counts(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """Count non-zero values per feature, ignoring stored sparse zeroes."""
    if sparse.issparse(matrix):
        nonzero_matrix = matrix.copy()
        nonzero_matrix.eliminate_zeros()
        return np.asarray(nonzero_matrix.getnnz(axis=0)).ravel()
    return np.count_nonzero(matrix, axis=0)


def _validate_matrix_frequency(matrix_frequency: float) -> float:
    try:
        frequency = float(matrix_frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix_frequency must be a finite value between 0 and 1") from exc
    if not np.isfinite(frequency) or not 0 <= frequency <= 1:
        raise ValueError("matrix_frequency must be a finite value between 0 and 1")
    return frequency


def _validate_tolerance(tolerance: float, unit: str) -> float:
    if unit not in {"ppm", "da"}:
        raise ValueError("unit must be either 'ppm' or 'da'")
    try:
        value = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be a finite positive value") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("tolerance must be a finite positive value")
    return value


def _reference_values(reference_mz: np.ndarray | None) -> np.ndarray:
    if reference_mz is None:
        raise ValueError("reference_mz is required for reference removal")
    try:
        reference = np.asarray(reference_mz, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference_mz must contain finite positive m/z values") from exc
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("reference_mz must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(reference)) or not np.all(reference > 0):
        raise ValueError("reference_mz must contain finite positive m/z values")
    return reference


def _ppm_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(query[:, None] - reference[None, :]) / query[:, None] * 1_000_000.0


def _edge_mask(spatial: np.ndarray, n_obs: int) -> np.ndarray:
    if spatial.shape != (n_obs, 2):
        raise ValueError("obsm['spatial'] must have shape (n_obs, 2)")
    if n_obs == 0:
        raise ValueError("obsm['spatial'] must contain at least one observation")
    if not np.issubdtype(spatial.dtype, np.number) or not np.all(np.isfinite(spatial)):
        raise ValueError("obsm['spatial'] must contain finite numeric coordinates")
    x = spatial[:, 0]
    y = spatial[:, 1]
    return (x == x.min()) | (x == x.max()) | (y == y.min()) | (y == y.max())


def remove_matrix_peaks(
    adata: anndata.AnnData,
    *,
    method: Literal["reference", "blank", "denovo"],
    reference_mz: np.ndarray | None = None,
    blank_datasets: list[str] | None = None,
    matrix_frequency: float = 0.8,
    tolerance: float = 10.0,
    unit: Literal["ppm", "da"] = "ppm",
) -> tuple[anndata.AnnData, pd.DataFrame]:
    """Remove matrix-associated features and return a stable removal report.

    Features are removed only when their occurrence fraction is strictly greater
    than ``matrix_frequency``. This makes a threshold of 1.0 remove no features.
    """
    mz = _mz_values(adata)
    if method not in {"reference", "blank", "denovo"}:
        raise ValueError(f"Unsupported matrix removal method: {method}")
    frequency = _validate_matrix_frequency(matrix_frequency)
    tolerance_value = _validate_tolerance(tolerance, unit)

    if method == "reference":
        reference = _reference_values(reference_mz)
        distances = _ppm_distance(mz, reference) if unit == "ppm" else np.abs(
            mz[:, None] - reference[None, :]
        )
        remove = distances.min(axis=1) <= tolerance_value
        reason = "reference_match"
    elif method == "blank":
        if "dataset" not in adata.obs:
            raise ValueError("adata.obs['dataset'] is required for blank removal")
        if not blank_datasets:
            raise ValueError("blank_datasets must select at least one blank observation")
        blank_mask = adata.obs["dataset"].isin(blank_datasets).to_numpy()
        if not np.any(blank_mask):
            raise ValueError("blank_datasets must select at least one blank observation")
        subset = adata[blank_mask]
        remove = _occurrence_counts(subset.X) / subset.n_obs > frequency
        reason = "blank_frequency"
    else:
        if "spatial" not in adata.obsm:
            raise ValueError("obsm['spatial'] is required for denovo removal")
        edge_mask = _edge_mask(np.asarray(adata.obsm["spatial"]), adata.n_obs)
        if not np.any(edge_mask):
            raise ValueError("obsm['spatial'] must select at least one edge observation")
        subset = adata[edge_mask]
        remove = _occurrence_counts(subset.X) / subset.n_obs > frequency
        reason = "edge_frequency"

    report = pd.DataFrame(
        {
            "feature_id": adata.var_names[remove].to_numpy(),
            "mz": mz[remove],
            "reason": reason,
        }
    )
    return adata[:, ~remove].copy(), report.reset_index(drop=True)


def _load_reference_mz(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    try:
        table = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise InputFormatError(f"Could not read matrix reference file: {path}") from exc
    if "mz" not in table:
        raise InputFormatError(f"Matrix reference file must contain an mz column: {path}")
    try:
        reference = table["mz"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputFormatError("Matrix reference file must contain numeric m/z values") from exc
    if reference.size == 0 or not np.all(np.isfinite(reference)) or not np.all(reference > 0):
        raise InputFormatError("Matrix reference file must contain finite positive m/z values")
    return reference


def preprocess_msi(
    path: str | Path,
    *,
    config: PreprocessingConfig,
    dataset_id: str | None = None,
    cardinal_output_dir: str | Path | None = None,
) -> anndata.AnnData:
    """Read, feature-filter, and normalize an imzML or h5ad MSI dataset."""
    source = Path(path)
    if config.backend == "cardinal":
        if cardinal_output_dir is None:
            raise ValueError("cardinal_output_dir is required for the Cardinal backend")
        from joint.backends.cardinal import run_cardinal

        adata = run_cardinal(
            source,
            cardinal_output_dir,
            tolerance=config.peak_tolerance,
            unit=config.tolerance_unit,
            snr=config.cardinal_snr,
            dataset_id=dataset_id,
        )
    elif source.suffix.lower() == ".h5ad":
        adata = read_h5ad(source)
    elif source.suffix.lower() == ".imzml":
        adata = read_imzml(
            source,
            dataset_id=dataset_id,
            tolerance=config.peak_tolerance,
            unit=config.tolerance_unit,
            profile_bin_size=config.profile_bin_size,
        )
    else:
        raise InputFormatError(f"Unsupported MSI input: {source.suffix}")
    mz = _mz_values(adata)
    adata = adata.copy()
    adata.var["mz"] = mz
    filtered = filter_features(adata, min_occurrence=config.min_occurrence)
    matrix_config = config.matrix_removal
    if matrix_config.enabled:
        filtered, matrix_report = remove_matrix_peaks(
            filtered,
            method=matrix_config.method,
            reference_mz=_load_reference_mz(matrix_config.reference_file),
            blank_datasets=matrix_config.blank_datasets,
            matrix_frequency=matrix_config.matrix_frequency,
            tolerance=matrix_config.tolerance,
            unit=matrix_config.unit,
        )
        filtered.uns.setdefault("joint", {})["matrix_removed_mz"] = matrix_report["mz"].tolist()
        filtered.uns["joint"]["matrix_removal_reasons"] = matrix_report["reason"].tolist()
    return normalize(filtered, method=config.normalization, layer=config.normalized_layer)
