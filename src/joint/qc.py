"""Quality-control validation, summaries, and diagnostics."""

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Number, Rational, Real
from typing import Any

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa as skbio_pcoa
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances, silhouette_samples

from joint.errors import InputFormatError
from joint.models import QuantificationResult, RegistrationResult


def _require_adata(adata: object) -> anndata.AnnData:
    if not isinstance(adata, anndata.AnnData):
        raise InputFormatError("adata must be an AnnData object")
    return adata


def _validate_numeric_matrix(matrix: object, *, name: str, shape: tuple[int, int]) -> None:
    if not hasattr(matrix, "shape") or matrix.shape != shape:
        raise InputFormatError(f"{name} must have shape (n_obs, n_vars)")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if values.dtype.kind not in "uif":
        raise InputFormatError(f"{name} must contain real numeric values")
    if not np.isfinite(values).all():
        raise InputFormatError(f"{name} must contain finite values")


def _validate_adata_matrices(adata: object) -> anndata.AnnData:
    checked = _require_adata(adata)
    shape = (checked.n_obs, checked.n_vars)
    _validate_numeric_matrix(checked.X, name="adata.X", shape=shape)
    for name, matrix in checked.layers.items():
        _validate_numeric_matrix(matrix, name=f'adata.layers["{name}"]', shape=shape)
    return checked


def validate_anndata(adata: anndata.AnnData, *, require_spatial: bool = False) -> None:
    """Validate the AnnData fields consumed by JOINT analysis functions."""
    if not isinstance(require_spatial, (bool, np.bool_)):
        raise InputFormatError("require_spatial must be boolean")
    checked = _validate_adata_matrices(adata)

    if "mz" not in checked.var:
        raise InputFormatError('adata.var must contain numeric column "mz"')
    mz_values = checked.var["mz"].astype(object).tolist()
    if not all(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        for value in mz_values
    ):
        raise InputFormatError('adata.var["mz"] must be numeric')
    try:
        mz = np.asarray(mz_values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputFormatError('adata.var["mz"] must be numeric') from exc
    if not np.isfinite(mz).all():
        raise InputFormatError('adata.var["mz"] must contain finite values')

    if require_spatial:
        if "spatial" not in checked.obsm:
            raise InputFormatError('adata.obsm must contain "spatial"')
        raw_spatial = np.asarray(checked.obsm["spatial"])
        if raw_spatial.dtype.kind == "b" or (
            raw_spatial.dtype.kind not in "uif"
            and not all(
                isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                for value in raw_spatial.flat
            )
        ):
            raise InputFormatError('adata.obsm["spatial"] must contain numeric coordinates')
        try:
            spatial = np.asarray(raw_spatial, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InputFormatError('adata.obsm["spatial"] must contain numeric coordinates') from exc
        if spatial.shape != (checked.n_obs, 2):
            raise InputFormatError('adata.obsm["spatial"] must have shape (n_obs, 2)')
        if not np.isfinite(spatial).all():
            raise InputFormatError('adata.obsm["spatial"] must contain finite coordinates')


def _matrix_for_layer(adata: anndata.AnnData, layer: str | None) -> Any:
    checked = _validate_adata_matrices(adata)
    if layer is None:
        return checked.X
    if not isinstance(layer, str) or not layer:
        raise InputFormatError("layer must be a non-empty string or None")
    if layer not in checked.layers:
        raise InputFormatError(f"adata.layers is missing requested layer {layer!r}")
    matrix = checked.layers[layer]
    _validate_numeric_matrix(
        matrix, name=f'adata.layers["{layer}"]', shape=(checked.n_obs, checked.n_vars)
    )
    return matrix


def spectral_qc(adata: anndata.AnnData, *, layer: str | None = None) -> pd.DataFrame:
    """Return per-observation intensity and feature-detection summaries."""
    matrix = _float64_matrix(_matrix_for_layer(adata, layer))
    total = np.asarray(matrix.sum(axis=1)).ravel()
    if sparse.issparse(matrix):
        detected = np.asarray((matrix != 0).sum(axis=1)).ravel()
    else:
        detected = np.count_nonzero(matrix, axis=1)
    return pd.DataFrame(
        {
            "total_intensity": total,
            "detected_features": detected,
        },
        index=adata.obs_names.copy(),
    )


def _validate_report_payload(
    payload: object, *, name: str, metric_prefix: str = ""
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise InputFormatError(f"{name} must be a mapping")
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise InputFormatError(f"{name} metric names must be non-empty strings")
        _validate_finite_report_value(value, metric=f"{metric_prefix}{key}")
    return payload


def _validate_finite_report_value(value: object, *, metric: str) -> None:
    if isinstance(value, Number) and not isinstance(value, (bool, np.bool_)):
        if not isinstance(value, Real):
            raise InputFormatError(f"report metric {metric!r} must be a real number")
        if isinstance(value, Rational):
            finite = True
        else:
            try:
                finite = isfinite(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InputFormatError(
                    f"report metric {metric!r} must be a finite real number"
                ) from exc
        if not finite:
            raise InputFormatError(f"report metric {metric!r} must be finite")
        return
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            _validate_finite_report_value(nested_value, metric=f"{metric}.{nested_key}")
        return
    if isinstance(value, np.ndarray):
        nested_values = [value.item()] if value.ndim == 0 else value.flat
        for index, nested_value in enumerate(nested_values):
            _validate_finite_report_value(nested_value, metric=f"{metric}[{index}]")
        return
    if isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            _validate_finite_report_value(nested_value, metric=f"{metric}[{index}]")


def registration_report(result: RegistrationResult) -> pd.DataFrame:
    """Convert a registration result report and transform into a tidy table."""
    if not isinstance(result, RegistrationResult):
        raise InputFormatError("result must be a RegistrationResult")
    report = _validate_report_payload(result.report, name="RegistrationResult.report")
    transform = _validate_report_payload(
        result.transform,
        name="RegistrationResult.transform",
        metric_prefix="transform_",
    )
    rows = [(metric, value) for metric, value in report.items()]
    rows.extend((f"transform_{metric}", value) for metric, value in transform.items())
    metrics = [metric for metric, _ in rows]
    if len(metrics) != len(set(metrics)):
        raise InputFormatError("registration report and transform metric names must not collide")
    return pd.DataFrame(rows, columns=["metric", "value"])


def _nonnegative_count(payload: Mapping[str, Any], name: str) -> int:
    if name not in payload:
        raise InputFormatError(f"report must contain {name}")
    value = payload[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise InputFormatError(f"{name} must be a non-negative integer")
    count = int(value)
    if count < 0:
        raise InputFormatError(f"{name} must be a non-negative integer")
    return count


def coverage_report(result: QuantificationResult) -> pd.DataFrame:
    """Convert a quantification report into a tidy table with derived coverage."""
    if not isinstance(result, QuantificationResult):
        raise InputFormatError("result must be a QuantificationResult")
    report = dict(_validate_report_payload(result.report, name="QuantificationResult.report"))
    included = _nonnegative_count(report, "included_cells")
    quantified = _nonnegative_count(report, "quantified_cells")
    if quantified > included:
        raise InputFormatError("quantified_cells must not exceed included_cells")
    report["coverage_fraction"] = quantified / included if included else 0.0
    return pd.DataFrame(report.items(), columns=["metric", "value"])


def _validate_contamination(contamination: object) -> float:
    if isinstance(contamination, (bool, np.bool_)) or not isinstance(contamination, Real):
        raise InputFormatError("contamination must be a finite number in (0, 0.5]")
    value = float(contamination)
    if not np.isfinite(value) or not 0 < value <= 0.5:
        raise InputFormatError("contamination must be a finite number in (0, 0.5]")
    return value


def _validate_random_seed(random_seed: object) -> int:
    if isinstance(random_seed, (bool, np.bool_)) or not isinstance(random_seed, Integral):
        raise InputFormatError("random_seed must be an integer in [0, 2**32 - 1]")
    value = int(random_seed)
    if not 0 <= value <= 2**32 - 1:
        raise InputFormatError("random_seed must be an integer in [0, 2**32 - 1]")
    return value


def detect_outliers(
    adata: anndata.AnnData,
    *,
    contamination: float = 0.05,
    random_seed: int = 0,
    layer: str | None = None,
) -> pd.Series:
    """Flag observation-level spectral outliers deterministically."""
    checked_contamination = _validate_contamination(contamination)
    checked_seed = _validate_random_seed(random_seed)
    report = spectral_qc(adata, layer=layer)
    if len(report) < 2:
        raise InputFormatError("outlier detection requires at least 2 observations")
    try:
        labels = IsolationForest(
            contamination=checked_contamination,
            random_state=checked_seed,
        ).fit_predict(report.to_numpy(copy=True))
    except (TypeError, ValueError) as exc:
        raise InputFormatError(f"outlier detection input is invalid: {exc}") from exc
    return pd.Series(labels == -1, index=adata.obs_names.copy(), name="outlier")


def _float64_matrix(matrix: Any) -> Any:
    if sparse.issparse(matrix):
        return matrix.astype(np.float64, copy=False)
    return np.asarray(matrix, dtype=np.float64)


def _dense_matrix(adata: anndata.AnnData, layer: str | None = None) -> np.ndarray:
    matrix = _float64_matrix(_matrix_for_layer(adata, layer))
    return matrix.toarray() if sparse.issparse(matrix) else matrix.copy()


def _validate_metric(metric: object) -> str:
    if not isinstance(metric, str) or not metric.strip():
        raise InputFormatError("metric must be a non-empty string")
    return metric


def _braycurtis_distances(matrix: np.ndarray) -> np.ndarray:
    if np.any(matrix < 0):
        raise InputFormatError("Bray-Curtis distance requires non-negative values")
    distances = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    for row in range(matrix.shape[0] - 1):
        differences = np.abs(matrix[row + 1 :] - matrix[row]).sum(axis=1)
        totals = np.abs(matrix[row + 1 :] + matrix[row]).sum(axis=1)
        row_distances = np.zeros_like(differences, dtype=float)
        np.divide(differences, totals, out=row_distances, where=totals != 0)
        distances[row, row + 1 :] = row_distances
        distances[row + 1 :, row] = row_distances
    return distances


def _raw_pcoa_eigenvalues(distances: np.ndarray) -> np.ndarray:
    observations = distances.shape[0]
    centering = np.eye(observations) - np.full((observations, observations), 1 / observations)
    centered = -0.5 * centering @ np.square(distances) @ centering
    return np.linalg.eigvalsh(centered)[::-1]


def _set_pcoa_attrs(
    coordinates: pd.DataFrame,
    *,
    eigenvalues: np.ndarray,
    proportions: np.ndarray,
) -> None:
    axes = [f"PCoA{index + 1}" for index in range(len(eigenvalues))]
    coordinates.attrs["eigenvalues"] = {
        axis: float(value) for axis, value in zip(axes, eigenvalues, strict=True)
    }
    coordinates.attrs["proportion_explained"] = {
        axis: float(value) for axis, value in zip(axes, proportions, strict=True)
    }


def pcoa(
    adata: anndata.AnnData,
    *,
    dimensions: int = 2,
    metric: str = "braycurtis",
    layer: str | None = None,
) -> pd.DataFrame:
    """Return principal-coordinate axes for observation spectra.

    The returned DataFrame attrs contain full-spectrum ``eigenvalues`` and
    ``proportion_explained`` dictionaries keyed by PCoA axis, even when fewer coordinate
    columns are requested. Material negative eigenvalues remain visible and retain
    scikit-bio's diagnostic warning.
    """
    matrix = _dense_matrix(adata, layer)
    if matrix.shape[0] < 2:
        raise InputFormatError("PCoA requires at least 2 observations")
    if isinstance(dimensions, (bool, np.bool_)) or not isinstance(dimensions, Integral):
        raise InputFormatError("dimensions must be a positive integer no greater than n_obs")
    checked_dimensions = int(dimensions)
    if not 1 <= checked_dimensions <= matrix.shape[0]:
        raise InputFormatError("dimensions must be a positive integer no greater than n_obs")
    checked_metric = _validate_metric(metric)
    if checked_metric == "precomputed":
        raise InputFormatError("metric 'precomputed' is invalid for spectral PCoA input")
    try:
        if checked_metric == "braycurtis":
            distances = _braycurtis_distances(matrix)
        else:
            distances = pairwise_distances(matrix, metric=checked_metric)
        if not np.isfinite(distances).all():
            raise InputFormatError(
                f"metric {checked_metric!r} produced non-finite PCoA distances"
            )
        eigenvalues = _raw_pcoa_eigenvalues(distances)
        if not np.any(distances):
            coordinates = pd.DataFrame(
                np.zeros((matrix.shape[0], checked_dimensions)),
                index=adata.obs_names.copy(),
                columns=[f"PCoA{index + 1}" for index in range(checked_dimensions)],
            )
            _set_pcoa_attrs(
                coordinates,
                eigenvalues=eigenvalues,
                proportions=np.zeros(matrix.shape[0]),
            )
            return coordinates
        distance_matrix = DistanceMatrix(distances)
        ordination = skbio_pcoa(
            distance_matrix,
            number_of_dimensions=matrix.shape[0],
        )
    except InputFormatError:
        raise
    except (TypeError, ValueError) as exc:
        raise InputFormatError(f"metric {checked_metric!r} is invalid for PCoA: {exc}") from exc
    values = ordination.samples.iloc[:, :checked_dimensions].copy()
    if values.shape[1] != checked_dimensions or not np.isfinite(values.to_numpy()).all():
        raise InputFormatError("PCoA did not produce the requested finite coordinates")
    values.columns = [f"PCoA{index + 1}" for index in range(checked_dimensions)]
    values.index = adata.obs_names.copy()
    _set_pcoa_attrs(
        values,
        eigenvalues=eigenvalues,
        proportions=ordination.proportion_explained.to_numpy(dtype=float, copy=True),
    )
    return values


def silhouette_summary(
    adata: anndata.AnnData,
    *,
    labels: str,
    metric: str = "euclidean",
    layer: str | None = None,
) -> pd.DataFrame:
    """Return cluster-level mean silhouette scores in first-observed group order."""
    matrix = _dense_matrix(adata, layer)
    if matrix.shape[0] < 3:
        raise InputFormatError("silhouette analysis requires at least 3 observations")
    if not isinstance(labels, str) or not labels or labels not in adata.obs:
        raise InputFormatError("labels must name an existing adata.obs column")
    groups = adata.obs[labels]
    if groups.isna().any():
        raise InputFormatError("labels must not contain missing values")
    group_strings = groups.astype(str)
    cluster_count = group_strings.nunique(dropna=False)
    if cluster_count < 2:
        raise InputFormatError("silhouette analysis requires at least 2 clusters")
    if cluster_count >= matrix.shape[0]:
        raise InputFormatError("silhouette analysis requires fewer clusters than observations")
    checked_metric = _validate_metric(metric)
    if checked_metric == "precomputed":
        raise InputFormatError(
            "metric 'precomputed' is invalid for silhouette feature-matrix input"
        )
    try:
        distances = pairwise_distances(matrix, metric=checked_metric)
        if not np.isfinite(distances).all():
            raise InputFormatError(
                f"metric {checked_metric!r} produced non-finite silhouette distances"
            )
        values = silhouette_samples(
            distances,
            group_strings.to_numpy(),
            metric="precomputed",
        )
    except InputFormatError:
        raise
    except (TypeError, ValueError) as exc:
        raise InputFormatError(f"metric {checked_metric!r} is invalid for silhouette analysis: {exc}") from exc
    return (
        pd.DataFrame({"cluster": group_strings.to_numpy(copy=True), "silhouette": values})
        .groupby("cluster", as_index=False, sort=False)
        .agg(mean_silhouette=("silhouette", "mean"), cells=("silhouette", "size"))
    )
