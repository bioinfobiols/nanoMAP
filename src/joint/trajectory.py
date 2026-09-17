"""Copy-safe spatial trajectory projection and optional feature-trend fitting."""

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

from joint.errors import InputFormatError, OptionalDependencyError

_DEPENDENCY_MESSAGE = "Trajectory trends require `python -m pip install 'joint-msi[trajectory]'`"
_TREND_COLUMNS = ["feature", "position", "fitted", "lower", "upper"]
_LONGDOUBLE_EPSILON = np.finfo(np.longdouble).eps
_PROJECTION_ERROR_ULPS = 16


def _require_anndata(adata: object, *, minimum_observations: int) -> anndata.AnnData:
    if not isinstance(adata, anndata.AnnData):
        raise InputFormatError("adata must be an AnnData object")
    if adata.n_obs < minimum_observations:
        raise InputFormatError(
            f"trajectory analysis requires at least {minimum_observations} observation"
            f"{'s' if minimum_observations != 1 else ''}"
        )
    if not adata.obs_names.is_unique:
        raise InputFormatError("AnnData observation identifiers must be unique")
    return adata


def _real_finite_array(value: object, *, name: str) -> np.ndarray:
    try:
        values = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise InputFormatError(f"{name} must contain real numeric values") from exc
    if values.dtype.kind not in "uif":
        raise InputFormatError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        checked = np.asarray(values, dtype=np.float64)
    if not np.isfinite(checked).all():
        raise InputFormatError(f"{name} must contain finite values")
    return checked


def _spatial_coordinates(adata: anndata.AnnData) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise InputFormatError('adata.obsm must contain spatial coordinates at key "spatial"')
    spatial = _real_finite_array(adata.obsm["spatial"], name='adata.obsm["spatial"]')
    if spatial.shape != (adata.n_obs, 2):
        raise InputFormatError('adata.obsm["spatial"] must have shape (n_obs, 2)')
    return spatial


def _path_geometry(
    path: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.longdouble]:
    try:
        candidate = np.asarray(path)
    except (TypeError, ValueError) as exc:
        raise InputFormatError("path must have shape (n_points, 2)") from exc
    if candidate.ndim != 2 or candidate.shape[1:] != (2,):
        raise InputFormatError("path must have shape (n_points, 2)")
    if len(candidate) < 2:
        raise InputFormatError("path must contain at least two points")
    points = _real_finite_array(candidate, name="path")
    extended_points = np.asarray(points, dtype=np.longdouble)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        segments = extended_points[1:] - extended_points[:-1]
        lengths = np.hypot(segments[:, 0], segments[:, 1])
    if not np.isfinite(segments).all() or not np.isfinite(lengths).all():
        raise InputFormatError("path segment lengths must be finite")
    if np.any(lengths == np.longdouble(0.0)):
        raise InputFormatError("path contains a zero-length segment")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cumulative = np.concatenate(
            (np.array([0.0], dtype=np.longdouble), np.cumsum(lengths, dtype=np.longdouble))
        )
    total_length = np.longdouble(cumulative[-1])
    if not np.isfinite(total_length) or total_length <= np.longdouble(0.0):
        raise InputFormatError("path total length must be finite and positive")
    return points, segments, lengths, total_length


def _clamped_fraction(
    numerator: np.longdouble,
    denominator: np.longdouble,
    offset_to_start: np.ndarray,
    offset_to_end: np.ndarray,
) -> np.longdouble:
    if np.all(offset_to_start == np.longdouble(0.0)):
        return np.longdouble(0.0)
    if np.all(offset_to_end == np.longdouble(0.0)):
        return np.longdouble(1.0)
    fraction = numerator / denominator
    return np.clip(fraction, np.longdouble(0.0), np.longdouble(1.0))


def _endpoint_distance(offset: np.ndarray) -> tuple[np.longdouble, np.longdouble]:
    distance = np.longdouble(np.hypot(offset[0], offset[1]))
    component_scale = abs(offset[0]) + abs(offset[1]) + distance
    error = np.longdouble(_PROJECTION_ERROR_ULPS) * _LONGDOUBLE_EPSILON * component_scale
    return distance, np.longdouble(error)


def _interior_distance(
    vector: np.ndarray,
    offset: np.ndarray,
    length: np.longdouble,
) -> tuple[np.longdouble, np.longdouble]:
    positive_product = vector[0] * offset[1]
    negative_product = vector[1] * offset[0]
    distance = np.longdouble(abs(positive_product - negative_product) / length)
    cancellation_scale = (abs(positive_product) + abs(negative_product)) / length
    error = (
        np.longdouble(_PROJECTION_ERROR_ULPS)
        * _LONGDOUBLE_EPSILON
        * (cancellation_scale + distance)
    )
    return distance, np.longdouble(error)


def _metadata_number(value: np.longdouble) -> float | np.longdouble:
    if abs(value) <= np.longdouble(np.finfo(np.float64).max):
        return float(value)
    return np.longdouble(value)


def _project_to_path(
    spatial: np.ndarray,
    points: np.ndarray,
    segments: np.ndarray,
    lengths: np.ndarray,
    total_length: np.longdouble,
) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.concatenate(
        (np.array([0.0], dtype=np.longdouble), np.cumsum(lengths, dtype=np.longdouble))
    )
    positions = np.empty(len(spatial), dtype=np.float64)
    distances = np.empty(len(spatial), dtype=np.longdouble)
    extended_spatial = np.asarray(spatial, dtype=np.longdouble)
    extended_points = np.asarray(points, dtype=np.longdouble)
    for point_index, point in enumerate(extended_spatial):
        best_distance: np.longdouble | None = None
        best_error = np.longdouble(0.0)
        best_position = np.longdouble(0.0)
        for segment_index, (start, vector, length) in enumerate(
            zip(extended_points[:-1], segments, lengths, strict=True)
        ):
            end = extended_points[segment_index + 1]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                offset = point - start
                offset_to_end = point - end
                denominator = np.dot(vector, vector)
                fraction = _clamped_fraction(
                    np.dot(offset, vector),
                    denominator,
                    offset,
                    offset_to_end,
                )
                if fraction == np.longdouble(0.0):
                    distance, error = _endpoint_distance(offset)
                elif fraction == np.longdouble(1.0):
                    distance, error = _endpoint_distance(offset_to_end)
                else:
                    distance, error = _interior_distance(vector, offset, length)
            if best_distance is None:
                best_distance = distance
                best_error = error
                best_position = np.longdouble(
                    (cumulative[segment_index] + fraction * length) / total_length
                )
                continue
            if distance + error < best_distance - best_error:
                best_distance = distance
                best_error = error
                best_position = np.longdouble(
                    (cumulative[segment_index] + fraction * length) / total_length
                )
        if (
            best_distance is None
            or not np.isfinite(best_distance)
            or not np.isfinite(best_position)
        ):
            raise InputFormatError(
                "trajectory projection must produce finite distances and positions"
            )
        positions[point_index] = float(best_position)
        distances[point_index] = best_distance
    if len(distances) == 0 or np.max(distances) <= np.longdouble(np.finfo(np.float64).max):
        return positions, np.asarray(distances, dtype=np.float64)
    return positions, distances


def fit_spatial_trajectory(adata: anndata.AnnData, path: np.ndarray) -> anndata.AnnData:
    """Project observations onto a finite polyline and return an annotated copy."""
    checked = _require_anndata(adata, minimum_observations=0)
    spatial = _spatial_coordinates(checked)
    points, segments, lengths, total_length = _path_geometry(path)
    existing_joint = checked.uns.get("joint", {})
    if not isinstance(existing_joint, Mapping):
        raise InputFormatError("AnnData uns['joint'] must be a mapping")
    positions, distances = _project_to_path(
        spatial,
        points,
        segments,
        lengths,
        total_length,
    )

    result = checked.copy()
    result.obs["trajectory_position"] = positions
    result.obs["trajectory_distance"] = distances
    joint_metadata = dict(result.uns.get("joint", {}))
    joint_metadata["trajectory_path"] = points.tolist()
    joint_metadata["trajectory"] = {
        "spatial_key": "spatial",
        "position_key": "trajectory_position",
        "distance_key": "trajectory_distance",
        "segment_lengths": [_metadata_number(length) for length in lengths],
        "total_length": _metadata_number(total_length),
        "projection_tie_breaker": "first segment in path order",
    }
    result.uns["joint"] = joint_metadata
    return result


def _trend_positions(adata: anndata.AnnData) -> np.ndarray:
    if "trajectory_position" not in adata.obs:
        raise InputFormatError('adata.obs must contain "trajectory_position"')
    positions = _real_finite_array(
        adata.obs["trajectory_position"].to_numpy(),
        name='adata.obs["trajectory_position"]',
    )
    if positions.shape != (adata.n_obs,):
        raise InputFormatError('adata.obs["trajectory_position"] must have shape (n_obs,)')
    if np.any((positions < 0.0) | (positions > 1.0)):
        raise InputFormatError('adata.obs["trajectory_position"] must contain values in [0, 1]')
    if np.unique(positions).size < 2:
        raise InputFormatError("trajectory trend fitting requires at least two distinct positions")
    return positions


def _trend_features(adata: anndata.AnnData, features: object) -> tuple[list[str], np.ndarray]:
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise InputFormatError("features must be a sequence of feature identifiers")
    checked = list(features)
    if not checked:
        raise InputFormatError("features must contain at least one feature identifier")
    if any(not isinstance(feature, str) or not feature.strip() for feature in checked):
        raise InputFormatError("features must contain non-empty string identifiers")
    if len(set(checked)) != len(checked):
        raise InputFormatError("features must contain unique identifiers")
    if not adata.var_names.is_unique:
        raise InputFormatError("AnnData feature identifiers must be unique")
    indices = adata.var_names.get_indexer(checked)
    if np.any(indices < 0):
        unknown = [feature for feature, index in zip(checked, indices, strict=True) if index < 0]
        raise InputFormatError(f"features must contain only known identifiers: {unknown}")
    return checked, indices


def _trend_points(points: object) -> int:
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, Integral):
        raise InputFormatError("points must be an integer greater than or equal to 2")
    checked = int(points)
    if checked < 2:
        raise InputFormatError("points must be an integer greater than or equal to 2")
    return checked


def _trend_source(adata: anndata.AnnData, layer: object) -> Any:
    if layer is None:
        name = "adata.X"
        matrix = adata.X
    else:
        if not isinstance(layer, str) or not layer.strip():
            raise InputFormatError("layer must be a non-empty string or None")
        if layer not in adata.layers:
            raise InputFormatError(f"adata.layers is missing requested layer {layer!r}")
        name = f'adata.layers["{layer}"]'
        matrix = adata.layers[layer]
    if not hasattr(matrix, "shape") or matrix.shape != adata.shape:
        raise InputFormatError(f"{name} must have shape (n_obs, n_vars)")
    try:
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    except (TypeError, ValueError) as exc:
        raise InputFormatError(f"{name} must contain real numeric values") from exc
    if values.dtype.kind not in "uif":
        raise InputFormatError(f"{name} must contain real numeric values")
    if not np.isfinite(values).all():
        raise InputFormatError(f"{name} must contain finite values")
    return matrix


def _pygam_entrypoints() -> tuple[Any, Any]:
    try:
        import pygam
    except ModuleNotFoundError as exc:
        if exc.name != "pygam":
            raise
        raise OptionalDependencyError(_DEPENDENCY_MESSAGE) from exc
    linear_gam = getattr(pygam, "LinearGAM", None)
    spline = getattr(pygam, "s", None)
    if not callable(linear_gam) or not callable(spline):
        cause = TypeError("pygam must provide callable LinearGAM and s entrypoints")
        raise OptionalDependencyError(_DEPENDENCY_MESSAGE) from cause
    return linear_gam, spline


def _validated_backend_output(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        candidate = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise InputFormatError(f"pyGAM {name} must contain real numeric values") from exc
    if candidate.shape != shape:
        raise InputFormatError(f"pyGAM {name} must have shape {shape}")
    if candidate.dtype.kind not in "uif":
        raise InputFormatError(f"pyGAM {name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        checked = np.asarray(candidate, dtype=np.float64)
    if not np.isfinite(checked).all():
        raise InputFormatError(f"pyGAM {name} must contain finite values")
    return checked


def calculate_feature_trends(
    adata: anndata.AnnData,
    features: list[str],
    *,
    points: int = 100,
    layer: str | None = None,
) -> pd.DataFrame:
    """Fit optional pyGAM trends on projected positions without mutating AnnData."""
    checked = _require_anndata(adata, minimum_observations=2)
    if checked.n_vars == 0:
        raise InputFormatError("trajectory trend fitting requires at least 1 feature")
    positions = _trend_positions(checked)
    checked_features, feature_indices = _trend_features(checked, features)
    checked_points = _trend_points(points)
    matrix = _trend_source(checked, layer)
    linear_gam, spline = _pygam_entrypoints()

    x = positions[:, None].copy()
    grid = np.linspace(0.0, 1.0, checked_points, dtype=np.float64)[:, None]
    records: list[pd.DataFrame] = []
    for feature, feature_index in zip(checked_features, feature_indices, strict=True):
        column = matrix[:, int(feature_index)]
        y = column.toarray().ravel() if sparse.issparse(column) else np.asarray(column).ravel()
        unfitted_model = linear_gam(spline(0))
        fit = getattr(unfitted_model, "fit", None)
        if not callable(fit):
            raise InputFormatError("pyGAM model must provide a callable fit method")
        model = fit(x.copy(), y.copy())
        predict = getattr(model, "predict", None)
        prediction_intervals = getattr(model, "prediction_intervals", None)
        if not callable(predict) or not callable(prediction_intervals):
            raise InputFormatError(
                "pyGAM fit must return a model with predict and prediction_intervals methods"
            )
        prediction = _validated_backend_output(
            predict(grid.copy()),
            name="prediction",
            shape=(checked_points,),
        )
        intervals = _validated_backend_output(
            prediction_intervals(grid.copy(), width=0.95),
            name="intervals",
            shape=(checked_points, 2),
        )
        if np.any(intervals[:, 0] > intervals[:, 1]):
            raise InputFormatError("pyGAM interval lower bounds must not exceed upper bounds")
        records.append(
            pd.DataFrame(
                {
                    "feature": feature,
                    "position": grid.ravel(),
                    "fitted": prediction,
                    "lower": intervals[:, 0],
                    "upper": intervals[:, 1],
                },
                columns=_TREND_COLUMNS,
            )
        )
    return pd.concat(records, ignore_index=True)
