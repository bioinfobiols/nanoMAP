"""Headless-safe plotting helpers for JOINT analysis outputs."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import sparse
from skimage.segmentation import find_boundaries

from joint.errors import InputFormatError
from joint.models import RegistrationResult


def _axes(ax: object) -> tuple[Any, Axes]:
    if ax is None:
        figure, axes = plt.subplots()
        return figure, axes
    if not isinstance(ax, Axes):
        raise InputFormatError("ax must be a matplotlib Axes or None")
    return ax.figure, ax


def _save(figure: Any, save: object) -> None:
    if save is None:
        return
    if not isinstance(save, (str, Path)):
        raise InputFormatError("save must be a path string, Path, or None")
    try:
        destination = Path(save).expanduser().resolve()
        if destination.exists() and destination.is_dir():
            raise InputFormatError("save must identify a file, not a directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, bbox_inches="tight")
    except InputFormatError:
        raise
    except (OSError, ValueError) as exc:
        raise InputFormatError(f"could not save plot to {save!s}: {exc}") from exc


def _labels(labels: object) -> np.ndarray:
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise InputFormatError("labels must be a two-dimensional numpy array")
    if labels.dtype.kind not in "iu":
        raise InputFormatError("labels must contain integer values")
    return labels


def _image(image: object, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise InputFormatError("image must be a two- or three-dimensional numpy array")
    if image.shape[:2] != shape:
        raise InputFormatError("image must have the same first two dimensions as labels")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise InputFormatError("image must have 1, 3, or 4 channels")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise InputFormatError("image must contain finite real numeric values")
    return image


def _spatial_adata(adata: object) -> tuple[anndata.AnnData, np.ndarray]:
    if not isinstance(adata, anndata.AnnData):
        raise InputFormatError("adata must be an AnnData object")
    if "spatial" not in adata.obsm:
        raise InputFormatError('adata.obsm must contain "spatial" coordinates')
    spatial = np.asarray(adata.obsm["spatial"])
    if spatial.dtype.kind not in "uif" or spatial.dtype.kind == "b":
        raise InputFormatError('adata.obsm["spatial"] must contain real numeric coordinates')
    if spatial.shape != (adata.n_obs, 2):
        raise InputFormatError('adata.obsm["spatial"] must have shape (n_obs, 2)')
    if not np.isfinite(spatial).all():
        raise InputFormatError('adata.obsm["spatial"] must contain finite coordinates')
    return adata, np.asarray(spatial, dtype=float)


def _layer(adata: anndata.AnnData, layer: object) -> str | None:
    if layer is None:
        return None
    if not isinstance(layer, str) or not layer.strip():
        raise InputFormatError("layer must be a non-empty string or None")
    if layer not in adata.layers:
        raise InputFormatError(f"adata.layers is missing requested layer {layer!r}")
    return layer


def _numeric_values(values: object, *, name: str, length: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (length,) or raw.dtype.kind not in "uif" or raw.dtype.kind == "b":
        raise InputFormatError(f"{name} must contain {length} real numeric values")
    numeric = np.asarray(raw, dtype=float)
    if not np.isfinite(numeric).all():
        raise InputFormatError(f"{name} must contain finite values")
    return numeric.copy()


def _feature_values(adata: anndata.AnnData, feature: object, layer: str | None) -> np.ndarray:
    if not isinstance(feature, str) or not feature.strip():
        raise InputFormatError("feature must be a non-empty string")
    if feature in adata.obs:
        series = adata.obs[feature]
        if pd.api.types.is_numeric_dtype(series):
            return _numeric_values(
                series.to_numpy(), name=f'adata.obs["{feature}"]', length=adata.n_obs
            )
        if series.isna().any():
            raise InputFormatError(f'adata.obs["{feature}"] must not contain missing values')
        return series.astype("category").cat.codes.to_numpy(dtype=float, copy=True)
    if not adata.var_names.is_unique:
        raise InputFormatError("AnnData feature identifiers must be unique")
    index = adata.var_names.get_indexer([feature])
    if index[0] < 0:
        raise InputFormatError(
            f"feature must be an observation column or known identifier: {feature!r}"
        )
    matrix = adata.layers[layer] if layer is not None else adata.X
    if not hasattr(matrix, "shape") or matrix.shape != adata.shape:
        raise InputFormatError("selected matrix must have shape (n_obs, n_vars)")
    column = matrix[:, index[0]]
    values = column.toarray().ravel() if sparse.issparse(column) else np.asarray(column).ravel()
    return _numeric_values(values, name=f"feature {feature!r}", length=adata.n_obs)


def _finish(figure: Any, save: object) -> None:
    _save(figure, save)


def plot_segmentation(
    labels: np.ndarray, *, save: str | Path | None = None, ax: Axes | None = None
):
    """Plot a two-dimensional integer segmentation label array."""
    checked_labels = _labels(labels)
    figure, axes = _axes(ax)
    axes.imshow(checked_labels, cmap="nipy_spectral", interpolation="nearest")
    axes.set_axis_off()
    _finish(figure, save)
    return figure, axes


def plot_cell_contours(
    labels: np.ndarray,
    *,
    image: np.ndarray | None = None,
    save: str | Path | None = None,
    ax: Axes | None = None,
):
    """Plot segmentation boundaries, optionally over a matching image."""
    checked_labels = _labels(labels)
    checked_image = None if image is None else _image(image, checked_labels.shape)
    figure, axes = _axes(ax)
    if checked_image is not None:
        axes.imshow(checked_image)
    axes.imshow(find_boundaries(checked_labels, mode="outer"), cmap="gray", alpha=0.8)
    axes.set_axis_off()
    _finish(figure, save)
    return figure, axes


def plot_spatial_feature(
    adata: anndata.AnnData,
    feature: str,
    *,
    layer: str | None = None,
    save: str | Path | None = None,
    ax: Axes | None = None,
):
    """Plot an observation annotation or molecular feature at spatial coordinates."""
    checked_adata, spatial = _spatial_adata(adata)
    checked_layer = _layer(checked_adata, layer)
    values = _feature_values(checked_adata, feature, checked_layer)
    figure, axes = _axes(ax)
    scatter = axes.scatter(spatial[:, 0], spatial[:, 1], c=values)
    figure.colorbar(scatter, ax=axes)
    axes.set_aspect("equal")
    axes.invert_yaxis()
    _finish(figure, save)
    return figure, axes


def _registration_mapping(result: object) -> tuple[pd.DataFrame, str]:
    if not isinstance(result, RegistrationResult):
        raise InputFormatError("result must be a RegistrationResult")
    if not isinstance(result.mapping, pd.DataFrame):
        raise InputFormatError("RegistrationResult.mapping must be a pandas DataFrame")
    required = ("centroid-0", "centroid-1")
    if any(column not in result.mapping for column in required):
        raise InputFormatError("RegistrationResult.mapping must contain centroid-0 and centroid-1")
    columns = {
        column: _numeric_values(
            result.mapping[column].to_numpy(),
            name=f'RegistrationResult.mapping["{column}"]',
            length=len(result.mapping),
        )
        for column in required
    }
    if not isinstance(result.transform, Mapping):
        raise InputFormatError("RegistrationResult.transform must be a mapping")
    orientation = result.transform.get("orientation")
    if not isinstance(orientation, str) or not orientation.strip():
        raise InputFormatError('RegistrationResult.transform must contain non-empty "orientation"')
    return pd.DataFrame(columns), orientation


def plot_registration(
    result: RegistrationResult,
    *,
    save: str | Path | None = None,
    ax: Axes | None = None,
):
    """Plot registered spatial centroids from a registration result."""
    mapping, orientation = _registration_mapping(result)
    figure, axes = _axes(ax)
    axes.scatter(mapping["centroid-1"], mapping["centroid-0"], s=12)
    axes.set_title(orientation)
    axes.set_aspect("equal")
    axes.invert_yaxis()
    _finish(figure, save)
    return figure, axes


def plot_cnmf_usage(
    adata: anndata.AnnData,
    usage: str,
    *,
    save: str | Path | None = None,
    ax: Axes | None = None,
):
    """Plot a numeric cNMF usage column at spatial coordinates."""
    checked_adata, spatial = _spatial_adata(adata)
    if not isinstance(usage, str) or not usage.strip():
        raise InputFormatError("usage must be a non-empty string")
    if usage not in checked_adata.obs:
        raise InputFormatError(f"adata.obs is missing requested usage {usage!r}")
    values = _numeric_values(
        checked_adata.obs[usage].to_numpy(),
        name=f'adata.obs["{usage}"]',
        length=checked_adata.n_obs,
    )
    figure, axes = _axes(ax)
    scatter = axes.scatter(spatial[:, 0], spatial[:, 1], c=values)
    figure.colorbar(scatter, ax=axes)
    axes.set_aspect("equal")
    axes.invert_yaxis()
    _finish(figure, save)
    return figure, axes


def _trend_table(trends: object) -> pd.DataFrame:
    required = ("feature", "position", "fitted", "lower", "upper")
    if not isinstance(trends, pd.DataFrame):
        raise InputFormatError("trends must be a pandas DataFrame")
    if trends.empty or any(column not in trends for column in required):
        raise InputFormatError(f"trends must contain non-empty columns: {list(required)}")
    feature = trends["feature"]
    if (
        feature.isna().any()
        or not feature.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    ):
        raise InputFormatError('trends["feature"] must contain non-empty strings')
    checked = trends.loc[:, required].copy()
    for column in required[1:]:
        checked[column] = _numeric_values(
            checked[column].to_numpy(), name=f'trends["{column}"]', length=len(checked)
        )
    if np.any(checked["lower"].to_numpy() > checked["upper"].to_numpy()):
        raise InputFormatError('trends["lower"] must not exceed trends["upper"]')
    return checked


def plot_trajectory(
    trends: pd.DataFrame,
    *,
    save: str | Path | None = None,
    ax: Axes | None = None,
):
    """Plot fitted trajectory trends and their confidence intervals by feature."""
    checked = _trend_table(trends)
    figure, axes = _axes(ax)
    for feature, table in checked.groupby("feature", sort=False):
        axes.plot(table["position"], table["fitted"], label=feature)
        axes.fill_between(table["position"], table["lower"], table["upper"], alpha=0.2)
    axes.legend()
    _finish(figure, save)
    return figure, axes
