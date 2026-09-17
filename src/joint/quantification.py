"""Cell-to-laser overlap extraction for JOINT quantification."""

from collections import Counter
from copy import deepcopy
from decimal import Decimal
from numbers import Integral, Rational, Real
from typing import Literal

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

from joint.errors import QuantificationError
from joint.models import QuantificationResult

_REQUIRED_OBS_COLUMNS = ("seg_label", "area", "centroid-0", "centroid-1")
_OVERLAP_COLUMNS = (
    "cell_id",
    "cell_label",
    "laser_label",
    "laser_obs_id",
    "occupied_pixels",
    "laser_area",
    "occupied_ratio",
)
_CELL_COLUMNS = ("label", "area", "centroid-0", "centroid-1", "cell_id")
_SELECTOR_COLUMNS = ("cell_id", "laser_label", "occupied_ratio")
_QUANTIFICATION_METHODS = ("specificity_filtered", "legacy_proportional")


def _validate_labels(
    cell_labels: object, laser_labels: object
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated, read-only-by-convention segmentation label arrays."""
    cells = np.asarray(cell_labels)
    lasers = np.asarray(laser_labels)
    if (
        cells.ndim != 2
        or lasers.ndim != 2
        or cells.shape != lasers.shape
        or 0 in cells.shape
    ):
        raise QuantificationError(
            "Cell and laser segmentations must be non-empty two-dimensional arrays with equal shapes"
        )
    if cells.dtype.kind not in "iu" or lasers.dtype.kind not in "iu":
        raise QuantificationError("Segmentation labels must use non-boolean integer dtypes")
    if (cells < 0).any() or (lasers < 0).any():
        raise QuantificationError("Segmentation labels must be non-negative")
    return cells, lasers


def _validate_boundary_margin(boundary_margin: object) -> float:
    if isinstance(boundary_margin, (bool, np.bool_)) or not isinstance(boundary_margin, Real):
        raise QuantificationError("boundary_margin must be a finite non-negative number")
    margin = float(boundary_margin)
    if not np.isfinite(margin) or margin < 0:
        raise QuantificationError("boundary_margin must be a finite non-negative number")
    return margin


def _validate_ratio(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise QuantificationError(f"{name} must be a finite ratio in [0, 1]")
    ratio = float(value)
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise QuantificationError(f"{name} must be a finite ratio in [0, 1]")
    return ratio


def _numeric_obs_column(
    adata: anndata.AnnData, column: str, *, non_negative: bool = False
) -> np.ndarray:
    values = adata.obs[column].astype(object).tolist()
    if not all(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) for value in values
    ):
        raise QuantificationError(f"Laser obs column {column!r} must be numeric")
    numbers = np.asarray(values, dtype=float)
    if not np.isfinite(numbers).all():
        raise QuantificationError(f"Laser obs column {column!r} must contain finite values")
    if non_negative and (numbers < 0).any():
        raise QuantificationError(f"Laser obs column {column!r} must be non-negative")
    return numbers


def _parse_seg_label(value: object) -> tuple[int, ...]:
    """Canonicalize one registered segmentation label into its raster constituents."""
    if value is None or value is pd.NA:
        raise QuantificationError("Laser obs column 'seg_label' must not contain missing values")

    if isinstance(value, str):
        parts = value.strip().split("_")
        if not value.strip() or any(not part.strip() for part in parts):
            raise QuantificationError("Laser obs column 'seg_label' must contain valid identifiers")
        raw_parts: tuple[object, ...] = tuple(part.strip() for part in parts)
    else:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise QuantificationError(
                "Laser obs column 'seg_label' must contain string or integral numeric identifiers"
            )
        if pd.isna(value):
            raise QuantificationError("Laser obs column 'seg_label' must not contain missing values")
        raw_parts = (value,)

    labels: list[int] = []
    for part in raw_parts:
        if isinstance(part, str):
            try:
                label = int(part)
            except ValueError as exc:
                raise QuantificationError(
                    "Laser obs column 'seg_label' must contain integral identifiers"
                ) from exc
        else:
            if isinstance(part, (bool, np.bool_)) or not isinstance(part, Real):
                raise QuantificationError(
                    "Laser obs column 'seg_label' must contain string or integral numeric identifiers"
                )
            numeric = float(part)
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise QuantificationError(
                    "Laser obs column 'seg_label' must contain finite integral identifiers"
                )
            label = int(part) if isinstance(part, Integral) else int(numeric)
        if label < 0:
            raise QuantificationError(
                "Laser obs column 'seg_label' must contain non-negative integral identifiers"
            )
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise QuantificationError("Laser obs column 'seg_label' must not repeat constituents")
    return tuple(labels)


def _validate_laser_adata(laser_adata: object) -> tuple[anndata.AnnData, np.ndarray, np.ndarray]:
    if not isinstance(laser_adata, anndata.AnnData):
        raise QuantificationError("laser_adata must be an AnnData object")
    if laser_adata.n_obs == 0:
        raise QuantificationError("laser_adata must contain at least one observation")
    if not laser_adata.obs_names.is_unique:
        raise QuantificationError("Laser AnnData must have unique observation identifiers")
    missing = sorted(set(_REQUIRED_OBS_COLUMNS) - set(laser_adata.obs.columns))
    if missing:
        raise QuantificationError(f"Laser AnnData is missing obs columns: {missing}")
    _numeric_obs_column(laser_adata, "area", non_negative=True)
    area = np.asarray(laser_adata.obs["area"].astype(object).tolist(), dtype=object)
    centroid_rows = _numeric_obs_column(laser_adata, "centroid-0")
    centroid_columns = _numeric_obs_column(laser_adata, "centroid-1")
    return laser_adata, area, np.column_stack((centroid_rows, centroid_columns))


def _laser_connections(laser_adata: anndata.AnnData) -> dict[int, str]:
    connections: dict[int, str] = {}
    for obs_id, value in laser_adata.obs["seg_label"].items():
        for label in _parse_seg_label(value):
            if label in connections:
                raise QuantificationError(
                    f"Laser label {label} maps to more than one observation"
                )
            connections[label] = str(obs_id)
    return connections


def _empty_overlap_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": pd.Series(dtype="string"),
            "cell_label": pd.Series(dtype="object"),
            "laser_label": pd.Series(dtype="object"),
            "laser_obs_id": pd.Series(dtype="string"),
            "occupied_pixels": pd.Series(dtype="int64"),
            "laser_area": pd.Series(dtype="int64"),
            "occupied_ratio": pd.Series(dtype="float64"),
        },
        columns=_OVERLAP_COLUMNS,
    )


def _group_pair_counts(
    cell_groups: np.ndarray, laser_groups: np.ndarray
) -> list[tuple[int, int, int]]:
    order = np.lexsort((laser_groups, cell_groups))
    if order.size == 0:
        return []
    ordered_cells = cell_groups[order]
    ordered_lasers = laser_groups[order]
    starts = np.empty(order.size, dtype=bool)
    starts[0] = True
    starts[1:] = (ordered_cells[1:] != ordered_cells[:-1]) | (
        ordered_lasers[1:] != ordered_lasers[:-1]
    )
    start_indices = np.flatnonzero(starts)
    counts = np.diff(np.append(start_indices, order.size))
    return [
        (int(ordered_cells[index]), int(ordered_lasers[index]), int(count))
        for index, count in zip(start_indices, counts, strict=True)
    ]


def _group_raster_pixels(
    cells: np.ndarray, lasers: np.ndarray
) -> tuple[pd.DataFrame, dict[int, int], list[tuple[int, int, int]]]:
    cell_labels, cell_inverse, cell_counts = np.unique(
        cells.reshape(-1), return_inverse=True, return_counts=True
    )
    positive_cell_indices = np.flatnonzero(cell_labels > 0)
    if positive_cell_indices.size == 0:
        cell_table = pd.DataFrame(
            {
                "label": pd.Series(dtype="object"),
                "area": pd.Series(dtype="float64"),
                "centroid-0": pd.Series(dtype="float64"),
                "centroid-1": pd.Series(dtype="float64"),
                "cell_id": pd.Series(dtype="string"),
            },
            columns=_CELL_COLUMNS,
        )
    else:
        group_count = len(cell_labels)
        row_sums = np.zeros(group_count, dtype=float)
        column_sums = np.zeros(group_count, dtype=float)
        column_coordinates = np.arange(cells.shape[1], dtype=float)
        inverse_image = cell_inverse.reshape(cells.shape)
        for row_coordinate, row_groups in enumerate(inverse_image):
            row_sums += np.bincount(row_groups, minlength=group_count) * row_coordinate
            column_sums += np.bincount(
                row_groups, weights=column_coordinates, minlength=group_count
            )
        records = [
            {
                "label": int(cell_labels[index]),
                "area": float(cell_counts[index]),
                "centroid-0": float(row_sums[index] / cell_counts[index]),
                "centroid-1": float(column_sums[index] / cell_counts[index]),
            }
            for index in positive_cell_indices
        ]
        cell_table = pd.DataFrame.from_records(records, columns=_CELL_COLUMNS[:-1])
        cell_table["label"] = cell_table["label"].map(int).astype(object)
        cell_table["cell_id"] = ("c" + cell_table["label"].astype(str)).astype("string")
        cell_table = cell_table.loc[:, _CELL_COLUMNS]

    laser_labels, laser_counts = np.unique(lasers.reshape(-1), return_counts=True)
    laser_areas = {
        int(label): int(count)
        for label, count in zip(laser_labels, laser_counts, strict=True)
        if int(label) > 0
    }
    foreground = (cells.reshape(-1) > 0) & (lasers.reshape(-1) > 0)
    if not foreground.any():
        return cell_table, laser_areas, []

    foreground_cell_groups = cell_inverse[foreground]
    foreground_laser_groups = np.searchsorted(laser_labels, lasers.reshape(-1)[foreground])
    grouped_overlaps = [
        (
            int(cell_labels[cell_group]),
            int(laser_labels[laser_group]),
            int(count),
        )
        for cell_group, laser_group, count in _group_pair_counts(
            foreground_cell_groups, foreground_laser_groups
        )
    ]
    return cell_table, laser_areas, grouped_overlaps


def build_overlap_table(
    laser_adata: anndata.AnnData,
    cell_segmentation: np.ndarray,
    laser_segmentation: np.ndarray,
    *,
    boundary_margin: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact raster overlaps for cells within the registered laser boundary.

    Raster labels that have no corresponding registered observation are omitted.  A
    registered label with zero raster pixels cannot produce a record, avoiding an
    undefined ratio before any abundance method is applied.
    """
    cells, lasers = _validate_labels(cell_segmentation, laser_segmentation)
    margin = _validate_boundary_margin(boundary_margin)
    adata, _, centroids = _validate_laser_adata(laser_adata)
    connections = _laser_connections(adata)

    cell_table, laser_areas, grouped_overlaps = _group_raster_pixels(cells, lasers)
    min_row, min_column = centroids.min(axis=0) - margin
    max_row, max_column = centroids.max(axis=0) + margin
    within_boundary = (
        cell_table["centroid-0"].between(min_row, max_row, inclusive="neither")
        & cell_table["centroid-1"].between(min_column, max_column, inclusive="neither")
    )
    cell_table = cell_table.loc[within_boundary].copy().reset_index(drop=True)

    included_cells = set(cell_table["label"].tolist())
    records: list[dict[str, object]] = []
    for cell_label, laser_label, count in grouped_overlaps:
        laser_area = laser_areas.get(laser_label, 0)
        if cell_label not in included_cells or laser_label not in connections or laser_area == 0:
            continue
        records.append(
            {
                "cell_id": f"c{cell_label}",
                "cell_label": cell_label,
                "laser_label": laser_label,
                "laser_obs_id": connections[laser_label],
                "occupied_pixels": count,
                "laser_area": laser_area,
                "occupied_ratio": float(count / laser_area),
            }
        )

    overlaps = _empty_overlap_table()
    if records:
        overlaps = pd.DataFrame.from_records(records, columns=_OVERLAP_COLUMNS).astype(
            {
                "cell_id": "string",
                "laser_obs_id": "string",
                "occupied_pixels": "int64",
                "laser_area": "int64",
                "occupied_ratio": "float64",
            }
        )
        overlaps["cell_label"] = overlaps["cell_label"].map(int).astype(object)
        overlaps["laser_label"] = overlaps["laser_label"].map(int).astype(object)
    return overlaps, cell_table


def _validate_selector_overlaps(overlaps: object) -> pd.DataFrame:
    if not isinstance(overlaps, pd.DataFrame):
        raise QuantificationError("overlaps must be a pandas DataFrame")
    missing = sorted(set(_SELECTOR_COLUMNS) - set(overlaps.columns))
    if missing:
        raise QuantificationError(f"Overlap table is missing required columns: {missing}")
    if overlaps.loc[:, _SELECTOR_COLUMNS].isna().any(axis=None):
        raise QuantificationError("Overlap table required columns must not contain missing values")
    try:
        duplicated = overlaps.duplicated(subset=["cell_id", "laser_label"], keep=False)
    except TypeError as exc:
        raise QuantificationError("Overlap cell and laser identifiers must be hashable") from exc
    if duplicated.any():
        raise QuantificationError("Overlap table must contain unique cell-laser mappings")
    for value in overlaps["occupied_ratio"].astype(object).tolist():
        _validate_ratio(value, "occupied_ratio")
    return overlaps.copy(deep=True)


def _empty_rejected(overlaps: pd.DataFrame) -> pd.DataFrame:
    rejected = overlaps.iloc[0:0].copy()
    rejected["rejection_reason"] = pd.Series(dtype="string")
    return rejected


def select_specificity_filtered(
    overlaps: pd.DataFrame,
    *,
    unique_min_overlap: float,
    dominant_min_overlap: float,
    secondary_max_overlap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select unambiguous laser-to-cell mappings for specificity quantification."""
    unique_threshold = _validate_ratio(unique_min_overlap, "unique_min_overlap")
    dominant_threshold = _validate_ratio(dominant_min_overlap, "dominant_min_overlap")
    secondary_threshold = _validate_ratio(secondary_max_overlap, "secondary_max_overlap")
    table = _validate_selector_overlaps(overlaps)

    accepted: list[pd.DataFrame] = []
    rejected: list[pd.DataFrame] = []
    for _, group in table.groupby("laser_label", sort=True):
        ordered = group.sort_values("occupied_ratio", ascending=False, kind="mergesort").copy()
        if len(ordered) == 1:
            if float(ordered.iloc[0]["occupied_ratio"]) > unique_threshold:
                accepted.append(ordered)
            else:
                ordered["rejection_reason"] = "unique_overlap_not_above_threshold"
                rejected.append(ordered)
            continue

        top_ratio = float(ordered.iloc[0]["occupied_ratio"])
        second_ratio = float(ordered.iloc[1]["occupied_ratio"])
        if (
            top_ratio > second_ratio
            and top_ratio > dominant_threshold
            and second_ratio < secondary_threshold
        ):
            accepted.append(ordered.iloc[[0]].copy())
            non_dominant = ordered.iloc[1:].copy()
            non_dominant["rejection_reason"] = "non_dominant_mapping"
            rejected.append(non_dominant)
        else:
            ordered["rejection_reason"] = "ambiguous_multi_mapping"
            rejected.append(ordered)

    accepted_df = pd.concat(accepted, ignore_index=True) if accepted else table.iloc[0:0].copy()
    rejected_df = pd.concat(rejected, ignore_index=True) if rejected else _empty_rejected(table)
    rejected_df["rejection_reason"] = rejected_df["rejection_reason"].astype("string")
    return accepted_df, rejected_df


def _spectrum(adata: anndata.AnnData, obs_id: str) -> np.ndarray:
    try:
        matrix = adata[[obs_id], :].X
    except KeyError as exc:
        raise QuantificationError(
            f"Overlap laser observation {obs_id!r} is absent from laser_adata"
        ) from exc
    if sparse.issparse(matrix):
        return matrix.toarray().ravel()
    return np.asarray(matrix).ravel()


def _quantify_legacy(
    laser_adata: anndata.AnnData, accepted: pd.DataFrame, cell_id: str
) -> np.ndarray:
    expression = np.zeros(laser_adata.n_vars, dtype=float)
    for row in accepted.loc[accepted["cell_id"] == cell_id].itertuples(index=False):
        expression += _spectrum(laser_adata, row.laser_obs_id) * float(row.occupied_ratio)
    return expression


def _quantify_specificity(
    laser_adata: anndata.AnnData,
    overlaps: pd.DataFrame,
    accepted: pd.DataFrame,
    cell_id: str,
) -> np.ndarray:
    rows = accepted.loc[accepted["cell_id"] == cell_id]
    if rows.empty:
        return np.zeros(laser_adata.n_vars, dtype=float)

    numerator = np.zeros(laser_adata.n_vars, dtype=float)
    specificity_sum = 0.0
    for row in rows.itertuples(index=False):
        total_occupied = float(
            overlaps.loc[overlaps["laser_label"] == row.laser_label, "occupied_pixels"].sum()
        )
        if total_occupied <= 0.0:
            raise QuantificationError("Specificity denominator must be positive")
        sampling_specificity = float(row.occupied_pixels) / total_occupied
        numerator += (
            _spectrum(laser_adata, row.laser_obs_id)
            * sampling_specificity
            * float(row.occupied_ratio)
        )
        specificity_sum += sampling_specificity
    return numerator / specificity_sum


def _mapping_text(mapped: pd.DataFrame, column: str) -> str:
    if mapped.empty:
        return "0"
    return "_".join(mapped[column].astype(str).tolist())


def quantify_cells(
    laser_adata: anndata.AnnData,
    cell_segmentation: np.ndarray,
    laser_segmentation: np.ndarray,
    *,
    method: Literal["specificity_filtered", "legacy_proportional"] = "specificity_filtered",
    boundary_margin: float = 20.0,
    unique_min_overlap: float = 0.10,
    dominant_min_overlap: float = 0.60,
    secondary_max_overlap: float = 0.15,
) -> QuantificationResult:
    """Quantify every in-boundary cell using one of the approved JOINT strategies."""
    if not isinstance(method, str) or method not in _QUANTIFICATION_METHODS:
        raise QuantificationError(f"Unsupported quantification method: {method!r}")
    unique_threshold = _validate_ratio(unique_min_overlap, "unique_min_overlap")
    dominant_threshold = _validate_ratio(dominant_min_overlap, "dominant_min_overlap")
    secondary_threshold = _validate_ratio(secondary_max_overlap, "secondary_max_overlap")

    overlaps, cells = build_overlap_table(
        laser_adata,
        cell_segmentation,
        laser_segmentation,
        boundary_margin=boundary_margin,
    )
    if method == "specificity_filtered":
        accepted, rejected = select_specificity_filtered(
            overlaps,
            unique_min_overlap=unique_threshold,
            dominant_min_overlap=dominant_threshold,
            secondary_max_overlap=secondary_threshold,
        )
    else:
        accepted = overlaps.copy(deep=True)
        rejected = _empty_rejected(overlaps)

    cell_ids = cells["cell_id"].tolist()
    matrix = np.zeros((len(cell_ids), laser_adata.n_vars), dtype=float)
    for index, cell_id in enumerate(cell_ids):
        if method == "specificity_filtered":
            matrix[index] = _quantify_specificity(
                laser_adata, overlaps, accepted, cell_id
            )
        else:
            matrix[index] = _quantify_legacy(laser_adata, accepted, cell_id)

    obs = cells.set_index("cell_id").copy(deep=True)
    label_values = obs["label"].tolist()
    label_dtype = "uint64" if max(label_values, default=0) > np.iinfo(np.int64).max else "int64"
    obs["label"] = pd.Series(label_values, index=obs.index, dtype=label_dtype)
    quantified_ids = set(accepted["cell_id"].tolist())
    obs["quantification_status"] = pd.Series(
        ["quantified" if cell_id in quantified_ids else "zero" for cell_id in cell_ids],
        index=obs.index,
        dtype="string",
    )
    obs["laser_labels"] = pd.Series(index=obs.index, dtype="string")
    obs["laser_coverage"] = pd.Series(index=obs.index, dtype="string")
    for cell_id in cell_ids:
        mapped = accepted.loc[accepted["cell_id"] == cell_id].sort_values(
            "laser_label", kind="mergesort"
        )
        obs.loc[cell_id, "laser_labels"] = _mapping_text(mapped, "laser_label")
        obs.loc[cell_id, "laser_coverage"] = _mapping_text(mapped, "occupied_pixels")

    cell_adata = anndata.AnnData(
        sparse.csr_matrix(matrix),
        obs=obs,
        var=laser_adata.var.copy(deep=True),
    )
    cell_adata.obsm["spatial"] = obs[["centroid-1", "centroid-0"]].to_numpy(dtype=float)
    cell_adata.uns["joint"] = {"quantification_method": method}
    report = {
        "included_cells": len(cell_ids),
        "quantified_cells": int(obs["quantification_status"].eq("quantified").sum()),
        "accepted_mappings": len(accepted),
        "rejected_mappings": len(rejected),
    }
    return QuantificationResult(cell_adata, overlaps, accepted, rejected, report)


def _exact_real_ratio(value: Real) -> tuple[int, int]:
    if isinstance(value, Integral):
        return int(value), 1
    if isinstance(value, Rational):
        return int(value.numerator), int(value.denominator)
    numerator, denominator = value.as_integer_ratio()
    return int(numerator), int(denominator)


def _validate_mixed_features(
    laser_adata: anndata.AnnData, cell_adata: anndata.AnnData
) -> None:
    mz_by_input: dict[str, list[tuple[int, int]]] = {}
    for name, adata in (("laser_adata", laser_adata), ("quantification.adata", cell_adata)):
        if not adata.var_names.is_unique:
            raise QuantificationError(f"{name} must have unique variable identifiers")
        if "mz" not in adata.var.columns:
            raise QuantificationError(f"{name}.var must contain numeric 'mz' values")
        values = adata.var["mz"].astype(object).tolist()
        if not all(
            isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            raise QuantificationError(f"{name}.var must contain numeric 'mz' values")
        if not all(
            isinstance(value, Rational) or bool(np.isfinite(value)) for value in values
        ):
            raise QuantificationError(f"{name}.var must contain finite 'mz' values")
        mz_by_input[name] = [_exact_real_ratio(value) for value in values]

    same_feature_count = laser_adata.n_vars == cell_adata.n_vars
    exact_mz_match = same_feature_count and all(
        laser_mz == cell_mz
        for laser_mz, cell_mz in zip(
            mz_by_input["laser_adata"],
            mz_by_input["quantification.adata"],
            strict=True,
        )
    )
    if (
        not same_feature_count
        or not laser_adata.var_names.equals(cell_adata.var_names)
        or not exact_mz_match
    ):
        raise QuantificationError(
            "laser_adata and quantification.adata features must have identical count, order, "
            "and identifiers"
        )


def _validate_cell_spatial(cell_adata: anndata.AnnData) -> None:
    if "spatial" not in cell_adata.obsm:
        raise QuantificationError('quantification.adata.obsm must contain "spatial"')
    try:
        spatial = np.asarray(cell_adata.obsm["spatial"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise QuantificationError(
            'quantification.adata.obsm["spatial"] must contain numeric coordinates'
        ) from exc
    if spatial.shape != (cell_adata.n_obs, 2) or not np.isfinite(spatial).all():
        raise QuantificationError(
            'quantification.adata.obsm["spatial"] must be a finite n_obs by 2 array'
        )


def _integral_accepted_column(
    accepted: pd.DataFrame, column: str, *, positive: bool
) -> np.ndarray:
    result: list[int] = []
    for value in accepted[column].astype(object).tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise QuantificationError(
                f"Accepted overlap column {column!r} must contain integral values"
            )
        if isinstance(value, Integral):
            integer = int(value)
        else:
            numeric = float(value)
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise QuantificationError(
                    f"Accepted overlap column {column!r} must contain finite integral values"
                )
            integer = int(numeric)
        if (positive and integer <= 0) or (not positive and integer < 0):
            boundary = "positive" if positive else "non-negative"
            raise QuantificationError(
                f"Accepted overlap column {column!r} must contain {boundary} values"
            )
        result.append(integer)
    return np.asarray(result, dtype=object)


def _validate_accepted_for_mixed(
    accepted: object,
    *,
    name: str = "accepted",
    laser_adata: anndata.AnnData,
    cell_adata: anndata.AnnData,
    laser_connections: dict[int, str],
) -> pd.DataFrame:
    if not isinstance(accepted, pd.DataFrame):
        raise QuantificationError(f"quantification.{name} must be a pandas DataFrame")
    missing = sorted(set(_OVERLAP_COLUMNS) - set(accepted.columns))
    if missing:
        raise QuantificationError(
            f"quantification.{name} overlap table is missing columns: {missing}"
        )
    table = accepted.loc[:, _OVERLAP_COLUMNS].copy(deep=True)
    if table.isna().any(axis=None):
        raise QuantificationError(
            f"quantification.{name} overlap table must not contain missing values"
        )

    for column, valid_ids in (
        ("cell_id", set(cell_adata.obs_names.tolist())),
        ("laser_obs_id", set(laser_adata.obs_names.tolist())),
    ):
        values = table[column].astype(object).tolist()
        if not all(isinstance(value, str) for value in values):
            raise QuantificationError(
                f"quantification.{name} column {column!r} must contain exact string identifiers"
            )
        unknown = sorted(set(values) - valid_ids)
        if unknown:
            raise QuantificationError(
                f"quantification.{name} column {column!r} contains unknown exact identifiers: "
                f"{unknown}"
            )

    cell_labels = _integral_accepted_column(table, "cell_label", positive=True)
    laser_labels = _integral_accepted_column(table, "laser_label", positive=True)
    occupied_pixels = _integral_accepted_column(table, "occupied_pixels", positive=False)
    laser_areas = _integral_accepted_column(table, "laser_area", positive=True)
    if any(
        int(occupied) > int(area)
        for occupied, area in zip(occupied_pixels, laser_areas, strict=True)
    ):
        raise QuantificationError(
            "Accepted overlap occupied_pixels must not exceed the constituent laser_area"
        )
    for value in table["occupied_ratio"].astype(object).tolist():
        try:
            _validate_ratio(value, "Accepted overlap occupied_ratio")
        except QuantificationError as exc:
            raise QuantificationError(str(exc)) from exc
    expected_ratios = np.asarray(
        [
            int(occupied) / int(area)
            for occupied, area in zip(occupied_pixels, laser_areas, strict=True)
        ],
        dtype=float,
    )
    actual_ratios = table["occupied_ratio"].to_numpy(dtype=float, copy=True)
    if not np.allclose(actual_ratios, expected_ratios, rtol=1e-12, atol=0.0):
        raise QuantificationError(
            f"quantification.{name} occupied_ratio must equal occupied_pixels / laser_area"
        )

    try:
        duplicated = table.duplicated(subset=["cell_id", "laser_label"], keep=False)
    except TypeError as exc:
        raise QuantificationError(
            f"quantification.{name} overlap identifiers must be hashable"
        ) from exc
    if duplicated.any():
        raise QuantificationError(
            f"quantification.{name} contains duplicate cell-laser mappings"
        )

    for obs_id, laser_label in zip(
        table["laser_obs_id"].tolist(), laser_labels.tolist(), strict=True
    ):
        if laser_connections.get(int(laser_label)) != obs_id:
            raise QuantificationError(
                f"quantification.{name} laser_label does not match its exact laser_obs_id"
            )

    if "label" in cell_adata.obs.columns:
        expected_cell_labels: dict[str, int] = {}
        for cell_id, value in cell_adata.obs["label"].items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise QuantificationError(
                    "quantification.adata obs column 'label' must contain integral values"
                )
            if isinstance(value, Integral):
                integer = int(value)
            else:
                numeric = float(value)
                if not np.isfinite(numeric) or not numeric.is_integer():
                    raise QuantificationError(
                        "quantification.adata obs column 'label' must contain integral values"
                    )
                integer = int(numeric)
            if integer <= 0:
                raise QuantificationError(
                    "quantification.adata obs column 'label' must contain positive integral values"
                )
            expected_cell_labels[str(cell_id)] = integer
        for cell_id, cell_label in zip(
            table["cell_id"].tolist(), cell_labels.tolist(), strict=True
        ):
            if expected_cell_labels[cell_id] != int(cell_label):
                raise QuantificationError(
                    f"quantification.{name} cell_label does not match its exact cell_id"
                )

    table["cell_label"] = cell_labels
    table["laser_label"] = laser_labels
    table["occupied_pixels"] = occupied_pixels
    table["laser_area"] = laser_areas
    return table


def _validate_result_table_columns(
    table: pd.DataFrame, name: str, *, extra_columns: tuple[str, ...] = ()
) -> None:
    missing = sorted((set(_OVERLAP_COLUMNS) | set(extra_columns)) - set(table.columns))
    if missing:
        raise QuantificationError(f"quantification.{name} is missing columns: {missing}")


def _overlap_row_counter(table: pd.DataFrame) -> Counter[tuple[object, ...]]:
    return Counter(table.loc[:, _OVERLAP_COLUMNS].itertuples(index=False, name=None))


def _validate_overlap_partition(
    overlaps: pd.DataFrame, accepted: pd.DataFrame, rejected: pd.DataFrame
) -> None:
    partition = _overlap_row_counter(accepted) + _overlap_row_counter(rejected)
    if partition != _overlap_row_counter(overlaps):
        raise QuantificationError(
            "quantification.accepted and quantification.rejected must form a coherent, "
            "disjoint partition of quantification.overlaps"
        )


def _decimal_real(value: Real) -> Decimal:
    if isinstance(value, Integral):
        return Decimal(int(value))
    return Decimal(str(value))


def build_mixed_anndata(
    laser_adata: anndata.AnnData,
    quantification: QuantificationResult,
    *,
    min_residual_fraction: float = 0.2,
) -> anndata.AnnData:
    """Combine quantified cells with residual laser-observation spectra.

    Residual fractions are calculated from accepted occupied pixels grouped by the
    exact registered laser observation identifier.  This keeps combined segmentation
    labels area-weighted instead of adding their constituent overlap ratios.
    """
    threshold = _validate_ratio(min_residual_fraction, "min_residual_fraction")
    if not isinstance(laser_adata, anndata.AnnData):
        raise QuantificationError("laser_adata must be an AnnData object")
    if not isinstance(quantification, QuantificationResult):
        raise QuantificationError("quantification must be a QuantificationResult")
    if not isinstance(quantification.adata, anndata.AnnData):
        raise QuantificationError("quantification.adata must be an AnnData object")
    if not isinstance(quantification.overlaps, pd.DataFrame):
        raise QuantificationError("quantification.overlaps must be a pandas DataFrame")
    if not isinstance(quantification.rejected, pd.DataFrame):
        raise QuantificationError("quantification.rejected must be a pandas DataFrame")
    if not isinstance(quantification.report, dict):
        raise QuantificationError("quantification.report must be a dictionary")
    _validate_result_table_columns(quantification.rejected, "rejected", extra_columns=("rejection_reason",))

    validated_laser, areas, _ = _validate_laser_adata(laser_adata)
    if any(value <= 0 for value in areas):
        raise QuantificationError("Laser obs column 'area' must contain positive values")
    laser_connections = _laser_connections(validated_laser)

    cell_adata = quantification.adata
    if not cell_adata.obs_names.is_unique:
        raise QuantificationError(
            "Quantification AnnData must have unique observation identifiers"
        )
    _validate_mixed_features(validated_laser, cell_adata)
    _validate_cell_spatial(cell_adata)
    overlaps = _validate_accepted_for_mixed(
        quantification.overlaps,
        name="overlaps",
        laser_adata=validated_laser,
        cell_adata=cell_adata,
        laser_connections=laser_connections,
    )
    accepted = _validate_accepted_for_mixed(
        quantification.accepted,
        laser_adata=validated_laser,
        cell_adata=cell_adata,
        laser_connections=laser_connections,
    )
    rejected = _validate_accepted_for_mixed(
        quantification.rejected,
        name="rejected",
        laser_adata=validated_laser,
        cell_adata=cell_adata,
        laser_connections=laser_connections,
    )
    if not all(
        isinstance(value, str) and bool(value.strip())
        for value in quantification.rejected["rejection_reason"].astype(object).tolist()
    ):
        raise QuantificationError(
            "quantification.rejected rejection_reason must contain non-empty strings"
        )
    _validate_overlap_partition(overlaps, accepted, rejected)

    cells = cell_adata.copy()
    cells.X = sparse.csr_matrix(cells.X, copy=True)
    cells.obs["data_type"] = pd.Series("cell", index=cells.obs_names, dtype="string")
    cells.uns = deepcopy(cell_adata.uns)

    occupied_by_obs = (
        accepted.groupby("laser_obs_id", sort=False, observed=False)["occupied_pixels"]
        .sum()
        .to_dict()
    )
    residual_rows: list[np.ndarray] = []
    residual_records: list[pd.Series] = []
    residual_ids: list[str] = []
    output_obs_ids = set(cells.obs_names.tolist())
    residual_spatial: list[tuple[float, float]] = []
    decimal_threshold = Decimal(str(threshold))
    for (obs_id, laser_row), area in zip(
        validated_laser.obs.iterrows(), areas, strict=True
    ):
        occupied_pixels = int(occupied_by_obs.get(obs_id, 0))
        decimal_residual = Decimal(1) - Decimal(occupied_pixels) / _decimal_real(area)
        decimal_residual = min(
            Decimal(1),
            max(Decimal(0), decimal_residual),
        )
        if decimal_residual < decimal_threshold:
            continue
        residual_fraction = float(decimal_residual)
        residual_id = f"laser_residual_{obs_id}"
        if residual_id in output_obs_ids:
            raise QuantificationError(
                f"Residual observation identifier collision prevents unique output: {residual_id!r}"
            )
        residual_row = laser_row.copy(deep=True)
        residual_row["data_type"] = "laser_residual"
        residual_row["residual_fraction"] = residual_fraction
        residual_rows.append(_spectrum(validated_laser, obs_id) * residual_fraction)
        residual_records.append(residual_row)
        residual_ids.append(residual_id)
        output_obs_ids.add(residual_id)
        residual_spatial.append(
            (float(laser_row["centroid-1"]), float(laser_row["centroid-0"]))
        )

    if not residual_rows:
        return cells

    residual_obs = pd.DataFrame(residual_records, index=pd.Index(residual_ids, dtype=str))
    residual_obs["data_type"] = pd.Series(
        "laser_residual", index=residual_obs.index, dtype="string"
    )
    residual_adata = anndata.AnnData(
        sparse.csr_matrix(np.vstack(residual_rows)),
        obs=residual_obs,
        var=cells.var.copy(deep=True),
        obsm={"spatial": np.asarray(residual_spatial, dtype=float)},
    )
    mixed = anndata.concat(
        [cells, residual_adata],
        join="outer",
        merge="same",
        index_unique=None,
    )
    if not mixed.obs_names.is_unique:
        raise QuantificationError("Mixed AnnData output observation identifiers must be unique")
    mixed.X = sparse.csr_matrix(mixed.X, copy=True)
    mixed.var = cells.var.copy(deep=True)
    mixed.uns = deepcopy(cells.uns)
    return mixed
