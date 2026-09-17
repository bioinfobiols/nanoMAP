from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Literal

import anndata
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import draw, feature, filters, measure, morphology, segmentation

from joint.errors import SegmentationError
from joint.models import SegmentationResult

_REGION_COLUMNS = ["label", "area", "centroid-0", "centroid-1"]


def _validate_region_table(regions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(regions, pd.DataFrame):
        raise SegmentationError("regions must be a pandas DataFrame")
    missing = set(_REGION_COLUMNS) - set(regions.columns)
    if missing:
        raise SegmentationError(f"Region table is missing columns: {sorted(missing)}")
    for column in _REGION_COLUMNS:
        try:
            values = pd.to_numeric(regions[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise SegmentationError(f"Region table column {column!r} must be numeric") from exc
        if not np.isfinite(values).all():
            raise SegmentationError(f"Region table column {column!r} must be finite")
        if column == "label" and ((values < 1).any() or not np.equal(values, np.floor(values)).all()):
            raise SegmentationError("Region table column 'label' must contain positive integer values")
    return regions.copy()


def _validate_positive_integer(name: str, value: int) -> int:
    return _validate_integer(name, value, minimum=1)


def _validate_finite_number(name: str, value: float, *, minimum: float | None = None) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SegmentationError(f"{name} must be a finite number") from exc
    if not np.isfinite(numeric) or (minimum is not None and numeric < minimum):
        qualifier = "a finite non-negative number" if minimum == 0 else "a finite number"
        raise SegmentationError(f"{name} must be {qualifier}")
    return numeric


def _validate_image(image: np.ndarray) -> np.ndarray:
    try:
        data = np.asarray(image, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SegmentationError("Laser image must contain numeric grayscale data") from exc
    if data.ndim != 2:
        raise SegmentationError("Laser image must be two-dimensional grayscale data")
    if data.size == 0:
        raise SegmentationError("Laser image must contain at least one pixel")
    if not np.isfinite(data).all():
        raise SegmentationError("Laser image contains non-finite values")
    return data.copy()


def _validate_integer(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SegmentationError(f"{name} must be an integer")
    if value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise SegmentationError(f"{name} must be {qualifier}")
    return int(value)


def _validate_intensity_cutoff(value: float) -> float:
    try:
        cutoff = float(value)
    except (TypeError, ValueError) as exc:
        raise SegmentationError("intensity_cutoff must be a finite non-negative number") from exc
    if not np.isfinite(cutoff) or cutoff < 0:
        raise SegmentationError("intensity_cutoff must be a finite non-negative number")
    return cutoff


def _validate_threshold(threshold: Literal["otsu"] | float) -> Literal["otsu"] | float:
    if isinstance(threshold, str):
        if threshold != "otsu":
            raise SegmentationError("threshold must be 'otsu' or a finite number")
        return threshold
    try:
        cutoff = float(threshold)
    except (TypeError, ValueError) as exc:
        raise SegmentationError("threshold must be 'otsu' or a finite number") from exc
    if not np.isfinite(cutoff):
        raise SegmentationError("threshold must be 'otsu' or a finite number")
    return cutoff


def region_table(labels: np.ndarray) -> pd.DataFrame:
    data = np.asarray(labels)
    if data.ndim != 2 or data.dtype.kind not in "iu" or np.any(data < 0):
        raise SegmentationError("labels must be a two-dimensional non-negative integer array")
    if data.size and int(data.max()) > np.iinfo(np.int32).max:
        raise SegmentationError("labels must fit in the supported int32 range")
    properties = measure.regionprops_table(
        data.astype(np.int32, copy=False), properties=("label", "area", "centroid")
    )
    return (
        pd.DataFrame(properties, columns=_REGION_COLUMNS)
        .sort_values("label")
        .reset_index(drop=True)
    )


def _remove_small_regions(labels: np.ndarray, min_size: int) -> np.ndarray:
    values, counts = np.unique(labels, return_counts=True)
    remove = values[(values != 0) & (counts < min_size)]
    filtered = labels.copy()
    filtered[np.isin(filtered, remove)] = 0
    return filtered


def segment_laser_marks(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    intensity_cutoff: float = 0.2,
    top_hat_radius: int = 10,
    top_hat_mode: Literal["enhance", "legacy_opening"] = "enhance",
    closing_size: int = 5,
    post_closing_size: int = 1,
    opening_size: int = 3,
    min_peak_distance: int = 25,
    peak_detection_mode: Literal["per_component", "legacy_global"] = "per_component",
    marker_connectivity: int = 2,
    min_size: int = 30,
    threshold: Literal["otsu"] | float = "otsu",
) -> SegmentationResult:
    processed = _validate_image(image)
    intensity_cutoff = _validate_intensity_cutoff(intensity_cutoff)
    top_hat_radius = _validate_integer("top_hat_radius", top_hat_radius, minimum=0)
    if not isinstance(top_hat_mode, str) or top_hat_mode not in {"enhance", "legacy_opening"}:
        raise SegmentationError("top_hat_mode must be 'enhance' or 'legacy_opening'")
    closing_size = _validate_integer("closing_size", closing_size, minimum=1)
    post_closing_size = _validate_integer("post_closing_size", post_closing_size, minimum=1)
    opening_size = _validate_integer("opening_size", opening_size, minimum=1)
    min_peak_distance = _validate_integer("min_peak_distance", min_peak_distance, minimum=1)
    if not isinstance(peak_detection_mode, str) or peak_detection_mode not in {
        "per_component",
        "legacy_global",
    }:
        raise SegmentationError("peak_detection_mode must be 'per_component' or 'legacy_global'")
    marker_connectivity = _validate_integer(
        "marker_connectivity", marker_connectivity, minimum=1
    )
    if marker_connectivity > 2:
        raise SegmentationError("marker_connectivity must be 1 or 2")
    min_size = _validate_integer("min_size", min_size, minimum=1)
    threshold = _validate_threshold(threshold)

    include = np.ones(processed.shape, dtype=bool)
    if mask is not None:
        mask_data = np.asarray(mask)
        if mask_data.shape != processed.shape:
            raise SegmentationError("mask and image must have identical shapes")
        try:
            finite_mask = np.asarray(mask_data, dtype=float)
        except (TypeError, ValueError) as exc:
            raise SegmentationError("mask must contain finite boolean-compatible values") from exc
        if not np.isfinite(finite_mask).all():
            raise SegmentationError("mask contains non-finite values")
        include = mask_data.astype(bool, copy=False)
        processed[~include] = 0

    processed[processed < intensity_cutoff] = 0
    if top_hat_radius > 0:
        top_hat = morphology.white_tophat(processed, morphology.disk(top_hat_radius))
        if top_hat_mode == "enhance":
            processed = top_hat
        else:
            processed -= top_hat

    cutoff = filters.threshold_otsu(processed) if threshold == "otsu" else threshold
    binary = processed > cutoff
    binary &= include
    if closing_size > 1:
        binary = morphology.closing(
            binary, footprint=np.ones((closing_size, closing_size), dtype=bool)
        )
        binary &= include
    binary = segmentation.clear_border(binary)
    if post_closing_size > 1:
        binary = morphology.closing(
            binary, footprint=np.ones((post_closing_size, post_closing_size), dtype=bool)
        )
        binary &= include
    if opening_size > 1:
        binary = morphology.opening(
            binary, footprint=np.ones((opening_size, opening_size), dtype=bool)
        )
        binary &= include

    distance = ndi.distance_transform_edt(binary)
    components = measure.label(binary, connectivity=2)
    if peak_detection_mode == "legacy_global":
        coordinates = feature.peak_local_max(distance, min_distance=min_peak_distance)
    else:
        coordinates = feature.peak_local_max(
            distance,
            min_distance=min_peak_distance,
            labels=components,
            exclude_border=False,
        )
    maxima = np.zeros_like(binary, dtype=bool)
    if coordinates.size:
        maxima[tuple(coordinates.T)] = True
    if peak_detection_mode == "per_component":
        for component_label in range(1, int(components.max()) + 1):
            component = components == component_label
            if not maxima[component].any():
                position = np.unravel_index(
                    np.argmax(np.where(component, distance, -np.inf)), distance.shape
                )
                maxima[position] = True
    if not maxima.any():
        raise SegmentationError("No laser markers were detected; inspect mask and thresholds")
    markers = measure.label(maxima, connectivity=marker_connectivity)
    labels = segmentation.watershed(
        -distance, markers, mask=binary & include, connectivity=2
    )
    labels = _remove_small_regions(labels, min_size)
    regions = region_table(labels)
    diagnostics = {
        "threshold": float(cutoff),
        "object_count": int(regions.shape[0]),
        "foreground_pixels": int(np.count_nonzero(labels)),
        "processed_image": processed,
        "binary_mask": binary,
    }
    return SegmentationResult(labels=labels, regions=regions, diagnostics=diagnostics)


def _row_breaks_by_gap(regions: pd.DataFrame, expected_rows: int | None) -> np.ndarray:
    coordinates = regions["centroid-0"].to_numpy(dtype=float)
    gaps = np.diff(coordinates)
    if expected_rows is not None:
        count = expected_rows - 1
        if count > gaps.size:
            raise SegmentationError(f"Cannot create expected {expected_rows} rows")
        if count == 0:
            return np.array([], dtype=int)
        return np.sort(np.argsort(gaps)[-count:] + 1)
    if gaps.size == 0:
        return np.array([], dtype=int)
    median = float(np.median(gaps))
    mad = float(np.median(np.abs(gaps - median)))
    threshold = median + max(5 * mad, np.finfo(float).eps)
    return np.flatnonzero(gaps > threshold) + 1


def _row_breaks_by_variance(
    regions: pd.DataFrame,
    *,
    window_size: int,
    variance_threshold: float,
    thinning: int,
) -> np.ndarray:
    coordinates = regions["centroid-0"].to_numpy(dtype=float)
    if window_size > coordinates.size:
        return np.array([], dtype=int)
    variances = np.array(
        [
            np.var(coordinates[index - window_size : index])
            for index in range(window_size, len(coordinates))
        ]
    )
    candidates = np.flatnonzero(variances > variance_threshold)[::thinning]
    return candidates + window_size - 1


def detect_rows(
    regions: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    method: Literal["coordinate_gap", "rolling_variance"] = "coordinate_gap",
    window_size: int = 5,
    variance_threshold: float = 50.0,
    thinning: int = 4,
) -> list[pd.DataFrame]:
    """Assign deterministic row and within-row column numbers to laser regions."""
    ordered = _validate_region_table(regions).sort_values(
        ["centroid-0", "centroid-1"], kind="stable"
    ).reset_index(drop=True)
    if expected_rows is not None:
        expected_rows = _validate_positive_integer("expected_rows", expected_rows)
    if method not in {"coordinate_gap", "rolling_variance"}:
        raise SegmentationError("method must be 'coordinate_gap' or 'rolling_variance'")
    window_size = _validate_positive_integer("window_size", window_size)
    variance_threshold = _validate_finite_number("variance_threshold", variance_threshold, minimum=0)
    thinning = _validate_positive_integer("thinning", thinning)
    if ordered.empty:
        if expected_rows is not None:
            raise SegmentationError(f"Detected 0 rows; expected {expected_rows}")
        return []

    breaks = (
        _row_breaks_by_gap(ordered, expected_rows)
        if method == "coordinate_gap"
        else _row_breaks_by_variance(
            ordered,
            window_size=window_size,
            variance_threshold=variance_threshold,
            thinning=thinning,
        )
    )
    edges = np.concatenate(([0], breaks, [len(ordered)]))
    pieces = [ordered.iloc[start:stop] for start, stop in pairwise(edges)]
    if expected_rows is not None and len(pieces) != expected_rows:
        raise SegmentationError(f"Detected {len(pieces)} rows; expected {expected_rows}")

    rows: list[pd.DataFrame] = []
    for row_number, piece in enumerate(pieces, start=1):
        row = piece.sort_values(["centroid-1", "centroid-0"], kind="stable").copy()
        row["seg_label"] = row["label"].astype(int).astype(str)
        row["morphology"] = "intact"
        row["row_number"] = row_number
        row["column_number"] = np.arange(1, len(row) + 1)
        row["right_to_left"] = np.arange(-len(row), 0)
        rows.append(row.reset_index(drop=True))
    return rows


def exclude_regions(
    regions: pd.DataFrame,
    *,
    min_row: float | None = None,
    max_row: float | None = None,
    min_column: float | None = None,
    max_column: float | None = None,
) -> pd.DataFrame:
    """Return a copy containing only regions inside inclusive image-coordinate bounds."""
    result = _validate_region_table(regions)
    bounds = {
        "min_row": min_row,
        "max_row": max_row,
        "min_column": min_column,
        "max_column": max_column,
    }
    checked = {
        name: None if value is None else _validate_finite_number(name, value)
        for name, value in bounds.items()
    }
    if checked["min_row"] is not None and checked["max_row"] is not None and checked["min_row"] > checked["max_row"]:
        raise SegmentationError("min_row must be less than or equal to max_row")
    if (
        checked["min_column"] is not None
        and checked["max_column"] is not None
        and checked["min_column"] > checked["max_column"]
    ):
        raise SegmentationError("min_column must be less than or equal to max_column")

    keep = np.ones(len(result), dtype=bool)
    if checked["min_row"] is not None:
        keep &= result["centroid-0"].to_numpy(dtype=float) >= checked["min_row"]
    if checked["max_row"] is not None:
        keep &= result["centroid-0"].to_numpy(dtype=float) <= checked["max_row"]
    if checked["min_column"] is not None:
        keep &= result["centroid-1"].to_numpy(dtype=float) >= checked["min_column"]
    if checked["max_column"] is not None:
        keep &= result["centroid-1"].to_numpy(dtype=float) <= checked["max_column"]
    return result.loc[keep].copy()


def _parse_source_labels(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise SegmentationError("Merge rule source_labels must be a non-empty string")
    normalized = value.replace(",", "|").replace("_", "|")
    labels = [item.strip() for item in normalized.split("|") if item.strip()]
    if len(labels) < 2:
        raise SegmentationError("Merge rule source_labels must contain at least two labels")
    if len(labels) != len(set(labels)):
        raise SegmentationError("Merge rule source_labels must not contain duplicate labels")
    return labels


def _validate_merge_rules(merge_rules: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(merge_rules, pd.DataFrame):
        raise SegmentationError("merge_rules must be a pandas DataFrame")
    required = {"row_number", "source_labels"}
    missing = required - set(merge_rules.columns)
    if missing:
        raise SegmentationError(f"Merge rules are missing columns: {sorted(missing)}")
    rules = merge_rules.copy()
    if rules["row_number"].isna().any():
        raise SegmentationError("Merge rule row_number must be a positive integer")
    try:
        row_numbers = pd.to_numeric(rules["row_number"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise SegmentationError("Merge rule row_number must be a positive integer") from exc
    if (
        not np.isfinite(row_numbers).all()
        or (row_numbers < 1).any()
        or not np.equal(row_numbers, np.floor(row_numbers)).all()
    ):
        raise SegmentationError("Merge rule row_number must be a positive integer")
    rules["row_number"] = row_numbers.astype(int)
    rules["source_labels"] = rules["source_labels"].map(_parse_source_labels)
    return rules


def load_merge_rules(path: str | Path) -> pd.DataFrame:
    """Load and validate CSV correction rules for multi-region laser marks."""
    try:
        source = Path(path).expanduser().resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise SegmentationError(f"Merge rules file does not exist or is invalid: {path}") from exc
    if not source.is_file():
        raise SegmentationError(f"Merge rules path is not a file: {source}")
    try:
        rules = pd.read_csv(source, dtype={"source_labels": str})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SegmentationError(f"Could not read merge rules: {source}") from exc
    validated = _validate_merge_rules(rules)
    return validated.assign(source_labels=validated["source_labels"].map("|".join))


def merge_regions(regions: pd.DataFrame, merge_rules: pd.DataFrame) -> pd.DataFrame:
    """Add combined regions and mark their constituent regions as ``spoilt``."""
    result = _validate_region_table(regions)
    required = {"seg_label", "morphology", "row_number"}
    missing = required - set(result.columns)
    if missing:
        raise SegmentationError(f"Region table is missing columns: {sorted(missing)}")
    if result["row_number"].isna().any():
        raise SegmentationError("Region table row_number must be a positive integer")
    try:
        row_numbers = pd.to_numeric(result["row_number"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise SegmentationError("Region table row_number must be a positive integer") from exc
    if (
        not np.isfinite(row_numbers).all()
        or (row_numbers < 1).any()
        or not np.equal(row_numbers, np.floor(row_numbers)).all()
    ):
        raise SegmentationError("Region table row_number must be a positive integer")
    result["row_number"] = row_numbers.astype(int)
    if result["seg_label"].isna().any() or result["morphology"].isna().any():
        raise SegmentationError("Region table seg_label and morphology must not contain missing values")
    result["seg_label"] = result["seg_label"].astype(str)
    rules = _validate_merge_rules(merge_rules)

    for rule in rules.itertuples(index=False):
        labels = rule.source_labels
        intact = result.loc[
            (result["row_number"] == rule.row_number)
            & (result["morphology"] == "intact")
        ]
        matches = {label: int((intact["seg_label"] == label).sum()) for label in labels}
        if any(count != 1 for count in matches.values()):
            raise SegmentationError(
                f"Merge rule row {rule.row_number} requires exactly one intact match "
                f"for each source label; matches: {matches}"
            )
        selected = intact.loc[intact["seg_label"].isin(labels)]
        result.loc[selected.index, "morphology"] = "spoilt"
        combined = selected.iloc[0].copy()
        combined["centroid-0"] = selected["centroid-0"].mean()
        combined["centroid-1"] = selected["centroid-1"].mean()
        combined["area"] = selected["area"].sum()
        combined["seg_label"] = "_".join(labels)
        combined["morphology"] = "combined"
        result.loc[len(result)] = combined
    result = result.sort_values(
        ["row_number", "centroid-1", "centroid-0"], kind="stable"
    ).reset_index(drop=True)
    active = result["morphology"] != "spoilt"
    for indices in result.loc[active].groupby("row_number", sort=False).groups.values():
        count = len(indices)
        result.loc[indices, "column_number"] = np.arange(1, count + 1)
        result.loc[indices, "right_to_left"] = np.arange(-count, 0)
    return result


def simulate_laser_marks(
    adata: anndata.AnnData,
    anchor_rows: Sequence[pd.DataFrame],
    image_shape: tuple[int, int],
    *,
    radius: int,
) -> SegmentationResult:
    if not isinstance(adata, anndata.AnnData):
        raise SegmentationError("adata must be an AnnData object")
    if "spatial" not in adata.obsm:
        raise SegmentationError('adata.obsm must contain "spatial"')
    try:
        spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise SegmentationError("adata.obsm['spatial'] must contain numeric data") from exc
    if spatial.ndim != 2 or spatial.shape != (adata.n_obs, 2):
        raise SegmentationError(
            "adata.obsm['spatial'] must be a two-dimensional n_obs by 2 array"
        )
    if spatial.shape[0] == 0:
        raise SegmentationError("adata.obsm['spatial'] must contain at least one point")
    if not np.isfinite(spatial).all():
        raise SegmentationError("adata.obsm['spatial'] must contain finite values")

    x_values = np.unique(spatial[:, 0])
    y_values = np.unique(spatial[:, 1])
    unique_points = np.unique(spatial, axis=0)
    if (
        unique_points.shape[0] != spatial.shape[0]
        or spatial.shape[0] != x_values.size * y_values.size
    ):
        raise SegmentationError(
            "adata.obsm['spatial'] must represent a complete rectangular grid"
        )
    points_per_row = int(x_values.size)

    if not isinstance(anchor_rows, Sequence) or isinstance(anchor_rows, (str, bytes)):
        raise SegmentationError("anchor_rows must be a sequence of pandas DataFrames")
    anchors_by_row = list(anchor_rows)
    if len(anchors_by_row) != y_values.size:
        raise SegmentationError(
            f"MSI contains {y_values.size} rows but {len(anchors_by_row)} anchor rows were provided"
        )

    if not isinstance(image_shape, tuple) or len(image_shape) != 2:
        raise SegmentationError("image_shape must contain two positive integer dimensions")
    checked_shape = tuple(
        _validate_positive_integer("image_shape dimensions", dimension)
        for dimension in image_shape
    )
    radius = _validate_positive_integer("radius", radius)

    endpoints_by_row: list[tuple[np.ndarray, np.ndarray]] = []
    for row_number, anchors in enumerate(anchors_by_row, start=1):
        if not isinstance(anchors, pd.DataFrame):
            raise SegmentationError(f"Anchor row {row_number} must be a pandas DataFrame")
        missing = {"centroid-0", "centroid-1"} - set(anchors.columns)
        if missing:
            raise SegmentationError(
                f"Anchor row {row_number} is missing columns: {sorted(missing)}"
            )
        if len(anchors) < 2:
            raise SegmentationError(f"Anchor row {row_number} must contain usable endpoints")
        try:
            coordinates = anchors.loc[:, ["centroid-0", "centroid-1"]].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise SegmentationError(
                f"Anchor row {row_number} centroid data must be numeric and finite"
            ) from exc
        if not np.isfinite(coordinates).all():
            raise SegmentationError(f"Anchor row {row_number} centroid data must be finite")
        order = np.argsort(coordinates[:, 1], kind="stable")
        start, end = coordinates[order[[0, -1]]]
        if np.array_equal(start, end):
            raise SegmentationError(f"Anchor row {row_number} must contain usable endpoints")
        endpoints_by_row.append((start, end))

    labels = np.zeros(checked_shape, dtype=np.int32)
    records: list[dict[str, float | int | str]] = []
    label = 1
    for row_number, (start, end) in enumerate(endpoints_by_row, start=1):
        centers = np.linspace(start, end, points_per_row)
        for column_number, (row, column) in enumerate(centers, start=1):
            rr, cc = draw.disk((row, column), radius, shape=checked_shape)
            if len(rr) == 0:
                raise SegmentationError(
                    f"Simulated label {label} has no pixels inside image_shape"
                )
            if np.any(labels[rr, cc] != 0):
                raise SegmentationError(
                    "Simulated laser-mark disks overlap; choose non-overlapping anchors or radius"
                )
            labels[rr, cc] = label
            records.append(
                {
                    "label": label,
                    "area": float(len(rr)),
                    "centroid-0": float(rr.mean()),
                    "centroid-1": float(cc.mean()),
                    "seg_label": str(label),
                    "morphology": "simulated",
                    "row_number": row_number,
                    "column_number": column_number,
                    "right_to_left": column_number - points_per_row - 1,
                }
            )
            label += 1
    regions = pd.DataFrame.from_records(records)
    return SegmentationResult(
        labels=labels,
        regions=regions,
        diagnostics={"method": "simulated", "radius": radius, "object_count": len(records)},
    )
