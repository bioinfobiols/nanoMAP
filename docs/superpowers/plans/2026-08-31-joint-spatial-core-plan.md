# JOINT Spatial Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement laser-mark segmentation and simulation, structured row corrections, explicit MSI-to-laser registration, both approved cell-quantification strategies, and mixed AnnData construction.

**Architecture:** Image operations return immutable `SegmentationResult` objects. Registration converts ordered MSI grids and ordered laser tables into an explicit mapping and never hides truncation or orientation. Quantification first builds one shared overlap table, then applies a pluggable strategy so d2 and d8 calculations remain independently testable.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pandas, scikit-image, AnnData, pytest, Ruff; foundation APIs from Plan 1.

## Global Constraints

- Execute after `2026-08-31-joint-foundation-msi-plan.md` passes.
- Image arrays use `(row, column)`; AnnData spatial coordinates use `(x, y)`.
- Segmentation arrays are two-dimensional integer labels and use zero for background.
- `specificity_filtered` is the default quantification method; `legacy_proportional` remains public.
- Hard-coded d2 merge calls become structured data consumed by `merge_regions`.
- Ambiguous `orientation="auto"` raises `RegistrationError` with candidate diagnostics.
- No function silently truncates unmatched MSI pixels; `edge_policy` must be explicit.
- Public functions return new AnnData objects by default.

---

## Planned File Map

```text
pyproject.toml                         add scikit-image dependency
src/joint/config.py                    complete orientation and spatial parameters
src/joint/segmentation.py              real segmentation, row detection, merges, simulation
src/joint/registration.py              registration, transforms, image mounting
src/joint/quantification.py            overlaps, filtering, abundance, mixed AnnData
src/joint/__init__.py                  curated spatial-core exports
tests/test_segmentation.py              segmentation and correction tests
tests/test_registration.py              transform and mapping tests
tests/test_quantification.py            numeric strategy tests
tests/test_spatial_integration.py        synthetic end-to-end core test
```

### Task 1: Real Laser-Mark Segmentation

**Files:**
- Modify: `pyproject.toml`
- Create: `src/joint/segmentation.py`
- Create: `tests/test_segmentation.py`

**Interfaces:**
- Consumes: `SegmentationResult`, `SegmentationError`.
- Produces: `segment_laser_marks(image, *, mask=None, ...) -> SegmentationResult` and `region_table(labels) -> DataFrame`.

- [ ] **Step 1: Add scikit-image to core dependencies**

Add to `[project].dependencies` in `pyproject.toml`:

```toml
"scikit-image>=0.22",
```

- [ ] **Step 2: Write failing segmentation tests**

```python
# tests/test_segmentation.py
import numpy as np
import pytest

from joint.segmentation import region_table, segment_laser_marks


def two_disks_image():
    image = np.zeros((80, 100), dtype=float)
    rr, cc = np.ogrid[:80, :100]
    image[(rr - 25) ** 2 + (cc - 25) ** 2 <= 8**2] = 1.0
    image[(rr - 55) ** 2 + (cc - 70) ** 2 <= 9**2] = 1.0
    return image


def test_segment_laser_marks_finds_two_objects():
    result = segment_laser_marks(
        two_disks_image(),
        intensity_cutoff=0.1,
        closing_size=1,
        opening_size=1,
        min_peak_distance=10,
        min_size=30,
    )
    assert result.labels.dtype.kind in "iu"
    assert result.regions.shape[0] == 2
    assert set(result.regions.columns) >= {"label", "area", "centroid-0", "centroid-1"}


def test_segment_laser_marks_respects_boolean_mask():
    mask = np.ones((80, 100), dtype=bool)
    mask[:, 50:] = False
    result = segment_laser_marks(two_disks_image(), mask=mask, min_peak_distance=10, min_size=30)
    assert result.regions.shape[0] == 1


def test_region_table_is_sorted_by_label():
    labels = np.array([[0, 2, 2], [1, 0, 0]], dtype=np.int32)
    table = region_table(labels)
    assert table["label"].tolist() == [1, 2]


def test_segment_laser_marks_rejects_image_without_markers():
    with pytest.raises(SegmentationError, match="No laser markers"):
        segment_laser_marks(np.zeros((20, 20)), intensity_cutoff=0.1)
```

- [ ] **Step 3: Run tests and verify the module is absent**

Run: `python -m pytest tests/test_segmentation.py -v`  
Expected: collection fails with `ModuleNotFoundError: No module named 'joint.segmentation'`.

- [ ] **Step 4: Implement image validation and watershed segmentation**

```python
# src/joint/segmentation.py
from collections.abc import Sequence
from typing import Literal

import anndata
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import draw, feature, filters, measure, morphology, segmentation

from joint.errors import SegmentationError
from joint.models import SegmentationResult


def _validate_image(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=float)
    if data.ndim != 2:
        raise SegmentationError("Laser image must be two-dimensional grayscale data")
    if not np.isfinite(data).all():
        raise SegmentationError("Laser image contains non-finite values")
    return data.copy()


def region_table(labels: np.ndarray) -> pd.DataFrame:
    properties = measure.regionprops_table(
        labels.astype(np.int32, copy=False), properties=("label", "area", "centroid")
    )
    return pd.DataFrame(properties).sort_values("label").reset_index(drop=True)


def segment_laser_marks(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    intensity_cutoff: float = 0.2,
    top_hat_radius: int = 1,
    closing_size: int = 5,
    post_closing_size: int = 1,
    opening_size: int = 3,
    min_peak_distance: int = 25,
    marker_connectivity: int = 2,
    min_size: int = 30,
    threshold: Literal["otsu"] | float = "otsu",
) -> SegmentationResult:
    processed = _validate_image(image)
    if mask is not None:
        include = np.asarray(mask, dtype=bool)
        if include.shape != processed.shape:
            raise SegmentationError("mask and image must have identical shapes")
        processed[~include] = 0
    processed[processed < intensity_cutoff] = 0
    if top_hat_radius > 0:
        processed -= morphology.white_tophat(processed, morphology.disk(top_hat_radius))
    cutoff = filters.threshold_otsu(processed) if threshold == "otsu" else float(threshold)
    binary = processed > cutoff
    if closing_size > 1:
        binary = morphology.binary_closing(binary, morphology.square(closing_size))
    binary = segmentation.clear_border(binary)
    if post_closing_size > 1:
        binary = morphology.binary_closing(binary, morphology.square(post_closing_size))
    if opening_size > 1:
        binary = morphology.binary_opening(binary, morphology.square(opening_size))
    distance = ndi.distance_transform_edt(binary)
    coordinates = feature.peak_local_max(distance, min_distance=min_peak_distance, labels=binary)
    maxima = np.zeros_like(binary, dtype=bool)
    if not coordinates.size:
        raise SegmentationError("No laser markers were detected; inspect mask and thresholds")
    maxima[tuple(coordinates.T)] = True
    markers = measure.label(maxima, connectivity=marker_connectivity)
    labels = segmentation.watershed(-distance, markers, mask=binary, connectivity=2)
    labels = morphology.remove_small_objects(labels, min_size=min_size).astype(np.int32)
    regions = region_table(labels)
    diagnostics = {
        "threshold": float(cutoff),
        "object_count": int(regions.shape[0]),
        "foreground_pixels": int(np.count_nonzero(labels)),
        "processed_image": processed,
        "binary_mask": binary,
    }
    return SegmentationResult(labels=labels, regions=regions, diagnostics=diagnostics)
```

- [ ] **Step 5: Run the segmentation tests**

Run: `python -m pytest tests/test_segmentation.py -v`  
Expected: 4 tests pass. If the synthetic disks touch an image border after morphology, move their centers inward rather than weakening border clearing.

- [ ] **Step 6: Commit real segmentation**

```bash
git add pyproject.toml src/joint/segmentation.py tests/test_segmentation.py
git commit -m "feat: add laser mark segmentation"
```

### Task 2: Row Detection, Ordering, Exclusions, and Structured Merge Rules

**Files:**
- Modify: `src/joint/segmentation.py`
- Modify: `tests/test_segmentation.py`

**Interfaces:**
- Consumes: region tables containing `label`, `area`, `centroid-0`, `centroid-1`.
- Produces: `detect_rows`, `exclude_regions`, `load_merge_rules`, and `merge_regions`.

- [ ] **Step 1: Add failing row and merge tests**

```python
# append to tests/test_segmentation.py
from pathlib import Path

import pandas as pd

from joint.errors import SegmentationError
from joint.segmentation import detect_rows, exclude_regions, load_merge_rules, merge_regions


def sample_regions():
    return pd.DataFrame(
        {
            "label": [1, 2, 3, 4],
            "area": [10.0, 12.0, 5.0, 7.0],
            "centroid-0": [10.0, 10.5, 40.0, 40.5],
            "centroid-1": [10.0, 20.0, 10.0, 20.0],
        }
    )


def test_detect_rows_assigns_row_and_column_numbers():
    rows = detect_rows(sample_regions(), expected_rows=2, method="coordinate_gap")
    assert len(rows) == 2
    assert rows[0]["row_number"].unique().tolist() == [1]
    assert rows[0]["column_number"].tolist() == [1, 2]
    assert rows[0]["seg_label"].tolist() == ["1", "2"]


def test_detect_rows_rejects_wrong_expected_count():
    with pytest.raises(SegmentationError, match="expected 5"):
        detect_rows(sample_regions(), expected_rows=5, method="coordinate_gap")


def test_exclude_regions_uses_explicit_coordinate_bounds():
    kept = exclude_regions(sample_regions(), min_column=15)
    assert kept["label"].tolist() == [2, 4]


def test_merge_regions_combines_labels_and_marks_sources():
    rows = pd.concat(detect_rows(sample_regions(), expected_rows=2), ignore_index=True)
    rules = pd.DataFrame({"row_number": [2], "source_labels": ["3,4"]})
    merged = merge_regions(rows, rules)
    combined = merged.query('morphology == "combined"').iloc[0]
    assert combined["seg_label"] == "3_4"
    assert combined["area"] == 12.0
    assert merged.query('morphology != "spoilt" and row_number == 2').shape[0] == 1


def test_load_merge_rules_validates_columns(tmp_path: Path):
    path = tmp_path / "rules.csv"
    path.write_text("row_number,source_labels\n2,3|4\n")
    rules = load_merge_rules(path)
    assert rules.iloc[0]["source_labels"] == "3|4"
```

- [ ] **Step 2: Run the focused tests and verify missing functions**

Run: `python -m pytest tests/test_segmentation.py -k 'rows or exclude or merge' -v`  
Expected: import fails for the new functions.

- [ ] **Step 3: Implement deterministic row detection and exclusions**

```python
# append to src/joint/segmentation.py
from pathlib import Path


def _row_breaks_by_gap(regions: pd.DataFrame, expected_rows: int | None) -> np.ndarray:
    ordered = regions.sort_values(["centroid-0", "centroid-1"])
    gaps = np.diff(ordered["centroid-0"].to_numpy(dtype=float))
    if gaps.size == 0:
        return np.array([], dtype=int)
    if expected_rows is not None:
        count = expected_rows - 1
        if count < 0 or count > gaps.size:
            raise SegmentationError(f"Cannot create expected {expected_rows} rows")
        return np.sort(np.argsort(gaps)[-count:] + 1) if count else np.array([], dtype=int)
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
    coordinates = regions.sort_values(["centroid-0", "centroid-1"])["centroid-0"].to_numpy()
    variances = np.array(
        [np.var(coordinates[index - window_size : index]) for index in range(window_size, len(coordinates))]
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
    required = {"label", "area", "centroid-0", "centroid-1"}
    missing = required - set(regions.columns)
    if missing:
        raise SegmentationError(f"Region table is missing columns: {sorted(missing)}")
    ordered = regions.sort_values(["centroid-0", "centroid-1"]).reset_index(drop=True).copy()
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
    pieces = np.split(ordered, breaks)
    if expected_rows is not None and len(pieces) != expected_rows:
        raise SegmentationError(f"Detected {len(pieces)} rows; expected {expected_rows}")
    rows: list[pd.DataFrame] = []
    for row_number, piece in enumerate(pieces, start=1):
        row = piece.sort_values(["centroid-1", "centroid-0"]).copy()
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
    keep = np.ones(len(regions), dtype=bool)
    if min_row is not None:
        keep &= regions["centroid-0"].to_numpy() >= min_row
    if max_row is not None:
        keep &= regions["centroid-0"].to_numpy() <= max_row
    if min_column is not None:
        keep &= regions["centroid-1"].to_numpy() >= min_column
    if max_column is not None:
        keep &= regions["centroid-1"].to_numpy() <= max_column
    return regions.loc[keep].copy()
```

- [ ] **Step 4: Implement correction-file loading and multi-label merges**

```python
# append to src/joint/segmentation.py
def load_merge_rules(path: str | Path) -> pd.DataFrame:
    source = Path(path).resolve()
    rules = pd.read_csv(source, dtype={"source_labels": str})
    required = {"row_number", "source_labels"}
    missing = required - set(rules.columns)
    if missing:
        raise SegmentationError(f"Merge rules are missing columns: {sorted(missing)}")
    return rules


def _parse_source_labels(value: str) -> list[str]:
    normalized = value.replace(",", "|").replace("_", "|")
    return [item.strip() for item in normalized.split("|") if item.strip()]


def merge_regions(regions: pd.DataFrame, merge_rules: pd.DataFrame) -> pd.DataFrame:
    result = regions.copy()
    for rule in merge_rules.itertuples(index=False):
        labels = _parse_source_labels(str(rule.source_labels))
        selected = result[
            (result["row_number"] == int(rule.row_number)) & result["seg_label"].isin(labels)
        ]
        if selected.shape[0] != len(labels):
            raise SegmentationError(
                f"Merge rule row {rule.row_number} expected labels {labels}, found {selected['seg_label'].tolist()}"
            )
        result.loc[selected.index, "morphology"] = "spoilt"
        first = selected.iloc[0].copy()
        first["centroid-0"] = selected["centroid-0"].mean()
        first["centroid-1"] = selected["centroid-1"].mean()
        first["area"] = selected["area"].sum()
        first["seg_label"] = "_".join(labels)
        first["morphology"] = "combined"
        result.loc[len(result)] = first
    return result.sort_values(["row_number", "centroid-1", "centroid-0"]).reset_index(drop=True)
```

- [ ] **Step 5: Run all segmentation tests**

Run: `python -m pytest tests/test_segmentation.py -v`  
Expected: all segmentation, row, exclusion, and merge tests pass.

- [ ] **Step 6: Commit row organization and corrections**

```bash
git add src/joint/segmentation.py tests/test_segmentation.py
git commit -m "feat: add laser row organization and corrections"
```

### Task 3: Simulated Laser Marks

**Files:**
- Modify: `src/joint/segmentation.py`
- Modify: `tests/test_segmentation.py`

**Interfaces:**
- Consumes: MSI AnnData with `obsm["spatial"]`, anchor-row DataFrames, and image shape.
- Produces: `simulate_laser_marks(adata, anchor_rows, image_shape, *, radius) -> SegmentationResult`.

- [ ] **Step 1: Add failing simulated-mark tests**

```python
# append to tests/test_segmentation.py
import anndata

from joint.segmentation import simulate_laser_marks


def test_simulate_laser_marks_matches_msi_grid_and_clips_disks():
    adata = anndata.AnnData(
        np.ones((6, 1)),
        obsm={"spatial": np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])},
    )
    anchors = [
        pd.DataFrame({"centroid-0": [2.0, 2.0], "centroid-1": [1.0, 8.0]}),
        pd.DataFrame({"centroid-0": [8.0, 8.0], "centroid-1": [1.0, 8.0]}),
    ]
    result = simulate_laser_marks(adata, anchors, (10, 10), radius=2)
    assert result.regions.shape[0] == 6
    assert result.labels.max() == 6
    assert result.labels.shape == (10, 10)
    assert result.regions["row_number"].value_counts().sort_index().tolist() == [3, 3]
```

- [ ] **Step 2: Run the test and verify the function is absent**

Run: `python -m pytest tests/test_segmentation.py::test_simulate_laser_marks_matches_msi_grid_and_clips_disks -v`  
Expected: import fails for `simulate_laser_marks`.

- [ ] **Step 3: Implement simulation with explicit shape clipping**

```python
# append to src/joint/segmentation.py
def simulate_laser_marks(
    adata: anndata.AnnData,
    anchor_rows: Sequence[pd.DataFrame],
    image_shape: tuple[int, int],
    *,
    radius: int,
) -> SegmentationResult:
    if "spatial" not in adata.obsm:
        raise SegmentationError('adata.obsm must contain "spatial"')
    spatial = np.asarray(adata.obsm["spatial"])
    points_per_row = np.unique(spatial[:, 0]).size
    row_count = np.unique(spatial[:, 1]).size
    if row_count != len(anchor_rows):
        raise SegmentationError(
            f"MSI contains {row_count} rows but {len(anchor_rows)} anchor rows were provided"
        )
    labels = np.zeros(image_shape, dtype=np.int32)
    records: list[dict[str, float | int | str]] = []
    label = 1
    for row_number, anchors in enumerate(anchor_rows, start=1):
        ordered = anchors.sort_values("centroid-1")
        start = ordered.iloc[0][["centroid-0", "centroid-1"]].to_numpy(dtype=float)
        end = ordered.iloc[-1][["centroid-0", "centroid-1"]].to_numpy(dtype=float)
        centers = np.linspace(start, end, points_per_row)
        for column_number, (row, column) in enumerate(centers, start=1):
            rr, cc = draw.disk((row, column), radius, shape=image_shape)
            labels[rr, cc] = label
            records.append(
                {
                    "label": label,
                    "area": float(len(rr)),
                    "centroid-0": float(row),
                    "centroid-1": float(column),
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
```

- [ ] **Step 4: Run segmentation tests**

Run: `python -m pytest tests/test_segmentation.py -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit laser simulation**

```bash
git add src/joint/segmentation.py tests/test_segmentation.py
git commit -m "feat: add simulated laser marks"
```

### Task 4: Explicit Orientation and MSI-to-Laser Registration

**Files:**
- Modify: `src/joint/config.py`
- Create: `src/joint/registration.py`
- Create: `tests/test_registration.py`

**Interfaces:**
- Consumes: MSI AnnData and a tidy laser-region table with `row_number` and `column_number`.
- Produces: `register_laser_points(..., orientation, edge_policy) -> RegistrationResult` and `mount_spatial_images`.

- [ ] **Step 1: Complete the orientation schema**

Replace `RegistrationConfig.orientation` with:

```python
orientation: Literal[
    "auto",
    "identity",
    "flip_x",
    "flip_y",
    "flip_xy",
    "swap_xy",
    "swap_flip_x",
    "swap_flip_y",
    "swap_flip_xy",
] = "auto"
edge_policy: Literal["error", "truncate_left", "truncate_right"] = "error"
```

- [ ] **Step 2: Write failing registration tests**

```python
# tests/test_registration.py
import anndata
import numpy as np
import pandas as pd
import pytest

from joint.errors import RegistrationError
from joint.registration import mount_spatial_images, register_laser_points


def grid_adata(columns=3):
    coords = np.array([(x, y) for y in (1, 2) for x in range(1, columns + 1)], dtype=float)
    adata = anndata.AnnData(np.arange(len(coords) * 2).reshape(len(coords), 2), obsm={"spatial": coords})
    adata.obs_names = [f"p{i + 1}" for i in range(len(coords))]
    adata.obs["dataset"] = "sample"
    return adata


def laser_regions(columns=3):
    records = []
    label = 1
    for row in (1, 2):
        for column in range(1, columns + 1):
            records.append(
                {
                    "label": label,
                    "seg_label": str(label),
                    "area": 20,
                    "centroid-0": row * 10,
                    "centroid-1": column * 10,
                    "row_number": row,
                    "column_number": column,
                    "morphology": "intact",
                }
            )
            label += 1
    return pd.DataFrame(records)


def test_identity_registration_maps_grid_and_replaces_spatial_coordinates():
    result = register_laser_points(grid_adata(), laser_regions(), orientation="identity")
    assert result.mapping[["pixel_id", "seg_label"]].values.tolist()[0] == ["p1", "1"]
    np.testing.assert_array_equal(result.adata.obsm["spatial"][0], [10, 10])
    assert result.transform["orientation"] == "identity"


def test_truncate_left_is_recorded_not_silent():
    result = register_laser_points(
        grid_adata(columns=4), laser_regions(columns=3), orientation="identity", edge_policy="truncate_left"
    )
    assert result.adata.n_obs == 6
    assert result.report["truncated_pixels"] == 2


def test_auto_orientation_raises_when_multiple_candidates_fit():
    with pytest.raises(RegistrationError, match="ambiguous"):
        register_laser_points(grid_adata(), laser_regions(), orientation="auto")


def test_registration_rejects_mixed_datasets():
    adata = grid_adata()
    adata.obs["dataset"] = ["a", "a", "a", "b", "b", "b"]
    with pytest.raises(RegistrationError, match="single dataset"):
        register_laser_points(adata, laser_regions(), orientation="identity")


def test_mount_spatial_images_returns_copy():
    adata = grid_adata()
    laser = np.zeros((20, 20, 3), dtype=np.uint8)
    mounted = mount_spatial_images(adata, laser_image=laser, library_id="sample")
    assert "spatial" not in adata.uns
    assert mounted.uns["spatial"]["sample"]["images"]["laser"].shape == laser.shape
```

- [ ] **Step 3: Run tests and verify registration is absent**

Run: `python -m pytest tests/test_registration.py -v`  
Expected: collection fails because `joint.registration` is missing.

- [ ] **Step 4: Implement transforms, row pairing, explicit truncation, and auto ambiguity**

```python
# src/joint/registration.py
from typing import Literal

import anndata
import numpy as np
import pandas as pd

from joint.errors import RegistrationError
from joint.models import RegistrationResult


ORIENTATIONS = (
    "identity", "flip_x", "flip_y", "flip_xy", "swap_xy", "swap_flip_x",
    "swap_flip_y", "swap_flip_xy",
)


def _orient(spatial: np.ndarray, orientation: str) -> np.ndarray:
    result = np.asarray(spatial, dtype=float).copy()
    if orientation.startswith("swap"):
        result = result[:, [1, 0]]
    if "flip_x" in orientation or orientation == "flip_xy" or orientation == "swap_flip_xy":
        result[:, 0] = result[:, 0].min() + result[:, 0].max() - result[:, 0]
    if "flip_y" in orientation or orientation == "flip_xy" or orientation == "swap_flip_xy":
        result[:, 1] = result[:, 1].min() + result[:, 1].max() - result[:, 1]
    return result


def _mapping_for_orientation(
    adata: anndata.AnnData,
    regions: pd.DataFrame,
    *,
    orientation: str,
    edge_policy: str,
) -> tuple[pd.DataFrame, int]:
    oriented = _orient(np.asarray(adata.obsm["spatial"]), orientation)
    pixels = pd.DataFrame({"pixel_id": adata.obs_names, "x": oriented[:, 0], "y": oriented[:, 1]})
    pixel_rows = [part.sort_values("x") for _, part in pixels.groupby("y", sort=True)]
    laser_rows = [part.sort_values("column_number") for _, part in regions.groupby("row_number", sort=True)]
    if len(pixel_rows) != len(laser_rows):
        raise RegistrationError(
            f"Orientation {orientation} gives {len(pixel_rows)} MSI rows and {len(laser_rows)} laser rows"
        )
    records: list[pd.DataFrame] = []
    truncated = 0
    for pixel_row, laser_row in zip(pixel_rows, laser_rows, strict=True):
        difference = len(pixel_row) - len(laser_row)
        if difference < 0:
            raise RegistrationError("Laser row contains more points than its MSI row")
        if difference and edge_policy == "error":
            raise RegistrationError("MSI and laser row lengths differ; choose an explicit edge_policy")
        if difference:
            truncated += difference
            pixel_row = pixel_row.iloc[difference:] if edge_policy == "truncate_left" else pixel_row.iloc[:-difference]
        paired = laser_row.copy().reset_index(drop=True)
        paired.insert(0, "pixel_id", pixel_row["pixel_id"].to_numpy())
        records.append(paired)
    return pd.concat(records, ignore_index=True), truncated


def register_laser_points(
    msi_adata: anndata.AnnData,
    laser_regions: pd.DataFrame,
    *,
    orientation: Literal[
        "auto", "identity", "flip_x", "flip_y", "flip_xy", "swap_xy",
        "swap_flip_x", "swap_flip_y", "swap_flip_xy",
    ] = "auto",
    edge_policy: Literal["error", "truncate_left", "truncate_right"] = "error",
) -> RegistrationResult:
    if "spatial" not in msi_adata.obsm:
        raise RegistrationError('MSI AnnData is missing obsm["spatial"]')
    if "dataset" in msi_adata.obs and msi_adata.obs["dataset"].nunique() != 1:
        raise RegistrationError("Registration requires AnnData from a single dataset")
    required = {"seg_label", "area", "centroid-0", "centroid-1", "row_number", "column_number"}
    missing = required - set(laser_regions.columns)
    if missing:
        raise RegistrationError(f"Laser regions are missing columns: {sorted(missing)}")
    if orientation == "auto":
        candidates: list[tuple[str, pd.DataFrame, int]] = []
        for candidate in ORIENTATIONS:
            try:
                mapping, truncated = _mapping_for_orientation(
                    msi_adata, laser_regions, orientation=candidate, edge_policy=edge_policy
                )
                candidates.append((candidate, mapping, truncated))
            except RegistrationError:
                continue
        if len(candidates) != 1:
            names = [item[0] for item in candidates]
            raise RegistrationError(f"Automatic orientation is ambiguous; valid candidates: {names}")
        orientation, mapping, truncated = candidates[0]
    else:
        mapping, truncated = _mapping_for_orientation(
            msi_adata, laser_regions, orientation=orientation, edge_policy=edge_policy
        )
    registered = msi_adata[mapping["pixel_id"].to_numpy()].copy()
    metadata = mapping.set_index("pixel_id").drop(columns=["label"], errors="ignore")
    registered.obs = registered.obs.join(metadata)
    registered.obsm["spatial"] = registered.obs[["centroid-1", "centroid-0"]].to_numpy(dtype=float)
    transform = {"orientation": orientation, "edge_policy": edge_policy}
    registered.uns.setdefault("joint", {})["registration"] = transform
    return RegistrationResult(
        adata=registered,
        mapping=mapping,
        transform=transform,
        report={"registered_pixels": registered.n_obs, "truncated_pixels": truncated},
    )


def mount_spatial_images(
    adata: anndata.AnnData,
    *,
    laser_image: np.ndarray | None = None,
    cell_image: np.ndarray | None = None,
    library_id: str = "sample",
    spot_diameter: float = 1.0,
) -> anndata.AnnData:
    result = adata.copy()
    images: dict[str, np.ndarray] = {}
    if laser_image is not None:
        images["laser"] = np.asarray(laser_image)
    if cell_image is not None:
        images["hires"] = np.asarray(cell_image)
    if not images:
        return result
    shapes = {image.shape[:2] for image in images.values()}
    if len(shapes) > 1:
        raise RegistrationError("Mounted images must have identical height and width")
    result.uns.setdefault("spatial", {})[library_id] = {
        "images": images,
        "scalefactors": {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": spot_diameter},
    }
    return result
```

- [ ] **Step 5: Run configuration and registration tests**

Run: `python -m pytest tests/test_config.py tests/test_registration.py -v`  
Expected: all tests pass.

- [ ] **Step 6: Commit registration**

```bash
git add src/joint/config.py src/joint/registration.py tests/test_registration.py
git commit -m "feat: add explicit laser registration"
```

### Task 5: Shared Cell-Laser Overlap Table

**Files:**
- Create: `src/joint/quantification.py`
- Create: `tests/test_quantification.py`

**Interfaces:**
- Consumes: laser AnnData, integer cell/laser label arrays.
- Produces: `build_overlap_table(...) -> tuple[DataFrame, DataFrame]` where the second table contains cells inside the analysis boundary.

- [ ] **Step 1: Write failing overlap tests**

```python
# tests/test_quantification.py
import anndata
import numpy as np
from scipy.sparse import csr_matrix

from joint.quantification import build_overlap_table


def laser_adata():
    adata = anndata.AnnData(
        csr_matrix([[10.0, 20.0], [30.0, 40.0]]),
        obs={
            "seg_label": ["1", "2"],
            "area": [4.0, 4.0],
            "centroid-0": [1.0, 1.0],
            "centroid-1": [1.0, 3.0],
        },
        var={"mz": [100.0, 200.0]},
    )
    adata.obs_names = ["p1", "p2"]
    return adata


def label_arrays():
    laser = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0]], dtype=np.int32)
    cells = np.array([[1, 1, 1, 2], [1, 1, 2, 2], [0, 0, 0, 0]], dtype=np.int32)
    return cells, laser


def test_build_overlap_table_records_area_and_ratios():
    cells, laser = label_arrays()
    overlaps, cell_table = build_overlap_table(laser_adata(), cells, laser, boundary_margin=10)
    first = overlaps.query("cell_id == 'c1' and laser_label == 1").iloc[0]
    assert first["occupied_pixels"] == 4
    assert first["laser_area"] == 4
    assert first["occupied_ratio"] == 1.0
    assert set(cell_table["cell_id"]) == {"c1", "c2"}


def test_combined_seg_labels_map_each_constituent_to_one_observation():
    adata = laser_adata()[["p1"]].copy()
    adata.obs["seg_label"] = "1_2"
    cells, laser = label_arrays()
    overlaps, _ = build_overlap_table(adata, cells, laser, boundary_margin=10)
    assert set(overlaps["laser_obs_id"]) == {"p1"}
```

- [ ] **Step 2: Run tests and verify quantification is absent**

Run: `python -m pytest tests/test_quantification.py -v`  
Expected: collection fails because `joint.quantification` is missing.

- [ ] **Step 3: Implement input validation and overlap construction**

```python
# src/joint/quantification.py
from typing import Literal

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from skimage import measure

from joint.errors import QuantificationError
from joint.models import QuantificationResult


def _validate_labels(cell_labels: np.ndarray, laser_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cell_labels)
    lasers = np.asarray(laser_labels)
    if cells.ndim != 2 or lasers.ndim != 2 or cells.shape != lasers.shape:
        raise QuantificationError("Cell and laser segmentations must be two-dimensional with equal shapes")
    if cells.dtype.kind not in "iu" or lasers.dtype.kind not in "iu":
        raise QuantificationError("Segmentation labels must use integer dtypes")
    return cells, lasers


def _laser_connections(laser_adata: anndata.AnnData) -> dict[int, str]:
    required = {"seg_label", "area", "centroid-0", "centroid-1"}
    missing = required - set(laser_adata.obs.columns)
    if missing:
        raise QuantificationError(f"Laser AnnData is missing obs columns: {sorted(missing)}")
    connections: dict[int, str] = {}
    for obs_id, value in laser_adata.obs["seg_label"].items():
        for label in str(value).split("_"):
            label_id = int(label)
            if label_id in connections:
                raise QuantificationError(f"Laser label {label_id} maps to more than one observation")
            connections[label_id] = str(obs_id)
    return connections


def build_overlap_table(
    laser_adata: anndata.AnnData,
    cell_segmentation: np.ndarray,
    laser_segmentation: np.ndarray,
    *,
    boundary_margin: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells, lasers = _validate_labels(cell_segmentation, laser_segmentation)
    connections = _laser_connections(laser_adata)
    cell_table = pd.DataFrame(
        measure.regionprops_table(cells, properties=("label", "area", "centroid"))
    )
    min_row = laser_adata.obs["centroid-0"].min() - boundary_margin
    max_row = laser_adata.obs["centroid-0"].max() + boundary_margin
    min_col = laser_adata.obs["centroid-1"].min() - boundary_margin
    max_col = laser_adata.obs["centroid-1"].max() + boundary_margin
    cell_table = cell_table[
        cell_table["centroid-0"].between(min_row, max_row, inclusive="neither")
        & cell_table["centroid-1"].between(min_col, max_col, inclusive="neither")
    ].copy()
    cell_table["cell_id"] = "c" + cell_table["label"].astype(int).astype(str)
    laser_areas = np.bincount(lasers.ravel())
    records: list[dict[str, object]] = []
    for row in cell_table.itertuples(index=False):
        covered = lasers[cells == row.label]
        labels, counts = np.unique(covered[covered > 0], return_counts=True)
        for label, count in zip(labels, counts, strict=True):
            label_id = int(label)
            if label_id not in connections:
                continue
            area = int(laser_areas[label_id])
            records.append(
                {
                    "cell_id": row.cell_id,
                    "cell_label": int(row.label),
                    "laser_label": label_id,
                    "laser_obs_id": connections[label_id],
                    "occupied_pixels": int(count),
                    "laser_area": area,
                    "occupied_ratio": float(count / area),
                }
            )
    columns = [
        "cell_id", "cell_label", "laser_label", "laser_obs_id", "occupied_pixels",
        "laser_area", "occupied_ratio",
    ]
    return pd.DataFrame.from_records(records, columns=columns), cell_table.reset_index(drop=True)
```

- [ ] **Step 4: Run overlap tests**

Run: `python -m pytest tests/test_quantification.py -v`  
Expected: 2 tests pass.

- [ ] **Step 5: Commit overlap construction**

```bash
git add src/joint/quantification.py tests/test_quantification.py
git commit -m "feat: add cell laser overlap tables"
```

### Task 6: Both Quantification Strategies

**Files:**
- Modify: `src/joint/quantification.py`
- Modify: `tests/test_quantification.py`

**Interfaces:**
- Consumes: shared overlap table from Task 5.
- Produces: `select_specificity_filtered`, `quantify_cells`, and numeric behavior for both approved methods.

- [ ] **Step 1: Add failing numeric and filtering tests**

```python
# append to tests/test_quantification.py
from joint.quantification import quantify_cells, select_specificity_filtered


def test_specificity_filter_keeps_unique_and_dominant_mappings_only():
    overlaps = pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c2", "c3"],
            "laser_label": [1, 2, 2, 3],
            "occupied_ratio": [0.2, 0.7, 0.1, 0.05],
        }
    )
    accepted, rejected = select_specificity_filtered(
        overlaps,
        unique_min_overlap=0.1,
        dominant_min_overlap=0.6,
        secondary_max_overlap=0.15,
    )
    assert accepted[["cell_id", "laser_label"]].values.tolist() == [["c1", 1], ["c1", 2]]
    assert set(rejected["cell_id"]) == {"c2", "c3"}


def test_legacy_proportional_matches_d2_formula():
    cells, laser = label_arrays()
    result = quantify_cells(
        laser_adata(), cells, laser, method="legacy_proportional", boundary_margin=10
    )
    np.testing.assert_allclose(result.adata["c1"].X.toarray(), [[17.5, 30.0]])


def test_specificity_filtered_matches_corrected_d8_formula():
    cells, laser = label_arrays()
    result = quantify_cells(
        laser_adata(),
        cells,
        laser,
        method="specificity_filtered",
        boundary_margin=10,
        unique_min_overlap=0.0,
        dominant_min_overlap=0.6,
        secondary_max_overlap=0.3,
    )
    assert result.adata.shape == (2, 2)
    assert set(result.accepted["cell_id"]) == {"c1", "c2"}
    assert result.adata.obs["quantification_status"].isin(["quantified", "zero"]).all()
```

Before running, add `import pandas as pd` to `tests/test_quantification.py`.

- [ ] **Step 2: Run focused tests and verify missing functions**

Run: `python -m pytest tests/test_quantification.py -k 'specificity or legacy' -v`  
Expected: import fails for the new functions.

- [ ] **Step 3: Implement d8 mapping selection with rejection reasons**

```python
# append to src/joint/quantification.py
def select_specificity_filtered(
    overlaps: pd.DataFrame,
    *,
    unique_min_overlap: float,
    dominant_min_overlap: float,
    secondary_max_overlap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted: list[pd.DataFrame] = []
    rejected: list[pd.DataFrame] = []
    for _, group in overlaps.groupby("laser_label", sort=True):
        ordered = group.sort_values("occupied_ratio", ascending=False).copy()
        if len(ordered) == 1:
            if ordered.iloc[0]["occupied_ratio"] > unique_min_overlap:
                accepted.append(ordered)
            else:
                ordered["rejection_reason"] = "unique_overlap_below_threshold"
                rejected.append(ordered)
            continue
        dominant = ordered.iloc[[0]].copy()
        secondary = ordered.iloc[1:].copy()
        if (
            dominant.iloc[0]["occupied_ratio"] > dominant_min_overlap
            and secondary.iloc[0]["occupied_ratio"] < secondary_max_overlap
        ):
            accepted.append(dominant)
            secondary["rejection_reason"] = "non_dominant_mapping"
            rejected.append(secondary)
        else:
            ordered["rejection_reason"] = "ambiguous_multi_mapping"
            rejected.append(ordered)
    accepted_df = pd.concat(accepted, ignore_index=True) if accepted else overlaps.iloc[0:0].copy()
    rejected_df = pd.concat(rejected, ignore_index=True) if rejected else overlaps.iloc[0:0].copy()
    return accepted_df, rejected_df
```

- [ ] **Step 4: Implement abundance calculation and cell AnnData construction**

```python
# append to src/joint/quantification.py
def _spectrum(adata: anndata.AnnData, obs_id: str) -> np.ndarray:
    matrix = adata[[obs_id]].X
    return matrix.toarray().ravel() if sparse.issparse(matrix) else np.asarray(matrix).ravel()


def _quantify_legacy(laser_adata: anndata.AnnData, accepted: pd.DataFrame, cell_id: str) -> np.ndarray:
    rows = accepted[accepted["cell_id"] == cell_id]
    expression = np.zeros(laser_adata.n_vars, dtype=float)
    for row in rows.itertuples(index=False):
        expression += _spectrum(laser_adata, row.laser_obs_id) * row.occupied_ratio
    return expression


def _quantify_specificity(laser_adata: anndata.AnnData, accepted: pd.DataFrame, cell_id: str) -> np.ndarray:
    rows = accepted[accepted["cell_id"] == cell_id]
    if rows.empty:
        return np.zeros(laser_adata.n_vars, dtype=float)
    contributions: list[np.ndarray] = []
    specificities: list[float] = []
    for row in rows.itertuples(index=False):
        total_occupied = accepted.loc[
            accepted["laser_label"] == row.laser_label, "occupied_pixels"
        ].sum()
        specificity = float(row.occupied_pixels / total_occupied)
        specificities.append(specificity)
        contributions.append(
            _spectrum(laser_adata, row.laser_obs_id) * specificity / row.occupied_ratio
        )
    return np.sum(contributions, axis=0) / np.sum(specificities)


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
    overlaps, cells = build_overlap_table(
        laser_adata, cell_segmentation, laser_segmentation, boundary_margin=boundary_margin
    )
    if method == "specificity_filtered":
        accepted, rejected = select_specificity_filtered(
            overlaps,
            unique_min_overlap=unique_min_overlap,
            dominant_min_overlap=dominant_min_overlap,
            secondary_max_overlap=secondary_max_overlap,
        )
        calculate = _quantify_specificity
    elif method == "legacy_proportional":
        accepted = overlaps.copy()
        rejected = overlaps.iloc[0:0].copy()
        calculate = _quantify_legacy
    else:
        raise QuantificationError(f"Unsupported quantification method: {method}")
    cell_ids = cells["cell_id"].tolist()
    matrix = np.vstack([calculate(laser_adata, accepted, cell_id) for cell_id in cell_ids])
    obs = cells.set_index("cell_id").copy()
    obs["quantification_status"] = [
        "quantified" if cell_id in set(accepted["cell_id"]) else "zero" for cell_id in cell_ids
    ]
    for cell_id in cell_ids:
        mapped = accepted[accepted["cell_id"] == cell_id]
        obs.loc[cell_id, "laser_labels"] = "_".join(mapped["laser_label"].astype(str)) or "0"
        obs.loc[cell_id, "laser_coverage"] = "_".join(mapped["occupied_pixels"].astype(str)) or "0"
    cell_adata = anndata.AnnData(sparse.csr_matrix(matrix), obs=obs, var=laser_adata.var.copy())
    cell_adata.obsm["spatial"] = obs[["centroid-1", "centroid-0"]].to_numpy(dtype=float)
    cell_adata.uns["joint"] = {"quantification_method": method}
    report = {
        "included_cells": len(cell_ids),
        "quantified_cells": int(obs["quantification_status"].eq("quantified").sum()),
        "accepted_mappings": len(accepted),
        "rejected_mappings": len(rejected),
    }
    return QuantificationResult(cell_adata, overlaps, accepted, rejected, report)
```

- [ ] **Step 5: Verify the expected legacy number and adjust only the test arithmetic if necessary**

Manually derive cell `c1`: laser 1 contributes `[10, 20] * 1.0`; laser 2 contributes `[30, 40] * 0.25`; expected `[17.5, 30.0]`. The implementation must not be changed to fit an incorrectly derived test.

Run: `python -m pytest tests/test_quantification.py -v`  
Expected: all quantification tests pass.

- [ ] **Step 6: Commit both strategies**

```bash
git add src/joint/quantification.py tests/test_quantification.py
git commit -m "feat: add JOINT cell quantification strategies"
```

### Task 7: Mixed AnnData and Spatial Core Integration

**Files:**
- Modify: `src/joint/quantification.py`
- Create: `tests/test_spatial_integration.py`
- Modify: `src/joint/__init__.py`

**Interfaces:**
- Consumes: laser AnnData, `QuantificationResult`, and accepted overlaps.
- Produces: `build_mixed_anndata` and an end-to-end synthetic spatial core.

- [ ] **Step 1: Write failing mixed-output and integration tests**

```python
# tests/test_spatial_integration.py
import anndata
import numpy as np
import pandas as pd

from joint.quantification import build_mixed_anndata, quantify_cells
from joint.registration import register_laser_points
from joint.segmentation import simulate_laser_marks


def test_spatial_core_runs_registration_quantification_and_mixed_output():
    msi = anndata.AnnData(
        np.array([[10.0], [20.0]]),
        obs={"dataset": ["sample", "sample"]},
        var={"mz": [100.0]},
        obsm={"spatial": np.array([[1.0, 1.0], [2.0, 1.0]])},
    )
    msi.obs_names = ["p1", "p2"]
    anchors = [pd.DataFrame({"centroid-0": [5.0, 5.0], "centroid-1": [4.0, 12.0]})]
    segmented = simulate_laser_marks(msi, anchors, (20, 20), radius=2)
    registered = register_laser_points(msi, segmented.regions, orientation="identity")
    cells = np.zeros((20, 20), dtype=np.int32)
    cells[:, :9] = 1
    cells[:, 9:] = 2
    quantified = quantify_cells(
        registered.adata,
        cells,
        segmented.labels,
        method="legacy_proportional",
        boundary_margin=20,
    )
    mixed = build_mixed_anndata(registered.adata, quantified)

    assert quantified.adata.n_obs == 2
    assert mixed.obs["data_type"].isin(["cell", "laser_residual"]).all()
    assert mixed.n_vars == 1
```

- [ ] **Step 2: Run the integration test and verify `build_mixed_anndata` is absent**

Run: `python -m pytest tests/test_spatial_integration.py -v`  
Expected: import fails for `build_mixed_anndata`.

- [ ] **Step 3: Implement residual-laser construction**

```python
# append to src/joint/quantification.py
def build_mixed_anndata(
    laser_adata: anndata.AnnData,
    quantification: QuantificationResult,
    *,
    min_residual_fraction: float = 0.2,
) -> anndata.AnnData:
    cells = quantification.adata.copy()
    cells.obs["data_type"] = "cell"
    residual_rows: list[np.ndarray] = []
    residual_obs: list[dict[str, object]] = []
    for obs_id, laser_row in laser_adata.obs.iterrows():
        labels = [int(value) for value in str(laser_row["seg_label"]).split("_")]
        occupied = quantification.accepted[
            quantification.accepted["laser_label"].isin(labels)
        ]["occupied_ratio"].sum()
        residual = max(0.0, 1.0 - float(occupied))
        if residual < min_residual_fraction:
            continue
        residual_rows.append(_spectrum(laser_adata, str(obs_id)) * residual)
        residual_obs.append(
            {
                "obs_id": f"laser_residual_{obs_id}",
                "data_type": "laser_residual",
                "seg_label": str(laser_row["seg_label"]),
                "residual_fraction": residual,
                "centroid-0": laser_row["centroid-0"],
                "centroid-1": laser_row["centroid-1"],
            }
        )
    if not residual_rows:
        return cells
    residual_meta = pd.DataFrame(residual_obs).set_index("obs_id")
    residual_adata = anndata.AnnData(
        sparse.csr_matrix(np.vstack(residual_rows)), obs=residual_meta, var=laser_adata.var.copy()
    )
    residual_adata.obsm["spatial"] = residual_meta[["centroid-1", "centroid-0"]].to_numpy()
    return anndata.concat([cells, residual_adata], join="outer", merge="same", index_unique=None)
```

- [ ] **Step 4: Export stable spatial APIs**

Add to `src/joint/__init__.py`:

```python
from joint.quantification import build_mixed_anndata, quantify_cells
from joint.registration import register_laser_points
from joint.segmentation import segment_laser_marks, simulate_laser_marks
```

Add the five names to `__all__`.

- [ ] **Step 5: Run the full spatial core suite**

Run: `python -m pytest tests/test_segmentation.py tests/test_registration.py tests/test_quantification.py tests/test_spatial_integration.py -v`  
Expected: all spatial tests pass.

Run: `ruff check src tests`  
Expected: `All checks passed!`.

- [ ] **Step 6: Commit the spatial milestone**

```bash
git add src/joint/quantification.py src/joint/__init__.py tests/test_spatial_integration.py
git commit -m "feat: complete JOINT spatial core"
```

## Plan 2 Completion Gate

Run:

```bash
python -m pytest -v
ruff check src tests
python -c "from joint import segment_laser_marks, simulate_laser_marks, register_laser_points, quantify_cells"
```

The plan is complete when the synthetic integration test passes, both quantification methods have numeric tests, ambiguous auto-orientation raises with candidates, and no d2/d8-specific label appears in library code.
