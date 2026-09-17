from copy import deepcopy
from inspect import signature

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse import csr_matrix

import joint.quantification as quantification_module
from joint.errors import QuantificationError
from joint.io import atomic_write_h5ad, read_h5ad
from joint.quantification import (
    build_overlap_table,
    quantify_cells,
    select_specificity_filtered,
)

OVERLAP_COLUMNS = [
    "cell_id",
    "cell_label",
    "laser_label",
    "laser_obs_id",
    "occupied_pixels",
    "laser_area",
    "occupied_ratio",
]
CELL_COLUMNS = ["label", "area", "centroid-0", "centroid-1", "cell_id"]


def laser_adata() -> anndata.AnnData:
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


def label_arrays() -> tuple[np.ndarray, np.ndarray]:
    laser = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0]], dtype=np.int32)
    cells = np.array([[1, 1, 1, 2], [1, 1, 2, 2], [0, 0, 0, 0]], dtype=np.int32)
    return cells, laser


def reference_grouped_raster_values(
    cells: np.ndarray, lasers: np.ndarray
) -> tuple[list[list[object]], list[list[object]]]:
    cell_rows: list[list[object]] = []
    overlap_rows: list[list[object]] = []
    laser_labels, laser_counts = np.unique(lasers, return_counts=True)
    laser_areas = {
        int(label): int(count)
        for label, count in zip(laser_labels, laser_counts, strict=True)
        if int(label) > 0
    }
    for label in np.unique(cells):
        cell_label = int(label)
        if cell_label == 0:
            continue
        rows, columns = np.nonzero(cells == label)
        cell_rows.append(
            [
                cell_label,
                float(rows.size),
                float(rows.mean()),
                float(columns.mean()),
                f"c{cell_label}",
            ]
        )
        covered_labels, occupied = np.unique(lasers[cells == label], return_counts=True)
        for laser_label, count in zip(covered_labels, occupied, strict=True):
            laser_id = int(laser_label)
            if laser_id > 0:
                overlap_rows.append([cell_label, laser_id, int(count), laser_areas[laser_id]])
    return cell_rows, overlap_rows


def selector_overlaps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c2", "c3"],
            "laser_label": [1, 2, 2, 3],
            "occupied_ratio": [0.2, 0.7, 0.1, 0.05],
        }
    )


def test_build_overlap_table_records_exact_raster_area_and_ratios() -> None:
    cells, laser = label_arrays()

    overlaps, cell_table = build_overlap_table(laser_adata(), cells, laser, boundary_margin=10)

    first = overlaps.query("cell_id == 'c1' and laser_label == 1").iloc[0]
    assert first["occupied_pixels"] == 4
    assert first["laser_area"] == 4
    assert first["occupied_ratio"] == 1.0
    assert set(cell_table["cell_id"]) == {"c1", "c2"}
    assert str(cell_table["cell_id"].dtype) == "string"


@pytest.mark.parametrize("seed", range(5))
def test_grouped_raster_aggregation_matches_small_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cells = rng.choice(np.array([0, 1, 2, 17], dtype=np.uint64), size=(9, 11))
    lasers = rng.choice(np.array([0, 1, 2, 3, 99], dtype=np.uint64), size=(9, 11))
    adata = anndata.AnnData(
        np.ones((3, 1)),
        obs={
            "seg_label": ["1", "2", "3"],
            "area": [1.0, 1.0, 1.0],
            "centroid-0": [0.0, 4.0, 8.0],
            "centroid-1": [0.0, 5.0, 10.0],
        },
        var={"mz": [100.0]},
    )
    adata.obs_names = ["p1", "p2", "p3"]
    expected_cells, expected_overlaps = reference_grouped_raster_values(cells, lasers)

    overlaps, cell_table = build_overlap_table(adata, cells, lasers, boundary_margin=10)

    assert cell_table[CELL_COLUMNS].values.tolist() == expected_cells
    expected_registered = [row for row in expected_overlaps if row[1] in {1, 2, 3}]
    assert (
        overlaps[["cell_label", "laser_label", "occupied_pixels", "laser_area"]].values.tolist()
        == expected_registered
    )


def test_pair_grouping_is_safe_at_signed_integer_limits() -> None:
    limit = np.iinfo(np.int64).max
    cell_groups = np.array([limit, 0, limit, limit - 1, 0], dtype=np.int64)
    laser_groups = np.array([limit, limit, limit, limit, limit - 1], dtype=np.int64)

    grouped = quantification_module._group_pair_counts(cell_groups, laser_groups)

    assert grouped == [
        (0, limit - 1, 1),
        (0, limit, 1),
        (limit - 1, limit, 1),
        (limit, limit, 2),
    ]


def test_large_sparse_labels_do_not_use_per_label_full_raster_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = np.uint64(np.iinfo(np.int64).max + 1)
    cells = np.zeros((512, 512), dtype=np.uint64)
    lasers = np.zeros_like(cells)
    cells[256, 256] = label
    lasers[256, 256] = label
    original_nonzero = np.nonzero

    def reject_full_raster_nonzero(values: np.ndarray) -> tuple[np.ndarray, ...]:
        if values.shape == cells.shape:
            raise AssertionError("legacy per-label full-raster scan was used")
        return original_nonzero(values)

    monkeypatch.setattr(quantification_module.np, "nonzero", reject_full_raster_nonzero)
    adata = laser_adata()[["p1"]].copy()
    adata.obs["seg_label"] = str(label)
    adata.obs["area"] = 1.0
    adata.obs["centroid-0"] = 256.0
    adata.obs["centroid-1"] = 256.0

    overlaps, cell_table = build_overlap_table(adata, cells, lasers, boundary_margin=1)

    assert cell_table["label"].tolist() == [int(label)]
    assert overlaps[["cell_label", "laser_label", "occupied_pixels"]].values.tolist() == [
        [int(label), int(label), 1]
    ]


def test_quantified_cell_labels_round_trip_through_h5ad(tmp_path):
    label = np.uint64(np.iinfo(np.int64).max + 1)
    adata = laser_adata()[["p1"]].copy()
    adata.obs["seg_label"] = str(label)
    adata.obs["area"] = 1.0
    adata.obs["centroid-0"] = 0.0
    adata.obs["centroid-1"] = 0.0
    cells = np.array([[label]], dtype=np.uint64)
    lasers = np.array([[1]], dtype=np.uint64)

    result = quantify_cells(adata, cells, lasers, method="legacy_proportional")
    destination = atomic_write_h5ad(result.adata, tmp_path / "cells.h5ad")
    reloaded = read_h5ad(destination)

    assert reloaded.obs["label"].tolist() == [int(label)]


def test_combined_and_csv_inferred_integral_seg_labels_are_canonicalized() -> None:
    adata = laser_adata()[["p1"]].copy()
    adata.obs["seg_label"] = pd.Series(" 1_02 ", index=adata.obs_names, dtype=object)
    cells, laser = label_arrays()

    overlaps, _ = build_overlap_table(adata, cells, laser, boundary_margin=10)

    assert set(overlaps["laser_obs_id"]) == {"p1"}
    assert set(overlaps["laser_label"]) == {1, 2}

    csv_inferred = laser_adata()
    csv_inferred.obs["seg_label"] = [1, 2]
    integral_overlaps, _ = build_overlap_table(csv_inferred, cells, laser, boundary_margin=10)
    assert integral_overlaps["laser_obs_id"].tolist() == ["p1", "p2", "p2"]


@pytest.mark.parametrize("value", [None, pd.NA, True, 1.5, np.inf, -np.inf, "", "   ", "1__2"])
def test_build_overlap_table_rejects_invalid_seg_labels(value: object) -> None:
    adata = laser_adata()
    adata.obs["seg_label"] = adata.obs["seg_label"].astype(object)
    adata.obs.loc["p1", "seg_label"] = value
    cells, laser = label_arrays()

    with pytest.raises(QuantificationError, match="seg_label"):
        build_overlap_table(adata, cells, laser)


def test_build_overlap_table_rejects_duplicate_constituent_mappings() -> None:
    adata = laser_adata()
    adata.obs.loc["p1", "seg_label"] = "1_2"
    cells, laser = label_arrays()

    with pytest.raises(QuantificationError, match="Laser label 2 maps"):
        build_overlap_table(adata, cells, laser)


@pytest.mark.parametrize(
    ("cell_labels", "laser_labels"),
    [
        (np.zeros((2, 2, 1), dtype=np.int32), np.zeros((2, 2, 1), dtype=np.int32)),
        (np.zeros((2, 2), dtype=np.int32), np.zeros((2, 3), dtype=np.int32)),
        (np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)),
        (np.zeros((2, 2), dtype=float), np.zeros((2, 2), dtype=np.int32)),
        (np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=np.int32)),
        (-np.ones((2, 2), dtype=np.int32), np.zeros((2, 2), dtype=np.int32)),
    ],
)
def test_build_overlap_table_rejects_invalid_label_arrays(
    cell_labels: np.ndarray, laser_labels: np.ndarray
) -> None:
    with pytest.raises(QuantificationError, match="segmentations|labels"):
        build_overlap_table(laser_adata(), cell_labels, laser_labels)


def test_build_overlap_table_preserves_unsigned_labels_above_signed_int64() -> None:
    label = np.uint64(np.iinfo(np.int64).max + 1)
    adata = laser_adata()[["p1"]].copy()
    adata.obs["seg_label"] = str(label)
    adata.obs["area"] = 1.0
    adata.obs["centroid-0"] = 0.0
    adata.obs["centroid-1"] = 0.0
    cells = np.array([[label]], dtype=np.uint64)
    lasers = np.array([[label]], dtype=np.uint64)

    overlaps, cell_table = build_overlap_table(adata, cells, lasers)

    assert overlaps["cell_label"].iloc[0] == int(label)
    assert overlaps["laser_label"].iloc[0] == int(label)
    assert cell_table["label"].iloc[0] == int(label)


def test_build_overlap_table_rejects_sequence_seg_label_with_domain_error() -> None:
    adata = laser_adata()
    adata.obs["seg_label"] = pd.Series([[], "2"], index=adata.obs_names, dtype=object)
    cells, laser = label_arrays()

    with pytest.raises(QuantificationError, match="seg_label"):
        build_overlap_table(adata, cells, laser)


@pytest.mark.parametrize("margin", [True, -0.1, np.nan, np.inf, "20"])
def test_build_overlap_table_rejects_invalid_boundary_margin(margin: object) -> None:
    cells, laser = label_arrays()

    with pytest.raises(QuantificationError, match="boundary_margin"):
        build_overlap_table(laser_adata(), cells, laser, boundary_margin=margin)  # type: ignore[arg-type]


def test_build_overlap_table_validates_adata_metadata_and_unique_obs_names() -> None:
    cells, laser = label_arrays()
    with pytest.raises(QuantificationError, match="AnnData"):
        build_overlap_table(object(), cells, laser)  # type: ignore[arg-type]

    for column, value in [("area", -1.0), ("centroid-0", np.nan), ("centroid-1", True)]:
        adata = laser_adata()
        adata.obs[column] = adata.obs[column].astype(object)
        adata.obs.loc["p1", column] = value
        with pytest.raises(QuantificationError, match=column):
            build_overlap_table(adata, cells, laser)

    missing = laser_adata()
    missing.obs = missing.obs.drop(columns="area")
    with pytest.raises(QuantificationError, match="missing obs columns"):
        build_overlap_table(missing, cells, laser)

    duplicate_names = laser_adata()
    duplicate_names.obs_names = ["p1", "p1"]
    with pytest.raises(QuantificationError, match="unique observation"):
        build_overlap_table(duplicate_names, cells, laser)


def test_zero_area_and_unregistered_raster_labels_are_omitted_with_stable_empty_tables() -> None:
    adata = laser_adata()[["p1"]].copy()
    adata.obs["area"] = 0.0
    cells = np.array([[1, 1], [0, 0]], dtype=np.int32)
    laser = np.array([[2, 2], [0, 0]], dtype=np.int32)

    overlaps, cell_table = build_overlap_table(adata, cells, laser, boundary_margin=10)

    assert overlaps.empty
    assert overlaps.columns.tolist() == OVERLAP_COLUMNS
    assert cell_table["cell_id"].tolist() == ["c1"]

    empty_overlaps, empty_cells = build_overlap_table(
        adata,
        np.zeros((2, 2), dtype=np.int32),
        np.zeros((2, 2), dtype=np.int32),
        boundary_margin=10,
    )
    assert empty_overlaps.empty
    assert empty_overlaps.columns.tolist() == OVERLAP_COLUMNS
    assert empty_cells.empty
    assert empty_cells.columns.tolist() == CELL_COLUMNS
    assert str(empty_cells["cell_id"].dtype) == "string"


def test_boundary_filtering_is_deterministic_and_does_not_mutate_inputs() -> None:
    adata = laser_adata()
    cells = np.array(
        [[0, 0, 0, 0, 0], [1, 2, 0, 0, 3], [0, 0, 0, 0, 0]], dtype=np.int32
    )
    laser = np.array(
        [[0, 0, 0, 0, 0], [1, 1, 0, 0, 2], [0, 0, 0, 0, 0]], dtype=np.int32
    )
    original_adata = adata.copy()
    original_cells = cells.copy()
    original_laser = laser.copy()

    _, cell_table = build_overlap_table(adata, cells, laser, boundary_margin=0.75)

    assert cell_table["cell_id"].tolist() == ["c2"]
    np.testing.assert_array_equal(cells, original_cells)
    np.testing.assert_array_equal(laser, original_laser)
    assert adata.obs.equals(original_adata.obs)
    assert deepcopy(adata.uns) == deepcopy(original_adata.uns)


def test_specificity_filter_keeps_unique_and_dominant_mappings_and_rejects_secondaries() -> None:
    overlaps = selector_overlaps()
    original = overlaps.copy(deep=True)

    accepted, rejected = select_specificity_filtered(
        overlaps,
        unique_min_overlap=0.1,
        dominant_min_overlap=0.6,
        secondary_max_overlap=0.15,
    )

    assert accepted[["cell_id", "laser_label"]].values.tolist() == [["c1", 1], ["c1", 2]]
    assert rejected[["cell_id", "laser_label"]].values.tolist() == [["c2", 2], ["c3", 3]]
    assert rejected["rejection_reason"].tolist() == [
        "non_dominant_mapping",
        "unique_overlap_not_above_threshold",
    ]
    pd.testing.assert_frame_equal(overlaps, original)


@pytest.mark.parametrize(
    ("ratios", "thresholds"),
    [
        ([0.1], (0.1, 0.6, 0.15)),
        ([0.6, 0.1], (0.1, 0.6, 0.15)),
        ([0.7, 0.15], (0.1, 0.6, 0.15)),
    ],
)
def test_specificity_filter_threshold_comparisons_are_strict(
    ratios: list[float], thresholds: tuple[float, float, float]
) -> None:
    overlaps = pd.DataFrame(
        {
            "cell_id": [f"c{index}" for index in range(1, len(ratios) + 1)],
            "laser_label": [1] * len(ratios),
            "occupied_ratio": ratios,
        }
    )

    accepted, rejected = select_specificity_filtered(
        overlaps,
        unique_min_overlap=thresholds[0],
        dominant_min_overlap=thresholds[1],
        secondary_max_overlap=thresholds[2],
    )

    assert accepted.empty
    assert len(rejected) == len(overlaps)
    assert rejected["rejection_reason"].notna().all()


def test_specificity_filter_rejects_all_mappings_when_dominant_ratios_tie() -> None:
    overlaps = pd.DataFrame(
        {
            "cell_id": ["c2", "c1", "c3"],
            "laser_label": [4, 4, 4],
            "occupied_ratio": [0.7, 0.7, 0.05],
        }
    )

    accepted, rejected = select_specificity_filtered(
        overlaps,
        unique_min_overlap=0.1,
        dominant_min_overlap=0.6,
        secondary_max_overlap=0.15,
    )

    assert accepted.empty
    assert rejected["cell_id"].tolist() == ["c2", "c1", "c3"]
    assert rejected["rejection_reason"].eq("ambiguous_multi_mapping").all()


@pytest.mark.parametrize("value", [True, False, -0.01, 1.01, np.nan, np.inf, -np.inf, "0.1"])
@pytest.mark.parametrize(
    "parameter", ["unique_min_overlap", "dominant_min_overlap", "secondary_max_overlap"]
)
def test_specificity_filter_validates_threshold_ratios(parameter: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "unique_min_overlap": 0.1,
        "dominant_min_overlap": 0.6,
        "secondary_max_overlap": 0.15,
    }
    kwargs[parameter] = value

    with pytest.raises(QuantificationError, match=parameter):
        select_specificity_filtered(selector_overlaps(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overlaps",
    [
        object(),
        selector_overlaps().drop(columns="cell_id"),
        selector_overlaps().assign(cell_id=["c1", "c1", "c2", pd.NA]),
        pd.concat([selector_overlaps(), selector_overlaps().iloc[[0]]], ignore_index=True),
        selector_overlaps().assign(occupied_ratio=[0.2, np.nan, 0.1, 0.05]),
        selector_overlaps().assign(occupied_ratio=[0.2, 1.1, 0.1, 0.05]),
        selector_overlaps().assign(occupied_ratio=[0.2, True, 0.1, 0.05]),
    ],
)
def test_specificity_filter_validates_public_overlap_tables(overlaps: object) -> None:
    with pytest.raises(QuantificationError, match="overlap|Overlap|occupied_ratio"):
        select_specificity_filtered(
            overlaps,  # type: ignore[arg-type]
            unique_min_overlap=0.1,
            dominant_min_overlap=0.6,
            secondary_max_overlap=0.15,
        )


def test_legacy_proportional_matches_exact_d2_formula_with_exact_obs_id_mapping() -> None:
    cells, laser = label_arrays()
    adata = laser_adata()[["p2", "p1"]].copy()

    result = quantify_cells(
        adata, cells, laser, method="legacy_proportional", boundary_margin=10
    )

    np.testing.assert_allclose(result.adata["c1"].X.toarray(), [[17.5, 30.0]])
    assert result.accepted["laser_obs_id"].tolist() == ["p1", "p2", "p2"]
    assert result.rejected.empty


def test_specificity_filtered_uses_full_prefilter_specificity_and_corrected_abundance() -> None:
    adata = anndata.AnnData(
        np.array([[100.0], [50.0]]),
        obs={
            "seg_label": ["1", "2"],
            "area": [2.0, 10.0],
            "centroid-0": [0.0, 1.0],
            "centroid-1": [0.5, 4.5],
        },
        var={"mz": [100.0]},
    )
    adata.obs_names = ["laser-a", "laser-b"]
    laser = np.array(
        [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
        dtype=np.int32,
    )
    cells = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 2, 0]],
        dtype=np.int32,
    )

    result = quantify_cells(adata, cells, laser, boundary_margin=10)

    assert result.adata["c1"].X.toarray()[0, 0] == pytest.approx(770 / 17)
    assert result.accepted[["cell_id", "laser_label"]].values.tolist() == [["c1", 1], ["c1", 2]]
    assert result.rejected[["cell_id", "laser_label"]].values.tolist() == [["c2", 2]]
    assert result.rejected["rejection_reason"].tolist() == ["non_dominant_mapping"]


def test_specificity_filtered_matches_corrected_d8_mapping_behavior() -> None:
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
    assert result.adata.obs["quantification_status"].eq("quantified").all()


@pytest.mark.parametrize("dense", [False, True])
def test_quantify_cells_builds_stable_cell_anndata_with_zero_rows_without_mutation(
    dense: bool,
) -> None:
    adata = laser_adata()
    if dense:
        adata.X = adata.X.toarray()
    cells, laser = label_arrays()
    cells = cells.copy()
    cells[2, 0] = 3
    original_adata = adata.copy()
    original_cells = cells.copy()
    original_laser = laser.copy()

    result = quantify_cells(
        adata, cells, laser, method="legacy_proportional", boundary_margin=10
    )

    assert sparse.isspmatrix_csr(result.adata.X)
    assert result.adata.shape == (3, 2)
    np.testing.assert_array_equal(result.adata["c3"].X.toarray(), np.zeros((1, 2)))
    assert result.adata.obs["quantification_status"].to_dict() == {
        "c1": "quantified",
        "c2": "quantified",
        "c3": "zero",
    }
    assert result.adata.obs["laser_labels"].to_dict() == {
        "c1": "1_2",
        "c2": "2",
        "c3": "0",
    }
    assert result.adata.obs["laser_coverage"].to_dict() == {
        "c1": "4_1",
        "c2": "3",
        "c3": "0",
    }
    np.testing.assert_allclose(
        result.adata.obsm["spatial"],
        result.adata.obs[["centroid-1", "centroid-0"]].to_numpy(dtype=float),
    )
    pd.testing.assert_frame_equal(result.adata.var, original_adata.var)
    result.adata.var.loc[result.adata.var.index[0], "mz"] = -1.0
    assert adata.var["mz"].tolist() == [100.0, 200.0]
    np.testing.assert_array_equal(cells, original_cells)
    np.testing.assert_array_equal(laser, original_laser)
    pd.testing.assert_frame_equal(adata.obs, original_adata.obs)
    assert deepcopy(adata.uns) == deepcopy(original_adata.uns)
    if sparse.issparse(adata.X):
        np.testing.assert_array_equal(adata.X.toarray(), original_adata.X.toarray())
    else:
        np.testing.assert_array_equal(adata.X, original_adata.X)


def test_quantify_cells_returns_stable_empty_sparse_shape() -> None:
    result = quantify_cells(
        laser_adata(),
        np.zeros((2, 2), dtype=np.int32),
        np.zeros((2, 2), dtype=np.int32),
        boundary_margin=10,
    )

    assert result.adata.shape == (0, 2)
    assert sparse.isspmatrix_csr(result.adata.X)
    assert result.adata.obsm["spatial"].shape == (0, 2)
    assert result.accepted.empty
    assert result.rejected.empty
    assert "rejection_reason" in result.rejected.columns


@pytest.mark.parametrize("method", [None, True, 1, "unknown"])
def test_quantify_cells_validates_method(method: object) -> None:
    cells, laser = label_arrays()

    with pytest.raises(QuantificationError, match="method"):
        quantify_cells(laser_adata(), cells, laser, method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, -0.1, 1.1, np.nan, np.inf, "0.1"])
@pytest.mark.parametrize(
    "parameter", ["unique_min_overlap", "dominant_min_overlap", "secondary_max_overlap"]
)
def test_quantify_cells_validates_every_threshold_for_both_methods(
    parameter: str, value: object
) -> None:
    cells, laser = label_arrays()
    kwargs: dict[str, object] = {
        "method": "legacy_proportional",
        "unique_min_overlap": 0.1,
        "dominant_min_overlap": 0.6,
        "secondary_max_overlap": 0.15,
    }
    kwargs[parameter] = value

    with pytest.raises(QuantificationError, match=parameter):
        quantify_cells(laser_adata(), cells, laser, **kwargs)  # type: ignore[arg-type]


def test_quantify_cells_public_unique_threshold_default_is_exactly_point_one() -> None:
    assert signature(quantify_cells).parameters["unique_min_overlap"].default == 0.10
