from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
from skimage import morphology

from joint.errors import SegmentationError
from joint.segmentation import (
    detect_rows,
    exclude_regions,
    load_merge_rules,
    merge_regions,
    region_table,
    segment_laser_marks,
    simulate_laser_marks,
)


def two_disks_image():
    image = np.zeros((80, 100), dtype=float)
    rr, cc = np.ogrid[:80, :100]
    image[(rr - 25) ** 2 + (cc - 25) ** 2 <= 8**2] = 1.0
    image[(rr - 55) ** 2 + (cc - 70) ** 2 <= 9**2] = 1.0
    return image


def top_hat_image():
    image = np.zeros((50, 50), dtype=float)
    rr, cc = np.ogrid[:50, :50]
    image[(rr - 25) ** 2 + (cc - 25) ** 2 <= 10**2] = 0.4
    image[(rr - 25) ** 2 + (cc - 25) ** 2 <= 2**2] = 1.0
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
    result = segment_laser_marks(
        two_disks_image(),
        mask=mask,
        min_peak_distance=10,
        min_size=30,
    )
    assert result.regions.shape[0] == 1


def test_segment_laser_marks_defaults_to_white_top_hat_enhancement_without_mutation():
    image = top_hat_image()
    original = image.copy()
    expected = morphology.white_tophat(image, morphology.disk(4))

    result = segment_laser_marks(
        image,
        intensity_cutoff=0.1,
        top_hat_radius=4,
        closing_size=1,
        post_closing_size=1,
        opening_size=1,
        min_peak_distance=2,
        min_size=3,
        threshold=0.1,
    )

    np.testing.assert_allclose(result.diagnostics["processed_image"], expected)
    np.testing.assert_array_equal(image, original)


def test_segment_laser_marks_supports_explicit_legacy_opening_mode():
    image = top_hat_image()
    expected = image - morphology.white_tophat(image, morphology.disk(4))

    result = segment_laser_marks(
        image,
        intensity_cutoff=0.1,
        top_hat_radius=4,
        top_hat_mode="legacy_opening",
        closing_size=1,
        post_closing_size=1,
        opening_size=1,
        min_peak_distance=2,
        min_size=3,
        threshold=0.1,
    )

    np.testing.assert_allclose(result.diagnostics["processed_image"], expected)


@pytest.mark.parametrize("top_hat_mode", ["enhance", "legacy_opening"])
def test_segment_laser_marks_top_hat_radius_zero_preserves_cutoff_image(top_hat_mode):
    image = top_hat_image()

    result = segment_laser_marks(
        image,
        intensity_cutoff=0.5,
        top_hat_radius=0,
        top_hat_mode=top_hat_mode,
        closing_size=1,
        post_closing_size=1,
        opening_size=1,
        min_peak_distance=2,
        min_size=3,
        threshold=0.5,
    )

    expected = image.copy()
    expected[expected < 0.5] = 0
    np.testing.assert_array_equal(result.diagnostics["processed_image"], expected)


def test_segment_laser_marks_rejects_invalid_top_hat_mode():
    with pytest.raises(SegmentationError, match="top_hat_mode"):
        segment_laser_marks(top_hat_image(), top_hat_mode="opening")


@pytest.mark.parametrize("keyword", ["top_hat_mode", "peak_detection_mode"])
def test_segment_laser_marks_rejects_non_string_mode_values(keyword):
    with pytest.raises(SegmentationError, match=keyword):
        segment_laser_marks(two_disks_image(), **{keyword: []})


def test_segment_laser_marks_preserves_single_pixel_mask_hole_after_closing():
    image = np.zeros((40, 40), dtype=float)
    rr, cc = np.ogrid[:40, :40]
    image[(rr - 20) ** 2 + (cc - 20) ** 2 <= 7**2] = 1.0
    mask = np.ones_like(image, dtype=bool)
    mask[20, 20] = False

    result = segment_laser_marks(
        image,
        mask=mask,
        intensity_cutoff=0.1,
        top_hat_radius=0,
        closing_size=3,
        post_closing_size=3,
        opening_size=1,
        min_peak_distance=3,
        min_size=20,
    )

    assert result.labels[20, 20] == 0
    assert not result.diagnostics["binary_mask"][20, 20]


def test_post_border_closing_and_marker_connectivity_are_configurable():
    image = np.zeros((30, 30), dtype=float)
    image[10:15, 8:13] = 1.0
    image[10:15, 14:19] = 1.0
    for connectivity in (1, 2):
        result = segment_laser_marks(
            image,
            intensity_cutoff=0.1,
            top_hat_radius=0,
            closing_size=1,
            post_closing_size=3,
            opening_size=1,
            min_peak_distance=2,
            marker_connectivity=connectivity,
            min_size=1,
            threshold=0.5,
        )
        assert result.diagnostics["binary_mask"][12, 13]


def test_segment_laser_marks_gives_each_nearby_disconnected_object_a_marker():
    image = np.zeros((40, 40), dtype=float)
    rr, cc = np.ogrid[:40, :40]
    image[(rr - 20) ** 2 + (cc - 15) ** 2 <= 3**2] = 1.0
    image[(rr - 20) ** 2 + (cc - 23) ** 2 <= 3**2] = 1.0

    result = segment_laser_marks(
        image,
        intensity_cutoff=0.1,
        top_hat_radius=0,
        closing_size=1,
        post_closing_size=1,
        opening_size=1,
        min_peak_distance=10,
        min_size=10,
    )

    assert result.regions.shape[0] == 2


def test_segment_laser_marks_supports_legacy_global_peak_detection():
    image = np.zeros((40, 40), dtype=float)
    rr, cc = np.ogrid[:40, :40]
    image[(rr - 20) ** 2 + (cc - 15) ** 2 <= 3**2] = 1.0
    image[(rr - 20) ** 2 + (cc - 23) ** 2 <= 3**2] = 1.0

    result = segment_laser_marks(
        image,
        intensity_cutoff=0.1,
        top_hat_radius=0,
        closing_size=1,
        post_closing_size=1,
        opening_size=1,
        min_peak_distance=10,
        peak_detection_mode="legacy_global",
        min_size=10,
    )

    assert result.regions.shape[0] == 1


def test_segment_laser_marks_rejects_invalid_peak_detection_mode():
    with pytest.raises(SegmentationError, match="peak_detection_mode"):
        segment_laser_marks(two_disks_image(), peak_detection_mode="unsupported")


def test_segment_laser_marks_clears_objects_touching_image_border():
    image = np.zeros((40, 40), dtype=float)
    rr, cc = np.ogrid[:40, :40]
    image[(rr - 0) ** 2 + (cc - 10) ** 2 <= 5**2] = 1.0
    image[(rr - 25) ** 2 + (cc - 25) ** 2 <= 5**2] = 1.0

    result = segment_laser_marks(
        image,
        intensity_cutoff=0.1,
        top_hat_radius=0,
        closing_size=1,
        post_closing_size=1,
        opening_size=1,
        min_peak_distance=5,
        min_size=20,
    )

    assert result.regions.shape[0] == 1
    assert not result.labels[0].any()


def test_segment_laser_marks_keeps_near_border_object_that_does_not_touch_border():
    image = np.zeros((30, 30), dtype=float)
    rr, cc = np.ogrid[:30, :30]
    image[(rr - 2) ** 2 + (cc - 15) ** 2 <= 1**2] = 1.0

    result = segment_laser_marks(
        image,
        intensity_cutoff=0.1,
        top_hat_radius=0,
        closing_size=1,
        post_closing_size=1,
        opening_size=1,
        min_peak_distance=5,
        min_size=3,
    )

    assert result.regions.shape[0] == 1


def test_region_table_is_sorted_by_label():
    labels = np.array([[0, 2, 2], [1, 0, 0]], dtype=np.int32)
    table = region_table(labels)
    assert table["label"].tolist() == [1, 2]


def test_region_table_rejects_labels_that_cannot_be_represented_safely():
    labels = np.array([[0, np.iinfo(np.int32).max + 1]], dtype=np.uint64)

    with pytest.raises(SegmentationError, match="int32"):
        region_table(labels)


def test_remove_small_regions_handles_sparse_uint64_labels_without_bincount():
    from joint.segmentation import _remove_small_regions

    label = np.uint64(np.iinfo(np.int32).max + 1)
    labels = np.array([[0, label], [label, 0]], dtype=np.uint64)

    filtered = _remove_small_regions(labels, min_size=2)

    assert filtered.dtype == np.uint64
    np.testing.assert_array_equal(filtered, labels)


def test_segment_laser_marks_rejects_image_without_markers():
    with pytest.raises(SegmentationError, match="No laser markers"):
        segment_laser_marks(np.zeros((20, 20)), intensity_cutoff=0.1)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((2, 3, 1)), "two-dimensional"),
        (np.array([[np.nan]]), "non-finite"),
    ],
)
def test_segment_laser_marks_validates_image(image, message):
    with pytest.raises(SegmentationError, match=message):
        segment_laser_marks(image)


def test_segment_laser_marks_validates_mask_shape():
    with pytest.raises(SegmentationError, match="identical shapes"):
        segment_laser_marks(np.zeros((3, 4)), mask=np.ones((4, 3), dtype=bool))


def test_segment_laser_marks_rejects_non_finite_mask():
    mask = np.ones((80, 100), dtype=float)
    mask[0, 0] = np.nan

    with pytest.raises(SegmentationError, match="mask.*non-finite"):
        segment_laser_marks(two_disks_image(), mask=mask)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("closing_size", 0),
        ("opening_size", 0),
        ("post_closing_size", 0),
        ("min_peak_distance", 0),
        ("marker_connectivity", 3),
        ("min_size", 0),
        ("top_hat_radius", -1),
        ("intensity_cutoff", -0.1),
        ("intensity_cutoff", np.nan),
    ],
)
def test_segment_laser_marks_rejects_invalid_parameters(keyword, value):
    with pytest.raises(SegmentationError, match=keyword):
        segment_laser_marks(two_disks_image(), **{keyword: value})


@pytest.mark.parametrize("threshold", ["triangle", np.nan, np.inf])
def test_segment_laser_marks_rejects_invalid_threshold(threshold):
    with pytest.raises(SegmentationError, match="threshold"):
        segment_laser_marks(two_disks_image(), threshold=threshold)


def test_region_table_returns_stable_columns_for_empty_labels():
    table = region_table(np.zeros((4, 5), dtype=np.int32))

    assert table.empty
    assert table.columns.tolist() == ["label", "area", "centroid-0", "centroid-1"]


@pytest.mark.parametrize(
    "labels",
    [np.zeros((2, 2, 1), dtype=np.int32), np.zeros((2, 2), dtype=float), -np.ones((2, 2))],
)
def test_region_table_rejects_invalid_labels(labels):
    with pytest.raises(SegmentationError, match="labels"):
        region_table(labels)


def test_segment_laser_marks_does_not_mutate_image_or_mask():
    image = two_disks_image()
    mask = np.ones(image.shape, dtype=bool)
    original_image = image.copy()
    original_mask = mask.copy()

    segment_laser_marks(
        image,
        mask=mask,
        top_hat_radius=0,
        min_peak_distance=10,
        min_size=30,
    )

    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(mask, original_mask)


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


def test_detect_rows_rejects_invalid_method_and_region_table():
    with pytest.raises(SegmentationError, match="method"):
        detect_rows(sample_regions(), method="unsupported")
    with pytest.raises(SegmentationError, match="missing columns"):
        detect_rows(sample_regions().drop(columns="area"))
    malformed = sample_regions().astype({"label": float})
    malformed.loc[0, "label"] = 1.5
    with pytest.raises(SegmentationError, match="label.*positive integer"):
        detect_rows(malformed)


def test_detect_rows_rejects_invalid_parameters():
    with pytest.raises(SegmentationError, match="window_size"):
        detect_rows(sample_regions(), method="rolling_variance", window_size=0)
    with pytest.raises(SegmentationError, match="variance_threshold"):
        detect_rows(sample_regions(), method="rolling_variance", variance_threshold=np.nan)
    with pytest.raises(SegmentationError, match="thinning"):
        detect_rows(sample_regions(), method="rolling_variance", thinning=0)


def test_detect_rows_rolling_variance_handles_empty_and_singleton_regions():
    empty = sample_regions().iloc[0:0]
    singleton = sample_regions().iloc[:1]

    assert detect_rows(empty, method="rolling_variance", window_size=5) == []
    rows = detect_rows(singleton, method="rolling_variance", window_size=5)
    assert [row["seg_label"].tolist() for row in rows] == [["1"]]


def test_detect_rows_rolling_variance_excludes_terminal_full_window():
    regions = pd.DataFrame(
        {
            "label": [1, 2, 3, 4, 5],
            "area": [1.0] * 5,
            "centroid-0": [0.0, 0.0, 0.0, 0.0, 100.0],
            "centroid-1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    rows = detect_rows(
        regions,
        method="rolling_variance",
        window_size=5,
        variance_threshold=1.0,
        thinning=1,
    )

    assert len(rows) == 1
    assert rows[0]["seg_label"].tolist() == ["1", "2", "3", "4", "5"]


def test_detect_rows_rolling_variance_does_not_split_when_window_exceeds_regions():
    regions = sample_regions().iloc[:2]

    rows = detect_rows(
        regions,
        method="rolling_variance",
        window_size=3,
        variance_threshold=0,
        thinning=1,
    )

    assert [row["seg_label"].tolist() for row in rows] == [["1", "2"]]


@pytest.mark.parametrize("window_size", [2, 3])
def test_detect_rows_rolling_variance_excludes_terminal_window(window_size):
    regions = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "area": [1.0, 1.0, 1.0],
            "centroid-0": [0.0, 0.1, 10.0],
            "centroid-1": [0.0, 1.0, 2.0],
        }
    )

    rows = detect_rows(
        regions,
        method="rolling_variance",
        window_size=window_size,
        variance_threshold=1.0,
        thinning=1,
    )

    assert [row["seg_label"].tolist() for row in rows] == [["1", "2", "3"]]


def test_exclude_regions_uses_explicit_coordinate_bounds():
    kept = exclude_regions(sample_regions(), min_column=15)

    assert kept["label"].tolist() == [2, 4]


def test_exclude_regions_rejects_malformed_regions_and_bounds():
    with pytest.raises(SegmentationError, match="missing columns"):
        exclude_regions(sample_regions().drop(columns="centroid-0"))
    with pytest.raises(SegmentationError, match="min_row"):
        exclude_regions(sample_regions(), min_row=np.nan)
    with pytest.raises(SegmentationError, match="min_column.*max_column"):
        exclude_regions(sample_regions(), min_column=20, max_column=10)


def test_merge_regions_combines_labels_and_marks_sources():
    rows = pd.concat(detect_rows(sample_regions(), expected_rows=2), ignore_index=True)
    rules = pd.DataFrame({"row_number": [2], "source_labels": ["3,4"]})

    merged = merge_regions(rows, rules)
    combined = merged.query('morphology == "combined"').iloc[0]

    assert combined["seg_label"] == "3_4"
    assert combined["area"] == 12.0
    assert merged.query('morphology != "spoilt" and row_number == 2').shape[0] == 1


def test_merge_regions_rejects_malformed_regions_and_rules():
    rows = pd.concat(detect_rows(sample_regions(), expected_rows=2), ignore_index=True)

    with pytest.raises(SegmentationError, match="missing columns"):
        merge_regions(rows.drop(columns="seg_label"), pd.DataFrame())
    with pytest.raises(SegmentationError, match="Merge rules are missing columns"):
        merge_regions(rows, pd.DataFrame({"row_number": [2]}))
    with pytest.raises(SegmentationError, match="at least two"):
        merge_regions(rows, pd.DataFrame({"row_number": [2], "source_labels": ["3"]}))
    with pytest.raises(SegmentationError, match="exactly one intact"):
        merge_regions(rows, pd.DataFrame({"row_number": [2], "source_labels": ["3,9"]}))


def test_merge_regions_refreshes_active_row_positions_after_multiple_merges():
    regions = pd.DataFrame(
        {
            "label": [1, 2, 3, 4, 5, 6],
            "area": [1.0] * 6,
            "centroid-0": [10.0] * 6,
            "centroid-1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    rows = pd.concat(detect_rows(regions, expected_rows=1), ignore_index=True)
    rules = pd.DataFrame({"row_number": [1, 1], "source_labels": ["1,2", "4,5"]})

    merged = merge_regions(rows, rules)
    active = merged.query('morphology != "spoilt"').sort_values("column_number")

    assert active["seg_label"].tolist() == ["1_2", "3", "4_5", "6"]
    assert active["column_number"].tolist() == [1, 2, 3, 4]
    assert active["right_to_left"].tolist() == [-4, -3, -2, -1]
    assert merged.query('morphology == "spoilt"').shape[0] == 4


def test_merge_regions_rejects_duplicate_intact_label_when_source_is_missing():
    rows = pd.concat(detect_rows(sample_regions(), expected_rows=2), ignore_index=True)
    duplicate = pd.concat([rows.iloc[:1], rows.iloc[:1]], ignore_index=True)
    duplicate.loc[:, "row_number"] = 1
    duplicate.loc[:, "seg_label"] = "1"
    rules = pd.DataFrame({"row_number": [1], "source_labels": ["1,2"]})

    with pytest.raises(SegmentationError, match="exactly one intact"):
        merge_regions(duplicate, rules)


def test_load_merge_rules_validates_columns(tmp_path: Path):
    path = tmp_path / "rules.csv"
    path.write_text("row_number,source_labels\n2,3|4\n")

    rules = load_merge_rules(path)

    assert rules.iloc[0]["source_labels"] == "3|4"


def test_load_merge_rules_rejects_missing_or_malformed_values(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(SegmentationError, match="does not exist"):
        load_merge_rules(missing)

    malformed = tmp_path / "malformed.csv"
    malformed.write_text("row_number\n2\n")
    with pytest.raises(SegmentationError, match="missing columns"):
        load_merge_rules(malformed)

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("row_number,source_labels\nnot-a-row,3|4\n")
    with pytest.raises(SegmentationError, match="row_number"):
        load_merge_rules(invalid)

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(SegmentationError, match="Could not read merge rules"):
        load_merge_rules(empty)


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


def simulated_mark_inputs():
    adata = anndata.AnnData(
        np.ones((6, 1)),
        obsm={"spatial": np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])},
    )
    anchors = [
        pd.DataFrame({"centroid-0": [2.0, 2.0], "centroid-1": [1.0, 8.0]}),
        pd.DataFrame({"centroid-0": [8.0, 8.0], "centroid-1": [1.0, 8.0]}),
    ]
    return adata, anchors


def test_simulate_laser_marks_rejects_missing_or_malformed_spatial_grid():
    adata, anchors = simulated_mark_inputs()
    invalid_spatial = [
        np.zeros((6, 3)),
        np.zeros((5, 2)),
        np.array([["x", "y"]] * 6),
        np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [np.nan, 1]]),
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[0, 0], [0, 0], [0, 1], [1, 1]]),
    ]

    missing = adata.copy()
    del missing.obsm["spatial"]
    with pytest.raises(SegmentationError, match="spatial"):
        simulate_laser_marks(missing, anchors, (10, 10), radius=1)

    for spatial in invalid_spatial:
        malformed = anndata.AnnData(np.ones((len(spatial), 1)), obsm={"spatial": spatial})
        with pytest.raises(SegmentationError, match="spatial|rectangular"):
            simulate_laser_marks(malformed, anchors, (10, 10), radius=1)


def test_simulate_laser_marks_validates_anchor_rows_shape_and_radius():
    adata, anchors = simulated_mark_inputs()

    with pytest.raises(SegmentationError, match="anchor rows"):
        simulate_laser_marks(adata, anchors[:1], (10, 10), radius=1)
    with pytest.raises(SegmentationError, match="centroid"):
        simulate_laser_marks(adata, [anchors[0].drop(columns="centroid-0"), anchors[1]], (10, 10), radius=1)
    with pytest.raises(SegmentationError, match="finite"):
        invalid = anchors[0].copy()
        invalid.loc[0, "centroid-0"] = np.nan
        simulate_laser_marks(adata, [invalid, anchors[1]], (10, 10), radius=1)
    with pytest.raises(SegmentationError, match="endpoints"):
        simulate_laser_marks(adata, [anchors[0].iloc[:1], anchors[1]], (10, 10), radius=1)
    with pytest.raises(SegmentationError, match="image_shape"):
        simulate_laser_marks(adata, anchors, (10, 0), radius=1)
    for radius in (0, True, 1.5):
        with pytest.raises(SegmentationError, match="radius"):
            simulate_laser_marks(adata, anchors, (10, 10), radius=radius)
    with pytest.raises(SegmentationError, match="anchor_rows"):
        simulate_laser_marks(adata, (anchor for anchor in anchors), (10, 10), radius=1)


def test_simulate_laser_marks_orders_numeric_string_anchor_coordinates_numerically():
    adata, anchors = simulated_mark_inputs()
    numeric_strings = pd.DataFrame(
        {"centroid-0": [2.0, 2.0], "centroid-1": ["10", "2"]}
    )

    result = simulate_laser_marks(adata, [numeric_strings, anchors[1]], (12, 12), radius=1)

    assert result.regions.query("row_number == 1")["centroid-1"].tolist() == [2.0, 6.0, 10.0]


def test_simulate_laser_marks_reports_centroid_of_clipped_disk_pixels():
    adata, _ = simulated_mark_inputs()
    anchors = [
        pd.DataFrame({"centroid-0": [0.0, 0.0], "centroid-1": [1.0, 7.0]}),
        pd.DataFrame({"centroid-0": [6.0, 6.0], "centroid-1": [1.0, 7.0]}),
    ]

    result = simulate_laser_marks(adata, anchors, (10, 10), radius=2)
    first = result.regions.query("label == 1").iloc[0]

    assert first["area"] == 6.0
    assert first["centroid-0"] == 0.5
    assert first["centroid-1"] == 1.0


def test_simulate_laser_marks_rejects_overlapping_or_empty_disks_without_mutating_inputs():
    adata, anchors = simulated_mark_inputs()
    original_spatial = adata.obsm["spatial"].copy()
    original_anchors = [anchor.copy(deep=True) for anchor in anchors]
    overlapping = [
        pd.DataFrame({"centroid-0": [2.0, 2.0], "centroid-1": [1.0, 2.0]}),
        anchors[1],
    ]

    with pytest.raises(SegmentationError, match="overlap"):
        simulate_laser_marks(adata, overlapping, (10, 10), radius=2)
    empty = [
        pd.DataFrame({"centroid-0": [-10.0, -10.0], "centroid-1": [-10.0, -5.0]}),
        anchors[1],
    ]
    with pytest.raises(SegmentationError, match="no pixels"):
        simulate_laser_marks(adata, empty, (10, 10), radius=1)

    np.testing.assert_array_equal(adata.obsm["spatial"], original_spatial)
    for actual, original in zip(anchors, original_anchors, strict=True):
        pd.testing.assert_frame_equal(actual, original)
