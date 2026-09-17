from copy import deepcopy
from fractions import Fraction

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import joint
from joint.errors import QuantificationError
from joint.models import QuantificationResult
from joint.quantification import build_mixed_anndata, quantify_cells
from joint.registration import register_laser_points
from joint.segmentation import simulate_laser_marks

ACCEPTED_COLUMNS = [
    "cell_id",
    "cell_label",
    "laser_label",
    "laser_obs_id",
    "occupied_pixels",
    "laser_area",
    "occupied_ratio",
]


def spatial_laser_adata() -> anndata.AnnData:
    adata = anndata.AnnData(
        sparse.csr_matrix([[10.0, 20.0], [30.0, 40.0]]),
        obs={
            "seg_label": ["1", "2_3"],
            "area": [10.0, 20.0],
            "centroid-0": [5.0, 6.0],
            "centroid-1": [7.0, 8.0],
            "sample_region": ["north", "south"],
        },
        var=pd.DataFrame(
            {"mz": [100.0, 200.0], "laser_annotation": ["a", "b"]},
            index=["feature-100", "feature-200"],
        ),
    )
    adata.obs_names = ["p1", "p2"]
    adata.obsm["spatial"] = np.array([[7.0, 5.0], [8.0, 6.0]])
    adata.uns = {"joint": {"registration": {"orientation": "identity"}}}
    return adata


def accepted_table(*rows: tuple[object, ...]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows, columns=ACCEPTED_COLUMNS)


def quantification_result(
    laser: anndata.AnnData, accepted: pd.DataFrame
) -> QuantificationResult:
    var = laser.var.copy(deep=True)
    var["cell_annotation"] = ["kept-first", "kept-second"]
    cells = anndata.AnnData(
        sparse.csc_matrix([[1.0, 2.0], [3.0, 4.0]]),
        obs=pd.DataFrame(
            {"label": [1, 2], "cell_quality": [0.9, 0.8]}, index=["c1", "c2"]
        ),
        var=var,
        obsm={"spatial": np.array([[1.5, 2.5], [3.5, 4.5]])},
        uns={
            "joint": {"quantification_method": "legacy_proportional"},
            "nested": {"values": [1, 2]},
        },
    )
    rejected = accepted.iloc[0:0].copy()
    rejected["rejection_reason"] = pd.Series(dtype="string")
    return QuantificationResult(
        adata=cells,
        overlaps=accepted.copy(deep=True),
        accepted=accepted.copy(deep=True),
        rejected=rejected,
        report={"accepted_mappings": len(accepted)},
    )


def matrix(adata: anndata.AnnData) -> np.ndarray:
    return adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)


def assert_adata_unchanged(actual: anndata.AnnData, expected: anndata.AnnData) -> None:
    np.testing.assert_array_equal(matrix(actual), matrix(expected))
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    assert set(actual.obsm) == set(expected.obsm)
    for key in expected.obsm:
        np.testing.assert_array_equal(actual.obsm[key], expected.obsm[key])
    assert deepcopy(actual.uns) == deepcopy(expected.uns)


def test_single_label_residual_scales_spectrum_and_includes_threshold_equality() -> None:
    laser = spatial_laser_adata()[["p1"]].copy()
    accepted = accepted_table(("c1", 1, 1, "p1", 8, 10, 0.8))
    result = quantification_result(spatial_laser_adata(), accepted)
    result_adata = result.adata[:, laser.var_names].copy()
    result = QuantificationResult(
        result_adata,
        result.overlaps,
        result.accepted,
        result.rejected,
        result.report,
    )

    mixed = build_mixed_anndata(laser, result, min_residual_fraction=0.2)

    assert mixed.obs_names.tolist() == ["c1", "c2", "laser_residual_p1"]
    assert mixed.obs.loc["laser_residual_p1", "residual_fraction"] == pytest.approx(0.2)
    np.testing.assert_allclose(matrix(mixed[["laser_residual_p1"]]), [[2.0, 4.0]])


def test_residual_fraction_just_below_threshold_is_excluded() -> None:
    laser = spatial_laser_adata()[["p1"]].copy()
    area = 2_000_000_000_000_000
    occupied = 1_600_000_000_000_001
    laser.obs["area"] = float(area)
    accepted = accepted_table(("c1", 1, 1, "p1", occupied, area, occupied / area))
    result = quantification_result(spatial_laser_adata(), accepted)
    result_adata = result.adata[:, laser.var_names].copy()
    result = QuantificationResult(
        result_adata,
        result.overlaps,
        result.accepted,
        result.rejected,
        result.report,
    )

    mixed = build_mixed_anndata(laser, result, min_residual_fraction=0.2)

    assert mixed.obs_names.tolist() == ["c1", "c2"]


def test_large_integral_laser_area_straddles_threshold_without_float_narrowing() -> None:
    laser = spatial_laser_adata()[["p1"]].copy()
    area = 2**53 + 1
    occupied = 8_106_479_329_266_893
    laser.obs["area"] = pd.Series([area], index=laser.obs_names, dtype=object)
    accepted = accepted_table(
        ("c1", 1, 1, "p1", occupied, area, occupied / area)
    )
    result = quantification_result(spatial_laser_adata(), accepted)
    result_adata = result.adata[:, laser.var_names].copy()
    result = QuantificationResult(
        result_adata,
        result.overlaps,
        result.accepted,
        result.rejected,
        result.report,
    )

    mixed = build_mixed_anndata(laser, result, min_residual_fraction=0.1)

    assert mixed.obs_names.tolist() == ["c1", "c2", "laser_residual_p1"]
    expected_fraction = (area - occupied) / area
    assert mixed.obs.loc["laser_residual_p1", "residual_fraction"] == expected_fraction
    np.testing.assert_allclose(
        matrix(mixed[["laser_residual_p1"]]),
        np.array([[10.0, 20.0]]) * expected_fraction,
    )


def test_combined_label_residual_groups_pixels_by_exact_laser_obs_id() -> None:
    laser = spatial_laser_adata()[["p2"]].copy()
    accepted = accepted_table(
        ("c1", 1, 2, "p2", 3, 5, 0.6),
        ("c2", 2, 3, "p2", 7, 10, 0.7),
    )
    result = quantification_result(spatial_laser_adata(), accepted)
    result_adata = result.adata[:, laser.var_names].copy()
    result = QuantificationResult(
        result_adata,
        result.overlaps,
        result.accepted,
        result.rejected,
        result.report,
    )

    mixed = build_mixed_anndata(laser, result)

    assert mixed.obs.loc["laser_residual_p2", "residual_fraction"] == pytest.approx(0.5)
    np.testing.assert_allclose(matrix(mixed[["laser_residual_p2"]]), [[15.0, 20.0]])


def test_mixed_output_preserves_metadata_spatial_uns_csr_and_inputs() -> None:
    laser = spatial_laser_adata()
    accepted = accepted_table(
        ("c1", 1, 1, "p1", 8, 10, 0.8),
        ("c1", 1, 2, "p2", 3, 5, 0.6),
        ("c2", 2, 3, "p2", 7, 10, 0.7),
    )
    result = quantification_result(laser, accepted)
    laser_before = laser.copy()
    cells_before = result.adata.copy()
    accepted_before = result.accepted.copy(deep=True)
    overlaps_before = result.overlaps.copy(deep=True)
    rejected_before = result.rejected.copy(deep=True)
    report_before = deepcopy(result.report)

    mixed = build_mixed_anndata(laser, result)

    assert sparse.isspmatrix_csr(mixed.X)
    assert mixed.obs_names.tolist() == [
        "c1",
        "c2",
        "laser_residual_p1",
        "laser_residual_p2",
    ]
    assert mixed.obs_names.is_unique
    assert mixed.obs["data_type"].tolist() == [
        "cell",
        "cell",
        "laser_residual",
        "laser_residual",
    ]
    assert mixed.obs.loc["laser_residual_p1", "sample_region"] == "north"
    assert mixed.obs.loc["laser_residual_p2", "seg_label"] == "2_3"
    np.testing.assert_allclose(
        mixed.obsm["spatial"],
        [[1.5, 2.5], [3.5, 4.5], [7.0, 5.0], [8.0, 6.0]],
    )
    pd.testing.assert_frame_equal(mixed.var, result.adata.var)
    assert deepcopy(mixed.uns) == deepcopy(result.adata.uns)
    mixed.uns["nested"]["values"].append(3)
    assert result.adata.uns["nested"]["values"] == [1, 2]
    assert_adata_unchanged(laser, laser_before)
    assert_adata_unchanged(result.adata, cells_before)
    pd.testing.assert_frame_equal(result.accepted, accepted_before)
    pd.testing.assert_frame_equal(result.overlaps, overlaps_before)
    pd.testing.assert_frame_equal(result.rejected, rejected_before)
    assert result.report == report_before


def test_no_residual_rows_returns_independent_cell_copy_with_data_type() -> None:
    laser = spatial_laser_adata()[["p1"]].copy()
    accepted = accepted_table(("c1", 1, 1, "p1", 10, 10, 1.0))
    result = quantification_result(spatial_laser_adata(), accepted)
    cells = result.adata[:, laser.var_names].copy()
    result = QuantificationResult(
        cells, result.overlaps, result.accepted, result.rejected, result.report
    )

    mixed = build_mixed_anndata(laser, result)

    assert mixed is not cells
    assert mixed.obs_names.tolist() == ["c1", "c2"]
    assert mixed.obs["data_type"].tolist() == ["cell", "cell"]
    assert sparse.isspmatrix_csr(mixed.X)
    np.testing.assert_array_equal(mixed.obsm["spatial"], cells.obsm["spatial"])
    pd.testing.assert_frame_equal(mixed.var, cells.var)
    assert deepcopy(mixed.uns) == deepcopy(cells.uns)
    assert "data_type" not in cells.obs


@pytest.mark.parametrize("value", [True, False, -0.01, 1.01, np.nan, np.inf, "0.2"])
def test_build_mixed_anndata_validates_min_residual_fraction(value: object) -> None:
    laser = spatial_laser_adata()
    result = quantification_result(laser, accepted_table())

    with pytest.raises(QuantificationError, match="min_residual_fraction"):
        build_mixed_anndata(laser, result, min_residual_fraction=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True, "10"])
def test_build_mixed_anndata_requires_positive_finite_numeric_laser_area(value: object) -> None:
    laser = spatial_laser_adata()
    laser.obs["area"] = laser.obs["area"].astype(object)
    laser.obs.loc["p1", "area"] = value
    result = quantification_result(spatial_laser_adata(), accepted_table())

    with pytest.raises(QuantificationError, match="area"):
        build_mixed_anndata(laser, result)


def test_build_mixed_anndata_validates_input_and_result_types() -> None:
    laser = spatial_laser_adata()
    result = quantification_result(laser, accepted_table())

    with pytest.raises(QuantificationError, match="laser_adata"):
        build_mixed_anndata(object(), result)  # type: ignore[arg-type]
    with pytest.raises(QuantificationError, match="QuantificationResult"):
        build_mixed_anndata(laser, object())  # type: ignore[arg-type]

    malformed = QuantificationResult(
        adata=object(),  # type: ignore[arg-type]
        overlaps=result.overlaps,
        accepted=result.accepted,
        rejected=result.rejected,
        report=result.report,
    )
    with pytest.raises(QuantificationError, match="quantification.adata"):
        build_mixed_anndata(laser, malformed)


@pytest.mark.parametrize("field", ["overlaps", "rejected", "report"])
def test_build_mixed_anndata_validates_quantification_result_fields(field: str) -> None:
    laser = spatial_laser_adata()
    result = quantification_result(laser, accepted_table())
    values = {
        "adata": result.adata,
        "overlaps": result.overlaps,
        "accepted": result.accepted,
        "rejected": result.rejected,
        "report": result.report,
    }
    values[field] = object()
    malformed = QuantificationResult(**values)  # type: ignore[arg-type]

    with pytest.raises(QuantificationError, match=field):
        build_mixed_anndata(laser, malformed)


@pytest.mark.parametrize("field", ["overlaps", "rejected"])
def test_build_mixed_anndata_validates_quantification_table_columns(field: str) -> None:
    laser = spatial_laser_adata()
    result = quantification_result(laser, accepted_table())
    values = {
        "adata": result.adata,
        "overlaps": result.overlaps,
        "accepted": result.accepted,
        "rejected": result.rejected,
        "report": result.report,
    }
    values[field] = values[field].drop(columns="laser_obs_id")  # type: ignore[union-attr]
    malformed = QuantificationResult(**values)  # type: ignore[arg-type]

    with pytest.raises(QuantificationError, match=field):
        build_mixed_anndata(laser, malformed)


def test_build_mixed_anndata_rejects_accepted_rows_incoherent_with_overlaps() -> None:
    laser = anndata.AnnData(
        sparse.csr_matrix([[10.0, 20.0]]),
        obs={
            "seg_label": ["1"],
            "area": [4.0],
            "centroid-0": [0.5],
            "centroid-1": [0.5],
        },
        var=pd.DataFrame({"mz": [100.0, 200.0]}, index=["feature-100", "feature-200"]),
    )
    laser.obs_names = ["p1"]
    labels = np.ones((2, 2), dtype=np.int32)
    result = quantify_cells(
        laser,
        labels,
        labels,
        method="legacy_proportional",
        boundary_margin=10,
    )
    mutated_accepted = result.accepted.copy(deep=True)
    mutated_accepted.loc[:, "occupied_pixels"] = 0
    mutated_accepted.loc[:, "occupied_ratio"] = 0.0
    malformed = QuantificationResult(
        result.adata,
        result.overlaps,
        mutated_accepted,
        result.rejected,
        result.report,
    )

    with pytest.raises(QuantificationError, match="partition|coherent"):
        build_mixed_anndata(laser, malformed)


@pytest.mark.parametrize(
    "accepted",
    [
        accepted_table(("c1", 1, 1, "p1", 1, 10, 0.9)),
        accepted_table(("c1", 2, 1, "p1", 1, 10, 0.1)),
    ],
)
def test_build_mixed_anndata_rejects_incoherent_overlap_row_values(
    accepted: pd.DataFrame,
) -> None:
    laser = spatial_laser_adata()
    malformed = quantification_result(laser, accepted)

    with pytest.raises(QuantificationError, match="occupied_ratio|cell_label"):
        build_mixed_anndata(laser, malformed)


def test_build_mixed_anndata_validates_required_metadata_and_unique_names() -> None:
    laser = spatial_laser_adata()
    result = quantification_result(laser, accepted_table())

    missing = laser.copy()
    missing.obs = missing.obs.drop(columns="centroid-1")
    with pytest.raises(QuantificationError, match="missing obs columns"):
        build_mixed_anndata(missing, result)

    duplicate_laser = laser.copy()
    duplicate_laser.obs_names = ["p1", "p1"]
    with pytest.raises(QuantificationError, match="unique observation"):
        build_mixed_anndata(duplicate_laser, result)

    duplicate_cells = result.adata.copy()
    duplicate_cells.obs_names = ["c1", "c1"]
    malformed = QuantificationResult(
        duplicate_cells,
        result.overlaps,
        result.accepted,
        result.rejected,
        result.report,
    )
    with pytest.raises(QuantificationError, match="unique observation"):
        build_mixed_anndata(laser, malformed)


@pytest.mark.parametrize(
    "accepted",
    [
        object(),
        accepted_table(("c1", 1, 1, "p1", 1, 10, 0.1)).drop(columns="occupied_pixels"),
        accepted_table(("c1", 1, 1, "P1", 1, 10, 0.1)),
        accepted_table(("c1", 1, 1, 1, 1, 10, 0.1)),
        accepted_table((1, 1, 1, "p1", 1, 10, 0.1)),
        accepted_table(("missing", 1, 1, "p1", 1, 10, 0.1)),
        accepted_table(("c1", True, 1, "p1", 1, 10, 0.1)),
        accepted_table(("c1", 1.5, 1, "p1", 1, 10, 0.1)),
        accepted_table(("c1", 0, 1, "p1", 1, 10, 0.1)),
        accepted_table(("c1", 1, True, "p1", 1, 10, 0.1)),
        accepted_table(("c1", 1, 1.5, "p1", 1, 10, 0.1)),
        accepted_table(("c1", 1, 0, "p1", 1, 10, 0.1)),
        accepted_table(("c1", 1, 2, "p1", 1, 10, 0.1)),
        accepted_table(("c1", 1, 1, "p1", True, 10, 0.1)),
        accepted_table(("c1", 1, 1, "p1", 1.5, 10, 0.1)),
        accepted_table(("c1", 1, 1, "p1", -1, 10, 0.1)),
        accepted_table(("c1", 1, 1, "p1", 11, 10, 1.0)),
        accepted_table(("c1", 1, 1, "p1", 1, True, 0.1)),
        accepted_table(("c1", 1, 1, "p1", 1, 10.5, 0.1)),
        accepted_table(("c1", 1, 1, "p1", 1, 0, 0.1)),
        accepted_table(("c1", 1, 1, "p1", 1, 10, np.nan)),
        accepted_table(("c1", 1, 1, "p1", 1, 10, True)),
        accepted_table(("c1", 1, 1, "p1", 1, 10, 1.1)),
        pd.concat(
            [
                accepted_table(("c1", 1, 1, "p1", 1, 10, 0.1)),
                accepted_table(("c1", 1, 1, "p1", 1, 10, 0.1)),
            ],
            ignore_index=True,
        ),
    ],
)
def test_build_mixed_anndata_validates_accepted_table(accepted: object) -> None:
    laser = spatial_laser_adata()
    valid = quantification_result(laser, accepted_table())
    malformed = QuantificationResult(
        valid.adata,
        valid.overlaps,
        accepted,  # type: ignore[arg-type]
        valid.rejected,
        valid.report,
    )

    with pytest.raises(QuantificationError, match="accepted|Accepted"):
        build_mixed_anndata(laser, malformed)


@pytest.mark.parametrize("mismatch", ["count", "order", "identity", "mz", "duplicate"])
def test_build_mixed_anndata_requires_exact_feature_compatibility(mismatch: str) -> None:
    laser = spatial_laser_adata()
    result = quantification_result(laser, accepted_table())
    cells = result.adata.copy()
    if mismatch == "count":
        cells = cells[:, :1].copy()
    elif mismatch == "order":
        cells = cells[:, [1, 0]].copy()
    elif mismatch == "identity":
        cells.var_names = ["feature-100", "other-feature"]
    elif mismatch == "mz":
        cells.var["mz"] = [100.0, 201.0]
    else:
        cells.var_names = ["feature-100", "feature-100"]
    malformed = QuantificationResult(
        cells, result.overlaps, result.accepted, result.rejected, result.report
    )

    with pytest.raises(QuantificationError, match="features|variable"):
        build_mixed_anndata(laser, malformed)


@pytest.mark.parametrize(
    ("laser_mz", "cell_mz"),
    [
        (2**53, 2**53 + 1),
        (2**53 + 1, np.float64(2**53)),
        (np.uint64(2**53 + 1), np.float64(2**53)),
        (np.float32(0.1), np.float64(0.1)),
        (Fraction(1, 10), np.float64(0.1)),
        (100.0, np.nextafter(100.0, np.inf)),
    ],
)
def test_build_mixed_anndata_compares_mz_identity_without_narrowing_or_tolerance(
    laser_mz: object, cell_mz: object
) -> None:
    laser = spatial_laser_adata()
    laser.var["mz"] = pd.Series(
        [laser_mz, 200.0], index=laser.var_names, dtype=object
    )
    result = quantification_result(laser, accepted_table())
    cells = result.adata.copy()
    cells.var["mz"] = pd.Series(
        [cell_mz, 200.0], index=cells.var_names, dtype=object
    )
    malformed = QuantificationResult(
        cells, result.overlaps, result.accepted, result.rejected, result.report
    )

    with pytest.raises(QuantificationError, match="features|identifiers"):
        build_mixed_anndata(laser, malformed)


@pytest.mark.parametrize(
    ("laser_mz", "cell_mz"),
    [
        (100, np.float64(100.0)),
        (np.float32(0.1), np.float64(float(np.float32(0.1)))),
        (Fraction(1, 10), Fraction(2, 20)),
    ],
)
def test_build_mixed_anndata_accepts_exact_mixed_type_mz_identity(
    laser_mz: object, cell_mz: object
) -> None:
    laser = spatial_laser_adata()
    laser.var["mz"] = pd.Series(
        [laser_mz, 200.0], index=laser.var_names, dtype=object
    )
    result = quantification_result(laser, accepted_table())
    cells = result.adata.copy()
    cells.var["mz"] = pd.Series(
        [cell_mz, 200.0], index=cells.var_names, dtype=object
    )
    mixed = QuantificationResult(
        cells, result.overlaps, result.accepted, result.rejected, result.report
    )

    output = build_mixed_anndata(laser, mixed)

    assert output.n_vars == laser.n_vars


@pytest.mark.parametrize("categorical", [False, True])
def test_build_mixed_anndata_preserves_large_integral_cell_labels_from_quantification(
    categorical: bool,
) -> None:
    cell_label = 2**53 + 1
    laser = anndata.AnnData(
        sparse.csr_matrix([[10.0]]),
        obs={
            "seg_label": ["1"],
            "area": [1],
            "centroid-0": [0.0],
            "centroid-1": [0.0],
        },
        var=pd.DataFrame({"mz": [100.0]}, index=["feature-100"]),
    )
    laser.obs_names = ["p1"]
    cells = np.array([[cell_label]], dtype=np.uint64)
    lasers = np.array([[1]], dtype=np.uint64)
    result = quantify_cells(
        laser,
        cells,
        lasers,
        method="legacy_proportional",
        boundary_margin=10,
    )
    if categorical:
        cell_adata = result.adata.copy()
        cell_adata.obs["label"] = pd.Categorical(cell_adata.obs["label"])
        result = QuantificationResult(
            cell_adata,
            result.overlaps,
            result.accepted,
            result.rejected,
            result.report,
        )

    mixed = build_mixed_anndata(laser, result)

    assert mixed.obs_names.tolist() == [f"c{cell_label}"]
    assert mixed.obs["label"].tolist() == [cell_label]
    assert mixed.obs["data_type"].tolist() == ["cell"]


def test_build_mixed_anndata_rejects_residual_identifier_collision() -> None:
    laser = spatial_laser_adata()
    result = quantification_result(laser, accepted_table())
    cells = result.adata.copy()
    cells.obs_names = ["laser_residual_p1", "c2"]
    malformed = QuantificationResult(
        cells, result.overlaps, result.accepted, result.rejected, result.report
    )

    with pytest.raises(QuantificationError, match="collision|unique"):
        build_mixed_anndata(laser, malformed)


def test_build_mixed_anndata_validates_cell_spatial_coordinates() -> None:
    laser = spatial_laser_adata()
    result = quantification_result(laser, accepted_table())
    cells = result.adata.copy()
    del cells.obsm["spatial"]
    malformed = QuantificationResult(
        cells, result.overlaps, result.accepted, result.rejected, result.report
    )

    with pytest.raises(QuantificationError, match="spatial"):
        build_mixed_anndata(laser, malformed)


def test_spatial_core_runs_registration_quantification_and_mixed_output() -> None:
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


def test_joint_exports_exact_stable_spatial_api() -> None:
    expected_api = {
        "JointConfig",
        "JointPipeline",
        "PipelineResult",
        "QuantificationResult",
        "RegistrationResult",
        "SegmentationResult",
        "__version__",
        "annotate_hmdb",
        "annotate_metaboscape",
        "build_mixed_anndata",
        "coverage_report",
        "load_config",
        "preprocess_msi",
        "quantify_cells",
        "read_h5ad",
        "read_imzml",
        "register_laser_points",
        "registration_report",
        "segment_laser_marks",
        "simulate_laser_marks",
        "spectral_qc",
        "write_results",
    }

    assert set(joint.__all__) == expected_api
    assert all(
        callable(getattr(joint, name))
        for name in expected_api
        if name not in {"__version__"}
    )
