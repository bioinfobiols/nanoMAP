from copy import deepcopy

import anndata
import numpy as np
import pandas as pd
import pytest

from joint.config import RegistrationConfig
from joint.errors import RegistrationError
from joint.registration import mount_spatial_images, register_laser_points


def grid_adata(columns: int = 3, rows: int = 2) -> anndata.AnnData:
    coords = np.array(
        [(x, y) for y in range(1, rows + 1) for x in range(1, columns + 1)],
        dtype=float,
    )
    adata = anndata.AnnData(
        np.arange(len(coords) * 2, dtype=float).reshape(len(coords), 2),
        obsm={"spatial": coords},
    )
    adata.obs_names = [f"x{int(x)}y{int(y)}" for x, y in coords]
    adata.obs["dataset"] = "sample"
    return adata


def laser_regions(columns: int = 3, rows: int = 2) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    label = 1
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            records.append(
                {
                    "label": label,
                    "seg_label": str(label),
                    "area": 20.0,
                    "centroid-0": float(row * 10),
                    "centroid-1": float(column * 10),
                    "row_number": row,
                    "column_number": column,
                    "morphology": "intact",
                }
            )
            label += 1
    return pd.DataFrame.from_records(records)


def test_registration_config_exposes_complete_orientation_and_edge_policy_schema():
    assert RegistrationConfig().orientation == "auto"
    assert RegistrationConfig().edge_policy == "error"
    assert RegistrationConfig(orientation="flip_xy").orientation == "flip_xy"
    assert RegistrationConfig(orientation="swap_flip_xy").orientation == "swap_flip_xy"
    assert RegistrationConfig(edge_policy="truncate_left").edge_policy == "truncate_left"
    assert RegistrationConfig(edge_policy="truncate_right").edge_policy == "truncate_right"


@pytest.mark.parametrize(
    ("orientation", "laser_columns", "laser_rows", "expected_pixels"),
    [
        (
            "identity",
            3,
            2,
            ["x1y1", "x2y1", "x3y1", "x1y2", "x2y2", "x3y2"],
        ),
        (
            "flip_x",
            3,
            2,
            ["x3y1", "x2y1", "x1y1", "x3y2", "x2y2", "x1y2"],
        ),
        (
            "flip_y",
            3,
            2,
            ["x1y2", "x2y2", "x3y2", "x1y1", "x2y1", "x3y1"],
        ),
        (
            "flip_xy",
            3,
            2,
            ["x3y2", "x2y2", "x1y2", "x3y1", "x2y1", "x1y1"],
        ),
        (
            "swap_xy",
            2,
            3,
            ["x1y1", "x1y2", "x2y1", "x2y2", "x3y1", "x3y2"],
        ),
        (
            "swap_flip_x",
            2,
            3,
            ["x1y2", "x1y1", "x2y2", "x2y1", "x3y2", "x3y1"],
        ),
        (
            "swap_flip_y",
            2,
            3,
            ["x3y1", "x3y2", "x2y1", "x2y2", "x1y1", "x1y2"],
        ),
        (
            "swap_flip_xy",
            2,
            3,
            ["x3y2", "x3y1", "x2y2", "x2y1", "x1y2", "x1y1"],
        ),
    ],
)
def test_registration_handles_every_orientation_on_non_square_grid(
    orientation: str,
    laser_columns: int,
    laser_rows: int,
    expected_pixels: list[str],
):
    result = register_laser_points(
        grid_adata(),
        laser_regions(columns=laser_columns, rows=laser_rows),
        orientation=orientation,
    )

    assert result.mapping["pixel_id"].tolist() == expected_pixels
    assert result.adata.obs_names.tolist() == expected_pixels
    assert result.adata.obs["seg_label"].tolist() == result.mapping["seg_label"].tolist()
    expected_spatial = result.mapping[["centroid-1", "centroid-0"]].to_numpy()
    np.testing.assert_array_equal(result.adata.obsm["spatial"], expected_spatial)
    assert result.transform == {"orientation": orientation, "edge_policy": "error"}


@pytest.mark.parametrize(
    ("edge_policy", "expected_pixels"),
    [
        (
            "truncate_left",
            ["x2y1", "x3y1", "x4y1", "x2y2", "x3y2", "x4y2"],
        ),
        (
            "truncate_right",
            ["x1y1", "x2y1", "x3y1", "x1y2", "x2y2", "x3y2"],
        ),
    ],
)
def test_explicit_truncation_records_exact_counts(
    edge_policy: str, expected_pixels: list[str]
):
    result = register_laser_points(
        grid_adata(columns=4),
        laser_regions(columns=3),
        orientation="identity",
        edge_policy=edge_policy,
    )

    assert result.mapping["pixel_id"].tolist() == expected_pixels
    assert result.adata.n_obs == 6
    assert result.report == {
        "registered_pixels": 6,
        "truncated_pixels": 2,
        "truncated_pixels_by_row": {1: 1, 2: 1},
    }


def test_length_mismatch_requires_explicit_truncation():
    with pytest.raises(RegistrationError, match="explicit edge_policy"):
        register_laser_points(
            grid_adata(columns=4),
            laser_regions(columns=3),
            orientation="identity",
        )


@pytest.mark.parametrize("edge_policy", ["error", "truncate_left", "truncate_right"])
def test_registration_never_truncates_laser_regions(edge_policy: str):
    with pytest.raises(RegistrationError, match="more points"):
        register_laser_points(
            grid_adata(columns=2),
            laser_regions(columns=3),
            orientation="identity",
            edge_policy=edge_policy,
        )


def test_auto_orientation_raises_when_multiple_candidates_fit():
    with pytest.raises(RegistrationError, match="ambiguous.*identity"):
        register_laser_points(grid_adata(), laser_regions(), orientation="auto")


def test_auto_orientation_reports_when_no_candidate_fits():
    with pytest.raises(RegistrationError, match="no valid candidate"):
        register_laser_points(
            grid_adata(), laser_regions(columns=4, rows=4), orientation="auto"
        )


def test_registration_excludes_spoilt_audit_rows_without_mutating_inputs():
    adata = grid_adata()
    regions = laser_regions()
    spoilt = regions.iloc[[0]].copy()
    spoilt["seg_label"] = "audit-source"
    spoilt["morphology"] = "spoilt"
    regions = pd.concat([regions, spoilt], ignore_index=True)
    original_spatial = adata.obsm["spatial"].copy()
    original_obs = adata.obs.copy(deep=True)
    original_regions = regions.copy(deep=True)

    result = register_laser_points(adata, regions, orientation="identity")

    assert "audit-source" not in result.mapping["seg_label"].tolist()
    assert set(result.mapping["morphology"]) == {"intact"}
    np.testing.assert_array_equal(adata.obsm["spatial"], original_spatial)
    pd.testing.assert_frame_equal(adata.obs, original_obs)
    pd.testing.assert_frame_equal(regions, original_regions)


def test_registration_accepts_combined_and_simulated_active_regions():
    regions = laser_regions()
    regions.loc[0, "morphology"] = "combined"
    regions.loc[1, "morphology"] = "simulated"

    result = register_laser_points(grid_adata(), regions, orientation="identity")

    assert result.mapping["morphology"].iloc[:2].tolist() == ["combined", "simulated"]


def test_registration_accepts_categorical_identifiers_morphology_and_dataset():
    adata = grid_adata()
    adata.obs["dataset"] = pd.Categorical(adata.obs["dataset"])
    regions = laser_regions()
    regions["seg_label"] = pd.Categorical(regions["seg_label"])
    regions["morphology"] = pd.Categorical(regions["morphology"])

    result = register_laser_points(adata, regions, orientation="identity")

    assert result.mapping["seg_label"].tolist() == ["1", "2", "3", "4", "5", "6"]
    assert result.mapping["morphology"].tolist() == ["intact"] * 6
    assert result.adata.obs["seg_label"].tolist() == ["1", "2", "3", "4", "5", "6"]


def test_registration_accepts_csv_inferred_integral_seg_labels(tmp_path):
    path = tmp_path / "laser-regions.csv"
    laser_regions().to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert pd.api.types.is_integer_dtype(reloaded["seg_label"])

    result = register_laser_points(grid_adata(), reloaded, orientation="identity")

    assert result.mapping["seg_label"].tolist() == ["1", "2", "3", "4", "5", "6"]
    assert result.adata.obs["seg_label"].tolist() == ["1", "2", "3", "4", "5", "6"]


def test_registration_preserves_large_integral_seg_label_exactly():
    regions = laser_regions()
    regions["seg_label"] = regions["seg_label"].astype(object)
    regions.loc[0, "seg_label"] = 2**63 + 1

    result = register_laser_points(grid_adata(), regions, orientation="identity")

    assert result.mapping.loc[0, "seg_label"] == str(2**63 + 1)
    assert result.adata.obs["seg_label"].iloc[0] == str(2**63 + 1)


@pytest.mark.parametrize(
    "seg_label", [None, pd.NA, True, 1.5, np.inf, -np.inf, "", "   "]
)
def test_registration_rejects_invalid_seg_label_identifiers(seg_label: object):
    regions = laser_regions()
    regions["seg_label"] = regions["seg_label"].astype(object)
    regions.loc[0, "seg_label"] = seg_label

    with pytest.raises(RegistrationError, match="seg_label"):
        register_laser_points(grid_adata(), regions, orientation="identity")


def test_registration_converts_invalid_categorical_values_to_registration_errors():
    invalid_identifier = laser_regions()
    invalid_identifier["seg_label"] = pd.Categorical(
        [1.5, "2", "3", "4", "5", "6"]
    )
    with pytest.raises(RegistrationError, match="seg_label"):
        register_laser_points(grid_adata(), invalid_identifier, orientation="identity")

    invalid_morphology = laser_regions()
    invalid_morphology["morphology"] = pd.Categorical([1, 1, 1, 1, 1, 1])
    with pytest.raises(RegistrationError, match="morphology"):
        register_laser_points(grid_adata(), invalid_morphology, orientation="identity")


@pytest.mark.parametrize("orientation", ["diagonal", "", None, 7])
def test_registration_rejects_invalid_orientation(orientation: object):
    with pytest.raises(RegistrationError, match="orientation"):
        register_laser_points(
            grid_adata(), laser_regions(), orientation=orientation  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("edge_policy", ["truncate", "", None, 7])
def test_registration_rejects_invalid_edge_policy(edge_policy: object):
    with pytest.raises(RegistrationError, match="edge_policy"):
        register_laser_points(
            grid_adata(),
            laser_regions(),
            orientation="identity",
            edge_policy=edge_policy,  # type: ignore[arg-type]
        )


def test_registration_requires_anndata_and_dataframe_inputs():
    with pytest.raises(RegistrationError, match="AnnData"):
        register_laser_points(  # type: ignore[arg-type]
            object(), laser_regions(), orientation="identity"
        )
    with pytest.raises(RegistrationError, match="DataFrame"):
        register_laser_points(  # type: ignore[arg-type]
            grid_adata(), [], orientation="identity"
        )


def test_registration_rejects_missing_or_malformed_spatial_coordinates():
    missing = anndata.AnnData(np.ones((2, 2)))
    with pytest.raises(RegistrationError, match="spatial"):
        register_laser_points(missing, laser_regions(columns=1), orientation="identity")

    wrong_shape = grid_adata()
    wrong_shape.obsm["spatial"] = np.ones((wrong_shape.n_obs, 3))
    with pytest.raises(RegistrationError, match="n_obs by 2"):
        register_laser_points(wrong_shape, laser_regions(), orientation="identity")

    non_numeric = grid_adata()
    non_numeric.obsm["spatial"] = non_numeric.obsm["spatial"].astype(str)
    non_numeric.obsm["spatial"][0, 0] = "not-a-number"
    with pytest.raises(RegistrationError, match="numeric"):
        register_laser_points(non_numeric, laser_regions(), orientation="identity")

    numeric_strings = grid_adata()
    numeric_strings.obsm["spatial"] = numeric_strings.obsm["spatial"].astype(str)
    with pytest.raises(RegistrationError, match="numeric"):
        register_laser_points(numeric_strings, laser_regions(), orientation="identity")

    booleans = grid_adata()
    booleans.obsm["spatial"] = booleans.obsm["spatial"].astype(bool)
    with pytest.raises(RegistrationError, match="numeric"):
        register_laser_points(booleans, laser_regions(), orientation="identity")

    non_finite = grid_adata()
    non_finite.obsm["spatial"][0, 0] = np.nan
    with pytest.raises(RegistrationError, match="finite"):
        register_laser_points(non_finite, laser_regions(), orientation="identity")


def test_registration_rejects_non_unique_observation_ids_and_coordinates():
    duplicate_ids = grid_adata()
    duplicate_ids.obs_names = ["duplicate"] * duplicate_ids.n_obs
    with pytest.raises(RegistrationError, match="unique observation identifiers"):
        register_laser_points(duplicate_ids, laser_regions(), orientation="identity")

    duplicate_coordinates = grid_adata()
    duplicate_coordinates.obsm["spatial"][1] = duplicate_coordinates.obsm["spatial"][0]
    with pytest.raises(RegistrationError, match="unique coordinate pairs"):
        register_laser_points(duplicate_coordinates, laser_regions(), orientation="identity")


def test_registration_rejects_mixed_or_missing_dataset_values():
    mixed = grid_adata()
    mixed.obs["dataset"] = ["a", "a", "a", "b", "b", "b"]
    with pytest.raises(RegistrationError, match="single dataset"):
        register_laser_points(mixed, laser_regions(), orientation="identity")

    missing = grid_adata()
    missing.obs.loc[missing.obs.index[0], "dataset"] = None
    with pytest.raises(RegistrationError, match="single dataset"):
        register_laser_points(missing, laser_regions(), orientation="identity")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("area", np.inf, "area.*finite"),
        ("area", -1, "area.*non-negative"),
        ("centroid-0", "bad", "centroid-0.*numeric"),
        ("centroid-1", np.nan, "centroid-1.*finite"),
        ("row_number", 1.5, "row_number.*positive integer"),
        ("column_number", 0, "column_number.*positive integer"),
        ("seg_label", None, "seg_label.*missing"),
        ("morphology", "", "morphology.*non-empty"),
        ("area", "20", "area.*numeric"),
        ("row_number", True, "row_number.*numeric"),
    ],
)
def test_registration_validates_required_laser_column_values(
    column: str, value: object, message: str
):
    regions = laser_regions()
    regions[column] = regions[column].astype(object)
    regions.loc[0, column] = value

    with pytest.raises(RegistrationError, match=message):
        register_laser_points(grid_adata(), regions, orientation="identity")


def test_registration_rejects_missing_columns_duplicate_positions_and_no_active_rows():
    with pytest.raises(RegistrationError, match="missing columns.*area"):
        register_laser_points(
            grid_adata(), laser_regions().drop(columns="area"), orientation="identity"
        )

    duplicate = laser_regions()
    duplicate.loc[1, ["row_number", "column_number"]] = duplicate.loc[
        0, ["row_number", "column_number"]
    ]
    with pytest.raises(RegistrationError, match="unique active.*row_number.*column_number"):
        register_laser_points(grid_adata(), duplicate, orientation="identity")

    spoilt = laser_regions()
    spoilt["morphology"] = "spoilt"
    with pytest.raises(RegistrationError, match="no active"):
        register_laser_points(grid_adata(), spoilt, orientation="identity")


def test_registration_rejects_metadata_columns_that_collide_with_observations():
    regions = laser_regions()
    regions["dataset"] = "laser"

    with pytest.raises(RegistrationError, match="overlap.*dataset"):
        register_laser_points(grid_adata(), regions, orientation="identity")


def test_registration_mapping_and_metadata_join_are_one_to_one_and_ordered():
    adata = grid_adata()
    adata.obs["batch"] = np.arange(adata.n_obs)
    result = register_laser_points(adata, laser_regions(), orientation="flip_xy")

    assert result.mapping["pixel_id"].is_unique
    assert result.adata.obs_names.tolist() == result.mapping["pixel_id"].tolist()
    assert result.adata.n_obs == len(result.mapping)
    assert result.adata.obs["batch"].tolist() == [5, 4, 3, 2, 1, 0]
    assert result.adata.obs["seg_label"].tolist() == result.mapping["seg_label"].tolist()


def test_mount_spatial_images_returns_copy_and_does_not_alias_caller_arrays():
    adata = grid_adata()
    original_uns = deepcopy(adata.uns)
    laser = np.zeros((20, 30, 3), dtype=np.uint8)
    cells = np.ones((20, 30), dtype=float)

    mounted = mount_spatial_images(
        adata,
        laser_image=laser,
        cell_image=cells,
        library_id="sample",
        spot_diameter=2.5,
    )

    assert mounted is not adata
    assert adata.uns == original_uns
    assert mounted.uns["spatial"]["sample"]["scalefactors"] == {
        "tissue_hires_scalef": 1.0,
        "spot_diameter_fullres": 2.5,
    }
    mounted_laser = mounted.uns["spatial"]["sample"]["images"]["laser"]
    mounted_cells = mounted.uns["spatial"]["sample"]["images"]["hires"]
    assert not np.shares_memory(mounted_laser, laser)
    assert not np.shares_memory(mounted_cells, cells)
    laser[0, 0] = 9
    mounted_cells[0, 0] = 8
    assert mounted_laser[0, 0].tolist() == [0, 0, 0]
    assert cells[0, 0] == 1


def test_mount_spatial_images_without_images_still_returns_copy():
    adata = grid_adata()
    mounted = mount_spatial_images(adata, library_id="sample")

    assert mounted is not adata
    assert "spatial" not in mounted.uns


@pytest.mark.parametrize("library_id", ["", "   ", None, 3])
def test_mount_spatial_images_validates_library_id(library_id: object):
    with pytest.raises(RegistrationError, match="library_id"):
        mount_spatial_images(
            grid_adata(),
            laser_image=np.zeros((2, 2)),
            library_id=library_id,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("spot_diameter", [0, -1, np.inf, np.nan, True, "wide"])
def test_mount_spatial_images_validates_spot_diameter(spot_diameter: object):
    with pytest.raises(RegistrationError, match="spot_diameter"):
        mount_spatial_images(
            grid_adata(),
            laser_image=np.zeros((2, 2)),
            spot_diameter=spot_diameter,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "image",
    [
        np.array([]),
        np.zeros((2,)),
        np.zeros((1, 1, 1, 1)),
        np.array([[np.nan]]),
        np.array([["not-numeric"]]),
    ],
)
def test_mount_spatial_images_validates_image_arrays(image: np.ndarray):
    with pytest.raises(RegistrationError, match="image"):
        mount_spatial_images(grid_adata(), laser_image=image)


def test_mount_spatial_images_requires_matching_height_and_width():
    with pytest.raises(RegistrationError, match="identical height and width"):
        mount_spatial_images(
            grid_adata(),
            laser_image=np.zeros((20, 30, 3)),
            cell_image=np.zeros((30, 20)),
        )


def test_mount_spatial_images_requires_anndata_input():
    with pytest.raises(RegistrationError, match="AnnData"):
        mount_spatial_images(  # type: ignore[arg-type]
            object(), laser_image=np.zeros((2, 2))
        )
