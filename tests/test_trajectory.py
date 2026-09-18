import builtins
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from joint.errors import InputFormatError, OptionalDependencyError
from joint.trajectory import calculate_feature_trends, fit_spatial_trajectory


def trajectory_adata() -> anndata.AnnData:
    return anndata.AnnData(
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        obs=pd.DataFrame(index=["cell-0", "cell-1", "cell-2"]),
        var=pd.DataFrame(index=["f0", "f1"]),
        obsm={"spatial": np.array([[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]])},
    )


def projected_adata() -> anndata.AnnData:
    return fit_spatial_trajectory(trajectory_adata(), np.array([[0.0, 0.0], [10.0, 0.0]]))


def assert_adata_unchanged(actual: anndata.AnnData, expected: anndata.AnnData) -> None:
    actual_x = actual.X.toarray() if sparse.issparse(actual.X) else np.asarray(actual.X)
    expected_x = expected.X.toarray() if sparse.issparse(expected.X) else np.asarray(expected.X)
    np.testing.assert_array_equal(actual_x, expected_x)
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    assert actual.uns == expected.uns
    assert actual.obsm.keys() == expected.obsm.keys()
    for key in actual.obsm:
        np.testing.assert_array_equal(actual.obsm[key], expected.obsm[key])
    assert actual.layers.keys() == expected.layers.keys()
    for key in actual.layers:
        actual_layer = (
            actual.layers[key].toarray()
            if sparse.issparse(actual.layers[key])
            else np.asarray(actual.layers[key])
        )
        expected_layer = (
            expected.layers[key].toarray()
            if sparse.issparse(expected.layers[key])
            else np.asarray(expected.layers[key])
        )
        np.testing.assert_array_equal(actual_layer, expected_layer)


def install_fake_pygam(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prediction: Any = None,
    intervals: Any = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    module = ModuleType("pygam")

    def fake_s(index: int) -> tuple[str, int]:
        return ("s", index)

    class FakeLinearGAM:
        def __init__(self, term: object) -> None:
            self.term = term
            self.y = np.array([], dtype=float)

        def fit(self, x: np.ndarray, y: np.ndarray) -> "FakeLinearGAM":
            self.y = np.asarray(y).copy()
            calls.append({"term": self.term, "x": np.asarray(x).copy(), "y": self.y.copy()})
            return self

        def predict(self, grid: np.ndarray) -> Any:
            if callable(prediction):
                return prediction(grid, self.y)
            if prediction is not None:
                return prediction
            return np.full(len(grid), float(np.mean(self.y)))

        def prediction_intervals(self, grid: np.ndarray, *, width: float) -> Any:
            calls[-1]["grid"] = np.asarray(grid).copy()
            calls[-1]["width"] = width
            if callable(intervals):
                return intervals(grid, self.y)
            if intervals is not None:
                return intervals
            center = np.full(len(grid), float(np.mean(self.y)))
            return np.column_stack([center - 0.25, center + 0.25])

    module.LinearGAM = FakeLinearGAM  # type: ignore[attr-defined]
    module.s = fake_s  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pygam", module)
    return calls


def test_fit_spatial_trajectory_projects_points_to_path_order_without_mutation() -> None:
    source = trajectory_adata()
    source.uns["joint"] = {"existing": {"value": 7}}
    before = source.copy()

    result = fit_spatial_trajectory(source, np.array([[0.0, 0.0], [10.0, 0.0]]))

    assert result is not source
    assert result.obs["trajectory_position"].tolist() == [0.0, 0.5, 1.0]
    assert result.obs["trajectory_distance"].tolist() == [0.0, 1.0, 0.0]
    assert result.uns["joint"]["existing"] == {"value": 7}
    assert result.uns["joint"]["trajectory_path"] == [[0.0, 0.0], [10.0, 0.0]]
    assert result.uns["joint"]["trajectory"] == {
        "spatial_key": "spatial",
        "position_key": "trajectory_position",
        "distance_key": "trajectory_distance",
        "segment_lengths": [10.0],
        "total_length": 10.0,
        "projection_tie_breaker": "first segment in path order",
    }
    assert_adata_unchanged(source, before)


def test_fit_spatial_trajectory_breaks_corner_ties_by_first_segment() -> None:
    adata = anndata.AnnData(
        np.ones((2, 1)),
        obsm={"spatial": np.array([[1.0, 1.0], [2.0, 0.0]])},
    )
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    first = fit_spatial_trajectory(adata, path)
    second = fit_spatial_trajectory(adata, path)

    assert first.obs["trajectory_position"].tolist() == [0.25, 0.5]
    assert first.obs["trajectory_distance"].tolist() == [1.0, 0.0]
    np.testing.assert_array_equal(
        first.obs[["trajectory_position", "trajectory_distance"]],
        second.obs[["trajectory_position", "trajectory_distance"]],
    )


def test_fit_spatial_trajectory_breaks_self_intersection_ties_by_first_segment() -> None:
    adata = anndata.AnnData(np.ones((1, 1)), obsm={"spatial": np.array([[0.0, 0.0]])})
    path = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    total_length = 4.0 * np.sqrt(2.0) + 2.0

    result = fit_spatial_trajectory(adata, path)

    assert result.obs["trajectory_position"].iloc[0] == pytest.approx(np.sqrt(2.0) / total_length)
    assert result.obs["trajectory_distance"].iloc[0] == 0.0


def test_fit_spatial_trajectory_snaps_exact_terminal_vertex_to_zero_distance() -> None:
    terminal = np.array([-1.0, -0.3])
    adata = anndata.AnnData(np.ones((1, 1)), obsm={"spatial": terminal[None, :]})
    path = np.array([[-2.0, -2.0], terminal])

    result = fit_spatial_trajectory(adata, path)

    assert result.obs["trajectory_position"].iloc[0] == 1.0
    assert result.obs["trajectory_distance"].iloc[0] == 0.0


def test_fit_spatial_trajectory_keeps_first_asymmetric_self_intersection_segment() -> None:
    intersection = np.array([-1.0, -0.3])
    path = np.array(
        [
            [-1.1, -0.4],
            [-0.9, -0.19999999999999998],
            [-1.1, 0.0],
            [-0.9, -0.6],
        ]
    )
    extended = path.astype(np.longdouble)
    lengths = np.sqrt(np.sum(np.diff(extended, axis=0) ** 2, axis=1))
    expected_position = float((lengths[0] / 2) / lengths.sum())
    adata = anndata.AnnData(np.ones((1, 1)), obsm={"spatial": intersection[None, :]})

    result = fit_spatial_trajectory(adata, path)

    assert result.obs["trajectory_position"].iloc[0] == pytest.approx(expected_position)
    assert result.obs["trajectory_distance"].iloc[0] == pytest.approx(0.0, abs=1e-16)


def test_fit_spatial_trajectory_translation_does_not_mask_materially_closer_segment() -> None:
    base_path = np.array([[-18.0, -4.0], [22.0, 4.0], [-2.0, 0.0], [2.0, 0.0]])
    outputs: list[tuple[float, float]] = []

    for shift in [0.0, 1e16]:
        adata = anndata.AnnData(
            np.ones((1, 1)),
            obsm={"spatial": np.array([[shift, shift]])},
        )
        result = fit_spatial_trajectory(adata, base_path + shift)
        outputs.append(
            (
                float(result.obs["trajectory_position"].iloc[0]),
                float(result.obs["trajectory_distance"].iloc[0]),
            )
        )

    assert outputs[0] == outputs[1]
    assert outputs[0][0] > 0.9
    assert outputs[0][1] == 0.0


def test_fit_spatial_trajectory_translation_preserves_first_exact_tie() -> None:
    base_path = np.array([[-8.0, -4.0], [8.0, 4.0], [-8.0, 4.0], [8.0, -4.0]])
    extended = base_path.astype(np.longdouble)
    lengths = np.sqrt(np.sum(np.diff(extended, axis=0) ** 2, axis=1))
    first_position = float((lengths[0] / 2) / lengths.sum())
    outputs: list[tuple[float, float]] = []

    for shift in [0.0, 1e16]:
        adata = anndata.AnnData(
            np.ones((1, 1)),
            obsm={"spatial": np.array([[shift, shift]])},
        )
        result = fit_spatial_trajectory(adata, base_path + shift)
        outputs.append(
            (
                float(result.obs["trajectory_position"].iloc[0]),
                float(result.obs["trajectory_distance"].iloc[0]),
            )
        )

    assert outputs[0] == outputs[1]
    assert outputs[0][0] == pytest.approx(first_position)
    assert outputs[0][1] == 0.0


def test_fit_spatial_trajectory_long_segment_cannot_mask_later_exact_zero() -> None:
    base_path = np.array(
        [[-5e16, 4.0], [5e16, 4.0], [-8.0, 0.0], [8.0, 0.0]],
    )
    outputs: list[tuple[float, float]] = []

    for shift in [0.0, 1e16]:
        adata = anndata.AnnData(
            np.ones((1, 1)),
            obsm={"spatial": np.array([[shift, shift]])},
        )
        result = fit_spatial_trajectory(adata, base_path + shift)
        outputs.append(
            (
                float(result.obs["trajectory_position"].iloc[0]),
                float(result.obs["trajectory_distance"].iloc[0]),
            )
        )

    assert outputs[0] == outputs[1]
    assert outputs[0][0] > 0.99
    assert outputs[0][1] == 0.0


def test_fit_spatial_trajectory_near_equal_material_improvement_wins() -> None:
    closer = np.nextafter(1.0, 0.0)
    path = np.array([[-2.0, 1.0], [2.0, 1.0], [-2.0, closer], [2.0, closer]])
    adata = anndata.AnnData(np.ones((1, 1)), obsm={"spatial": np.array([[0.0, 0.0]])})

    result = fit_spatial_trajectory(adata, path)

    assert result.obs["trajectory_position"].iloc[0] > 0.8
    assert result.obs["trajectory_distance"].iloc[0] == closer


def test_fit_spatial_trajectory_handles_tiny_mixed_scale_segment() -> None:
    adata = anndata.AnnData(
        np.ones((1, 1)),
        obsm={"spatial": np.array([[1.0, -1.0]])},
    )
    path = np.array([[0.0, 0.0], [1e-309, 1e-309]])

    result = fit_spatial_trajectory(adata, path)

    assert result.obs["trajectory_position"].iloc[0] == 0.0
    assert result.obs["trajectory_distance"].iloc[0] == pytest.approx(np.sqrt(2.0))


@pytest.mark.parametrize(
    ("path_end", "point_x"),
    [
        (1.0, 1e-18),
        (1e308, 1e290),
    ],
)
def test_fit_spatial_trajectory_preserves_small_positive_interior_fraction(
    path_end: float,
    point_x: float,
) -> None:
    adata = anndata.AnnData(
        np.ones((1, 1)),
        obsm={"spatial": np.array([[point_x, 0.0]])},
    )
    expected = float(np.longdouble(point_x) / np.longdouble(path_end))

    result = fit_spatial_trajectory(adata, np.array([[0.0, 0.0], [path_end, 0.0]]))

    position = result.obs["trajectory_position"].iloc[0]
    assert position > 0.0
    assert position == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert result.obs["trajectory_distance"].iloc[0] == 0.0


def test_fit_spatial_trajectory_handles_huge_finite_geometry_and_distance() -> None:
    scale = 1e308
    adata = anndata.AnnData(
        np.ones((2, 1)),
        obsm={"spatial": np.array([[5e307, 5e307], [5e307, -5e307]])},
    )
    path = np.array([[0.0, 0.0], [scale, scale]])

    result = fit_spatial_trajectory(adata, path)

    assert result.obs["trajectory_position"].tolist() == [0.5, 0.0]
    assert result.obs["trajectory_distance"].iloc[0] == 0.0
    assert np.isfinite(result.obs["trajectory_distance"].iloc[1])
    assert result.obs["trajectory_distance"].iloc[1] == pytest.approx(np.hypot(5e307, 5e307))
    assert np.isfinite(result.uns["joint"]["trajectory"]["segment_lengths"]).all()
    assert np.isfinite(result.uns["joint"]["trajectory"]["total_length"])


def test_fit_spatial_trajectory_handles_float64_endpoint_difference_overflow() -> None:
    maximum = np.finfo(np.float64).max
    adata = anndata.AnnData(np.ones((1, 1)), obsm={"spatial": np.array([[0.0, 0.0]])})
    path = np.array([[-maximum, 0.0], [maximum, 0.0]])

    result = fit_spatial_trajectory(adata, path)

    assert result.obs["trajectory_position"].iloc[0] == 0.5
    assert result.obs["trajectory_distance"].iloc[0] == 0.0
    assert np.isfinite(result.uns["joint"]["trajectory"]["segment_lengths"]).all()
    assert np.isfinite(result.uns["joint"]["trajectory"]["total_length"])


def test_fit_spatial_trajectory_returns_finite_distance_above_float64_range() -> None:
    maximum = np.finfo(np.float64).max
    adata = anndata.AnnData(
        np.ones((1, 1)),
        obsm={"spatial": np.array([[maximum, -maximum]])},
    )
    path = np.array([[-maximum, -maximum], [-maximum, maximum]])

    result = fit_spatial_trajectory(adata, path)

    distance = result.obs["trajectory_distance"].iloc[0]
    assert result.obs["trajectory_position"].iloc[0] == 0.0
    assert np.isfinite(distance)
    assert distance == 2 * np.longdouble(maximum)


def test_fit_spatial_trajectory_returns_empty_copy_with_metadata() -> None:
    source = anndata.AnnData(
        np.empty((0, 1)),
        var=pd.DataFrame(index=["f0"]),
        obsm={"spatial": np.empty((0, 2))},
    )
    source.uns["joint"] = {"existing": True}
    before = source.copy()

    result = fit_spatial_trajectory(source, np.array([[0.0, 0.0], [2.0, 0.0]]))

    assert result is not source
    assert result.obs["trajectory_position"].empty
    assert result.obs["trajectory_distance"].empty
    assert result.uns["joint"]["existing"] is True
    assert result.uns["joint"]["trajectory_path"] == [[0.0, 0.0], [2.0, 0.0]]
    assert result.uns["joint"]["trajectory"]["total_length"] == 2.0
    assert_adata_unchanged(source, before)


@pytest.mark.parametrize(
    ("adata", "match"),
    [
        (None, "AnnData"),
    ],
)
def test_fit_spatial_trajectory_rejects_invalid_anndata(adata: object, match: str) -> None:
    with pytest.raises(InputFormatError, match=match):
        fit_spatial_trajectory(adata, np.array([[0.0, 0.0], [1.0, 0.0]]))  # type: ignore[arg-type]


def test_fit_spatial_trajectory_requires_unique_observation_identifiers() -> None:
    adata = trajectory_adata()
    adata.obs_names = ["duplicate"] * adata.n_obs

    with pytest.raises(InputFormatError, match="observation.*unique"):
        fit_spatial_trajectory(adata, np.array([[0.0, 0.0], [1.0, 0.0]]))


@pytest.mark.parametrize(
    ("spatial", "match"),
    [
        (None, "spatial"),
        (np.ones((3, 3)), "shape"),
        (np.ones((2, 2)), "shape"),
        (np.full((3, 2), "1"), "real numeric"),
        (np.ones((3, 2), dtype=bool), "real numeric"),
        (np.ones((3, 2), dtype=complex), "real numeric"),
        (np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0]]), "finite"),
        (np.array([[0.0, 0.0], [np.inf, 1.0], [2.0, 2.0]]), "finite"),
    ],
)
def test_fit_spatial_trajectory_validates_spatial_coordinates(spatial: object, match: str) -> None:
    adata = trajectory_adata()
    if spatial is None:
        del adata.obsm["spatial"]
    else:
        adata.obsm._data["spatial"] = spatial

    with pytest.raises(InputFormatError, match=match):
        fit_spatial_trajectory(adata, np.array([[0.0, 0.0], [1.0, 0.0]]))


@pytest.mark.parametrize(
    ("path", "match"),
    [
        (None, "path"),
        ([0.0, 1.0], "shape"),
        ([[0.0, 0.0]], "at least two"),
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], "shape"),
        ([["0", "0"], ["1", "1"]], "real numeric"),
        (np.ones((2, 2), dtype=bool), "real numeric"),
        (np.ones((2, 2), dtype=complex), "real numeric"),
        ([[0.0, 0.0], [np.nan, 1.0]], "finite"),
        ([[0.0, 0.0], [np.inf, 1.0]], "finite"),
        (
            np.array([[0.0, 0.0], [np.finfo(np.longdouble).max, 0.0]], dtype=np.longdouble),
            "finite",
        ),
        ([[0.0, 0.0], [0.0, 0.0]], "zero-length"),
    ],
)
def test_fit_spatial_trajectory_rejects_invalid_path(path: object, match: str) -> None:
    with pytest.raises(InputFormatError, match=match):
        fit_spatial_trajectory(trajectory_adata(), path)  # type: ignore[arg-type]


def test_fit_spatial_trajectory_rejects_nonmapping_joint_metadata_without_mutation() -> None:
    adata = trajectory_adata()
    adata.uns["joint"] = "invalid"
    before = adata.copy()

    with pytest.raises(InputFormatError, match=r"uns\['joint'\].*mapping"):
        fit_spatial_trajectory(adata, np.array([[0.0, 0.0], [1.0, 0.0]]))

    assert_adata_unchanged(adata, before)


def test_importing_trajectory_in_clean_process_does_not_import_pygam() -> None:
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; assert 'pygam' not in sys.modules; "
                "import joint.trajectory; assert 'pygam' not in sys.modules"
            ),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_calculate_feature_trends_has_precise_chained_missing_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pygam", None)

    with pytest.raises(OptionalDependencyError) as error:
        calculate_feature_trends(projected_adata(), ["f0"], points=10)

    assert str(error.value) == (
        "Trajectory trends require `python -m pip install 'joint-msi[trajectory]'`"
    )
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_calculate_feature_trends_propagates_transitive_import_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "pygam", raising=False)
    original_import = builtins.__import__
    failure = ModuleNotFoundError("No module named 'pygam_backend'", name="pygam_backend")

    def transitive_failure(name: str, *args: object, **kwargs: object) -> Any:
        if name == "pygam":
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", transitive_failure)

    with pytest.raises(ModuleNotFoundError) as error:
        calculate_feature_trends(projected_adata(), ["f0"])

    assert error.value is failure


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("layer", [None, "signal"])
def test_calculate_feature_trends_selects_dense_or_sparse_features_in_exact_order(
    monkeypatch: pytest.MonkeyPatch,
    as_sparse: bool,
    layer: str | None,
) -> None:
    adata = projected_adata()
    selected = np.array([[101.0, 1001.0], [102.0, 1002.0], [103.0, 1003.0]])
    selected_source = sparse.csr_matrix(selected) if as_sparse else selected
    if layer is None:
        adata.X = selected_source
        adata.layers["ignored"] = np.full(adata.shape, np.nan)
    else:
        adata.X = np.full(adata.shape, np.nan)
        adata.layers[layer] = selected_source
    before = adata.copy()
    calls = install_fake_pygam(monkeypatch)

    trends = calculate_feature_trends(adata, ["f1", "f0"], points=4, layer=layer)

    assert trends.columns.tolist() == ["feature", "position", "fitted", "lower", "upper"]
    assert trends["feature"].drop_duplicates().tolist() == ["f1", "f0"]
    assert trends.groupby("feature", sort=False).size().tolist() == [4, 4]
    assert trends["position"].tolist() == [0.0, 1 / 3, 2 / 3, 1.0] * 2
    np.testing.assert_array_equal(calls[0]["y"], [1001.0, 1002.0, 1003.0])
    np.testing.assert_array_equal(calls[1]["y"], [101.0, 102.0, 103.0])
    for call in calls:
        assert call["term"] == ("s", 0)
        np.testing.assert_array_equal(call["x"].ravel(), [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(call["grid"].ravel(), [0.0, 1 / 3, 2 / 3, 1.0])
        assert call["width"] == 0.95
    assert np.isfinite(trends[["fitted", "lower", "upper"]].to_numpy()).all()
    assert (trends["lower"] <= trends["upper"]).all()
    assert_adata_unchanged(adata, before)


@pytest.mark.parametrize(
    ("positions", "match"),
    [
        (None, "trajectory_position"),
        ([0.0, 0.5, np.nan], "finite"),
        ([0.0, 0.5, np.inf], "finite"),
        ([-0.1, 0.5, 1.0], r"\[0, 1\]"),
        ([0.0, 0.5, 1.1], r"\[0, 1\]"),
        ([0.5, 0.5, 0.5], "distinct"),
        ([False, True, False], "real numeric"),
        (["0", "0.5", "1"], "real numeric"),
        ([0j, 0.5 + 0j, 1 + 0j], "real numeric"),
    ],
)
def test_calculate_feature_trends_validates_projected_positions(
    positions: object, match: str
) -> None:
    adata = projected_adata()
    if positions is None:
        del adata.obs["trajectory_position"]
    else:
        adata.obs["trajectory_position"] = positions

    with pytest.raises(InputFormatError, match=match):
        calculate_feature_trends(adata, ["f0"])


@pytest.mark.parametrize(
    ("features", "match"),
    [
        ([], "at least one"),
        (["f0", "f0"], "unique"),
        (["missing"], "known"),
        ([""], "non-empty string"),
        (["   "], "non-empty string"),
        ([0], "non-empty string"),
        ("f0", "sequence"),
    ],
)
def test_calculate_feature_trends_validates_features(features: object, match: str) -> None:
    with pytest.raises(InputFormatError, match=match):
        calculate_feature_trends(projected_adata(), features)  # type: ignore[arg-type]


@pytest.mark.parametrize("points", [True, 1, 0, -1, 2.5, "2"])
def test_calculate_feature_trends_requires_at_least_two_integer_points(points: object) -> None:
    with pytest.raises(InputFormatError, match="points"):
        calculate_feature_trends(projected_adata(), ["f0"], points=points)  # type: ignore[arg-type]


@pytest.mark.parametrize("layer", ["", "   ", 1, "missing"])
def test_calculate_feature_trends_rejects_invalid_or_missing_layer(layer: object) -> None:
    with pytest.raises(InputFormatError, match="layer"):
        calculate_feature_trends(projected_adata(), ["f0"], layer=layer)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (np.full((3, 2), "1"), "real numeric"),
        (np.ones((3, 2), dtype=bool), "real numeric"),
        (np.ones((3, 2), dtype=complex), "real numeric"),
        (np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]), "finite"),
        (np.array([[1.0, 2.0], [3.0, np.inf], [5.0, 6.0]]), "finite"),
        (np.ones((1, 1)), "shape"),
    ],
)
def test_calculate_feature_trends_validates_the_entire_selected_source(
    source: object, match: str
) -> None:
    adata = projected_adata()
    adata.layers._data["broken"] = source

    with pytest.raises(InputFormatError, match=match):
        calculate_feature_trends(adata, ["f0"], layer="broken")


def test_calculate_feature_trends_translates_unreadable_source_to_domain_error() -> None:
    class UnreadableArray:
        shape = (3, 2)

        def __array__(self, dtype: object = None) -> np.ndarray:
            raise ValueError("cannot materialize")

    adata = projected_adata()
    adata.layers._data["broken"] = UnreadableArray()

    with pytest.raises(InputFormatError, match="real numeric"):
        calculate_feature_trends(adata, ["f0"], layer="broken")


def test_calculate_feature_trends_requires_unique_observation_and_feature_ids() -> None:
    duplicate_obs = projected_adata()
    duplicate_obs.obs_names = ["duplicate"] * duplicate_obs.n_obs
    with pytest.raises(InputFormatError, match="observation.*unique"):
        calculate_feature_trends(duplicate_obs, ["f0"])

    duplicate_var = projected_adata()
    duplicate_var.var_names = ["duplicate"] * duplicate_var.n_vars
    with pytest.raises(InputFormatError, match="feature.*unique"):
        calculate_feature_trends(duplicate_var, ["duplicate"])


def test_calculate_feature_trends_validates_before_importing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pygam", None)

    with pytest.raises(InputFormatError, match="features"):
        calculate_feature_trends(projected_adata(), [])


def test_calculate_feature_trends_shields_anndata_from_backend_array_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adata = projected_adata()
    before = adata.copy()
    module = ModuleType("pygam")

    class MutatingLinearGAM:
        def __init__(self, term: object) -> None:
            self.term = term

        def fit(self, x: np.ndarray, y: np.ndarray) -> "MutatingLinearGAM":
            x[:] = -10.0
            y[:] = -20.0
            return self

        def predict(self, grid: np.ndarray) -> np.ndarray:
            grid[:] = -30.0
            return np.ones(len(grid))

        def prediction_intervals(self, grid: np.ndarray, *, width: float) -> np.ndarray:
            grid[:] = -40.0
            return np.column_stack([np.zeros(len(grid)), np.ones(len(grid))])

    module.LinearGAM = MutatingLinearGAM  # type: ignore[attr-defined]
    module.s = lambda index: index  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pygam", module)

    trends = calculate_feature_trends(adata, ["f0"], points=4)

    assert trends["position"].tolist() == [0.0, 1 / 3, 2 / 3, 1.0]
    assert_adata_unchanged(adata, before)


def test_calculate_feature_trends_rejects_model_without_callable_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("pygam")
    module.LinearGAM = lambda term: object()  # type: ignore[attr-defined]
    module.s = lambda index: index  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pygam", module)

    with pytest.raises(InputFormatError, match="callable fit"):
        calculate_feature_trends(projected_adata(), ["f0"])


@pytest.mark.parametrize(
    ("prediction", "match"),
    [
        (np.ones((4, 1)), "prediction.*shape"),
        (np.full(4, "1"), "prediction.*real numeric"),
        (np.ones(4, dtype=complex), "prediction.*real numeric"),
        (np.array([1.0, 2.0, np.nan, 4.0]), "prediction.*finite"),
        (np.array([1.0, 2.0, np.inf, 4.0]), "prediction.*finite"),
        (np.full(4, np.finfo(np.longdouble).max, dtype=np.longdouble), "prediction.*finite"),
    ],
)
def test_calculate_feature_trends_rejects_malformed_pygam_predictions(
    monkeypatch: pytest.MonkeyPatch,
    prediction: object,
    match: str,
) -> None:
    install_fake_pygam(monkeypatch, prediction=prediction)

    with pytest.raises(InputFormatError, match=match):
        calculate_feature_trends(projected_adata(), ["f0"], points=4)


@pytest.mark.parametrize(
    ("intervals", "match"),
    [
        (np.ones((4, 1)), "interval.*shape"),
        (np.full((4, 2), "1"), "interval.*real numeric"),
        (np.ones((4, 2), dtype=complex), "interval.*real numeric"),
        (
            np.array([[0.0, 1.0], [0.0, np.nan], [0.0, 1.0], [0.0, 1.0]]),
            "interval.*finite",
        ),
        (
            np.array([[0.0, 1.0], [0.0, np.inf], [0.0, 1.0], [0.0, 1.0]]),
            "interval.*finite",
        ),
        (
            np.array([[0.0, 1.0], [2.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            "lower.*upper",
        ),
    ],
)
def test_calculate_feature_trends_rejects_malformed_pygam_intervals(
    monkeypatch: pytest.MonkeyPatch,
    intervals: object,
    match: str,
) -> None:
    install_fake_pygam(monkeypatch, intervals=intervals)

    with pytest.raises(InputFormatError, match=match):
        calculate_feature_trends(projected_adata(), ["f0"], points=4)
