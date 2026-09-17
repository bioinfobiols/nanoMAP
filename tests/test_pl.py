import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from joint.errors import InputFormatError
from joint.models import RegistrationResult
from joint.pl import (
    plot_cell_contours,
    plot_cnmf_usage,
    plot_registration,
    plot_segmentation,
    plot_spatial_feature,
    plot_trajectory,
)


def plot_adata() -> anndata.AnnData:
    adata = anndata.AnnData(
        np.array([[1.0], [2.0]]),
        obs={"Usage_1": [0.2, 0.8]},
        obsm={"spatial": np.array([[2.0, 3.0], [7.0, 8.0]])},
    )
    adata.var_names = ["m1"]
    adata.obs["cluster"] = ["a", "b"]
    return adata


def registration_result() -> RegistrationResult:
    return RegistrationResult(
        plot_adata(),
        pd.DataFrame({"centroid-1": [2.0, 7.0], "centroid-0": [3.0, 8.0]}),
        {"orientation": "identity"},
        {},
    )


def trajectory_trends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["m1", "m1"],
            "position": [0.0, 1.0],
            "fitted": [1.0, 2.0],
            "lower": [0.8, 1.8],
            "upper": [1.2, 2.2],
        }
    )


def test_plotting_functions_return_figures_and_save_when_requested(tmp_path: Path) -> None:
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    adata = plot_adata()
    trends = trajectory_trends()
    calls = [
        plot_segmentation(labels),
        plot_cell_contours(labels),
        plot_spatial_feature(adata, "m1"),
        plot_spatial_feature(adata, "cluster"),
        plot_registration(registration_result()),
        plot_cnmf_usage(adata, "Usage_1"),
        plot_trajectory(trends),
    ]
    assert all(figure is not None and axes is not None for figure, axes in calls)
    path = tmp_path / "nested" / "segmentation.png"
    plot_segmentation(labels, save=path)
    assert path.is_file()


def test_plots_reuse_axes_and_do_not_modify_input_data() -> None:
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    image = np.arange(4, dtype=float).reshape(2, 2)
    adata = plot_adata()
    adata_before = adata.copy()
    trends = trajectory_trends()
    trends_before = trends.copy(deep=True)
    result = registration_result()
    mapping_before = result.mapping.copy(deep=True)
    calls = [
        (plot_segmentation, (labels,)),
        (plot_cell_contours, (labels,)),
        (plot_spatial_feature, (adata, "m1")),
        (plot_registration, (result,)),
        (plot_cnmf_usage, (adata, "Usage_1")),
        (plot_trajectory, (trends,)),
    ]
    for function, args in calls:
        figure, axes = plt.subplots()
        returned_figure, returned_axes = function(*args, ax=axes)
        assert returned_figure is figure
        assert returned_axes is axes
        plt.close(figure)
    plot_cell_contours(labels, image=image)
    np.testing.assert_array_equal(image, np.arange(4, dtype=float).reshape(2, 2))
    assert adata.obs.equals(adata_before.obs)
    np.testing.assert_array_equal(adata.X, adata_before.X)
    np.testing.assert_array_equal(adata.obsm["spatial"], adata_before.obsm["spatial"])
    assert trends.equals(trends_before)
    assert result.mapping.equals(mapping_before)


@pytest.mark.parametrize(
    ("function", "args", "kwargs", "match"),
    [
        (plot_segmentation, (np.array([0, 1]),), {}, "labels"),
        (plot_cell_contours, (np.array([[0, 1], [2, 2]]),), {"image": np.ones(3)}, "image"),
        (plot_spatial_feature, (plot_adata(), "missing"), {}, "feature"),
        (plot_spatial_feature, (plot_adata(), "m1"), {"layer": "missing"}, "layer"),
        (plot_cnmf_usage, (plot_adata(), "missing"), {}, "usage"),
        (plot_registration, (object(),), {}, "RegistrationResult"),
        (plot_trajectory, (pd.DataFrame({"feature": ["m1"]}),), {}, "trends"),
        (plot_segmentation, (np.ones((2, 2), dtype=np.int32),), {"save": 7}, "save"),
        (plot_segmentation, (np.ones((2, 2), dtype=np.int32),), {"ax": object()}, "ax"),
    ],
)
def test_plotting_apis_reject_invalid_public_inputs(function, args, kwargs, match: str) -> None:
    with pytest.raises(InputFormatError, match=match):
        function(*args, **kwargs)


@pytest.mark.parametrize(
    ("mutate", "function", "args"),
    [
        (
            lambda adata: adata.obsm.__delitem__("spatial"),
            plot_spatial_feature,
            lambda adata: (adata, "m1"),
        ),
        (
            lambda adata: adata.obsm.__setitem__("spatial", np.ones((2, 3))),
            plot_cnmf_usage,
            lambda adata: (adata, "Usage_1"),
        ),
    ],
)
def test_spatial_plots_validate_spatial_coordinates(mutate, function, args) -> None:
    adata = plot_adata()
    mutate(adata)

    with pytest.raises(InputFormatError, match="spatial"):
        function(*args(adata))


def test_plot_registration_validates_mapping_and_orientation() -> None:
    for mapping, transform, match in [
        (pd.DataFrame({"centroid-0": [1.0]}), {"orientation": "identity"}, "mapping"),
        (
            pd.DataFrame({"centroid-0": [1.0], "centroid-1": [np.nan]}),
            {"orientation": "identity"},
            "centroid-1",
        ),
        (
            pd.DataFrame({"centroid-0": [1.0], "centroid-1": [2.0]}),
            {},
            "orientation",
        ),
    ]:
        result = RegistrationResult(plot_adata(), mapping, transform, {})
        with pytest.raises(InputFormatError, match=match):
            plot_registration(result)


def test_plot_trajectory_validates_numeric_intervals() -> None:
    trends = trajectory_trends()
    trends.loc[0, "lower"] = 2.0
    trends.loc[0, "upper"] = 1.0

    with pytest.raises(InputFormatError, match="lower"):
        plot_trajectory(trends)


def test_cell_contours_rejects_images_with_unsupported_channel_count() -> None:
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)

    with pytest.raises(InputFormatError, match="image"):
        plot_cell_contours(labels, image=np.ones((2, 2, 2)))
