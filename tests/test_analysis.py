import builtins
import copy
import subprocess
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import ClassVar

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse
from scipy.sparse import csr_array, csr_matrix

import joint.analysis as analysis_module
from joint.analysis import (
    PreparedCnmf,
    cluster_cells,
    consensus_cnmf,
    factorize_cnmf,
    load_cnmf_results,
    prepare_cnmf,
    rank_metabolites,
    run_cnmf,
    run_cosg,
    select_cnmf_k,
)
from joint.errors import InputFormatError, OptionalDependencyError


def analysis_adata() -> anndata.AnnData:
    rng = np.random.default_rng(7)
    matrix = rng.normal(1.0, 0.05, (12, 5))
    matrix[:6, :2] += 3.0
    matrix[6:, 2:4] += 3.0
    adata = anndata.AnnData(
        matrix,
        obs=pd.DataFrame(
            {
                "known_group": pd.Categorical(
                    ["a"] * 6 + ["b"] * 6,
                    categories=["b", "a"],
                    ordered=True,
                )
            },
            index=[f"cell-{index}" for index in range(12)],
        ),
        var=pd.DataFrame(index=[f"m{index}" for index in range(5)]),
    )
    adata.layers["signal"] = matrix.copy()
    adata.layers["other"] = matrix[:, ::-1].copy()
    adata.raw = adata.copy()
    adata.uns["source"] = {"name": "fixture"}
    return adata


def assert_matrix_equal(actual, expected) -> None:
    if sparse.issparse(actual) or sparse.issparse(expected):
        assert sparse.issparse(actual) and sparse.issparse(expected)
        difference = actual != expected
        assert difference.nnz == 0
    else:
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def assert_adata_payload_unchanged(actual: anndata.AnnData, expected: anndata.AnnData) -> None:
    assert actual.obs.equals(expected.obs)
    assert actual.var.equals(expected.var)
    assert actual.obs_names.equals(expected.obs_names)
    assert actual.var_names.equals(expected.var_names)
    assert_matrix_equal(actual.X, expected.X)
    assert list(actual.layers) == list(expected.layers)
    for name in actual.layers:
        assert_matrix_equal(actual.layers[name], expected.layers[name])
    assert list(actual.obsm) == list(expected.obsm)
    for name in actual.obsm:
        assert_matrix_equal(actual.obsm[name], expected.obsm[name])
    assert copy.deepcopy(actual.uns) == copy.deepcopy(expected.uns)
    assert (actual.raw is None) == (expected.raw is None)
    if actual.raw is not None and expected.raw is not None:
        assert actual.raw.var.equals(expected.raw.var)
        assert actual.raw.obs_names.equals(expected.raw.obs_names)
        assert_matrix_equal(actual.raw.X, expected.raw.X)


def test_cluster_cells_returns_deterministic_copy_and_preserves_matrix_payload() -> None:
    source = analysis_adata()
    before = source.copy()

    first = cluster_cells(source, n_neighbors=4, resolution=0.5, random_seed=17)
    second = cluster_cells(source, n_neighbors=4, resolution=0.5, random_seed=17)

    assert first is not source
    assert "cluster" not in source.obs
    assert "cluster" in first.obs
    assert "X_pca" in first.obsm
    assert first.obs["cluster"].astype(str).tolist() == second.obs["cluster"].astype(str).tolist()
    np.testing.assert_allclose(first.obsm["X_pca"], second.obsm["X_pca"])
    assert_matrix_equal(first.X, before.X)
    for name in before.layers:
        assert_matrix_equal(first.layers[name], before.layers[name])
    assert first.raw is not None and before.raw is not None
    assert_matrix_equal(first.raw.X, before.raw.X)
    assert_adata_payload_unchanged(source, before)


def test_cluster_cells_uses_layer_for_hvg_and_pca_and_passes_every_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, dict[str, object]] = {}
    original_pca = sc.tl.pca
    original_neighbors = sc.pp.neighbors
    original_leiden = sc.tl.leiden

    def record_pca(*args, **kwargs):
        calls["pca"] = kwargs.copy()
        return original_pca(*args, **kwargs)

    def record_neighbors(*args, **kwargs):
        calls["neighbors"] = kwargs.copy()
        return original_neighbors(*args, **kwargs)

    def record_leiden(*args, **kwargs):
        calls["leiden"] = kwargs.copy()
        return original_leiden(*args, **kwargs)

    monkeypatch.setattr(analysis_module.sc.tl, "pca", record_pca)
    monkeypatch.setattr(analysis_module.sc.pp, "neighbors", record_neighbors)
    monkeypatch.setattr(analysis_module.sc.tl, "leiden", record_leiden)

    clustered = cluster_cells(
        analysis_adata(),
        layer="signal",
        n_top_features=3,
        n_neighbors=4,
        resolution=0.7,
        random_seed=23,
        key_added="partition",
    )

    assert clustered.var["highly_variable"].sum() == 3
    assert calls["pca"]["layer"] == "signal"
    np.testing.assert_array_equal(
        calls["pca"]["mask_var"], clustered.var["highly_variable"].to_numpy()
    )
    assert calls["pca"]["random_state"] == 23
    assert calls["neighbors"]["random_state"] == 23
    assert calls["neighbors"]["n_neighbors"] == 4
    assert calls["leiden"]["random_state"] == 23
    assert calls["leiden"]["key_added"] == "partition"
    assert calls["leiden"]["flavor"] == "igraph"
    assert clustered.uns["joint"]["clustering"] == {
        "layer": "signal",
        "n_top_features": 3,
        "n_neighbors": 4,
        "resolution": 0.7,
        "random_seed": 23,
        "key_added": "partition",
        "n_pcs": 3,
        "pca_solver": "full",
        "leiden_flavor": "igraph",
        "leiden_iterations": 2,
    }


def test_cluster_cells_supports_narrow_matrices_with_valid_pca_dimensions() -> None:
    adata = anndata.AnnData(
        np.array([[0.0], [0.1], [4.0], [4.1]]),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=["feature"]),
    )

    clustered = cluster_cells(adata, n_neighbors=2, random_seed=3)

    assert clustered.obsm["X_pca"].shape == (4, 1)
    assert "cluster" in clustered.obs


@pytest.mark.parametrize("constructor", [csr_matrix, csr_array])
def test_cluster_cells_supports_sparse_selected_matrices(constructor) -> None:
    adata = anndata.AnnData(
        constructor([[0.0, 1.0], [0.1, 1.1], [4.0, 0.0], [4.1, 0.1]]),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=["m0", "m1"]),
    )

    clustered = cluster_cells(adata, n_neighbors=2, random_seed=3)

    assert sparse.issparse(clustered.X)
    assert clustered.obsm["X_pca"].shape == (4, 1)
    assert "cluster" in clustered.obs
    assert clustered.uns["joint"]["clustering"]["pca_solver"] == "arpack"
    assert clustered.uns["joint"]["clustering"]["n_pcs"] == 1


def test_cluster_cells_passes_sklearn_floor_compatible_sparse_pca_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    original_pca = sc.tl.pca

    def record_pca(*args, **kwargs):
        calls.update(kwargs)
        return original_pca(*args, **kwargs)

    monkeypatch.setattr(analysis_module.sc.tl, "pca", record_pca)
    adata = anndata.AnnData(
        csr_matrix([[0.0, 1.0], [0.1, 1.1], [4.0, 0.0], [4.1, 0.1]]),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=["m0", "m1"]),
    )

    clustered = cluster_cells(adata, n_neighbors=2, random_seed=3)

    assert calls["svd_solver"] == "arpack"
    assert calls["n_comps"] == 1
    assert clustered.uns["joint"]["clustering"]["pca_solver"] == "arpack"


@pytest.mark.parametrize("constructor", [csr_matrix, csr_array])
def test_cluster_cells_has_deterministic_single_sparse_feature_path(constructor) -> None:
    adata = anndata.AnnData(
        constructor([[0.0], [0.1], [4.0], [4.1]]),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=["feature"]),
    )

    first = cluster_cells(adata, n_neighbors=2, random_seed=3)
    second = cluster_cells(adata, n_neighbors=2, random_seed=3)

    np.testing.assert_array_equal(first.obsm["X_pca"], second.obsm["X_pca"])
    assert first.obsm["X_pca"].shape == (4, 1)
    assert first.uns["joint"]["clustering"]["pca_solver"] == "manual_single_feature"
    assert first.uns["joint"]["clustering"]["n_pcs"] == 1


def test_cluster_cells_caps_pcs_at_50_and_excludes_observation_null_dimension() -> None:
    rng = np.random.default_rng(31)
    adata = anndata.AnnData(
        rng.normal(size=(60, 100)),
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(60)]),
        var=pd.DataFrame(index=[f"m{index}" for index in range(100)]),
    )

    clustered = cluster_cells(adata, n_neighbors=5, random_seed=11)

    assert clustered.obsm["X_pca"].shape == (60, 50)
    assert clustered.uns["neighbors"]["params"]["n_pcs"] == 50
    assert clustered.uns["joint"]["clustering"]["n_pcs"] == 50
    assert clustered.uns["joint"]["clustering"]["pca_solver"] == "full"
    assert np.all(clustered.uns["pca"]["variance"] > 0)


def test_cluster_cells_uses_at_most_n_obs_minus_one_dense_pcs() -> None:
    rng = np.random.default_rng(37)
    adata = anndata.AnnData(
        rng.normal(size=(4, 10)),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=[f"m{index}" for index in range(10)]),
    )

    clustered = cluster_cells(adata, n_neighbors=2, random_seed=3)

    assert clustered.obsm["X_pca"].shape == (4, 3)
    assert clustered.uns["joint"]["clustering"]["n_pcs"] == 3


def test_cluster_cells_rejects_selected_matrix_without_variable_features() -> None:
    adata = anndata.AnnData(
        np.ones((4, 2)),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=["m0", "m1"]),
    )

    with pytest.raises(InputFormatError, match="variable feature"):
        cluster_cells(adata, n_neighbors=2)


def test_cluster_cells_rejects_top_feature_request_larger_than_variable_feature_count() -> None:
    adata = anndata.AnnData(
        np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 3.0, 1.0], [1.0, 4.0, 1.0]]),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=["m0", "m1", "m2"]),
    )

    with pytest.raises(InputFormatError, match="n_top_features.*variable"):
        cluster_cells(adata, n_top_features=2, n_neighbors=2)


def test_cluster_cells_selects_top_variances_stably_on_high_raw_abundances() -> None:
    matrix = 1_000.0 + np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 5.0, 0.0, 0.0],
            [3.0, 1.0, 1.0, 0.0],
            [4.0, 4.0, 1.0, 0.0],
            [5.0, 3.0, 1.0, 0.0],
        ]
    )
    # m0 and m1 have equal variance; their original order must break the tie.
    adata = anndata.AnnData(
        matrix,
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(6)]),
        var=pd.DataFrame(index=["m0", "m1", "m2", "m3"]),
    )

    clustered = cluster_cells(adata, n_top_features=2, n_neighbors=2, random_seed=5)

    assert clustered.var["highly_variable"].tolist() == [True, True, False, False]
    assert clustered.obsm["X_pca"].shape == (6, 2)


@pytest.mark.parametrize(
    ("adata", "match"),
    [
        (None, "AnnData"),
        (anndata.AnnData(np.empty((0, 2))), "observation"),
        (anndata.AnnData(np.empty((2, 0))), "feature"),
    ],
)
def test_cluster_cells_rejects_invalid_or_empty_anndata(adata: object, match: str) -> None:
    with pytest.raises(InputFormatError, match=match):
        cluster_cells(adata, n_neighbors=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("axis", ["obs", "var"])
def test_cluster_cells_requires_unique_identifiers(axis: str) -> None:
    adata = analysis_adata()
    if axis == "obs":
        adata.obs_names = ["duplicate"] * adata.n_obs
    else:
        adata.var_names = ["duplicate"] * adata.n_vars

    with pytest.raises(InputFormatError, match=f"{axis}ervation|{axis}.*unique|feature.*unique"):
        cluster_cells(adata, n_neighbors=4)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_cluster_cells_validates_only_the_selected_finite_numeric_matrix(value: float) -> None:
    broken_x = analysis_adata()
    broken_x.X[0, 0] = value
    with pytest.raises(InputFormatError, match="adata.X.*finite"):
        cluster_cells(broken_x, n_neighbors=4)

    broken_layer = analysis_adata()
    broken_layer.layers["signal"][0, 0] = value
    with pytest.raises(InputFormatError, match=r'layers\["signal"\].*finite'):
        cluster_cells(broken_layer, layer="signal", n_neighbors=4)

    ignored_layer = analysis_adata()
    ignored_layer.layers["other"][0, 0] = value
    clustered = cluster_cells(ignored_layer, layer="signal", n_neighbors=4)
    assert "cluster" in clustered.obs


def test_cluster_cells_rejects_nonnumeric_and_malformed_selected_matrices() -> None:
    nonnumeric = analysis_adata()
    nonnumeric.X = np.full(nonnumeric.shape, "bad")
    with pytest.raises(InputFormatError, match="numeric"):
        cluster_cells(nonnumeric, n_neighbors=4)

    malformed = analysis_adata()
    malformed.layers._data["bad"] = np.ones((1, 1))
    with pytest.raises(InputFormatError, match="shape"):
        cluster_cells(malformed, layer="bad", n_neighbors=4)


@pytest.mark.parametrize("layer", ["", "   ", 1, "missing"])
def test_cluster_cells_rejects_invalid_or_missing_layer(layer: object) -> None:
    with pytest.raises(InputFormatError, match="layer"):
        cluster_cells(analysis_adata(), layer=layer, n_neighbors=4)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_top_features", [True, 0, -1, 1.5, 6])
def test_cluster_cells_rejects_invalid_n_top_features(n_top_features: object) -> None:
    with pytest.raises(InputFormatError, match="n_top_features"):
        cluster_cells(
            analysis_adata(),
            n_top_features=n_top_features,  # type: ignore[arg-type]
            n_neighbors=4,
        )


@pytest.mark.parametrize("n_neighbors", [True, 0, 1, -1, 1.5, 12])
def test_cluster_cells_rejects_invalid_n_neighbors(n_neighbors: object) -> None:
    with pytest.raises(InputFormatError, match="n_neighbors"):
        cluster_cells(analysis_adata(), n_neighbors=n_neighbors)  # type: ignore[arg-type]


@pytest.mark.parametrize("resolution", [True, 0, -1, np.nan, np.inf, "1"])
def test_cluster_cells_rejects_invalid_resolution(resolution: object) -> None:
    with pytest.raises(InputFormatError, match="resolution"):
        cluster_cells(analysis_adata(), resolution=resolution, n_neighbors=4)  # type: ignore[arg-type]


@pytest.mark.parametrize("random_seed", [True, -1, 2**32, 1.5])
def test_cluster_cells_rejects_invalid_random_seed(random_seed: object) -> None:
    with pytest.raises(InputFormatError, match="random_seed"):
        cluster_cells(analysis_adata(), random_seed=random_seed, n_neighbors=4)  # type: ignore[arg-type]


@pytest.mark.parametrize("key_added", [None, "", "   ", 1])
def test_cluster_cells_rejects_invalid_key_added(key_added: object) -> None:
    with pytest.raises(InputFormatError, match="key_added"):
        cluster_cells(analysis_adata(), key_added=key_added, n_neighbors=4)  # type: ignore[arg-type]


def test_rank_metabolites_returns_stable_tidy_table_in_group_order_without_mutation() -> None:
    source = analysis_adata()
    before = source.copy()

    ranked = rank_metabolites(source, groupby="known_group", method="wilcoxon")

    assert ranked.columns.tolist() == [
        "group",
        "names",
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
    ]
    assert ranked["group"].map(type).eq(str).all()
    assert ranked["group"].drop_duplicates().tolist() == ["b", "a"]
    assert ranked.groupby("group", sort=False, observed=True).size().tolist() == [5, 5]
    assert np.isfinite(ranked[["scores", "pvals", "pvals_adj"]].to_numpy()).all()
    assert_adata_payload_unchanged(source, before)


def test_rank_metabolites_uses_requested_layer_and_never_implicit_raw() -> None:
    adata = analysis_adata()
    x = np.ones(adata.shape)
    x[:6, 0] = 5.0
    x[6:, 0] = 1.0
    layer = np.ones(adata.shape)
    layer[:6, 1] = 6.0
    layer[6:, 1] = 1.0
    raw = np.ones(adata.shape)
    raw[:6, 2] = 7.0
    raw[6:, 2] = 1.0
    adata.X = x
    adata.layers["signal"] = layer
    adata.raw = anndata.AnnData(raw, obs=adata.obs.copy(), var=adata.var.copy())

    ranked_x = rank_metabolites(adata, groupby="known_group", method="wilcoxon")
    ranked_layer = rank_metabolites(adata, groupby="known_group", method="wilcoxon", layer="signal")

    top_x = ranked_x.loc[ranked_x["group"] == "a", "names"].iloc[0]
    top_layer = ranked_layer.loc[ranked_layer["group"] == "a", "names"].iloc[0]
    assert top_x == "m0"
    assert top_layer == "m1"


def test_rank_metabolites_stringifies_numeric_groups_in_first_seen_order() -> None:
    adata = analysis_adata()
    adata.obs["numeric_group"] = [2] * 6 + [1] * 6

    ranked = rank_metabolites(adata, groupby="numeric_group", method="t-test")

    assert ranked["group"].drop_duplicates().tolist() == ["2", "1"]


def test_rank_metabolites_accepts_finite_negative_values_without_warning_or_mutation() -> None:
    adata = analysis_adata()[:, ["m0", "m1"]].copy()
    adata.X = np.array(
        [
            [-2.0, -1.0],
            [-2.1, -1.1],
            [-1.9, -0.9],
            [-2.2, -1.2],
            [-1.8, -0.8],
            [-2.0, -1.0],
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [1.2, 2.2],
            [0.8, 1.8],
            [1.0, 2.0],
        ]
    )
    before = adata.copy()

    ranked = rank_metabolites(adata, groupby="known_group", method="wilcoxon")

    assert np.isfinite(ranked[["scores", "pvals", "pvals_adj"]].to_numpy()).all()
    assert_adata_payload_unchanged(adata, before)


def test_rank_metabolites_handles_high_raw_values_without_logfold_warning() -> None:
    adata = analysis_adata()
    adata.X = 1_000.0 + np.asarray(adata.X)

    ranked = rank_metabolites(adata, groupby="known_group", method="wilcoxon")

    assert np.isfinite(ranked[["scores", "pvals", "pvals_adj"]].to_numpy()).all()


def test_rank_metabolites_keeps_infinite_perfect_separation_t_test_scores() -> None:
    adata = analysis_adata()[:, ["m0", "m1"]].copy()
    adata.X = np.array([[1.0, 2.0]] * 6 + [[3.0, 4.0]] * 6)

    ranked = rank_metabolites(adata, groupby="known_group", method="t-test")

    assert np.isinf(ranked["scores"]).any()
    assert np.isfinite(ranked[["pvals", "pvals_adj"]].to_numpy()).all()
    assert ranked[["pvals", "pvals_adj"]].ge(0).all().all()
    assert ranked[["pvals", "pvals_adj"]].le(1).all().all()


@pytest.mark.parametrize(
    ("adata", "match"),
    [
        (None, "AnnData"),
        (anndata.AnnData(np.empty((0, 2))), "observation"),
        (anndata.AnnData(np.empty((2, 0))), "feature"),
    ],
)
def test_rank_metabolites_rejects_invalid_or_empty_anndata(adata: object, match: str) -> None:
    with pytest.raises(InputFormatError, match=match):
        rank_metabolites(adata, groupby="group")  # type: ignore[arg-type]


@pytest.mark.parametrize("axis", ["obs", "var"])
def test_rank_metabolites_requires_unique_identifiers(axis: str) -> None:
    adata = analysis_adata()
    if axis == "obs":
        adata.obs_names = ["duplicate"] * adata.n_obs
    else:
        adata.var_names = ["duplicate"] * adata.n_vars

    with pytest.raises(InputFormatError, match="unique"):
        rank_metabolites(adata, groupby="known_group")


@pytest.mark.parametrize("groupby", [None, "", "   ", 1, "missing"])
def test_rank_metabolites_rejects_invalid_or_missing_groupby(groupby: object) -> None:
    with pytest.raises(InputFormatError, match="groupby"):
        rank_metabolites(analysis_adata(), groupby=groupby)  # type: ignore[arg-type]


def test_rank_metabolites_excludes_missing_groups_without_source_mutation() -> None:
    missing = analysis_adata()
    missing.obs["known_group"] = ["a"] * 5 + [None] + ["b"] * 6
    before = missing.copy()

    ranked = rank_metabolites(missing, groupby="known_group")

    assert ranked["group"].drop_duplicates().tolist() == ["a", "b"]
    assert_adata_payload_unchanged(missing, before)


def test_rank_metabolites_requires_multiple_well_sampled_present_groups() -> None:

    one_group = analysis_adata()
    one_group.obs["known_group"] = "a"
    with pytest.raises(InputFormatError, match="at least 2 groups"):
        rank_metabolites(one_group, groupby="known_group")

    singleton = analysis_adata()
    singleton.obs["known_group"] = ["rare"] + ["common"] * 11
    with pytest.raises(InputFormatError, match="at least 2 observations"):
        rank_metabolites(singleton, groupby="known_group")


@pytest.mark.parametrize("method", [None, "", "logreg", "cosg", "cnmf", 1])
def test_rank_metabolites_rejects_unsupported_method(method: object) -> None:
    with pytest.raises(InputFormatError, match="method"):
        rank_metabolites(analysis_adata(), groupby="known_group", method=method)  # type: ignore[arg-type]


def test_rank_metabolites_validates_selected_layer_and_ignores_unselected_layers() -> None:
    missing = analysis_adata()
    with pytest.raises(InputFormatError, match="layer"):
        rank_metabolites(missing, groupby="known_group", layer="missing")

    malformed = analysis_adata()
    malformed.layers._data["bad"] = np.ones((1, 1))
    with pytest.raises(InputFormatError, match="shape"):
        rank_metabolites(malformed, groupby="known_group", layer="bad")

    selected = analysis_adata()
    selected.layers["signal"][0, 0] = np.nan
    with pytest.raises(InputFormatError, match="finite"):
        rank_metabolites(selected, groupby="known_group", layer="signal")

    ignored = analysis_adata()
    ignored.layers["other"][0, 0] = np.nan
    ranked = rank_metabolites(ignored, groupby="known_group", layer="signal")
    assert not ranked.empty


def test_rank_metabolites_rejects_nonfinite_scanpy_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_table(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": ["a"],
                "names": ["m0"],
                "scores": [np.nan],
                "logfoldchanges": [1.0],
                "pvals": [0.1],
                "pvals_adj": [0.2],
            }
        )

    monkeypatch.setattr(analysis_module.sc.get, "rank_genes_groups_df", invalid_table)

    with pytest.raises(InputFormatError, match="finite"):
        rank_metabolites(analysis_adata(), groupby="known_group")


@pytest.mark.parametrize("bad_pvalue", [np.nan, np.inf, -0.1, 1.1])
def test_rank_metabolites_rejects_invalid_scanpy_pvalues(
    monkeypatch: pytest.MonkeyPatch, bad_pvalue: float
) -> None:
    def invalid_table(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": ["b", "a"],
                "names": ["m0", "m0"],
                "scores": [np.inf, -np.inf],
                "logfoldchanges": [1.0, -1.0],
                "pvals": [bad_pvalue, 0.1],
                "pvals_adj": [0.2, 0.2],
            }
        )

    monkeypatch.setattr(analysis_module.sc.get, "rank_genes_groups_df", invalid_table)

    with pytest.raises(InputFormatError, match="pvals"):
        rank_metabolites(analysis_adata(), groupby="known_group")


def test_importing_joint_and_analysis_does_not_import_cosg_in_a_clean_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import joint; import joint.analysis; assert 'cosg' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_run_cosg_has_precise_missing_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cosg", None)

    with pytest.raises(OptionalDependencyError, match=r"joint-msi\[cosg\]") as raised:
        run_cosg(analysis_adata(), groupby="known_group", n_genes=3)

    assert isinstance(raised.value.__cause__, ImportError)


def test_run_cosg_does_not_mislabel_arbitrary_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def broken_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "cosg":
            raise RuntimeError("broken COSG installation")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "cosg", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(RuntimeError, match="broken COSG installation"):
        run_cosg(analysis_adata(), groupby="known_group", n_genes=3)


def test_run_cosg_chains_a_broken_backend_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cosg", types.SimpleNamespace(cosg=None))

    with pytest.raises(OptionalDependencyError, match=r"joint-msi\[cosg\]") as raised:
        run_cosg(analysis_adata(), groupby="known_group", n_genes=3)

    assert isinstance(raised.value.__cause__, TypeError)


def test_run_cosg_returns_original_wide_names_in_backend_field_order_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = analysis_adata()
    before = source.copy()
    calls: dict[str, object] = {}
    module = types.SimpleNamespace()

    def fake_cosg(
        adata: anndata.AnnData,
        *,
        key_added: str,
        groupby: str,
        n_genes_user: int,
        **kwargs: object,
    ) -> None:
        calls.update(
            {
                "adata": adata,
                "key_added": key_added,
                "groupby": groupby,
                "n_genes_user": n_genes_user,
                "kwargs": kwargs,
            }
        )
        adata.uns[key_added] = {
            "names": np.rec.array(
                [("m0", "m1"), ("m2", "m3")],
                dtype=[("a", "O"), ("b", "O")],
            )
        }

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)

    result = run_cosg(
        source,
        groupby="known_group",
        n_genes=2,
        key_added="markers",
        mu=3,
    )

    assert calls["adata"] is not source
    assert calls["key_added"] == "markers"
    assert calls["groupby"] == "known_group"
    assert calls["n_genes_user"] == 2
    assert calls["kwargs"] == {"mu": 3}
    assert result.to_dict("records") == [
        {"rank": 1, "a": "m0", "b": "m1"},
        {"rank": 2, "a": "m2", "b": "m3"},
    ]
    assert_adata_payload_unchanged(source, before)


def test_run_cosg_excludes_missing_group_values_on_the_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = analysis_adata()
    source.obs["known_group"] = ["a"] * 5 + [None] + ["b"] * 6
    module = types.SimpleNamespace()

    def fake_cosg(
        adata: anndata.AnnData,
        *,
        key_added: str,
        groupby: str,
        n_genes_user: int,
        **kwargs: object,
    ) -> None:
        assert adata.n_obs == 11
        assert adata.obs[groupby].cat.categories.tolist() == ["a", "b"]
        adata.uns[key_added] = {
            "names": np.array(
                [("m0", "m1")],
                dtype=[("a", "O"), ("b", "O")],
            )
        }

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)

    result = run_cosg(source, groupby="known_group", n_genes=1)

    assert result.to_dict("records") == [{"rank": 1, "a": "m0", "b": "m1"}]
    assert source.obs["known_group"].isna().sum() == 1


def test_run_cosg_uses_current_feature_source_by_default_despite_wider_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = analysis_adata()
    source.raw = anndata.AnnData(
        np.ones((source.n_obs, 7)),
        obs=source.obs.copy(),
        var=pd.DataFrame(index=[f"raw-{index}" for index in range(7)]),
    )
    module = types.SimpleNamespace()

    def fake_cosg(
        adata: anndata.AnnData,
        *,
        key_added: str,
        groupby: str,
        n_genes_user: int,
        **kwargs: object,
    ) -> None:
        assert n_genes_user == 5
        adata.uns[key_added] = {
            "names": np.array(
                [(f"m{rank}", f"m{rank}") for rank in range(5)],
                dtype=[("a", "O"), ("b", "O")],
            )
        }

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)

    result = run_cosg(source, groupby="known_group", n_genes=5)

    assert result.columns.tolist() == ["rank", "a", "b"]
    assert result["a"].tolist() == [f"m{rank}" for rank in range(5)]


def test_run_cosg_rebuilds_explicit_raw_working_copy_on_the_raw_feature_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = analysis_adata()
    raw_matrix = np.arange(source.n_obs * 7, dtype=float).reshape(source.n_obs, 7)
    source.raw = anndata.AnnData(
        raw_matrix,
        obs=source.obs.copy(),
        var=pd.DataFrame(index=[f"raw-{index}" for index in range(7)]),
    )
    before = source.copy()
    module = types.SimpleNamespace()

    def fake_cosg(
        adata: anndata.AnnData,
        *,
        key_added: str,
        groupby: str,
        n_genes_user: int,
        **kwargs: object,
    ) -> None:
        assert kwargs == {"groups": ["a"], "use_raw": True}
        assert adata.n_obs == 6
        assert adata.obs_names.tolist() == source.obs_names[:6].tolist()
        assert adata.obs[groupby].cat.categories.tolist() == ["a"]
        assert adata.var_names.tolist() == [f"raw-{index}" for index in range(7)]
        np.testing.assert_array_equal(adata.X, raw_matrix[:6])
        assert adata.raw is not None
        assert adata.raw.var_names.tolist() == [f"raw-{index}" for index in range(7)]
        np.testing.assert_array_equal(adata.raw.X, raw_matrix[:6])
        adata.uns[key_added] = {
            "names": np.array(
                [(name,) for name in adata.var_names],
                dtype=[("a", "O")],
            )
        }

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)

    result = run_cosg(
        source,
        groupby="known_group",
        n_genes=7,
        groups=["a"],
        use_raw=True,
    )

    assert result.to_dict("records") == [
        {"rank": rank + 1, "a": f"raw-{rank}"} for rank in range(7)
    ]
    assert_adata_payload_unchanged(source, before)


def test_run_cosg_forwards_groups_layer_use_raw_false_and_copy_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = analysis_adata()
    source.raw = anndata.AnnData(
        np.ones((source.n_obs, 7)),
        obs=source.obs.copy(),
        var=pd.DataFrame(index=[f"raw-{index}" for index in range(7)]),
    )
    calls: dict[str, object] = {}
    module = types.SimpleNamespace()

    def fake_cosg(
        adata: anndata.AnnData,
        *,
        key_added: str,
        groupby: str,
        n_genes_user: int,
        **kwargs: object,
    ) -> None:
        calls.update(kwargs)
        adata.uns[key_added] = {"names": np.array([("m0",)], dtype=[("a", "O")])}

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)

    result = run_cosg(
        source,
        groupby="known_group",
        n_genes=1,
        groups=["a"],
        layer="signal",
        use_raw=False,
        copy=False,
    )

    assert calls == {
        "groups": ["a"],
        "layer": "signal",
        "use_raw": False,
        "copy": False,
    }
    assert result.to_dict("records") == [{"rank": 1, "a": "m0"}]


@pytest.mark.parametrize("use_raw", [None, 1, "yes"])
def test_run_cosg_requires_boolean_explicit_use_raw_before_loading_dependency(
    use_raw: object,
) -> None:
    with pytest.raises(InputFormatError, match="use_raw.*boolean"):
        run_cosg(analysis_adata(), groupby="known_group", n_genes=1, use_raw=use_raw)


def test_run_cosg_rejects_layer_with_use_raw_true_before_loading_dependency() -> None:
    with pytest.raises(InputFormatError, match="COSG.*layer.*use_raw"):
        run_cosg(
            analysis_adata(),
            groupby="known_group",
            n_genes=1,
            layer="signal",
            use_raw=True,
        )


def test_run_cosg_limits_backend_copy_to_selected_groups_and_removes_unused_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = analysis_adata()
    source.obs["known_group"] = pd.Categorical(
        source.obs["known_group"],
        categories=["unused", "b", "a"],
        ordered=True,
    )
    module = types.SimpleNamespace()

    def fake_cosg(
        adata: anndata.AnnData,
        *,
        key_added: str,
        groupby: str,
        n_genes_user: int,
        **kwargs: object,
    ) -> None:
        assert kwargs["groups"] == ["a"]
        assert adata.n_obs == 6
        assert adata.obs[groupby].cat.categories.tolist() == ["a"]
        assert adata.obs[groupby].tolist() == ["a"] * 6
        adata.uns[key_added] = {"names": np.array([("m0",)], dtype=[("a", "O")])}

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)

    result = run_cosg(source, groupby="known_group", n_genes=1, groups=["a"])

    assert result.to_dict("records") == [{"rank": 1, "a": "m0"}]


def test_run_cosg_validates_numpy_group_selection_without_reordering_backend_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_groups = np.array(["b", "a"])
    module = types.SimpleNamespace()

    def fake_cosg(
        adata: anndata.AnnData,
        *,
        key_added: str,
        groupby: str,
        n_genes_user: int,
        **kwargs: object,
    ) -> None:
        assert kwargs["groups"] is requested_groups
        adata.uns[key_added] = {"names": np.array([("m0", "m1")], dtype=[("a", "O"), ("b", "O")])}

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)

    result = run_cosg(
        analysis_adata(),
        groupby="known_group",
        n_genes=1,
        groups=requested_groups,
    )

    assert result.to_dict("records") == [{"rank": 1, "a": "m0", "b": "m1"}]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_raw": False},
        {"layer": "signal"},
    ],
)
def test_run_cosg_rejects_marker_ids_from_an_inactive_feature_source(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
) -> None:
    source = analysis_adata()
    source.raw = anndata.AnnData(
        source.X.copy(),
        obs=source.obs.copy(),
        var=pd.DataFrame(index=[f"raw-{index}" for index in range(source.n_vars)]),
    )
    module = types.SimpleNamespace()

    def fake_cosg(
        adata: anndata.AnnData,
        *,
        key_added: str,
        groupby: str,
        n_genes_user: int,
        **cosg_kwargs: object,
    ) -> None:
        adata.uns[key_added] = {
            "names": np.array([("raw-0", "raw-1")], dtype=[("a", "O"), ("b", "O")])
        }

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)

    with pytest.raises(InputFormatError, match="feature IDs"):
        run_cosg(source, groupby="known_group", n_genes=1, **kwargs)


@pytest.mark.parametrize(
    "groupby",
    [None, "", "   ", "missing"],
)
def test_run_cosg_validates_groupby_before_loading_optional_dependency(groupby: object) -> None:
    with pytest.raises(InputFormatError, match="groupby"):
        run_cosg(analysis_adata(), groupby=groupby, n_genes=2)  # type: ignore[arg-type]


def test_run_cosg_rejects_invalid_group_values_before_loading_optional_dependency() -> None:
    one_group = analysis_adata()
    one_group.obs["known_group"] = "a"
    with pytest.raises(InputFormatError, match="at least 2 groups"):
        run_cosg(one_group, groupby="known_group", n_genes=2)

    ambiguous_strings = analysis_adata()
    ambiguous_strings.obs["known_group"] = [1] * 6 + ["1"] * 6
    with pytest.raises(InputFormatError, match="unique when represented as strings"):
        run_cosg(ambiguous_strings, groupby="known_group", n_genes=2)


@pytest.mark.parametrize("n_genes", [True, 0, -1, 1.5, 6])
def test_run_cosg_validates_n_genes_before_loading_optional_dependency(n_genes: object) -> None:
    with pytest.raises(InputFormatError, match="n_genes"):
        run_cosg(analysis_adata(), groupby="known_group", n_genes=n_genes)  # type: ignore[arg-type]


@pytest.mark.parametrize("key_added", [None, "", "   ", 1])
def test_run_cosg_validates_key_added_before_loading_optional_dependency(key_added: object) -> None:
    with pytest.raises(InputFormatError, match="key_added"):
        run_cosg(
            analysis_adata(),
            groupby="known_group",
            n_genes=2,
            key_added=key_added,  # type: ignore[arg-type]
        )


def test_run_cosg_rejects_existing_output_key_and_true_conflicting_kwargs_before_loading_dependency() -> (
    None
):
    existing = analysis_adata()
    existing.uns["markers"] = {"prior": "result"}
    with pytest.raises(InputFormatError, match="already contains"):
        run_cosg(existing, groupby="known_group", n_genes=2, key_added="markers")

    with pytest.raises(InputFormatError, match="reserved"):
        run_cosg(analysis_adata(), groupby="known_group", n_genes=2, n_genes_user=3)

    for name, value in {"copy": True, "copy_invalid": "not-forwarded"}.items():
        if name == "copy_invalid":
            name = "copy"
        with pytest.raises(InputFormatError, match="reserved"):
            run_cosg(analysis_adata(), groupby="known_group", n_genes=2, **{name: value})


@pytest.mark.parametrize(
    "broken_x",
    [
        np.full((12, 5), "bad"),
        np.full((12, 5), np.nan),
    ],
)
def test_run_cosg_validates_x_before_loading_optional_dependency(broken_x: np.ndarray) -> None:
    source = analysis_adata()
    source.X = broken_x

    with pytest.raises(InputFormatError, match="adata.X"):
        run_cosg(source, groupby="known_group", n_genes=2, use_raw=False)


def test_run_cosg_validates_raw_and_layer_feature_sources_before_loading_optional_dependency() -> (
    None
):
    raw_broken = analysis_adata()
    raw_broken.raw = anndata.AnnData(
        np.full((raw_broken.n_obs, 6), np.nan),
        obs=raw_broken.obs.copy(),
        var=pd.DataFrame(index=[f"raw-{index}" for index in range(6)]),
    )
    with pytest.raises(InputFormatError, match="adata.raw.X"):
        run_cosg(raw_broken, groupby="known_group", n_genes=2, use_raw=True)

    layer_broken = analysis_adata()
    layer_broken.layers["signal"][0, 0] = np.nan
    with pytest.raises(InputFormatError, match=r'layers\["signal"\]'):
        run_cosg(layer_broken, groupby="known_group", n_genes=2, layer="signal")


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"names": np.array(["m0"], dtype=object)},
        {"names": np.array([("m0", "m1")], dtype=[("a", "O"), ("wrong", "O")])},
        {"names": np.array([("missing", "m1")], dtype=[("a", "O"), ("b", "O")])},
        {"names": np.array([(None, "m1")], dtype=[("a", "O"), ("b", "O")])},
    ],
)
def test_run_cosg_translates_malformed_backend_output_to_input_format_error(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    module = types.SimpleNamespace()

    def fake_cosg(
        adata: anndata.AnnData,
        *,
        key_added: str,
        groupby: str,
        n_genes_user: int,
        **kwargs: object,
    ) -> None:
        if payload is not None:
            adata.uns[key_added] = payload

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)

    with pytest.raises(InputFormatError, match="COSG"):
        run_cosg(analysis_adata(), groupby="known_group", n_genes=1)


class FakeCnmf:
    calls: ClassVar[list[tuple[str, object]]] = []

    def __init__(self, output_dir: str, name: str) -> None:
        self.output_dir = output_dir
        self.name = name
        self.calls.append(("init", {"output_dir": output_dir, "name": name}))

    def prepare(
        self,
        *,
        counts_fn: str,
        components: np.ndarray,
        seed: int,
        num_highvar_genes: int,
    ) -> None:
        self.calls.append(
            (
                "prepare",
                {
                    "counts_fn": counts_fn,
                    "components": components,
                    "seed": seed,
                    "num_highvar_genes": num_highvar_genes,
                },
            )
        )

    def factorize(self, *, worker_i: int, total_workers: int) -> None:
        self.calls.append(("factorize", {"worker_i": worker_i, "total_workers": total_workers}))

    def combine(self) -> None:
        self.calls.append(("combine", {}))

    def k_selection_plot(self) -> None:
        self.calls.append(("k_selection_plot", {}))

    def consensus(self, *, k: int, density_threshold: float) -> None:
        self.calls.append(("consensus", {"k": k, "density_threshold": density_threshold}))


@pytest.fixture
def fake_cnmf(monkeypatch: pytest.MonkeyPatch) -> type[FakeCnmf]:
    FakeCnmf.calls = []
    monkeypatch.setitem(sys.modules, "cnmf", types.SimpleNamespace(cNMF=FakeCnmf))
    return FakeCnmf


def test_importing_joint_and_analysis_does_not_import_cnmf_in_a_clean_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import joint; import joint.analysis; assert 'cnmf' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_prepare_cnmf_imports_backend_before_mutating_filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "not-created"
    monkeypatch.setitem(sys.modules, "cnmf", None)

    with pytest.raises(OptionalDependencyError) as raised:
        prepare_cnmf(
            analysis_adata(),
            destination,
            name="sample",
            components=[5, 6],
            seed=14,
            num_highvar_genes=3,
        )

    assert str(raised.value) == ("cNMF support requires `python -m pip install 'joint-msi[cnmf]'`")
    assert isinstance(raised.value.__cause__, ImportError)
    assert not destination.exists()


def test_prepare_cnmf_does_not_mislabel_arbitrary_import_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_import = builtins.__import__

    def broken_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "cnmf":
            raise RuntimeError("broken cNMF installation")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "cnmf", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(RuntimeError, match="broken cNMF installation"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path / "not-created",
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )


def test_prepare_cnmf_chains_broken_backend_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "cnmf", types.SimpleNamespace(cNMF=None))

    with pytest.raises(TypeError, match="callable cNMF"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path / "not-created",
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )


def test_prepare_cnmf_propagates_transitive_module_not_found_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_import = builtins.__import__

    def broken_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "cnmf":
            raise ModuleNotFoundError(
                "No module named 'transitive_backend'", name="transitive_backend"
            )
        return original_import(name, *args, **kwargs)

    destination = tmp_path / "not-created"
    monkeypatch.delitem(sys.modules, "cnmf", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(ModuleNotFoundError) as raised:
        prepare_cnmf(
            analysis_adata(),
            destination,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert raised.value.name == "transitive_backend"
    assert not destination.exists()


def test_prepare_and_run_cnmf_call_documented_backend_methods_and_retain_settings(
    fake_cnmf: type[FakeCnmf], tmp_path: Path
) -> None:
    source = analysis_adata()
    before = source.copy()

    prepared = prepare_cnmf(
        source,
        tmp_path,
        name="sample",
        components=[5, 6],
        seed=14,
        num_highvar_genes=3,
    )
    returned = run_cnmf(prepared, worker_index=0, total_workers=1, selected_k=6)

    assert isinstance(prepared, PreparedCnmf)
    assert returned is prepared
    assert [name for name, _ in fake_cnmf.calls] == [
        "init",
        "prepare",
        "factorize",
        "combine",
        "k_selection_plot",
        "consensus",
    ]
    assert fake_cnmf.calls[0][1] == {"output_dir": str(tmp_path.resolve()), "name": "sample"}
    prepare_kwargs = fake_cnmf.calls[1][1]
    assert isinstance(prepare_kwargs, dict)
    assert prepare_kwargs["counts_fn"] == str(tmp_path.resolve() / "sample.h5ad")
    np.testing.assert_array_equal(prepare_kwargs["components"], np.array([5, 6]))
    assert prepare_kwargs["seed"] == 14
    assert prepare_kwargs["num_highvar_genes"] == 3
    assert fake_cnmf.calls[2][1] == {"worker_i": 0, "total_workers": 1}
    assert fake_cnmf.calls[3][1] == {}
    assert fake_cnmf.calls[4][1] == {}
    assert fake_cnmf.calls[5][1] == {"k": 6, "density_threshold": 0.5}
    assert prepared.counts_path == tmp_path.resolve() / "sample.h5ad"
    assert prepared.output_dir == tmp_path.resolve()
    assert prepared.name == "sample"
    assert prepared.task_dir == tmp_path.resolve() / "sample"
    assert prepared.task_dir.is_dir()
    assert prepared.components == (5, 6)
    assert prepared.seed == 14
    assert prepared.num_highvar_genes == 3
    assert anndata.read_h5ad(prepared.counts_path).obs_names.equals(source.obs_names)
    assert prepared.staging_dir.parent == tmp_path.resolve()
    assert prepared.staging_dir.name.startswith(".joint-cnmf-")
    staged_counts = prepared.staging_dir / "counts.h5ad"
    assert staged_counts.is_file()
    assert staged_counts.stat().st_ino == prepared.counts_path.stat().st_ino
    assert_adata_payload_unchanged(source, before)
    with pytest.raises(FrozenInstanceError):
        prepared.seed = 15  # type: ignore[misc]


@pytest.mark.parametrize("axis", ["obs", "var"])
def test_prepare_cnmf_requires_unique_anndata_axes_before_backend_construction(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, axis: str
) -> None:
    source = analysis_adata()
    if axis == "obs":
        source.obs_names = ["duplicate"] * source.n_obs
    else:
        source.var_names = ["duplicate"] * source.n_vars

    with pytest.raises(InputFormatError, match="unique"):
        prepare_cnmf(
            source,
            tmp_path,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert fake_cnmf.calls == []


@pytest.mark.parametrize(
    "matrix",
    [
        np.full((12, 5), "bad"),
        np.full((12, 5), np.nan),
        np.full((12, 5), np.inf),
        -np.ones((12, 5)),
    ],
)
def test_prepare_cnmf_requires_finite_real_nonnegative_counts(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, matrix: np.ndarray
) -> None:
    source = analysis_adata()
    source.X = matrix

    with pytest.raises(InputFormatError, match="adata.X|counts"):
        prepare_cnmf(
            source,
            tmp_path,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert fake_cnmf.calls == []
    assert not (tmp_path / "sample.h5ad").exists()


@pytest.mark.parametrize("output_dir", [None, b"bad", 1, "", "   "])
def test_prepare_cnmf_rejects_unsafe_output_path_types(
    fake_cnmf: type[FakeCnmf], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output_dir: object
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(InputFormatError, match="output_dir"):
        prepare_cnmf(
            analysis_adata(),
            output_dir,  # type: ignore[arg-type]
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )
    assert fake_cnmf.calls == []


@pytest.mark.parametrize("name", [None, "", "   ", ".", "..", "../sample", "a/b", r"a\b"])
def test_prepare_cnmf_requires_safe_single_component_name(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, name: object
) -> None:
    with pytest.raises(InputFormatError, match="name"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name=name,  # type: ignore[arg-type]
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )
    assert fake_cnmf.calls == []


@pytest.mark.parametrize("name", ["bad\nname", "bad\x1fname", "bad\x7fname", "bad\u200bname"])
def test_prepare_cnmf_rejects_control_and_format_characters_before_path_checks(
    fake_cnmf: type[FakeCnmf], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    def unexpected_exists(self: Path) -> bool:
        raise AssertionError(f"path existence checked for invalid name: {self}")

    monkeypatch.setattr(Path, "exists", unexpected_exists)

    with pytest.raises(InputFormatError, match="name"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name=name,
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert fake_cnmf.calls == []


@pytest.mark.parametrize("name", ["a" * 101, "é" * 51, "\ud800"])
def test_prepare_cnmf_rejects_overlong_or_non_utf8_name_before_path_checks(
    fake_cnmf: type[FakeCnmf], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    def unexpected_exists(self: Path) -> bool:
        raise AssertionError(f"path existence checked for invalid name: {self}")

    monkeypatch.setattr(Path, "exists", unexpected_exists)

    with pytest.raises(InputFormatError, match="name"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name=name,
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert fake_cnmf.calls == []


def test_prepare_cnmf_supports_maximum_conservative_name_length(
    fake_cnmf: type[FakeCnmf], tmp_path: Path
) -> None:
    name = "a" * 100

    prepared = prepare_cnmf(
        analysis_adata(),
        tmp_path,
        name=name,
        components=[5],
        seed=14,
        num_highvar_genes=3,
    )

    assert prepared.counts_path.name == f"{name}.h5ad"
    assert prepared.counts_path.exists()
    assert (prepared.staging_dir / "counts.h5ad").exists()


@pytest.mark.parametrize(
    "components",
    [None, [], [0], [-1], [True], [1.5], [5, 5], "5"],
)
def test_prepare_cnmf_requires_nonempty_unique_positive_components(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, components: object
) -> None:
    with pytest.raises(InputFormatError, match="components"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name="sample",
            components=components,  # type: ignore[arg-type]
            seed=14,
            num_highvar_genes=3,
        )
    assert fake_cnmf.calls == []


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_prepare_cnmf_validates_seed(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, seed: object
) -> None:
    with pytest.raises(InputFormatError, match="seed"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name="sample",
            components=[5],
            seed=seed,  # type: ignore[arg-type]
            num_highvar_genes=3,
        )
    assert fake_cnmf.calls == []


@pytest.mark.parametrize("num_highvar_genes", [True, 0, -1, 1.5, 6])
def test_prepare_cnmf_bounds_high_variable_gene_count_to_feature_axis(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, num_highvar_genes: object
) -> None:
    with pytest.raises(InputFormatError, match="num_highvar_genes"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=num_highvar_genes,  # type: ignore[arg-type]
        )
    assert fake_cnmf.calls == []


@pytest.mark.parametrize("existing", ["counts", "task"])
def test_prepare_cnmf_never_silently_overwrites_existing_outputs(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, existing: str
) -> None:
    if existing == "counts":
        (tmp_path / "sample.h5ad").write_text("keep me")
    else:
        (tmp_path / "sample").mkdir()

    with pytest.raises(InputFormatError, match="already exists"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert fake_cnmf.calls == []
    if existing == "counts":
        assert (tmp_path / "sample.h5ad").read_text() == "keep me"


def test_prepare_cnmf_atomically_reserves_task_directory_before_backend_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class BackendMustNotConstruct(FakeCnmf):
        def __init__(self, output_dir: str, name: str) -> None:
            calls.append("init")
            super().__init__(output_dir, name)

    original_mkdir = Path.mkdir
    injected = False

    def race_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        nonlocal injected
        original_mkdir(self, *args, **kwargs)
        if self == tmp_path.resolve() and not injected:
            injected = True
            original_mkdir(tmp_path / "sample")

    monkeypatch.setitem(sys.modules, "cnmf", types.SimpleNamespace(cNMF=BackendMustNotConstruct))
    monkeypatch.setattr(Path, "mkdir", race_mkdir)

    with pytest.raises(InputFormatError, match="task output already exists"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert calls == []
    assert (tmp_path / "sample").is_dir()
    assert not (tmp_path / "sample.h5ad").exists()


def test_prepare_cnmf_leaves_owned_temporary_directory_after_write_failure(
    fake_cnmf: type[FakeCnmf], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_write(
        self: anndata.AnnData, filename: object, *args: object, **kwargs: object
    ) -> None:
        del self, filename, args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(anndata.AnnData, "write_h5ad", fail_write)

    with pytest.raises(InputFormatError, match="Could not write cNMF counts"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert fake_cnmf.calls == []
    assert not (tmp_path / "sample.h5ad").exists()
    temporary_directories = list(tmp_path.glob(".joint-cnmf-*"))
    assert len(temporary_directories) == 1
    assert temporary_directories[0].is_dir()


def test_prepare_cnmf_does_not_unlink_replaced_temporary_file(
    fake_cnmf: type[FakeCnmf], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def replace_then_fail(
        self: anndata.AnnData, filename: object, *args: object, **kwargs: object
    ) -> None:
        del self, args, kwargs
        path = Path(filename)  # type: ignore[arg-type]
        path.write_text("concurrent replacement")
        raise OSError("disk full")

    monkeypatch.setattr(anndata.AnnData, "write_h5ad", replace_then_fail)

    with pytest.raises(InputFormatError, match="Could not write cNMF counts"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    replacements = list(tmp_path.glob(".joint-cnmf-*/counts.h5ad"))
    assert len(replacements) == 1
    assert replacements[0].read_text() == "concurrent replacement"
    assert fake_cnmf.calls == []


def test_prepare_cnmf_never_deletes_staging_paths_after_successful_promotion(
    fake_cnmf: type[FakeCnmf], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden_unlink(self: Path, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError(f"successful staging path was unlinked: {self}")

    def forbidden_rmdir(self: Path) -> None:
        raise AssertionError(f"successful staging directory was removed: {self}")

    monkeypatch.setattr(Path, "unlink", forbidden_unlink)
    monkeypatch.setattr(Path, "rmdir", forbidden_rmdir)

    prepared = prepare_cnmf(
        analysis_adata(),
        tmp_path,
        name="sample",
        components=[5],
        seed=14,
        num_highvar_genes=3,
    )

    assert (prepared.staging_dir / "counts.h5ad").exists()
    assert prepared.counts_path.exists()


def test_prepare_cnmf_cleanup_failure_does_not_mask_count_write_error(
    fake_cnmf: type[FakeCnmf], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_write(
        self: anndata.AnnData, filename: object, *args: object, **kwargs: object
    ) -> None:
        del self, filename, args, kwargs
        raise OSError("disk full")

    def fail_unlink(self: Path, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(anndata.AnnData, "write_h5ad", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(InputFormatError, match="disk full") as raised:
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert fake_cnmf.calls == []


def test_prepare_cnmf_wraps_output_directory_creation_failure(
    fake_cnmf: type[FakeCnmf], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "blocked"

    def fail_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(InputFormatError, match="Could not create cNMF output directory"):
        prepare_cnmf(
            analysis_adata(),
            destination,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert fake_cnmf.calls == []


def test_prepare_cnmf_preserves_promoted_counts_and_backend_artifacts_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingCnmf(FakeCnmf):
        def prepare(self, **kwargs: object) -> None:
            counts = Path(str(kwargs["counts_fn"]))
            counts.unlink()
            counts.write_text("concurrent replacement")
            task_dir = Path(self.output_dir) / self.name
            assert task_dir.is_dir()
            (task_dir / "diagnostic.txt").write_text("keep for diagnosis")
            raise RuntimeError("prepare failed")

    monkeypatch.setitem(sys.modules, "cnmf", types.SimpleNamespace(cNMF=FailingCnmf))

    with pytest.raises(RuntimeError, match="prepare failed"):
        prepare_cnmf(
            analysis_adata(),
            tmp_path,
            name="sample",
            components=[5],
            seed=14,
            num_highvar_genes=3,
        )

    assert (tmp_path / "sample.h5ad").read_text() == "concurrent replacement"
    assert (tmp_path / "sample" / "diagnostic.txt").read_text() == "keep for diagnosis"
    staged = list(tmp_path.glob(".joint-cnmf-*/counts.h5ad"))
    assert len(staged) == 1
    assert staged[0].exists()


@pytest.mark.parametrize("prepared", [None, object(), "prepared"])
def test_run_cnmf_validates_prepared_type(prepared: object) -> None:
    with pytest.raises(InputFormatError, match="PreparedCnmf"):
        run_cnmf(prepared, worker_index=0, total_workers=1, selected_k=5)  # type: ignore[arg-type]


@pytest.mark.parametrize("phase", [factorize_cnmf, select_cnmf_k, consensus_cnmf])
def test_cnmf_phase_apis_validate_prepared_type(phase) -> None:
    with pytest.raises(InputFormatError, match="PreparedCnmf"):
        if phase is factorize_cnmf:
            phase(object(), worker_index=0, total_workers=1)
        elif phase is consensus_cnmf:
            phase(object(), selected_k=5)
        else:
            phase(object())


def test_cnmf_separate_phases_support_multiworker_execution(
    fake_cnmf: type[FakeCnmf], tmp_path: Path
) -> None:
    prepared = prepare_cnmf(
        analysis_adata(),
        tmp_path,
        name="sample",
        components=[5, 6],
        seed=14,
        num_highvar_genes=3,
    )

    factorize_cnmf(prepared, worker_index=0, total_workers=2)
    factorize_cnmf(prepared, worker_index=1, total_workers=2)
    select_cnmf_k(prepared)
    consensus_cnmf(prepared, selected_k=6, density_threshold=0.4)

    assert fake_cnmf.calls[2:] == [
        ("factorize", {"worker_i": 0, "total_workers": 2}),
        ("factorize", {"worker_i": 1, "total_workers": 2}),
        ("combine", {}),
        ("k_selection_plot", {}),
        ("consensus", {"k": 6, "density_threshold": 0.4}),
    ]


def test_run_cnmf_rejects_multiworker_orchestration_before_backend_calls(
    fake_cnmf: type[FakeCnmf], tmp_path: Path
) -> None:
    prepared = prepare_cnmf(
        analysis_adata(),
        tmp_path,
        name="sample",
        components=[5],
        seed=14,
        num_highvar_genes=3,
    )
    calls_before = list(fake_cnmf.calls)

    with pytest.raises(
        InputFormatError,
        match="factorize_cnmf.*select_cnmf_k.*consensus_cnmf",
    ):
        run_cnmf(prepared, worker_index=0, total_workers=2, selected_k=5)

    assert fake_cnmf.calls == calls_before


@pytest.mark.parametrize(
    ("worker_index", "total_workers"),
    [(True, 1), (0, True), (-1, 1), (1, 1), (0, 0), (0, -1), (0, 1.5)],
)
def test_run_cnmf_validates_zero_based_worker_range(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, worker_index: object, total_workers: object
) -> None:
    prepared = prepare_cnmf(
        analysis_adata(),
        tmp_path,
        name="sample",
        components=[5, 6],
        seed=14,
        num_highvar_genes=3,
    )
    calls_before = list(fake_cnmf.calls)

    with pytest.raises(InputFormatError, match="worker"):
        run_cnmf(
            prepared,
            worker_index=worker_index,  # type: ignore[arg-type]
            total_workers=total_workers,  # type: ignore[arg-type]
            selected_k=5,
        )

    assert fake_cnmf.calls == calls_before


@pytest.mark.parametrize("selected_k", [True, 0, 1.5, 7])
def test_run_cnmf_requires_selected_k_from_prepared_components(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, selected_k: object
) -> None:
    prepared = prepare_cnmf(
        analysis_adata(),
        tmp_path,
        name="sample",
        components=[5, 6],
        seed=14,
        num_highvar_genes=3,
    )
    calls_before = list(fake_cnmf.calls)

    with pytest.raises(InputFormatError, match="selected_k"):
        run_cnmf(
            prepared,
            worker_index=0,
            total_workers=1,
            selected_k=selected_k,  # type: ignore[arg-type]
        )

    assert fake_cnmf.calls == calls_before


@pytest.mark.parametrize("density_threshold", [True, np.nan, np.inf, -np.inf, "0.5"])
def test_run_cnmf_requires_finite_real_density_threshold(
    fake_cnmf: type[FakeCnmf], tmp_path: Path, density_threshold: object
) -> None:
    prepared = prepare_cnmf(
        analysis_adata(),
        tmp_path,
        name="sample",
        components=[5],
        seed=14,
        num_highvar_genes=3,
    )
    calls_before = list(fake_cnmf.calls)

    with pytest.raises(InputFormatError, match="density_threshold"):
        run_cnmf(
            prepared,
            worker_index=0,
            total_workers=1,
            selected_k=5,
            density_threshold=density_threshold,  # type: ignore[arg-type]
        )

    assert fake_cnmf.calls == calls_before


def test_load_cnmf_results_aligns_exact_usage_set_by_observation_name_without_mutation(
    tmp_path: Path,
) -> None:
    source = analysis_adata()
    before = source.copy()
    usage = pd.DataFrame(
        {"Usage_1": np.arange(source.n_obs, dtype=float)},
        index=source.obs_names[::-1],
    )
    path = tmp_path / "usage.tsv"
    usage.to_csv(path, sep="\t")

    loaded = load_cnmf_results(source, path)

    assert loaded is not source
    assert loaded.obs_names.equals(source.obs_names)
    assert loaded.obs["Usage_1"].tolist() == list(reversed(range(source.n_obs)))
    assert loaded.uns["joint"]["cnmf_usage_path"] == str(path.resolve())
    assert sparse.issparse(loaded.X) == sparse.issparse(source.X)
    assert_adata_payload_unchanged(source, before)


@pytest.mark.parametrize("usage_path", [None, b"bad", 1, "", "   "])
def test_load_cnmf_results_rejects_unsafe_path_types(usage_path: object) -> None:
    with pytest.raises(InputFormatError, match="usage_path"):
        load_cnmf_results(analysis_adata(), usage_path)  # type: ignore[arg-type]


def test_load_cnmf_results_wraps_missing_and_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(InputFormatError, match="Could not read cNMF usage"):
        load_cnmf_results(analysis_adata(), tmp_path / "missing.tsv")
    with pytest.raises(InputFormatError, match="Could not read cNMF usage"):
        load_cnmf_results(analysis_adata(), tmp_path)


def test_load_cnmf_results_requires_unique_exact_observation_set(tmp_path: Path) -> None:
    source = analysis_adata()
    duplicate = pd.DataFrame(
        {"Usage_1": np.arange(source.n_obs, dtype=float)},
        index=[source.obs_names[0]] * 2 + source.obs_names[2:].tolist(),
    )
    duplicate_path = tmp_path / "duplicate.tsv"
    duplicate.to_csv(duplicate_path, sep="\t")
    with pytest.raises(InputFormatError, match="identifiers.*unique"):
        load_cnmf_results(source, duplicate_path)

    mismatched = pd.DataFrame(
        {"Usage_1": np.arange(source.n_obs, dtype=float)},
        index=source.obs_names[:-1].tolist() + ["extra"],
    )
    mismatch_path = tmp_path / "mismatch.tsv"
    mismatched.to_csv(mismatch_path, sep="\t")
    with pytest.raises(InputFormatError, match="exactly match"):
        load_cnmf_results(source, mismatch_path)


def test_load_cnmf_results_preserves_leading_zero_observation_identifiers(tmp_path: Path) -> None:
    source = anndata.AnnData(
        np.ones((2, 1)),
        obs=pd.DataFrame(index=["001", "01"]),
        var=pd.DataFrame(index=["m0"]),
    )
    path = tmp_path / "leading-zero.tsv"
    path.write_text("\tUsage_1\n01\t2\n001\t1\n")

    loaded = load_cnmf_results(source, path)

    assert loaded.obs["Usage_1"].tolist() == [1, 2]


def test_load_cnmf_results_requires_unique_noncolliding_usage_columns(tmp_path: Path) -> None:
    source = analysis_adata()
    source.obs["existing"] = np.arange(source.n_obs)
    duplicate_path = tmp_path / "duplicate-columns.tsv"
    duplicate_path.write_text(
        "\tUsage_1\tUsage_1\n" + "\n".join(f"{name}\t1\t2" for name in source.obs_names) + "\n"
    )
    with pytest.raises(InputFormatError, match="columns.*unique"):
        load_cnmf_results(source, duplicate_path)

    collision = pd.DataFrame({"existing": np.ones(source.n_obs)}, index=source.obs_names)
    collision_path = tmp_path / "collision.tsv"
    collision.to_csv(collision_path, sep="\t")
    with pytest.raises(InputFormatError, match="collide"):
        load_cnmf_results(source, collision_path)


@pytest.mark.parametrize("value", ["bad", np.nan, np.inf, -1.0])
def test_load_cnmf_results_requires_numeric_finite_nonnegative_usage(
    tmp_path: Path, value: object
) -> None:
    source = analysis_adata()
    usage = pd.DataFrame(
        {"Usage_1": [value, *np.ones(source.n_obs - 1)]},
        index=source.obs_names,
    )
    path = tmp_path / "bad-usage.tsv"
    usage.to_csv(path, sep="\t")

    with pytest.raises(InputFormatError, match="numeric|finite|nonnegative"):
        load_cnmf_results(source, path)


def test_load_cnmf_results_validates_source_and_joint_metadata(tmp_path: Path) -> None:
    source = analysis_adata()
    source.uns["joint"] = "invalid"
    usage = pd.DataFrame({"Usage_1": np.ones(source.n_obs)}, index=source.obs_names)
    path = tmp_path / "usage.tsv"
    usage.to_csv(path, sep="\t")

    with pytest.raises(InputFormatError, match=r"uns\['joint'\].*mapping"):
        load_cnmf_results(source, path)

    duplicate_obs = analysis_adata()
    duplicate_obs.obs_names = ["duplicate"] * duplicate_obs.n_obs
    with pytest.raises(InputFormatError, match="unique"):
        load_cnmf_results(duplicate_obs, path)
