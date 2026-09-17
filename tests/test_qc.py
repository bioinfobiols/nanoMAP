from decimal import Decimal
from fractions import Fraction

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array, csr_matrix
from scipy.spatial.distance import pdist, squareform

import joint.qc as qc_module
from joint.errors import InputFormatError
from joint.models import QuantificationResult, RegistrationResult
from joint.qc import (
    coverage_report,
    detect_outliers,
    pcoa,
    registration_report,
    silhouette_summary,
    spectral_qc,
    validate_anndata,
)


def sample_adata() -> anndata.AnnData:
    return anndata.AnnData(
        csr_matrix([[1.0, 0.0], [2.0, 3.0], [100.0, 100.0]]),
        obs=pd.DataFrame({"dataset": ["s", "s", "s"]}, index=["z", "a", "m"]),
        var={"mz": [100.0, 200.0]},
        obsm={"spatial": np.array([[0, 0], [1, 0], [2, 0]])},
    )


def test_validate_anndata_requires_anndata() -> None:
    with pytest.raises(InputFormatError, match="AnnData"):
        validate_anndata(np.ones((2, 2)))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mz", "match"),
    [
        (None, 'column "mz"'),
        (["one", "two"], "numeric"),
        ([100.0, np.nan], "finite"),
        ([100.0, np.inf], "finite"),
    ],
)
def test_validate_anndata_requires_numeric_finite_mz(mz, match: str) -> None:
    adata = sample_adata()
    if mz is None:
        del adata.var["mz"]
    else:
        adata.var["mz"] = mz

    with pytest.raises(InputFormatError, match=match):
        validate_anndata(adata)


@pytest.mark.parametrize(
    ("spatial", "match"),
    [
        (None, "spatial"),
        (np.ones((3, 3)), "shape"),
        (np.array([[0, 0], [1, 0], ["x", 0]], dtype=object), "numeric"),
        (np.array([[0, 0], [1, 0], [np.nan, 0]]), "finite"),
    ],
)
def test_validate_anndata_requires_valid_spatial_coordinates(spatial, match: str) -> None:
    adata = sample_adata()
    if spatial is None:
        del adata.obsm["spatial"]
    else:
        adata.obsm["spatial"] = spatial

    with pytest.raises(InputFormatError, match=match):
        validate_anndata(adata, require_spatial=True)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_validate_anndata_rejects_nonfinite_matrix_and_layers(value: float) -> None:
    broken_x = sample_adata()
    broken_x.X = np.array([[1.0, value], [2.0, 3.0], [4.0, 5.0]])
    with pytest.raises(InputFormatError, match="adata.X.*finite"):
        validate_anndata(broken_x)

    broken_layer = sample_adata()
    broken_layer.layers["bad"] = csr_matrix([[1.0, value], [2.0, 3.0], [4.0, 5.0]])
    with pytest.raises(InputFormatError, match=r'layers\["bad"\].*finite'):
        validate_anndata(broken_layer)


def test_validate_anndata_rejects_boolean_spectra_mz_and_spatial() -> None:
    broken_x = sample_adata()
    broken_x.X = np.ones((3, 2), dtype=bool)
    with pytest.raises(InputFormatError, match="numeric"):
        validate_anndata(broken_x)

    broken_mz = sample_adata()
    broken_mz.var["mz"] = [True, False]
    with pytest.raises(InputFormatError, match="numeric"):
        validate_anndata(broken_mz)

    broken_spatial = sample_adata()
    broken_spatial.obsm["spatial"] = np.ones((3, 2), dtype=bool)
    with pytest.raises(InputFormatError, match="numeric"):
        validate_anndata(broken_spatial, require_spatial=True)


def test_validate_anndata_requires_boolean_require_spatial() -> None:
    with pytest.raises(InputFormatError, match="require_spatial"):
        validate_anndata(sample_adata(), require_spatial="yes")  # type: ignore[arg-type]


def test_spectral_qc_reports_sparse_total_and_detected_features_in_obs_order() -> None:
    adata = sample_adata()
    original_x = adata.X.copy()
    report = spectral_qc(adata)

    assert report.index.tolist() == ["z", "a", "m"]
    assert report.loc["z", "total_intensity"] == 1.0
    assert report.loc["a", "detected_features"] == 2
    assert (adata.X != original_x).nnz == 0


def test_spectral_qc_does_not_count_explicit_sparse_zeros() -> None:
    matrix = csr_matrix(([0.0, 2.0], ([0, 1], [0, 1])), shape=(2, 2))
    adata = anndata.AnnData(matrix, var={"mz": [100.0, 200.0]})

    report = spectral_qc(adata)

    assert report["detected_features"].tolist() == [0, 1]


def test_spectral_qc_supports_scipy_sparse_arrays() -> None:
    adata = anndata.AnnData(csr_array([[0.0, 2.0], [3.0, 0.0]]), var={"mz": [1.0, 2.0]})

    report = spectral_qc(adata)

    assert report["total_intensity"].tolist() == [2.0, 3.0]
    assert report["detected_features"].tolist() == [1, 1]


@pytest.mark.parametrize("constructor", [np.asarray, csr_matrix])
def test_spectral_qc_promotes_large_integer_totals_to_float64(constructor) -> None:
    maximum = np.iinfo(np.int64).max
    matrix = constructor(np.array([[maximum, maximum], [1, 2]], dtype=np.int64))
    adata = anndata.AnnData(matrix, var={"mz": [1.0, 2.0]})

    report = spectral_qc(adata)

    assert report["total_intensity"].dtype == np.float64
    assert report["total_intensity"].tolist() == [float(maximum) * 2, 3.0]


def test_spectral_qc_validates_layer_and_does_not_mutate_it() -> None:
    adata = sample_adata()
    adata.layers["normalized"] = csr_matrix([[0.5, 0.0], [1.0, 1.5], [50.0, 50.0]])
    original = adata.layers["normalized"].copy()

    report = spectral_qc(adata, layer="normalized")
    assert report.loc["z", "total_intensity"] == 0.5
    assert (adata.layers["normalized"] != original).nnz == 0

    with pytest.raises(InputFormatError, match="missing"):
        spectral_qc(adata, layer="missing")


def test_registration_and_coverage_reports_are_tidy_and_truthful() -> None:
    registered = RegistrationResult(
        sample_adata(), None, {"orientation": "identity"}, {"registered_pixels": 3}
    )
    quantified = QuantificationResult(
        sample_adata(),
        None,
        None,
        None,
        {"included_cells": 3, "quantified_cells": 2, "rejected_mappings": 1},
    )

    registration = registration_report(registered)
    coverage = coverage_report(quantified)
    assert registration.columns.tolist() == ["metric", "value"]
    assert registration.set_index("metric").loc["registered_pixels", "value"] == 3
    assert registration.set_index("metric").loc["transform_orientation", "value"] == "identity"
    assert coverage.columns.tolist() == ["metric", "value"]
    assert coverage.set_index("metric").loc["coverage_fraction", "value"] == 2 / 3


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"included_cells": 3},
        {"included_cells": -1, "quantified_cells": 0},
        {"included_cells": 1, "quantified_cells": 2},
        {"included_cells": "three", "quantified_cells": 2},
    ],
)
def test_coverage_report_rejects_invalid_counts(report: dict[str, object]) -> None:
    result = QuantificationResult(sample_adata(), None, None, None, report)
    with pytest.raises(InputFormatError, match="included_cells|quantified_cells"):
        coverage_report(result)


def test_report_functions_validate_result_types_and_payloads() -> None:
    with pytest.raises(InputFormatError, match="RegistrationResult"):
        registration_report(object())  # type: ignore[arg-type]
    with pytest.raises(InputFormatError, match="QuantificationResult"):
        coverage_report(object())  # type: ignore[arg-type]

    invalid = RegistrationResult(sample_adata(), None, {"scale": np.nan}, {})
    with pytest.raises(InputFormatError, match="transform_scale"):
        registration_report(invalid)


def test_registration_report_accepts_finite_real_fraction_without_coercion() -> None:
    ratio = Fraction(1, 3)
    result = RegistrationResult(sample_adata(), None, {"scale": ratio}, {})

    report = registration_report(result)

    assert report.set_index("metric").loc["transform_scale", "value"] == ratio


@pytest.mark.parametrize("value", [1 + 2j, np.complex128(2 + 0j), Decimal("1.5")])
def test_registration_report_rejects_non_real_numeric_values(value) -> None:
    result = RegistrationResult(sample_adata(), None, {"scale": value}, {})
    with pytest.raises(InputFormatError, match="transform_scale.*real"):
        registration_report(result)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_registration_report_rejects_nonfinite_real_values(value: float) -> None:
    result = RegistrationResult(sample_adata(), None, {"scale": value}, {})
    with pytest.raises(InputFormatError, match="transform_scale.*finite"):
        registration_report(result)


def test_detect_outliers_returns_deterministic_boolean_series() -> None:
    first = detect_outliers(sample_adata(), contamination=1 / 3, random_seed=7)
    second = detect_outliers(sample_adata(), contamination=1 / 3, random_seed=7)

    assert first.dtype == bool
    assert first.name == "outlier"
    assert first.index.tolist() == ["z", "a", "m"]
    assert first.sum() == 1
    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize("contamination", [0, -0.1, 0.51, np.nan, "bad", True])
def test_detect_outliers_validates_contamination(contamination) -> None:
    with pytest.raises(InputFormatError, match="contamination"):
        detect_outliers(sample_adata(), contamination=contamination)


@pytest.mark.parametrize("random_seed", [-1, 2**32, 1.2, "7", True])
def test_detect_outliers_validates_random_seed(random_seed) -> None:
    with pytest.raises(InputFormatError, match="random_seed"):
        detect_outliers(sample_adata(), random_seed=random_seed)


def test_detect_outliers_requires_at_least_two_observations() -> None:
    adata = sample_adata()[:1].copy()
    with pytest.raises(InputFormatError, match="at least 2"):
        detect_outliers(adata)


def test_pcoa_default_braycurtis_handles_zero_spectra_deterministically() -> None:
    adata = anndata.AnnData(
        csr_matrix([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 1.0]]),
        obs=pd.DataFrame(index=["zero-a", "zero-b", "x", "y"]),
    )

    first = pcoa(adata, dimensions=2)
    second = pcoa(adata, dimensions=2)

    assert first.columns.tolist() == ["PCoA1", "PCoA2"]
    assert first.index.tolist() == adata.obs_names.tolist()
    assert np.isfinite(first.to_numpy()).all()
    pd.testing.assert_frame_equal(first, second)


def test_pcoa_uint8_results_are_invariant_to_observation_order() -> None:
    matrix = np.array([[0, 1], [2, 3], [4, 0]], dtype=np.uint8)
    forward = pcoa(anndata.AnnData(matrix), dimensions=2)
    reverse = pcoa(anndata.AnnData(matrix[::-1]), dimensions=2).iloc[::-1]

    assert np.allclose(
        squareform(pdist(forward.to_numpy())),
        squareform(pdist(reverse.to_numpy())),
    )


def test_pcoa_exposes_full_eigenspectrum_and_warns_for_material_negative_values() -> None:
    matrix = np.array([[1, 0], [1, 4], [2, 4], [2, 1]], dtype=float)

    with pytest.warns(RuntimeWarning, match="negative eigenvalues"):
        coordinates = pcoa(anndata.AnnData(matrix), dimensions=1)

    assert coordinates.columns.tolist() == ["PCoA1"]
    assert len(coordinates.attrs["eigenvalues"]) == 4
    assert min(coordinates.attrs["eigenvalues"].values()) < 0
    assert len(coordinates.attrs["proportion_explained"]) == 4


def test_pcoa_default_dimensions_support_two_observations() -> None:
    coordinates = pcoa(anndata.AnnData(np.array([[1.0, 0.0], [0.0, 1.0]])))

    assert coordinates.shape == (2, 2)
    assert coordinates.columns.tolist() == ["PCoA1", "PCoA2"]


def test_pcoa_uses_scikit_bio_06_dimension_keyword(monkeypatch) -> None:
    actual_pcoa = qc_module.skbio_pcoa
    called: dict[str, int] = {}

    def compatible_pcoa(distance_matrix, *, number_of_dimensions):
        called["number_of_dimensions"] = number_of_dimensions
        return actual_pcoa(
            distance_matrix,
            number_of_dimensions=number_of_dimensions,
        )

    monkeypatch.setattr(qc_module, "skbio_pcoa", compatible_pcoa)

    coordinates = pcoa(anndata.AnnData(np.array([[1.0, 0.0], [0.0, 1.0]])))

    assert called == {"number_of_dimensions": 2}
    assert coordinates.shape == (2, 2)


@pytest.mark.parametrize(
    "matrix",
    [np.zeros((3, 2)), np.ones((3, 2))],
)
def test_pcoa_returns_zero_coordinates_for_degenerate_distances(matrix: np.ndarray) -> None:
    coordinates = pcoa(anndata.AnnData(csr_array(matrix)), dimensions=2)

    assert coordinates.columns.tolist() == ["PCoA1", "PCoA2"]
    assert np.array_equal(coordinates.to_numpy(), np.zeros((3, 2)))


@pytest.mark.parametrize("dimensions", [0, -1, 1.5, True, 4])
def test_pcoa_validates_dimensions(dimensions) -> None:
    with pytest.raises(InputFormatError, match="dimensions"):
        pcoa(sample_adata(), dimensions=dimensions)


@pytest.mark.parametrize("metric", ["not-a-metric", "precomputed", "", 1])
def test_pcoa_wraps_invalid_metric_errors(metric) -> None:
    with pytest.raises(InputFormatError, match="metric"):
        pcoa(sample_adata(), dimensions=2, metric=metric)


def test_pcoa_wraps_nonfinite_metric_distances() -> None:
    with pytest.raises(InputFormatError, match="metric"):
        pcoa(
            anndata.AnnData(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])),
            metric="correlation",
        )


def test_pcoa_requires_nonnegative_values_for_braycurtis() -> None:
    adata = anndata.AnnData(np.array([[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]]))
    with pytest.raises(InputFormatError, match="non-negative"):
        pcoa(adata)


def test_pcoa_requires_at_least_two_observations() -> None:
    with pytest.raises(InputFormatError, match="at least 2"):
        pcoa(anndata.AnnData(np.ones((1, 2))), dimensions=1)


def test_silhouette_summary_returns_tidy_table_without_mutation() -> None:
    adata = anndata.AnnData(
        csr_matrix([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]),
        obs=pd.DataFrame({"cluster": ["b", "b", "a", "a"]}, index=["d", "c", "b", "a"]),
    )
    original_obs = adata.obs.copy(deep=True)

    summary = silhouette_summary(adata, labels="cluster")

    assert summary.columns.tolist() == ["cluster", "mean_silhouette", "cells"]
    assert summary["cluster"].tolist() == ["b", "a"]
    assert summary.set_index("cluster").loc["a", "cells"] == 2
    pd.testing.assert_frame_equal(adata.obs, original_obs)


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        ("missing", "labels"),
        ("one", "at least 2 clusters"),
        ("unique", "fewer clusters than observations"),
        ("null", "missing"),
    ],
)
def test_silhouette_summary_validates_labels_and_cluster_counts(labels: str, match: str) -> None:
    adata = sample_adata()
    adata.obs["one"] = ["a", "a", "a"]
    adata.obs["unique"] = ["a", "b", "c"]
    adata.obs["null"] = ["a", None, "b"]

    with pytest.raises(InputFormatError, match=match):
        silhouette_summary(adata, labels=labels)


@pytest.mark.parametrize("metric", ["not-a-metric", "", 1])
def test_silhouette_summary_wraps_invalid_metric_errors(metric) -> None:
    adata = sample_adata()
    adata.obs["cluster"] = ["a", "a", "b"]
    with pytest.raises(InputFormatError, match="metric"):
        silhouette_summary(adata, labels="cluster", metric=metric)


def test_silhouette_summary_rejects_precomputed_metric_for_square_feature_matrix() -> None:
    adata = anndata.AnnData(
        np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    )
    adata.obs["cluster"] = ["a", "a", "b"]

    with pytest.raises(InputFormatError, match="precomputed"):
        silhouette_summary(adata, labels="cluster", metric="precomputed")


def test_silhouette_summary_wraps_nonfinite_metric_distances() -> None:
    adata = anndata.AnnData(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
    adata.obs["cluster"] = ["a", "a", "b"]
    with pytest.raises(InputFormatError, match="metric"):
        silhouette_summary(adata, labels="cluster", metric="correlation")
