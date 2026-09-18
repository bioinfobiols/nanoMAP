from pathlib import Path

import anndata
import numpy as np
from scipy.sparse import csr_matrix

from joint.config import PreprocessingConfig
from joint.errors import InputFormatError
from joint.pp import align_peaks, filter_features, normalize, preprocess_msi, remove_matrix_peaks


def make_adata():
    return anndata.AnnData(
        csr_matrix([[1.0, 0.0, 3.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
        var={"mz": [100.0, 200.0, 300.0]},
    )


def test_filter_features_uses_nonzero_occurrence_without_mutating_input():
    adata = make_adata()
    filtered = filter_features(adata, min_occurrence=0.5)

    assert adata.shape == (3, 3)
    assert filtered.var["mz"].tolist() == [100.0]


def test_rms_normalization_preserves_raw_x_and_zero_rows():
    adata = anndata.AnnData(csr_matrix([[3.0, 4.0], [0.0, 0.0]]))
    normalized = normalize(adata, method="rms", layer="normalized")

    np.testing.assert_allclose(normalized.X.toarray(), [[3.0, 4.0], [0.0, 0.0]])
    np.testing.assert_allclose(normalized.layers["normalized"].toarray()[1], [0.0, 0.0])


def test_preprocess_msi_applies_filter_and_normalization(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.imzML"
    source.write_text("xml")
    monkeypatch.setattr("joint.pp.read_imzml", lambda *args, **kwargs: make_adata())
    config = PreprocessingConfig(normalization="tic", min_occurrence=0.5)

    result = preprocess_msi(source, config=config)

    assert result.shape == (3, 1)
    assert "normalized" in result.layers


def test_preprocess_msi_h5ad_requires_numeric_mz(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.h5ad"
    monkeypatch.setattr(
        "joint.pp.read_h5ad", lambda *args, **kwargs: anndata.AnnData(csr_matrix([[1.0]]))
    )

    with np.testing.assert_raises_regex(InputFormatError, 'must contain numeric column "mz"'):
        preprocess_msi(source, config=PreprocessingConfig())


def test_preprocess_msi_h5ad_canonicalizes_numeric_mz_to_floats(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.h5ad"
    adata = anndata.AnnData(csr_matrix([[1.0]]), var={"mz": ["100.0"]})
    monkeypatch.setattr("joint.pp.read_h5ad", lambda *args, **kwargs: adata)

    result = preprocess_msi(source, config=PreprocessingConfig(min_occurrence=0.0))

    assert result.var["mz"].tolist() == [100.0]
    assert result.var["mz"].dtype.kind == "f"


def test_align_peaks_merges_adjacent_features_and_sums_intensity():
    adata = anndata.AnnData(
        csr_matrix([[1.0, 2.0, 4.0]]),
        var={"mz": [100.000, 100.006, 200.0]},
    )
    aligned = align_peaks(adata, tolerance=0.01, unit="da")
    np.testing.assert_allclose(aligned.X.toarray(), [[3.0, 4.0]])
    np.testing.assert_allclose(aligned.var["mz"], [100.003, 200.0])


def test_align_peaks_keeps_membership_from_consensus_grouping():
    adata = anndata.AnnData(
        csr_matrix([[1.0, 2.0, 4.0]]),
        var={"mz": [100.0, 102.0, 102.1]},
    )

    aligned = align_peaks(adata, tolerance=1.0, unit="da")

    np.testing.assert_allclose(aligned.var["mz"], [101.0, 102.1])
    np.testing.assert_allclose(aligned.X.toarray(), [[3.0, 4.0]])


def test_align_peaks_deep_copies_nested_uns_metadata():
    adata = make_adata()
    adata.uns["joint"] = {"labels": ["source"]}

    aligned = align_peaks(adata, tolerance=0.01, unit="da")
    aligned.uns["joint"]["labels"].append("result")

    assert adata.uns["joint"]["labels"] == ["source"]


def test_filter_features_excludes_zero_occurrence_at_zero_threshold():
    filtered = filter_features(make_adata(), min_occurrence=0.0)

    assert filtered.var["mz"].tolist() == [100.0, 200.0, 300.0]


def test_filter_features_does_not_count_explicit_sparse_zeros_as_occurrences():
    matrix = csr_matrix(([1.0, 0.0], ([0, 0], [0, 1])), shape=(1, 2))
    adata = anndata.AnnData(matrix, var={"mz": [100.0, 200.0]})

    assert filter_features(adata, min_occurrence=0.0).var["mz"].tolist() == [100.0]


def test_filter_features_handles_anndata_with_no_observations():
    adata = anndata.AnnData(csr_matrix((0, 2)), var={"mz": [100.0, 200.0]})

    assert filter_features(adata, min_occurrence=0.0).shape == (0, 0)


def test_normalize_handles_empty_observation_and_feature_dimensions():
    no_observations = anndata.AnnData(csr_matrix((0, 2)))
    no_features = anndata.AnnData(csr_matrix((2, 0)))

    assert normalize(no_observations).layers["normalized"].shape == (0, 2)
    assert normalize(no_features).layers["normalized"].shape == (2, 0)


def test_rms_normalization_casts_integer_intensities_before_squaring():
    adata = anndata.AnnData(csr_matrix(np.array([[50_000, 50_000]], dtype=np.int32)))

    np.testing.assert_allclose(normalize(adata).layers["normalized"].toarray(), [[1.0, 1.0]])


def test_normalize_rejects_empty_output_layer_name():
    with np.testing.assert_raises_regex(ValueError, "layer"):
        normalize(make_adata(), layer="")


def test_align_peaks_reports_non_numeric_mz_values():
    adata = anndata.AnnData(csr_matrix([[1.0]]), var={"mz": ["not-a-number"]})

    with np.testing.assert_raises_regex(InputFormatError, "numeric"):
        align_peaks(adata, tolerance=0.01, unit="da")


def test_reference_matrix_removal_uses_ppm_tolerance():
    adata = make_adata()
    cleaned, report = remove_matrix_peaks(
        adata,
        method="reference",
        reference_mz=np.array([100.0005]),
        tolerance=10,
        unit="ppm",
    )

    assert cleaned.var["mz"].tolist() == [200.0, 300.0]
    assert report["reason"].tolist() == ["reference_match"]


def test_blank_matrix_removal_uses_blank_occurrence():
    adata = make_adata()
    adata.obs["dataset"] = ["blank", "blank", "sample"]
    cleaned, report = remove_matrix_peaks(
        adata, method="blank", blank_datasets=["blank"], matrix_frequency=0.8
    )

    assert cleaned.var["mz"].tolist() == [200.0, 300.0]
    assert report.iloc[0]["mz"] == 100.0


def test_denovo_matrix_removal_uses_grid_edge_pixels():
    adata = anndata.AnnData(
        csr_matrix(
            [
                [5.0, 0.0],
                [5.0, 0.0],
                [5.0, 0.0],
                [5.0, 2.0],
                [0.0, 2.0],
                [5.0, 2.0],
                [5.0, 0.0],
                [5.0, 0.0],
                [5.0, 0.0],
            ]
        ),
        var={"mz": [100.0, 200.0]},
        obsm={"spatial": np.array([(x, y) for y in range(3) for x in range(3)])},
    )
    cleaned, report = remove_matrix_peaks(adata, method="denovo", matrix_frequency=0.8)

    assert cleaned.var["mz"].tolist() == [200.0]
    assert report.iloc[0]["reason"] == "edge_frequency"


def test_preprocess_msi_applies_configured_reference_removal(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.imzML"
    source.write_text("xml")
    reference = tmp_path / "matrix.csv"
    reference.write_text("mz\n100.0005\n")
    monkeypatch.setattr("joint.pp.read_imzml", lambda *args, **kwargs: make_adata())
    config = PreprocessingConfig(
        normalization="none",
        min_occurrence=0.0,
        matrix_removal={
            "enabled": True,
            "method": "reference",
            "reference_file": reference,
            "tolerance": 10,
            "unit": "ppm",
        },
    )

    result = preprocess_msi(source, config=config)

    assert result.var["mz"].tolist() == [200.0, 300.0]
    assert result.uns["joint"]["matrix_removed_mz"] == [100.0]


def test_cardinal_backend_keeps_configured_matrix_removal(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.imzML"
    source.write_text("xml")
    reference = tmp_path / "matrix.csv"
    reference.write_text("mz\n100.0005\n")
    monkeypatch.setattr(
        "joint.backends.cardinal.run_cardinal", lambda *args, **kwargs: make_adata()
    )
    config = PreprocessingConfig(
        backend="cardinal",
        normalization="none",
        min_occurrence=0.0,
        matrix_removal={
            "enabled": True,
            "method": "reference",
            "reference_file": reference,
            "tolerance": 10,
            "unit": "ppm",
        },
    )

    result = preprocess_msi(source, config=config, cardinal_output_dir=tmp_path / "cardinal")

    assert result.var["mz"].tolist() == [200.0, 300.0]
    assert result.uns["joint"]["matrix_removed_mz"] == [100.0]


def test_cardinal_backend_preserves_dataset_id_for_blank_matrix_removal(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.imzML"
    source.write_text("xml")
    arguments = {}

    def fake_run_cardinal(*args, dataset_id=None, **kwargs):
        arguments["dataset_id"] = dataset_id
        result = make_adata()
        result.obs["dataset"] = dataset_id
        return result

    monkeypatch.setattr("joint.backends.cardinal.run_cardinal", fake_run_cardinal)
    config = PreprocessingConfig(
        backend="cardinal",
        normalization="none",
        min_occurrence=0.0,
        matrix_removal={
            "enabled": True,
            "method": "blank",
            "blank_datasets": ["sample"],
            "matrix_frequency": 0.8,
        },
    )

    result = preprocess_msi(
        source,
        config=config,
        dataset_id="sample",
        cardinal_output_dir=tmp_path / "cardinal",
    )

    assert arguments["dataset_id"] == "sample"
    assert result.obs["dataset"].tolist() == ["sample", "sample", "sample"]
    assert result.var["mz"].tolist() == [200.0, 300.0]


def test_matrix_removal_validates_unit_and_tolerance_for_every_method():
    blank = make_adata()
    blank.obs["dataset"] = ["blank", "blank", "sample"]
    denovo = make_adata()
    denovo.obsm["spatial"] = np.array([(0, 0), (1, 0), (0, 1)])

    with np.testing.assert_raises_regex(ValueError, "unit"):
        remove_matrix_peaks(
            blank,
            method="blank",
            blank_datasets=["blank"],
            unit="invalid",  # type: ignore[arg-type]
        )
    with np.testing.assert_raises_regex(ValueError, "tolerance"):
        remove_matrix_peaks(denovo, method="denovo", tolerance=0)


def test_preprocess_msi_wraps_empty_matrix_reference_file(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.imzML"
    source.write_text("xml")
    reference = tmp_path / "empty-matrix.csv"
    reference.write_text("")
    monkeypatch.setattr("joint.pp.read_imzml", lambda *args, **kwargs: make_adata())
    config = PreprocessingConfig(
        min_occurrence=0.0,
        matrix_removal={"enabled": True, "method": "reference", "reference_file": reference},
    )

    with np.testing.assert_raises_regex(InputFormatError, "Could not read matrix reference file"):
        preprocess_msi(source, config=config)


def test_preprocess_msi_wraps_invalid_utf8_matrix_reference_file(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.imzML"
    source.write_text("xml")
    reference = tmp_path / "invalid-utf8-matrix.csv"
    reference.write_bytes(b"mz\n\xff\n")
    monkeypatch.setattr("joint.pp.read_imzml", lambda *args, **kwargs: make_adata())
    config = PreprocessingConfig(
        min_occurrence=0.0,
        matrix_removal={"enabled": True, "method": "reference", "reference_file": reference},
    )

    with np.testing.assert_raises_regex(InputFormatError, "Could not read matrix reference file") as error:
        preprocess_msi(source, config=config)

    assert isinstance(error.exception.__cause__, UnicodeDecodeError)
