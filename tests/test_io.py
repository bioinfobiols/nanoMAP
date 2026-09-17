import json
import pickle
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from joint.errors import InputFormatError, PeakAlignmentError
from joint.io import (
    atomic_write_h5ad,
    read_h5ad,
    read_imzml,
    read_segmentation,
    write_results,
)
from joint.models import QuantificationResult, RegistrationResult, SegmentationResult


class FakeParser:
    coordinates = ((1, 2, 1), (2, 2, 1))

    def __init__(self, path, *, ibd_file):
        self.path = path
        self.m = ibd_file

    def getspectrum(self, index):
        spectra = [
            (np.array([100.0, 200.0]), np.array([1.0, 2.0])),
            (np.array([100.004, 300.0]), np.array([3.0, 4.0])),
        ]
        return spectra[index]


class UnsortedSharedAxisParser:
    coordinates = ((1, 2, 1), (2, 2, 1))

    def __init__(self, path, *, ibd_file):
        self.path = path
        self.m = ibd_file

    def getspectrum(self, index):
        return np.array([200.0, 100.0]), np.array([1.0, 2.0])


class NonNumericSpectrumParser:
    coordinates = ((1, 2, 1),)

    def __init__(self, path, *, ibd_file):
        self.path = path
        self.m = ibd_file

    def getspectrum(self, index):
        return np.array(["not-an-mz"]), np.array([1.0])


class ClosableParser:
    coordinates = ((1, 2, 1),)
    last = None

    def __init__(self, path, *, ibd_file):
        self.path = path
        self.m = ibd_file
        type(self).last = self

    def getspectrum(self, index):
        return np.array([100.0]), np.array([1.0])


class FailingSpectrumParser(ClosableParser):
    def getspectrum(self, index):
        raise RuntimeError("spectrum read failed")


class FailingCloseHandle:
    closed = False

    def close(self):
        raise RuntimeError("binary close failed")


class FailingCloseParser(ClosableParser):
    pass


class FailingSpectrumAndCloseParser(FailingCloseParser):
    def getspectrum(self, index):
        raise RuntimeError("spectrum read failed")


class FailingConstructorParser:
    last_handle = None

    def __init__(self, path, *, ibd_file):
        type(self).last_handle = ibd_file
        raise RuntimeError("metadata parse failed")


class UncoercibleLabels:
    def __array__(self, dtype=None, copy=None):
        raise RuntimeError("cannot make array")


def test_read_imzml_builds_anndata(monkeypatch, tmp_path: Path):
    imzml = tmp_path / "sample.imzML"
    ibd = tmp_path / "sample.ibd"
    imzml.write_text("xml")
    ibd.write_bytes(b"ibd")
    monkeypatch.setattr("joint.io.ImzMLParser", FakeParser)

    adata = read_imzml(imzml, dataset_id="sample", tolerance=0.01, unit="da")

    assert adata.shape == (2, 3)
    np.testing.assert_array_equal(adata.obsm["spatial"], [[1, 2], [2, 2]])
    assert adata.obs["dataset"].tolist() == ["sample", "sample"]
    assert adata.var["mz"].dtype.kind == "f"


def test_read_imzml_requires_matching_ibd_file(tmp_path: Path):
    imzml = tmp_path / "sample.imzML"
    imzml.write_text("xml")

    with pytest.raises(InputFormatError, match="Matching ibd"):
        read_imzml(imzml)


def test_read_imzml_rejects_unsorted_shared_axis(monkeypatch, tmp_path: Path):
    imzml = tmp_path / "unsorted.imzML"
    imzml.write_text("xml")
    (tmp_path / "unsorted.ibd").write_bytes(b"ibd")
    monkeypatch.setattr("joint.io.ImzMLParser", UnsortedSharedAxisParser)

    with pytest.raises(InputFormatError, match="strictly increasing") as error:
        read_imzml(imzml)

    assert isinstance(error.value.__cause__, PeakAlignmentError)


def test_read_imzml_wraps_non_numeric_spectrum_values(monkeypatch, tmp_path: Path):
    imzml = tmp_path / "non-numeric.imzML"
    imzml.write_text("xml")
    (tmp_path / "non-numeric.ibd").write_bytes(b"ibd")
    monkeypatch.setattr("joint.io.ImzMLParser", NonNumericSpectrumParser)

    with pytest.raises(InputFormatError, match="Could not parse imzML") as error:
        read_imzml(imzml)

    assert isinstance(error.value.__cause__, PeakAlignmentError)


@pytest.mark.parametrize("spectrum", [
    (np.array([np.nan]), np.array([1.0])),
    (np.array([100.0]), np.array([np.inf])),
])
def test_read_imzml_wraps_non_finite_spectrum_values(monkeypatch, tmp_path: Path, spectrum):
    class NonFiniteParser:
        coordinates = ((1, 2, 1),)

        def __init__(self, path, *, ibd_file):
            self.path = path

        def getspectrum(self, index):
            return spectrum

    imzml = tmp_path / "non-finite.imzML"
    imzml.write_text("xml")
    (tmp_path / "non-finite.ibd").write_bytes(b"ibd")
    monkeypatch.setattr("joint.io.ImzMLParser", NonFiniteParser)

    with pytest.raises(InputFormatError, match="Could not parse imzML") as error:
        read_imzml(imzml)

    assert isinstance(error.value.__cause__, PeakAlignmentError)


def test_read_imzml_rejects_empty_pixel_set(monkeypatch, tmp_path: Path):
    class EmptyParser:
        coordinates = ()

        def __init__(self, path, *, ibd_file):
            self.path = path
            self.m = ibd_file

    imzml = tmp_path / "empty.imzML"
    imzml.write_text("xml")
    (tmp_path / "empty.ibd").write_bytes(b"ibd")
    monkeypatch.setattr("joint.io.ImzMLParser", EmptyParser)

    with pytest.raises(InputFormatError, match="no pixel spectra"):
        read_imzml(imzml)


def test_read_imzml_closes_parser_binary_handle(monkeypatch, tmp_path: Path):
    imzml = tmp_path / "sample.imzML"
    imzml.write_text("xml")
    (tmp_path / "sample.ibd").write_bytes(b"ibd")
    monkeypatch.setattr("joint.io.ImzMLParser", ClosableParser)

    read_imzml(imzml)

    assert ClosableParser.last is not None
    assert ClosableParser.last.m.closed


def test_read_imzml_closes_parser_binary_handle_after_spectrum_error(monkeypatch, tmp_path: Path):
    imzml = tmp_path / "sample.imzML"
    imzml.write_text("xml")
    (tmp_path / "sample.ibd").write_bytes(b"ibd")
    monkeypatch.setattr("joint.io.ImzMLParser", FailingSpectrumParser)

    with pytest.raises(InputFormatError, match="spectrum read failed"):
        read_imzml(imzml)

    assert FailingSpectrumParser.last is not None
    assert FailingSpectrumParser.last.m.closed


def test_read_imzml_closes_binary_handle_after_parser_constructor_error(monkeypatch, tmp_path):
    imzml = tmp_path / "sample.imzML"
    imzml.write_text("xml")
    (tmp_path / "sample.ibd").write_bytes(b"ibd")
    monkeypatch.setattr("joint.io.ImzMLParser", FailingConstructorParser)

    with pytest.raises(InputFormatError, match="metadata parse failed"):
        read_imzml(imzml)

    handle = FailingConstructorParser.last_handle
    assert handle is not None
    try:
        assert handle.closed
    finally:
        handle.close()


def test_read_imzml_reports_close_error_after_success(monkeypatch, tmp_path: Path):
    imzml = tmp_path / "sample.imzML"
    imzml.write_text("xml")
    (tmp_path / "sample.ibd").write_bytes(b"ibd")
    handle = FailingCloseHandle()
    monkeypatch.setattr("joint.io.Path.open", lambda self, *args, **kwargs: handle)
    monkeypatch.setattr("joint.io.ImzMLParser", FailingCloseParser)

    with pytest.raises(InputFormatError, match="Could not close imzML binary handle"):
        read_imzml(imzml)


def test_read_imzml_close_error_does_not_mask_spectrum_error(monkeypatch, tmp_path: Path):
    imzml = tmp_path / "sample.imzML"
    imzml.write_text("xml")
    (tmp_path / "sample.ibd").write_bytes(b"ibd")
    handle = FailingCloseHandle()
    monkeypatch.setattr("joint.io.Path.open", lambda self, *args, **kwargs: handle)
    monkeypatch.setattr("joint.io.ImzMLParser", FailingSpectrumAndCloseParser)

    with pytest.raises(InputFormatError, match="spectrum read failed") as error:
        read_imzml(imzml)

    assert error.value.__notes__ == ["Could not close imzML binary handle: binary close failed"]


def test_read_segmentation_supports_npy_and_trusted_pickle(tmp_path: Path):
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    np.save(tmp_path / "labels.npy", labels)
    with (tmp_path / "labels.pkl").open("wb") as handle:
        pickle.dump(labels, handle)

    np.testing.assert_array_equal(read_segmentation(tmp_path / "labels.npy"), labels)
    np.testing.assert_array_equal(
        read_segmentation(tmp_path / "labels.pkl", trusted_pickle=True), labels
    )


def test_read_segmentation_rejects_pickle_without_explicit_trust(tmp_path: Path):
    path = tmp_path / "labels.pkl"
    with path.open("wb") as handle:
        pickle.dump(np.array([[1]], dtype=np.int32), handle)

    with pytest.raises(InputFormatError, match="trusted_pickle=True"):
        read_segmentation(path)


def test_read_segmentation_supports_legacy_singleton_pickle_wrappers(tmp_path: Path):
    labels = np.array([[0, 1], [2, 2]], dtype=np.int64)
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump(labels[None, :, :, None], handle)

    result = read_segmentation(path, trusted_pickle=True)

    np.testing.assert_array_equal(result, labels)


@pytest.mark.parametrize("shape", [(2, 2, 2, 1), (1, 2, 2, 2), (1, 1, 2, 2)])
def test_read_segmentation_rejects_ambiguous_high_dimensional_pickle(
    tmp_path: Path, shape: tuple[int, ...]
):
    path = tmp_path / "ambiguous.pkl"
    with path.open("wb") as handle:
        pickle.dump(np.zeros(shape, dtype=np.int32), handle)

    with pytest.raises(InputFormatError, match="two-dimensional integer"):
        read_segmentation(path, trusted_pickle=True)


def test_read_segmentation_rejects_non_integer_or_non_2d_labels(tmp_path: Path):
    path = tmp_path / "labels.npy"
    np.save(path, np.array([1.0, 2.0]))

    with pytest.raises(InputFormatError, match="two-dimensional integer"):
        read_segmentation(path)


def test_read_segmentation_wraps_label_coercion_errors(tmp_path: Path):
    path = tmp_path / "labels.pkl"
    with path.open("wb") as handle:
        pickle.dump(UncoercibleLabels(), handle)

    with pytest.raises(InputFormatError, match="Could not validate segmentation"):
        read_segmentation(path, trusted_pickle=True)


def test_atomic_h5ad_round_trip(tmp_path: Path):
    adata = anndata.AnnData(csr_matrix([[1.0, 2.0]]))
    destination = atomic_write_h5ad(adata, tmp_path / "result.h5ad")

    assert destination.is_file()
    assert not list(tmp_path.glob("*.tmp.h5ad"))
    assert read_h5ad(destination).shape == (1, 2)


def test_read_h5ad_wraps_anndata_reader_errors(tmp_path: Path):
    path = tmp_path / "invalid.h5ad"
    path.write_text("not an h5ad")

    with pytest.raises(InputFormatError, match="Could not read h5ad"):
        read_h5ad(path)


def test_write_results_serializes_segmentation_bundle(tmp_path: Path):
    result = SegmentationResult(
        labels=np.array([[0, 1]], dtype=np.int32),
        regions=pd.DataFrame({"label": [1], "area": [1]}),
        diagnostics={"object_count": 1},
    )

    artifacts = write_results(result, tmp_path / "bundle")

    assert set(artifacts) == {"labels", "regions", "report"}
    assert all(path.is_file() for path in artifacts.values())


def test_write_results_filters_non_json_segmentation_diagnostics_without_mutation(tmp_path: Path):
    labels = np.array([[0, 1]], dtype=np.int32)
    regions = pd.DataFrame({"label": [1], "area": [1]})
    diagnostics = {"object_count": 1, "non_serializable": object()}
    result = SegmentationResult(labels=labels, regions=regions, diagnostics=diagnostics)

    artifacts = write_results(result, tmp_path / "bundle")

    assert json.loads(artifacts["report"].read_text()) == {"object_count": 1}
    np.testing.assert_array_equal(result.labels, labels)
    pd.testing.assert_frame_equal(result.regions, regions)
    assert result.diagnostics is diagnostics
    assert "non_serializable" in result.diagnostics


def test_write_results_serializes_anndata_without_mutating_input(tmp_path: Path):
    adata = anndata.AnnData(csr_matrix([[1.0, 2.0]]))

    artifacts = write_results(adata, tmp_path / "adata")

    assert set(artifacts) == {"adata"}
    assert read_h5ad(artifacts["adata"]).shape == adata.shape
    np.testing.assert_array_equal(adata.X.toarray(), [[1.0, 2.0]])


def test_write_results_serializes_registration_bundle(tmp_path: Path):
    result = RegistrationResult(
        adata=anndata.AnnData(csr_matrix([[3.0]])),
        mapping=pd.DataFrame({"pixel": ["p1"], "cell": ["c1"]}),
        transform={"scale": 2.0},
        report={"matched": 1},
    )

    artifacts = write_results(result, tmp_path / "registration")

    assert set(artifacts) == {"adata", "mapping", "report"}
    assert read_h5ad(artifacts["adata"]).shape == (1, 1)
    pd.testing.assert_frame_equal(pd.read_csv(artifacts["mapping"]), result.mapping)
    assert json.loads(artifacts["report"].read_text()) == {"matched": 1, "scale": 2.0}


def test_write_results_serializes_quantification_bundle(tmp_path: Path):
    table = pd.DataFrame({"cell": ["c1"], "score": [0.8]})
    result = QuantificationResult(
        adata=anndata.AnnData(csr_matrix([[4.0]])),
        overlaps=table,
        accepted=table,
        rejected=pd.DataFrame(columns=table.columns),
        report={"accepted_count": 1},
    )

    artifacts = write_results(result, tmp_path / "quantification")

    assert set(artifacts) == {"adata", "overlaps", "accepted", "rejected", "report"}
    assert read_h5ad(artifacts["adata"]).shape == (1, 1)
    pd.testing.assert_frame_equal(pd.read_csv(artifacts["overlaps"]), result.overlaps)
    assert json.loads(artifacts["report"].read_text()) == {"accepted_count": 1}


def test_write_results_rejects_unsupported_results(tmp_path: Path):
    with pytest.raises(TypeError, match="Unsupported JOINT result type: object"):
        write_results(object(), tmp_path / "unsupported")


def test_write_results_cleans_temporary_file_after_artifact_write_error(monkeypatch, tmp_path: Path):
    result = SegmentationResult(
        labels=np.array([[0, 1]], dtype=np.int32),
        regions=pd.DataFrame({"label": [1]}),
    )

    def fail_to_csv(self, path_or_buf, *args, **kwargs):
        Path(path_or_buf).write_text("partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(InputFormatError, match="Could not write CSV file.*regions.csv") as error:
        write_results(result, tmp_path / "failed")

    assert isinstance(error.value.__cause__, OSError)
    assert not list((tmp_path / "failed").glob(".*.tmp"))
