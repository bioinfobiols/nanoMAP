import numpy as np
import pytest
from scipy.sparse import csr_matrix

from joint._peaks import bin_profile_spectra, build_consensus_axis, project_spectra
from joint.errors import PeakAlignmentError


def test_build_consensus_axis_groups_peaks_within_da_tolerance():
    axis = build_consensus_axis(
        [np.array([100.000, 200.000]), np.array([100.006, 300.000])],
        tolerance=0.01,
        unit="da",
    )
    np.testing.assert_allclose(axis, [100.003, 200.0, 300.0])


def test_build_consensus_axis_returns_empty_float_axis_for_no_peaks():
    axis = build_consensus_axis([], tolerance=0.01, unit="da")

    assert axis.dtype.kind == "f"
    assert axis.size == 0


def test_build_consensus_axis_rejects_bad_tolerance_unit():
    with pytest.raises(PeakAlignmentError, match="Unsupported tolerance unit"):
        build_consensus_axis([np.array([100.0])], tolerance=0.01, unit="mz")


def test_build_consensus_axis_wraps_non_numeric_mz_values():
    with pytest.raises(PeakAlignmentError, match="numeric"):
        build_consensus_axis([np.array(["not-an-mz"])], tolerance=0.01, unit="da")


@pytest.mark.parametrize("bad_mz", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_peak_alignment_rejects_non_positive_or_non_finite_mz(bad_mz):
    with pytest.raises(PeakAlignmentError, match="m/z"):
        build_consensus_axis([np.array([bad_mz])], tolerance=0.01, unit="da")


@pytest.mark.parametrize("bad_intensity", [np.nan, np.inf, -np.inf])
def test_peak_alignment_rejects_non_finite_intensity(bad_intensity):
    with pytest.raises(PeakAlignmentError, match="intensity"):
        project_spectra(
            [(np.array([100.0]), np.array([bad_intensity]))],
            np.array([100.0]),
            tolerance=0.01,
            unit="da",
        )


@pytest.mark.parametrize("bad_axis", [np.array([0.0]), np.array([np.nan]), np.array([np.inf])])
def test_project_spectra_rejects_non_positive_or_non_finite_axis(bad_axis):
    with pytest.raises(PeakAlignmentError, match="axis"):
        project_spectra(
            [(np.array([100.0]), np.array([1.0]))],
            bad_axis,
            tolerance=0.01,
            unit="da",
        )


@pytest.mark.parametrize("bad_tolerance", [np.nan, np.inf, -np.inf, 0.0])
def test_peak_alignment_rejects_non_finite_or_non_positive_tolerance(bad_tolerance):
    with pytest.raises(PeakAlignmentError, match="tolerance"):
        build_consensus_axis([np.array([100.0])], tolerance=bad_tolerance, unit="da")


def test_project_spectra_sums_duplicate_assignments():
    axis = np.array([100.0, 200.0])
    matrix = project_spectra(
        [
            (np.array([99.999, 100.001, 200.0]), np.array([2.0, 3.0, 7.0])),
            (np.array([100.0]), np.array([11.0])),
        ],
        axis,
        tolerance=0.01,
        unit="da",
    )
    assert isinstance(matrix, csr_matrix)
    np.testing.assert_allclose(matrix.toarray(), [[5.0, 7.0], [11.0, 0.0]])


def test_consensus_axis_keeps_every_grouped_peak_within_projection_tolerance():
    mzs = np.array([100.0, 101.0, 101.5, 101.75])
    axis = build_consensus_axis([mzs], tolerance=1.0, unit="da")

    matrix = project_spectra(
        [(mzs, np.ones(mzs.size))], axis, tolerance=1.0, unit="da"
    )

    assert matrix.sum() == mzs.size


def test_project_spectra_rejects_mismatched_peak_and_intensity_lengths():
    with pytest.raises(PeakAlignmentError, match="same length"):
        project_spectra(
            [(np.array([100.0, 200.0]), np.array([1.0]))],
            np.array([100.0, 200.0]),
            tolerance=0.01,
            unit="da",
        )


def test_project_spectra_wraps_non_numeric_intensity_values():
    with pytest.raises(PeakAlignmentError, match="numeric"):
        project_spectra(
            [(np.array([100.0]), np.array(["not-an-intensity"]))],
            np.array([100.0]),
            tolerance=0.01,
            unit="da",
        )


def test_project_spectra_rejects_non_strictly_increasing_axis():
    with pytest.raises(PeakAlignmentError, match="strictly increasing"):
        project_spectra(
            [(np.array([100.0]), np.array([1.0]))],
            np.array([200.0, 100.0]),
            tolerance=0.01,
            unit="da",
        )


def test_bin_profile_spectra_uses_fixed_width_centers():
    axis, matrix = bin_profile_spectra(
        [
            (np.array([100.1, 100.7, 101.2]), np.array([1.0, 2.0, 3.0])),
            (np.array([100.2, 101.8]), np.array([4.0, 5.0])),
        ],
        bin_size=1.0,
    )
    assert isinstance(matrix, csr_matrix)
    np.testing.assert_allclose(axis, [100.5, 101.5])
    np.testing.assert_allclose(matrix.toarray(), [[3.0, 3.0], [4.0, 5.0]])


def test_bin_profile_spectra_returns_empty_sparse_matrix_for_no_spectra():
    axis, matrix = bin_profile_spectra([], bin_size=1.0)

    assert axis.size == 0
    assert matrix.shape == (0, 0)


@pytest.mark.parametrize("bad_bin_size", [np.nan, np.inf, -np.inf, 0.0])
def test_bin_profile_spectra_rejects_non_finite_or_non_positive_bin_size(bad_bin_size):
    with pytest.raises(PeakAlignmentError, match="bin_size"):
        bin_profile_spectra([], bin_size=bad_bin_size)


def test_bin_profile_spectra_keeps_empty_spectrum_as_empty_row():
    axis, matrix = bin_profile_spectra(
        [(np.array([]), np.array([])), (np.array([100.2]), np.array([3.0]))],
        bin_size=1.0,
    )

    np.testing.assert_allclose(axis, [100.5])
    np.testing.assert_allclose(matrix.toarray(), [[0.0], [3.0]])
