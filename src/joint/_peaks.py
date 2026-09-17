"""Peak-axis construction and sparse MSI spectrum projection."""

from collections.abc import Iterable, Sequence
from itertools import pairwise

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from joint.errors import PeakAlignmentError


def _distance_limit(mz: float, tolerance: float, unit: str) -> float:
    if unit == "da":
        return tolerance
    if unit == "ppm":
        return mz * tolerance / 1_000_000.0
    raise PeakAlignmentError(f"Unsupported tolerance unit: {unit}")


def _validate_tolerance(tolerance: float, unit: str) -> None:
    try:
        numeric_tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise PeakAlignmentError("tolerance must be finite and positive") from exc
    if not np.isfinite(numeric_tolerance) or numeric_tolerance <= 0:
        raise PeakAlignmentError("tolerance must be finite and positive")
    _distance_limit(1.0, numeric_tolerance, unit)


def _validated_spectrum(
    mzs: np.ndarray, intensities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mz_array = _numeric_array(mzs, "m/z")
    intensity_array = _numeric_array(intensities, "intensity")
    if mz_array.ndim != 1 or intensity_array.ndim != 1:
        raise PeakAlignmentError("m/z and intensity arrays must be one-dimensional")
    if len(mz_array) != len(intensity_array):
        raise PeakAlignmentError("m/z and intensity arrays must have the same length")
    _validate_positive_finite_values(mz_array, "m/z")
    _validate_finite_values(intensity_array, "intensity")
    return mz_array, intensity_array


def _numeric_array(values: np.ndarray, name: str) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PeakAlignmentError(f"{name} values must be numeric") from exc


def _validate_finite_values(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise PeakAlignmentError(f"{name} values must be finite")


def _validate_positive_finite_values(values: np.ndarray, name: str) -> None:
    _validate_finite_values(values, name)
    if np.any(values <= 0):
        raise PeakAlignmentError(f"{name} values must be positive")


def build_consensus_axis(
    mz_arrays: Sequence[np.ndarray], *, tolerance: float, unit: str
) -> np.ndarray:
    """Group sorted m/z values into tolerance-bounded consensus peaks."""
    axis, _ = _build_consensus_axis_with_assignments(
        mz_arrays, tolerance=tolerance, unit=unit
    )
    return axis


def _build_consensus_axis_with_assignments(
    mz_arrays: Sequence[np.ndarray], *, tolerance: float, unit: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build a consensus axis and map every input peak to its consensus member."""
    _validate_tolerance(tolerance, unit)
    arrays = [_numeric_array(array, "m/z") for array in mz_arrays]
    if not arrays:
        return np.array([], dtype=float), []
    if any(array.ndim != 1 for array in arrays):
        raise PeakAlignmentError("m/z arrays must be one-dimensional")
    for array in arrays:
        _validate_positive_finite_values(array, "m/z")
    lengths = [len(array) for array in arrays]
    values = np.concatenate(arrays)
    if values.size == 0:
        return values, [np.array([], dtype=int) for _ in arrays]

    order = np.argsort(values)
    assignments = np.empty(len(values), dtype=int)
    first = int(order[0])
    groups: list[list[float]] = [[float(values[first])]]
    assignments[first] = 0
    for member in order[1:]:
        member = int(member)
        value = float(values[member])
        group = groups[-1]
        center = float((sum(group) + value) / (len(group) + 1))
        if (
            abs(group[0] - center) <= _distance_limit(group[0], tolerance, unit)
            and abs(value - center) <= _distance_limit(value, tolerance, unit)
        ):
            group.append(value)
        else:
            groups.append([value])
        assignments[member] = len(groups) - 1

    offsets = np.cumsum([0, *lengths])
    per_array = [
        assignments[start:end] for start, end in pairwise(offsets)
    ]
    axis = np.array([np.mean(group) for group in groups], dtype=float)
    return axis, per_array


def project_spectra(
    spectra: Iterable[tuple[np.ndarray, np.ndarray]],
    axis: np.ndarray,
    *,
    tolerance: float,
    unit: str,
) -> csr_matrix:
    """Project spectra onto an axis, summing intensities sharing an axis peak."""
    _validate_tolerance(tolerance, unit)
    axis_array = _numeric_array(axis, "axis")
    if axis_array.ndim != 1:
        raise PeakAlignmentError("axis must be one-dimensional")
    _validate_positive_finite_values(axis_array, "axis")
    if not np.all(np.diff(axis_array) > 0):
        raise PeakAlignmentError("axis must be strictly increasing")

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    spectrum_count = 0
    for row, (mzs, intensities) in enumerate(spectra):
        spectrum_count += 1
        mz_array, intensity_array = _validated_spectrum(mzs, intensities)
        for mz, intensity in zip(mz_array, intensity_array, strict=True):
            insertion = int(np.searchsorted(axis_array, mz))
            candidates = [
                index
                for index in (insertion - 1, insertion)
                if 0 <= index < len(axis_array)
            ]
            if not candidates:
                continue
            column = min(candidates, key=lambda index: abs(axis_array[index] - mz))
            if abs(axis_array[column] - mz) <= _distance_limit(float(mz), tolerance, unit):
                rows.append(row)
                columns.append(column)
                data.append(float(intensity))
    return coo_matrix((data, (rows, columns)), shape=(spectrum_count, len(axis_array))).tocsr()


def bin_profile_spectra(
    spectra: Sequence[tuple[np.ndarray, np.ndarray]], *, bin_size: float
) -> tuple[np.ndarray, csr_matrix]:
    """Bin profile spectra into fixed-width m/z intervals."""
    try:
        numeric_bin_size = float(bin_size)
    except (TypeError, ValueError) as exc:
        raise PeakAlignmentError("bin_size must be finite and positive") from exc
    if not np.isfinite(numeric_bin_size) or numeric_bin_size <= 0:
        raise PeakAlignmentError("bin_size must be finite and positive")

    checked_spectra = [_validated_spectrum(mzs, intensities) for mzs, intensities in spectra]
    nonempty_mzs = [mzs for mzs, _ in checked_spectra if mzs.size]
    if not nonempty_mzs:
        return np.array([], dtype=float), csr_matrix((len(checked_spectra), 0))

    minimum = min(float(np.min(mzs)) for mzs in nonempty_mzs)
    maximum = max(float(np.max(mzs)) for mzs in nonempty_mzs)
    start = np.floor(minimum / numeric_bin_size) * numeric_bin_size
    bin_count = int(np.floor((maximum - start) / numeric_bin_size)) + 1
    axis = start + (np.arange(bin_count) + 0.5) * numeric_bin_size

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for row, (mzs, intensities) in enumerate(checked_spectra):
        indices = np.floor((mzs - start) / numeric_bin_size).astype(int)
        rows.extend([row] * len(indices))
        columns.extend(indices.tolist())
        data.extend(intensities.tolist())
    matrix = coo_matrix((data, (rows, columns)), shape=(len(checked_spectra), bin_count))
    return axis, matrix.tocsr()
