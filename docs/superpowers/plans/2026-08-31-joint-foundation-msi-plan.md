# JOINT Foundation and MSI Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable JOINT package with stable shared contracts, validated YAML configuration, imzML/h5ad/segmentation I/O, native MSI preprocessing, matrix-peak filtering, and an optional Cardinal adapter.

**Architecture:** Use a `src/` package layout. AnnData is the public data container, Pydantic models validate configuration, and sparse matrices are retained throughout preprocessing. Optional R/Cardinal behavior is isolated under `joint.backends` and is never imported by the core package.

**Tech Stack:** Python 3.11+, hatchling, NumPy, SciPy, pandas, AnnData, pyimzML, PyYAML, Pydantic 2, pytest, Ruff.

## Global Constraints

- The primary interface is typed Python functions; the CLI is added in Plan 4.
- Core import must not import COSG, cNMF, R bridges, or trajectory-only dependencies.
- `adata.obsm["spatial"]` uses `(x, y)` coordinates and `adata.var["mz"]` uses numeric m/z values.
- Public functions return new objects by default and never call `exit()`.
- Paths loaded from YAML resolve relative to the YAML file.
- Native Python preprocessing is the default; Cardinal is optional.
- Raw matrices remain available when normalization is written to a layer.
- No REST service, graphical interface, or cell-segmentation model is added.

---

## Planned File Map

```text
pyproject.toml                         packaging, dependencies, test/lint settings
README.md                              minimal package identity and install command
src/joint/__init__.py                  version and curated top-level exports
src/joint/errors.py                    JOINT exception hierarchy
src/joint/models.py                    shared result dataclasses
src/joint/config.py                    typed YAML schema and path resolution
src/joint/_peaks.py                    internal peak-axis and projection helpers
src/joint/io.py                        imzML, h5ad, segmentation, and atomic output I/O
src/joint/pp.py                        filtering, normalization, matrix removal, preprocessing facade
src/joint/backends/__init__.py         backend namespace
src/joint/backends/cardinal.py         R/Cardinal subprocess adapter
src/joint/backends/cardinal_pipeline.R versioned Cardinal entry script
tests/test_package.py                   package and shared-contract tests
tests/test_config.py                    configuration tests
tests/test_peaks.py                     peak alignment/projection tests
tests/test_io.py                        data I/O tests
tests/test_pp.py                        preprocessing tests
tests/test_cardinal_backend.py          optional backend tests
```

### Task 1: Package Skeleton and Shared Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/joint/__init__.py`
- Create: `src/joint/errors.py`
- Create: `src/joint/models.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Consumes: none.
- Produces: `JointError` hierarchy and `SegmentationResult`, `RegistrationResult`, `QuantificationResult`, `PipelineResult` dataclasses used by all later plans.

- [ ] **Step 1: Write the failing package-contract tests**

```python
# tests/test_package.py
import joint
from joint.errors import (
    ConfigurationError,
    InputFormatError,
    JointError,
    OptionalDependencyError,
    PeakAlignmentError,
    QuantificationError,
    RegistrationError,
    SegmentationError,
)
from joint.models import SegmentationResult


def test_package_exports_version():
    assert joint.__version__ == "0.1.0"


def test_all_specific_errors_are_joint_errors():
    errors = [
        ConfigurationError,
        InputFormatError,
        OptionalDependencyError,
        PeakAlignmentError,
        QuantificationError,
        RegistrationError,
        SegmentationError,
    ]
    assert all(issubclass(error, JointError) for error in errors)


def test_segmentation_result_retains_diagnostics():
    result = SegmentationResult(labels=None, regions=None, diagnostics={"objects": 3})
    assert result.diagnostics["objects"] == 3
```

- [ ] **Step 2: Run the tests and verify the package is absent**

Run: `python -m pytest tests/test_package.py -v`  
Expected: collection fails with `ModuleNotFoundError: No module named 'joint'`.

- [ ] **Step 3: Add packaging metadata**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "joint-msi"
version = "0.1.0"
description = "Joint spatial MSI, laser-mark, and single-cell metabolomics analysis"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "anndata>=0.10",
  "numpy>=1.26",
  "pandas>=2.1",
  "pydantic>=2.7",
  "pyimzml>=1.5",
  "pyyaml>=6.0",
  "scipy>=1.11",
]

[project.optional-dependencies]
test = ["pytest>=8.0", "ruff>=0.6"]
cardinal = []

[tool.hatch.build.targets.wheel]
packages = ["src/joint"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py311"
```

````markdown
<!-- README.md -->
# JOINT

JOINT connects MSI spectra, laser-ablation marks, microscopy segmentations, and single-cell metabolomics analysis.

Development install:

```bash
python -m pip install -e '.[test]'
```
````

- [ ] **Step 4: Implement the error hierarchy and dataclasses**

```python
# src/joint/errors.py
class JointError(Exception):
    """Base class for all public JOINT errors."""


class ConfigurationError(JointError):
    pass


class InputFormatError(JointError):
    pass


class PeakAlignmentError(JointError):
    pass


class SegmentationError(JointError):
    pass


class RegistrationError(JointError):
    pass


class QuantificationError(JointError):
    pass


class OptionalDependencyError(JointError):
    pass
```

```python
# src/joint/models.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    regions: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrationResult:
    adata: anndata.AnnData
    mapping: pd.DataFrame
    transform: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantificationResult:
    adata: anndata.AnnData
    overlaps: pd.DataFrame
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    artifacts: dict[str, Path]
    manifest: Path
    reports: dict[str, Any] = field(default_factory=dict)
```

```python
# src/joint/__init__.py
from joint.models import PipelineResult, QuantificationResult, RegistrationResult, SegmentationResult

__version__ = "0.1.0"

__all__ = [
    "PipelineResult",
    "QuantificationResult",
    "RegistrationResult",
    "SegmentationResult",
    "__version__",
]
```

- [ ] **Step 5: Install and run the package tests**

Run: `python -m pip install -e '.[test]'`  
Expected: editable installation succeeds.

Run: `python -m pytest tests/test_package.py -v`  
Expected: 3 tests pass.

- [ ] **Step 6: Commit the package contracts**

```bash
git add pyproject.toml README.md src/joint tests/test_package.py
git commit -m "build: scaffold JOINT package contracts"
```

### Task 2: Typed YAML Configuration and Relative Path Resolution

**Files:**
- Create: `src/joint/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: `ConfigurationError` from Task 1.
- Produces: `JointConfig`, `load_config(path: str | Path) -> JointConfig`, and `write_resolved_config(config, path) -> Path`.

- [ ] **Step 1: Write failing configuration tests**

```python
# tests/test_config.py
from pathlib import Path

import pytest

from joint.config import load_config, write_resolved_config
from joint.errors import ConfigurationError


def test_load_config_resolves_paths_relative_to_yaml(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "sample.yaml"
    config_path.write_text(
        """
project:
  name: sample
  output_dir: ../results/sample
input:
  imzml: ../data/sample.imzML
laser_segmentation:
  method: simulated
  radius: 18
quantification:
  method: specificity_filtered
""".strip()
    )

    config = load_config(config_path)

    assert config.project.output_dir == (tmp_path / "results/sample").resolve()
    assert config.input.imzml == (tmp_path / "data/sample.imzML").resolve()
    assert config.laser_segmentation.radius == 18


def test_invalid_quantification_method_has_domain_error(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
project: {name: bad, output_dir: results}
input: {h5ad: input.h5ad}
quantification: {method: unsupported}
""".strip()
    )

    with pytest.raises(ConfigurationError, match="unsupported"):
        load_config(config_path)


def test_write_resolved_config_emits_absolute_paths(tmp_path: Path):
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        "project: {name: sample, output_dir: results}\ninput: {h5ad: input.h5ad}\n"
    )
    config = load_config(config_path)

    written = write_resolved_config(config, tmp_path / "resolved.yaml")

    assert written.is_file()
    assert str((tmp_path / "input.h5ad").resolve()) in written.read_text()
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run: `python -m pytest tests/test_config.py -v`  
Expected: collection fails with `ModuleNotFoundError: No module named 'joint.config'`.

- [ ] **Step 3: Implement the complete configuration schema**

```python
# src/joint/config.py
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from joint.errors import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str
    output_dir: Path
    random_seed: int = 0


class InputConfig(StrictModel):
    imzml: Path | None = None
    h5ad: Path | None = None
    laser_image: Path | None = None
    cell_segmentation: Path | None = None


class MatrixRemovalConfig(StrictModel):
    enabled: bool = False
    method: Literal["reference", "blank", "denovo"] = "denovo"
    reference_file: Path | None = None
    blank_datasets: list[str] = Field(default_factory=list)
    matrix_frequency: float = Field(default=0.8, ge=0.0, le=1.0)
    tolerance: float = Field(default=10.0, gt=0.0)
    unit: Literal["ppm", "da"] = "ppm"


class PreprocessingConfig(StrictModel):
    backend: Literal["python", "cardinal"] = "python"
    normalization: Literal["rms", "tic", "none"] = "rms"
    normalized_layer: str = "normalized"
    min_occurrence: float = Field(default=0.05, ge=0.0, le=1.0)
    peak_tolerance: float = Field(default=10.0, gt=0.0)
    tolerance_unit: Literal["ppm", "da"] = "ppm"
    profile_bin_size: float | None = Field(default=None, gt=0.0)
    cardinal_snr: float = Field(default=3.0, gt=0.0)
    matrix_removal: MatrixRemovalConfig = Field(default_factory=MatrixRemovalConfig)


class LaserSegmentationConfig(StrictModel):
    method: Literal["real", "simulated"] = "real"
    radius: int = Field(default=18, gt=0)
    expected_rows: int | None = Field(default=None, gt=0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    merge_rules: Path | None = None


class RegistrationConfig(StrictModel):
    orientation: Literal[
        "auto", "identity", "swap_xy", "flip_x", "flip_y", "swap_flip_x", "swap_flip_y"
    ] = "auto"


class QuantificationConfig(StrictModel):
    method: Literal["specificity_filtered", "legacy_proportional"] = "specificity_filtered"
    boundary_margin: float = Field(default=20.0, ge=0.0)
    unique_min_overlap: float = Field(default=0.10, ge=0.0, le=1.0)
    dominant_min_overlap: float = Field(default=0.60, ge=0.0, le=1.0)
    secondary_max_overlap: float = Field(default=0.15, ge=0.0, le=1.0)


class CosgConfig(StrictModel):
    enabled: bool = False
    groupby: str = "cluster"
    n_genes: int = Field(default=20, gt=0)


class CnmfConfig(StrictModel):
    enabled: bool = False
    components: list[int] = Field(default_factory=lambda: list(range(5, 20)))
    selected_k: int = 14
    seed: int = 14
    num_highvar_genes: int = Field(default=300, gt=0)


class AnalysisConfig(StrictModel):
    cosg: CosgConfig = Field(default_factory=CosgConfig)
    cnmf: CnmfConfig = Field(default_factory=CnmfConfig)


class JointConfig(StrictModel):
    project: ProjectConfig
    input: InputConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    laser_segmentation: LaserSegmentationConfig = Field(default_factory=LaserSegmentationConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    quantification: QuantificationConfig = Field(default_factory=QuantificationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


_PATH_KEYS = {
    "output_dir",
    "imzml",
    "h5ad",
    "laser_image",
    "cell_segmentation",
    "merge_rules",
    "reference_file",
}


def _resolve_paths(value: Any, base: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {name: _resolve_paths(item, base, name) for name, item in value.items()}
    if key in _PATH_KEYS and value is not None:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()
    return value


def load_config(path: str | Path) -> JointConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text()) or {}
        resolved = _resolve_paths(payload, config_path.parent)
        config = JointConfig.model_validate(resolved)
    except (yaml.YAMLError, ValidationError, TypeError) as exc:
        raise ConfigurationError(str(exc)) from exc
    if (config.input.imzml is None) == (config.input.h5ad is None):
        raise ConfigurationError("Exactly one of input.imzml or input.h5ad must be provided")
    return config


def write_resolved_config(config: JointConfig, path: str | Path) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    return destination
```

- [ ] **Step 4: Run configuration tests**

Run: `python -m pytest tests/test_config.py -v`  
Expected: 3 tests pass.

- [ ] **Step 5: Commit configuration support**

```bash
git add src/joint/config.py tests/test_config.py
git commit -m "feat: add typed JOINT configuration"
```

### Task 3: Peak-Axis Helpers and Data I/O

**Files:**
- Create: `src/joint/_peaks.py`
- Create: `src/joint/io.py`
- Create: `tests/test_peaks.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Consumes: `InputFormatError`, `PeakAlignmentError`.
- Produces: `build_consensus_axis`, `project_spectra`, `read_imzml`, `read_h5ad`, `read_segmentation`, and `atomic_write_h5ad`.

- [ ] **Step 1: Write failing peak-axis tests**

```python
# tests/test_peaks.py
import numpy as np

from joint._peaks import build_consensus_axis, bin_profile_spectra, project_spectra


def test_build_consensus_axis_groups_peaks_within_da_tolerance():
    axis = build_consensus_axis(
        [np.array([100.000, 200.000]), np.array([100.006, 300.000])],
        tolerance=0.01,
        unit="da",
    )
    np.testing.assert_allclose(axis, [100.003, 200.0, 300.0])


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
    np.testing.assert_allclose(matrix.toarray(), [[5.0, 7.0], [11.0, 0.0]])


def test_bin_profile_spectra_uses_fixed_width_centers():
    axis, matrix = bin_profile_spectra(
        [
            (np.array([100.1, 100.7, 101.2]), np.array([1.0, 2.0, 3.0])),
            (np.array([100.2, 101.8]), np.array([4.0, 5.0])),
        ],
        bin_size=1.0,
    )
    np.testing.assert_allclose(axis, [100.5, 101.5])
    np.testing.assert_allclose(matrix.toarray(), [[3.0, 3.0], [4.0, 5.0]])
```

- [ ] **Step 2: Write failing I/O tests with a fake imzML parser**

```python
# tests/test_io.py
import pickle
from pathlib import Path

import anndata
import numpy as np
from scipy.sparse import csr_matrix

from joint.io import atomic_write_h5ad, read_h5ad, read_imzml, read_segmentation


class FakeParser:
    coordinates = [(1, 2, 1), (2, 2, 1)]

    def __init__(self, path):
        self.path = path

    def getspectrum(self, index):
        spectra = [
            (np.array([100.0, 200.0]), np.array([1.0, 2.0])),
            (np.array([100.004, 300.0]), np.array([3.0, 4.0])),
        ]
        return spectra[index]


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


def test_read_segmentation_supports_npy_and_trusted_pickle(tmp_path: Path):
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    np.save(tmp_path / "labels.npy", labels)
    with (tmp_path / "labels.pkl").open("wb") as handle:
        pickle.dump(labels, handle)

    np.testing.assert_array_equal(read_segmentation(tmp_path / "labels.npy"), labels)
    np.testing.assert_array_equal(read_segmentation(tmp_path / "labels.pkl"), labels)


def test_atomic_h5ad_round_trip(tmp_path: Path):
    adata = anndata.AnnData(csr_matrix([[1.0, 2.0]]))
    destination = atomic_write_h5ad(adata, tmp_path / "result.h5ad")

    assert destination.is_file()
    assert not list(tmp_path.glob("*.tmp.h5ad"))
    assert read_h5ad(destination).shape == (1, 2)
```

- [ ] **Step 3: Run the focused tests and verify missing modules**

Run: `python -m pytest tests/test_peaks.py tests/test_io.py -v`  
Expected: collection fails because `joint._peaks` and `joint.io` do not exist.

- [ ] **Step 4: Implement peak grouping and sparse projection**

```python
# src/joint/_peaks.py
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from joint.errors import PeakAlignmentError


def _distance_limit(mz: float, tolerance: float, unit: str) -> float:
    if unit == "da":
        return tolerance
    if unit == "ppm":
        return mz * tolerance / 1_000_000.0
    raise PeakAlignmentError(f"Unsupported tolerance unit: {unit}")


def build_consensus_axis(
    mz_arrays: Sequence[np.ndarray], *, tolerance: float, unit: str
) -> np.ndarray:
    if tolerance <= 0:
        raise PeakAlignmentError("tolerance must be positive")
    values = np.sort(np.concatenate([np.asarray(array, dtype=float) for array in mz_arrays]))
    if values.size == 0:
        return values
    groups: list[list[float]] = [[float(values[0])]]
    for value in values[1:]:
        center = float(np.mean(groups[-1]))
        if abs(float(value) - center) <= _distance_limit(center, tolerance, unit):
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return np.array([np.mean(group) for group in groups], dtype=float)


def project_spectra(
    spectra: Iterable[tuple[np.ndarray, np.ndarray]],
    axis: np.ndarray,
    *,
    tolerance: float,
    unit: str,
) -> csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    spectrum_count = 0
    for row, (mzs, intensities) in enumerate(spectra):
        spectrum_count += 1
        for mz, intensity in zip(mzs, intensities, strict=True):
            insertion = int(np.searchsorted(axis, mz))
            candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(axis)]
            if not candidates:
                continue
            column = min(candidates, key=lambda index: abs(axis[index] - mz))
            if abs(axis[column] - mz) <= _distance_limit(float(mz), tolerance, unit):
                rows.append(row)
                columns.append(column)
                data.append(float(intensity))
    matrix = coo_matrix((data, (rows, columns)), shape=(spectrum_count, len(axis)))
    return matrix.tocsr()


def bin_profile_spectra(
    spectra: Sequence[tuple[np.ndarray, np.ndarray]], *, bin_size: float
) -> tuple[np.ndarray, csr_matrix]:
    if bin_size <= 0:
        raise PeakAlignmentError("bin_size must be positive")
    minimum = min(float(np.min(mzs)) for mzs, _ in spectra)
    maximum = max(float(np.max(mzs)) for mzs, _ in spectra)
    start = np.floor(minimum / bin_size) * bin_size
    bin_count = int(np.floor((maximum - start) / bin_size)) + 1
    axis = start + (np.arange(bin_count) + 0.5) * bin_size
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for row, (mzs, intensities) in enumerate(spectra):
        indices = np.floor((np.asarray(mzs) - start) / bin_size).astype(int)
        rows.extend([row] * len(indices))
        columns.extend(indices.tolist())
        data.extend(np.asarray(intensities, dtype=float).tolist())
    matrix = coo_matrix((data, (rows, columns)), shape=(len(spectra), bin_count)).tocsr()
    return axis, matrix
```

- [ ] **Step 5: Implement imzML, h5ad, segmentation, and atomic output I/O**

```python
# src/joint/io.py
import os
import pickle
import tempfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser

from joint._peaks import build_consensus_axis, bin_profile_spectra, project_spectra
from joint.errors import InputFormatError


def read_imzml(
    path: str | Path,
    *,
    dataset_id: str | None = None,
    tolerance: float = 10.0,
    unit: str = "ppm",
    profile_bin_size: float | None = None,
) -> anndata.AnnData:
    imzml = Path(path).expanduser().resolve()
    if not imzml.is_file():
        raise InputFormatError(f"imzML file does not exist: {imzml}")
    ibd = imzml.with_suffix(".ibd")
    if not ibd.is_file():
        raise InputFormatError(f"Matching ibd file does not exist: {ibd}")
    try:
        parser = ImzMLParser(str(imzml))
        spectra = [parser.getspectrum(index) for index in range(len(parser.coordinates))]
    except Exception as exc:
        raise InputFormatError(f"Could not parse imzML file {imzml}: {exc}") from exc
    mz_arrays = [np.asarray(mzs, dtype=float) for mzs, _ in spectra]
    shared_axis = bool(mz_arrays) and all(np.array_equal(mz_arrays[0], item) for item in mz_arrays[1:])
    if profile_bin_size is not None:
        axis, matrix = bin_profile_spectra(spectra, bin_size=profile_bin_size)
    else:
        axis = mz_arrays[0].copy() if shared_axis else build_consensus_axis(
            mz_arrays, tolerance=tolerance, unit=unit
        )
        matrix = project_spectra(spectra, axis, tolerance=tolerance, unit=unit)
    coords = np.asarray([(x, y) for x, y, _ in parser.coordinates], dtype=float)
    sample = dataset_id or imzml.stem
    obs = pd.DataFrame(index=[f"p{index + 1}" for index in range(matrix.shape[0])])
    obs["dataset"] = sample
    var = pd.DataFrame(index=[f"m{index + 1}" for index in range(matrix.shape[1])])
    var["mz"] = axis
    adata = anndata.AnnData(matrix, obs=obs, var=var, obsm={"spatial": coords})
    adata.uns["joint"] = {"source": str(imzml), "shared_mz_axis": shared_axis}
    return adata


def read_h5ad(path: str | Path) -> anndata.AnnData:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputFormatError(f"h5ad file does not exist: {source}")
    return anndata.read_h5ad(source)


def read_segmentation(path: str | Path) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputFormatError(f"Segmentation file does not exist: {source}")
    if source.suffix == ".npy":
        labels = np.load(source, allow_pickle=False)
    elif source.suffix in {".pkl", ".pickle"}:
        with source.open("rb") as handle:
            labels = pickle.load(handle)
    else:
        raise InputFormatError(f"Unsupported segmentation format: {source.suffix}")
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype.kind not in "iu":
        raise InputFormatError("Segmentation must be a two-dimensional integer label array")
    return labels


def atomic_write_h5ad(adata: anndata.AnnData, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.h5ad", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        adata.write_h5ad(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
```

- [ ] **Step 6: Run I/O and peak tests**

Run: `python -m pytest tests/test_peaks.py tests/test_io.py -v`  
Expected: 6 tests pass.

- [ ] **Step 7: Commit peak and I/O support**

```bash
git add src/joint/_peaks.py src/joint/io.py tests/test_peaks.py tests/test_io.py
git commit -m "feat: add MSI and segmentation IO"
```

### Task 4: Feature Filtering, Normalization, and Preprocessing Facade

**Files:**
- Create: `src/joint/pp.py`
- Create: `tests/test_pp.py`

**Interfaces:**
- Consumes: `read_imzml`, `read_h5ad`, `PreprocessingConfig`.
- Produces: `filter_features`, `normalize`, and `preprocess_msi`.

- [ ] **Step 1: Write failing preprocessing tests**

```python
# tests/test_pp.py
from pathlib import Path

import anndata
import numpy as np
from scipy.sparse import csr_matrix

from joint.config import PreprocessingConfig
from joint.pp import align_peaks, filter_features, normalize, preprocess_msi


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


def test_align_peaks_merges_adjacent_features_and_sums_intensity():
    adata = anndata.AnnData(
        csr_matrix([[1.0, 2.0, 4.0]]),
        var={"mz": [100.000, 100.006, 200.0]},
    )
    aligned = align_peaks(adata, tolerance=0.01, unit="da")
    np.testing.assert_allclose(aligned.X.toarray(), [[3.0, 4.0]])
    np.testing.assert_allclose(aligned.var["mz"], [100.003, 200.0])
```

- [ ] **Step 2: Run the tests and verify `joint.pp` is missing**

Run: `python -m pytest tests/test_pp.py -v`  
Expected: collection fails with `ModuleNotFoundError: No module named 'joint.pp'`.

- [ ] **Step 3: Implement filtering and normalization**

```python
# src/joint/pp.py
from pathlib import Path
from typing import Literal

import anndata
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix

from joint.config import PreprocessingConfig
from joint._peaks import build_consensus_axis
from joint.errors import InputFormatError
from joint.io import read_h5ad, read_imzml


def align_peaks(
    adata: anndata.AnnData, *, tolerance: float, unit: Literal["ppm", "da"] = "ppm"
) -> anndata.AnnData:
    if "mz" not in adata.var:
        raise InputFormatError('adata.var must contain numeric column "mz"')
    mz = adata.var["mz"].astype(float).to_numpy()
    axis = build_consensus_axis([mz], tolerance=tolerance, unit=unit)
    insertion = np.searchsorted(axis, mz)
    assignments = []
    for value, index in zip(mz, insertion, strict=True):
        candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(axis)]
        assignments.append(min(candidates, key=lambda candidate: abs(axis[candidate] - value)))
    projector = coo_matrix(
        (np.ones(len(mz)), (np.arange(len(mz)), assignments)), shape=(len(mz), len(axis))
    ).tocsr()
    matrix = adata.X @ projector
    result = anndata.AnnData(
        sparse.csr_matrix(matrix),
        obs=adata.obs.copy(),
        var={"mz": axis},
        obsm={key: value.copy() for key, value in adata.obsm.items()},
    )
    result.var_names = [f"m{index + 1}" for index in range(result.n_vars)]
    result.uns = adata.uns.copy()
    return result


def _row_factors(matrix, method: str) -> np.ndarray:
    squared = matrix.multiply(matrix) if sparse.issparse(matrix) else np.square(matrix)
    if method == "rms":
        factors = np.sqrt(np.asarray(squared.mean(axis=1)).ravel())
    elif method == "tic":
        factors = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    factors[factors == 0] = 1.0
    return factors


def filter_features(adata: anndata.AnnData, *, min_occurrence: float = 0.05) -> anndata.AnnData:
    if not 0 <= min_occurrence <= 1:
        raise ValueError("min_occurrence must be between 0 and 1")
    matrix = adata.X
    counts = matrix.getnnz(axis=0) if sparse.issparse(matrix) else np.count_nonzero(matrix, axis=0)
    keep = np.asarray(counts).ravel() / adata.n_obs > min_occurrence
    result = adata[:, keep].copy()
    result.uns.setdefault("joint", {})["min_occurrence"] = min_occurrence
    return result


def normalize(
    adata: anndata.AnnData,
    *,
    method: Literal["rms", "tic", "none"] = "rms",
    layer: str = "normalized",
) -> anndata.AnnData:
    result = adata.copy()
    if method == "none":
        result.layers[layer] = result.X.copy()
        return result
    factors = _row_factors(result.X, method)
    inverse = sparse.diags(1.0 / factors)
    normalized = inverse @ result.X if sparse.issparse(result.X) else result.X / factors[:, None]
    result.layers[layer] = sparse.csr_matrix(normalized)
    result.uns.setdefault("joint", {})["normalization"] = method
    return result


def preprocess_msi(
    path: str | Path,
    *,
    config: PreprocessingConfig,
    dataset_id: str | None = None,
) -> anndata.AnnData:
    source = Path(path)
    if source.suffix.lower() == ".h5ad":
        adata = read_h5ad(source)
    elif source.suffix.lower() == ".imzml":
        adata = read_imzml(
            source,
            dataset_id=dataset_id,
            tolerance=config.peak_tolerance,
            unit=config.tolerance_unit,
            profile_bin_size=config.profile_bin_size,
        )
    else:
        raise InputFormatError(f"Unsupported MSI input: {source.suffix}")
    filtered = filter_features(adata, min_occurrence=config.min_occurrence)
    matrix_config = config.matrix_removal
    if matrix_config.enabled:
        filtered, matrix_report = remove_matrix_peaks(
            filtered,
            method=matrix_config.method,
            reference_mz=_load_reference_mz(matrix_config.reference_file),
            blank_datasets=matrix_config.blank_datasets,
            matrix_frequency=matrix_config.matrix_frequency,
            tolerance=matrix_config.tolerance,
            unit=matrix_config.unit,
        )
        filtered.uns.setdefault("joint", {})["matrix_removed_mz"] = matrix_report["mz"].tolist()
        filtered.uns["joint"]["matrix_removal_reasons"] = matrix_report["reason"].tolist()
    return normalize(filtered, method=config.normalization, layer=config.normalized_layer)
```

- [ ] **Step 4: Run preprocessing tests**

Run: `python -m pytest tests/test_pp.py -v`  
Expected: 4 tests pass.

- [ ] **Step 5: Commit native preprocessing**

```bash
git add src/joint/pp.py tests/test_pp.py
git commit -m "feat: add native MSI preprocessing"
```

### Task 5: Matrix-Peak Removal Strategies

**Files:**
- Modify: `src/joint/pp.py`
- Modify: `tests/test_pp.py`

**Interfaces:**
- Consumes: AnnData with numeric `var["mz"]`, optional `obs["dataset"]`, and optional `obsm["spatial"]`.
- Produces: `remove_matrix_peaks(adata, method, ...) -> tuple[AnnData, DataFrame]`.

- [ ] **Step 1: Add failing tests for reference, blank, and edge-based de novo removal**

```python
# append to tests/test_pp.py
from joint.pp import remove_matrix_peaks


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
                [5.0, 0.0], [5.0, 0.0], [5.0, 0.0],
                [5.0, 2.0], [0.0, 2.0], [5.0, 2.0],
                [5.0, 0.0], [5.0, 0.0], [5.0, 0.0],
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
```

- [ ] **Step 2: Run the focused tests and verify the API is absent**

Run: `python -m pytest tests/test_pp.py -k matrix -v`  
Expected: import fails because `remove_matrix_peaks` is not defined.

- [ ] **Step 3: Implement deterministic matrix-peak removal**

```python
# append to src/joint/pp.py
import pandas as pd


def _ppm_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(query[:, None] - reference[None, :]) / query[:, None] * 1_000_000.0


def _edge_mask(spatial: np.ndarray) -> np.ndarray:
    x = spatial[:, 0]
    y = spatial[:, 1]
    return (x == x.min()) | (x == x.max()) | (y == y.min()) | (y == y.max())


def remove_matrix_peaks(
    adata: anndata.AnnData,
    *,
    method: Literal["reference", "blank", "denovo"],
    reference_mz: np.ndarray | None = None,
    blank_datasets: list[str] | None = None,
    matrix_frequency: float = 0.8,
    tolerance: float = 10.0,
    unit: Literal["ppm", "da"] = "ppm",
) -> tuple[anndata.AnnData, pd.DataFrame]:
    if "mz" not in adata.var:
        raise InputFormatError('adata.var must contain numeric column "mz"')
    mz = adata.var["mz"].astype(float).to_numpy()
    if method == "reference":
        if reference_mz is None or len(reference_mz) == 0:
            raise ValueError("reference_mz is required for reference removal")
        reference = np.asarray(reference_mz, dtype=float)
        distances = _ppm_distance(mz, reference) if unit == "ppm" else np.abs(
            mz[:, None] - reference[None, :]
        )
        remove = distances.min(axis=1) <= tolerance
        reason = "reference_match"
    elif method == "blank":
        if not blank_datasets or "dataset" not in adata.obs:
            raise ValueError("blank_datasets and adata.obs['dataset'] are required")
        subset = adata[adata.obs["dataset"].isin(blank_datasets)]
        counts = subset.X.getnnz(axis=0) if sparse.issparse(subset.X) else np.count_nonzero(
            subset.X, axis=0
        )
        remove = np.asarray(counts).ravel() / subset.n_obs > matrix_frequency
        reason = "blank_frequency"
    else:
        if "spatial" not in adata.obsm:
            raise ValueError("obsm['spatial'] is required for denovo removal")
        subset = adata[_edge_mask(np.asarray(adata.obsm["spatial"]))]
        counts = subset.X.getnnz(axis=0) if sparse.issparse(subset.X) else np.count_nonzero(
            subset.X, axis=0
        )
        remove = np.asarray(counts).ravel() / subset.n_obs > matrix_frequency
        reason = "edge_frequency"
    report = pd.DataFrame({"feature_id": adata.var_names[remove], "mz": mz[remove], "reason": reason})
    return adata[:, ~remove].copy(), report.reset_index(drop=True)


def _load_reference_mz(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    table = pd.read_csv(path)
    if "mz" not in table:
        raise InputFormatError(f"Matrix reference file must contain an mz column: {path}")
    return table["mz"].astype(float).to_numpy()
```

Replace the final two lines of `preprocess_msi` with:

```python
filtered = filter_features(adata, min_occurrence=config.min_occurrence)
matrix_config = config.matrix_removal
if matrix_config.enabled:
    filtered, matrix_report = remove_matrix_peaks(
        filtered,
        method=matrix_config.method,
        reference_mz=_load_reference_mz(matrix_config.reference_file),
        blank_datasets=matrix_config.blank_datasets,
        matrix_frequency=matrix_config.matrix_frequency,
        tolerance=matrix_config.tolerance,
        unit=matrix_config.unit,
    )
    filtered.uns.setdefault("joint", {})["matrix_removed_mz"] = matrix_report["mz"].tolist()
    filtered.uns["joint"]["matrix_removal_reasons"] = matrix_report["reason"].tolist()
return normalize(filtered, method=config.normalization, layer=config.normalized_layer)
```

- [ ] **Step 4: Run all preprocessing tests**

Run: `python -m pytest tests/test_pp.py -v`  
Expected: 8 tests pass.

- [ ] **Step 5: Commit matrix removal**

```bash
git add src/joint/pp.py tests/test_pp.py
git commit -m "feat: add matrix peak filtering"
```

### Task 6: Optional Cardinal Backend

**Files:**
- Create: `src/joint/backends/__init__.py`
- Create: `src/joint/backends/cardinal.py`
- Create: `src/joint/backends/cardinal_pipeline.R`
- Create: `tests/test_cardinal_backend.py`
- Modify: `src/joint/pp.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: input imzML path and output directory.
- Produces: `run_cardinal(input_imzml, output_dir, *, rscript="Rscript") -> AnnData` and `preprocess_msi(..., backend="cardinal")` dispatch.

- [ ] **Step 1: Write failing backend tests**

```python
# tests/test_cardinal_backend.py
from pathlib import Path

import pytest

from joint.backends.cardinal import run_cardinal
from joint.errors import OptionalDependencyError


def test_missing_rscript_has_precise_optional_dependency_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: None)
    with pytest.raises(OptionalDependencyError, match="Rscript"):
        run_cardinal(tmp_path / "sample.imzML", tmp_path / "out")


def test_cardinal_runs_versioned_script_and_reads_h5ad(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "sample.imzML"
    input_path.write_text("xml")
    output_dir = tmp_path / "out"
    monkeypatch.setattr("joint.backends.cardinal.shutil.which", lambda command: "/usr/bin/Rscript")

    def fake_run(command, check, capture_output, text):
        assert command[-3:] == ["10.0", "ppm", "3.0"]
        output_dir.mkdir()
        (output_dir / "cardinal.imzML").write_text("xml")
        (output_dir / "cardinal.ibd").write_bytes(b"ibd")
        return type("Completed", (), {"stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("joint.backends.cardinal.subprocess.run", fake_run)
    monkeypatch.setattr(
        "joint.backends.cardinal.read_imzml",
        lambda path: type("Result", (), {"shape": (1, 1)})(),
    )
    result = run_cardinal(input_path, output_dir)
    assert result.shape == (1, 1)
```

- [ ] **Step 2: Run tests and verify backend is absent**

Run: `python -m pytest tests/test_cardinal_backend.py -v`  
Expected: collection fails because `joint.backends.cardinal` is absent.

- [ ] **Step 3: Implement the subprocess adapter**

```python
# src/joint/backends/__init__.py
"""Optional JOINT preprocessing backends."""
```

```python
# src/joint/backends/cardinal.py
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

from anndata import AnnData

from joint.errors import InputFormatError, OptionalDependencyError
from joint.io import read_imzml


def run_cardinal(
    input_imzml: str | Path,
    output_dir: str | Path,
    *,
    rscript: str = "Rscript",
    tolerance: float = 10.0,
    unit: str = "ppm",
    snr: float = 3.0,
) -> AnnData:
    executable = shutil.which(rscript)
    if executable is None:
        raise OptionalDependencyError(
            "Rscript is required for the Cardinal backend; install R and Cardinal"
        )
    source = Path(input_imzml).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    script = files("joint.backends").joinpath("cardinal_pipeline.R")
    command = [
        executable,
        str(script),
        str(source),
        str(destination),
        str(tolerance),
        unit,
        str(snr),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result_path = destination / "cardinal.imzML"
    if not result_path.is_file():
        raise InputFormatError(
            f"Cardinal completed without {result_path}; stdout={completed.stdout}; stderr={completed.stderr}"
        )
    return read_imzml(result_path)
```

- [ ] **Step 4: Add the versioned R entry script**

```r
# src/joint/backends/cardinal_pipeline.R
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("usage: cardinal_pipeline.R INPUT_IMZML OUTPUT_DIR TOLERANCE UNIT SNR")
}
if (!requireNamespace("Cardinal", quietly = TRUE)) {
  stop("The Cardinal R package is required")
}
input_imzml <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- normalizePath(args[[2]], mustWork = FALSE)
tolerance <- as.numeric(args[[3]])
unit <- args[[4]]
snr <- as.numeric(args[[5]])
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
msi <- Cardinal::readMSIData(input_imzml)
processed <- Cardinal::peakProcess(msi, SNR = snr)
aligned <- Cardinal::peakAlign(processed, tolerance = tolerance, units = unit)
Cardinal::writeMSIData(aligned, file = file.path(output_dir, "cardinal.imzML"))
```

- [ ] **Step 5: Dispatch `preprocess_msi` to Cardinal only when requested**

```python
# replace preprocess_msi signature/body dispatch in src/joint/pp.py
def preprocess_msi(
    path: str | Path,
    *,
    config: PreprocessingConfig,
    dataset_id: str | None = None,
    cardinal_output_dir: str | Path | None = None,
) -> anndata.AnnData:
    source = Path(path)
    if config.backend == "cardinal":
        if cardinal_output_dir is None:
            raise ValueError("cardinal_output_dir is required for the Cardinal backend")
        from joint.backends.cardinal import run_cardinal

        adata = run_cardinal(
            source,
            cardinal_output_dir,
            tolerance=config.peak_tolerance,
            unit=config.tolerance_unit,
            snr=config.cardinal_snr,
        )
    elif source.suffix.lower() == ".h5ad":
        adata = read_h5ad(source)
    elif source.suffix.lower() == ".imzml":
        adata = read_imzml(
            source,
            dataset_id=dataset_id,
            tolerance=config.peak_tolerance,
            unit=config.tolerance_unit,
            profile_bin_size=config.profile_bin_size,
        )
    else:
        raise InputFormatError(f"Unsupported MSI input: {source.suffix}")
    filtered = filter_features(adata, min_occurrence=config.min_occurrence)
    return normalize(filtered, method=config.normalization, layer=config.normalized_layer)
```

- [ ] **Step 6: Package the R script and run tests**

Add to `pyproject.toml`:

```toml
[tool.hatch.build]
include = ["src/joint/**/*.py", "src/joint/backends/*.R"]
```

Run: `python -m pytest tests/test_cardinal_backend.py tests/test_pp.py -v`  
Expected: all backend and preprocessing tests pass with subprocess behavior mocked.

- [ ] **Step 7: Commit Cardinal isolation**

```bash
git add pyproject.toml src/joint/backends src/joint/pp.py tests/test_cardinal_backend.py
git commit -m "feat: add optional Cardinal backend contract"
```

### Task 7: Foundation Verification and API Export

**Files:**
- Modify: `src/joint/__init__.py`
- Modify: `src/joint/io.py`
- Modify: `README.md`
- Modify: `tests/test_io.py`

**Interfaces:**
- Consumes: all APIs in this plan.
- Produces: `write_results` plus curated top-level `load_config`, `preprocess_msi`, `read_h5ad`, and `read_imzml` exports.

- [ ] **Step 1: Add a top-level export test**

```python
# append to tests/test_package.py
def test_foundation_api_is_available_from_joint():
    assert callable(joint.load_config)
    assert callable(joint.preprocess_msi)
    assert callable(joint.read_h5ad)
    assert callable(joint.read_imzml)
```

Append to `tests/test_io.py`:

```python
from joint.io import write_results
from joint.models import SegmentationResult


def test_write_results_serializes_segmentation_bundle(tmp_path: Path):
    result = SegmentationResult(
        labels=np.array([[0, 1]], dtype=np.int32),
        regions=pd.DataFrame({"label": [1], "area": [1]}),
        diagnostics={"object_count": 1},
    )
    artifacts = write_results(result, tmp_path / "bundle")
    assert set(artifacts) == {"labels", "regions", "report"}
    assert all(path.is_file() for path in artifacts.values())
```

Add `import pandas as pd` to `tests/test_io.py`.

- [ ] **Step 2: Run the export test and verify failure**

Run: `python -m pytest tests/test_package.py::test_foundation_api_is_available_from_joint tests/test_io.py::test_write_results_serializes_segmentation_bundle -v`  
Expected: the package export test fails with `AttributeError` and the I/O test fails because `write_results` is absent.

- [ ] **Step 3: Implement generic result serialization**

```python
# add imports to src/joint/io.py
import json

from joint.models import QuantificationResult, RegistrationResult, SegmentationResult


# append to src/joint/io.py
def _atomic_save_npy(array: np.ndarray, destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_write_text(text: str, destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_write_dataframe(table: pd.DataFrame, destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_results(
    result: anndata.AnnData | SegmentationResult | RegistrationResult | QuantificationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(result, anndata.AnnData):
        return {"adata": atomic_write_h5ad(result, destination / "result.h5ad")}
    if isinstance(result, SegmentationResult):
        labels_path = destination / "labels.npy"
        _atomic_save_npy(result.labels, labels_path)
        regions_path = destination / "regions.csv"
        _atomic_write_dataframe(result.regions, regions_path)
        report_path = destination / "report.json"
        serializable = {
            key: value
            for key, value in result.diagnostics.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        _atomic_write_text(json.dumps(serializable, indent=2, sort_keys=True), report_path)
        return {"labels": labels_path, "regions": regions_path, "report": report_path}
    if isinstance(result, RegistrationResult):
        adata_path = atomic_write_h5ad(result.adata, destination / "laser.h5ad")
        mapping_path = destination / "mapping.csv"
        _atomic_write_dataframe(result.mapping, mapping_path)
        report_path = destination / "report.json"
        _atomic_write_text(
            json.dumps({**result.transform, **result.report}, indent=2, default=str), report_path
        )
        return {"adata": adata_path, "mapping": mapping_path, "report": report_path}
    if isinstance(result, QuantificationResult):
        adata_path = atomic_write_h5ad(result.adata, destination / "cells.h5ad")
        artifacts = {"adata": adata_path}
        for name, table in {
            "overlaps": result.overlaps,
            "accepted": result.accepted,
            "rejected": result.rejected,
        }.items():
            path = destination / f"{name}.csv"
            _atomic_write_dataframe(table, path)
            artifacts[name] = path
        report_path = destination / "report.json"
        _atomic_write_text(json.dumps(result.report, indent=2, default=str), report_path)
        artifacts["report"] = report_path
        return artifacts
    raise TypeError(f"Unsupported JOINT result type: {type(result).__name__}")
```

- [ ] **Step 4: Export the stable foundation API**

```python
# src/joint/__init__.py
from joint.config import JointConfig, load_config
from joint.io import read_h5ad, read_imzml, write_results
from joint.models import PipelineResult, QuantificationResult, RegistrationResult, SegmentationResult
from joint.pp import preprocess_msi

__version__ = "0.1.0"

__all__ = [
    "JointConfig",
    "PipelineResult",
    "QuantificationResult",
    "RegistrationResult",
    "SegmentationResult",
    "load_config",
    "preprocess_msi",
    "read_h5ad",
    "read_imzml",
    "write_results",
    "__version__",
]
```

- [ ] **Step 5: Add a foundation usage example to README**

```python
from joint.config import load_config
from joint.pp import preprocess_msi

config = load_config("configs/d8.yaml")
source = config.input.imzml or config.input.h5ad
adata = preprocess_msi(source, config=config.preprocessing, dataset_id=config.project.name)
```

- [ ] **Step 6: Run the complete foundation verification**

Run: `python -m pytest -v`  
Expected: all foundation tests pass.

Run: `ruff check src tests`  
Expected: `All checks passed!`.

Run: `python -c "import joint; print(joint.__version__)"`  
Expected: `0.1.0`.

- [ ] **Step 7: Commit the foundation milestone**

```bash
git add src/joint/__init__.py src/joint/io.py README.md tests/test_package.py tests/test_io.py
git commit -m "docs: expose JOINT foundation API"
```

## Plan 1 Completion Gate

Run:

```bash
python -m pip install -e '.[test]'
python -m pytest -v
ruff check src tests
python -c "import joint; from joint import load_config, preprocess_msi"
```

The plan is complete when all commands succeed, core import works without optional packages, native preprocessing returns sparse AnnData with numeric `var["mz"]` and `obsm["spatial"]`, and Cardinal failures are isolated behind `OptionalDependencyError` or captured subprocess errors.
