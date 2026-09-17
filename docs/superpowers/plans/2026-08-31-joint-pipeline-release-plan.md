# JOINT Pipeline, Reproduction, and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the tested JOINT modules into a checkpointed `JointPipeline`, expose the approved CLI, add exact d2/d8 notebook-reproduction configurations plus raw-imzML variants, verify regression baselines, and complete user documentation and release checks.

**Architecture:** A manifest-backed stage store records input/configuration signatures and artifact paths. `JointPipeline` owns orchestration only; scientific calculations stay in the independent modules implemented in Plans 1–3. Typer commands call the same pipeline methods used by Python clients.

**Tech Stack:** Python 3.11+, Typer, Pydantic, PyYAML, JSON, hashlib, pathlib, AnnData, NumPy, pandas, scikit-image, pytest, nbformat; all APIs from Plans 1–3.

## Global Constraints

- Execute after Plans 1–3 pass.
- Stage outputs are written temporarily and promoted only after success.
- `--resume` reuses a stage only when its signature and artifacts match.
- Existing outputs require explicit `--overwrite`.
- CLI failures return nonzero status codes and library functions raise exceptions.
- d2 exact reproduction uses `legacy_proportional`; d8 uses `specificity_filtered`.
- d2 contains 40 rows with 37 retained marks per row after corrections: 1,480 registered laser observations.
- d8 contains 41 rows with 40 simulated marks per row: 1,640 registered laser observations.
- Exact notebook regression uses the existing h5ad inputs; separate raw configurations exercise imzML preprocessing.
- Public configs contain no user-specific absolute paths; `${JOINT_DATA_ROOT}` supplies the Figure5/5B data root.

---

## Planned File Map

```text
pyproject.toml                         Typer, nbformat, CLI entry point, pytest markers
src/joint/config.py                    environment expansion and downstream config models
src/joint/logging.py                   terminal and file logging setup
src/joint/pipeline.py                  stage store, manifest, JointPipeline
src/joint/cli.py                       `joint` commands
configs/d2.yaml                        d2 notebook reproduction from h5ad
configs/d2-raw.yaml                    d2 native imzML workflow
configs/d2-merges.csv                  exact d2 manual corrections
configs/d8.yaml                        d8 notebook reproduction from h5ad
configs/d8-raw.yaml                    d8 native imzML workflow
tests/test_pipeline.py                 stage/cache/orchestration tests
tests/test_cli.py                      CLI command tests
tests/test_provenance.py               logging and manifest metadata tests
tests/regression/test_reference_data.py d2/d8 baselines and raw-loader tests
examples/joint_quickstart.ipynb        Python API and pipeline example
README.md                              Chinese quick start and install/use guidance
docs/cardinal.md                       R/Cardinal setup and execution
docs/api.md                            public module/API map
```

### Task 1: Manifest-Backed Stage Store

**Files:**
- Create: `src/joint/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `JointConfig`, filesystem paths, producer callables.
- Produces: `file_signature`, `stage_signature`, and `StageStore.run`.

- [ ] **Step 1: Write failing stage-store tests**

```python
# tests/test_pipeline.py
import json
from pathlib import Path

import pytest

from joint.errors import JointError
from joint.pipeline import StageStore, file_signature, stage_signature


def test_file_signature_changes_when_input_changes(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("first")
    first = file_signature(path)
    path.write_text("second")
    second = file_signature(path)
    assert first != second


def test_stage_store_resumes_only_matching_completed_stage(tmp_path: Path):
    store = StageStore(tmp_path)
    artifact = tmp_path / "preprocessing" / "msi.h5ad"
    calls = []

    def producer():
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("result")
        calls.append("run")
        return {"adata": artifact}

    signature = stage_signature({"normalization": "rms"}, [])
    first = store.run("preprocessing", signature, producer, resume=False, overwrite=False)
    second = store.run("preprocessing", signature, producer, resume=True, overwrite=False)

    assert first == second == {"adata": artifact.resolve()}
    assert calls == ["run"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stages"]["preprocessing"]["status"] == "complete"


def test_stage_store_rejects_existing_mismatched_output_without_overwrite(tmp_path: Path):
    store = StageStore(tmp_path)
    artifact = tmp_path / "segmentation" / "labels.npy"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("old")
    with pytest.raises(JointError, match="overwrite"):
        store.run(
            "segmentation",
            "new-signature",
            lambda: {"labels": artifact},
            resume=False,
            overwrite=False,
        )
```

- [ ] **Step 2: Run tests and verify pipeline is absent**

Run: `python -m pytest tests/test_pipeline.py -v`  
Expected: collection fails because `joint.pipeline` is absent.

- [ ] **Step 3: Implement signatures and atomic JSON**

```python
# src/joint/pipeline.py
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from joint.errors import JointError


def file_signature(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    stat = source.stat()
    return {
        "path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def stage_signature(configuration: dict[str, Any], inputs: Iterable[str | Path]) -> str:
    payload = {
        "configuration": configuration,
        "inputs": [file_signature(path) for path in inputs],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
```

- [ ] **Step 4: Implement `StageStore` cache and overwrite behavior**

```python
# append to src/joint/pipeline.py
class StageStore:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"

    def _load(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"stages": {}}
        return json.loads(self.manifest_path.read_text())

    def run(
        self,
        stage: str,
        signature: str,
        producer: Callable[[], dict[str, Path]],
        *,
        resume: bool,
        overwrite: bool,
    ) -> dict[str, Path]:
        manifest = self._load()
        record = manifest["stages"].get(stage)
        if resume and record and record.get("status") == "complete" and record.get("signature") == signature:
            artifacts = {name: Path(path) for name, path in record["artifacts"].items()}
            if all(path.is_file() for path in artifacts.values()):
                return artifacts
        stage_dir = self.output_dir / stage
        if stage_dir.exists() and any(stage_dir.iterdir()) and not overwrite:
            raise JointError(f"Stage {stage} already has outputs; pass overwrite=True")
        artifacts = {name: Path(path).resolve() for name, path in producer().items()}
        missing = [str(path) for path in artifacts.values() if not path.is_file()]
        if missing:
            raise JointError(f"Stage {stage} did not create expected artifacts: {missing}")
        manifest["stages"][stage] = {
            "status": "complete",
            "signature": signature,
            "artifacts": {name: str(path) for name, path in artifacts.items()},
        }
        _atomic_json(manifest, self.manifest_path)
        return artifacts
```

- [ ] **Step 5: Run stage-store tests**

Run: `python -m pytest tests/test_pipeline.py -v`  
Expected: 3 tests pass.

- [ ] **Step 6: Commit the stage store**

```bash
git add src/joint/pipeline.py tests/test_pipeline.py
git commit -m "feat: add checkpointed stage store"
```

### Task 2: Complete Configuration for Reproducible Stages

**Files:**
- Modify: `src/joint/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: YAML from public d2/d8 configs.
- Produces: environment-variable expansion plus typed QC, annotation, clustering, and trajectory fields.

- [ ] **Step 1: Add failing environment and downstream-config tests**

```python
# append to tests/test_config.py
import os


def test_load_config_expands_joint_data_root(monkeypatch, tmp_path: Path):
    data_root = tmp_path / "reference"
    monkeypatch.setenv("JOINT_DATA_ROOT", str(data_root))
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        """
project: {name: sample, output_dir: results}
input: {h5ad: "${JOINT_DATA_ROOT}/adatas/d2.h5ad"}
""".strip()
    )
    config = load_config(config_path)
    assert config.input.h5ad == (data_root / "adatas/d2.h5ad").resolve()


def test_unexpanded_environment_variable_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("JOINT_DATA_ROOT", raising=False)
    path = tmp_path / "sample.yaml"
    path.write_text(
        "project: {name: sample, output_dir: results}\ninput: {h5ad: '${JOINT_DATA_ROOT}/d2.h5ad'}\n"
    )
    with pytest.raises(ConfigurationError, match="JOINT_DATA_ROOT"):
        load_config(path)


def test_downstream_sections_have_stable_defaults(tmp_path: Path):
    path = tmp_path / "sample.yaml"
    path.write_text("project: {name: sample, output_dir: results}\ninput: {h5ad: input.h5ad}\n")
    config = load_config(path)
    assert config.qc.enabled is True
    assert config.clustering.resolution == 0.6
    assert config.annotation.enabled is False
    assert config.trajectory.enabled is False
```

- [ ] **Step 2: Run tests and verify failures**

Run: `python -m pytest tests/test_config.py -k 'environment or downstream' -v`  
Expected: tests fail because environment expansion and sections are absent.

- [ ] **Step 3: Add downstream configuration models**

```python
# add to src/joint/config.py
class QcConfig(StrictModel):
    enabled: bool = True
    outlier_contamination: float = Field(default=0.05, gt=0.0, lt=0.5)


class AnnotationConfig(StrictModel):
    enabled: bool = False
    hmdb_reference: Path | None = None
    metaboscape_reference: Path | None = None
    ion_mode: Literal["pos", "neg"] = "pos"
    hmdb_ppm: float = Field(default=5.0, gt=0.0)
    metaboscape_ppm: float = Field(default=3.0, gt=0.0)


class ClusteringConfig(StrictModel):
    enabled: bool = True
    n_neighbors: int = Field(default=15, gt=1)
    resolution: float = Field(default=0.6, gt=0.0)
    n_top_features: int | None = Field(default=None, gt=0)


class TrajectoryConfig(StrictModel):
    enabled: bool = False
    path_file: Path | None = None
    features: list[str] = Field(default_factory=list)
    points: int = Field(default=100, gt=1)
```

Add fields to `JointConfig`:

```python
qc: QcConfig = Field(default_factory=QcConfig)
annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
```

Add `hmdb_reference`, `metaboscape_reference`, and `path_file` to `_PATH_KEYS`.

- [ ] **Step 4: Expand environment variables before path conversion**

```python
# replace the path branch in _resolve_paths
if key in _PATH_KEYS and value is not None:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        variable = expanded[expanded.index("$") :].split("/")[0]
        raise ConfigurationError(f"Environment variable is not defined: {variable}")
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
```

Add `import os` to `src/joint/config.py`.

- [ ] **Step 5: Run configuration tests**

Run: `python -m pytest tests/test_config.py -v`  
Expected: all configuration tests pass.

- [ ] **Step 6: Commit final configuration schema**

```bash
git add src/joint/config.py tests/test_config.py
git commit -m "feat: complete pipeline configuration schema"
```

### Task 3: `JointPipeline` Through Cell Quantification

**Files:**
- Modify: `src/joint/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `JointConfig`, preprocessing, segmentation, registration, quantification APIs.
- Produces: `JointPipeline.from_config`, `preprocess`, `segment`, `register`, `quantify`, and `run_spatial_core`.

- [ ] **Step 1: Write failing orchestration tests using lightweight fakes**

```python
# append to tests/test_pipeline.py
import anndata
import numpy as np
import pandas as pd

from joint.config import load_config
from joint.models import QuantificationResult, RegistrationResult, SegmentationResult
from joint.pipeline import JointPipeline


def pipeline_config(tmp_path: Path):
    image = tmp_path / "laser.npy"
    np.save(image, np.zeros((5, 5)))
    cells = tmp_path / "cells.npy"
    np.save(cells, np.zeros((5, 5), dtype=np.int32))
    source = tmp_path / "input.h5ad"
    anndata.AnnData(
        np.ones((1, 1)),
        obs={"dataset": ["sample"]},
        var={"mz": [100.0]},
        obsm={"spatial": np.array([[1.0, 1.0]])},
    ).write_h5ad(source)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
project: {{name: sample, output_dir: {tmp_path / 'results'}}}
input:
  h5ad: {source}
  laser_image: {image}
  cell_segmentation: {cells}
laser_segmentation:
  method: real
registration: {{orientation: identity, edge_policy: error}}
quantification: {{method: legacy_proportional}}
""".strip()
    )
    return load_config(config_path)


def test_joint_pipeline_runs_spatial_core_and_writes_stage_artifacts(monkeypatch, tmp_path: Path):
    config = pipeline_config(tmp_path)
    pipeline = JointPipeline(config)
    msi = anndata.read_h5ad(config.input.h5ad)
    segmented = SegmentationResult(
        labels=np.array([[1]], dtype=np.int32),
        regions=pd.DataFrame(
            {
                "label": [1], "area": [1], "centroid-0": [1.0], "centroid-1": [1.0],
                "seg_label": ["1"], "morphology": ["intact"], "row_number": [1],
                "column_number": [1], "right_to_left": [-1],
            }
        ),
        diagnostics={},
    )
    registered = RegistrationResult(msi, segmented.regions.assign(pixel_id=msi.obs_names), {"orientation": "identity"}, {})
    empty_mapping = pd.DataFrame({"cell_id": pd.Series(dtype=str)})
    quantified = QuantificationResult(
        msi,
        empty_mapping.copy(),
        empty_mapping.copy(),
        empty_mapping.copy(),
        {},
    )
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    monkeypatch.setattr("joint.pipeline._segment_from_config", lambda *args, **kwargs: segmented)
    monkeypatch.setattr("joint.pipeline.register_laser_points", lambda *args, **kwargs: registered)
    monkeypatch.setattr("joint.pipeline.quantify_cells", lambda *args, **kwargs: quantified)

    result = pipeline.run_spatial_core(overwrite=False, resume=False)

    assert result.adata.shape == (1, 1)
    assert (config.project.output_dir / "preprocessing/msi.h5ad").is_file()
    assert (config.project.output_dir / "segmentation/laser_labels.npy").is_file()
    assert (config.project.output_dir / "registration/laser.h5ad").is_file()
    assert (config.project.output_dir / "quantification/cells.h5ad").is_file()
```

- [ ] **Step 2: Run the test and verify `JointPipeline` is absent**

Run: `python -m pytest tests/test_pipeline.py::test_joint_pipeline_runs_spatial_core_and_writes_stage_artifacts -v`  
Expected: import fails for `JointPipeline`.

- [ ] **Step 3: Add atomic NumPy, CSV, and JSON helpers**

```python
# append to src/joint/pipeline.py
import anndata
import numpy as np
import pandas as pd
from skimage import io as skio

from joint.config import JointConfig, load_config, write_resolved_config
from joint.io import atomic_write_h5ad, read_h5ad, read_segmentation
from joint.models import QuantificationResult, RegistrationResult, SegmentationResult
from joint.pp import preprocess_msi
from joint.quantification import quantify_cells
from joint.registration import register_laser_points
from joint.segmentation import (
    detect_rows,
    exclude_regions,
    load_merge_rules,
    merge_regions,
    segment_laser_marks,
    simulate_laser_marks,
)


def _atomic_save_npy(array: np.ndarray, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, destination)
    return destination


def _atomic_save_csv(table: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return destination
```

- [ ] **Step 4: Implement configuration-driven segmentation helper**

```python
# append to src/joint/pipeline.py
def _image_and_mask(config: JointConfig) -> tuple[np.ndarray, np.ndarray | None]:
    if config.input.laser_image is None:
        raise JointError("input.laser_image is required for segmentation")
    source = config.input.laser_image
    image = np.load(source) if source.suffix == ".npy" else skio.imread(source, as_gray=True)
    rectangle = config.laser_segmentation.parameters.get("include_rectangle")
    if rectangle is None:
        return np.asarray(image), None
    row0, col0, row1, col1 = map(int, rectangle)
    mask = np.zeros(np.asarray(image).shape[:2], dtype=bool)
    mask[row0 : row1 + 1, col0 : col1 + 1] = True
    return np.asarray(image), mask


def _segment_from_config(config: JointConfig, msi: anndata.AnnData) -> SegmentationResult:
    image, mask = _image_and_mask(config)
    parameters = dict(config.laser_segmentation.parameters)
    parameters.pop("include_rectangle", None)
    row_parameters = parameters.pop("row_detection", {})
    exclusion = parameters.pop("exclusion", {})
    segmented = segment_laser_marks(image, mask=mask, **parameters)
    rows = detect_rows(
        segmented.regions,
        expected_rows=config.laser_segmentation.expected_rows,
        **row_parameters,
    )
    if exclusion:
        rows = [exclude_regions(row, **exclusion) for row in rows]
    if config.laser_segmentation.method == "simulated":
        return simulate_laser_marks(
            msi,
            rows,
            image.shape[:2],
            radius=config.laser_segmentation.radius,
        )
    regions = pd.concat(rows, ignore_index=True)
    if config.laser_segmentation.merge_rules is not None:
        regions = merge_regions(regions, load_merge_rules(config.laser_segmentation.merge_rules))
    return SegmentationResult(segmented.labels, regions, segmented.diagnostics)
```

- [ ] **Step 5: Implement `JointPipeline` preprocessing and segmentation stages**

```python
# append to src/joint/pipeline.py
class JointPipeline:
    def __init__(self, config: JointConfig):
        self.config = config
        self.store = StageStore(config.project.output_dir)
        write_resolved_config(config, config.project.output_dir / "resolved-config.yaml")

    @classmethod
    def from_config(cls, path: str | Path) -> "JointPipeline":
        return cls(load_config(path))

    def preprocess(self, *, resume: bool, overwrite: bool) -> anndata.AnnData:
        source = self.config.input.imzml or self.config.input.h5ad
        source_inputs = [source]
        if source.suffix.lower() == ".imzml":
            source_inputs.append(source.with_suffix(".ibd"))
        signature = stage_signature(
            self.config.preprocessing.model_dump(mode="json"), source_inputs
        )

        def producer():
            adata = preprocess_msi(
                source,
                config=self.config.preprocessing,
                dataset_id=self.config.project.name,
                cardinal_output_dir=self.config.project.output_dir / "preprocessing/cardinal",
            )
            path = atomic_write_h5ad(adata, self.config.project.output_dir / "preprocessing/msi.h5ad")
            return {"adata": path}

        artifacts = self.store.run(
            "preprocessing", signature, producer, resume=resume, overwrite=overwrite
        )
        return read_h5ad(artifacts["adata"])

    def segment(
        self, msi: anndata.AnnData, *, resume: bool, overwrite: bool
    ) -> SegmentationResult:
        inputs = [self.config.input.laser_image]
        if self.config.laser_segmentation.merge_rules is not None:
            inputs.append(self.config.laser_segmentation.merge_rules)
        signature = stage_signature(
            self.config.laser_segmentation.model_dump(mode="json"),
            [path for path in inputs if path is not None],
        )
        labels_path = self.config.project.output_dir / "segmentation/laser_labels.npy"
        regions_path = self.config.project.output_dir / "segmentation/laser_regions.csv"

        def producer():
            result = _segment_from_config(self.config, msi)
            _atomic_save_npy(result.labels, labels_path)
            _atomic_save_csv(result.regions, regions_path)
            return {"labels": labels_path, "regions": regions_path}

        artifacts = self.store.run(
            "segmentation", signature, producer, resume=resume, overwrite=overwrite
        )
        return SegmentationResult(
            labels=np.load(artifacts["labels"]),
            regions=pd.read_csv(artifacts["regions"], dtype={"seg_label": str}),
            diagnostics={},
        )
```

- [ ] **Step 6: Implement registration and quantification stages**

```python
# append methods inside JointPipeline
    def register(
        self,
        msi: anndata.AnnData,
        segmented: SegmentationResult,
        *,
        resume: bool,
        overwrite: bool,
    ) -> RegistrationResult:
        signature = stage_signature(
            self.config.registration.model_dump(mode="json"),
            [
                self.config.project.output_dir / "preprocessing/msi.h5ad",
                self.config.project.output_dir / "segmentation/laser_regions.csv",
            ],
        )
        adata_path = self.config.project.output_dir / "registration/laser.h5ad"
        mapping_path = self.config.project.output_dir / "registration/mapping.csv"

        def producer():
            result = register_laser_points(
                msi,
                segmented.regions.query('morphology != "spoilt"').copy(),
                orientation=self.config.registration.orientation,
                edge_policy=self.config.registration.edge_policy,
            )
            atomic_write_h5ad(result.adata, adata_path)
            _atomic_save_csv(result.mapping, mapping_path)
            return {"adata": adata_path, "mapping": mapping_path}

        artifacts = self.store.run(
            "registration", signature, producer, resume=resume, overwrite=overwrite
        )
        registered = read_h5ad(artifacts["adata"])
        mapping = pd.read_csv(artifacts["mapping"], dtype={"seg_label": str})
        return RegistrationResult(
            registered,
            mapping,
            self.config.registration.model_dump(mode="json"),
            {"registered_pixels": registered.n_obs},
        )

    def quantify(
        self,
        registered: RegistrationResult,
        segmented: SegmentationResult,
        *,
        resume: bool,
        overwrite: bool,
    ) -> QuantificationResult:
        if self.config.input.cell_segmentation is None:
            raise JointError("input.cell_segmentation is required for quantification")
        signature = stage_signature(
            self.config.quantification.model_dump(mode="json"),
            [
                self.config.project.output_dir / "registration/laser.h5ad",
                self.config.project.output_dir / "segmentation/laser_labels.npy",
                self.config.input.cell_segmentation,
            ],
        )
        cell_path = self.config.project.output_dir / "quantification/cells.h5ad"
        overlap_path = self.config.project.output_dir / "quantification/overlaps.csv"
        accepted_path = self.config.project.output_dir / "quantification/accepted.csv"
        rejected_path = self.config.project.output_dir / "quantification/rejected.csv"

        def producer():
            result = quantify_cells(
                registered.adata,
                read_segmentation(self.config.input.cell_segmentation),
                segmented.labels,
                **self.config.quantification.model_dump(),
            )
            atomic_write_h5ad(result.adata, cell_path)
            _atomic_save_csv(result.overlaps, overlap_path)
            _atomic_save_csv(result.accepted, accepted_path)
            _atomic_save_csv(result.rejected, rejected_path)
            return {
                "adata": cell_path,
                "overlaps": overlap_path,
                "accepted": accepted_path,
                "rejected": rejected_path,
            }

        artifacts = self.store.run(
            "quantification", signature, producer, resume=resume, overwrite=overwrite
        )
        return QuantificationResult(
            read_h5ad(artifacts["adata"]),
            pd.read_csv(artifacts["overlaps"]),
            pd.read_csv(artifacts["accepted"]),
            pd.read_csv(artifacts["rejected"]),
            {},
        )

    def run_spatial_core(self, *, resume: bool = False, overwrite: bool = False):
        msi = self.preprocess(resume=resume, overwrite=overwrite)
        segmented = self.segment(msi, resume=resume, overwrite=overwrite)
        registered = self.register(msi, segmented, resume=resume, overwrite=overwrite)
        return self.quantify(registered, segmented, resume=resume, overwrite=overwrite)
```

- [ ] **Step 7: Run pipeline tests**

Run: `python -m pytest tests/test_pipeline.py -v`  
Expected: all stage-store and orchestration tests pass.

- [ ] **Step 8: Commit the spatial pipeline**

```bash
git add src/joint/pipeline.py tests/test_pipeline.py
git commit -m "feat: orchestrate JOINT spatial stages"
```

### Task 4: QC, Annotation, Analysis, and Trajectory Pipeline Stages

**Files:**
- Modify: `src/joint/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: quantified cell AnnData and downstream configuration.
- Produces: `qc`, `annotate`, `analyze`, `trajectory`, and full `run` methods.

- [ ] **Step 1: Add a failing full-run dispatch test**

```python
# append to tests/test_pipeline.py
def test_run_calls_spatial_core_then_enabled_downstream_stages(monkeypatch, tmp_path: Path):
    pipeline = JointPipeline(pipeline_config(tmp_path))
    cell_result = QuantificationResult(anndata.AnnData([[1.0]], var={"mz": [100.0]}), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
    calls = []
    monkeypatch.setattr(pipeline, "run_spatial_core", lambda **kwargs: cell_result)
    monkeypatch.setattr(pipeline, "qc", lambda adata, **kwargs: calls.append("qc") or adata)
    monkeypatch.setattr(pipeline, "annotate", lambda adata, **kwargs: calls.append("annotate") or adata)
    monkeypatch.setattr(pipeline, "analyze", lambda adata, **kwargs: calls.append("analyze") or adata)
    monkeypatch.setattr(pipeline, "trajectory", lambda adata, **kwargs: calls.append("trajectory") or adata)

    pipeline.config.annotation.enabled = True
    pipeline.config.trajectory.enabled = True
    pipeline.run()

    assert calls == ["qc", "annotate", "analyze", "trajectory"]
```

Because Pydantic models are immutable by convention but not frozen, direct assignment is allowed here. If the final model configuration is frozen, create the test config with enabled values in YAML instead.

- [ ] **Step 2: Run the test and verify downstream methods are absent**

Run: `python -m pytest tests/test_pipeline.py::test_run_calls_spatial_core_then_enabled_downstream_stages -v`  
Expected: FAIL because `qc`, `annotate`, `analyze`, and `trajectory` are absent.

- [ ] **Step 3: Implement QC and annotation stages**

```python
# add imports to src/joint/pipeline.py
from joint.annotation import annotate_hmdb, annotate_metaboscape
from joint.qc import detect_outliers, spectral_qc


# add methods inside JointPipeline
    def qc(self, adata: anndata.AnnData, *, overwrite=False, resume=False) -> anndata.AnnData:
        input_path = self.config.project.output_dir / "quantification/cells.h5ad"
        signature = stage_signature(self.config.qc.model_dump(mode="json"), [input_path])
        report_path = self.config.project.output_dir / "qc/spectral.csv"
        adata_path = self.config.project.output_dir / "qc/cells_qc.h5ad"

        def producer():
            _atomic_save_csv(spectral_qc(adata).reset_index(names="obs_id"), report_path)
            result = adata.copy()
            result.obs["outlier"] = detect_outliers(
                result,
                contamination=self.config.qc.outlier_contamination,
                random_seed=self.config.project.random_seed,
            )
            atomic_write_h5ad(result, adata_path)
            return {"adata": adata_path, "spectral": report_path}

        artifacts = self.store.run(
            "qc", signature, producer, resume=resume, overwrite=overwrite
        )
        return read_h5ad(artifacts["adata"])

    def annotate(self, adata: anndata.AnnData, *, overwrite=False, resume=False) -> anndata.AnnData:
        input_path = (
            self.config.project.output_dir / "qc/cells_qc.h5ad"
            if self.config.qc.enabled
            else self.config.project.output_dir / "quantification/cells.h5ad"
        )
        reference_paths = [
            path
            for path in (
                self.config.annotation.hmdb_reference,
                self.config.annotation.metaboscape_reference,
            )
            if path is not None
        ]
        signature = stage_signature(
            self.config.annotation.model_dump(mode="json"), [input_path, *reference_paths]
        )
        adata_path = self.config.project.output_dir / "annotation/cells_annotated.h5ad"

        def producer():
            result = adata.copy()
            if self.config.annotation.hmdb_reference is not None:
                result = annotate_hmdb(
                    result,
                    self.config.annotation.hmdb_reference,
                    mode=self.config.annotation.ion_mode,
                    ppm=self.config.annotation.hmdb_ppm,
                )
            if self.config.annotation.metaboscape_reference is not None:
                result = annotate_metaboscape(
                    result,
                    self.config.annotation.metaboscape_reference,
                    ppm=self.config.annotation.metaboscape_ppm,
                )
            atomic_write_h5ad(result, adata_path)
            return {"adata": adata_path}

        artifacts = self.store.run(
            "annotation", signature, producer, resume=resume, overwrite=overwrite
        )
        return read_h5ad(artifacts["adata"])
```

- [ ] **Step 4: Implement clustering, COSG, cNMF, and trajectory stages**

```python
# add imports to src/joint/pipeline.py
from joint.analysis import cluster_cells, load_cnmf_results, prepare_cnmf, run_cnmf, run_cosg
from joint.trajectory import calculate_feature_trends, fit_spatial_trajectory


# add methods inside JointPipeline
    def analyze(self, adata: anndata.AnnData, *, overwrite=False, resume=False) -> anndata.AnnData:
        if self.config.annotation.enabled:
            input_path = self.config.project.output_dir / "annotation/cells_annotated.h5ad"
        elif self.config.qc.enabled:
            input_path = self.config.project.output_dir / "qc/cells_qc.h5ad"
        else:
            input_path = self.config.project.output_dir / "quantification/cells.h5ad"
        signature = stage_signature(
            {
                "clustering": self.config.clustering.model_dump(mode="json"),
                "analysis": self.config.analysis.model_dump(mode="json"),
            },
            [input_path],
        )
        analysis_dir = self.config.project.output_dir / "analysis"
        adata_path = analysis_dir / "cells_analyzed.h5ad"
        cosg_path = analysis_dir / "cosg.csv"
        cnmf_marker = analysis_dir / "cnmf-complete.json"

        def producer():
            result = adata.copy()
            artifacts = {"adata": adata_path}
            if self.config.clustering.enabled:
                result = cluster_cells(
                    result,
                    n_neighbors=self.config.clustering.n_neighbors,
                    resolution=self.config.clustering.resolution,
                    n_top_features=self.config.clustering.n_top_features,
                    random_seed=self.config.project.random_seed,
                )
            if self.config.analysis.cosg.enabled:
                table = run_cosg(
                    result,
                    groupby=self.config.analysis.cosg.groupby,
                    n_genes=self.config.analysis.cosg.n_genes,
                )
                _atomic_save_csv(table, cosg_path)
                artifacts["cosg"] = cosg_path
            if self.config.analysis.cnmf.enabled:
                prepared = prepare_cnmf(
                    result,
                    analysis_dir / "cnmf",
                    name=self.config.project.name,
                    components=self.config.analysis.cnmf.components,
                    seed=self.config.analysis.cnmf.seed,
                    num_highvar_genes=self.config.analysis.cnmf.num_highvar_genes,
                )
                run_cnmf(
                    prepared,
                    worker_index=0,
                    total_workers=1,
                    selected_k=self.config.analysis.cnmf.selected_k,
                )
                usage_pattern = (
                    analysis_dir
                    / "cnmf"
                    / self.config.project.name
                    / f"{self.config.project.name}.usages.k_{self.config.analysis.cnmf.selected_k}.*.txt"
                )
                candidates = sorted(usage_pattern.parent.glob(usage_pattern.name))
                if len(candidates) != 1:
                    raise JointError(f"Expected one cNMF usage file, found: {candidates}")
                result = load_cnmf_results(result, candidates[0])
                _atomic_json(
                    {"usage": str(candidates[0]), "selected_k": self.config.analysis.cnmf.selected_k},
                    cnmf_marker,
                )
                artifacts["cnmf"] = cnmf_marker
            atomic_write_h5ad(result, adata_path)
            return artifacts

        artifacts = self.store.run(
            "analysis", signature, producer, resume=resume, overwrite=overwrite
        )
        return read_h5ad(artifacts["adata"])

    def trajectory(self, adata: anndata.AnnData, *, overwrite=False, resume=False) -> anndata.AnnData:
        if self.config.trajectory.path_file is None:
            raise JointError("trajectory.path_file is required when trajectory is enabled")
        input_path = self.config.project.output_dir / "analysis/cells_analyzed.h5ad"
        signature = stage_signature(
            self.config.trajectory.model_dump(mode="json"),
            [input_path, self.config.trajectory.path_file],
        )
        trajectory_dir = self.config.project.output_dir / "trajectory"
        adata_path = trajectory_dir / "cells_trajectory.h5ad"
        trends_path = trajectory_dir / "trends.csv"

        def producer():
            path = pd.read_csv(self.config.trajectory.path_file)[["x", "y"]].to_numpy()
            result = fit_spatial_trajectory(adata, path)
            trends = calculate_feature_trends(
                result,
                self.config.trajectory.features,
                points=self.config.trajectory.points,
            )
            _atomic_save_csv(trends, trends_path)
            atomic_write_h5ad(result, adata_path)
            return {"adata": adata_path, "trends": trends_path}

        artifacts = self.store.run(
            "trajectory", signature, producer, resume=resume, overwrite=overwrite
        )
        return read_h5ad(artifacts["adata"])

    def run(self, *, resume: bool = False, overwrite: bool = False) -> anndata.AnnData:
        result = self.run_spatial_core(resume=resume, overwrite=overwrite).adata
        if self.config.qc.enabled:
            result = self.qc(result, resume=resume, overwrite=overwrite)
        if self.config.annotation.enabled:
            result = self.annotate(result, resume=resume, overwrite=overwrite)
        result = self.analyze(result, resume=resume, overwrite=overwrite)
        if self.config.trajectory.enabled:
            result = self.trajectory(result, resume=resume, overwrite=overwrite)
        atomic_write_h5ad(result, self.config.project.output_dir / "joint-final.h5ad")
        return result
```

- [ ] **Step 5: Run pipeline tests**

Run: `python -m pytest tests/test_pipeline.py -v`  
Expected: all tests pass, including a test added for downstream `StageStore` reuse.

- [ ] **Step 6: Commit the complete pipeline**

```bash
git add src/joint/pipeline.py tests/test_pipeline.py
git commit -m "feat: complete JOINT analysis pipeline"
```

### Task 5: Typer CLI

**Files:**
- Modify: `pyproject.toml`
- Create: `src/joint/cli.py`
- Create: `tests/test_cli.py`
- Modify: `src/joint/__init__.py`

**Interfaces:**
- Consumes: `JointPipeline` stage methods.
- Produces: `joint run`, `preprocess`, `segment`, `register`, `quantify`, `qc`, `annotate`, `analyze`, and `trajectory` commands.

- [ ] **Step 1: Add Typer and CLI entry metadata**

Add to core dependencies:

```toml
"typer>=0.12",
```

Add:

```toml
[project.scripts]
joint = "joint.cli:app"
```

- [ ] **Step 2: Write failing CLI tests**

```python
# tests/test_cli.py
from pathlib import Path

from typer.testing import CliRunner

from joint.cli import app


runner = CliRunner()


def test_help_lists_all_stage_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["run", "preprocess", "segment", "register", "quantify", "qc", "annotate", "analyze", "trajectory"]:
        assert command in result.stdout


def test_run_forwards_resume_and_overwrite(monkeypatch, tmp_path: Path):
    calls = []

    class FakePipeline:
        @classmethod
        def from_config(cls, path):
            calls.append(("config", Path(path)))
            return cls()

        def run(self, *, resume, overwrite):
            calls.append(("run", resume, overwrite))

    monkeypatch.setattr("joint.cli.JointPipeline", FakePipeline)
    config = tmp_path / "config.yaml"
    config.write_text("config")
    result = runner.invoke(app, ["run", "--config", str(config), "--resume", "--overwrite"])
    assert result.exit_code == 0
    assert calls[-1] == ("run", True, True)


def test_domain_error_returns_nonzero_status(monkeypatch, tmp_path: Path):
    class BrokenPipeline:
        @classmethod
        def from_config(cls, path):
            raise RuntimeError("bad config")

    monkeypatch.setattr("joint.cli.JointPipeline", BrokenPipeline)
    bad = tmp_path / "bad.yaml"
    bad.write_text("bad")
    result = runner.invoke(app, ["run", "--config", str(bad)])
    assert result.exit_code != 0
    assert "bad config" in result.stdout
```

- [ ] **Step 3: Run tests and verify CLI is absent**

Run: `python -m pytest tests/test_cli.py -v`  
Expected: collection fails because `joint.cli` is absent.

- [ ] **Step 4: Implement one shared command runner and stage commands**

```python
# src/joint/cli.py
from pathlib import Path

import typer

from joint.pipeline import JointPipeline


app = typer.Typer(no_args_is_help=True, help="JOINT spatial MSI analysis")


def _pipeline(config: Path) -> JointPipeline:
    return JointPipeline.from_config(config)


def _execute(action):
    try:
        return action()
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    resume: bool = False,
    overwrite: bool = False,
):
    _execute(lambda: _pipeline(config).run(resume=resume, overwrite=overwrite))


@app.command()
def preprocess(config: Path = typer.Option(..., exists=True), resume: bool = False, overwrite: bool = False):
    _execute(lambda: _pipeline(config).preprocess(resume=resume, overwrite=overwrite))


@app.command()
def segment(config: Path = typer.Option(..., exists=True), resume: bool = False, overwrite: bool = False):
    def action():
        pipeline = _pipeline(config)
        msi = pipeline.preprocess(resume=True, overwrite=False)
        pipeline.segment(msi, resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def register(config: Path = typer.Option(..., exists=True), resume: bool = False, overwrite: bool = False):
    def action():
        pipeline = _pipeline(config)
        msi = pipeline.preprocess(resume=True, overwrite=False)
        segmented = pipeline.segment(msi, resume=True, overwrite=False)
        pipeline.register(msi, segmented, resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def quantify(config: Path = typer.Option(..., exists=True), resume: bool = False, overwrite: bool = False):
    def action():
        pipeline = _pipeline(config)
        msi = pipeline.preprocess(resume=True, overwrite=False)
        segmented = pipeline.segment(msi, resume=True, overwrite=False)
        registered = pipeline.register(msi, segmented, resume=True, overwrite=False)
        pipeline.quantify(registered, segmented, resume=resume, overwrite=overwrite)

    _execute(action)


def _prepared_cells(pipeline: JointPipeline, target: str):
    result = pipeline.run_spatial_core(resume=True, overwrite=False).adata
    if target in {"qc", "annotate", "analyze", "trajectory"} and pipeline.config.qc.enabled:
        result = pipeline.qc(result, resume=True, overwrite=False)
    if target in {"annotate", "analyze", "trajectory"} and pipeline.config.annotation.enabled:
        result = pipeline.annotate(result, resume=True, overwrite=False)
    if target == "trajectory":
        result = pipeline.analyze(result, resume=True, overwrite=False)
    return result


@app.command()
def qc(config: Path = typer.Option(..., exists=True), resume: bool = False, overwrite: bool = False):
    def action():
        pipeline = _pipeline(config)
        cells = pipeline.run_spatial_core(resume=True, overwrite=False).adata
        pipeline.qc(cells, resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def annotate(config: Path = typer.Option(..., exists=True), resume: bool = False, overwrite: bool = False):
    def action():
        pipeline = _pipeline(config)
        pipeline.annotate(_prepared_cells(pipeline, "annotate"), resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def analyze(config: Path = typer.Option(..., exists=True), resume: bool = False, overwrite: bool = False):
    def action():
        pipeline = _pipeline(config)
        pipeline.analyze(_prepared_cells(pipeline, "analyze"), resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def trajectory(config: Path = typer.Option(..., exists=True), resume: bool = False, overwrite: bool = False):
    def action():
        pipeline = _pipeline(config)
        pipeline.trajectory(_prepared_cells(pipeline, "trajectory"), resume=resume, overwrite=overwrite)

    _execute(action)
```

- [ ] **Step 5: Export `JointPipeline` and run CLI tests**

Add to `src/joint/__init__.py`:

```python
from joint.pipeline import JointPipeline
```

Add `JointPipeline` to `__all__`.

Run: `python -m pip install -e '.[test]'`  
Expected: reinstall succeeds and creates the `joint` executable.

Run: `python -m pytest tests/test_cli.py -v`  
Expected: 3 tests pass.

Run: `joint --help`  
Expected: all nine commands appear.

- [ ] **Step 6: Commit the CLI**

```bash
git add pyproject.toml src/joint/cli.py src/joint/__init__.py tests/test_cli.py
git commit -m "feat: add JOINT command line interface"
```

### Task 6: d2/d8 Configurations, Corrections, and Regression Tests

**Files:**
- Create: `configs/d2.yaml`
- Create: `configs/d2-raw.yaml`
- Create: `configs/d2-merges.csv`
- Create: `configs/d8.yaml`
- Create: `configs/d8-raw.yaml`
- Create: `tests/regression/test_reference_data.py`
- Modify: `pyproject.toml`
- Modify: `src/joint/segmentation.py`
- Modify: `tests/test_segmentation.py`

**Interfaces:**
- Consumes: source data rooted at `${JOINT_DATA_ROOT}`.
- Produces: exact notebook-compatible workflows and real-data regression baselines.

- [ ] **Step 1: Add the d2 correction table exactly as encoded in the notebook**

```csv
# configs/d2-merges.csv
row_number,source_labels,note
7,276|284,notebook row 7
9,370|363,notebook row 9
11,448|455,notebook row 11
13,539|540,notebook row 13 first merge
13,536|532,notebook row 13 second merge
15,618|616,notebook row 15 first merge
15,607|603,notebook row 15 second merge
21,871|870,notebook row 21
23,951|952,notebook row 23 first merge
23,964|959,notebook row 23 second merge
25,1024|1027,notebook row 25 first merge
25,1045|1047,notebook row 25 second merge
28,1181|1176,notebook row 28 first merge
28,1159|1149,notebook row 28 second merge
29,1216|1220,notebook row 29 first merge
29,1210|1208,notebook row 29 second merge
31,1304|1302,notebook row 31
32,1339|1336,notebook row 32 first merge
32,1334|1335,notebook row 32 second merge
33,1383|1371,notebook row 33 first merge
33,1380|1381,notebook row 33 second merge
34,1427|1425,notebook row 34 first merge
34,1422|1423,notebook row 34 second merge
35,1477|1478,notebook row 35 first merge
35,1464|1465,notebook row 35 second merge
36,1527|1520,notebook row 36 first merge
36,1522|1516,notebook row 36 second merge
36,1513|1514,notebook row 36 third merge
38,1587|1582,notebook row 38
40,1692|1693,notebook row 40
```

- [ ] **Step 2: Support d8 post-border closing and marker connectivity**

Add parameters to `segment_laser_marks`:

```python
post_closing_size: int = 1,
marker_connectivity: int = 2,
```

Replace marker and post-border code with:

```python
binary = segmentation.clear_border(binary)
if post_closing_size > 1:
    binary = morphology.binary_closing(binary, morphology.square(post_closing_size))
if opening_size > 1:
    binary = morphology.binary_opening(binary, morphology.square(opening_size))
# after peak detection
markers = measure.label(maxima, connectivity=marker_connectivity)
```

Append this exact test to `tests/test_segmentation.py`:

```python
def test_post_border_closing_and_marker_connectivity_are_configurable():
    image = np.zeros((30, 30), dtype=float)
    image[10:15, 8:13] = 1.0
    image[10:15, 14:19] = 1.0
    for connectivity in (1, 2):
        result = segment_laser_marks(
            image,
            intensity_cutoff=0.1,
            top_hat_radius=0,
            closing_size=1,
            post_closing_size=3,
            opening_size=1,
            min_peak_distance=2,
            marker_connectivity=connectivity,
            min_size=1,
            threshold=0.5,
        )
        assert result.diagnostics["binary_mask"][12, 13]
```

- [ ] **Step 3: Add exact h5ad-based d2 and d8 reproduction configs**

```yaml
# configs/d2.yaml
project:
  name: d2
  output_dir: ../results/d2
  random_seed: 14
input:
  h5ad: ${JOINT_DATA_ROOT}/adatas/d2.h5ad
  laser_image: ${JOINT_DATA_ROOT}/pics/02/white.jpeg
  cell_segmentation: ${JOINT_DATA_ROOT}/segementation_d2/segmentation_cell_raw.pkl
preprocessing:
  backend: python
  normalization: none
  min_occurrence: 0.0
laser_segmentation:
  method: real
  expected_rows: 40
  merge_rules: d2-merges.csv
  parameters:
    include_rectangle: [1250, 1250, 3650, 3750]
    intensity_cutoff: 0.2
    top_hat_radius: 1
    closing_size: 5
    post_closing_size: 1
    opening_size: 3
    min_peak_distance: 25
    marker_connectivity: 2
    min_size: 30
    row_detection:
      method: rolling_variance
      window_size: 5
      variance_threshold: 50.0
      thinning: 4
    exclusion:
      min_column: 1610
registration:
  orientation: identity
  edge_policy: truncate_left
quantification:
  method: legacy_proportional
  boundary_margin: 50
analysis:
  cosg: {enabled: false}
  cnmf: {enabled: false}
```

```yaml
# configs/d8.yaml
project:
  name: d8
  output_dir: ../results/d8
  random_seed: 14
input:
  h5ad: ${JOINT_DATA_ROOT}/adatas/d8.h5ad
  laser_image: ${JOINT_DATA_ROOT}/pics/08/white.jpeg
  cell_segmentation: ${JOINT_DATA_ROOT}/segementation_d8/segmentation_cell_raw_08.pkl
preprocessing:
  backend: python
  normalization: none
  min_occurrence: 0.0
laser_segmentation:
  method: simulated
  radius: 18
  expected_rows: 41
  parameters:
    include_rectangle: [1200, 1300, 4350, 4350]
    intensity_cutoff: 0.2
    top_hat_radius: 1
    closing_size: 10
    post_closing_size: 10
    opening_size: 1
    min_peak_distance: 20
    marker_connectivity: 1
    min_size: 1
    row_detection:
      method: rolling_variance
      window_size: 5
      variance_threshold: 50.0
      thinning: 4
registration:
  orientation: identity
  edge_policy: error
quantification:
  method: specificity_filtered
  boundary_margin: 20
  unique_min_overlap: 0.0
  dominant_min_overlap: 0.6
  secondary_max_overlap: 0.15
analysis:
  cosg:
    enabled: true
    groupby: cluster
    n_genes: 20
  cnmf:
    enabled: true
    components: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    selected_k: 14
    seed: 14
    num_highvar_genes: 300
```

- [ ] **Step 4: Add raw-imzML variants by changing only input and preprocessing**

```yaml
# configs/d2-raw.yaml
project:
  name: d2-raw
  output_dir: ../results/d2-raw
  random_seed: 14
input:
  imzml: ${JOINT_DATA_ROOT}/rawdata/02.imzML
  laser_image: ${JOINT_DATA_ROOT}/pics/02/white.jpeg
  cell_segmentation: ${JOINT_DATA_ROOT}/segementation_d2/segmentation_cell_raw.pkl
preprocessing:
  backend: python
  normalization: rms
  normalized_layer: normalized
  min_occurrence: 0.05
  peak_tolerance: 10
  tolerance_unit: ppm
laser_segmentation:
  method: real
  expected_rows: 40
  merge_rules: d2-merges.csv
  parameters:
    include_rectangle: [1250, 1250, 3650, 3750]
    intensity_cutoff: 0.2
    top_hat_radius: 1
    closing_size: 5
    post_closing_size: 1
    opening_size: 3
    min_peak_distance: 25
    marker_connectivity: 2
    min_size: 30
    row_detection:
      method: rolling_variance
      window_size: 5
      variance_threshold: 50.0
      thinning: 4
    exclusion:
      min_column: 1610
registration:
  orientation: identity
  edge_policy: truncate_left
quantification:
  method: legacy_proportional
  boundary_margin: 50
analysis:
  cosg: {enabled: false}
  cnmf: {enabled: false}
```

```yaml
# configs/d8-raw.yaml
project:
  name: d8-raw
  output_dir: ../results/d8-raw
  random_seed: 14
input:
  imzml: ${JOINT_DATA_ROOT}/rawdata/08.imzML
  laser_image: ${JOINT_DATA_ROOT}/pics/08/white.jpeg
  cell_segmentation: ${JOINT_DATA_ROOT}/segementation_d8/segmentation_cell_raw_08.pkl
preprocessing:
  backend: python
  normalization: rms
  normalized_layer: normalized
  min_occurrence: 0.05
  peak_tolerance: 10
  tolerance_unit: ppm
laser_segmentation:
  method: simulated
  radius: 18
  expected_rows: 41
  parameters:
    include_rectangle: [1200, 1300, 4350, 4350]
    intensity_cutoff: 0.2
    top_hat_radius: 1
    closing_size: 10
    post_closing_size: 10
    opening_size: 1
    min_peak_distance: 20
    marker_connectivity: 1
    min_size: 1
    row_detection:
      method: rolling_variance
      window_size: 5
      variance_threshold: 50.0
      thinning: 4
registration:
  orientation: identity
  edge_policy: error
quantification:
  method: specificity_filtered
  boundary_margin: 20
  unique_min_overlap: 0.0
  dominant_min_overlap: 0.6
  secondary_max_overlap: 0.15
analysis:
  cosg:
    enabled: true
    groupby: cluster
    n_genes: 20
  cnmf:
    enabled: true
    components: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    selected_k: 14
    seed: 14
    num_highvar_genes: 300
```

- [ ] **Step 5: Add real-data regression tests with explicit baselines**

```python
# tests/regression/test_reference_data.py
import os
from pathlib import Path

import pytest

from joint.config import load_config
from joint.io import read_imzml
from joint.pipeline import JointPipeline


REFERENCE = os.environ.get("JOINT_DATA_ROOT")
pytestmark = pytest.mark.regression


def require_reference():
    if REFERENCE is None or not Path(REFERENCE).is_dir():
        pytest.skip("Set JOINT_DATA_ROOT to the Figure5/5B directory")


@pytest.mark.parametrize(
    ("config_name", "laser_count", "cell_count"),
    [("d2.yaml", 1480, 2936), ("d8.yaml", 1640, 4986)],
)
def test_notebook_reproduction_baselines(config_name, laser_count, cell_count):
    require_reference()
    config = load_config(Path("configs") / config_name)
    pipeline = JointPipeline(config)
    cells = pipeline.run_spatial_core(overwrite=True, resume=False)
    laser = config.project.output_dir / "registration/laser.h5ad"
    import anndata

    assert anndata.read_h5ad(laser).n_obs == laser_count
    assert cells.adata.n_obs == cell_count


@pytest.mark.parametrize(("filename", "pixel_count"), [("02.imzML", 1760), ("08.imzML", 1640)])
def test_raw_imzml_pixel_counts(filename, pixel_count):
    require_reference()
    adata = read_imzml(Path(REFERENCE) / "rawdata" / filename)
    assert adata.n_obs == pixel_count
```

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = ["regression: requires the external d2/d8 reference data"]
```

- [ ] **Step 6: Run config, segmentation, and local regression checks**

Run: `python -m pytest tests/test_config.py tests/test_segmentation.py -v`  
Expected: all focused tests pass.

Run: `python -m pytest -m 'not regression' -v`  
Expected: all non-regression tests pass.

Run with the available reference data:

```bash
JOINT_DATA_ROOT=/Users/mac/Desktop/nanoMap/nanoMAP_figures/Figure5/5B \
python -m pytest -m regression tests/regression/test_reference_data.py -v
```

Expected: d2 reports 1,480 laser observations and 2,936 cells; d8 reports 1,640 laser observations and 4,986 cells; raw imzML pixel counts are 1,760 and 1,640. If a count differs, save the stage diagnostics and compare the exact row/object counts to notebook outputs before changing a baseline.

- [ ] **Step 7: Commit reproduction workflows**

```bash
git add configs pyproject.toml src/joint/segmentation.py tests/test_segmentation.py tests/regression
git commit -m "test: add d2 and d8 reproduction workflows"
```

### Task 7: Example Notebook, Documentation, and Release Verification

**Files:**
- Modify: `pyproject.toml`
- Create: `examples/joint_quickstart.ipynb`
- Modify: `README.md`
- Create: `docs/cardinal.md`
- Create: `docs/api.md`
- Create: `docs/test-summary.md`

**Interfaces:**
- Consumes: final public Python and CLI APIs.
- Produces: user-facing installation, examples, API map, Cardinal setup, and verified test summary.

- [ ] **Step 1: Add notebook tooling to the test extra**

Add `nbformat>=5.10` and `nbclient>=0.10` to the `test` optional dependency list.

- [ ] **Step 2: Create an executable quick-start notebook**

Run this script once to create `examples/joint_quickstart.ipynb`:

```python
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


cells = [
    new_markdown_cell(
        "# JOINT quick start\n\n"
        "Set `JOINT_DATA_ROOT` to the nanoMAP `Figure5/5B` directory before executing this notebook."
    ),
    new_code_cell("from pathlib import Path\nfrom joint import JointPipeline, load_config"),
    new_code_cell(
        'config = load_config(Path("../configs/d8.yaml"))\n'
        "pipeline = JointPipeline(config)"
    ),
    new_code_cell(
        "msi = pipeline.preprocess(resume=True, overwrite=False)\n"
        "segmented = pipeline.segment(msi, resume=True, overwrite=False)\n"
        "registered = pipeline.register(msi, segmented, resume=True, overwrite=False)\n"
        "cells = pipeline.quantify(registered, segmented, resume=True, overwrite=False)\n"
        "cells.adata"
    ),
    new_code_cell(
        "from joint.pl import plot_registration, plot_spatial_feature\n"
        "plot_registration(registered)\n"
        "plot_spatial_feature(cells.adata, cells.adata.var_names[0])"
    ),
]
notebook = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
destination = Path("examples/joint_quickstart.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, destination)
```

- [ ] **Step 3: Replace README with a Chinese quick start**

The README must contain these sections with runnable commands:

````markdown
# JOINT

JOINT 用于连接 MSI 光谱、激光烧蚀点、细胞分割和单细胞代谢组学分析。

## 安装

```bash
python -m pip install -e '.[test]'
python -m pip install -e '.[cosg,cnmf,trajectory]'
```

## 配置参考数据

```bash
export JOINT_DATA_ROOT=/path/to/nanoMAP_figures/Figure5/5B
```

## 命令行

```bash
joint run --config configs/d8.yaml
joint quantify --config configs/d2.yaml --resume
```

## Python API

```python
from joint import JointPipeline

pipeline = JointPipeline.from_config("configs/d8.yaml")
result = pipeline.run(resume=True)
```
````

Also document the two quantification methods, h5ad versus raw configs, output folders, `--resume`, `--overwrite`, and optional extras.

- [ ] **Step 4: Write Cardinal and API documentation**

Write `docs/cardinal.md` with this content:

````markdown
# Cardinal backend

Install R and Cardinal:

```bash
R -q -e 'install.packages("BiocManager"); BiocManager::install("Cardinal")'
Rscript -e 'library(Cardinal); packageVersion("Cardinal")'
```

Set `preprocessing.backend: cardinal` in a raw-data configuration, then run:

```bash
joint preprocess --config configs/d8-raw.yaml
```

JOINT invokes `Rscript` with explicit input and output paths. Cardinal writes a processed imZML/ibd pair, and JOINT imports that pair into AnnData using the same coordinate and feature conventions as the native backend. Cardinal stdout and stderr are retained in the run log.
````

Write `docs/api.md` with this module map:

```markdown
# JOINT public API

- `joint.io`: `read_imzml`, `read_h5ad`, `read_segmentation`, `atomic_write_h5ad`
- `joint.pp`: `filter_features`, `normalize`, `remove_matrix_peaks`, `preprocess_msi`
- `joint.segmentation`: `segment_laser_marks`, `detect_rows`, `exclude_regions`, `merge_regions`, `simulate_laser_marks`
- `joint.registration`: `register_laser_points`, `mount_spatial_images`
- `joint.quantification`: `build_overlap_table`, `quantify_cells`, `build_mixed_anndata`
- `joint.qc`: `validate_anndata`, `spectral_qc`, `registration_report`, `coverage_report`, `detect_outliers`, `pcoa`, `silhouette_summary`
- `joint.annotation`: `annotate_hmdb`, `annotate_metaboscape`
- `joint.analysis`: `cluster_cells`, `rank_metabolites`, `run_cosg`, `prepare_cnmf`, `run_cnmf`, `load_cnmf_results`
- `joint.trajectory`: `fit_spatial_trajectory`, `calculate_feature_trends`
- `joint.pl`: segmentation, registration, spatial-feature, cell-contour, cNMF-usage, and trajectory plots
- `joint.pipeline`: `JointPipeline`

`configs/d2.yaml` reproduces the notebook with `legacy_proportional`. `configs/d8.yaml` reproduces the notebook with the default `specificity_filtered` method.
```

- [ ] **Step 5: Execute notebook and all automated checks**

Run:

```bash
python -m pip install -e '.[test]'
python -m pytest -m 'not regression' -v
ruff check src tests
python - <<'PY'
import nbformat
from nbclient import NotebookClient

path = "examples/joint_quickstart.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=1200, kernel_name="python3").execute(cwd="examples")
nbformat.write(notebook, path)
PY
joint --help
joint run --help
```

Expected: tests and lint pass, notebook executes using existing/resumable d8 artifacts when `JOINT_DATA_ROOT` is set, and both help commands exit zero.

- [ ] **Step 6: Run reference regression and write the exact summary**

Run:

```bash
JOINT_DATA_ROOT=/Users/mac/Desktop/nanoMap/nanoMAP_figures/Figure5/5B \
python -m pytest -m regression tests/regression/test_reference_data.py -v
```

Write `docs/test-summary.md` with the date, Python version, JOINT commit, exact commands, pass/fail counts, d2/d8 laser and cell counts, and any optional dependency tests skipped because the real package was unavailable. Do not claim a pass for a command that was not executed successfully.

- [ ] **Step 7: Verify a clean install and source distribution**

Run:

```bash
python -m pip install build
python -m build
python -m pip install --force-reinstall dist/joint_msi-0.1.0-py3-none-any.whl
python -c "import joint; print(joint.__version__)"
joint --help
```

Expected: wheel and source distribution build; installed version prints `0.1.0`; CLI help exits zero.

- [ ] **Step 8: Commit documentation and release evidence**

```bash
git add pyproject.toml README.md docs examples/joint_quickstart.ipynb
git commit -m "docs: complete JOINT quick start and release evidence"
```

### Task 8: Provenance Metadata, Failure Records, Logs, and Diagnostic Figures

**Files:**
- Create: `src/joint/logging.py`
- Create: `tests/test_provenance.py`
- Modify: `src/joint/pipeline.py`

**Interfaces:**
- Consumes: `JointConfig`, stage inputs, artifacts, and stage exceptions.
- Produces: `configure_logging`, `build_run_metadata`, enriched manifest records, `logs/joint.log`, and automatic segmentation/registration diagnostic figures.

- [ ] **Step 1: Write failing logging and provenance tests**

```python
# tests/test_provenance.py
import json
from pathlib import Path

import pytest

from joint.logging import configure_logging
from joint.pipeline import StageStore


def test_configure_logging_writes_terminal_messages_to_run_log(tmp_path: Path):
    logger = configure_logging(tmp_path)
    logger.info("stage message")
    for handler in logger.handlers:
        handler.flush()
    assert "stage message" in (tmp_path / "logs/joint.log").read_text()


def test_stage_store_records_failure_and_artifact_hashes(tmp_path: Path):
    store = StageStore(tmp_path, logger=configure_logging(tmp_path))
    store.initialize({"joint_version": "0.1.0", "random_seed": 14, "inputs": []})

    with pytest.raises(RuntimeError, match="boom"):
        store.run(
            "broken",
            "signature",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            resume=False,
            overwrite=False,
        )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run"]["random_seed"] == 14
    assert manifest["stages"]["broken"]["status"] == "failed"

    artifact = tmp_path / "working" / "result.txt"

    def producer():
        artifact.parent.mkdir()
        artifact.write_text("result")
        return {"result": artifact}

    store.run("working", "signature", producer, resume=False, overwrite=False)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["stages"]["working"]["artifact_signatures"]["result"]["sha256"]) == 64
```

- [ ] **Step 2: Run tests and verify logging/provenance support is absent**

Run: `python -m pytest tests/test_provenance.py -v`  
Expected: collection fails because `joint.logging` is absent.

- [ ] **Step 3: Implement idempotent terminal and file logging**

```python
# src/joint/logging.py
import logging
import sys
from pathlib import Path


def configure_logging(output_dir: str | Path, *, verbose: bool = False) -> logging.Logger:
    destination = Path(output_dir).resolve()
    log_dir = destination / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"joint.{destination}")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_dir / "joint.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
```

- [ ] **Step 4: Add complete run metadata and input fingerprints**

```python
# add imports to src/joint/pipeline.py
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from joint.errors import InputFormatError
from joint.logging import configure_logging


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_metadata(config: JointConfig) -> dict[str, Any]:
    try:
        joint_version = version("joint-msi")
    except PackageNotFoundError:
        joint_version = "0.1.0+editable"
    candidates = [
        config.input.imzml,
        config.input.h5ad,
        config.input.laser_image,
        config.input.cell_segmentation,
        config.laser_segmentation.merge_rules,
        config.annotation.hmdb_reference,
        config.annotation.metaboscape_reference,
        config.trajectory.path_file,
    ]
    if config.input.imzml is not None:
        candidates.append(config.input.imzml.with_suffix(".ibd"))
    inputs = []
    for path in [item for item in candidates if item is not None]:
        if not path.is_file():
            raise InputFormatError(f"Configured input does not exist: {path}")
        inputs.append(file_signature(path))
    dependency_versions = {}
    for package in [
        "anndata", "numpy", "pandas", "pydantic", "pyimzml", "pyyaml", "scipy",
        "scikit-image", "scanpy", "cosg", "cnmf", "pygam",
    ]:
        try:
            dependency_versions[package] = version(package)
        except PackageNotFoundError:
            continue
    cardinal_version = None
    rscript = shutil.which("Rscript")
    if config.preprocessing.backend == "cardinal" and rscript is not None:
        completed = subprocess.run(
            [
                rscript,
                "-e",
                'cat(as.character(getRversion()), "|", as.character(packageVersion("Cardinal")))',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cardinal_version = completed.stdout.strip()
    return {
        "joint_version": joint_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": dependency_versions,
        "r_cardinal_version": cardinal_version,
        "random_seed": config.project.random_seed,
        "configuration": config.model_dump(mode="json"),
        "inputs": inputs,
        "created_at": _utc_now(),
    }
```

- [ ] **Step 5: Enrich `StageStore` with run initialization, timestamps, failures, hashes, and logs**

Replace `StageStore.__init__` and `run` with:

```python
class StageStore:
    def __init__(self, output_dir: str | Path, logger=None):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.logger = logger

    def _load(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"stages": {}}
        return json.loads(self.manifest_path.read_text())

    def initialize(self, metadata: dict[str, Any]) -> None:
        manifest = self._load()
        manifest["run"] = metadata
        manifest.setdefault("stages", {})
        _atomic_json(manifest, self.manifest_path)

    def run(
        self,
        stage: str,
        signature: str,
        producer: Callable[[], dict[str, Path]],
        *,
        resume: bool,
        overwrite: bool,
    ) -> dict[str, Path]:
        manifest = self._load()
        record = manifest["stages"].get(stage)
        if resume and record and record.get("status") == "complete" and record.get("signature") == signature:
            artifacts = {name: Path(path) for name, path in record["artifacts"].items()}
            if all(path.is_file() for path in artifacts.values()):
                if self.logger:
                    self.logger.info("Reusing stage %s", stage)
                return artifacts
        stage_dir = self.output_dir / stage
        if stage_dir.exists() and any(stage_dir.iterdir()) and not overwrite:
            raise JointError(f"Stage {stage} already has outputs; pass overwrite=True")
        started_at = _utc_now()
        manifest["stages"][stage] = {
            "status": "running",
            "signature": signature,
            "started_at": started_at,
        }
        _atomic_json(manifest, self.manifest_path)
        if self.logger:
            self.logger.info("Starting stage %s", stage)
        try:
            artifacts = {name: Path(path).resolve() for name, path in producer().items()}
            missing = [str(path) for path in artifacts.values() if not path.is_file()]
            if missing:
                raise JointError(f"Stage {stage} did not create expected artifacts: {missing}")
        except Exception as exc:
            manifest = self._load()
            manifest["stages"][stage].update(
                {"status": "failed", "completed_at": _utc_now(), "error": str(exc)}
            )
            _atomic_json(manifest, self.manifest_path)
            if self.logger:
                self.logger.exception("Stage %s failed", stage)
            raise
        manifest = self._load()
        manifest["stages"][stage] = {
            "status": "complete",
            "signature": signature,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "artifact_signatures": {name: file_signature(path) for name, path in artifacts.items()},
        }
        _atomic_json(manifest, self.manifest_path)
        if self.logger:
            self.logger.info("Completed stage %s", stage)
        return artifacts
```

- [ ] **Step 6: Initialize provenance in `JointPipeline`**

Replace `JointPipeline.__init__` with:

```python
def __init__(self, config: JointConfig):
    self.config = config
    self.logger = configure_logging(config.project.output_dir)
    self.store = StageStore(config.project.output_dir, logger=self.logger)
    write_resolved_config(config, config.project.output_dir / "resolved-config.yaml")
    self.store.initialize(build_run_metadata(config))
```

- [ ] **Step 7: Save diagnostic plots and coverage reports during stage production**

Add imports:

```python
from joint.pl import plot_registration, plot_segmentation
from joint.qc import coverage_report
```

In the segmentation producer, after saving labels and regions:

```python
figure_path = self.config.project.output_dir / "figures/laser-segmentation.png"
figure, _ = plot_segmentation(result.labels, save=figure_path)
figure.clear()
return {"labels": labels_path, "regions": regions_path, "figure": figure_path}
```

In the registration producer, after saving the mapping:

```python
figure_path = self.config.project.output_dir / "figures/registration.png"
figure, _ = plot_registration(result, save=figure_path)
figure.clear()
return {"adata": adata_path, "mapping": mapping_path, "figure": figure_path}
```

In the quantification producer, before returning artifacts:

```python
coverage_path = self.config.project.output_dir / "quantification/coverage.csv"
_atomic_save_csv(coverage_report(result), coverage_path)
return {
    "adata": cell_path,
    "overlaps": overlap_path,
    "accepted": accepted_path,
    "rejected": rejected_path,
    "coverage": coverage_path,
}
```

Update the stage readers to ignore the extra diagnostic artifact names; their existing named lookups for `adata`, `labels`, `regions`, and mapping tables remain unchanged.

- [ ] **Step 8: Run provenance and full non-regression tests**

Run: `python -m pytest tests/test_provenance.py tests/test_pipeline.py -v`  
Expected: provenance and pipeline tests pass.

Run: `python -m pytest -m 'not regression' -v`  
Expected: all non-regression tests pass.

- [ ] **Step 9: Refresh release evidence after provenance changes**

Run:

```bash
ruff check src tests
python -m build
JOINT_DATA_ROOT=/Users/mac/Desktop/nanoMap/nanoMAP_figures/Figure5/5B python -m pytest -m regression -v
```

Update `docs/test-summary.md` with these final command results and confirm the manifest contains run metadata, completed stage timestamps, and artifact SHA-256 values.

- [ ] **Step 10: Commit logs and provenance**

```bash
git add src/joint/logging.py src/joint/pipeline.py tests/test_provenance.py docs/test-summary.md
git commit -m "feat: add JOINT run provenance and logging"
```

## Plan 4 Completion Gate

Run:

```bash
python -m pytest -m 'not regression' -v
JOINT_DATA_ROOT=/Users/mac/Desktop/nanoMap/nanoMAP_figures/Figure5/5B python -m pytest -m regression -v
ruff check src tests
python -m build
joint --help
git status --short
```

The software is ready for completion review when all required commands pass, `git status --short` is empty, d2/d8 baseline counts match, the example notebook runs, and the test summary contains only observed evidence.
