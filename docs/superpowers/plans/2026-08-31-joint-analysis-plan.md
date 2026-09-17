# JOINT Analysis and Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add QC, metabolite annotation, clustering and ranking, optional COSG/cNMF integrations, spatial trajectory analysis, and reusable plotting APIs to the tested JOINT spatial core.

**Architecture:** Core analysis functions operate on copies of AnnData and attach results under documented keys. COSG, cNMF, and pyGAM imports are lazy and raise `OptionalDependencyError` only when invoked. Plotting functions consume result objects or AnnData and return Matplotlib figure/axes objects without saving unless a path is supplied.

**Tech Stack:** Python 3.11+, Scanpy, scikit-learn, scikit-bio, Matplotlib, scikit-image, NumPy, pandas, AnnData; optional COSG, cNMF, and pyGAM; pytest and Ruff.

## Global Constraints

- Execute after Plans 1 and 2 pass.
- Original feature IDs and raw abundance matrices are preserved.
- Annotation uses numeric m/z values and retains multiple matches.
- Optional dependencies are not imported by `import joint`.
- Analysis randomness uses explicit seeds.
- Plotting functions return objects and save only when `save` is provided.
- Analysis results use stable keys under `adata.uns["joint"]` or documented `obs` columns.
- No automatic biological interpretation or hard-coded sample labels is added.

---

## Planned File Map

```text
pyproject.toml                    core analysis and optional extra dependencies
src/joint/qc.py                   validation, spectral, registration, and coverage QC
src/joint/annotation.py           HMDB and MetaboScape matching
src/joint/analysis.py             clustering, ranking, COSG, and cNMF adapters
src/joint/trajectory.py           path projection and feature-trend fitting
src/joint/pl.py                   reusable plotting functions
src/joint/__init__.py             curated analysis exports
tests/test_qc.py                  QC tests
tests/test_annotation.py          ppm matching tests
tests/test_analysis.py            clustering/ranking and lazy optional dependency tests
tests/test_trajectory.py          path and trend tests
tests/test_pl.py                  plotting smoke tests
```

### Task 1: QC Validation and Reports

**Files:**
- Modify: `pyproject.toml`
- Create: `src/joint/qc.py`
- Create: `tests/test_qc.py`

**Interfaces:**
- Consumes: AnnData, `RegistrationResult`, `QuantificationResult`, segmentation arrays.
- Produces: `validate_anndata`, `spectral_qc`, `registration_report`, `coverage_report`, and `detect_outliers`.

- [ ] **Step 1: Add analysis dependencies**

Add to `[project].dependencies`:

```toml
"matplotlib>=3.8",
"scanpy>=1.10",
"igraph>=0.11",
"leidenalg>=0.10",
"scikit-bio>=0.6",
"scikit-learn>=1.4",
```

Add optional extras:

```toml
cosg = ["cosg>=1.0"]
cnmf = ["cnmf>=1.3"]
trajectory = ["pygam>=0.9"]
all = ["cosg>=1.0", "cnmf>=1.3", "pygam>=0.9"]
```

- [ ] **Step 2: Write failing QC tests**

```python
# tests/test_qc.py
import anndata
import numpy as np
import pytest
from scipy.sparse import csr_matrix

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


def sample_adata():
    return anndata.AnnData(
        csr_matrix([[1.0, 0.0], [2.0, 3.0], [100.0, 100.0]]),
        obs={"dataset": ["s", "s", "s"]},
        var={"mz": [100.0, 200.0]},
        obsm={"spatial": np.array([[0, 0], [1, 0], [2, 0]])},
    )


def test_validate_anndata_requires_spatial_and_numeric_mz():
    validate_anndata(sample_adata(), require_spatial=True)
    broken = sample_adata()
    del broken.obsm["spatial"]
    with pytest.raises(InputFormatError, match="spatial"):
        validate_anndata(broken, require_spatial=True)


def test_spectral_qc_reports_total_and_detected_features():
    report = spectral_qc(sample_adata())
    assert report.loc[report.index[0], "total_intensity"] == 1.0
    assert report.loc[report.index[1], "detected_features"] == 2


def test_registration_and_coverage_reports_are_tidy():
    registered = RegistrationResult(sample_adata(), None, {"orientation": "identity"}, {"registered_pixels": 3})
    quantified = QuantificationResult(
        sample_adata(),
        None,
        None,
        None,
        {"included_cells": 3, "quantified_cells": 2, "rejected_mappings": 1},
    )
    assert registration_report(registered).set_index("metric").loc["registered_pixels", "value"] == 3
    assert coverage_report(quantified).set_index("metric").loc["coverage_fraction", "value"] == 2 / 3


def test_detect_outliers_returns_boolean_series():
    outliers = detect_outliers(sample_adata(), contamination=1 / 3, random_seed=7)
    assert outliers.dtype == bool
    assert outliers.sum() == 1


def test_pcoa_and_silhouette_return_tidy_tables():
    adata = anndata.AnnData(np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]))
    adata.obs["cluster"] = ["a", "a", "b", "b"]
    coordinates = pcoa(adata, dimensions=2)
    summary = silhouette_summary(adata, labels="cluster")
    assert coordinates.columns.tolist() == ["PCoA1", "PCoA2"]
    assert set(summary["cluster"]) == {"a", "b"}
```

- [ ] **Step 3: Run tests and verify QC is absent**

Run: `python -m pytest tests/test_qc.py -v`  
Expected: collection fails because `joint.qc` is absent.

- [ ] **Step 4: Implement QC APIs**

```python
# src/joint/qc.py
import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances, silhouette_samples
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa as skbio_pcoa

from joint.errors import InputFormatError
from joint.models import QuantificationResult, RegistrationResult


def validate_anndata(adata: anndata.AnnData, *, require_spatial: bool = False) -> None:
    if "mz" not in adata.var:
        raise InputFormatError('adata.var must contain numeric column "mz"')
    try:
        adata.var["mz"].astype(float)
    except (TypeError, ValueError) as exc:
        raise InputFormatError('adata.var["mz"] must be numeric') from exc
    if require_spatial:
        if "spatial" not in adata.obsm or np.asarray(adata.obsm["spatial"]).shape != (adata.n_obs, 2):
            raise InputFormatError('adata.obsm["spatial"] must have shape (n_obs, 2)')


def spectral_qc(adata: anndata.AnnData, *, layer: str | None = None) -> pd.DataFrame:
    matrix = adata.layers[layer] if layer else adata.X
    total = np.asarray(matrix.sum(axis=1)).ravel()
    detected = matrix.getnnz(axis=1) if sparse.issparse(matrix) else np.count_nonzero(matrix, axis=1)
    return pd.DataFrame(
        {"total_intensity": total, "detected_features": np.asarray(detected).ravel()},
        index=adata.obs_names.copy(),
    )


def registration_report(result: RegistrationResult) -> pd.DataFrame:
    payload = {**result.report, **{f"transform_{k}": v for k, v in result.transform.items()}}
    return pd.DataFrame({"metric": payload.keys(), "value": payload.values()})


def coverage_report(result: QuantificationResult) -> pd.DataFrame:
    payload = dict(result.report)
    included = int(payload.get("included_cells", 0))
    quantified = int(payload.get("quantified_cells", 0))
    payload["coverage_fraction"] = quantified / included if included else 0.0
    return pd.DataFrame({"metric": payload.keys(), "value": payload.values()})


def detect_outliers(
    adata: anndata.AnnData,
    *,
    contamination: float = 0.05,
    random_seed: int = 0,
    layer: str | None = None,
) -> pd.Series:
    report = spectral_qc(adata, layer=layer)
    labels = IsolationForest(contamination=contamination, random_state=random_seed).fit_predict(report)
    return pd.Series(labels == -1, index=adata.obs_names, name="outlier")


def _dense_matrix(adata: anndata.AnnData, layer: str | None = None) -> np.ndarray:
    matrix = adata.layers[layer] if layer else adata.X
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def pcoa(
    adata: anndata.AnnData,
    *,
    dimensions: int = 2,
    metric: str = "braycurtis",
    layer: str | None = None,
) -> pd.DataFrame:
    distances = pairwise_distances(_dense_matrix(adata, layer), metric=metric)
    ordination = skbio_pcoa(DistanceMatrix(distances, ids=adata.obs_names.astype(str)))
    values = ordination.samples.iloc[:, :dimensions].copy()
    values.columns = [f"PCoA{index + 1}" for index in range(values.shape[1])]
    return values


def silhouette_summary(
    adata: anndata.AnnData,
    *,
    labels: str,
    metric: str = "euclidean",
    layer: str | None = None,
) -> pd.DataFrame:
    groups = adata.obs[labels].astype(str)
    values = silhouette_samples(_dense_matrix(adata, layer), groups, metric=metric)
    return (
        pd.DataFrame({"cluster": groups.to_numpy(), "silhouette": values})
        .groupby("cluster", as_index=False)
        .agg(mean_silhouette=("silhouette", "mean"), cells=("silhouette", "size"))
    )
```

- [ ] **Step 5: Run QC tests**

Run: `python -m pytest tests/test_qc.py -v`  
Expected: 5 tests pass.

- [ ] **Step 6: Commit QC support**

```bash
git add pyproject.toml src/joint/qc.py tests/test_qc.py
git commit -m "feat: add JOINT quality control reports"
```

### Task 2: HMDB and MetaboScape Annotation

**Files:**
- Create: `src/joint/annotation.py`
- Create: `tests/test_annotation.py`

**Interfaces:**
- Consumes: AnnData with numeric `var["mz"]` and reference DataFrames or files.
- Produces: `annotate_hmdb` and `annotate_metaboscape`, each returning a copied AnnData and retaining multiple matches as JSON-compatible lists.

- [ ] **Step 1: Write failing annotation tests**

```python
# tests/test_annotation.py
import json

import anndata
import numpy as np
import pandas as pd

from joint.annotation import annotate_hmdb, annotate_metaboscape


def query_adata():
    return anndata.AnnData(
        np.ones((1, 2)),
        var=pd.DataFrame({"mz": [100.0, 200.0]}, index=["m1", "m2"]),
    )


def test_hmdb_annotation_retains_multiple_matches_without_changing_ids():
    reference = pd.DataFrame(
        {
            "accession": ["HMDB1", "HMDB2", "HMDB3"],
            "name": ["A", "B", "C"],
            "mz": [100.0002, 100.0004, 300.0],
            "mode": ["pos", "pos", "pos"],
        }
    )
    annotated = annotate_hmdb(query_adata(), reference, mode="pos", ppm=5)
    assert annotated.var_names.tolist() == ["m1", "m2"]
    assert json.loads(annotated.var.loc["m1", "hmdb_accessions"]) == ["HMDB1", "HMDB2"]
    assert json.loads(annotated.var.loc["m2", "hmdb_accessions"]) == []


def test_metaboscape_annotation_validates_and_matches_columns():
    reference = pd.DataFrame(
        {"Measured m/z": [199.9995], "Molecular Formula": ["C6H8"], "Name": ["candidate"]}
    )
    annotated = annotate_metaboscape(query_adata(), reference, ppm=3)
    assert json.loads(annotated.var.loc["m2", "metaboscape_names"]) == ["candidate"]
```

- [ ] **Step 2: Run tests and verify annotation is absent**

Run: `python -m pytest tests/test_annotation.py -v`  
Expected: collection fails because `joint.annotation` is absent.

- [ ] **Step 3: Implement one shared ppm matcher and both adapters**

```python
# src/joint/annotation.py
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from joint.errors import InputFormatError


def _load_table(reference: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(reference, pd.DataFrame):
        return reference.copy()
    source = Path(reference).resolve()
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    return pd.read_table(source)


def _matches(query_mz: float, reference_mz: np.ndarray, ppm: float) -> np.ndarray:
    return np.abs(reference_mz - query_mz) / query_mz * 1_000_000.0 <= ppm


def annotate_hmdb(
    adata: anndata.AnnData,
    reference: pd.DataFrame | str | Path,
    *,
    mode: str,
    ppm: float = 5.0,
) -> anndata.AnnData:
    table = _load_table(reference)
    required = {"accession", "name", "mz", "mode"}
    missing = required - set(table.columns)
    if missing:
        raise InputFormatError(f"HMDB reference is missing columns: {sorted(missing)}")
    table = table[table["mode"].astype(str).str.lower() == mode.lower()].copy()
    reference_mz = table["mz"].astype(float).to_numpy()
    result = adata.copy()
    accessions: list[list[str]] = []
    names: list[list[str]] = []
    for query in result.var["mz"].astype(float):
        selected = table.loc[_matches(query, reference_mz, ppm)]
        accessions.append(selected["accession"].astype(str).tolist())
        names.append(selected["name"].astype(str).tolist())
    result.var["hmdb_accessions"] = [json.dumps(values) for values in accessions]
    result.var["hmdb_names"] = [json.dumps(values) for values in names]
    return result


def annotate_metaboscape(
    adata: anndata.AnnData,
    reference: pd.DataFrame | str | Path,
    *,
    ppm: float = 3.0,
) -> anndata.AnnData:
    table = _load_table(reference)
    required = {"Measured m/z", "Molecular Formula", "Name"}
    missing = required - set(table.columns)
    if missing:
        raise InputFormatError(f"MetaboScape reference is missing columns: {sorted(missing)}")
    reference_mz = table["Measured m/z"].astype(float).to_numpy()
    result = adata.copy()
    formulas: list[list[str]] = []
    names: list[list[str]] = []
    for query in result.var["mz"].astype(float):
        selected = table.loc[_matches(query, reference_mz, ppm)]
        formulas.append(selected["Molecular Formula"].astype(str).tolist())
        names.append(selected["Name"].astype(str).tolist())
    result.var["metaboscape_formulas"] = [json.dumps(values) for values in formulas]
    result.var["metaboscape_names"] = [json.dumps(values) for values in names]
    return result
```

- [ ] **Step 4: Run annotation tests**

Run: `python -m pytest tests/test_annotation.py -v`  
Expected: 2 tests pass.

- [ ] **Step 5: Commit annotation support**

```bash
git add src/joint/annotation.py tests/test_annotation.py
git commit -m "feat: add metabolite annotation adapters"
```

### Task 3: Clustering and Differential Ranking

**Files:**
- Create: `src/joint/analysis.py`
- Create: `tests/test_analysis.py`

**Interfaces:**
- Consumes: cell AnnData.
- Produces: `cluster_cells` and `rank_metabolites`.

- [ ] **Step 1: Write failing Scanpy analysis tests**

```python
# tests/test_analysis.py
import anndata
import numpy as np

from joint.analysis import cluster_cells, rank_metabolites


def analysis_adata():
    rng = np.random.default_rng(7)
    matrix = np.vstack([rng.normal(1, 0.1, (10, 5)), rng.normal(5, 0.1, (10, 5))])
    adata = anndata.AnnData(matrix)
    adata.var_names = [f"m{i}" for i in range(5)]
    adata.obs["known_group"] = ["a"] * 10 + ["b"] * 10
    return adata


def test_cluster_cells_returns_copy_with_embedding_and_cluster():
    source = analysis_adata()
    clustered = cluster_cells(source, n_neighbors=5, resolution=0.5, random_seed=7)
    assert "cluster" not in source.obs
    assert "cluster" in clustered.obs
    assert "X_pca" in clustered.obsm


def test_rank_metabolites_returns_tidy_table():
    ranked = rank_metabolites(analysis_adata(), groupby="known_group", method="wilcoxon")
    assert set(ranked.columns) >= {"group", "names", "scores", "pvals_adj"}
    assert set(ranked["group"]) == {"a", "b"}
```

- [ ] **Step 2: Run tests and verify analysis is absent**

Run: `python -m pytest tests/test_analysis.py -v`  
Expected: collection fails because `joint.analysis` is absent.

- [ ] **Step 3: Implement copy-safe clustering and tidy ranking**

```python
# src/joint/analysis.py
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from joint.errors import OptionalDependencyError


def cluster_cells(
    adata: anndata.AnnData,
    *,
    layer: str | None = None,
    n_top_features: int | None = None,
    n_neighbors: int = 15,
    resolution: float = 0.6,
    random_seed: int = 0,
    key_added: str = "cluster",
) -> anndata.AnnData:
    result = adata.copy()
    if n_top_features is not None and n_top_features < result.n_vars:
        sc.pp.highly_variable_genes(result, n_top_genes=n_top_features, flavor="seurat")
        use_highly_variable = True
    else:
        use_highly_variable = False
    sc.tl.pca(result, use_highly_variable=use_highly_variable, layer=layer, random_state=random_seed)
    sc.pp.neighbors(result, n_neighbors=n_neighbors, random_state=random_seed)
    sc.tl.leiden(result, resolution=resolution, key_added=key_added, random_state=random_seed)
    result.uns.setdefault("joint", {})["clustering"] = {
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "random_seed": random_seed,
    }
    return result


def rank_metabolites(
    adata: anndata.AnnData,
    *,
    groupby: str,
    method: str = "wilcoxon",
    layer: str | None = None,
) -> pd.DataFrame:
    result = adata.copy()
    sc.tl.rank_genes_groups(result, groupby=groupby, method=method, layer=layer)
    return sc.get.rank_genes_groups_df(result, group=None).rename(columns={"group": "group"})
```

- [ ] **Step 4: Run analysis tests**

Run: `python -m pytest tests/test_analysis.py -v`  
Expected: 2 tests pass with the declared `igraph` and `leidenalg` dependencies installed.

- [ ] **Step 5: Commit clustering and ranking**

```bash
git add src/joint/analysis.py tests/test_analysis.py pyproject.toml
git commit -m "feat: add cell clustering and metabolite ranking"
```

### Task 4: Lazy COSG Integration

**Files:**
- Modify: `src/joint/analysis.py`
- Modify: `tests/test_analysis.py`

**Interfaces:**
- Consumes: AnnData and COSG grouping parameters.
- Produces: `run_cosg(adata, *, groupby, n_genes, **kwargs) -> DataFrame`.

- [ ] **Step 1: Add failing COSG tests**

```python
# append to tests/test_analysis.py
import sys
import types
import pytest

from joint.analysis import run_cosg
from joint.errors import OptionalDependencyError


def test_run_cosg_has_precise_missing_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cosg", None)
    with pytest.raises(OptionalDependencyError, match="joint-msi\[cosg\]"):
        run_cosg(analysis_adata(), groupby="known_group", n_genes=3)


def test_run_cosg_returns_tidy_names(monkeypatch):
    module = types.SimpleNamespace()

    def fake_cosg(adata, key_added, groupby, n_genes_user, **kwargs):
        adata.uns[key_added] = {
            "names": np.array([("m0", "m1"), ("m2", "m3")], dtype=[("a", "O"), ("b", "O")])
        }

    module.cosg = fake_cosg
    monkeypatch.setitem(sys.modules, "cosg", module)
    result = run_cosg(analysis_adata(), groupby="known_group", n_genes=2)
    assert result.to_dict("records") == [
        {"rank": 1, "a": "m0", "b": "m1"},
        {"rank": 2, "a": "m2", "b": "m3"},
    ]
```

- [ ] **Step 2: Run COSG tests and verify the function is absent**

Run: `python -m pytest tests/test_analysis.py -k cosg -v`  
Expected: import fails for `run_cosg`.

- [ ] **Step 3: Implement lazy import and tidy output**

```python
# append to src/joint/analysis.py
def run_cosg(
    adata: anndata.AnnData,
    *,
    groupby: str,
    n_genes: int = 20,
    key_added: str = "cosg",
    **kwargs: Any,
) -> pd.DataFrame:
    try:
        import cosg
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalDependencyError(
            "COSG support requires `python -m pip install 'joint-msi[cosg]'`"
        ) from exc
    working = adata.copy()
    cosg.cosg(working, key_added=key_added, groupby=groupby, n_genes_user=n_genes, **kwargs)
    names = working.uns[key_added]["names"]
    table = pd.DataFrame({group: names[group] for group in names.dtype.names})
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table
```

- [ ] **Step 4: Run COSG tests**

Run: `python -m pytest tests/test_analysis.py -k cosg -v`  
Expected: 2 COSG tests pass without installing COSG.

- [ ] **Step 5: Commit COSG adapter**

```bash
git add src/joint/analysis.py tests/test_analysis.py
git commit -m "feat: add optional COSG analysis"
```

### Task 5: Lazy cNMF Workflow

**Files:**
- Modify: `src/joint/analysis.py`
- Modify: `tests/test_analysis.py`

**Interfaces:**
- Consumes: AnnData, output directory, component range, seed, worker parameters.
- Produces: `prepare_cnmf`, `run_cnmf`, and `load_cnmf_results`.

- [ ] **Step 1: Add failing cNMF tests with a fake backend**

```python
# append to tests/test_analysis.py
from pathlib import Path

from joint.analysis import load_cnmf_results, prepare_cnmf, run_cnmf


class FakeCnmf:
    calls = []

    def __init__(self, output_dir, name):
        self.output_dir = output_dir
        self.name = name

    def prepare(self, **kwargs):
        self.calls.append(("prepare", kwargs))

    def factorize(self, **kwargs):
        self.calls.append(("factorize", kwargs))

    def consensus(self, **kwargs):
        self.calls.append(("consensus", kwargs))


def test_prepare_and_run_cnmf_call_documented_backend_methods(monkeypatch, tmp_path: Path):
    module = types.SimpleNamespace(cNMF=FakeCnmf)
    monkeypatch.setitem(sys.modules, "cnmf", module)
    prepared = prepare_cnmf(
        analysis_adata(), tmp_path, name="sample", components=[5, 6], seed=14, num_highvar_genes=3
    )
    run_cnmf(prepared, worker_index=0, total_workers=1, selected_k=6)
    assert [name for name, _ in FakeCnmf.calls[-3:]] == ["prepare", "factorize", "consensus"]


def test_load_cnmf_results_joins_usage_by_obs_name(tmp_path: Path):
    adata = analysis_adata()
    usage = pd.DataFrame({"Usage_1": np.arange(adata.n_obs)}, index=adata.obs_names)
    path = tmp_path / "usage.tsv"
    usage.to_csv(path, sep="\t")
    loaded = load_cnmf_results(adata, path)
    assert loaded.obs["Usage_1"].tolist() == list(range(adata.n_obs))
```

- [ ] **Step 2: Run cNMF tests and verify functions are absent**

Run: `python -m pytest tests/test_analysis.py -k cnmf -v`  
Expected: imports fail for the cNMF APIs.

- [ ] **Step 3: Implement the lazy cNMF adapter and stable prepared object**

```python
# append to src/joint/analysis.py
from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedCnmf:
    backend: Any
    counts_path: Path
    output_dir: Path
    name: str


def _cnmf_class():
    try:
        from cnmf import cNMF
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalDependencyError(
            "cNMF support requires `python -m pip install 'joint-msi[cnmf]'`"
        ) from exc
    return cNMF


def prepare_cnmf(
    adata: anndata.AnnData,
    output_dir: str | Path,
    *,
    name: str,
    components: list[int],
    seed: int,
    num_highvar_genes: int,
) -> PreparedCnmf:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    counts_path = destination / f"{name}.h5ad"
    adata.write_h5ad(counts_path)
    backend = _cnmf_class()(output_dir=str(destination), name=name)
    backend.prepare(
        counts_fn=str(counts_path),
        components=np.asarray(components),
        seed=seed,
        num_highvar_genes=num_highvar_genes,
    )
    return PreparedCnmf(backend, counts_path, destination, name)


def run_cnmf(
    prepared: PreparedCnmf,
    *,
    worker_index: int,
    total_workers: int,
    selected_k: int,
    density_threshold: float = 0.5,
) -> PreparedCnmf:
    prepared.backend.factorize(worker_i=worker_index, total_workers=total_workers)
    prepared.backend.consensus(k=selected_k, density_threshold=density_threshold)
    return prepared


def load_cnmf_results(adata: anndata.AnnData, usage_path: str | Path) -> anndata.AnnData:
    usage = pd.read_csv(usage_path, sep="\t", index_col=0)
    missing = set(adata.obs_names) - set(usage.index)
    if missing:
        raise ValueError(f"cNMF usage is missing observations: {sorted(missing)[:5]}")
    result = adata.copy()
    result.obs = result.obs.join(usage.loc[result.obs_names])
    result.uns.setdefault("joint", {})["cnmf_usage_path"] = str(Path(usage_path).resolve())
    return result
```

- [ ] **Step 4: Run cNMF tests**

Run: `python -m pytest tests/test_analysis.py -k cnmf -v`  
Expected: 2 cNMF tests pass without installing cNMF.

- [ ] **Step 5: Commit cNMF support**

```bash
git add src/joint/analysis.py tests/test_analysis.py
git commit -m "feat: add optional cNMF workflow"
```

### Task 6: Spatial Trajectory Projection and Optional Trend Fitting

**Files:**
- Create: `src/joint/trajectory.py`
- Create: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: AnnData spatial coordinates and a polyline path.
- Produces: `fit_spatial_trajectory` and `calculate_feature_trends`.

- [ ] **Step 1: Write failing trajectory tests**

```python
# tests/test_trajectory.py
import sys

import anndata
import numpy as np
import pytest

from joint.errors import OptionalDependencyError
from joint.trajectory import calculate_feature_trends, fit_spatial_trajectory


def trajectory_adata():
    return anndata.AnnData(
        np.array([[0.0], [1.0], [2.0]]),
        obsm={"spatial": np.array([[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]])},
    )


def test_fit_spatial_trajectory_projects_points_to_path_order():
    result = fit_spatial_trajectory(trajectory_adata(), np.array([[0.0, 0.0], [10.0, 0.0]]))
    assert result.obs["trajectory_position"].tolist() == [0.0, 0.5, 1.0]
    assert result.obs["trajectory_distance"].tolist() == [0.0, 1.0, 0.0]


def test_calculate_feature_trends_has_precise_missing_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygam", None)
    projected = fit_spatial_trajectory(trajectory_adata(), np.array([[0.0, 0.0], [10.0, 0.0]]))
    with pytest.raises(OptionalDependencyError, match="joint-msi\[trajectory\]"):
        calculate_feature_trends(projected, [projected.var_names[0]], points=10)
```

- [ ] **Step 2: Run tests and verify trajectory is absent**

Run: `python -m pytest tests/test_trajectory.py -v`  
Expected: collection fails because `joint.trajectory` is absent.

- [ ] **Step 3: Implement polyline projection**

```python
# src/joint/trajectory.py
import anndata
import numpy as np
import pandas as pd
from scipy import sparse

from joint.errors import OptionalDependencyError


def fit_spatial_trajectory(adata: anndata.AnnData, path: np.ndarray) -> anndata.AnnData:
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path must have shape (n_points, 2) with at least two points")
    segments = points[1:] - points[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths == 0):
        raise ValueError("path contains a zero-length segment")
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    positions: list[float] = []
    distances: list[float] = []
    for point in spatial:
        best_distance = np.inf
        best_position = 0.0
        for index, (start, vector, length) in enumerate(zip(points[:-1], segments, lengths, strict=True)):
            fraction = np.clip(np.dot(point - start, vector) / (length**2), 0.0, 1.0)
            projection = start + fraction * vector
            distance = float(np.linalg.norm(point - projection))
            if distance < best_distance:
                best_distance = distance
                best_position = float((cumulative[index] + fraction * length) / cumulative[-1])
        positions.append(best_position)
        distances.append(best_distance)
    result = adata.copy()
    result.obs["trajectory_position"] = positions
    result.obs["trajectory_distance"] = distances
    result.uns.setdefault("joint", {})["trajectory_path"] = points.tolist()
    return result


def calculate_feature_trends(
    adata: anndata.AnnData,
    features: list[str],
    *,
    points: int = 100,
    layer: str | None = None,
) -> pd.DataFrame:
    try:
        from pygam import LinearGAM, s
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalDependencyError(
            "Trajectory trends require `python -m pip install 'joint-msi[trajectory]'`"
        ) from exc
    x = adata.obs["trajectory_position"].to_numpy()[:, None]
    grid = np.linspace(0.0, 1.0, points)[:, None]
    records: list[pd.DataFrame] = []
    for feature in features:
        matrix = adata[:, feature].layers[layer] if layer else adata[:, feature].X
        y = matrix.toarray().ravel() if sparse.issparse(matrix) else np.asarray(matrix).ravel()
        model = LinearGAM(s(0)).fit(x, y)
        prediction = model.predict(grid)
        intervals = model.prediction_intervals(grid, width=0.95)
        records.append(
            pd.DataFrame(
                {
                    "feature": feature,
                    "position": grid.ravel(),
                    "fitted": prediction,
                    "lower": intervals[:, 0],
                    "upper": intervals[:, 1],
                }
            )
        )
    return pd.concat(records, ignore_index=True)
```

- [ ] **Step 4: Run trajectory tests**

Run: `python -m pytest tests/test_trajectory.py -v`  
Expected: 2 tests pass without pyGAM installed.

- [ ] **Step 5: Commit trajectory support**

```bash
git add src/joint/trajectory.py tests/test_trajectory.py
git commit -m "feat: add spatial trajectory analysis"
```

### Task 7: Plotting APIs and Analysis Exports

**Files:**
- Create: `src/joint/pl.py`
- Create: `tests/test_pl.py`
- Modify: `src/joint/__init__.py`

**Interfaces:**
- Consumes: AnnData, label arrays, `RegistrationResult`, cNMF usage columns, trajectory tables.
- Produces: plotting functions returning `(Figure, Axes)` and curated analysis exports.

- [ ] **Step 1: Write plotting smoke tests**

```python
# tests/test_pl.py
import matplotlib
matplotlib.use("Agg")

import anndata
import numpy as np
import pandas as pd

from joint.models import RegistrationResult
from joint.pl import (
    plot_cell_contours,
    plot_cnmf_usage,
    plot_registration,
    plot_segmentation,
    plot_spatial_feature,
    plot_trajectory,
)


def plot_adata():
    adata = anndata.AnnData(
        np.array([[1.0], [2.0]]),
        obs={"Usage_1": [0.2, 0.8]},
        obsm={"spatial": np.array([[2.0, 3.0], [7.0, 8.0]])},
    )
    adata.var_names = ["m1"]
    adata.obs["cluster"] = ["a", "b"]
    return adata


def test_plotting_functions_return_figures_and_save_when_requested(tmp_path):
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    adata = plot_adata()
    registration = RegistrationResult(
        adata,
        pd.DataFrame({"centroid-1": [2.0, 7.0], "centroid-0": [3.0, 8.0]}),
        {"orientation": "identity"},
        {},
    )
    trends = pd.DataFrame(
        {"feature": ["m1", "m1"], "position": [0.0, 1.0], "fitted": [1.0, 2.0], "lower": [0.8, 1.8], "upper": [1.2, 2.2]}
    )
    calls = [
        plot_segmentation(labels),
        plot_cell_contours(labels),
        plot_spatial_feature(adata, "m1"),
        plot_spatial_feature(adata, "cluster"),
        plot_registration(registration),
        plot_cnmf_usage(adata, "Usage_1"),
        plot_trajectory(trends),
    ]
    assert all(figure is not None and axes is not None for figure, axes in calls)
    path = tmp_path / "segmentation.png"
    plot_segmentation(labels, save=path)
    assert path.is_file()
```

- [ ] **Step 2: Run tests and verify plotting is absent**

Run: `python -m pytest tests/test_pl.py -v`  
Expected: collection fails because `joint.pl` is absent.

- [ ] **Step 3: Implement shared plotting helpers and six public plots**

```python
# src/joint/pl.py
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from skimage.segmentation import find_boundaries

from joint.models import RegistrationResult


def _finish(fig, save: str | Path | None):
    if save is not None:
        destination = Path(save).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")


def plot_segmentation(labels: np.ndarray, *, save=None, ax=None):
    fig, axes = plt.subplots() if ax is None else (ax.figure, ax)
    axes.imshow(labels, cmap="nipy_spectral", interpolation="nearest")
    axes.set_axis_off()
    _finish(fig, save)
    return fig, axes


def plot_cell_contours(labels: np.ndarray, *, image=None, save=None, ax=None):
    fig, axes = plt.subplots() if ax is None else (ax.figure, ax)
    if image is not None:
        axes.imshow(image)
    axes.imshow(find_boundaries(labels, mode="outer"), cmap="gray", alpha=0.8)
    axes.set_axis_off()
    _finish(fig, save)
    return fig, axes


def plot_spatial_feature(adata: anndata.AnnData, feature: str, *, layer=None, save=None, ax=None):
    fig, axes = plt.subplots() if ax is None else (ax.figure, ax)
    if feature in adata.obs:
        series = adata.obs[feature]
        values = (
            series.to_numpy()
            if pd.api.types.is_numeric_dtype(series)
            else series.astype("category").cat.codes.to_numpy()
        )
    else:
        matrix = adata[:, feature].layers[layer] if layer else adata[:, feature].X
        values = matrix.toarray().ravel() if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    scatter = axes.scatter(adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1], c=values)
    fig.colorbar(scatter, ax=axes)
    axes.set_aspect("equal")
    axes.invert_yaxis()
    _finish(fig, save)
    return fig, axes


def plot_registration(result: RegistrationResult, *, save=None, ax=None):
    fig, axes = plt.subplots() if ax is None else (ax.figure, ax)
    mapping = result.mapping
    axes.scatter(mapping["centroid-1"], mapping["centroid-0"], s=12)
    axes.set_title(result.transform["orientation"])
    axes.set_aspect("equal")
    axes.invert_yaxis()
    _finish(fig, save)
    return fig, axes


def plot_cnmf_usage(adata: anndata.AnnData, usage: str, *, save=None, ax=None):
    fig, axes = plt.subplots() if ax is None else (ax.figure, ax)
    scatter = axes.scatter(
        adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1], c=adata.obs[usage]
    )
    fig.colorbar(scatter, ax=axes)
    axes.set_aspect("equal")
    axes.invert_yaxis()
    _finish(fig, save)
    return fig, axes


def plot_trajectory(trends: pd.DataFrame, *, save=None, ax=None):
    fig, axes = plt.subplots() if ax is None else (ax.figure, ax)
    for feature, table in trends.groupby("feature"):
        axes.plot(table["position"], table["fitted"], label=feature)
        axes.fill_between(table["position"], table["lower"], table["upper"], alpha=0.2)
    axes.legend()
    _finish(fig, save)
    return fig, axes
```

- [ ] **Step 4: Export stable analysis APIs without optional imports**

Add to `src/joint/__init__.py`:

```python
from joint.annotation import annotate_hmdb, annotate_metaboscape
from joint.qc import coverage_report, registration_report, spectral_qc
```

Add these names to `__all__`. Do not export `run_cosg`, cNMF, or pyGAM functions at package root because importing their modules must remain optional and explicit.

- [ ] **Step 5: Run the complete analysis suite**

Run: `python -m pytest tests/test_qc.py tests/test_annotation.py tests/test_analysis.py tests/test_trajectory.py tests/test_pl.py -v`  
Expected: all analysis and plotting tests pass.

Run: `ruff check src tests`  
Expected: `All checks passed!`.

- [ ] **Step 6: Commit the analysis milestone**

```bash
git add src/joint/pl.py src/joint/__init__.py tests/test_pl.py
git commit -m "feat: add JOINT visualization APIs"
```

## Plan 3 Completion Gate

Run:

```bash
python -m pip install -e '.[test]'
python -m pytest -v
ruff check src tests
python -c "import sys, joint; assert 'cosg' not in sys.modules and 'cnmf' not in sys.modules and 'pygam' not in sys.modules"
```

The plan is complete when core analysis and plots pass without optional packages, each optional adapter passes with a fake backend, multiple annotation matches are retained, and plotting tests run headlessly.
