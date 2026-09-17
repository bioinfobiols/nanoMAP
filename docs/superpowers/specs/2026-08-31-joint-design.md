# JOINT Software Design

Date: 2026-08-31  
Status: approved design, implementation not started

## 1. Purpose

JOINT is a reusable Python package and command-line application for spatial MSI workflows that connect laser-ablation marks, MSI spectra, microscopy-derived cell segmentations, and downstream single-cell metabolomics analysis.

The design is derived from:

- `laserPoints-d2.ipynb`, which uses image-derived laser segmentation, manual region merging, MSI registration, and proportional cell quantification.
- `laserPoints-d8.ipynb`, which can simulate regular laser marks, applies filtered cell-to-laser mapping, and performs clustering, COSG, cNMF, and spatial visualization.
- The user's earlier SMAF modules, used as a reference for module responsibilities and public API style.

Notebook text, comments, and hard-coded commands are treated as source material rather than instructions. JOINT removes notebook global state, absolute paths, sample-specific constants, and silent in-place mutations.

## 2. Confirmed Product Decisions

- Product name: **JOINT**.
- Primary interface: typed Python function APIs.
- Secondary interface: the `joint` command-line program.
- Architecture: modular Python package with independent functions plus a lightweight `JointPipeline` facade.
- Usage target: both reusable analysis of new samples and reproducible d2/d8 workflows.
- Input scope: raw `.imzML/.ibd` data as well as prebuilt `.h5ad` data.
- MSI preprocessing backends:
  - Native Python backend by default.
  - Optional R/Cardinal backend.
- Cell quantification methods:
  - `specificity_filtered` is the default and preserves the intent of the d8 method.
  - `legacy_proportional` preserves the d2 proportional method.
- Downstream scope includes QC, annotation, clustering, COSG, cNMF, spatial trajectory analysis, and plotting.
- d2 and d8 receive dedicated, versioned configuration files and regression workflows.
- There is no REST/HTTP service in the initial release.

## 3. Scope and Non-goals

### 3.1 Included

1. Read centroid or profile imzML data and construct sparse AnnData objects.
2. Align or bin non-identical m/z axes, filter low-occurrence features, normalize spectra, and optionally remove matrix-associated peaks.
3. Segment laser marks from microscopy images or simulate regularly spaced marks from row anchors.
4. Detect rows, order marks, apply exclusion regions, and merge over-segmented marks through structured rules.
5. Register MSI pixels to laser marks while handling coordinate-system orientation explicitly.
6. Quantify metabolite abundance in segmented cells with either supported strategy.
7. Create laser-only, cell-only, and mixed AnnData outputs.
8. Run QC, HMDB/MetaboScape annotation, clustering, COSG, cNMF, differential ranking, and spatial trajectory analysis.
9. Plot laser segmentation, registration, spatial features, cell contours, clusters, cNMF usage, and trajectory trends.
10. Run individual stages or the complete pipeline from Python or the CLI.

### 3.2 Not included in the initial release

- A graphical desktop or web interface.
- An HTTP API.
- Training a general-purpose cell-segmentation model. JOINT accepts an existing labeled cell-segmentation array.
- Reimplementing the entire Cardinal package in Python.
- Automatic biological interpretation of COSG/cNMF results.
- Support for arbitrary proprietary MSI formats that cannot be converted to imzML.

## 4. Package Architecture

```text
JOINT/
├── src/joint/
│   ├── __init__.py
│   ├── io.py
│   ├── pp.py
│   ├── segmentation.py
│   ├── registration.py
│   ├── quantification.py
│   ├── qc.py
│   ├── annotation.py
│   ├── analysis.py
│   ├── trajectory.py
│   ├── pl.py
│   ├── config.py
│   ├── pipeline.py
│   ├── cli.py
│   └── backends/
│       ├── __init__.py
│       └── cardinal.py
├── configs/
│   ├── d2.yaml
│   ├── d2-merges.csv
│   └── d8.yaml
├── examples/
├── tests/
├── docs/
├── README.md
└── pyproject.toml
```

Each public module has a documented Python API. Public names are deliberately exported rather than exposing all implementation helpers.

### 4.1 Module responsibilities

| Module | Responsibility |
|---|---|
| `io` | Read and write imzML, h5ad, images, NumPy arrays, pickle-compatible legacy arrays, CSV tables, and stage artifacts. |
| `pp` | Peak alignment/binning, occurrence filtering, normalization, matrix-peak removal, and backend selection. |
| `segmentation` | Laser-image preprocessing, watershed labeling, region tables, row detection, merge rules, exclusions, and simulated laser marks. |
| `registration` | Coordinate orientation, MSI-to-laser matching, image mounting, and registration diagnostics. |
| `quantification` | Cell-laser overlap tables, filtering policies, abundance calculation, and mixed AnnData construction. |
| `qc` | Input validation, spectral QC, segmentation QC, registration metrics, coverage metrics, and outlier detection. |
| `annotation` | HMDB and MetaboScape annotation adapters with ppm-based matching. |
| `analysis` | Scanpy clustering, differential ranking, COSG, cNMF preparation/execution/import, and result attachment. |
| `trajectory` | User-defined spatial paths and feature-trend fitting. |
| `pl` | Plotting functions that return Matplotlib objects and save only when requested. |
| `config` | Typed YAML schema, defaults, validation, path resolution, and configuration snapshots. |
| `pipeline` | Stage orchestration, checkpointing, resume behavior, output manifests, and the `JointPipeline` facade. |
| `cli` | User-facing commands and nonzero error status behavior. |
| `backends.cardinal` | Optional subprocess bridge to a validated Cardinal workflow and import of its results. |

## 5. Data Model

### 5.1 AnnData conventions

- Observations are MSI pixels, laser marks, or cells depending on the stage.
- `adata.X` stores the active abundance matrix for that artifact.
- Raw or normalized alternatives are stored in named layers when both must be retained.
- `adata.obsm["spatial"]` stores coordinates as `(x, y)`.
- Image-array indices use `(row, column)`. Conversion between these systems is centralized in `registration`.
- `adata.var["mz"]` stores numeric m/z values.
- `adata.obs["dataset"]` stores the dataset identifier. A single-sample registration rejects mixed datasets unless a dataset is explicitly selected.
- Laser AnnData observations include `seg_label`, `area`, `centroid-0`, `centroid-1`, `row_number`, and `column_number`.
- Cell AnnData observations include cell label, area, centroid, mapped laser labels, coverage values, and quantification status.
- Provenance, parsed configuration, versions, coordinate transforms, and random seeds are recorded under `adata.uns["joint"]`.

### 5.2 Structured result objects

Functions with substantial diagnostics return dataclasses rather than loose tuples:

- `SegmentationResult(labels, regions, diagnostics)`
- `RegistrationResult(adata, mapping, transform, report)`
- `QuantificationResult(adata, overlaps, accepted, rejected, report)`
- `PipelineResult(artifacts, manifest, reports)`

These objects retain intermediate tables needed to audit a result. The primary AnnData remains directly accessible.

## 6. End-to-end Data Flow

```text
imzML/ibd or h5ad
  -> MSI loading
  -> peak alignment/binning
  -> occurrence filtering and normalization
  -> MSI AnnData
  -> real laser segmentation OR simulated laser marks
  -> row detection, ordering, exclusion, and merge rules
  -> MSI-to-laser registration
  -> laser AnnData
  -> cell-laser overlap extraction
  -> selected quantification method
  -> cell AnnData and optional mixed AnnData
  -> QC and annotation
  -> clustering, COSG, cNMF, ranking, trajectory
  -> plots, reports, and saved artifacts
```

All stages can run independently from saved inputs. The pipeline does not require a notebook kernel or hidden global state.

## 7. Public Python API

The exact optional parameters will be finalized in the implementation plan, but the stable API boundaries are:

```python
# joint.io
read_imzml(path, *, dataset_id=None) -> AnnData
read_h5ad(path) -> AnnData
read_segmentation(path) -> np.ndarray
write_results(result, output_dir) -> None

# joint.pp
align_peaks(adata, *, tolerance, unit="ppm") -> AnnData
filter_features(adata, *, min_occurrence=0.05) -> AnnData
normalize(adata, *, method="rms", layer="normalized") -> AnnData
remove_matrix_peaks(adata, *, method="denovo", **kwargs) -> AnnData
preprocess_msi(path, *, backend="python", **kwargs) -> AnnData

# joint.segmentation
segment_laser_marks(image, *, mask=None, **kwargs) -> SegmentationResult
detect_rows(regions, *, expected_rows=None, **kwargs) -> list[pd.DataFrame]
merge_regions(regions, merge_rules) -> pd.DataFrame
simulate_laser_marks(adata, anchor_rows, *, radius) -> SegmentationResult

# joint.registration
register_laser_points(msi_adata, laser_regions, *, orientation="auto") -> RegistrationResult
mount_spatial_images(adata, *, laser_image=None, cell_image=None) -> AnnData

# joint.quantification
quantify_cells(
    laser_adata,
    cell_segmentation,
    laser_segmentation,
    *,
    method="specificity_filtered",
    **kwargs,
) -> QuantificationResult
build_mixed_anndata(laser_adata, cell_adata, laser_segmentation, **kwargs) -> AnnData

# joint.qc
validate_inputs(**inputs) -> None
registration_report(result) -> pd.DataFrame
coverage_report(result) -> pd.DataFrame
detect_outliers(adata, **kwargs) -> pd.DataFrame

# joint.annotation
annotate_hmdb(adata, reference, *, mode, ppm=5) -> AnnData
annotate_metaboscape(adata, reference, *, ppm=3) -> AnnData

# joint.analysis
cluster_cells(adata, **kwargs) -> AnnData
run_cosg(adata, **kwargs) -> pd.DataFrame
prepare_cnmf(adata, output_dir, **kwargs) -> object
run_cnmf(prepared, **kwargs) -> object
load_cnmf_results(adata, result_dir, **kwargs) -> AnnData
rank_metabolites(adata, *, groupby, **kwargs) -> pd.DataFrame

# joint.trajectory
fit_spatial_trajectory(adata, path, **kwargs) -> object
calculate_feature_trends(trajectory, features, **kwargs) -> pd.DataFrame

# joint.pl
plot_segmentation(...)
plot_spatial_feature(...)
plot_cell_contours(...)
plot_registration(...)
plot_cnmf_usage(...)
plot_trajectory(...)
```

Public functions return new objects by default. Any API that supports mutation exposes an explicit `inplace=True` option and documents its return value.

## 8. MSI Preprocessing Backends

### 8.1 Native Python backend

The default backend uses `pyimzML`, NumPy, SciPy, pandas, sparse matrices, and AnnData.

- Continuous centroid data with a shared m/z axis, such as d2 and d8, can be loaded without unnecessary realignment.
- Processed centroid data with differing m/z axes is aligned using a configurable Da or ppm tolerance.
- Profile data is binned before matrix construction.
- Sparse construction avoids creating a dense pixels-by-features matrix when possible.
- Occurrence filtering is applied after alignment.
- Supported normalization initially includes RMS, TIC, and no normalization.
- The input matrix is preserved when normalization is written to a layer.

### 8.2 Cardinal backend

- Cardinal is an optional dependency and is not imported during core-package startup.
- JOINT invokes a versioned R script with explicit input and output paths rather than embedding interactive R commands.
- Cardinal failures are captured with stdout, stderr, and the exact command in the run report.
- Cardinal output is converted into the same AnnData conventions as the Python backend.

## 9. Laser Segmentation and Simulation

### 9.1 Real segmentation

The d2-derived workflow becomes a parameterized sequence:

1. Load grayscale image.
2. Apply an optional rectangular, polygonal, or externally supplied mask.
3. Apply intensity cutoff and white top-hat denoising.
4. Compute Otsu or user-specified threshold.
5. Apply configurable morphological closing/opening and border clearing.
6. Compute distance transform and local maxima.
7. Apply watershed segmentation.
8. Remove small objects.
9. Generate a region-property table.
10. Detect rows, apply coordinate exclusions, order marks, and apply merge rules.

Every hard-coded notebook value becomes a configuration field. Diagnostic images are generated without requiring interactive plotting.

### 9.2 Simulated marks

The d8-derived workflow uses detected row endpoints and the MSI grid dimensions to interpolate laser centers. Disk radius is configurable, disk coordinates are clipped to the image boundary, labels are integer-valued, and the returned region table follows the same schema as real segmentation.

### 9.3 Manual corrections

Manual corrections remain supported because some over-segmented images cannot be resolved reliably from morphology alone. Corrections are data, not code. They are stored as YAML or CSV records containing dataset, row, source labels, operation, and optional note. d2 receives a supplied correction table derived from the notebook.

## 10. Registration

Registration maps the ordered MSI grid to ordered laser marks while preserving both coordinate systems.

- Orientation options include explicit axis swap and horizontal/vertical reversal.
- `orientation="auto"` evaluates valid transforms using row count, point count, monotonicity, and spatial fit.
- If more than one orientation is equally plausible, the function raises `RegistrationError` and produces diagnostic candidates rather than silently choosing one.
- Registration returns a mapping table from MSI observation ID to laser segmentation label.
- Truncation or exclusion of unmatched edge points is explicit and recorded.
- Images mounted in `adata.uns["spatial"]` use consistent library IDs and scale factors.

## 11. Cell Quantification

### 11.1 Shared overlap table

Both methods first create a normalized table with:

- cell ID
- laser segmentation label
- occupied pixels
- total laser area
- occupied ratio
- laser AnnData observation ID

Zero-area labels, labels absent from laser AnnData, and cells outside the configured analysis boundary are handled before abundance calculation.

### 11.2 `specificity_filtered` default

This method preserves the d8 intent:

- Unique cell-laser mappings must exceed `unique_min_overlap`, default `0.10`.
- For lasers overlapping multiple cells, the dominant cell must exceed `dominant_min_overlap`, default `0.60`.
- The second-largest overlap must remain below `secondary_max_overlap`, default `0.15`.
- Accepted cell abundance uses overlap-area and sampling-specificity corrections.
- Rejected mappings and reasons are retained in the result.

The implementation explicitly handles empty unique or multi-mapping groups instead of calling `pd.concat` on an empty list.

### 11.3 `legacy_proportional`

This method preserves the d2 calculation: each laser spectrum is multiplied by the fraction of its segmented area covered by the cell, then contributions are summed. It is available for backward comparison and used in the d2 reproduction configuration.

## 12. Downstream Analysis

### 12.1 QC

QC includes spectral totals, detected-feature counts, occurrence distributions, segmentation object-size distributions, row and column counts, registration residuals, cell coverage, rejected mapping rates, and optional PCoA/silhouette summaries.

### 12.2 Annotation

- HMDB and MetaboScape importers validate required columns.
- Matching uses numeric m/z values and configurable ppm tolerances.
- Multiple matches are retained in structured columns rather than silently discarded.
- Annotation does not overwrite original feature IDs.

### 12.3 Clustering and COSG

Clustering follows explicit preprocessing choices recorded in configuration. COSG is optional and runs only when its extra dependency is installed. Results are attached under stable `adata.uns["joint"]` keys and returned as tidy tables.

### 12.4 cNMF

cNMF preparation, factorization, k-selection, consensus, and result loading are separate APIs. Component ranges, random seeds, workers, selected k, density threshold, and high-variable-feature count are configurable. The d8 configuration preserves the notebook's component range and selected `k=14` as reproduction defaults.

### 12.5 Trajectory

Trajectory analysis accepts a user-provided spatial path or ordered landmarks. It returns path positions and fitted feature trends without mutating unrelated AnnData fields.

## 13. Configuration

JOINT uses YAML validated by typed models. Paths are resolved relative to the configuration file.

Representative d8 configuration:

```yaml
project:
  name: d8
  output_dir: ../results/d8
  random_seed: 14

input:
  imzml: ../data/d8/rawdata/08.imzML
  laser_image: ../data/d8/pics/08/white.jpeg
  cell_segmentation: ../data/d8/segementation_d8/segmentation_cell_raw_08.pkl

preprocessing:
  backend: python
  normalization: rms
  min_occurrence: 0.05
  peak_tolerance: 10
  tolerance_unit: ppm

laser_segmentation:
  method: simulated
  radius: 18
  expected_rows: 41

registration:
  orientation: auto

quantification:
  method: specificity_filtered
  boundary_margin: 20
  unique_min_overlap: 0.10
  dominant_min_overlap: 0.60
  secondary_max_overlap: 0.15

analysis:
  cosg:
    enabled: true
  cnmf:
    enabled: true
    components: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    selected_k: 14
```

The d2 configuration uses real laser segmentation, a structured merge-rules file, and `legacy_proportional`.

## 14. CLI Design

```text
joint run --config CONFIG [--resume] [--overwrite]
joint preprocess --config CONFIG
joint segment --config CONFIG
joint register --config CONFIG
joint quantify --config CONFIG
joint qc --config CONFIG
joint annotate --config CONFIG
joint analyze --config CONFIG
joint trajectory --config CONFIG
```

- All commands support `--help`.
- Errors return nonzero status codes.
- `--resume` reuses validated completed stages.
- Existing final outputs are not overwritten without `--overwrite`.
- Commands print concise progress and write detailed logs to the output directory.

## 15. Output Layout and Reproducibility

```text
results/<project>/
├── manifest.json
├── resolved-config.yaml
├── logs/
├── preprocessing/
├── segmentation/
├── registration/
├── quantification/
├── qc/
├── annotation/
├── analysis/
├── trajectory/
└── figures/
```

The manifest records:

- JOINT and dependency versions
- Python and optional R/Cardinal versions
- random seeds
- resolved configuration
- input paths, sizes, modification times, and content digests when practical
- stage start/end times and status
- output artifact paths and summaries

Stage outputs are written to temporary files and atomically promoted after successful validation. A stage is reusable only when its recorded input/configuration signature matches the current run.

## 16. Error Handling

```text
JointError
├── ConfigurationError
├── InputFormatError
├── PeakAlignmentError
├── SegmentationError
├── RegistrationError
├── QuantificationError
└── OptionalDependencyError
```

Principles:

- Python APIs raise exceptions; library code never calls `exit()`.
- imzML and ibd pairing, XML readability, image dimensions, segmentation dtypes, required AnnData fields, and optional dependencies are checked before expensive work.
- Cell and laser segmentations must be two-dimensional integer label arrays with identical shapes.
- Ambiguous coordinate orientation is an error with diagnostics.
- Filtered or rejected records are reported rather than silently discarded.
- Optional-dependency errors include the exact install extra required.
- Logging goes to both the terminal and a run log.

## 17. Dependency Strategy

Core dependencies include NumPy, SciPy, pandas, scikit-image, matplotlib, AnnData, Scanpy-compatible sparse handling, pyimzML, PyYAML, and a typed configuration library.

Optional extras are isolated:

```text
joint[cosg]
joint[cnmf]
joint[cardinal]
joint[trajectory]
joint[all]
```

Importing `joint` must not import COSG, cNMF, R bridges, or trajectory-only dependencies.

## 18. Testing Strategy

### 18.1 Unit tests

- imzML metadata and coordinate parsing
- shared-axis and non-shared-axis peak handling
- TIC/RMS normalization including zero spectra
- occurrence filtering
- row-break detection
- row slicing and ordering
- merge-rule application
- disk simulation and boundary clipping
- orientation transforms
- overlap-table construction
- both abundance formulas
- empty, unique, and multi-mapped laser cases
- annotation ppm matching
- configuration validation and relative paths

### 18.2 Synthetic integration tests

Small deterministic arrays verify segmentation-to-registration-to-quantification behavior and exact expected metabolite intensities.

### 18.3 d2/d8 regression tests

Where source data is available, dedicated tests compare:

- laser-mark and row counts
- coordinate ranges and orientation
- registered pixel count
- included cell count
- accepted/rejected mapping counts
- representative metabolite values within documented numeric tolerance
- expected d8 cNMF configuration and result-loading shapes

Regression tests distinguish intended compatibility from notebook defects. Tests do not require reproducing failures caused by missing imports, undefined variables, invalid boolean checks, or hard-coded path mismatches in legacy code.

### 18.4 CLI tests

- install and `joint --help`
- each stage command
- full run on a small fixture
- resume and overwrite behavior
- nonzero status on invalid input
- optional-dependency messages

## 19. Documentation and Deliverables

The first release contains:

- installable Python package
- `joint` CLI
- d2 and d8 configurations
- d2 manual merge-rules file
- example notebook covering function-level and pipeline usage
- Chinese quick-start README
- English API docstrings
- dependency and Cardinal setup guidance
- unit, synthetic integration, regression, and CLI tests
- test summary

## 20. Acceptance Criteria

1. `pip install -e .` succeeds in the documented environment.
2. `import joint` and every documented public module API succeed.
3. `joint --help` and all stage commands behave as documented.
4. The native backend reads the d2 and d8 raw imzML/ibd pairs into valid AnnData objects.
5. d2 and d8 configurations run through laser and cell AnnData creation with the available source data.
6. `specificity_filtered` and `legacy_proportional` have independent numeric tests.
7. Core import and core tests do not require COSG, cNMF, Cardinal, or trajectory extras.
8. Optional features either run when installed or raise a precise `OptionalDependencyError`.
9. Core tests pass and a reproducible test summary is produced.
10. No public workflow depends on notebook execution order, absolute user paths, or hidden global variables.
