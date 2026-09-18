# JOINT d2-only distribution verification

Verified on 2026-09-17 with Python 3.12.13 on macOS 13.7.8 (x86_64).
This report describes the bundled d2 distribution, replacing the earlier
external-data and synthetic-fixture release workflow.

## Fresh installation

The source was copied into a new temporary directory without Git metadata,
virtual environments, caches, or results, as for a downloaded GitHub ZIP.
A new Python 3.12 virtual environment was created without system site packages.
The following installation resolved dependencies successfully:

```bash
python -m pip install -c requirements/constraints-py312.txt -e '.[test]'
python -m pip check
```

`pip check` reported `No broken requirements found.` Import inspection confirmed
that JOINT was loaded from the new source copy and Python from the new virtual
environment. Package downloads used the Python package index and available wheel
cache; no dependencies were inherited from the development environment.

## Complete real d2 run

With `JOINT_DATA_ROOT` and `JOINT_RUN_ROOT` unset:

```bash
python scripts/run_d2_demo.py
```

The command exited successfully from an initially empty output directory. All
seven stages completed: preprocessing, segmentation, registration, quantification,
qc, analysis, and final. The final AnnData shape is `(2936, 360)`.

Fresh checks confirmed:

- 1,760 raw MSI observations in the supplied preprocessed reference;
- 1,707 raw laser regions before exclusions and merges;
- 1,480 registered laser observations;
- 2,936 cells and 360 features;
- exactly 371 actual PNGs matching the figure manifest, with 360 feature maps.

The segmentation overlay was visually inspected. Data checksum verification
also passed when the runner was called by absolute path from an unrelated working
directory. No input path depended on the original author's desktop data tree.

The documented raw-input command also completed all seven stages:

```bash
joint run --config configs/d2-raw.yaml --resume
```

Its final shape was `(2936, 2411)`. That feature count reflects raw peak alignment
and normalization and is intentionally separate from the preprocessed 360-feature
reference. The imzML reader reported correction of the source metadata term
`pixel size x` to `pixel size (x)`; this did not prevent execution.

## Automated checks

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

The full suite in the fresh environment reported **1127 passed, 1 warning in
117.25s**, with **no skipped tests**. This includes complete d2 analysis, raw d2
imzML reading, lossless NPZ label verification, input checksum checks, and release
archive inspection. Ruff reported `All checks passed!`.

The warning is SciPy's `SparseEfficiencyWarning` about modifying a CSR matrix.
It is a performance advisory; the run completed and the numerical baselines
matched. A prior packaging assertion checking a literal configuration string was
replaced with tests for the actual Cardinal R script in both release archives.

## Notebook

The shipped quickstart was executed with `nbclient` using a standard separate-process
Jupyter kernel from the fresh virtual environment, with working directory `examples`.
All four original code cells completed (10.67 seconds), resumed the verified d2
outputs, and produced the spatial plot. An injected assertion confirmed the kernel
Python executable. The source notebook remained unchanged and output-free.
The local kernel emitted a transport advisory; execution exited successfully.

## Data and archives

The five real d2 payloads total **33,378,796 bytes**. Four are byte-identical copies
of the supplied source inputs. The cell-label conversion preserves all original
integer values and the int64 dtype, removes only singleton dimensions, and uses
NPZ compression. The resulting file is **1,796,694 bytes**, below GitHub's file
limit. All payload SHA-256 values and conversion provenance are in
`data/d2/manifest.json`; `.gitattributes` disables Git text conversion for them.

The source archive contains the data, example scripts, tests, README, notebook,
reference configurations, numerical dependency constraints, and CI configuration.
The wheel contains the library and d2 reference resources. Tests inspect both
formats and compare every source-archive d2 payload to its checksum manifest.
The isolated `python -m build --wheel --sdist` command also completed successfully.
Neither archive includes generated results, the old synthetic fixture, d8
configurations, development plans, virtual environments, or Python caches.

## Platform scope

The local evidence is for the macOS/Python 3.12 environment above. GitHub Actions
is configured for a fresh Ubuntu/Python 3.12 installation, full tests, and the
d2 figure run; no hosted GitHub execution is claimed before the repository is
uploaded. Native Windows is not supported because the pipeline uses POSIX file
APIs; Windows users need WSL2/Linux. Other Python versions, optional R/Cardinal,
COSG, cNMF, and trajectory integrations were not newly validated by this demo.
