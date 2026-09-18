# JOINT

JOINT connects mass spectrometry imaging (MSI), laser-ablation marks, cell
segmentations, and single-cell metabolomics analysis. It provides Python APIs
and a command-line pipeline for preprocessing, laser segmentation, registration,
cell quantification, quality control, clustering, and visualization.

**d2 is the only example and reference dataset.** The real data are included in
[`data/d2`](data/d2/README.md). No synthetic dataset generator, Git LFS checkout,
external data download, or machine-specific path is needed for the demo.

## Installation

Download this repository using GitHub **Code → Download ZIP**, extract it, and
open a terminal in the extracted folder containing `pyproject.toml` and this
README. A Git clone works as well. Use **Python 3.12** for the verified setup.
The Python dependency installation requires internet access; the d2 run itself
uses only local files. R, a GPU, and optional analysis packages are not required.

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements/constraints-py312.txt -e '.[test]'
python -m pip check
```

JOINT's artifact protection uses POSIX filesystem APIs. On Windows, use **WSL2
with Ubuntu**, place the extracted repository in the Linux filesystem, and run
the Linux commands above inside WSL. Native Windows Python is not supported.

The constraints file records the numerical
library versions used for verification. The `test` extra also installs build,
notebook, and testing tools. A minimal installation can replace `-e '.[test]'`
with `-e .`; it still supports the full d2 analysis and figure generation.

## Run the d2 demo

From the source root, after installation:

```bash
python scripts/run_d2_demo.py --check-data
python scripts/run_d2_demo.py
```

The first command checks the bundled data against their SHA-256 manifest. The
second runs the complete configured analysis, creates `results/d2`, and produces
371 PNG files including 360 metabolite feature maps. It validates the final
matrix shape and figure inventory. Allow several minutes for the first run;
Numba compilation and figure rendering contribute to the runtime. Use a machine
with at least 8 GB RAM and enough free disk space for Python dependencies and
generated results. See [verification notes](docs/test-summary.md) for measured
results and tested platforms.

No `JOINT_DATA_ROOT` environment variable is needed. Paths in
[`configs/d2.yaml`](configs/d2.yaml) resolve relative to that configuration file.
The runner also works when called by absolute path from another directory.

The default run safely resumes matching completed stages. For a new output
location or explicit regeneration:

```bash
python scripts/run_d2_demo.py --output-dir results/d2-new
python scripts/run_d2_demo.py --overwrite
```

`--overwrite` permits replacing existing analysis outputs. If results came from
an older JOINT configuration, choose a new output folder or explicitly overwrite.

Expected d2 reproduction counts:

| Measurement | Expected value |
| --- | ---: |
| Preprocessed MSI observations | 1,760 |
| Raw laser regions before exclusions and merges | 1,707 |
| Registered laser observations | 1,480 |
| Cells | 2,936 |
| Metabolite features | 360 |

## Results

All demo outputs are placed under `results/d2/` unless `--output-dir` is supplied:

| Path | Contents |
| --- | --- |
| `joint-final.h5ad` | Final 2,936 × 360 AnnData, with QC and clustering results |
| `manifest.json` | Stage status, configuration, input signatures, and artifact hashes |
| `logs/joint.log` | Pipeline progress and diagnostics |
| `segmentation/laser_labels.npy` | Laser segmentation label image |
| `registration/laser.h5ad` | Registered laser-level data |
| `quantification/cells.h5ad` | Cell-level quantification |
| `figures/02_laser_segmentation_crop.png` | Laser segmentation overview |
| `figures/03_registration_overlay.png` | Registration overlay |
| `figures/06_cluster_map.png` and `figures/08_umap.png` | Spatial clusters and UMAP |
| `figures/metabolites/` | 360 feature maps and a representative contact sheet |
| `d2-figures-manifest.json` | Figure filenames and summary counts |

Results are generated locally and excluded from the repository and release archives.

## Command line

Run individual stages or the full pipeline without the additional figure gallery:

```bash
joint --help
joint run --config configs/d2.yaml --resume
joint quantify --config configs/d2.yaml --resume
python scripts/render_d2_figures.py
```

The renderer requires completed d2 results and defaults to the bundled data and
`results/d2`. For optional raw MSI preprocessing using the bundled imzML/ibd pair:

```bash
joint run --config configs/d2-raw.yaml --resume
```

Raw preprocessing writes `results/d2-raw`. Its peak alignment and normalization
are separate from the preprocessed notebook reference, so its feature matrix is
not expected to be identical to the 360-feature d2 reproduction.

## Python API

Run analysis from the repository root:

```python
from joint import JointPipeline

pipeline = JointPipeline.from_config("configs/d2.yaml")
result = pipeline.run(resume=True, overwrite=False)
print(result.shape)  # (2936, 360)
```

Run the complete demo, including checksum validation and the figure gallery:

```python
from scripts.run_d2_demo import run_d2_demo, verify_d2_data

verify_d2_data()
output = run_d2_demo(output_dir="results/d2-api")
```

Render previously completed results:

```python
from scripts.render_d2_figures import render

figures = render(data_root="data/d2", run_root="results/d2-api")
```

The example scripts are source-checkout utilities. The installable wheel contains
the library and reference configurations; use the complete GitHub source ZIP or
source distribution for the bundled data, scripts, and tests.
See [public APIs](docs/api.md) for the module interfaces.

## Notebook

[`examples/joint_quickstart.ipynb`](examples/joint_quickstart.ipynb) runs the real
d2 pipeline and plots a spatial feature. Open it from the repository root or
`examples/`, selecting the same Python environment used above. If using an
external Jupyter application, register that environment first:

```bash
python -m ipykernel install --user --name joint --display-name "JOINT (Python 3.12)"
```

## Tests and continuous integration

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

The full test suite includes the actual d2 pipeline and bundled raw-data checks;
it does not need external datasets or environment variables. Unit tests also use
small in-memory arrays to check individual algorithms. There is no generated
simulation dataset or d8 reference workflow in this distribution.
For faster developer checks, run `python -m pytest -m 'not regression' -q`.

GitHub Actions installs a fresh Python 3.12 environment, runs the tests and lint,
and executes the d2 demo with figures. Local validation is documented separately
in [docs/test-summary.md](docs/test-summary.md); configuring CI does not imply
that a hosted GitHub run has already occurred.

## Analyze your own data

Copy the d2 YAML and set your MSI input (`h5ad` or `imzml`), laser image, and
cell segmentation paths. Cell labels can be a 2-D integer `.npy` array or an
`.npz` containing only an array named `labels`. Legacy pickle inputs require
explicit `trusted_pickle: true`; prefer the NumPy formats for shared data.
Use your own segmentation parameters, orientation, and merge rules. The d2 crop
and merge CSV are dataset-specific and should not be reused blindly.

`legacy_proportional` reproduces the d2 notebook's overlap-weighted quantification.
The library default `specificity_filtered` instead retains unique or dominant
laser-to-cell mappings. Both are available through `quantification.method`.
The low-level `simulate_laser_marks` API reconstructs geometric marks from actual
MSI coordinates and image anchors; it does not create a synthetic MSI dataset,
and the d2 demo uses measured laser segmentation.

Annotation, COSG, cNMF, and trajectory analysis are disabled for d2 because their
required references and choices are not defined by the original dataset. For
other data, optional packages can be installed with
`python -m pip install -e '.[cosg,cnmf,trajectory]'`. R/Cardinal is an optional
alternative preprocessing backend; see [Cardinal setup](docs/cardinal.md).

## Data provenance

The d2 inputs originate from the project owner's nanoMAP Figure5/5A-C workflow.
[The dataset README](data/d2/README.md) documents the files, checksum manifest,
and lossless conversion of the original large pickle segmentation. No software
or data license has been selected in this source tree; no third-party rights are
implied by its availability. The project owner can add the intended licenses
and citation details before public release.
