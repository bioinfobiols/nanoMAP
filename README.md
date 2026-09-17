# JOINT

JOINT connects MSI spectra, laser-ablation marks, cell segmentations, and single-cell
metabolomics analysis.

## Installation

```bash
python -m pip install joint-msi
# For development or testing:
python -m pip install -e '.[test]'
python -m pip install -e '.[cosg,cnmf,trajectory]'
```

The `test` extra installs testing, linting, and notebook execution tools. The `cosg`, `cnmf`,
and `trajectory` extras install optional dependencies for differential metabolite analysis,
cNMF, and trajectory analysis. Install them only when the corresponding configuration is enabled.

The published wheel and source distribution include reference configurations, Cardinal
documentation, and the quickstart notebook. After installation, use `importlib.resources` to
locate them, copy the files into your working directory, and update their input paths:

```python
from importlib.resources import files

print(files("joint").joinpath("resources/configs/d8.yaml"))
print(files("joint").joinpath("resources/examples/joint_quickstart.ipynb"))
```

## Reference Data Configuration

```bash
export JOINT_DATA_ROOT=/path/to/nanoMAP_figures/Figure5/5A-C
```

`configs/d2.yaml` and `configs/d8.yaml` read existing `.h5ad` MSI data and reproduce the
notebook workflows. `configs/d2-raw.yaml` and `configs/d8-raw.yaml` start from raw imZML/ibd
data and run preprocessing again. The four configurations write results to their respective
`project.output_dir` values (by default, `results/d2`, `results/d8`, `results/d2-raw`, and
`results/d8-raw`). Each output directory contains `preprocessing`, `segmentation`,
`registration`, `quantification`, and any enabled downstream stages, plus `joint-final.h5ad` at
the root.

### Real d2 reference run

The d2 notebook data are located in `Figure5/5A-C` (the d2 assets are not in a `5B` directory).
Run the complete real-data workflow and render the notebook-equivalent review figures with:

```bash
export JOINT_DATA_ROOT=/Users/mac/Desktop/nanoMap/nanoMAP_figures/Figure5/5A-C
joint run --config configs/d2.yaml --overwrite
python scripts/render_d2_figures.py
```

The completed run produces 1,480 registered laser observations, 2,936 segmented cells, and
360 metabolite feature maps. The full artifact and figure inventory is documented in
[`results/d2/README.md`](results/d2/README.md), with a machine-readable figure manifest at
[`results/d2/d2-figures-manifest.json`](results/d2/d2-figures-manifest.json).

## Command Line

```bash
joint run --config configs/d8.yaml
joint quantify --config configs/d2.yaml --resume
```

`--resume` reuses completed stages whose input signatures still match, which is useful after an
interruption or when running a downstream command. `--overwrite` explicitly permits existing
stage artifacts to be regenerated; do not use it when existing results must be preserved.

## Synthetic demonstration data

The repository includes a reproducible 32×32 demonstration fixture with 1,024 MSI observations,
ten m/z features, a laser image, and four synthetic cell regions. It is intended for installation
checks and pipeline smoke tests, not as a scientific benchmark.
The fixture contains `synthetic_demo.h5ad` and the matching `synthetic_demo.imzML`/`synthetic_demo.ibd`
pair.

Regenerate the fixture when needed:

```bash
.venv/bin/python tests/data/synthetic_demo/generate_dataset.py \
  --output-dir tests/data/synthetic_demo \
  --seed 20260912 \
  --overwrite
```

Run the complete workflow from the preprocessed h5ad input:

```bash
joint run --config tests/data/synthetic_demo/configs/synthetic-h5ad.yaml \
  --overwrite
```

Run the same workflow from the raw imzML/ibd pair:

```bash
joint run --config tests/data/synthetic_demo/configs/synthetic-imzml.yaml \
  --overwrite
```

The configurations write to `results/synthetic-h5ad` and `results/synthetic-imzml` respectively.
Each run records `manifest.json`, `logs/joint.log`, and stage artifacts such as
`preprocessing/msi.h5ad`, `segmentation/laser_labels.npy`, `registration/laser.h5ad`, and
`quantification/cells.h5ad`. The same workflow is available from Python:

```python
from joint import JointPipeline

pipeline = JointPipeline.from_config("tests/data/synthetic_demo/configs/synthetic-h5ad.yaml")
result = pipeline.run(resume=True, overwrite=False)
```

## Quantification Methods

`configs/d8.yaml` uses the default `specificity_filtered` method. It keeps only unique or
dominant-overlap laser-to-cell mappings to reduce cross-contamination from mixed pixels.
`configs/d2.yaml` uses `legacy_proportional`, which distributes a laser point's signal across
all intersecting cells according to overlap area to reproduce the historical notebook results.
Select either method with `quantification.method`.

## Python API

```python
from joint import JointPipeline

pipeline = JointPipeline.from_config("configs/d8.yaml")
result = pipeline.run(resume=True)
```

`result` is the final `AnnData` object. To inspect intermediate artifacts, call `preprocess`,
`segment`, `register`, and `quantify` individually. See the [public API](docs/api.md) reference
for the complete function list and [Cardinal backend](docs/cardinal.md) for Cardinal configuration.
