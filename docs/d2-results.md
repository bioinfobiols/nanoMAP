# d2 results guide

Run `python scripts/run_d2_demo.py` from the installed JOINT source checkout.
All analysis artifacts and images are written to `results/d2`; choose another
location with `--output-dir`. These generated files are not distributed as inputs.

## Analysis outputs

The seven completed stages are preprocessing, segmentation, registration,
quantification, qc, analysis, and final. `manifest.json` records the configuration,
source fingerprints, stage status, and hashes for the outputs. `logs/joint.log`
records the run. The final `joint-final.h5ad` has 2,936 cells and 360 features;
read it with `anndata.read_h5ad` or inspect the `AnnData` returned by the Python API.

The reference starts from 1,760 MSI observations and 1,707 laser regions before
the dataset's exclusions and merges. Registration retains 1,480 laser observations.
The segmentation label image is stored in `segmentation/laser_labels.npy`,
registered spectra in `registration/laser.h5ad`, and quantified cells in
`quantification/cells.h5ad`.

## Figure gallery

`figures/` contains the original laser image, cropped laser/cell segmentation,
registration overlay, quantified cell status, spatial cluster map, QC outliers,
UMAP, and two pipeline diagnostic plots. `figures/metabolites/` contains 360
feature maps and `contact_sheet_representative.png`. There are 371 PNGs in total.
`d2-figures-manifest.json` lists all paths and the feature-map count.

The d2 renderer uses the historical d2 image crop and produces descriptive
plots. Cluster identifiers and UMAP coordinates depend on the numerical stack
and should not be interpreted as cell identities without further validation.

## Reuse and regeneration

Running the demo again resumes stages only if their configuration, inputs, and
artifact signatures still match; the figure gallery is rendered again. Use
`--overwrite` to explicitly replace a previous run, or `--output-dir` to preserve
it and create another run. Do not copy a `manifest.json` from a different output
directory: it records paths and provenance for its own run.
