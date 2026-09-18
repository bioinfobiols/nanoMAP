# Cardinal backend

Install R and Cardinal:

```bash
R -q -e 'install.packages("BiocManager"); BiocManager::install("Cardinal")'
Rscript -e 'library(Cardinal); packageVersion("Cardinal")'
```

Set `preprocessing.backend: cardinal` in `configs/d2-raw.yaml` (which defaults to the Python
backend), then run:

```bash
joint preprocess --config configs/d2-raw.yaml
```

The same backend is available to new compatible raw-data configurations.

JOINT invokes `Rscript` with explicit input and output paths. Cardinal writes a processed
imZML/ibd pair, and JOINT imports that pair into AnnData using the same coordinate and feature
conventions as the native backend. On success, the `Rscript` command, stdout, and stderr are
recorded in `adata.uns["joint"]["cardinal"]`; they therefore travel with the resulting h5ad
artifact. If Cardinal fails, JOINT includes the command and captured stdout/stderr in the raised
error. Run-scoped logs are written to `logs/joint.log` under the configured output directory.
