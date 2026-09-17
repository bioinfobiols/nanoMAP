# JOINT synthetic demonstration fixture

This directory contains a deterministic, medium-sized JOINT demonstration dataset. It is
synthetic test data for validating the file readers and pipeline wiring; it is not a scientific
benchmark and must not be used as experimental evidence.

## Regeneration

From the repository root, run:

```bash
.venv/bin/python tests/data/synthetic_demo/generate_dataset.py \
  --output-dir tests/data/synthetic_demo \
  --seed 20260912 \
  --overwrite
```

The generator uses seed `20260912`, produces a 32×32 grid (1,024 observations), and writes ten
strictly increasing m/z features from 100.0 to 325.0 Da. It refuses to replace a non-empty
directory unless `--overwrite` is provided.

## Files

| File | Description |
| --- | --- |
| `synthetic_demo.h5ad` | Sparse AnnData input with 1,024 observations and 10 features |
| `synthetic_demo.imzML` / `synthetic_demo.ibd` | Centroid imzML/ibd pair with the same spectra and coordinates |
| `laser_image.png` | Deterministic 32×32 grayscale laser image |
| `cell_segmentation.npy` | 32×32 integer labels: background `0` and cells `1`–`4` |
| `manifest.json` | Dataset dimensions, seed, file descriptions, and SHA-256 hashes |
| `configs/synthetic-h5ad.yaml` | Pipeline configuration starting from the h5ad input |
| `configs/synthetic-imzml.yaml` | Pipeline configuration starting from the imzML input |

The two configurations disable optional annotation, QC, COSG, cNMF, and trajectory stages so
they can be used as a compact end-to-end smoke test with the base JOINT installation.
