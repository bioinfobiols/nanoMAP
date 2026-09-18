# Real d2 reference dataset

This is JOINT's only example and reference dataset. These are real experimental
inputs from the project owner's nanoMAP Figure5/5A-C d2 workflow, not generated data.
The five payloads total approximately 33 MB and are ordinary repository files:
Git LFS, external downloads, and private filesystem paths are not required.

| Input | Purpose |
| --- | --- |
| `adatas/d2.h5ad` | Preprocessed MSI input: 1,760 observations and 360 features |
| `pics/02/white.jpeg` | Original laser image |
| `segementation_d2/segmentation_cell_raw.npz` | Original cell labels, losslessly compressed |
| `rawdata/02.imzML` and `rawdata/02.ibd` | Optional raw d2 MSI preprocessing input |

`manifest.json` records sizes, SHA-256 checksums, original relative source paths,
and the expected analysis counts. The image, h5ad, and raw MSI pair are byte-for-byte
copies. The original 191 MB pickle label array had shape `(1, 5000, 5000, 1)`;
only its singleton axes were removed, then its unchanged int64 labels were saved
as a compressed NPZ array named `labels`. Loading does not execute pickle code.
The manifest includes both the original pickle hash and the uncompressed array
hash. The legacy `segementation_d2` spelling is retained for traceability.

From the JOINT source root, verify the files with:

```bash
python scripts/run_d2_demo.py --check-data
```

Use `configs/d2.yaml` for notebook-equivalent reproduction and
`configs/d2-raw.yaml` to exercise raw preprocessing. Raw preprocessing uses its own
feature alignment and normalization settings; its feature counts and intensities
are not asserted to equal the preprocessed notebook input.

The project owner supplied these data for this distribution. No separate data
license or formal citation metadata was supplied; this file does not assign
third-party rights or invent licensing terms.
