# JOINT Implementation Roadmap

The approved JOINT design is implemented through four plans. Execute them in order because later plans consume APIs established by earlier plans.

1. [Foundation and MSI preprocessing](2026-08-31-joint-foundation-msi-plan.md)
   - Installable package, shared contracts, configuration, imzML/h5ad/segmentation I/O, native preprocessing, matrix-peak filtering, and Cardinal adapter.
2. [Spatial core](2026-08-31-joint-spatial-core-plan.md)
   - Laser segmentation/simulation, row corrections, registration, both cell-quantification strategies, and mixed AnnData construction.
3. [Analysis and visualization](2026-08-31-joint-analysis-plan.md)
   - QC, annotation, clustering, COSG, cNMF, trajectory analysis, and plotting.
4. [Pipeline and release](2026-08-31-joint-pipeline-release-plan.md)
   - Checkpointed pipeline, CLI, d2/d8 configurations and regression workflows, example notebook, documentation, and release verification.

Each plan produces working, testable software and ends with its own verification gate. Use the same Git repository and preserve task-level commits so a reviewer can approve or reject changes independently.
