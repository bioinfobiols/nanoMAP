# JOINT public API

- `joint.io`: `read_imzml`, `read_h5ad`, `read_segmentation`, `atomic_write_h5ad`
- `joint.pp`: `filter_features`, `normalize`, `remove_matrix_peaks`, `preprocess_msi`
- `joint.segmentation`: `segment_laser_marks`, `detect_rows`, `exclude_regions`, `merge_regions`, `simulate_laser_marks`
- `joint.registration`: `register_laser_points`, `mount_spatial_images`
- `joint.quantification`: `build_overlap_table`, `quantify_cells`, `build_mixed_anndata`
- `joint.qc`: `validate_anndata`, `spectral_qc`, `registration_report`, `coverage_report`, `detect_outliers`, `pcoa`, `silhouette_summary`
- `joint.annotation`: `annotate_hmdb`, `annotate_metaboscape`
- `joint.analysis`: `cluster_cells`, `rank_metabolites`, `run_cosg`, `prepare_cnmf`, `run_cnmf`, `load_cnmf_results`
- `joint.trajectory`: `fit_spatial_trajectory`, `calculate_feature_trends`
- `joint.pl`: segmentation, registration, spatial-feature, cell-contour, cNMF-usage, and trajectory plots
- `joint.pipeline`: `JointPipeline`

`configs/d2.yaml` reproduces the notebook with `legacy_proportional`. `configs/d8.yaml`
reproduces the notebook with the default `specificity_filtered` method.
