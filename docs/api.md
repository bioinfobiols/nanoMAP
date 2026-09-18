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
- `scripts.run_d2_demo`: `verify_d2_data(data_root=None)`, `run_d2_demo(output_dir=None, overwrite=False)` (keyword-only run arguments; source checkout)
- `scripts.render_d2_figures`: `render(data_root=None, run_root=None)` (keyword-only arguments; source checkout)

`configs/d2.yaml` uses the bundled real d2 data with `legacy_proportional` to
reproduce the notebook. The library also supports `specificity_filtered` for
other compatible datasets. YAML paths resolve relative to the YAML file.

`read_segmentation` accepts `.npy`, or `.npz` containing exactly one 2-D integer
array named `labels` (loaded with `allow_pickle=False`). Legacy pickle requires
explicit `trusted_pickle=True`. No synthetic MSI dataset generator is provided.
The geometric `simulate_laser_marks` API uses actual MSI coordinates and image
anchors; it remains available for analytical reconstruction but is not used by d2.
