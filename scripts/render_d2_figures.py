"""Render reviewable d2 figures from completed JOINT pipeline artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import anndata
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from skimage import io
from skimage.segmentation import find_boundaries
from skimage.transform import resize

from joint.io import read_segmentation

CROP = (1250, 3750, 1250, 3750)  # row start, row stop, column start, column stop
ROOT = Path(__file__).resolve().parents[1]


class D2DemoError(RuntimeError):
    """The d2 renderer is missing required demo inputs or pipeline artifacts."""


def _data_root(value: str | Path | None) -> Path:
    candidate = value if value is not None else os.environ.get("JOINT_DATA_ROOT", ROOT / "data/d2")
    if candidate is None or not str(candidate).strip():
        raise D2DemoError(
            "Set JOINT_DATA_ROOT to a complete d2 data directory "
            "or pass data_root=..."
        )
    root = Path(candidate).expanduser().resolve()
    required = (
        Path("pics/02/white.jpeg"),
        Path("segementation_d2/segmentation_cell_raw.npz"),
    )
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise D2DemoError(f"d2 data root is missing {missing[0].as_posix()}: {root}")
    return root


def _run_root(value: str | Path | None) -> Path:
    candidate = value if value is not None else os.environ.get("JOINT_RUN_ROOT", ROOT / "results/d2")
    return Path(candidate).expanduser().resolve()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _crop_extent() -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = CROP
    return c0, c1, r1, r0


def _image_axes(ax: plt.Axes, image: np.ndarray, *, crop: bool = True) -> None:
    if crop:
        r0, r1, c0, c1 = CROP
        image = image[r0:r1, c0:c1]
        ax.imshow(image, cmap="gray", extent=_crop_extent())
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)
    else:
        ax.imshow(image, cmap="gray", extent=(0, image.shape[1], image.shape[0], 0))
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
    ax.set_aspect("equal")


def _scatter_spatial(ax: plt.Axes, adata: anndata.AnnData, **kwargs: object) -> None:
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    ax.scatter(spatial[:, 0], spatial[:, 1], **kwargs)
    ax.set_aspect("equal")
    ax.invert_yaxis()


def _feature_values(adata: anndata.AnnData, index: int) -> np.ndarray:
    matrix = adata.X
    values = matrix[:, index]
    if hasattr(values, "toarray"):
        values = values.toarray().ravel()
    return np.asarray(values, dtype=float).ravel()


def render(
    *,
    data_root: str | Path | None = None,
    run_root: str | Path | None = None,
) -> Path:
    data = _data_root(data_root)
    run = _run_root(run_root)
    required = (
        Path("segmentation/laser_labels.npy"),
        Path("registration/laser.h5ad"),
        Path("quantification/cells.h5ad"),
        Path("qc/cells_qc.h5ad"),
        Path("joint-final.h5ad"),
    )
    missing = [relative for relative in required if not (run / relative).is_file()]
    if missing:
        raise D2DemoError(f"d2 run root is missing {missing[0].as_posix()}: {run}")
    output = run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    image = io.imread(data / "pics/02/white.jpeg")
    gray = image.mean(axis=2) if image.ndim == 3 else image
    laser_labels = np.load(run / "segmentation/laser_labels.npy", allow_pickle=False)
    cell_labels = read_segmentation(
        data / "segementation_d2/segmentation_cell_raw.npz"
    )
    laser = anndata.read_h5ad(run / "registration/laser.h5ad")
    cells = anndata.read_h5ad(run / "quantification/cells.h5ad")
    final = anndata.read_h5ad(run / "joint-final.h5ad")
    qc = anndata.read_h5ad(run / "qc/cells_qc.h5ad")

    # Raw image and segmentation diagnostics.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(gray, cmap="gray")
    ax.set_title("d2 laser image")
    ax.set_axis_off()
    _save(fig, output / "01_raw_laser_image.png")

    r0, r1, c0, c1 = CROP
    laser_crop = laser_labels[r0:r1, c0:c1]
    active_label_ids = {
        int(part)
        for value in laser.obs["seg_label"].astype(str)
        for part in value.split("_")
        if part.isdigit()
    }
    active_laser_crop = np.where(np.isin(laser_crop, list(active_label_ids)), laser_crop, 0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(resize(gray[r0:r1, c0:c1], (1000, 1000), preserve_range=True), cmap="gray")
    axes[0].imshow(
        resize(find_boundaries(active_laser_crop, mode="outer"), (1000, 1000), order=0),
        cmap="autumn",
        alpha=0.8,
    )
    axes[0].set_title("Active laser boundaries on image")
    axes[0].set_axis_off()
    axes[1].imshow(active_laser_crop, cmap="nipy_spectral", interpolation="nearest")
    axes[1].set_title(f"Active laser observations: {laser.n_obs}")
    axes[1].set_axis_off()
    _save(fig, output / "02_laser_segmentation_crop.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    _image_axes(ax, gray)
    spatial = np.asarray(laser.obsm["spatial"], dtype=float)
    ax.scatter(spatial[:, 0], spatial[:, 1], s=5, facecolors="none", edgecolors="cyan", linewidths=0.4)
    ax.set_title("d2 registered laser points (1480 expected)")
    _save(fig, output / "03_registration_overlay.png")

    # Cell segmentation and quantified cell status.
    cell_crop = cell_labels[r0:r1, c0:c1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(resize(gray[r0:r1, c0:c1], (1000, 1000), preserve_range=True), cmap="gray")
    axes[0].imshow(
        resize(find_boundaries(cell_crop, mode="outer"), (1000, 1000), order=0),
        cmap="winter",
        alpha=0.75,
    )
    axes[0].set_title("Cell segmentation boundaries")
    axes[0].set_axis_off()
    axes[1].imshow(cell_crop, cmap="nipy_spectral", interpolation="nearest")
    axes[1].set_title(f"Cell labels: {cells.n_obs}")
    axes[1].set_axis_off()
    _save(fig, output / "04_cell_segmentation_crop.png")

    status = cells.obs["quantification_status"].astype(str).to_numpy()
    status_codes = pd.Categorical(status).codes
    fig, ax = plt.subplots(figsize=(8, 7))
    _scatter_spatial(ax, cells, c=status_codes, cmap="Set1", s=4, alpha=0.8)
    ax.set_title("Quantified cell status")
    _save(fig, output / "05_quantified_cell_status.png")

    if "cluster" in final.obs:
        fig, ax = plt.subplots(figsize=(8, 7))
        clusters = final.obs["cluster"].astype(str).astype("category").cat.codes.to_numpy()
        _scatter_spatial(ax, final, c=clusters, cmap="tab10", s=4, alpha=0.85)
        ax.set_title("JOINT d2 cell clusters")
        _save(fig, output / "06_cluster_map.png")

    if "outlier" in qc.obs:
        fig, ax = plt.subplots(figsize=(8, 7))
        _scatter_spatial(ax, qc, c=qc.obs["outlier"].astype(int), cmap="coolwarm", s=4)
        ax.set_title("QC outlier calls")
        _save(fig, output / "07_qc_outliers.png")

    # The d2 notebook also computes a two-dimensional UMAP after PCA/neighbors.
    # Reuse the stored neighbor graph so this figure is derived from the exact
    # analysis artifact written by JOINT, rather than from a re-segmented input.
    umap_rendered = False
    if "connectivities" in final.obsp and "distances" in final.obsp:
        umap_data = final.copy()
        if "X_umap" not in umap_data.obsm:
            sc.tl.umap(umap_data, random_state=14)
        fig, ax = plt.subplots(figsize=(8, 7))
        umap_coordinates = np.asarray(umap_data.obsm["X_umap"], dtype=float)
        clusters = umap_data.obs["cluster"].astype(str).astype("category").cat.codes.to_numpy()
        ax.scatter(
            umap_coordinates[:, 0],
            umap_coordinates[:, 1],
            c=clusters,
            cmap="tab10",
            s=4,
            alpha=0.85,
        )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title("JOINT d2 UMAP")
        _save(fig, output / "08_umap.png")
        umap_rendered = True

    # Notebook-equivalent positive-mean feature maps: one PNG per output feature.
    feature_dir = output / "metabolites"
    feature_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for index, feature in enumerate(final.var_names.astype(str)):
        values = _feature_values(final, index)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        vmin, vmax = np.percentile(finite, [1, 99])
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max()) + 1.0
        fig, ax = plt.subplots(figsize=(5, 4.5))
        _scatter_spatial(ax, final, c=values, cmap="viridis", s=4, vmin=vmin, vmax=vmax)
        ax.set_title(feature)
        fig.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04)
        _save(fig, feature_dir / f"{index + 1:03d}_{feature}.png")
        summary.append({"feature": feature, "index": index, "mean": float(np.mean(finite))})

    # A compact contact sheet for fast review of representative d2 metabolites.
    preferred = ["m12", "m170", "m459"]
    selected = [item for item in preferred if item in final.var_names]
    selected.extend([item["feature"] for item in summary if item["feature"] not in selected][:9])
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, feature in zip(axes.flat, selected, strict=False):
        index = int(final.var_names.get_loc(feature))
        values = _feature_values(final, index)
        _scatter_spatial(ax, final, c=values, cmap="viridis", s=2)
        ax.set_title(feature)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(selected) :]:
        ax.set_visible(False)
    _save(fig, feature_dir / "contact_sheet_representative.png")

    manifest = {
        "source": str(data),
        "pipeline_output": str(run),
        "laser_observations": int(laser.n_obs),
        "cell_observations": int(cells.n_obs),
        "feature_count": int(final.n_vars),
        "feature_maps": len(summary),
        "umap_rendered": umap_rendered,
        "representative_features": selected,
        "crop": {"row_start": r0, "row_stop": r1, "column_start": c0, "column_stop": c1},
        "files": sorted(str(path.relative_to(run)) for path in output.rglob("*.png")),
    }
    (run / "d2-figures-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    print(render())
