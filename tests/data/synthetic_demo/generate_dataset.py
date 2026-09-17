"""Generate the reproducible medium-sized JOINT demonstration fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from importlib import import_module
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from PIL import Image
from pyimzml.ImzMLWriter import ImzMLWriter
from scipy import sparse

from joint.io import atomic_write_h5ad

DEFAULT_SEED = 20260912
GRID_SHAPE = (32, 32)
FEATURE_COUNT = 10
GENERATED_FILES = (
    "synthetic_demo.h5ad",
    "synthetic_demo.imzML",
    "synthetic_demo.ibd",
    "laser_image.png",
    "cell_segmentation.npy",
    "manifest.json",
)


def build_synthetic_arrays(seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """Build deterministic coordinates, spectra, image, and cell labels."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(int(seed))

    rows, columns = np.indices(GRID_SHAPE, dtype=np.int32)
    coordinates = np.column_stack((columns.ravel() + 1, rows.ravel() + 1)).astype(np.int32)
    mz = (100.0 + 25.0 * np.arange(FEATURE_COUNT, dtype=np.float64)).astype(np.float64)

    x = coordinates[:, 0].astype(np.float32)
    y = coordinates[:, 1].astype(np.float32)
    spatial_gradient = (0.55 * x + 0.35 * y)[:, None]
    feature_scale = np.linspace(0.8, 1.6, FEATURE_COUNT, dtype=np.float32)[None, :]
    feature_profile = (1.0 + 0.25 * np.sin(np.arange(FEATURE_COUNT, dtype=np.float32)))
    noise = rng.normal(0.0, 0.15, size=(coordinates.shape[0], FEATURE_COUNT)).astype(np.float32)
    intensities = spatial_gradient * feature_scale * feature_profile[None, :] + 2.0 + noise
    intensities = np.clip(intensities, 0.0, None).astype(np.float32)

    cell_labels = np.zeros(GRID_SHAPE, dtype=np.int32)
    cell_labels[1:16, 1:16] = 1
    cell_labels[1:16, 16:31] = 2
    cell_labels[16:31, 1:16] = 3
    cell_labels[16:31, 16:31] = 4

    laser_image = np.full(GRID_SHAPE, 10, dtype=np.uint8)
    laser_image[::4, 1:31] = 235
    laser_image[:, ::8] = np.maximum(laser_image[:, ::8], 150)
    laser_image[1:31, 1:31] = np.maximum(laser_image[1:31, 1:31], 25)

    return {
        "coordinates": coordinates,
        "mz": mz,
        "intensities": intensities,
        "laser_image": laser_image,
        "cell_labels": cell_labels,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_h5ad(arrays: dict[str, np.ndarray], destination: Path, seed: int) -> None:
    coordinates = arrays["coordinates"]
    intensities = arrays["intensities"]
    obs = pd.DataFrame(
        {"dataset": np.full(coordinates.shape[0], "synthetic_demo", dtype=object)},
        index=[f"p{index + 1}" for index in range(coordinates.shape[0])],
    )
    var = pd.DataFrame(
        {"mz": arrays["mz"]},
        index=[f"m{index + 1}" for index in range(arrays["mz"].size)],
    )
    adata = anndata.AnnData(
        X=sparse.csr_matrix(intensities),
        obs=obs,
        var=var,
        obsm={"spatial": coordinates},
    )
    adata.uns["joint"] = {"dataset": "synthetic_demo", "seed": int(seed)}
    atomic_write_h5ad(adata, destination)


def _write_imzml(arrays: dict[str, np.ndarray], destination: Path, seed: int) -> None:
    writer_module = import_module("pyimzml.ImzMLWriter")
    original_uuid4 = writer_module.uuid.uuid4
    writer_module.uuid.uuid4 = lambda: uuid.uuid5(
        uuid.NAMESPACE_URL, f"joint-synthetic-demo:{int(seed)}"
    )
    try:
        with ImzMLWriter(
            str(destination),
            mz_dtype=np.float64,
            intensity_dtype=np.float32,
            spec_type="centroid",
        ) as writer:
            for coordinate, spectrum in zip(
                arrays["coordinates"], arrays["intensities"], strict=True
            ):
                writer.addSpectrum(
                    arrays["mz"],
                    spectrum,
                    (int(coordinate[0]), int(coordinate[1]), 1),
                )
    finally:
        writer_module.uuid.uuid4 = original_uuid4
    metadata = destination.read_text(encoding="ISO-8859-1")
    metadata = metadata.replace(str(destination.with_suffix("")), "synthetic_demo")
    destination.write_text(metadata, encoding="ISO-8859-1")


def _write_manifest(output_dir: Path, arrays: dict[str, np.ndarray], seed: int) -> None:
    files = {
        "synthetic_demo.h5ad": "preprocessed sparse AnnData matrix",
        "synthetic_demo.imzML": "centroid imzML metadata",
        "synthetic_demo.ibd": "centroid imzML binary spectra",
        "laser_image.png": "synthetic 8-bit grayscale laser image",
        "cell_segmentation.npy": "integer cell labels",
    }
    manifest = {
        "seed": int(seed),
        "shape": list(GRID_SHAPE),
        "pixel_count": int(arrays["coordinates"].shape[0]),
        "feature_count": int(arrays["mz"].size),
        "m/z": arrays["mz"].tolist(),
        "cell_labels": [int(value) for value in np.unique(arrays["cell_labels"])],
        "files": files,
        "sha256": {name: _sha256(output_dir / name) for name in files},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate_dataset(
    output_dir: str | Path, *, seed: int = DEFAULT_SEED, overwrite: bool = False
) -> Path:
    """Generate the fixture into ``output_dir`` and return its resolved path."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"Output path is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty; pass overwrite=True: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = build_synthetic_arrays(int(seed))

    temporary = Path(tempfile.mkdtemp(prefix=".synthetic-build-", dir=destination.parent))
    try:
        _write_h5ad(arrays, temporary / "synthetic_demo.h5ad", int(seed))
        _write_imzml(arrays, temporary / "synthetic_demo.imzML", int(seed))
        Image.fromarray(arrays["laser_image"], mode="L").save(temporary / "laser_image.png")
        with (temporary / "cell_segmentation.npy").open("wb") as handle:
            np.save(handle, arrays["cell_labels"], allow_pickle=False)
        _write_manifest(temporary, arrays, int(seed))
        destination.mkdir(parents=True, exist_ok=True)
        for name in GENERATED_FILES:
            os.replace(temporary / name, destination / name)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="destination fixture directory (default: this directory)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        generated = generate_dataset(args.output_dir, seed=args.seed, overwrite=args.overwrite)
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated JOINT synthetic dataset in {generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
