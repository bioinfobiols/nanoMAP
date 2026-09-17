import hashlib
import json
from pathlib import Path

import anndata
import numpy as np
import pytest

from joint.pipeline import JointPipeline
from tests.data.synthetic_demo.generate_dataset import (
    build_synthetic_arrays,
    generate_dataset,
)

SYNTHETIC_ROOT = Path(__file__).parent / "data" / "synthetic_demo"


def test_build_synthetic_arrays_is_deterministic():
    first = build_synthetic_arrays(seed=20260912)
    second = build_synthetic_arrays(seed=20260912)
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_build_synthetic_arrays_has_medium_demo_contract():
    arrays = build_synthetic_arrays(seed=20260912)
    assert arrays["coordinates"].shape == (1024, 2)
    assert arrays["coordinates"].dtype == np.int32
    assert arrays["coordinates"].min() == 1
    assert arrays["coordinates"].max() == 32
    assert arrays["mz"].shape == (10,)
    assert np.all(np.diff(arrays["mz"]) > 0)
    assert arrays["intensities"].shape == (1024, 10)
    assert np.isfinite(arrays["intensities"]).all()
    assert (arrays["intensities"] >= 0).all()
    assert arrays["laser_image"].shape == (32, 32)
    assert arrays["laser_image"].dtype == np.uint8
    assert arrays["cell_labels"].shape == (32, 32)
    assert arrays["cell_labels"].dtype == np.int32
    assert set(np.unique(arrays["cell_labels"])) == {0, 1, 2, 3, 4}


def test_generate_dataset_rejects_nonempty_output_without_overwrite(tmp_path: Path):
    output = tmp_path / "synthetic_demo"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        generate_dataset(output)


def test_generate_dataset_writes_expected_files(tmp_path: Path):
    output = generate_dataset(tmp_path / "synthetic_demo")
    for name in (
        "synthetic_demo.h5ad",
        "synthetic_demo.imzML",
        "synthetic_demo.ibd",
        "laser_image.png",
        "cell_segmentation.npy",
        "manifest.json",
    ):
        assert (output / name).is_file(), name


def test_fixture_shapes_and_manifest():
    adata = anndata.read_h5ad(SYNTHETIC_ROOT / "synthetic_demo.h5ad")
    labels = np.load(SYNTHETIC_ROOT / "cell_segmentation.npy")
    manifest = json.loads((SYNTHETIC_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert adata.shape == (1024, 10)
    assert adata.obsm["spatial"].shape == (1024, 2)
    assert labels.shape == (32, 32)
    assert set(np.unique(labels)) == {0, 1, 2, 3, 4}
    assert manifest["seed"] == 20260912
    assert manifest["pixel_count"] == 1024
    assert manifest["feature_count"] == 10
    for name, digest in manifest["sha256"].items():
        actual = hashlib.sha256((SYNTHETIC_ROOT / name).read_bytes()).hexdigest()
        assert actual == digest


@pytest.mark.parametrize("config_name", ["synthetic-h5ad.yaml", "synthetic-imzml.yaml"])
def test_synthetic_pipeline_smoke(tmp_path: Path, config_name: str):
    pipeline = JointPipeline.from_config(SYNTHETIC_ROOT / "configs" / config_name)
    pipeline.config.project.output_dir = tmp_path / config_name.removesuffix(".yaml")
    pipeline.output_dir = pipeline.config.project.output_dir.resolve()

    result = pipeline.run(resume=False, overwrite=False)

    assert result.shape == (4, 10)
    output = pipeline.output_dir
    for relative in (
        "manifest.json",
        "logs/joint.log",
        "preprocessing/msi.h5ad",
        "segmentation/laser_labels.npy",
        "registration/laser.h5ad",
        "quantification/cells.h5ad",
    ):
        assert (output / relative).is_file(), relative


def test_readme_documents_synthetic_workflow():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "generate_dataset.py",
        "synthetic-h5ad.yaml",
        "synthetic-imzml.yaml",
        "synthetic_demo.imzML",
        "joint run",
        "JointPipeline.from_config",
        "--resume",
        "--overwrite",
        "JOINT_DATA_ROOT",
        "logs/joint.log",
    ):
        assert phrase in readme, phrase
