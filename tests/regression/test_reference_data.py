"""End-to-end regression against the bundled real d2 dataset."""
import hashlib
import json
from pathlib import Path

import anndata
import numpy as np
import pytest

from joint.config import load_config
from joint.io import read_imzml, read_segmentation
from joint.pipeline import JointPipeline
from scripts.run_d2_demo import verify_d2_data

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.regression


def test_bundled_d2_pipeline(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("JOINT_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    data = verify_d2_data()
    config = load_config(ROOT / "configs/d2.yaml")
    config.project.output_dir = tmp_path / "d2"
    assert config.input.h5ad == data / "adatas/d2.h5ad"
    assert not config.input.trusted_pickle
    result = JointPipeline(config).run(resume=False, overwrite=False)
    run = config.project.output_dir
    assert anndata.read_h5ad(run / "preprocessing/msi.h5ad").shape == (1760, 360)
    labels = np.load(run / "segmentation/laser_labels.npy", allow_pickle=False)
    assert np.count_nonzero(np.unique(labels)) == 1707
    assert anndata.read_h5ad(run / "registration/laser.h5ad").n_obs == 1480
    assert result.shape == (2936, 360)
    assert anndata.read_h5ad(run / "joint-final.h5ad").shape == result.shape
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(manifest["stages"]) == 7
    assert all(stage["status"] == "complete" for stage in manifest["stages"].values())
    assert (run / "figures/laser-segmentation.png").is_file()
    assert (run / "figures/registration.png").is_file()


def test_bundled_cell_labels_are_lossless_integer_data():
    data = ROOT / "data/d2"
    name = "segementation_d2/segmentation_cell_raw.npz"
    labels = read_segmentation(data / name)
    record = json.loads((data / "manifest.json").read_text())["files"][name]
    assert labels.shape == (5000, 5000)
    assert labels.dtype == np.dtype("int64")
    assert hashlib.sha256(labels.tobytes(order="C")).hexdigest() == record["array_sha256"]


@pytest.mark.filterwarnings(
    r'ignore:Accession IMS\x3a1000046 found with incorrect name "pixel size x"\. '
    r'Updating name to "pixel size \(x\)"\.:UserWarning:pyimzml\.ontology\.ontology'
)
def test_bundled_d2_raw_imzml():
    adata = read_imzml(ROOT / "data/d2/rawdata/02.imzML")
    assert adata.n_obs == 1760
