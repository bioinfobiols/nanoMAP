import os
from pathlib import Path

import anndata
import pytest

from joint.config import load_config
from joint.io import read_imzml
from joint.pipeline import JointPipeline

REFERENCE = os.environ.get("JOINT_DATA_ROOT")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
pytestmark = pytest.mark.regression


def require_reference() -> Path:
    if REFERENCE is None or not Path(REFERENCE).is_dir():
        pytest.skip("Set JOINT_DATA_ROOT to the Figure5/5A-C directory")
    return Path(REFERENCE)


@pytest.mark.parametrize(
    ("config_name", "laser_count", "cell_count"),
    [("d2.yaml", 1480, 2936), ("d8.yaml", 1640, 4986)],
)
def test_notebook_reproduction_baselines(
    tmp_path: Path, config_name: str, laser_count: int, cell_count: int
):
    require_reference()
    config = load_config(CONFIG_DIR / config_name)
    config.project.output_dir = tmp_path / config.project.name
    pipeline = JointPipeline(config)

    cells = pipeline.run_spatial_core(overwrite=False, resume=False)
    laser = config.project.output_dir / "registration/laser.h5ad"

    assert anndata.read_h5ad(laser).n_obs == laser_count
    assert cells.adata.n_obs == cell_count


@pytest.mark.parametrize(("filename", "pixel_count"), [("02.imzML", 1760), ("08.imzML", 1640)])
@pytest.mark.filterwarnings(
    r'ignore:Accession IMS\x3a1000046 found with incorrect name "pixel size x"\. '
    r'Updating name to "pixel size \(x\)"\.:UserWarning:pyimzml\.ontology\.ontology'
)
def test_raw_imzml_pixel_counts(filename: str, pixel_count: int):
    reference = require_reference()
    adata = read_imzml(reference / "rawdata" / filename)
    assert adata.n_obs == pixel_count
