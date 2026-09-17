import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import joint
from joint.errors import (
    ConfigurationError,
    InputFormatError,
    JointError,
    OptionalDependencyError,
    PeakAlignmentError,
    QuantificationError,
    RegistrationError,
    SegmentationError,
)
from joint.models import SegmentationResult


def test_package_exports_version():
    assert joint.__version__ == "0.1.0"


def test_release_metadata_caps_numba_to_verified_python312_wheels():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    assert "numba>=0.60,<0.62" in dependencies


def test_test_extra_declares_build_tool_used_by_package_artifact_test():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert any(requirement.startswith("build") for requirement in metadata["project"]["optional-dependencies"]["test"])


def test_foundation_api_is_available_from_joint():
    assert callable(joint.load_config)
    assert callable(joint.preprocess_msi)
    assert callable(joint.read_h5ad)
    assert callable(joint.read_imzml)


def test_all_specific_errors_are_joint_errors():
    errors = [
        ConfigurationError,
        InputFormatError,
        OptionalDependencyError,
        PeakAlignmentError,
        QuantificationError,
        RegistrationError,
        SegmentationError,
    ]
    assert all(issubclass(error, JointError) for error in errors)


def test_segmentation_result_retains_diagnostics():
    result = SegmentationResult(labels=None, regions=None, diagnostics={"objects": 3})
    assert result.diagnostics["objects"] == 3


def test_build_artifacts_include_reference_configs_docs_and_quickstart(tmp_path: Path):
    root = Path(__file__).parents[1]
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    sdist = next(tmp_path.glob("*.tar.gz"))

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    wheel_resources = [
        "configs/d2.yaml",
        "configs/d2-raw.yaml",
        "configs/d8.yaml",
        "configs/d8-raw.yaml",
        "configs/d2-merges.csv",
        "docs/cardinal.md",
        "examples/joint_quickstart.ipynb",
    ]
    assert all(
        any(name.endswith(f"/{resource}") or name == resource for name in members)
        for resource in wheel_resources
    )

    with tarfile.open(sdist) as archive:
        members = {Path(member.name).relative_to(next(iter(archive.getnames())).split("/")[0]).as_posix()
                   for member in archive.getmembers() if "/" in member.name}
    assert all(resource in members for resource in wheel_resources)
