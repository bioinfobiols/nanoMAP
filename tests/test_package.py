import hashlib
import json
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
    assert metadata["project"]["requires-python"] == ">=3.11,<3.14"


def test_test_extra_declares_build_tools_used_by_package_artifact_test():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    requirements = metadata["project"]["optional-dependencies"]["test"]
    for name in ("build", "hatchling"):
        assert any(requirement.startswith(f"{name}>=") for requirement in requirements)


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


def test_build_artifacts_ship_d2_source_demo_and_library_wheel(tmp_path: Path):
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
        wheel_members = set(archive.namelist())
    resources = {
        "configs/d2.yaml",
        "configs/d2-raw.yaml",
        "configs/d2-merges.csv",
        "docs/cardinal.md",
        "examples/joint_quickstart.ipynb",
    }
    assert {f"joint/resources/{resource}" for resource in resources} <= wheel_members
    assert "joint/__init__.py" in wheel_members
    assert "joint/backends/cardinal_pipeline.R" in wheel_members
    assert all(
        name.startswith(("joint/", "joint_msi-0.1.0.dist-info/"))
        for name in wheel_members
    )
    assert not any(
        {"data", "scripts", "tests"} & set(Path(name).parts)
        or Path(name).suffix.lower() in {".h5ad", ".imzml", ".ibd", ".npz", ".pkl"}
        or name.endswith("docs/api.md")
        for name in wheel_members
    )

    with tarfile.open(sdist) as archive:
        files = {
            Path(member.name).relative_to(Path(member.name).parts[0]).as_posix(): member
            for member in archive.getmembers()
            if member.isfile()
        }
        sdist_members = set(files)
        source_files = {
            "README.md",
            ".gitattributes",
            "pyproject.toml",
            "src/joint/__init__.py",
            "src/joint/backends/cardinal_pipeline.R",
            "scripts/__init__.py",
            "scripts/run_d2_demo.py",
            "scripts/render_d2_figures.py",
            "data/d2/manifest.json",
            "data/d2/README.md",
            "tests/test_package.py",
            "tests/regression/test_reference_data.py",
            "docs/api.md",
            "docs/d2-results.md",
            "docs/test-summary.md",
            ".github/workflows/tests.yml",
            "requirements/constraints-py312.txt",
        }
        assert resources | source_files <= sdist_members

        manifest = json.loads((root / "data/d2/manifest.json").read_text(encoding="utf-8"))
        for relative_path, expected in manifest["files"].items():
            name = f"data/d2/{relative_path}"
            assert name in sdist_members
            payload = archive.extractfile(files[name]).read()
            assert len(payload) == expected["bytes"]
            assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
            assert payload == (root / name).read_bytes()

    allowed_source_roots = {
        "src", "configs", "scripts", "data", "tests", "docs", "examples",
        ".github", ".gitignore", ".gitattributes", "requirements", "README.md", "pyproject.toml", "PKG-INFO",
    }
    assert all(Path(name).parts[0] in allowed_source_roots for name in sdist_members)
    assert {name for name in sdist_members if name.startswith("docs/")} == {
        "docs/api.md", "docs/cardinal.md", "docs/d2-results.md", "docs/test-summary.md",
    }
    for members in (wheel_members, sdist_members):
        assert not any(
            {"results", ".superpowers", "superpowers", ".venv", "__pycache__",
             ".pytest_cache", ".ruff_cache"} & set(Path(name).parts)
            or "synthetic" in name.lower()
            or any(part.startswith("d8") for part in Path(name).parts)
            or Path(name).suffix.lower() in {".pyc", ".pyo", ".pkl"}
            for name in members
        )
