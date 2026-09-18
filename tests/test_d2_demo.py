import json
from pathlib import Path

import pytest

from scripts.render_d2_figures import D2DemoError, render
from scripts.run_d2_demo import verify_d2_data

ROOT = Path(__file__).parents[1]


def test_readme_presents_d2_as_primary_demo():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'JointPipeline.from_config("configs/d2.yaml")' in text
    assert "python scripts/run_d2_demo.py" in text
    assert "data/d2" in text
    assert "generate_dataset" not in text


def test_quickstart_uses_d2_reference_data():
    notebook = json.loads((ROOT / "examples/joint_quickstart.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "data/d2" in source
    assert "configs/d2.yaml" in source
    assert "result = pipeline.run(resume=True, overwrite=False)" in source
    assert "plot_spatial_feature(result, result.var_names[0])" in source
    assert "Figure5/5B" not in source
    assert "../configs/d8.yaml" not in source


def test_quickstart_has_four_clean_code_cells():
    notebook = json.loads((ROOT / "examples/joint_quickstart.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 4
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)
    assert all("execution" not in cell.get("metadata", {}) for cell in notebook["cells"])


def test_current_user_docs_do_not_reference_old_figure5_path():
    paths = (ROOT / "README.md", ROOT / "docs/cardinal.md", ROOT / "docs/test-summary.md")
    assert not [path for path in paths if "Figure5/5B" in path.read_text(encoding="utf-8")]


def test_verification_summary_distinguishes_d2_baseline_counts():
    text = (ROOT / "docs/test-summary.md").read_text(encoding="utf-8")
    assert "1,760 raw MSI observations" in text
    assert "1,707 raw laser regions" in text
    assert "1,480 registered laser observations" in text
    assert "2,936 cells" in text


def test_d2_renderer_defaults_to_bundled_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("JOINT_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(D2DemoError, match="segmentation/laser_labels.npy"):
        render(run_root=tmp_path)


def test_d2_renderer_rejects_incomplete_data_root(tmp_path: Path):
    with pytest.raises(D2DemoError, match="pics/02/white.jpeg"):
        render(data_root=tmp_path, run_root=tmp_path / "run")


@pytest.mark.parametrize("value", ["", "   "])
def test_d2_renderer_rejects_blank_data_root(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("JOINT_DATA_ROOT", value)
    with pytest.raises(D2DemoError, match="JOINT_DATA_ROOT"):
        render()


def test_d2_renderer_requires_cell_segmentation(tmp_path: Path):
    image = tmp_path / "pics/02/white.jpeg"
    image.parent.mkdir(parents=True)
    image.touch()
    with pytest.raises(D2DemoError, match="segementation_d2/segmentation_cell_raw.npz"):
        render(data_root=tmp_path, run_root=tmp_path / "run")


@pytest.mark.parametrize("explicit_roots", [False, True])
@pytest.mark.parametrize(
    "missing_artifact",
    [
        "segmentation/laser_labels.npy",
        "registration/laser.h5ad",
        "quantification/cells.h5ad",
        "qc/cells_qc.h5ad",
        "joint-final.h5ad",
    ],
)
def test_d2_renderer_validates_run_before_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    explicit_roots: bool,
    missing_artifact: str,
):
    data = tmp_path / "data"
    run = tmp_path / "run"
    for relative in ("pics/02/white.jpeg", "segementation_d2/segmentation_cell_raw.npz"):
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for relative in (
        "segmentation/laser_labels.npy",
        "registration/laser.h5ad",
        "quantification/cells.h5ad",
        "qc/cells_qc.h5ad",
        "joint-final.h5ad",
    ):
        if relative != missing_artifact:
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JOINT_DATA_ROOT", "missing-data" if explicit_roots else "data")
    monkeypatch.setenv("JOINT_RUN_ROOT", "missing-run" if explicit_roots else "run")
    kwargs = {"data_root": Path("data"), "run_root": "run"} if explicit_roots else {}
    with pytest.raises(D2DemoError, match=f"d2 run root is missing {missing_artifact}: {run}"):
        render(**kwargs)
    assert not (run / "figures").exists()


def test_d2_renderer_has_callable_api():
    assert callable(render)


def test_demo_inputs_are_separate_from_library_code():
    forbidden = {".h5ad", ".imzML", ".ibd", ".pkl"}
    packaged_roots = [ROOT / "src", ROOT / "configs", ROOT / "examples"]
    assert not [
        path
        for root in packaged_roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in forbidden
    ]


def test_verify_bundled_data_from_an_unrelated_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JOINT_DATA_ROOT", str(tmp_path / "nonexistent"))
    assert verify_d2_data() == ROOT / "data/d2"


def test_verify_d2_reports_missing_data(tmp_path):
    (tmp_path / "manifest.json").write_bytes((ROOT / "data/d2/manifest.json").read_bytes())
    with pytest.raises(D2DemoError, match="Missing d2 input.*adatas/d2.h5ad"):
        verify_d2_data(tmp_path)


def test_verify_d2_rejects_changed_payload(tmp_path):
    (tmp_path / "manifest.json").write_bytes((ROOT / "data/d2/manifest.json").read_bytes())
    (tmp_path / "adatas").mkdir()
    (tmp_path / "adatas/d2.h5ad").write_bytes(b"damaged download")
    with pytest.raises(D2DemoError, match="checksum mismatch"):
        verify_d2_data(tmp_path)
