import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from joint.config import JointConfig
from joint.errors import JointError
from joint.logging import configure_logging
from joint.pipeline import JointPipeline, StageStore, build_run_metadata


def test_configure_logging_writes_terminal_messages_to_run_log(tmp_path: Path):
    logger = configure_logging(tmp_path)
    logger.info("stage message")
    for handler in logger.handlers:
        handler.flush()
    assert "stage message" in (tmp_path / "logs/joint.log").read_text()


def test_stage_store_records_failure_and_artifact_hashes(tmp_path: Path):
    store = StageStore(tmp_path, logger=configure_logging(tmp_path))
    store.initialize({"joint_version": "0.1.0", "random_seed": 14, "inputs": []})

    def failing_producer(staging_dir: Path):
        del staging_dir
        raise RuntimeError("boom")

    with pytest.raises(JointError, match="boom"):
        store.run(
            "broken",
            "signature",
            failing_producer,
            resume=False,
            overwrite=False,
        )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run"]["random_seed"] == 14
    assert manifest["stages"]["broken"]["status"] == "failed"

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("result")
        return {"result": artifact}

    store.run("working", "signature", producer, resume=False, overwrite=False)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["stages"]["working"]["artifact_signatures"]["result"]["sha256"]) == 64


def test_pipeline_initializes_run_metadata(tmp_path: Path):
    input_path = tmp_path / "input.h5ad"
    input_path.write_bytes(b"placeholder")
    config = JointConfig.model_validate(
        {
            "project": {"name": "sample", "output_dir": tmp_path / "results", "random_seed": 14},
            "input": {"h5ad": input_path},
        }
    )

    pipeline = JointPipeline(config)
    pipeline._get_store()

    manifest = json.loads((config.project.output_dir / "manifest.json").read_text())
    assert manifest["run"]["random_seed"] == 14
    assert manifest["run"]["inputs"][0]["sha256"]


def test_failed_stage_is_recorded_without_logger(tmp_path: Path):
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="boom"):
        store.run("broken", "sig", lambda staging: (_ for _ in ()).throw(RuntimeError("boom")), resume=False, overwrite=False)

    assert json.loads(store.manifest_path.read_text())["stages"]["broken"]["status"] == "failed"


def test_initialize_preserves_created_at(tmp_path: Path):
    store = StageStore(tmp_path)
    store.initialize({"created_at": "first", "random_seed": 1})
    store.initialize({"created_at": "second", "random_seed": 2})
    run = json.loads(store.manifest_path.read_text())["run"]
    assert run["created_at"] == "first"
    assert run["last_initialized_at"]


def test_diagnostic_figure_sync_rejects_symlink_and_repairs_tampering(tmp_path: Path):
    source = tmp_path / "source.png"
    source.write_bytes(b"fresh")
    pipeline = object.__new__(JointPipeline)
    pipeline.output_dir = tmp_path
    pipeline._sync_diagnostic_figure(source, "figure.png")
    destination = tmp_path / "figures" / "figure.png"
    destination.write_bytes(b"tampered")
    pipeline._sync_diagnostic_figure(source, "figure.png")
    assert destination.read_bytes() == b"fresh"
    destination.unlink()
    destination.symlink_to(source)
    with pytest.raises(JointError, match="regular file"):
        pipeline._sync_diagnostic_figure(source, "figure.png")


def test_diagnostic_figure_sync_rejects_preexisting_temporary_symlink(tmp_path: Path):
    source = tmp_path / "source.png"
    source.write_bytes(b"fresh")
    marker = tmp_path / "marker"
    marker.write_bytes(b"untouched")
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    temporary = figure_dir / ".figure.png.tmp"
    temporary.symlink_to(marker)
    pipeline = object.__new__(JointPipeline)
    pipeline.output_dir = tmp_path

    with pytest.raises(JointError, match="temporary"):
        pipeline._sync_diagnostic_figure(source, "figure.png")

    assert marker.read_bytes() == b"untouched"


def test_diagnostic_figure_sync_rejects_symlink_figures_parent(tmp_path: Path):
    source = tmp_path / "source.png"
    source.write_bytes(b"fresh")
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "figures").symlink_to(external, target_is_directory=True)
    pipeline = object.__new__(JointPipeline)
    pipeline.output_dir = tmp_path
    with pytest.raises(JointError, match="figures"):
        pipeline._sync_diagnostic_figure(source, "figure.png")
    assert not (external / "figure.png").exists()


def test_diagnostic_figure_sync_honors_overwrite_policy(tmp_path: Path):
    source = tmp_path / "source.png"
    source.write_bytes(b"fresh")
    pipeline = object.__new__(JointPipeline)
    pipeline.output_dir = tmp_path
    pipeline._sync_diagnostic_figure(source, "figure.png", resume=False, overwrite=False)
    source.write_bytes(b"new")
    with pytest.raises(JointError, match="overwrite"):
        pipeline._sync_diagnostic_figure(source, "figure.png", resume=False, overwrite=False)
    pipeline._sync_diagnostic_figure(source, "figure.png", resume=True, overwrite=False)
    assert (tmp_path / "figures/figure.png").read_bytes() == b"new"


def test_logging_reopens_deleted_log(tmp_path: Path):
    first = configure_logging(tmp_path)
    for handler in first.handlers:
        handler.close()
    (tmp_path / "logs/joint.log").unlink()
    second = configure_logging(tmp_path)
    second.info("reopened")
    for handler in second.handlers:
        handler.flush()
    assert "reopened" in (tmp_path / "logs/joint.log").read_text()


def test_logging_closes_stale_fd_backed_stream(tmp_path: Path):
    logger = configure_logging(tmp_path)
    handler = next(handler for handler in logger.handlers if getattr(handler, "_joint_log_path", None))
    stream = handler.stream
    (tmp_path / "logs/joint.log").unlink()

    configure_logging(tmp_path)

    assert stream.closed


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_logging_rejects_unsafe_log_destination(tmp_path: Path, kind: str):
    logs = tmp_path / "logs"
    logs.mkdir()
    target = tmp_path / "external.log"
    if kind == "symlink":
        target.write_text("")
        (logs / "joint.log").symlink_to(target)
    else:
        (logs / "joint.log").mkdir()
    with pytest.raises(JointError, match="joint.log"):
        configure_logging(tmp_path)
    assert target.read_text() == "" if target.exists() and target.is_file() else True


def test_logging_rejects_dangling_log_symlink(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "joint.log").symlink_to(tmp_path / "missing.log")
    with pytest.raises(JointError, match="joint.log"):
        configure_logging(tmp_path)


def test_logging_rejects_dangling_logs_directory_symlink(tmp_path: Path):
    (tmp_path / "logs").symlink_to(tmp_path / "missing-logs", target_is_directory=True)
    with pytest.raises(JointError, match="Log directory"):
        configure_logging(tmp_path)


def test_logging_uses_fd_backed_handler(tmp_path: Path):
    import logging

    logger = configure_logging(tmp_path)
    assert not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    assert any(getattr(handler, "_joint_log_path", None) for handler in logger.handlers)
    assert any(
        isinstance(handler, logging.StreamHandler)
        and not getattr(handler, "_joint_log_path", None)
        for handler in logger.handlers
    )


def test_logging_removes_stale_handler_before_rejecting_replaced_symlink(tmp_path: Path):
    logger = configure_logging(tmp_path)
    log_path = tmp_path / "logs/joint.log"
    external = tmp_path / "external.log"
    external.write_text("must remain unchanged")
    log_path.unlink()
    log_path.symlink_to(external)
    with pytest.raises(JointError, match="joint.log"):
        configure_logging(tmp_path)
    assert not any(getattr(handler, "_joint_log_path", None) for handler in logger.handlers)
    assert external.read_text() == "must remain unchanged"


def test_logging_reopens_handler_after_log_rotation(tmp_path: Path):
    logger = configure_logging(tmp_path)
    log_path = tmp_path / "logs/joint.log"
    rotated_log_path = tmp_path / "logs/joint.log.1"
    logger.info("before rotation")
    for handler in logger.handlers:
        handler.flush()
    log_path.rename(rotated_log_path)
    log_path.write_text("")

    configure_logging(tmp_path).info("after rotation")
    for handler in logger.handlers:
        handler.flush()

    assert "after rotation" in log_path.read_text()
    assert "after rotation" not in rotated_log_path.read_text()


def test_cardinal_probe_errors_are_recorded(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "input.h5ad"
    input_path.write_bytes(b"placeholder")
    config = JointConfig.model_validate({"project": {"name": "x", "output_dir": tmp_path / "out"}, "input": {"h5ad": input_path}, "preprocessing": {"backend": "cardinal"}})
    monkeypatch.setattr("joint.pipeline.shutil.which", lambda _: "/usr/bin/Rscript")
    monkeypatch.setattr("joint.pipeline.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("probe failed")))
    metadata = build_run_metadata(config)
    assert metadata["r_cardinal_version"] is None
    assert "probe failed" in metadata["r_cardinal_probe_error"]


def test_metadata_fingerprints_enabled_matrix_reference(tmp_path: Path):
    input_path = tmp_path / "input.h5ad"
    input_path.write_bytes(b"input")
    reference = tmp_path / "matrix.csv"
    reference.write_bytes(b"reference")
    config = JointConfig.model_validate({"project": {"name": "x", "output_dir": tmp_path / "out"}, "input": {"h5ad": input_path}, "preprocessing": {"matrix_removal": {"enabled": True, "method": "reference", "reference_file": reference}}})
    paths = {item["path"] for item in build_run_metadata(config)["inputs"]}
    assert str(reference.resolve()) in paths


def test_plot_figures_can_be_closed():
    figure = plt.figure()
    plt.close(figure)
    assert figure.number not in plt.get_fignums()
