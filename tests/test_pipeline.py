import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from joint import pipeline
from joint.config import JointConfig
from joint.errors import JointError
from joint.pipeline import StageStore, file_signature, stage_signature


def test_file_signature_changes_when_input_changes(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("first")
    first = file_signature(path)
    path.write_text("second")
    second = file_signature(path)
    assert first != second


def test_file_signature_wraps_missing_file_error(tmp_path: Path):
    with pytest.raises(JointError, match="file signature"):
        file_signature(tmp_path / "missing.txt")


def test_file_signature_wraps_unresolvable_path(tmp_path: Path):
    with pytest.raises(JointError, match="file signature"):
        file_signature(str(tmp_path / "invalid") + "\x00")


def test_file_signature_wraps_symlink_loop(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second.name)
    second.symlink_to(first.name)

    with pytest.raises(JointError, match="file signature"):
        file_signature(first)


@pytest.mark.parametrize("writer", ["npy", "csv", "text", "bytes", "json"])
def test_atomic_artifact_writers_do_not_follow_preplaced_temporary_symlink(
    tmp_path: Path, writer: str
):
    destinations = {
        "npy": "labels.npy",
        "csv": "table.csv",
        "text": "resolved-config.yaml",
        "bytes": "payload.bin",
        "json": "report.json",
    }
    target = tmp_path / destinations[writer]
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not overwrite")
    target.with_name(f".{target.name}.tmp").symlink_to(outside)

    if writer == "npy":
        pipeline._atomic_save_npy(np.array([1]), target)
        np.testing.assert_array_equal(np.load(target), [1])
    elif writer == "csv":
        pipeline._atomic_save_csv(pd.DataFrame({"value": [1]}), target)
        assert pd.read_csv(target)["value"].tolist() == [1]
    elif writer == "text":
        pipeline._atomic_save_text("safe", target)
        assert target.read_text() == "safe"
    elif writer == "bytes":
        pipeline._atomic_save_bytes(b"safe", target)
        assert target.read_bytes() == b"safe"
    else:
        pipeline._atomic_save_json({"value": 1}, target)
        assert pipeline._read_json_mapping(target) == {"value": 1}

    assert outside.read_bytes() == b"do not overwrite"


@pytest.mark.parametrize("writer", ["npy", "csv", "text", "bytes", "json"])
def test_atomic_artifact_writers_close_fd_when_fdopen_fails(monkeypatch, tmp_path: Path, writer: str):
    captured: list[int] = []

    def failing_fdopen(fd, *args, **kwargs):
        captured.append(fd)
        raise OSError("fdopen failed")

    monkeypatch.setattr(pipeline.os, "fdopen", failing_fdopen)
    target = tmp_path / f"{writer}.artifact"

    with pytest.raises(OSError, match="fdopen failed"):
        if writer == "npy":
            pipeline._atomic_save_npy(np.array([1]), target)
        elif writer == "csv":
            pipeline._atomic_save_csv(pd.DataFrame({"value": [1]}), target)
        elif writer == "text":
            pipeline._atomic_save_text("safe", target)
        elif writer == "bytes":
            pipeline._atomic_save_bytes(b"safe", target)
        else:
            pipeline._atomic_save_json({"value": 1}, target)

    assert captured
    for fd in captured:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_stage_signature_accepts_joint_config(tmp_path: Path):
    config = JointConfig.model_validate(
        {
            "project": {"name": "example", "output_dir": tmp_path / "output"},
            "input": {"h5ad": tmp_path / "input.h5ad"},
        }
    )

    assert stage_signature(config, []) == stage_signature(config.model_dump(mode="json"), [])


def test_stage_signature_changes_when_input_content_changes(tmp_path: Path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("first")
    first = stage_signature({"method": "example"}, [input_path])
    input_path.write_text("second")

    assert stage_signature({"method": "example"}, [input_path]) != first


def test_stage_store_wraps_invalid_output_directory(tmp_path: Path):
    output = tmp_path / "output"
    output.write_text("not a directory")

    with pytest.raises(JointError, match="output directory"):
        StageStore(output)


def test_stage_store_resumes_only_matching_completed_stage(tmp_path: Path):
    store = StageStore(tmp_path)
    calls = []

    def producer(staging_dir: Path):
        artifact = staging_dir / "msi.h5ad"
        artifact.write_text("result")
        calls.append("run")
        return {"adata": artifact}

    signature = stage_signature({"normalization": "rms"}, [])
    first = store.run("preprocessing", signature, producer, resume=False, overwrite=False)
    second = store.run("preprocessing", signature, producer, resume=True, overwrite=False)

    artifact = (tmp_path / "preprocessing" / "msi.h5ad").resolve()
    assert first == second == {"adata": artifact}
    assert calls == ["run"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stages"]["preprocessing"]["status"] == "complete"
    recorded = manifest["stages"]["preprocessing"]["artifacts"]["adata"]
    assert recorded["path"] == str(artifact)
    assert recorded["signature"]["size"] == len("result")
    assert len(recorded["signature"]["sha256"]) == 64


def test_stage_store_does_not_resume_tampered_artifact(tmp_path: Path):
    store = StageStore(tmp_path)
    calls = []

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("original")
        calls.append("run")
        return {"result": artifact}

    store.run("analysis", "signature", producer, resume=False, overwrite=False)
    (tmp_path / "analysis" / "result.txt").write_text("tampered")

    with pytest.raises(JointError, match="overwrite"):
        store.run("analysis", "signature", producer, resume=True, overwrite=False)

    assert calls == ["run"]


def test_stage_store_does_not_resume_missing_artifact(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("result")
        return {"result": artifact}

    store.run("analysis", "signature", producer, resume=False, overwrite=False)
    (tmp_path / "analysis" / "result.txt").unlink()

    with pytest.raises(JointError, match="overwrite"):
        store.run("analysis", "signature", producer, resume=True, overwrite=False)


def test_stage_store_does_not_resume_non_complete_record(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("result")
        return {"result": artifact}

    store.run("analysis", "signature", producer, resume=False, overwrite=False)
    manifest = json.loads(store.manifest_path.read_text())
    manifest["stages"]["analysis"]["status"] = "running"
    store.manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(JointError, match="overwrite"):
        store.run("analysis", "signature", producer, resume=True, overwrite=False)


def test_stage_store_rejects_existing_output_without_overwrite(tmp_path: Path):
    store = StageStore(tmp_path)
    artifact = tmp_path / "segmentation" / "labels.npy"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("old")

    with pytest.raises(JointError, match="overwrite"):
        store.run(
            "segmentation",
            "new-signature",
            lambda staging_dir: {"labels": staging_dir / "labels.npy"},
            resume=False,
            overwrite=False,
        )


def test_stage_store_protects_broken_stage_symlink_without_overwrite(tmp_path: Path):
    store = StageStore(tmp_path)
    stage_link = tmp_path / "analysis"
    stage_link.symlink_to(tmp_path / "missing-stage", target_is_directory=True)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("new")
        return {"result": artifact}

    with pytest.raises(JointError, match="overwrite"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert stage_link.is_symlink()


def test_stage_store_replaces_whole_stage_for_mismatched_signature(tmp_path: Path):
    store = StageStore(tmp_path)

    def first_producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("old")
        (staging_dir / "obsolete.txt").write_text("obsolete")
        return {"result": artifact}

    store.run("analysis", "signature-v1", first_producer, resume=False, overwrite=False)

    def second_producer(staging_dir: Path):
        assert staging_dir.parent == tmp_path.resolve()
        assert staging_dir != tmp_path / "analysis"
        assert not (tmp_path / "analysis").exists()
        artifact = staging_dir / "result.txt"
        artifact.write_text("new")
        return {"result": artifact}

    artifacts = store.run(
        "analysis",
        "signature-v2",
        second_producer,
        resume=True,
        overwrite=True,
    )

    assert artifacts == {"result": (tmp_path / "analysis" / "result.txt").resolve()}
    assert artifacts["result"].read_text() == "new"
    assert not (tmp_path / "analysis" / "obsolete.txt").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stages"]["analysis"]["signature"] == "signature-v2"


def test_stage_store_preserves_existing_stage_when_overwrite_producer_fails(tmp_path: Path):
    store = StageStore(tmp_path)

    def first_producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("old")
        return {"result": artifact}

    store.run("analysis", "signature-v1", first_producer, resume=False, overwrite=False)
    old_manifest = (tmp_path / "manifest.json").read_bytes()

    def failing_producer(staging_dir: Path):
        (staging_dir / "result.txt").write_text("new")
        raise RuntimeError("backend failed")

    with pytest.raises(JointError, match="backend failed"):
        store.run(
            "analysis",
            "signature-v2",
            failing_producer,
            resume=False,
            overwrite=True,
        )

    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert (tmp_path / "manifest.json").read_bytes() == old_manifest
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []


def test_stage_store_restores_old_stage_after_producer_writes_final_then_fails(
    tmp_path: Path,
):
    store = StageStore(tmp_path)

    def first_producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("old")
        return {"result": artifact}

    store.run("analysis", "signature-v1", first_producer, resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()

    def failing_producer(staging_dir: Path):
        final_dir = tmp_path / "analysis"
        final_dir.mkdir(exist_ok=True)
        (final_dir / "result.txt").write_text("corrupt")
        (final_dir / "unvalidated.txt").write_text("bad")
        raise RuntimeError("backend failed")

    with pytest.raises(JointError, match="producer.*backend failed"):
        store.run(
            "analysis",
            "signature-v2",
            failing_producer,
            resume=False,
            overwrite=True,
        )

    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert not (tmp_path / "analysis" / "unvalidated.txt").exists()
    assert store.manifest_path.read_bytes() == old_manifest
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []


@pytest.mark.parametrize("phase", ["producer", "validation", "promotion", "manifest"])
def test_stage_store_restores_old_stage_when_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
):
    store = StageStore(tmp_path)

    def first_producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("old")
        return {"result": artifact}

    store.run("analysis", "signature-v1", first_producer, resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()
    interrupt = KeyboardInterrupt(f"interrupted during {phase}")

    def producer(staging_dir: Path):
        if phase == "producer":
            raise interrupt
        artifact = staging_dir / "result.txt"
        artifact.write_text("new")
        return {"result": artifact}

    if phase == "validation":
        monkeypatch.setattr(
            pipeline,
            "_validate_directory_identity",
            lambda staging_dir, expected: (_ for _ in ()).throw(interrupt),
        )
    elif phase == "promotion":
        original_replace = pipeline.os.replace

        def interrupt_promotion(source, destination):
            if Path(source).name.startswith(".analysis.staging-"):
                raise interrupt
            original_replace(source, destination)

        monkeypatch.setattr(pipeline.os, "replace", interrupt_promotion)
    elif phase == "manifest":
        monkeypatch.setattr(
            pipeline,
            "_atomic_json",
            lambda payload, destination: (_ for _ in ()).throw(interrupt),
        )

    with pytest.raises(KeyboardInterrupt) as captured:
        store.run(
            "analysis",
            "signature-v2",
            producer,
            resume=False,
            overwrite=True,
        )

    assert captured.value is interrupt
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert store.manifest_path.read_bytes() == old_manifest
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []


def test_stage_store_recovers_orphan_backup_and_discards_staging(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("old")
        return {"result": artifact}

    expected = store.run("analysis", "signature", producer, resume=False, overwrite=False)
    backup = tmp_path / ".analysis.backup-orphan"
    pipeline.os.replace(tmp_path / "analysis", backup)
    orphan_staging = tmp_path / ".analysis.staging-orphan"
    orphan_staging.mkdir()
    (orphan_staging / "partial.txt").write_text("partial")
    pipeline._create_transaction_marker(orphan_staging, "analysis")

    resumed = store.run(
        "analysis",
        "signature",
        lambda staging_dir: pytest.fail("producer must not run"),
        resume=True,
        overwrite=False,
    )

    assert resumed == expected
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert not backup.exists()
    assert not orphan_staging.exists()


def test_stage_store_recovers_backup_after_uncommitted_stage_promotion(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("old")
        return {"result": artifact}

    expected = store.run("analysis", "signature", producer, resume=False, overwrite=False)
    backup = tmp_path / ".analysis.backup-orphan"
    pipeline.os.replace(tmp_path / "analysis", backup)
    uncommitted = tmp_path / "analysis"
    uncommitted.mkdir()
    (uncommitted / "result.txt").write_text("uncommitted")

    resumed = store.run(
        "analysis",
        "signature",
        lambda staging_dir: pytest.fail("producer must not run"),
        resume=True,
        overwrite=False,
    )

    assert resumed == expected
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert not backup.exists()


def test_stage_store_refuses_ambiguous_orphan_backups(tmp_path: Path):
    store = StageStore(tmp_path)
    first = tmp_path / ".analysis.backup-one"
    second = tmp_path / ".analysis.backup-two"
    first.mkdir()
    second.mkdir()
    (first / "result.txt").write_text("one")
    (second / "result.txt").write_text("two")

    with pytest.raises(JointError, match="ambiguous orphan backups"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=True,
            overwrite=False,
        )

    assert not (tmp_path / "analysis").exists()
    assert first.exists()
    assert second.exists()


def test_stage_store_does_not_restore_orphan_backup_with_escaping_parent_symlink(
    tmp_path: Path,
):
    store = StageStore(tmp_path / "output")

    def producer(staging_dir: Path):
        nested = staging_dir / "nested"
        nested.mkdir()
        artifact = nested / "result.txt"
        artifact.write_text("old")
        return {"result": artifact}

    store.run("analysis", "signature", producer, resume=False, overwrite=False)
    shutil.rmtree(store.output_dir / "analysis")
    external = tmp_path / "external"
    external.mkdir()
    (external / "result.txt").write_text("old")
    backup = store.output_dir / ".analysis.backup-orphan"
    backup.mkdir()
    (backup / "nested").symlink_to(external, target_is_directory=True)

    with pytest.raises(JointError, match="ambiguous orphan backups"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=True,
            overwrite=False,
        )

    assert not (store.output_dir / "analysis").exists()
    assert backup.exists()
    assert (external / "result.txt").read_text() == "old"


def test_stage_store_wraps_artifact_parent_symlink_loop_and_restores_old_stage(
    tmp_path: Path,
):
    store = StageStore(tmp_path)

    def first_producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("old")
        return {"result": artifact}

    store.run("analysis", "signature-v1", first_producer, resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()

    def loop_producer(staging_dir: Path):
        first = staging_dir / "first"
        second = staging_dir / "second"
        first.symlink_to(second.name, target_is_directory=True)
        second.symlink_to(first.name, target_is_directory=True)
        return {"result": first / "result.txt"}

    with pytest.raises(JointError, match="artifact path"):
        store.run(
            "analysis",
            "signature-v2",
            loop_producer,
            resume=False,
            overwrite=True,
        )

    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert store.manifest_path.read_bytes() == old_manifest
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []


def test_stage_store_cleanup_failure_does_not_mask_producer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def first_producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("old")
        return {"result": artifact}

    store.run("analysis", "signature-v1", first_producer, resume=False, overwrite=False)
    original_rmtree = pipeline.shutil.rmtree

    def fail_staging_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".analysis.staging-"):
            raise OSError("cleanup failed")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(pipeline.shutil, "rmtree", fail_staging_cleanup)

    with pytest.raises(JointError, match="producer.*primary failure") as captured:
        store.run(
            "analysis",
            "signature-v2",
            lambda staging_dir: (_ for _ in ()).throw(RuntimeError("primary failure")),
            resume=False,
            overwrite=True,
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert any("cleanup failed" in note for note in captured.value.__notes__)
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert list(tmp_path.glob(".analysis.backup-*")) == []
    assert len(list(tmp_path.glob(".analysis.staging-*"))) == 1

    monkeypatch.undo()
    store.run(
        "analysis",
        "signature-v1",
        lambda staging_dir: pytest.fail("producer must not run"),
        resume=True,
        overwrite=False,
    )
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_stage_store_post_commit_backup_cleanup_failure_is_successful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    original_rmtree = pipeline.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".analysis.backup-"):
            raise OSError("backup cleanup failed")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(pipeline.shutil, "rmtree", fail_backup_cleanup)

    artifacts = store.run(
        "analysis",
        "signature-v2",
        producer_with("new"),
        resume=False,
        overwrite=True,
    )

    assert artifacts["result"].read_text() == "new"
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["stages"]["analysis"]["signature"] == "signature-v2"
    assert len(list(tmp_path.glob(".analysis.backup-*"))) == 1

    monkeypatch.undo()
    resumed = store.run(
        "analysis",
        "signature-v2",
        lambda staging_dir: pytest.fail("producer must not run"),
        resume=True,
        overwrite=False,
    )
    assert resumed == artifacts
    assert list(tmp_path.glob(".analysis.backup-*")) == []


def test_stage_store_refuses_replacement_while_committed_orphan_backup_cannot_be_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    original_rmtree = pipeline.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".analysis.backup-"):
            raise OSError("backup cleanup failed")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(pipeline.shutil, "rmtree", fail_backup_cleanup)
    store.run(
        "analysis",
        "signature-v2",
        producer_with("new"),
        resume=False,
        overwrite=True,
    )

    with pytest.raises(JointError, match="orphan backup"):
        store.run(
            "analysis",
            "signature-v3",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=False,
            overwrite=True,
        )

    assert (tmp_path / "analysis" / "result.txt").read_text() == "new"
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["stages"]["analysis"]["signature"] == "signature-v2"


def test_stage_store_restores_existing_stage_when_manifest_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    old_manifest = (tmp_path / "manifest.json").read_bytes()

    def fail_manifest(payload, destination):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "_atomic_json", fail_manifest, raising=False)

    with pytest.raises(JointError, match="manifest.*disk full"):
        store.run(
            "analysis",
            "signature-v2",
            producer_with("new"),
            resume=False,
            overwrite=True,
        )

    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert (tmp_path / "manifest.json").read_bytes() == old_manifest
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []


def test_stage_store_rolls_back_if_interrupted_after_manifest_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()
    interrupt = KeyboardInterrupt("interrupted after manifest replace")
    original_atomic_json = pipeline._atomic_json

    def commit_then_interrupt(payload, destination):
        original_atomic_json(payload, destination)
        raise interrupt

    monkeypatch.setattr(pipeline, "_atomic_json", commit_then_interrupt)

    with pytest.raises(KeyboardInterrupt) as captured:
        store.run(
            "analysis",
            "signature-v2",
            producer_with("new"),
            resume=False,
            overwrite=True,
        )

    assert captured.value is interrupt
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert store.manifest_path.read_bytes() == old_manifest
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_stage_store_restores_regular_manifest_after_producer_replaces_node(
    tmp_path: Path, replacement: str
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()
    external = tmp_path / "external.json"
    external.write_bytes(old_manifest)

    def replacing_producer(staging_dir: Path):
        store.manifest_path.unlink()
        if replacement == "symlink":
            store.manifest_path.symlink_to(external)
        else:
            store.manifest_path.mkdir()
            (store.manifest_path / "keep.txt").write_text("keep")
        raise RuntimeError("primary failure")

    with pytest.raises(JointError, match="producer.*primary failure") as captured:
        store.run(
            "analysis",
            "signature-v2",
            replacing_producer,
            resume=False,
            overwrite=True,
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert store.manifest_path.is_file()
    assert not store.manifest_path.is_symlink()
    assert store.manifest_path.read_bytes() == old_manifest
    assert external.read_bytes() == old_manifest
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []


def test_stage_store_leaves_unchanged_regular_manifest_if_restore_path_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()
    restore_called = False

    def fail_restore(payload: bytes, destination: Path):
        nonlocal restore_called
        restore_called = True
        return ["manifest rollback failed: injected restore failure"]

    monkeypatch.setattr(pipeline, "_restore_file_bytes", fail_restore)

    with pytest.raises(JointError, match="producer.*primary failure") as captured:
        store.run(
            "analysis",
            "signature-v2",
            lambda staging_dir: (_ for _ in ()).throw(RuntimeError("primary failure")),
            resume=False,
            overwrite=True,
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert restore_called is False
    assert store.manifest_path.is_file()
    assert not store.manifest_path.is_symlink()
    assert store.manifest_path.read_bytes() == old_manifest
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"


def test_stage_store_manifest_cleanup_failure_preserves_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    manifest = store.manifest_path
    original_rmtree = pipeline.shutil.rmtree

    def fail_manifest_cleanup(path, *args, **kwargs):
        if Path(path) == manifest:
            raise OSError("manifest directory cleanup failed")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(pipeline.shutil, "rmtree", fail_manifest_cleanup)

    def replacing_producer(staging_dir: Path):
        manifest.unlink()
        manifest.mkdir()
        (manifest / "keep.txt").write_text("keep")
        raise RuntimeError("primary failure")

    with pytest.raises(JointError, match="producer.*primary failure") as captured:
        store.run(
            "analysis",
            "signature-v2",
            replacing_producer,
            resume=False,
            overwrite=True,
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert any("manifest directory cleanup failed" in note for note in captured.value.__notes__)
    assert manifest.is_dir()
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_stage_store_marker_unlink_interrupt_rolls_back_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()
    interrupt = interrupt_type("marker unlink interrupted")
    original_remove_path = pipeline._remove_path
    interrupted = False

    def interrupt_marker_unlink(path: Path):
        nonlocal interrupted
        if not interrupted and Path(path).name.startswith(".joint-transaction-.analysis.staging-"):
            interrupted = True
            raise interrupt
        return original_remove_path(path)

    monkeypatch.setattr(pipeline, "_remove_path", interrupt_marker_unlink)

    with pytest.raises(interrupt_type) as captured:
        store.run(
            "analysis",
            "signature-v2",
            producer_with("new"),
            resume=False,
            overwrite=True,
        )

    assert captured.value is interrupt
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert store.manifest_path.is_file()
    assert store.manifest_path.read_bytes() == old_manifest
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []
    assert list(tmp_path.glob(".joint-transaction-*")) == []


def test_stage_store_keeps_staging_marker_until_promotion_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    original_replace = pipeline.os.replace
    original_remove_path = pipeline._remove_path
    events: list[str] = []

    def observe_promotion(source, destination):
        if Path(source).name.startswith(".analysis.staging-"):
            marker = pipeline._transaction_marker_path(Path(source))
            assert marker.is_file()
            events.append("promotion")
        return original_replace(source, destination)

    def observe_marker_removal(path: Path):
        if Path(path).name.startswith(".joint-transaction-.analysis.staging-"):
            events.append("marker removal")
        return original_remove_path(path)

    monkeypatch.setattr(pipeline.os, "replace", observe_promotion)
    monkeypatch.setattr(pipeline, "_remove_path", observe_marker_removal)

    store.run("analysis", "signature-v2", producer_with("new"), resume=False, overwrite=True)

    assert events == ["promotion", "marker removal"]


def test_stage_store_restore_failure_is_not_masked_by_temp_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()
    external = tmp_path / "external.json"
    external.write_bytes(old_manifest)
    original_replace = pipeline.os.replace
    original_unlink = Path.unlink

    def fail_restore_replace(source, destination):
        if Path(destination) == store.manifest_path and Path(source).name.startswith(
            ".manifest.json.rollback-"
        ):
            raise OSError("restore replace failed")
        return original_replace(source, destination)

    def fail_restore_temp_unlink(path: Path, *args, **kwargs):
        if path.name.startswith(".manifest.json.rollback-"):
            raise OSError("restore temp unlink failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pipeline.os, "replace", fail_restore_replace)
    monkeypatch.setattr(Path, "unlink", fail_restore_temp_unlink)

    def replacing_producer(staging_dir: Path):
        store.manifest_path.unlink()
        store.manifest_path.symlink_to(external)
        raise RuntimeError("primary failure")

    with pytest.raises(JointError, match="producer.*primary failure") as captured:
        store.run(
            "analysis",
            "signature-v2",
            replacing_producer,
            resume=False,
            overwrite=True,
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert any("restore replace failed" in note for note in captured.value.__notes__)
    assert any("restore temp unlink failed" in note for note in captured.value.__notes__)
    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert store.manifest_path.is_symlink()
    assert store.manifest_path.read_bytes() == old_manifest
    assert external.read_bytes() == old_manifest


def test_stage_store_removes_first_stage_when_manifest_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("result")
        return {"result": artifact}

    def fail_manifest(payload, destination):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "_atomic_json", fail_manifest)

    with pytest.raises(JointError, match="manifest.*disk full"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert not (tmp_path / "analysis").exists()
    assert not store.manifest_path.exists()
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_atomic_manifest_write_preserves_existing_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "manifest.json"
    destination.write_text("old manifest")
    original_replace = pipeline.os.replace

    def fail_manifest_replace(source, target):
        if Path(target) == destination:
            raise OSError("replace failed")
        original_replace(source, target)

    monkeypatch.setattr(pipeline.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="replace failed"):
        pipeline._atomic_json({"stages": {}}, destination)

    assert destination.read_text() == "old manifest"
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_atomic_manifest_replace_failure_is_not_masked_by_temp_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "manifest.json"
    destination.write_text("old manifest")
    original_unlink = Path.unlink

    def fail_manifest_replace(source, target):
        if Path(target) == destination:
            raise OSError("replace failed")
        raise AssertionError("unexpected replace")

    def fail_temp_unlink(path: Path, *args, **kwargs):
        if path.name.startswith(".manifest.json.") and path.name.endswith(".tmp"):
            raise OSError("temp unlink failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pipeline.os, "replace", fail_manifest_replace)
    monkeypatch.setattr(Path, "unlink", fail_temp_unlink)

    with pytest.raises(OSError, match="replace failed") as captured:
        pipeline._atomic_json({"stages": {}}, destination)

    assert any("temp unlink failed" in note for note in captured.value.__notes__)
    assert destination.read_text() == "old manifest"
    assert len(list(tmp_path.glob(".manifest.json.*.tmp"))) == 1


def test_stage_store_restores_existing_stage_when_promotion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = StageStore(tmp_path)

    def producer_with(content: str):
        def producer(staging_dir: Path):
            artifact = staging_dir / "result.txt"
            artifact.write_text(content)
            return {"result": artifact}

        return producer

    store.run("analysis", "signature-v1", producer_with("old"), resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()
    original_replace = pipeline.os.replace

    def fail_promotion(source, destination):
        if Path(source).name.startswith(".analysis.staging-"):
            raise OSError("promotion failed")
        original_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", fail_promotion)

    with pytest.raises(JointError, match="promotion.*promotion failed"):
        store.run(
            "analysis",
            "signature-v2",
            producer_with("new"),
            resume=False,
            overwrite=True,
        )

    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert store.manifest_path.read_bytes() == old_manifest
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []


def test_stage_store_rejects_corrupt_manifest(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{not-json")
    store = StageStore(tmp_path)
    called = False

    def producer(staging_dir: Path):
        nonlocal called
        called = True
        return {}

    with pytest.raises(JointError, match="manifest"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert called is False
    assert (tmp_path / "manifest.json").read_text() == "{not-json"


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_stage_store_rejects_non_regular_manifest(tmp_path: Path, kind: str):
    manifest = tmp_path / "manifest.json"
    if kind == "directory":
        manifest.mkdir()
    else:
        external = tmp_path / "external.json"
        external.write_text(json.dumps({"stages": {}}))
        manifest.symlink_to(external)
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="manifest.*regular file"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=False,
            overwrite=False,
        )


@pytest.mark.parametrize(
    "payload",
    [[], {}, {"stages": []}, {"stages": {"analysis": []}}],
)
def test_stage_store_rejects_invalid_manifest_structure(tmp_path: Path, payload):
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="manifest"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=True,
            overwrite=False,
        )


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"status": "complete", "signature": "signature", "artifacts": {}},
        {
            "status": "complete",
            "signature": "signature",
            "artifacts": {"../result": {"path": "/tmp/result", "signature": {}}},
        },
        {
            "status": "complete",
            "signature": "signature",
            "artifacts": {"result": {"path": 123, "signature": {}}},
        },
        {
            "status": "complete",
            "signature": 123,
            "artifacts": {"result": {"path": "/tmp/result", "signature": {}}},
        },
    ],
)
def test_stage_store_rejects_invalid_manifest_record(tmp_path: Path, record):
    payload = {"stages": {"analysis": record}}
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="manifest"):
        store.run(
            "other-stage",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=False,
            overwrite=False,
        )


def test_stage_store_rejects_unsafe_stage_name_in_manifest(tmp_path: Path):
    payload = {"stages": {"../escape": {}}}
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="manifest.*stage name"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=False,
            overwrite=False,
        )


def test_stage_store_rejects_manifest_artifact_path_outside_stage(tmp_path: Path):
    store = StageStore(tmp_path / "output")

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("result")
        return {"result": artifact}

    store.run("analysis", "signature", producer, resume=False, overwrite=False)
    outside = tmp_path / "outside.txt"
    outside.write_text("result")
    manifest = json.loads(store.manifest_path.read_text())
    record = manifest["stages"]["analysis"]["artifacts"]["result"]
    record["path"] = str(outside.resolve())
    record["signature"] = file_signature(outside)
    store.manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(JointError, match="manifest artifact path"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=True,
            overwrite=False,
        )


def test_stage_store_rejects_manifest_artifact_equal_to_stage_path(tmp_path: Path):
    stage_path = tmp_path / "analysis"
    stage_path.write_text("not a stage directory")
    signature = file_signature(stage_path)
    manifest = {
        "stages": {
            "analysis": {
                "status": "complete",
                "signature": "signature",
                "artifacts": {
                    "result": {"path": str(stage_path.resolve()), "signature": signature}
                },
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="manifest artifact path"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=True,
            overwrite=False,
        )


def test_stage_store_wraps_unresolvable_manifest_artifact_path(tmp_path: Path):
    invalid_path = str((tmp_path / "analysis" / "result.txt").resolve()) + "\x00"
    manifest = {
        "stages": {
            "analysis": {
                "status": "complete",
                "signature": "signature",
                "artifacts": {
                    "result": {
                        "path": invalid_path,
                        "signature": {"size": 1, "sha256": "0" * 64},
                    }
                },
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="manifest artifact path"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=True,
            overwrite=False,
        )


def test_stage_store_cleans_staging_when_producer_fails(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        (staging_dir / "partial.txt").write_text("partial")
        raise RuntimeError("backend failed")

    with pytest.raises(JointError, match="producer.*backend failed"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert not (tmp_path / "analysis").exists()
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_stage_store_validates_missing_artifact_before_promotion(tmp_path: Path):
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="expected artifacts"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: {"result": staging_dir / "missing.txt"},
            resume=False,
            overwrite=False,
        )

    assert not (tmp_path / "analysis").exists()
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_stage_store_restores_old_stage_when_overwrite_artifact_is_missing(tmp_path: Path):
    store = StageStore(tmp_path)

    def first_producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("old")
        return {"result": artifact}

    store.run("analysis", "signature-v1", first_producer, resume=False, overwrite=False)
    old_manifest = store.manifest_path.read_bytes()

    with pytest.raises(JointError, match="expected artifacts"):
        store.run(
            "analysis",
            "signature-v2",
            lambda staging_dir: {"result": staging_dir / "missing.txt"},
            resume=False,
            overwrite=True,
        )

    assert (tmp_path / "analysis" / "result.txt").read_text() == "old"
    assert store.manifest_path.read_bytes() == old_manifest
    assert list(tmp_path.glob(".analysis.staging-*")) == []
    assert list(tmp_path.glob(".analysis.backup-*")) == []


def test_stage_store_rejects_artifact_outside_staging_directory(tmp_path: Path):
    store = StageStore(tmp_path / "output")
    outside = tmp_path / "outside.txt"
    outside.write_text("external")

    with pytest.raises(JointError, match="staging directory"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: {"result": outside},
            resume=False,
            overwrite=False,
        )

    assert outside.read_text() == "external"
    assert not (store.output_dir / "analysis").exists()
    assert list(store.output_dir.glob(".analysis.staging-*")) == []


def test_stage_store_rejects_producer_writing_final_stage_directly(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        final_dir = tmp_path / "analysis"
        final_dir.mkdir()
        (final_dir / "unvalidated.txt").write_text("bad")
        artifact = staging_dir / "result.txt"
        artifact.write_text("good")
        return {"result": artifact}

    with pytest.raises(JointError, match="final stage"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert not (tmp_path / "analysis").exists()
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_stage_store_rejects_staging_root_replaced_by_external_symlink(tmp_path: Path):
    store = StageStore(tmp_path / "output")
    external = tmp_path / "external"
    external.mkdir()
    artifact = external / "result.txt"
    artifact.write_text("external")

    def producer(staging_dir: Path):
        shutil.rmtree(staging_dir)
        staging_dir.symlink_to(external, target_is_directory=True)
        return {"result": staging_dir / artifact.name}

    with pytest.raises(JointError, match="staging directory"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert external.is_dir()
    assert artifact.read_text() == "external"
    assert not (store.output_dir / "analysis").exists()
    assert json.loads(store.manifest_path.read_text())["stages"]["analysis"]["status"] == "failed"
    assert list(store.output_dir.glob(".analysis.staging-*")) == []


def test_stage_store_rejects_staging_root_recreated_as_directory(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        shutil.rmtree(staging_dir)
        staging_dir.mkdir()
        artifact = staging_dir / "result.txt"
        artifact.write_text("recreated")
        return {"result": artifact}

    with pytest.raises(JointError, match="identity changed"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert not (tmp_path / "analysis").exists()
    assert json.loads(store.manifest_path.read_text())["stages"]["analysis"]["status"] == "failed"
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_stage_store_rejects_non_mapping_producer_result(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("result")
        return [artifact]

    with pytest.raises(JointError, match="artifact mapping"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert not (tmp_path / "analysis").exists()
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_stage_store_rejects_empty_artifact_mapping(tmp_path: Path):
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="at least one artifact"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: {},
            resume=False,
            overwrite=False,
        )

    assert not (tmp_path / "analysis").exists()
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_stage_store_rejects_unsafe_artifact_name(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("result")
        return {"../result": artifact}

    with pytest.raises(JointError, match="artifact name"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert not (tmp_path / "analysis").exists()
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_stage_store_rejects_invalid_artifact_path_value(tmp_path: Path):
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="artifact path"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: {"result": object()},
            resume=False,
            overwrite=False,
        )

    assert not (tmp_path / "analysis").exists()
    assert list(tmp_path.glob(".analysis.staging-*")) == []


def test_stage_store_rejects_symlink_artifact(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        target = staging_dir / "target.txt"
        target.write_text("result")
        link = staging_dir / "result.txt"
        link.symlink_to(target.name)
        return {"result": link}

    with pytest.raises(JointError, match="regular files"):
        store.run("analysis", "signature", producer, resume=False, overwrite=False)

    assert not (tmp_path / "analysis").exists()
    assert list(tmp_path.glob(".analysis.staging-*")) == []


@pytest.mark.parametrize("stage", ["../escape", "nested/stage", "", "manifest.json"])
def test_stage_store_rejects_unsafe_stage_name(tmp_path: Path, stage: str):
    store = StageStore(tmp_path / "output")
    called = False

    def producer(staging_dir: Path):
        nonlocal called
        called = True
        return {}

    with pytest.raises(JointError, match="stage name"):
        store.run(stage, "signature", producer, resume=False, overwrite=False)

    assert called is False
    assert not (tmp_path / "escape").exists()


def test_stage_store_rejects_internal_namespace_stage_name(tmp_path: Path):
    store = StageStore(tmp_path)

    with pytest.raises(JointError, match="stage name"):
        store.run(
            ".analysis.backup-victim",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=False,
            overwrite=False,
        )


def test_stage_store_resume_does_not_delete_hidden_user_stage_with_internal_prefix(
    tmp_path: Path,
):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("result")
        return {"result": artifact}

    store.run("analysis", "signature", producer, resume=False, overwrite=False)
    user_stage = tmp_path / ".analysis.backup-victim"
    user_stage.mkdir()
    (user_stage / "user-data.txt").write_text("keep")

    resumed = store.run(
        "analysis",
        "signature",
        lambda staging_dir: pytest.fail("producer must not run"),
        resume=True,
        overwrite=False,
    )

    assert resumed == {"result": (tmp_path / "analysis" / "result.txt").resolve()}
    assert (user_stage / "user-data.txt").read_text() == "keep"


def test_stage_store_resume_preserves_backup_with_matching_layout_but_wrong_content(
    tmp_path: Path,
):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("committed")
        return {"result": artifact}

    store.run("analysis", "signature", producer, resume=False, overwrite=False)
    user_stage = tmp_path / ".analysis.backup-victim"
    user_stage.mkdir()
    (user_stage / "result.txt").write_text("user data")

    store.run(
        "analysis",
        "signature",
        lambda staging_dir: pytest.fail("producer must not run"),
        resume=True,
        overwrite=False,
    )

    assert (user_stage / "result.txt").read_text() == "user data"


def test_stage_store_resume_preserves_untrusted_staging_prefix_path(tmp_path: Path):
    store = StageStore(tmp_path)

    def producer(staging_dir: Path):
        artifact = staging_dir / "result.txt"
        artifact.write_text("committed")
        return {"result": artifact}

    store.run("analysis", "signature", producer, resume=False, overwrite=False)
    user_stage = tmp_path / ".analysis.staging-victim"
    user_stage.mkdir()
    (user_stage / "user-data.txt").write_text("keep")

    store.run(
        "analysis",
        "signature",
        lambda staging_dir: pytest.fail("producer must not run"),
        resume=True,
        overwrite=False,
    )

    assert (user_stage / "user-data.txt").read_text() == "keep"


def test_stage_store_preserves_untrusted_single_backup_without_manifest(tmp_path: Path):
    store = StageStore(tmp_path)
    backup = tmp_path / ".analysis.backup-fake"
    backup.mkdir()
    (backup / "user-data.txt").write_text("keep")

    with pytest.raises(JointError, match="untrusted orphan"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=False,
            overwrite=True,
        )

    assert not (tmp_path / "analysis").exists()
    assert (backup / "user-data.txt").read_text() == "keep"


def test_stage_store_overwrite_false_checks_final_before_fake_backup_recovery(tmp_path: Path):
    store = StageStore(tmp_path)
    final = tmp_path / "analysis"
    final.mkdir()
    (final / "user-data.txt").write_text("keep")
    backup = tmp_path / ".analysis.backup-fake"
    backup.mkdir()
    (backup / "replacement.txt").write_text("bad")

    with pytest.raises(JointError, match="overwrite"):
        store.run(
            "analysis",
            "signature",
            lambda staging_dir: pytest.fail("producer must not run"),
            resume=False,
            overwrite=False,
        )

    assert (final / "user-data.txt").read_text() == "keep"
    assert (backup / "replacement.txt").read_text() == "bad"


def pipeline_config(tmp_path: Path):
    import anndata
    import numpy as np

    image = tmp_path / "laser.npy"
    np.save(image, np.zeros((5, 5)))
    cells = tmp_path / "cells.npy"
    np.save(cells, np.zeros((5, 5), dtype=np.int32))
    source = tmp_path / "input.h5ad"
    anndata.AnnData(
        np.ones((1, 1)),
        obs={"dataset": ["sample"]},
        var={"mz": [100.0]},
        obsm={"spatial": np.array([[1.0, 1.0]])},
    ).write_h5ad(source)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
project: {{name: sample, output_dir: {tmp_path / "results"}}}
input:
  h5ad: {source}
  laser_image: {image}
  cell_segmentation: {cells}
laser_segmentation:
  method: real
registration: {{orientation: identity, edge_policy: error}}
quantification: {{method: legacy_proportional}}
""".strip()
    )
    from joint.config import load_config

    return load_config(config_path)


def downstream_adata():
    import anndata
    import numpy as np

    return anndata.AnnData(
        np.array([[1.0, 0.0], [2.0, 3.0], [4.0, 1.0]]),
        obs={"sample": ["a", "b", "c"]},
        var={"mz": [100.0, 200.0]},
        obsm={"spatial": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])},
    )


def write_downstream_input(config, relative_path: str, adata=None):
    from joint.io import atomic_write_h5ad

    value = downstream_adata() if adata is None else adata
    path = config.project.output_dir / relative_path
    atomic_write_h5ad(value, path)
    return value, path


def test_run_calls_spatial_core_then_enabled_downstream_stages(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import QuantificationResult
    from joint.pipeline import JointPipeline

    pipeline_obj = JointPipeline(pipeline_config(tmp_path))
    cell_result = QuantificationResult(
        anndata.AnnData(np.array([[1.0]]), var={"mz": [100.0]}),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
    )
    calls = []
    monkeypatch.setattr(pipeline_obj, "run_spatial_core", lambda **kwargs: cell_result)
    monkeypatch.setattr(pipeline_obj, "qc", lambda adata, **kwargs: calls.append("qc") or adata)
    monkeypatch.setattr(
        pipeline_obj, "annotate", lambda adata, **kwargs: calls.append("annotate") or adata
    )
    monkeypatch.setattr(
        pipeline_obj, "analyze", lambda adata, **kwargs: calls.append("analyze") or adata
    )
    monkeypatch.setattr(
        pipeline_obj,
        "trajectory",
        lambda adata, **kwargs: calls.append("trajectory") or adata,
    )
    monkeypatch.setattr(
        pipeline_obj,
        "_finalize",
        lambda adata, input_path, **kwargs: calls.append("final") or adata,
        raising=False,
    )

    pipeline_obj.config.annotation.enabled = True
    pipeline_obj.config.trajectory.enabled = True
    pipeline_obj.run()

    assert calls == ["qc", "annotate", "analyze", "trajectory", "final"]


def test_qc_persists_typed_spectral_report_and_reads_final_artifact(monkeypatch, tmp_path: Path):
    import json

    import pandas as pd

    from joint.pipeline import JointPipeline, _read_typed_csv

    config = pipeline_config(tmp_path)
    caller, _ = write_downstream_input(config, "quantification/cells.h5ad")
    original = caller.copy()
    spectral = pd.DataFrame(
        {
            "total_intensity": pd.Series([1.0, 5.0, 5.0], index=caller.obs_names, dtype="Float64"),
            "detected_features": pd.Series([1, 2, 2], index=caller.obs_names, dtype="Int64"),
        },
        index=caller.obs_names,
    )
    monkeypatch.setattr("joint.pipeline.spectral_qc", lambda adata: spectral.copy())
    monkeypatch.setattr(
        "joint.pipeline.detect_outliers",
        lambda adata, **kwargs: pd.Series(
            [False, True, False], index=adata.obs_names, name="outlier"
        ),
    )

    result = JointPipeline(config).qc(caller, resume=False, overwrite=False)

    assert "outlier" not in caller.obs
    assert "outlier" not in original.obs
    assert result.obs["outlier"].tolist() == [False, True, False]
    output = config.project.output_dir / "qc"
    report_path = output / "spectral.csv"
    schema_path = output / "spectral.csv.schema.json"
    expected_spectral = spectral.reset_index(names="obs_id")
    pd.testing.assert_frame_equal(_read_typed_csv(report_path, schema_path), expected_spectral)
    manifest = json.loads((config.project.output_dir / "manifest.json").read_text())
    assert set(manifest["stages"]["qc"]["artifacts"]) == {
        "adata",
        "spectral",
        "spectral_schema",
    }


def test_annotation_uses_persisted_qc_input_and_enabled_references(monkeypatch, tmp_path: Path):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    config.annotation.enabled = True
    config.qc.enabled = True
    source, _ = write_downstream_input(config, "qc/cells_qc.h5ad")
    reference = tmp_path / "hmdb.csv"
    reference.write_text("accession,name,mz,mode\nHMDB1,Metabolite,100.0,pos\n")
    config.annotation.hmdb_reference = reference
    calls = []

    def annotate(source_adata, reference_path, **kwargs):
        calls.append((source_adata, Path(reference_path), kwargs))
        result = source_adata.copy()
        result.var["hmdb_accessions"] = ['["HMDB1"]'] * result.n_vars
        return result

    monkeypatch.setattr("joint.pipeline.annotate_hmdb", annotate)
    monkeypatch.setattr(
        "joint.pipeline.annotate_metaboscape",
        lambda *args, **kwargs: pytest.fail("disabled reference must not be called"),
    )
    caller = source.copy()
    result = JointPipeline(config).annotate(caller, resume=False, overwrite=False)

    assert len(calls) == 1
    assert calls[0][1] == reference.resolve()
    assert result.var["hmdb_accessions"].tolist() == ['["HMDB1"]'] * source.n_vars
    assert "hmdb_accessions" not in caller.var
    assert (config.project.output_dir / "annotation/cells_annotated.h5ad").is_file()


def test_analyze_persists_clustered_adata_and_skips_disabled_optional_adapters(
    monkeypatch, tmp_path: Path
):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    config.qc.enabled = False
    config.annotation.enabled = False
    config.analysis.cosg.enabled = False
    config.analysis.cnmf.enabled = False
    source, _ = write_downstream_input(config, "quantification/cells.h5ad")
    calls = []

    def cluster(source_adata, **kwargs):
        calls.append(kwargs)
        result = source_adata.copy()
        result.obs["cluster"] = ["0", "1", "0"]
        return result

    monkeypatch.setattr("joint.pipeline.cluster_cells", cluster)
    monkeypatch.setattr(
        "joint.pipeline.run_cosg", lambda *args, **kwargs: pytest.fail("COSG disabled")
    )
    monkeypatch.setattr(
        "joint.pipeline.prepare_cnmf", lambda *args, **kwargs: pytest.fail("cNMF disabled")
    )

    result = JointPipeline(config).analyze(source.copy(), resume=False, overwrite=False)

    assert result.obs["cluster"].tolist() == ["0", "1", "0"]
    assert calls[0]["random_seed"] == config.project.random_seed
    assert (config.project.output_dir / "analysis/cells_analyzed.h5ad").is_file()


def test_analyze_persists_cosg_typed_table_and_cnmf_usage_artifact(monkeypatch, tmp_path: Path):
    import pandas as pd

    from joint.analysis import PreparedCnmf
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    config.qc.enabled = False
    config.annotation.enabled = False
    config.clustering.enabled = False
    config.analysis.cosg.enabled = True
    config.analysis.cnmf.enabled = True
    config.analysis.cnmf.components = [2]
    config.analysis.cnmf.selected_k = 2
    source, _ = write_downstream_input(config, "quantification/cells.h5ad")
    cosg_table = pd.DataFrame({"rank": [1], "0": ["100.0"]})
    calls = []

    monkeypatch.setattr("joint.pipeline.run_cosg", lambda adata, **kwargs: cosg_table.copy())

    class FakeBackend:
        def prepare(self, **kwargs):
            calls.append(("prepare", kwargs))

    def prepare(adata, output_dir, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        task_dir = output_dir / config.project.name
        task_dir.mkdir()
        usage = task_dir / f"{config.project.name}.usages.k_2.txt"
        prepared = PreparedCnmf(
            backend=FakeBackend(),
            counts_path=output_dir / "counts.h5ad",
            output_dir=output_dir,
            name=config.project.name,
            task_dir=task_dir,
            staging_dir=output_dir,
            components=(2,),
            seed=kwargs["seed"],
            num_highvar_genes=kwargs["num_highvar_genes"],
        )
        usage.write_text(
            "Cell_ID\tUsage_1\n" + "\n".join(f"{name}\t1.0" for name in adata.obs_names) + "\n"
        )
        return prepared

    monkeypatch.setattr("joint.pipeline.prepare_cnmf", prepare)
    monkeypatch.setattr(
        "joint.pipeline.run_cnmf",
        lambda prepared, **kwargs: calls.append(("run", kwargs)) or prepared,
    )
    monkeypatch.setattr("joint.pipeline.load_cnmf_results", lambda adata, path: adata.copy())

    JointPipeline(config).analyze(source.copy(), resume=False, overwrite=False)

    output = config.project.output_dir / "analysis"
    assert (output / "cosg.csv").is_file()
    assert (output / "cosg.csv.schema.json").is_file()
    assert len(list((output / "cnmf").rglob("*.txt"))) == 1
    assert (output / "cnmf-complete.json").is_file()
    assert calls[0][0] == "run"


def test_trajectory_validates_path_and_persists_typed_trends(monkeypatch, tmp_path: Path):
    import json

    import pandas as pd

    from joint.pipeline import JointPipeline, _read_typed_csv

    config = pipeline_config(tmp_path)
    config.trajectory.enabled = True
    config.trajectory.features = ["0"]
    config.trajectory.points = 3
    path_file = tmp_path / "path.csv"
    path_file.write_text("x,y\n0,0\n2,0\n")
    config.trajectory.path_file = path_file
    source, _ = write_downstream_input(config, "analysis/cells_analyzed.h5ad")
    trends = pd.DataFrame(
        {
            "feature": pd.Series(["0", "0", "0"], dtype="string"),
            "position": [0.0, 0.5, 1.0],
            "fitted": [1.0, 2.0, 3.0],
            "lower": [0.0, 1.0, 2.0],
            "upper": [2.0, 3.0, 4.0],
        }
    )

    def fit(adata, path):
        assert path.tolist() == [[0.0, 0.0], [2.0, 0.0]]
        result = adata.copy()
        result.obs["trajectory_position"] = [0.0, 0.5, 1.0]
        return result

    monkeypatch.setattr("joint.pipeline.fit_spatial_trajectory", fit)
    monkeypatch.setattr(
        "joint.pipeline.calculate_feature_trends",
        lambda adata, features, **kwargs: trends.copy(),
    )

    result = JointPipeline(config).trajectory(source.copy(), resume=False, overwrite=False)

    assert result.obs["trajectory_position"].tolist() == [0.0, 0.5, 1.0]
    output = config.project.output_dir / "trajectory"
    pd.testing.assert_frame_equal(
        _read_typed_csv(output / "trends.csv", output / "trends.csv.schema.json"), trends
    )
    manifest = json.loads((config.project.output_dir / "manifest.json").read_text())
    assert set(manifest["stages"]["trajectory"]["artifacts"]) == {
        "adata",
        "trends",
        "trends_schema",
    }


def test_trajectory_propagates_optional_dependency_error(monkeypatch, tmp_path: Path):
    from joint.errors import OptionalDependencyError
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    config.trajectory.enabled = True
    config.trajectory.features = ["0"]
    path_file = tmp_path / "path.csv"
    path_file.write_text("x,y\n0,0\n1,0\n")
    config.trajectory.path_file = path_file
    source, _ = write_downstream_input(config, "analysis/cells_analyzed.h5ad")
    monkeypatch.setattr(
        "joint.pipeline.fit_spatial_trajectory",
        lambda adata, path: adata.copy(),
    )
    expected = OptionalDependencyError("pygam missing")
    monkeypatch.setattr(
        "joint.pipeline.calculate_feature_trends",
        lambda *args, **kwargs: (_ for _ in ()).throw(expected),
    )

    with pytest.raises(OptionalDependencyError) as caught:
        JointPipeline(config).trajectory(source.copy(), resume=False, overwrite=False)
    assert caught.value is expected


def test_run_persists_hash_verified_final_stage_and_root_copy(monkeypatch, tmp_path: Path):
    import json

    import pandas as pd

    from joint.models import QuantificationResult
    from joint.pipeline import JointPipeline, file_signature

    config = pipeline_config(tmp_path)
    config.qc.enabled = False
    config.annotation.enabled = False
    config.trajectory.enabled = False
    quantified, _ = write_downstream_input(config, "quantification/cells.h5ad")
    analyzed = quantified.copy()
    analyzed.obs["cluster"] = ["0", "1", "0"]
    write_downstream_input(config, "analysis/cells_analyzed.h5ad", analyzed)
    empty = pd.DataFrame()
    pipeline_obj = JointPipeline(config)
    monkeypatch.setattr(
        pipeline_obj,
        "run_spatial_core",
        lambda **kwargs: QuantificationResult(
            quantified.copy(), empty.copy(), empty.copy(), empty.copy(), {}
        ),
    )
    monkeypatch.setattr(pipeline_obj, "analyze", lambda adata, **kwargs: analyzed.copy())

    result = pipeline_obj.run(resume=False, overwrite=False)

    final_stage = config.project.output_dir / "final/joint-final.h5ad"
    root_final = config.project.output_dir / "joint-final.h5ad"
    assert final_stage.is_file()
    assert root_final.is_file()
    assert file_signature(final_stage)["sha256"] == file_signature(root_final)["sha256"]
    assert result.obs["cluster"].tolist() == ["0", "1", "0"]
    manifest = json.loads((config.project.output_dir / "manifest.json").read_text())
    assert set(manifest["stages"]["final"]["artifacts"]) == {"adata"}


def test_final_root_requires_overwrite_unless_resume_repairs_tampering(tmp_path: Path):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    pipeline_obj = JointPipeline(config)
    stage = tmp_path / "stage.h5ad"
    root = config.project.output_dir / "joint-final.h5ad"
    stage.parent.mkdir(parents=True, exist_ok=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    stage.write_bytes(b"new")
    root.write_bytes(b"old")

    with pytest.raises(JointError, match="joint-final.h5ad"):
        pipeline_obj._sync_final_root(stage, resume=False, overwrite=False)
    pipeline_obj._sync_final_root(stage, resume=True, overwrite=False)
    assert root.read_bytes() == b"new"


def test_final_root_write_does_not_follow_preplaced_temporary_symlink(tmp_path: Path):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    pipeline_obj = JointPipeline(config)
    stage = tmp_path / "stage.h5ad"
    root = config.project.output_dir / "joint-final.h5ad"
    stage.write_bytes(b"new")
    root.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not overwrite")
    (root.parent / ".joint-final.h5ad.tmp").symlink_to(outside)

    pipeline_obj._sync_final_root(stage, resume=False, overwrite=False)

    assert root.read_bytes() == b"new"
    assert outside.read_bytes() == b"do not overwrite"


def test_final_root_copies_through_the_exclusively_created_descriptor(
    monkeypatch, tmp_path: Path
):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    pipeline_obj = JointPipeline(config)
    stage = tmp_path / "stage.h5ad"
    stage.write_bytes(b"new")
    config.project.output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "joint.pipeline.shutil.copyfile",
        lambda *args, **kwargs: pytest.fail("final root must not reopen the temporary path"),
    )

    destination = pipeline_obj._sync_final_root(stage, resume=False, overwrite=False)

    assert destination.read_bytes() == b"new"


def test_final_stage_requires_the_persisted_last_h5ad(tmp_path: Path):
    from joint.pipeline import JointPipeline

    pipeline_obj = JointPipeline(pipeline_config(tmp_path))

    with pytest.raises(JointError):
        pipeline_obj._finalize(
            downstream_adata(),
            pipeline_obj.output_dir / "analysis/cells_analyzed.h5ad",
            resume=False,
            overwrite=False,
        )


@pytest.mark.parametrize("slot", ["uns", "raw", "obsp", "varm", "varp"])
def test_downstream_rejects_caller_identity_changes_in_semantic_slots(tmp_path: Path, slot: str):
    import numpy as np

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    source, _ = write_downstream_input(config, "quantification/cells.h5ad")
    source.uns["semantic"] = {"values": np.array([1.0, 2.0])}
    source.raw = source.copy()
    source.obsp["connectivities"] = np.eye(source.n_obs)
    source.varm["loadings"] = np.ones((source.n_vars, 1))
    source.varp["covariance"] = np.eye(source.n_vars)
    _, input_path = write_downstream_input(config, "quantification/cells.h5ad", source)
    caller = source.copy()
    if slot == "uns":
        caller.uns["semantic"]["values"][0] = 99.0
    elif slot == "raw":
        caller.raw = caller.raw.to_adata()[:, [1, 0]].copy()
    elif slot == "obsp":
        caller.obsp["connectivities"][0, 0] = 99.0
    elif slot == "varm":
        caller.varm["loadings"][0, 0] = 99.0
    else:
        caller.varp["covariance"][0, 0] = 99.0

    with pytest.raises(JointError, match="persisted upstream"):
        JointPipeline(config)._persisted_stage_input(caller, input_path)


def test_final_conflict_fails_before_commit_and_valid_resume_repairs_root(
    monkeypatch, tmp_path: Path
):
    import json

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    source, input_path = write_downstream_input(config, "analysis/cells_analyzed.h5ad")
    root = config.project.output_dir / "joint-final.h5ad"
    root.write_bytes(b"protected-root")
    pipeline_obj = JointPipeline(config)

    with pytest.raises(JointError, match="joint-final.h5ad"):
        pipeline_obj._finalize(source.copy(), input_path, resume=False, overwrite=False)

    assert root.read_bytes() == b"protected-root"
    assert not (config.project.output_dir / "final").exists()
    manifest_path = config.project.output_dir / "manifest.json"
    if manifest_path.exists():
        assert "final" not in json.loads(manifest_path.read_text())["stages"]

    with pytest.raises(JointError, match="joint-final.h5ad"):
        pipeline_obj._finalize(source.copy(), input_path, resume=True, overwrite=False)
    assert not (config.project.output_dir / "final").exists()

    first = pipeline_obj._finalize(source.copy(), input_path, resume=False, overwrite=True)
    expected_root = root.read_bytes()
    root.write_bytes(b"tampered-root")
    monkeypatch.setattr(
        "joint.pipeline.atomic_write_h5ad",
        lambda *args, **kwargs: pytest.fail("valid resume must skip final producer"),
    )
    resumed = pipeline_obj._finalize(source.copy(), input_path, resume=True, overwrite=False)

    assert root.read_bytes() == expected_root
    assert resumed.shape == first.shape == source.shape


def test_qc_real_stage_resume_skips_producer_and_rejects_tampered_artifacts_or_input(
    monkeypatch, tmp_path: Path
):
    import numpy as np
    import pandas as pd

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    source, input_path = write_downstream_input(config, "quantification/cells.h5ad")
    calls = []
    monkeypatch.setattr(
        "joint.pipeline.spectral_qc",
        lambda adata: (
            calls.append("spectral")
            or pd.DataFrame(
                {"total_intensity": [1.0, 5.0, 5.0], "detected_features": [1, 2, 2]},
                index=adata.obs_names,
            )
        ),
    )
    monkeypatch.setattr(
        "joint.pipeline.detect_outliers",
        lambda adata, **kwargs: (
            calls.append("outliers")
            or pd.Series([False, True, False], index=adata.obs_names, name="outlier")
        ),
    )
    pipeline_obj = JointPipeline(config)
    first = pipeline_obj.qc(source.copy(), resume=False, overwrite=False)
    monkeypatch.setattr(
        "joint.pipeline.spectral_qc",
        lambda *args, **kwargs: pytest.fail("valid resume must skip spectral producer"),
    )
    monkeypatch.setattr(
        "joint.pipeline.detect_outliers",
        lambda *args, **kwargs: pytest.fail("valid resume must skip outlier producer"),
    )

    resumed = pipeline_obj.qc(source.copy(), resume=True, overwrite=False)

    assert calls == ["spectral", "outliers"]
    assert resumed.obs["outlier"].tolist() == first.obs["outlier"].tolist()

    qc_path = config.project.output_dir / "qc/cells_qc.h5ad"
    qc_path.write_bytes(b"tampered")
    with pytest.raises(JointError, match="overwrite"):
        pipeline_obj.qc(source.copy(), resume=True, overwrite=False)

    restored = source.copy()
    restored.X = np.asarray(restored.X) + 1.0
    from joint.io import atomic_write_h5ad

    atomic_write_h5ad(restored, input_path)
    with pytest.raises(JointError, match="overwrite"):
        pipeline_obj.qc(restored.copy(), resume=True, overwrite=False)


def test_final_root_write_failure_leaves_recoverable_verified_stage(monkeypatch, tmp_path: Path):
    import json

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    source, input_path = write_downstream_input(config, "analysis/cells_analyzed.h5ad")
    pipeline_obj = JointPipeline(config)
    original_replace = pipeline.os.replace

    def fail_root_replace(source_path, destination_path):
        if Path(destination_path) == config.project.output_dir / "joint-final.h5ad":
            raise OSError("root sync failed")
        return original_replace(source_path, destination_path)

    monkeypatch.setattr(pipeline.os, "replace", fail_root_replace)
    with pytest.raises(JointError, match="synchronize.*root sync failed"):
        pipeline_obj._finalize(source.copy(), input_path, resume=False, overwrite=False)

    final_path = config.project.output_dir / "final/joint-final.h5ad"
    root = config.project.output_dir / "joint-final.h5ad"
    assert final_path.is_file()
    assert not root.exists()
    manifest = json.loads((config.project.output_dir / "manifest.json").read_text())
    assert manifest["stages"]["final"]["status"] == "complete"
    assert not (config.project.output_dir / ".joint-final.h5ad.tmp").exists()

    monkeypatch.setattr(pipeline.os, "replace", original_replace)
    monkeypatch.setattr(
        "joint.pipeline.atomic_write_h5ad",
        lambda *args, **kwargs: pytest.fail("recovery resume must skip final producer"),
    )
    recovered = pipeline_obj._finalize(source.copy(), input_path, resume=True, overwrite=False)
    assert root.read_bytes() == final_path.read_bytes()
    assert recovered.shape == source.shape


def test_downstream_rejects_caller_raw_varm_change(tmp_path: Path):
    import numpy as np

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    source = downstream_adata()
    raw = source.copy()
    raw.varm["raw_loadings"] = np.ones((raw.n_vars, 1))
    source.raw = raw
    _, input_path = write_downstream_input(config, "quantification/cells.h5ad", source)
    caller = source.copy()
    changed_raw = caller.raw.to_adata()
    changed_raw.varm["raw_loadings"][0, 0] = 99.0
    caller.raw = changed_raw

    with pytest.raises(JointError, match="persisted upstream"):
        JointPipeline(config)._persisted_stage_input(caller, input_path)


def test_downstream_accepts_unchanged_structured_uns_roundtrip(tmp_path: Path):
    import numpy as np

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    source = downstream_adata()
    source.uns["records"] = np.array(
        [(1, 2.5), (3, 4.5)], dtype=[("identifier", "i4"), ("score", "f8")]
    )
    _, input_path = write_downstream_input(config, "quantification/cells.h5ad", source)

    result = JointPipeline(config)._persisted_stage_input(source.copy(), input_path)

    assert result.shape == source.shape


def test_joint_pipeline_runs_spatial_core_and_writes_stage_artifacts(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import QuantificationResult, RegistrationResult, SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    pipeline_obj = JointPipeline(config)
    msi = anndata.read_h5ad(config.input.h5ad)
    segmented = SegmentationResult(
        labels=np.array([[1]], dtype=np.int32),
        regions=pd.DataFrame(
            {
                "label": [1],
                "area": [1],
                "centroid-0": [1.0],
                "centroid-1": [1.0],
                "seg_label": ["1"],
                "morphology": ["intact"],
                "row_number": [1],
                "column_number": [1],
                "right_to_left": [-1],
            }
        ),
        diagnostics={},
    )
    registered = RegistrationResult(
        msi, segmented.regions.assign(pixel_id=msi.obs_names), {"orientation": "identity"}, {}
    )
    empty_mapping = pd.DataFrame({"cell_id": pd.Series(dtype=str)})
    quantified = QuantificationResult(
        msi,
        empty_mapping.copy(),
        empty_mapping.copy(),
        empty_mapping.copy(),
        {"included_cells": 0, "quantified_cells": 0},
    )
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    monkeypatch.setattr("joint.pipeline._segment_from_config", lambda *args, **kwargs: segmented)
    monkeypatch.setattr("joint.pipeline.register_laser_points", lambda *args, **kwargs: registered)
    monkeypatch.setattr("joint.pipeline.quantify_cells", lambda *args, **kwargs: quantified)

    result = pipeline_obj.run_spatial_core(overwrite=False, resume=False)

    assert result.adata.shape == (1, 1)
    assert (config.project.output_dir / "preprocessing/msi.h5ad").is_file()
    assert (config.project.output_dir / "segmentation/laser_labels.npy").is_file()
    assert (config.project.output_dir / "registration/laser.h5ad").is_file()
    assert (config.project.output_dir / "quantification/cells.h5ad").is_file()


def test_pipeline_rebuilds_registration_report_with_integer_row_keys(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import RegistrationResult, SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    pipeline_obj = JointPipeline(config)
    msi = anndata.read_h5ad(config.input.h5ad)
    segmented = SegmentationResult(
        np.array([[1]], dtype=np.int32),
        pd.DataFrame(
            {
                "label": [1],
                "area": [1.0],
                "centroid-0": [1.0],
                "centroid-1": [1.0],
                "seg_label": ["1"],
                "morphology": ["intact"],
                "row_number": [1],
                "column_number": [1],
                "right_to_left": [-1],
            }
        ),
        {"object_count": 1},
    )
    original = RegistrationResult(
        msi,
        segmented.regions.assign(pixel_id=msi.obs_names),
        {"orientation": "identity", "edge_policy": "error"},
        {"registered_pixels": 1, "truncated_pixels_by_row": {1: 0}},
    )
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    monkeypatch.setattr("joint.pipeline._segment_from_config", lambda *args, **kwargs: segmented)
    monkeypatch.setattr("joint.pipeline.register_laser_points", lambda *args, **kwargs: original)

    pipeline_obj.preprocess(resume=False, overwrite=False)
    saved = pipeline_obj.segment(msi, resume=False, overwrite=False)
    rebuilt = pipeline_obj.register(msi, saved, resume=False, overwrite=False)

    assert saved.diagnostics == segmented.diagnostics
    assert rebuilt.transform == original.transform
    assert rebuilt.report == original.report


def test_pipeline_resume_uses_persisted_stage_artifacts(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import QuantificationResult, RegistrationResult, SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    pipeline_obj = JointPipeline(config)
    msi = anndata.read_h5ad(config.input.h5ad)
    regions = pd.DataFrame(
        {
            "label": [1],
            "area": [1.0],
            "centroid-0": [1.0],
            "centroid-1": [1.0],
            "seg_label": ["1"],
            "morphology": ["intact"],
            "row_number": [1],
            "column_number": [1],
            "right_to_left": [-1],
        }
    )
    segmented = SegmentationResult(np.array([[1]], dtype=np.int32), regions, {"object_count": 1})
    registered = RegistrationResult(
        msi,
        regions.assign(pixel_id=msi.obs_names),
        {"orientation": "identity"},
        {"registered_pixels": 1},
    )
    empty = pd.DataFrame({"cell_id": pd.Series(dtype=str)})
    quantified = QuantificationResult(
        msi, empty, empty, empty, {"included_cells": 0, "quantified_cells": 0}
    )
    calls = {"preprocess": 0, "segment": 0, "register": 0, "quantify": 0}

    def preprocess(*args, **kwargs):
        calls["preprocess"] += 1
        assert kwargs["cardinal_output_dir"].parent.name.startswith(".preprocessing.staging-")
        return msi

    def segment(*args, **kwargs):
        calls["segment"] += 1
        return segmented

    def register(*args, **kwargs):
        calls["register"] += 1
        return registered

    def quantify(*args, **kwargs):
        calls["quantify"] += 1
        return quantified

    monkeypatch.setattr("joint.pipeline.preprocess_msi", preprocess)
    monkeypatch.setattr("joint.pipeline._segment_from_config", segment)
    monkeypatch.setattr("joint.pipeline.register_laser_points", register)
    monkeypatch.setattr("joint.pipeline.quantify_cells", quantify)

    pipeline_obj.run_spatial_core(resume=False, overwrite=False)
    resumed = pipeline_obj.run_spatial_core(resume=True, overwrite=False)

    assert calls == {"preprocess": 1, "segment": 1, "register": 1, "quantify": 1}
    assert resumed.report == {"included_cells": 0, "quantified_cells": 0}


def test_simulated_segmentation_signature_includes_persisted_msi(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    config.laser_segmentation.method = "simulated"
    pipeline_obj = JointPipeline(config)
    msi = anndata.read_h5ad(config.input.h5ad)
    result = SegmentationResult(
        np.array([[1]], dtype=np.int32),
        pd.DataFrame(
            {
                "label": [1],
                "area": [1.0],
                "centroid-0": [1.0],
                "centroid-1": [1.0],
                "seg_label": ["1"],
                "morphology": ["simulated"],
                "row_number": [1],
                "column_number": [1],
                "right_to_left": [-1],
            }
        ),
        {"method": "simulated"},
    )
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    monkeypatch.setattr("joint.pipeline._segment_from_config", lambda *args, **kwargs: result)

    pipeline_obj.preprocess(resume=False, overwrite=False)
    pipeline_obj.segment(msi, resume=False, overwrite=False)
    (config.project.output_dir / "preprocessing" / "msi.h5ad").write_bytes(b"tampered")

    with pytest.raises(JointError, match="overwrite"):
        pipeline_obj.segment(msi, resume=True, overwrite=False)


def test_quantify_forwards_configured_trusted_pickle(monkeypatch, tmp_path: Path):
    import pickle

    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import QuantificationResult, RegistrationResult, SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    pickle_path = tmp_path / "cells.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(np.zeros((1, 1), dtype=np.int32), handle)
    config.input.cell_segmentation = pickle_path
    config.input.trusted_pickle = True
    pipeline_obj = JointPipeline(config)
    msi = anndata.read_h5ad(config.input.h5ad)
    regions = pd.DataFrame(
        {
            "label": [1],
            "area": [1.0],
            "centroid-0": [1.0],
            "centroid-1": [1.0],
            "seg_label": ["1"],
            "morphology": ["intact"],
            "row_number": [1],
            "column_number": [1],
            "right_to_left": [-1],
        }
    )
    registered = RegistrationResult(
        msi, regions.assign(pixel_id=msi.obs_names), {"orientation": "identity"}, {}
    )
    segmented = SegmentationResult(np.array([[1]], dtype=np.int32), regions, {})
    registration_path = config.project.output_dir / "registration" / "laser.h5ad"
    registration_path.parent.mkdir(parents=True)
    msi.write_h5ad(registration_path)
    labels_path = config.project.output_dir / "segmentation" / "laser_labels.npy"
    labels_path.parent.mkdir(parents=True)
    np.save(labels_path, segmented.labels)
    captured = {}
    empty = pd.DataFrame({"cell_id": pd.Series(dtype=str)})

    def read_cells(path, *, trusted_pickle):
        captured["path"] = path
        captured["trusted_pickle"] = trusted_pickle
        return np.zeros((1, 1), dtype=np.int32)

    monkeypatch.setattr("joint.pipeline.read_segmentation", read_cells)
    monkeypatch.setattr(
        "joint.pipeline.quantify_cells",
        lambda *args, **kwargs: QuantificationResult(
            msi, empty, empty, empty, {"included_cells": 0, "quantified_cells": 0}
        ),
    )

    rebuilt = pipeline_obj.quantify(registered, segmented, resume=False, overwrite=False)

    assert captured == {"path": pickle_path, "trusted_pickle": True}
    assert rebuilt.report == {"included_cells": 0, "quantified_cells": 0}


def test_real_segmentation_resume_does_not_depend_on_persisted_msi(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    pipeline_obj = JointPipeline(config)
    msi = anndata.read_h5ad(config.input.h5ad)
    result = SegmentationResult(
        np.array([[1]], dtype=np.int32),
        pd.DataFrame(
            {
                "label": [1],
                "area": [1.0],
                "centroid-0": [1.0],
                "centroid-1": [1.0],
                "seg_label": ["1"],
                "morphology": ["intact"],
                "row_number": [1],
                "column_number": [1],
                "right_to_left": [-1],
            }
        ),
        {},
    )
    calls = []
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)

    def segment(*args, **kwargs):
        calls.append("segment")
        return result

    monkeypatch.setattr("joint.pipeline._segment_from_config", segment)

    pipeline_obj.preprocess(resume=False, overwrite=False)
    pipeline_obj.segment(msi, resume=False, overwrite=False)
    (config.project.output_dir / "preprocessing" / "msi.h5ad").write_bytes(b"tampered")
    resumed = pipeline_obj.segment(msi, resume=True, overwrite=False)

    assert calls == ["segment"]
    assert resumed.labels.tolist() == [[1]]


def test_preprocess_rejects_missing_configured_source_before_producer(monkeypatch, tmp_path: Path):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    config.input.h5ad.unlink()
    pipeline_obj = JointPipeline(config)
    monkeypatch.setattr(
        "joint.pipeline.preprocess_msi",
        lambda *args, **kwargs: pytest.fail("producer must not run"),
    )

    with pytest.raises(JointError, match="file signature"):
        pipeline_obj.preprocess(resume=False, overwrite=False)


def test_preprocess_signature_tracks_reference_matrix_file(monkeypatch, tmp_path: Path):
    import anndata

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    reference = tmp_path / "reference.csv"
    reference.write_text("100.0\n")
    config.preprocessing.matrix_removal.enabled = True
    config.preprocessing.matrix_removal.method = "reference"
    config.preprocessing.matrix_removal.reference_file = reference
    msi = anndata.read_h5ad(config.input.h5ad)
    calls = []
    monkeypatch.setattr(
        "joint.pipeline.preprocess_msi", lambda *args, **kwargs: calls.append(1) or msi
    )
    pipeline_obj = JointPipeline(config)

    pipeline_obj.preprocess(resume=False, overwrite=False)
    reference.write_text("101.0\n")

    with pytest.raises(JointError, match="overwrite"):
        pipeline_obj.preprocess(resume=True, overwrite=False)
    assert calls == [1]


def test_pipeline_construction_does_not_overwrite_resolved_config(tmp_path: Path):
    config = pipeline_config(tmp_path)
    output = config.project.output_dir
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved-config.yaml"
    resolved.write_text("user-owned-config\n")

    from joint.pipeline import JointPipeline

    with pytest.raises(JointError, match="overwrite"):
        JointPipeline(config).preprocess(resume=False, overwrite=False)
    assert resolved.read_text() == "user-owned-config\n"


def test_resolved_config_same_content_is_idempotent(tmp_path: Path):
    config = pipeline_config(tmp_path)
    from joint.pipeline import JointPipeline

    pipeline_obj = JointPipeline(config)
    assert not (config.project.output_dir / "resolved-config.yaml").exists()
    pipeline_obj._sync_resolved_config(resume=False, overwrite=False)
    first = (config.project.output_dir / "resolved-config.yaml").read_bytes()
    pipeline_obj._sync_resolved_config(resume=True, overwrite=False)
    assert (config.project.output_dir / "resolved-config.yaml").read_bytes() == first


def test_resolved_config_write_does_not_follow_preplaced_temporary_symlink(tmp_path: Path):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    output = config.project.output_dir
    output.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("do not overwrite")
    (output / ".resolved-config.yaml.tmp").symlink_to(outside)

    JointPipeline(config)._sync_resolved_config(resume=False, overwrite=False)

    assert (output / "resolved-config.yaml").is_file()
    assert outside.read_text() == "do not overwrite"


def test_segmentation_ndarray_diagnostics_roundtrip(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np

    from joint.pipeline import JointPipeline, segment_laser_marks

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    image = np.zeros((40, 40), dtype=np.float32)
    rows, columns = np.ogrid[:40, :40]
    image[(rows - 20) ** 2 + (columns - 20) ** 2 < 49] = 1.0
    np.save(config.input.laser_image, image)
    config.laser_segmentation.parameters = {
        "intensity_cutoff": 0.0,
        "top_hat_radius": 0,
        "closing_size": 1,
        "post_closing_size": 1,
        "opening_size": 1,
        "min_peak_distance": 1,
        "min_size": 1,
        "threshold": 0.5,
    }
    expected = segment_laser_marks(image, **config.laser_segmentation.parameters)
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    pipeline_obj = JointPipeline(config)
    pipeline_obj.preprocess(resume=False, overwrite=False)
    first = pipeline_obj.segment(msi, resume=False, overwrite=False)
    resumed = pipeline_obj.segment(msi, resume=True, overwrite=False)

    for key in ("processed_image", "binary_mask"):
        assert isinstance(first.diagnostics[key], np.ndarray)
        assert isinstance(resumed.diagnostics[key], np.ndarray)
        np.testing.assert_array_equal(first.diagnostics[key], expected.diagnostics[key])
        np.testing.assert_array_equal(resumed.diagnostics[key], expected.diagnostics[key])


def test_csv_reconstruction_preserves_identifier_and_schema(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    regions = pd.DataFrame(
        {
            "label": pd.Series([1], dtype="Int64"),
            "area": pd.Series([1.0], dtype="float64"),
            "centroid-0": pd.Series([1.0], dtype="float64"),
            "centroid-1": pd.Series([1.0], dtype="float64"),
            "seg_label": pd.Series(["001"], dtype="string"),
            "morphology": pd.Series(["intact"], dtype="string"),
            "row_number": pd.Series([1], dtype="Int64"),
            "column_number": pd.Series([1], dtype="Int64"),
            "right_to_left": pd.Series([True], dtype="boolean"),
        }
    )
    segmented = SegmentationResult(np.array([[1]], dtype=np.int32), regions, {})
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    monkeypatch.setattr("joint.pipeline._segment_from_config", lambda *args, **kwargs: segmented)
    pipeline_obj = JointPipeline(config)
    pipeline_obj.preprocess(resume=False, overwrite=False)
    rebuilt = pipeline_obj.segment(msi, resume=False, overwrite=False)
    resumed = pipeline_obj.segment(msi, resume=True, overwrite=False)
    for candidate in (rebuilt.regions, resumed.regions):
        assert list(candidate.columns) == list(regions.columns)
        assert candidate.loc[0, "seg_label"] == "001"
        assert str(candidate["seg_label"].dtype) == "string"
        assert str(candidate["label"].dtype) == "Int64"
        assert str(candidate["right_to_left"].dtype) == "boolean"


def test_quantification_signature_tracks_trusted_pickle_policy(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import QuantificationResult, RegistrationResult, SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    regions = pd.DataFrame(
        {
            "label": [1],
            "area": [1.0],
            "centroid-0": [1.0],
            "centroid-1": [1.0],
            "seg_label": ["1"],
            "morphology": ["intact"],
            "row_number": [1],
            "column_number": [1],
            "right_to_left": [-1],
        }
    )
    segmented = SegmentationResult(np.array([[1]], dtype=np.int32), regions, {})
    registered = RegistrationResult(
        msi, regions.assign(pixel_id=msi.obs_names), {"orientation": "identity"}, {}
    )
    empty = pd.DataFrame({"cell_id": pd.Series(dtype=str)})
    quantified = QuantificationResult(
        msi, empty, empty, empty, {"included_cells": 0, "quantified_cells": 0}
    )
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    monkeypatch.setattr("joint.pipeline._segment_from_config", lambda *args, **kwargs: segmented)
    monkeypatch.setattr("joint.pipeline.register_laser_points", lambda *args, **kwargs: registered)
    monkeypatch.setattr(
        "joint.pipeline.read_segmentation", lambda *args, **kwargs: np.zeros((1, 1), dtype=np.int32)
    )
    monkeypatch.setattr("joint.pipeline.quantify_cells", lambda *args, **kwargs: quantified)
    pipeline_obj = JointPipeline(config)
    pipeline_obj.run_spatial_core(resume=False, overwrite=False)
    config.input.trusted_pickle = True

    with pytest.raises(JointError, match="overwrite"):
        pipeline_obj.quantify(registered, segmented, resume=True, overwrite=False)


def test_simulated_segmentation_does_not_require_merge_rules(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    config.laser_segmentation.method = "simulated"
    config.laser_segmentation.merge_rules = tmp_path / "missing-rules.csv"
    msi = anndata.read_h5ad(config.input.h5ad)
    result = SegmentationResult(np.array([[1]], dtype=np.int32), pd.DataFrame(), {})
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    monkeypatch.setattr("joint.pipeline._segment_from_config", lambda *args, **kwargs: result)
    pipeline_obj = JointPipeline(config)
    pipeline_obj.preprocess(resume=False, overwrite=False)
    assert pipeline_obj.segment(msi, resume=False, overwrite=False).labels.tolist() == [[1]]


def test_segment_from_config_real_and_simulated_flows(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import SegmentationResult
    from joint.pipeline import _segment_from_config

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    regions = pd.DataFrame(
        {
            "label": [1],
            "area": [1.0],
            "centroid-0": [1.0],
            "centroid-1": [1.0],
            "seg_label": ["1"],
            "morphology": ["intact"],
            "row_number": [1],
            "column_number": [1],
            "right_to_left": [-1],
        }
    )
    raw = SegmentationResult(
        np.array([[1]], dtype=np.int32),
        regions,
        {"processed_image": np.ones((1, 1), dtype=np.float32)},
    )
    monkeypatch.setattr("joint.pipeline.segment_laser_marks", lambda *args, **kwargs: raw)
    monkeypatch.setattr("joint.pipeline.detect_rows", lambda *args, **kwargs: [regions])
    real = _segment_from_config(config, msi)
    assert real.regions.loc[0, "seg_label"] == "1"

    config.laser_segmentation.method = "simulated"
    simulated = SegmentationResult(
        np.array([[2]], dtype=np.int32), regions, {"method": "simulated"}
    )
    monkeypatch.setattr("joint.pipeline.simulate_laser_marks", lambda *args, **kwargs: simulated)
    result = _segment_from_config(config, msi)
    assert result.diagnostics == {"method": "simulated"}


def test_resolved_config_restores_missing_root_only_with_explicit_policy(tmp_path: Path):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    pipeline_obj = JointPipeline(config)
    resolved = pipeline_obj._sync_resolved_config(resume=False, overwrite=False)
    expected = resolved.read_bytes()
    resolved.unlink()
    pipeline_obj._sync_resolved_config(resume=True, overwrite=False)
    assert resolved.read_bytes() == expected
    resolved.write_text("corrupt\n")
    with pytest.raises(JointError, match="overwrite"):
        pipeline_obj._sync_resolved_config(resume=True, overwrite=False)
    pipeline_obj._sync_resolved_config(resume=True, overwrite=True)
    assert resolved.read_bytes() == expected


def test_resolved_config_rejects_different_config_without_overwrite(tmp_path: Path):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    first = JointPipeline(config)
    resolved = first._sync_resolved_config(resume=False, overwrite=False)
    original = resolved.read_bytes()
    different = config.model_copy(deep=True)
    different.project.name = "different"
    with pytest.raises(JointError, match="overwrite"):
        JointPipeline(different)._sync_resolved_config(resume=False, overwrite=False)
    assert resolved.read_bytes() == original


def test_typed_csv_rejects_malformed_schema_and_values(tmp_path: Path):
    import pandas as pd

    from joint.pipeline import _atomic_save_json, _atomic_save_typed_csv, _read_typed_csv

    csv_path, schema_path = _atomic_save_typed_csv(
        pd.DataFrame({"identifier": pd.Series(["001"], dtype="string")}),
        tmp_path / "table.csv",
    )
    _atomic_save_json({"format": "wrong", "columns": []}, schema_path)
    with pytest.raises(JointError, match="invalid format"):
        _read_typed_csv(csv_path, schema_path)
    _atomic_save_typed_csv(
        pd.DataFrame({"identifier": pd.Series(["001"], dtype="string")}), csv_path
    )
    csv_path.write_text("other\n001\n")
    with pytest.raises(JointError, match="columns do not match"):
        _read_typed_csv(csv_path, schema_path)


def test_typed_csv_roundtrips_arbitrary_precision_object_integers(tmp_path: Path):
    import pandas as pd

    from joint.pipeline import _atomic_save_typed_csv, _read_typed_csv

    values = [0, -17, 2**63, -(2**80), 2**100]
    expected = pd.DataFrame({"label": pd.Series(values, dtype=object)})

    csv_path, schema_path = _atomic_save_typed_csv(expected, tmp_path / "integers.csv")
    reloaded = _read_typed_csv(csv_path, schema_path)

    pd.testing.assert_frame_equal(reloaded, expected)
    assert all(type(value) is int for value in reloaded["label"])


@pytest.mark.parametrize(
    ("values", "expected_values"),
    [([], []), (["001", "text"], ["001", "text"])],
)
def test_typed_csv_preserves_empty_and_generic_object_columns(
    tmp_path: Path, values: list[str], expected_values: list[str]
):
    import pandas as pd

    from joint.pipeline import _atomic_save_typed_csv, _read_typed_csv

    expected = pd.DataFrame({"value": pd.Series(values, dtype=object)})
    csv_path, schema_path = _atomic_save_typed_csv(expected, tmp_path / "objects.csv")
    reloaded = _read_typed_csv(csv_path, schema_path)

    assert reloaded["value"].tolist() == expected_values
    assert str(reloaded["value"].dtype) == "object"


@pytest.mark.parametrize("malformed", ["1.0", "1e3", "abc"])
def test_typed_csv_rejects_malformed_object_integers(tmp_path: Path, malformed: str):
    import pandas as pd

    from joint.pipeline import _atomic_save_typed_csv, _read_typed_csv

    csv_path, schema_path = _atomic_save_typed_csv(
        pd.DataFrame({"label": pd.Series([1], dtype=object)}),
        tmp_path / "integers.csv",
    )
    csv_path.write_text(f"label\n{malformed}\n")

    with pytest.raises(JointError, match="values do not match schema"):
        _read_typed_csv(csv_path, schema_path)


def test_real_and_simulated_segment_helpers_run_with_small_arrays(tmp_path: Path):
    import anndata
    import numpy as np

    from joint.pipeline import _segment_from_config

    config = pipeline_config(tmp_path)
    image = np.zeros((80, 80), dtype=float)
    rows, columns = np.ogrid[:80, :80]
    image[(rows - 40) ** 2 + (columns - 20) ** 2 < 36] = 1
    image[(rows - 40) ** 2 + (columns - 60) ** 2 < 36] = 1
    np.save(config.input.laser_image, image)
    config.laser_segmentation.parameters = {
        "intensity_cutoff": 0.0,
        "top_hat_radius": 0,
        "closing_size": 1,
        "post_closing_size": 1,
        "opening_size": 1,
        "min_peak_distance": 1,
        "min_size": 1,
        "threshold": 0.5,
    }
    msi = anndata.AnnData(
        np.ones((2, 1)),
        obs={"dataset": ["sample", "sample"]},
        var={"mz": [100.0]},
        obsm={"spatial": np.array([[0.0, 0.0], [1.0, 0.0]])},
    )
    real = _segment_from_config(config, msi)
    assert isinstance(real.diagnostics["processed_image"], np.ndarray)
    assert len(real.regions) == 2
    config.laser_segmentation.method = "simulated"
    config.laser_segmentation.radius = 2
    simulated = _segment_from_config(config, msi)
    assert simulated.diagnostics["method"] == "simulated"
    assert len(simulated.regions) == 2


def test_pipeline_json_rejects_malformed_ndarray_payload(tmp_path: Path):
    import json

    from joint.pipeline import _read_json_mapping

    artifact = tmp_path / "diagnostics.json"
    artifact.write_text(
        json.dumps(
            {
                "format": "joint-pipeline-json-v1",
                "value": {
                    "__joint_json_type__": "mapping",
                    "items": [
                        [
                            "array",
                            {
                                "__joint_json_type__": "ndarray",
                                "dtype": "|O",
                                "shape": [1],
                                "data": "AA==",
                            },
                        ]
                    ],
                },
            }
        )
    )
    with pytest.raises(JointError, match="invalid dtype, shape, or data"):
        _read_json_mapping(artifact)


def test_registration_and_quantification_typed_csvs_roundtrip(monkeypatch, tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.models import QuantificationResult, RegistrationResult, SegmentationResult
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    regions = pd.DataFrame(
        {
            "label": [1],
            "area": [1.0],
            "centroid-0": [1.0],
            "centroid-1": [1.0],
            "seg_label": pd.Series(["001"], dtype="string"),
            "morphology": ["intact"],
            "row_number": [1],
            "column_number": [1],
            "right_to_left": [-1],
        }
    )
    segmented = SegmentationResult(np.array([[1]], dtype=np.int32), regions, {})
    mapping = pd.DataFrame(
        {
            "pixel_id": pd.Series(["001"], dtype="string"),
            "centroid-0": pd.Series([1.0], dtype="Float64"),
            "centroid-1": pd.Series([1.0], dtype="Float64"),
            "count": pd.Series([1], dtype="Int64"),
            "score": pd.Series([0.5], dtype="Float64"),
            "flag": pd.Series([True], dtype="boolean"),
        }
    )
    registered = RegistrationResult(msi, mapping, {"orientation": "identity"}, {})
    typed = pd.DataFrame(
        {
            "identifier": pd.Series(["001", pd.NA], dtype="string"),
            "count": pd.Series([1, pd.NA], dtype="Int64"),
            "score": pd.Series([0.5, pd.NA], dtype="Float64"),
            "flag": pd.Series([True, pd.NA], dtype="boolean"),
        }
    )
    empty = typed.iloc[:0].copy()
    quantified = QuantificationResult(
        msi, typed, typed.copy(), empty, {"included_cells": 0, "quantified_cells": 0}
    )
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    monkeypatch.setattr("joint.pipeline._segment_from_config", lambda *args, **kwargs: segmented)
    monkeypatch.setattr("joint.pipeline.register_laser_points", lambda *args, **kwargs: registered)
    monkeypatch.setattr(
        "joint.pipeline.read_segmentation", lambda *args, **kwargs: np.zeros((1, 1), dtype=np.int32)
    )
    monkeypatch.setattr("joint.pipeline.quantify_cells", lambda *args, **kwargs: quantified)
    pipeline_obj = JointPipeline(config)
    prepared = pipeline_obj.preprocess(resume=False, overwrite=False)
    saved_segmentation = pipeline_obj.segment(prepared, resume=False, overwrite=False)
    saved_registration = pipeline_obj.register(
        prepared, saved_segmentation, resume=False, overwrite=False
    )
    saved_quantification = pipeline_obj.quantify(
        saved_registration, saved_segmentation, resume=False, overwrite=False
    )

    pd.testing.assert_frame_equal(saved_registration.mapping, mapping)
    pd.testing.assert_frame_equal(saved_quantification.overlaps, typed)
    pd.testing.assert_frame_equal(saved_quantification.accepted, typed)
    pd.testing.assert_frame_equal(saved_quantification.rejected, empty)


def test_real_quantification_stage_roundtrips_uint64_labels(tmp_path: Path):
    import anndata
    import numpy as np
    import pandas as pd

    from joint.io import atomic_write_h5ad
    from joint.models import RegistrationResult, SegmentationResult
    from joint.pipeline import JointPipeline, _atomic_save_npy

    label = np.uint64(np.iinfo(np.int64).max + 1)
    config = pipeline_config(tmp_path)
    np.save(config.input.cell_segmentation, np.array([[label]], dtype=np.uint64))
    laser = anndata.AnnData(
        np.ones((1, 1)),
        obs={
            "seg_label": [str(label)],
            "area": [1.0],
            "centroid-0": [0.0],
            "centroid-1": [0.0],
        },
        var={"mz": [100.0]},
    )
    laser.obs_names = ["p1"]
    registered = RegistrationResult(laser, pd.DataFrame(), {}, {})
    segmented = SegmentationResult(np.array([[label]], dtype=np.uint64), pd.DataFrame(), {})
    atomic_write_h5ad(laser, config.project.output_dir / "registration" / "laser.h5ad")
    _atomic_save_npy(
        segmented.labels,
        config.project.output_dir / "segmentation" / "laser_labels.npy",
    )
    pipeline_obj = JointPipeline(config)

    first = pipeline_obj.quantify(registered, segmented, resume=False, overwrite=False)
    resumed = pipeline_obj.quantify(registered, segmented, resume=True, overwrite=False)

    for result in (first, resumed):
        for table in (result.overlaps, result.accepted):
            assert table["cell_label"].tolist() == [int(label)]
            assert table["laser_label"].tolist() == [int(label)]
            assert str(table["cell_label"].dtype) == "object"
            assert str(table["laser_label"].dtype) == "object"
        assert result.rejected.empty
        assert str(result.rejected["cell_label"].dtype) == "object"
        assert str(result.rejected["laser_label"].dtype) == "object"


def test_pipeline_construction_does_not_create_output_root(tmp_path: Path):
    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    assert not config.project.output_dir.exists()
    JointPipeline(config)
    assert not config.project.output_dir.exists()


def test_first_real_stage_call_creates_output_root(monkeypatch, tmp_path: Path):
    import anndata

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    pipeline_obj = JointPipeline(config)
    assert not config.project.output_dir.exists()
    pipeline_obj.preprocess(resume=False, overwrite=False)
    assert (config.project.output_dir / "preprocessing" / "msi.h5ad").is_file()


def test_stage_failure_restores_previous_resolved_config(monkeypatch, tmp_path: Path):
    import anndata

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    first = JointPipeline(config)
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    first.preprocess(resume=False, overwrite=False)
    previous = (config.project.output_dir / "resolved-config.yaml").read_bytes()
    different = config.model_copy(deep=True)
    different.project.name = "changed"
    monkeypatch.setattr(
        "joint.pipeline.preprocess_msi",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("producer failed")),
    )

    with pytest.raises(JointError, match="producer failed"):
        JointPipeline(different).preprocess(resume=False, overwrite=True)
    assert (config.project.output_dir / "resolved-config.yaml").read_bytes() == previous


def test_manifest_failure_restores_previous_resolved_config(monkeypatch, tmp_path: Path):
    import anndata

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    first = JointPipeline(config)
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    first.preprocess(resume=False, overwrite=False)
    resolved = config.project.output_dir / "resolved-config.yaml"
    previous = resolved.read_bytes()
    previous_manifest = (config.project.output_dir / "manifest.json").read_bytes()
    different = config.model_copy(deep=True)
    different.project.name = "changed"
    monkeypatch.setattr(
        "joint.pipeline._atomic_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("manifest failed")),
    )

    with pytest.raises(JointError, match="manifest failed"):
        JointPipeline(different).preprocess(resume=False, overwrite=True)
    assert resolved.read_bytes() == previous
    assert (config.project.output_dir / "manifest.json").read_bytes() == previous_manifest


def test_resume_with_corrupt_resolved_config_requires_overwrite(monkeypatch, tmp_path: Path):
    import anndata

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    pipeline_obj = JointPipeline(config)
    pipeline_obj.preprocess(resume=False, overwrite=False)
    resolved = config.project.output_dir / "resolved-config.yaml"
    expected = resolved.read_bytes()
    resolved.write_text("corrupt\n")
    with pytest.raises(JointError, match="overwrite"):
        pipeline_obj.preprocess(resume=True, overwrite=False)
    pipeline_obj.preprocess(resume=True, overwrite=True)
    assert resolved.read_bytes() == expected


def test_json_ndarray_overflow_is_domain_error(tmp_path: Path):
    import json

    from joint.pipeline import _read_json_mapping

    artifact = tmp_path / "overflow.json"
    artifact.write_text(
        json.dumps(
            {
                "format": "joint-pipeline-json-v1",
                "value": {
                    "__joint_json_type__": "mapping",
                    "items": [
                        [
                            "array",
                            {
                                "__joint_json_type__": "ndarray",
                                "dtype": "|u1",
                                "shape": [10**100],
                                "data": "",
                            },
                        ]
                    ],
                },
            }
        )
    )
    with pytest.raises(JointError, match="invalid dtype, shape, or data"):
        _read_json_mapping(artifact)


@pytest.mark.parametrize("dtype,value", [("<i8", 10**1000), ("<f2", 1e100)])
def test_json_scalar_overflow_is_domain_error(tmp_path: Path, dtype: str, value: object):
    import json

    from joint.pipeline import _read_json_mapping

    artifact = tmp_path / "scalar-overflow.json"
    artifact.write_text(
        json.dumps(
            {
                "format": "joint-pipeline-json-v1",
                "value": {
                    "__joint_json_type__": "mapping",
                    "items": [
                        [
                            "value",
                            {
                                "__joint_json_type__": "scalar",
                                "dtype": dtype,
                                "value": value,
                            },
                        ]
                    ],
                },
            }
        )
    )
    with pytest.raises(JointError, match="invalid dtype or value"):
        _read_json_mapping(artifact)


def test_non_utf8_resolved_snapshot_restored_after_producer_failure(monkeypatch, tmp_path: Path):
    import anndata

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    pipeline_obj = JointPipeline(config)
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    pipeline_obj.preprocess(resume=False, overwrite=False)
    resolved = config.project.output_dir / "resolved-config.yaml"
    original = b"\xffcorrupt"
    resolved.write_bytes(original)
    different = config.model_copy(deep=True)
    different.project.name = "changed"
    monkeypatch.setattr(
        "joint.pipeline.preprocess_msi",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("producer failed")),
    )

    with pytest.raises(JointError, match="producer failed"):
        JointPipeline(different).preprocess(resume=False, overwrite=True)
    assert resolved.read_bytes() == original
    assert resolved.is_file() and not resolved.is_symlink()


def test_non_utf8_resolved_snapshot_restored_after_manifest_failure(monkeypatch, tmp_path: Path):
    import anndata

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    pipeline_obj = JointPipeline(config)
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    pipeline_obj.preprocess(resume=False, overwrite=False)
    resolved = config.project.output_dir / "resolved-config.yaml"
    original = b"\xffcorrupt"
    resolved.write_bytes(original)
    different = config.model_copy(deep=True)
    different.project.name = "changed"
    monkeypatch.setattr(
        "joint.pipeline._atomic_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("manifest failed")),
    )
    with pytest.raises(JointError, match="manifest failed"):
        JointPipeline(different).preprocess(resume=False, overwrite=True)
    assert resolved.read_bytes() == original
    assert resolved.is_file() and not resolved.is_symlink()


def test_snapshot_restore_failure_keeps_primary_error_note(monkeypatch, tmp_path: Path):
    import anndata

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    pipeline_obj = JointPipeline(config)
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    pipeline_obj.preprocess(resume=False, overwrite=False)
    resolved = config.project.output_dir / "resolved-config.yaml"
    resolved.write_bytes(b"\xffcorrupt")
    different = config.model_copy(deep=True)
    different.project.name = "changed"
    monkeypatch.setattr(
        "joint.pipeline.preprocess_msi",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("producer failed")),
    )
    monkeypatch.setattr(
        "joint.pipeline._atomic_save_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("restore failed")),
    )
    with pytest.raises(JointError, match="producer failed") as caught:
        JointPipeline(different).preprocess(resume=False, overwrite=True)
    assert any("restore failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize(
    "restore_error", [KeyboardInterrupt("restore interrupt"), SystemExit("restore exit")]
)
def test_base_exception_restore_failure_keeps_primary_error(
    monkeypatch, tmp_path: Path, restore_error: BaseException
):
    import anndata

    from joint.pipeline import JointPipeline

    config = pipeline_config(tmp_path)
    msi = anndata.read_h5ad(config.input.h5ad)
    pipeline_obj = JointPipeline(config)
    monkeypatch.setattr("joint.pipeline.preprocess_msi", lambda *args, **kwargs: msi)
    pipeline_obj.preprocess(resume=False, overwrite=False)
    (config.project.output_dir / "resolved-config.yaml").write_bytes(b"\xffcorrupt")
    different = config.model_copy(deep=True)
    different.project.name = "changed"
    monkeypatch.setattr(
        "joint.pipeline.preprocess_msi",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("producer failed")),
    )
    monkeypatch.setattr(
        "joint.pipeline._atomic_save_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(restore_error),
    )

    with pytest.raises(JointError, match="producer failed") as caught:
        JointPipeline(different).preprocess(resume=False, overwrite=True)
    assert any("resolved configuration restore failed" in note for note in caught.value.__notes__)
