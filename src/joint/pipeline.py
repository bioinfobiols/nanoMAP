import base64
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from skimage import io as skio

from joint.analysis import (
    cluster_cells,
    load_cnmf_results,
    prepare_cnmf,
    run_cnmf,
    run_cosg,
)
from joint.annotation import annotate_hmdb, annotate_metaboscape
from joint.config import JointConfig, load_config
from joint.errors import InputFormatError, JointError, OptionalDependencyError
from joint.io import atomic_write_h5ad, read_h5ad, read_segmentation
from joint.logging import configure_logging
from joint.models import QuantificationResult, RegistrationResult, SegmentationResult
from joint.pl import plot_registration, plot_segmentation
from joint.pp import preprocess_msi
from joint.qc import coverage_report, detect_outliers, spectral_qc
from joint.quantification import quantify_cells
from joint.registration import register_laser_points
from joint.segmentation import (
    detect_rows,
    exclude_regions,
    load_merge_rules,
    merge_regions,
    segment_laser_marks,
    simulate_laser_marks,
)
from joint.trajectory import calculate_feature_trends, fit_spatial_trajectory

DirectoryIdentity = tuple[Path, int, int]
_TRANSACTION_MARKER_PREFIX = ".joint-transaction-"


def _is_safe_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
    )


def _is_reserved_stage_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(".")
        and any(
            marker in value
            for marker in (
                ".backup-",
                ".staging-",
                ".quarantine-",
                ".rollback-",
                _TRANSACTION_MARKER_PREFIX,
            )
        )
    )


def file_signature(path: str | Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    try:
        source = Path(path).resolve()
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        stat = source.stat()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise JointError(f"Could not compute file signature for {path!s}: {exc}") from exc
    return {
        "path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_run_metadata(config: JointConfig) -> dict[str, Any]:
    """Build reproducibility metadata for a pipeline run."""
    try:
        joint_version = version("joint-msi")
    except PackageNotFoundError:
        joint_version = "0.1.0+editable"
    candidates: list[Path | None] = [
        config.input.imzml,
        config.input.h5ad,
        config.input.laser_image,
        config.input.cell_segmentation,
        config.annotation.hmdb_reference,
        config.annotation.metaboscape_reference,
        config.trajectory.path_file,
    ]
    if config.laser_segmentation.method == "real":
        candidates.append(config.laser_segmentation.merge_rules)
    if (
        config.preprocessing.matrix_removal.enabled
        and config.preprocessing.matrix_removal.method == "reference"
    ):
        candidates.append(config.preprocessing.matrix_removal.reference_file)
    if config.input.imzml is not None:
        candidates.append(config.input.imzml.with_suffix(".ibd"))
    inputs: list[dict[str, Any]] = []
    for path in (item for item in candidates if item is not None):
        if not path.is_file():
            raise InputFormatError(f"Configured input does not exist: {path}")
        inputs.append(file_signature(path))
    dependency_versions: dict[str, str] = {}
    for package in (
        "anndata", "numpy", "pandas", "pydantic", "pyimzml", "pyyaml", "scipy",
        "scikit-image", "scanpy", "cosg", "cnmf", "pygam",
    ):
        try:
            dependency_versions[package] = version(package)
        except PackageNotFoundError:
            continue
    cardinal_version = None
    rscript = shutil.which("Rscript")
    if config.preprocessing.backend == "cardinal" and rscript is not None:
        try:
            completed = subprocess.run(
                [rscript, "-e", 'cat(as.character(getRversion()), "|", as.character(packageVersion("Cardinal")))'],
                check=True,
                capture_output=True,
                text=True,
            )
            cardinal_version = completed.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            cardinal_probe_error = str(exc)
        else:
            cardinal_probe_error = None
    else:
        cardinal_probe_error = None
    return {
        "joint_version": joint_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": dependency_versions,
        "r_cardinal_version": cardinal_version,
        "r_cardinal_probe_error": cardinal_probe_error,
        "random_seed": config.project.random_seed,
        "configuration": config.model_dump(mode="json"),
        "inputs": inputs,
        "created_at": _utc_now(),
    }


def stage_signature(
    configuration: JointConfig | Mapping[str, Any], inputs: Iterable[str | Path]
) -> str:
    configuration_payload = (
        configuration.model_dump(mode="json")
        if isinstance(configuration, JointConfig)
        else configuration
    )
    payload = {
        "configuration": configuration_payload,
        "inputs": [file_signature(path) for path in inputs],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    primary_error: BaseException | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"manifest temporary cleanup failed: {cleanup_error}")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _directory_identity(path: Path) -> DirectoryIdentity:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointError(f"Stage staging directory is unsafe: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise JointError("Stage staging directory must remain a non-symlink directory")
    return resolved, metadata.st_dev, metadata.st_ino


def _validate_directory_identity(path: Path, expected: DirectoryIdentity) -> None:
    if _directory_identity(path) != expected:
        raise JointError("Stage staging directory identity changed during production")


def _best_effort_remove(path: Path, action: str, *, preserve_primary: bool = False) -> list[str]:
    try:
        _remove_path(path)
    except BaseException as exc:
        if not preserve_primary and not isinstance(exc, Exception):
            raise
        return [f"{action} failed: {exc}"]
    return []


def _restore_file_bytes(payload: bytes, destination: Path) -> list[str]:
    descriptor: int | None = None
    temporary: Path | None = None
    primary_error: BaseException | None = None
    notes: list[str] = []
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.rollback-", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException as exc:  # noqa: BLE001 - preserve the primary failure
        primary_error = exc
        notes.append(f"manifest rollback failed: {exc}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                cleanup_note = f"manifest rollback temporary cleanup failed: {cleanup_error}"
                primary_error.add_note(cleanup_note)
                notes.append(cleanup_note)
    return notes


def _add_cleanup_notes(error: BaseException, notes: Iterable[str]) -> None:
    for note in notes:
        error.add_note(note)


def _transaction_marker_path(path: Path) -> Path:
    return path.with_name(f"{_TRANSACTION_MARKER_PREFIX}{path.name}")


def _create_transaction_marker(path: Path, stage: str) -> None:
    marker = _transaction_marker_path(path)
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(f"{stage}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _has_transaction_marker(path: Path, stage: str) -> bool:
    marker = _transaction_marker_path(path)
    try:
        return not marker.is_symlink() and marker.is_file() and marker.read_text() == f"{stage}\n"
    except (OSError, UnicodeError):
        return False


def _remove_transaction_marker(path: Path, *, preserve_primary: bool = False) -> list[str]:
    marker = _transaction_marker_path(path)
    if not _path_exists(marker):
        return []
    return _best_effort_remove(
        marker, "transaction marker cleanup", preserve_primary=preserve_primary
    )


@dataclass
class _StageTransaction:
    output_dir: Path
    stage: str
    stage_dir: Path
    had_existing_stage: bool
    manifest_path: Path
    original_manifest: bytes | None
    backup_dir: Path | None = None
    backup_contains_old: bool = False
    staging_dir: Path | None = None
    cleanup_notes: list[str] = field(default_factory=list)

    def reserve_backup(self) -> None:
        backup = Path(tempfile.mkdtemp(prefix=f".{self.stage}.backup-", dir=self.output_dir))
        try:
            backup.rmdir()
        except BaseException:
            self.cleanup_notes.extend(
                _best_effort_remove(backup, "backup placeholder cleanup", preserve_primary=True)
            )
            raise
        self.backup_dir = backup
        os.replace(self.stage_dir, backup)
        _create_transaction_marker(backup, self.stage)
        self.backup_contains_old = True

    def create_staging(self) -> DirectoryIdentity:
        self.staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{self.stage}.staging-", dir=self.output_dir)
        )
        _create_transaction_marker(self.staging_dir, self.stage)
        return _directory_identity(self.staging_dir)

    def rollback(self) -> list[str]:
        notes: list[str] = list(self.cleanup_notes)
        backup_available = (
            self.backup_dir is not None
            and _path_exists(self.backup_dir)
            and (self.backup_contains_old or not _path_exists(self.stage_dir))
        )
        should_remove_final = (backup_available or not self.had_existing_stage) and _path_exists(
            self.stage_dir
        )
        if should_remove_final:
            notes.extend(
                _best_effort_remove(
                    self.stage_dir, "uncommitted final cleanup", preserve_primary=True
                )
            )
        if backup_available:
            if _path_exists(self.stage_dir):
                notes.append(
                    "old stage restore skipped because the final stage could not be cleared"
                )
            else:
                try:
                    assert self.backup_dir is not None
                    notes.extend(_remove_transaction_marker(self.backup_dir, preserve_primary=True))
                    os.replace(self.backup_dir, self.stage_dir)
                except BaseException as exc:  # noqa: BLE001 - preserve the primary failure
                    notes.append(f"old stage restore failed: {exc}")
        elif self.backup_dir is not None and _path_exists(self.backup_dir):
            notes.extend(
                _best_effort_remove(
                    self.backup_dir, "backup placeholder cleanup", preserve_primary=True
                )
            )
        if self.staging_dir is not None:
            if _path_exists(self.staging_dir):
                notes.extend(
                    _best_effort_remove(self.staging_dir, "staging cleanup", preserve_primary=True)
                )
            if not _path_exists(self.staging_dir):
                notes.extend(_remove_transaction_marker(self.staging_dir, preserve_primary=True))
        try:
            if self.original_manifest is None:
                if _path_exists(self.manifest_path):
                    notes.extend(
                        _best_effort_remove(
                            self.manifest_path, "manifest rollback", preserve_primary=True
                        )
                    )
            elif self.original_manifest is not None:
                if not _path_exists(self.manifest_path):
                    notes.extend(_restore_file_bytes(self.original_manifest, self.manifest_path))
                else:
                    metadata = self.manifest_path.lstat()
                    unchanged_regular = (
                        stat.S_ISREG(metadata.st_mode)
                        and self.manifest_path.read_bytes() == self.original_manifest
                    )
                    if stat.S_ISDIR(metadata.st_mode):
                        notes.extend(
                            _best_effort_remove(
                                self.manifest_path, "manifest rollback", preserve_primary=True
                            )
                        )
                        if not _path_exists(self.manifest_path):
                            notes.extend(
                                _restore_file_bytes(self.original_manifest, self.manifest_path)
                            )
                    elif not unchanged_regular:
                        notes.extend(
                            _restore_file_bytes(self.original_manifest, self.manifest_path)
                        )
        except BaseException as exc:  # noqa: BLE001 - preserve the primary failure
            notes.append(f"manifest rollback failed: {exc}")
        self.cleanup_notes = notes
        return notes

    def discard_backup(self) -> None:
        if self.backup_dir is not None and _path_exists(self.backup_dir):
            self.cleanup_notes.extend(
                _best_effort_remove(self.backup_dir, "post-commit backup cleanup")
            )
            if not _path_exists(self.backup_dir):
                self.cleanup_notes.extend(_remove_transaction_marker(self.backup_dir))


class StageStore:
    def __init__(self, output_dir: str | Path, logger=None):
        self.output_dir = Path(output_dir).resolve()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise JointError(f"Could not create stage output directory: {exc}") from exc
        self.manifest_path = self.output_dir / "manifest.json"
        self.logger = logger

    def _load(self) -> dict[str, Any]:
        if not _path_exists(self.manifest_path):
            return {"stages": {}}
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise JointError("Stage manifest must be a regular file")
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise JointError(f"Could not read stage manifest: {exc}") from exc
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("stages"), dict)
            or any(not isinstance(record, dict) for record in manifest["stages"].values())
        ):
            raise JointError("Stage manifest has an invalid structure")
        if any(
            not _is_safe_name(stage)
            or _is_reserved_stage_name(stage)
            or stage == self.manifest_path.name
            for stage in manifest["stages"]
        ):
            raise JointError("Stage manifest contains an unsafe stage name")
        for stage, record in manifest["stages"].items():
            self._validate_manifest_record(stage, record)
        return manifest

    def initialize(self, metadata: dict[str, Any]) -> None:
        manifest = self._load()
        previous_run = manifest.get("run")
        if isinstance(previous_run, Mapping) and "created_at" in previous_run:
            metadata = dict(metadata)
            metadata["created_at"] = previous_run["created_at"]
            metadata["last_initialized_at"] = _utc_now()
        manifest["run"] = metadata
        manifest.setdefault("stages", {})
        _atomic_json(manifest, self.manifest_path)

    def _validate_manifest_record(self, stage: str, record: Mapping[str, Any]) -> None:
        status = record.get("status")
        if not isinstance(status, str):
            raise JointError(f"Stage manifest has an invalid record for {stage}")
        if status != "complete":
            return
        signature = record.get("signature")
        recorded_artifacts = record.get("artifacts")
        if not isinstance(signature, str) or not isinstance(recorded_artifacts, dict):
            raise JointError(f"Stage manifest has an invalid complete record for {stage}")
        if not recorded_artifacts:
            raise JointError(f"Stage manifest has no artifacts for complete stage {stage}")
        stage_dir = self.output_dir / stage
        for name, details in recorded_artifacts.items():
            if not _is_safe_name(name) or not isinstance(details, dict):
                raise JointError(f"Stage manifest has an invalid artifact name for {stage}")
            raw_path = details.get("path")
            recorded_signature = details.get("signature")
            if not isinstance(raw_path, str) or not isinstance(recorded_signature, dict):
                raise JointError(f"Stage manifest has an invalid artifact record for {stage}")
            try:
                path = Path(raw_path)
                resolved = path.resolve()
                stage_resolved = stage_dir.resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                raise JointError(f"Stage manifest artifact path is unsafe for {stage}") from exc
            if (
                not path.is_absolute()
                or path != resolved
                or stage_dir.is_symlink()
                or resolved == stage_resolved
                or not resolved.is_relative_to(stage_resolved)
            ):
                raise JointError(f"Stage manifest artifact path is unsafe for {stage}")
            size = recorded_signature.get("size")
            sha256 = recorded_signature.get("sha256")
            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise JointError(f"Stage manifest has an invalid artifact signature for {stage}")
        aliases = record.get("artifact_signatures")
        if aliases is not None:
            expected = {name: details["signature"] for name, details in recorded_artifacts.items()}
            if aliases != expected:
                raise JointError(f"Stage manifest artifact_signatures mismatch for {stage}")

    def _resume_artifacts(self, stage: str, record: Mapping[str, Any]) -> dict[str, Path] | None:
        stage_dir = self.output_dir / stage
        if stage_dir.is_symlink():
            raise JointError(f"Stage manifest artifact path is unsafe for {stage}")
        try:
            recorded_artifacts = record["artifacts"]
            if not isinstance(recorded_artifacts, Mapping) or not recorded_artifacts:
                raise TypeError("artifacts must be a non-empty mapping")
            artifacts: dict[str, Path] = {}
            for name, details in recorded_artifacts.items():
                if not _is_safe_name(name) or not isinstance(details, Mapping):
                    raise TypeError("invalid artifact record")
                raw_path = details["path"]
                if not isinstance(raw_path, str):
                    raise TypeError("artifact path must be a string")
                path = Path(raw_path)
                resolved = path.resolve()
                stage_resolved = stage_dir.resolve()
                if (
                    not path.is_absolute()
                    or path != resolved
                    or path.is_symlink()
                    or resolved == stage_resolved
                    or not resolved.is_relative_to(stage_resolved)
                ):
                    raise JointError(f"Stage manifest artifact path is unsafe for {stage}")
                artifacts[name] = resolved
                recorded_signature = details["signature"]
                if not isinstance(recorded_signature, Mapping):
                    raise TypeError("artifact signature must be a mapping")
                if not resolved.is_file():
                    return None
                current_signature = file_signature(resolved)
                if (
                    current_signature["size"] != recorded_signature["size"]
                    or current_signature["sha256"] != recorded_signature["sha256"]
                ):
                    return None
        except JointError:
            raise
        except (KeyError, TypeError, OSError, RuntimeError, ValueError) as exc:
            raise JointError(f"Stage manifest has an invalid artifact record for {stage}") from exc
        return artifacts

    def _record_matches_directory(
        self, stage: str, record: Mapping[str, Any], directory: Path
    ) -> bool:
        if directory.is_symlink() or not directory.is_dir():
            return False
        final_dir = self.output_dir / stage
        try:
            directory_resolved = directory.resolve(strict=True)
            for details in record["artifacts"].values():
                final_path = Path(details["path"])
                relative_path = final_path.relative_to(final_dir)
                candidate = directory / relative_path
                candidate_resolved = candidate.resolve(strict=True)
                if (
                    candidate.is_symlink()
                    or candidate != candidate_resolved
                    or not candidate_resolved.is_relative_to(directory_resolved)
                    or not candidate_resolved.is_file()
                ):
                    return False
                current = file_signature(candidate_resolved)
                recorded = details["signature"]
                if current["size"] != recorded["size"] or current["sha256"] != recorded["sha256"]:
                    return False
        except (JointError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            return False
        return True

    def _recover_orphans(self, stage: str, manifest: Mapping[str, Any]) -> bool:
        stage_dir = self.output_dir / stage
        try:
            siblings = list(self.output_dir.iterdir())
        except OSError as exc:
            raise JointError(f"Stage {stage} orphan scan failed: {exc}") from exc
        backups = sorted(path for path in siblings if path.name.startswith(f".{stage}.backup-"))
        staging_paths = sorted(
            path for path in siblings if path.name.startswith(f".{stage}.staging-")
        )
        record = manifest["stages"].get(stage)
        complete_record = (
            record if isinstance(record, Mapping) and record.get("status") == "complete" else None
        )

        unresolved_orphans = False
        if backups:
            final_matches = complete_record is not None and self._record_matches_directory(
                stage, complete_record, stage_dir
            )
            if final_matches:
                for backup in backups:
                    if _has_transaction_marker(backup, stage):
                        _best_effort_remove(backup, "orphan backup cleanup")
                        if not _path_exists(backup):
                            _remove_transaction_marker(backup)
                unresolved_orphans = any(_path_exists(backup) for backup in backups)
            else:
                if complete_record is not None:
                    candidates = [
                        backup
                        for backup in backups
                        if self._record_matches_directory(stage, complete_record, backup)
                    ]
                else:
                    if len(backups) == 1:
                        raise JointError(f"Stage {stage} has an untrusted orphan backup")
                    candidates = []
                if len(candidates) != 1:
                    raise JointError(f"Stage {stage} has ambiguous orphan backups")
                chosen = candidates[0]
                if _path_exists(stage_dir):
                    try:
                        _remove_path(stage_dir)
                    except BaseException as exc:
                        raise JointError(
                            f"Stage {stage} orphan recovery could not clear the final stage: {exc}"
                        ) from exc
                try:
                    _remove_transaction_marker(chosen)
                    os.replace(chosen, stage_dir)
                except BaseException as exc:
                    if not isinstance(exc, Exception):
                        raise
                    raise JointError(f"Stage {stage} orphan backup restore failed: {exc}") from exc
                for backup in backups:
                    if backup != chosen and _has_transaction_marker(backup, stage):
                        _best_effort_remove(backup, "obsolete orphan backup cleanup")
                        if not _path_exists(backup):
                            _remove_transaction_marker(backup)
            if not final_matches:
                unresolved_orphans = any(_path_exists(backup) for backup in backups)

        for staging_path in staging_paths:
            if _has_transaction_marker(staging_path, stage):
                _best_effort_remove(staging_path, "orphan staging cleanup")
                if not _path_exists(staging_path):
                    _remove_transaction_marker(staging_path)
        return unresolved_orphans

    def _validate_produced_artifacts(
        self,
        stage: str,
        produced: object,
        staging_dir: Path,
        staging_identity: DirectoryIdentity,
    ) -> dict[str, Path]:
        _validate_directory_identity(staging_dir, staging_identity)
        if not isinstance(produced, Mapping):
            raise JointError(f"Stage {stage} producer must return an artifact mapping")
        if not produced:
            raise JointError(f"Stage {stage} producer must declare at least one artifact")
        if any(not _is_safe_name(name) for name in produced):
            raise JointError(f"Stage {stage} producer returned an unsafe artifact name")
        try:
            declared_artifacts = {name: Path(path) for name, path in produced.items()}
            invalid_files = [
                path
                for path in declared_artifacts.values()
                if path.is_symlink() or (path.exists() and not path.is_file())
            ]
            if invalid_files:
                raise JointError(f"Stage {stage} artifacts must be regular files")
            staging_root = staging_identity[0]
            relative_artifacts = {
                name: path.resolve().relative_to(staging_root)
                for name, path in declared_artifacts.items()
            }
        except JointError:
            raise
        except (OSError, RuntimeError, TypeError) as exc:
            raise JointError(f"Stage {stage} producer returned an invalid artifact path") from exc
        except ValueError as exc:
            raise JointError(
                f"Stage {stage} declared an artifact outside its staging directory"
            ) from exc
        missing = [
            str(staging_dir / relative_path)
            for relative_path in relative_artifacts.values()
            if not (staging_dir / relative_path).is_file()
        ]
        if missing:
            raise JointError(f"Stage {stage} did not create expected artifacts: {missing}")
        _validate_directory_identity(staging_dir, staging_identity)
        return relative_artifacts

    def run(
        self,
        stage: str,
        signature: str,
        producer: Callable[[Path], Mapping[str, Path]],
        *,
        resume: bool,
        overwrite: bool,
    ) -> dict[str, Path]:
        if (
            not _is_safe_name(stage)
            or _is_reserved_stage_name(stage)
            or stage == self.manifest_path.name
        ):
            raise JointError(f"Unsafe stage name: {stage!r}")
        manifest = self._load()
        record = manifest["stages"].get(stage)
        matching_record = (
            resume
            and isinstance(record, Mapping)
            and record.get("status") == "complete"
            and record.get("signature") == signature
        )
        stage_dir = self.output_dir / stage
        if matching_record and _path_exists(stage_dir):
            artifacts = self._resume_artifacts(stage, record)
            if artifacts is not None:
                self._recover_orphans(stage, manifest)
                return artifacts

        stage_preexisted = _path_exists(stage_dir)
        if stage_preexisted and not overwrite and not matching_record:
            raise JointError(f"Stage {stage} already has outputs; pass overwrite=True")

        unresolved_backups = self._recover_orphans(stage, manifest)
        if matching_record:
            artifacts = self._resume_artifacts(stage, record)
            if artifacts is not None:
                return artifacts
        if unresolved_backups:
            raise JointError(
                f"Stage {stage} has an orphan backup that must be cleaned before replacement"
            )

        stage_dir = self.output_dir / stage
        stage_preexisted = _path_exists(stage_dir)
        if stage_preexisted and not overwrite:
            raise JointError(f"Stage {stage} already has outputs; pass overwrite=True")

        try:
            original_manifest = (
                self.manifest_path.read_bytes() if _path_exists(self.manifest_path) else None
            )
        except OSError as exc:
            raise JointError(f"Could not snapshot stage manifest: {exc}") from exc
        transaction = _StageTransaction(
            output_dir=self.output_dir,
            stage=stage,
            stage_dir=stage_dir,
            had_existing_stage=stage_preexisted,
            manifest_path=self.manifest_path,
            original_manifest=original_manifest,
        )
        phase = "backup"
        try:
            if stage_preexisted:
                transaction.reserve_backup()
            phase = "staging setup"
            staging_identity = transaction.create_staging()
            assert transaction.staging_dir is not None
            staging_dir = transaction.staging_dir
            started_at = _utc_now()
            if self.logger:
                self.logger.info("Starting stage %s", stage)
            phase = "producer"
            produced = producer(staging_dir)
            phase = "validation"
            relative_artifacts = self._validate_produced_artifacts(
                stage, produced, staging_dir, staging_identity
            )
            if _path_exists(stage_dir):
                raise JointError(f"Stage {stage} producer wrote directly to the final stage")
            _validate_directory_identity(staging_dir, staging_identity)
            phase = "promotion"
            os.replace(staging_dir, stage_dir)
            _remove_transaction_marker(staging_dir)
            artifacts = {
                name: (stage_dir / relative_path).resolve()
                for name, relative_path in relative_artifacts.items()
            }
            phase = "manifest update"
            manifest["stages"][stage] = {
                "status": "complete",
                "signature": signature,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "artifacts": {
                    name: {"path": str(path), "signature": file_signature(path)}
                    for name, path in artifacts.items()
                },
            }
            manifest["stages"][stage]["artifact_signatures"] = {
                name: details["signature"]
                for name, details in manifest["stages"][stage]["artifacts"].items()
            }
            _atomic_json(manifest, self.manifest_path)
            if self.logger:
                self.logger.info("Completed stage %s", stage)
        except BaseException as exc:
            cleanup_notes = transaction.rollback()
            if not isinstance(exc, Exception):
                _add_cleanup_notes(exc, cleanup_notes)
                raise
            if isinstance(exc, OptionalDependencyError) or (
                isinstance(exc, JointError) and phase == "validation"
            ):
                failure = exc
            else:
                failure = JointError(f"Stage {stage} {phase} failed: {exc}")
            _add_cleanup_notes(failure, cleanup_notes)
            # Preserve a failed record for first-run stages; an overwrite failure
            # deliberately restores the previous complete manifest transactionally.
            if not transaction.had_existing_stage:
                try:
                    failed_manifest = self._load()
                    failed_manifest["stages"][stage] = {
                        "status": "failed",
                        "signature": signature,
                        "started_at": locals().get("started_at", _utc_now()),
                        "completed_at": _utc_now(),
                        "error": str(exc),
                    }
                    _atomic_json(failed_manifest, self.manifest_path)
                except Exception as record_error:  # noqa: BLE001
                    failure.add_note(f"failed-stage manifest update failed: {record_error}")
            if self.logger:
                self.logger.exception("Stage %s failed", stage)
            if failure is exc:
                raise
            raise failure from exc
        transaction.discard_backup()
        return artifacts


def _atomic_save_npy(array: np.ndarray, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            fd = -1
            raise
        fd = -1
        with handle:
            np.save(handle, array)
        os.replace(temporary, destination)
    finally:
        if fd != -1:
            os.close(fd)
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_save_csv(table: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8", newline="")
        except BaseException:
            os.close(fd)
            fd = -1
            raise
        fd = -1
        with handle:
            table.to_csv(handle, index=False)
        os.replace(temporary, destination)
    finally:
        if fd != -1:
            os.close(fd)
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_save_text(payload: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            fd = -1
            raise
        fd = -1
        with handle:
            handle.write(payload)
        os.replace(temporary, destination)
    finally:
        if fd != -1:
            os.close(fd)
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_save_bytes(payload: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            fd = -1
            raise
        fd = -1
        with handle:
            handle.write(payload)
        os.replace(temporary, destination)
    finally:
        if fd != -1:
            os.close(fd)
        temporary.unlink(missing_ok=True)
    return destination


def _frame_schema(table: pd.DataFrame) -> dict[str, Any]:
    return {
        "format": "joint-pipeline-csv-v1",
        "columns": [
            {
                "name": str(name),
                "dtype": str(dtype),
                **(
                    {"object_kind": pd.api.types.infer_dtype(table[name], skipna=False)}
                    if str(dtype) == "object"
                    else {}
                ),
            }
            for name, dtype in zip(table.columns, table.dtypes, strict=True)
        ],
    }


def _atomic_save_typed_csv(table: pd.DataFrame, csv_path: Path) -> tuple[Path, Path]:
    _atomic_save_csv(table, csv_path)
    schema_path = csv_path.with_suffix(csv_path.suffix + ".schema.json")
    _atomic_save_json(_frame_schema(table), schema_path)
    return csv_path, schema_path


def _parse_object_integers(values: pd.Series) -> pd.Series:
    parsed: list[int] = []
    for value in values.array:
        if not isinstance(value, str):
            raise TypeError("object integer value is missing or is not text")
        digits = value[1:] if value.startswith(("+", "-")) else value
        if not digits or not digits.isascii() or not digits.isdecimal():
            raise ValueError("object integer value is not a decimal integer")
        parsed.append(int(value, 10))
    return pd.Series(parsed, index=values.index, name=values.name, dtype=object)


def _read_typed_csv(csv_path: Path, schema_path: Path) -> pd.DataFrame:
    schema = _read_json_mapping(schema_path)
    if schema.get("format") != "joint-pipeline-csv-v1" or not isinstance(
        schema.get("columns"), list
    ):
        raise JointError(f"CSV schema has an invalid format: {schema_path}")
    columns = schema["columns"]
    if any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("name"), str)
        or not isinstance(item.get("dtype"), str)
        or (item["dtype"] == "object" and not isinstance(item.get("object_kind"), str))
        for item in columns
    ):
        raise JointError(f"CSV schema has invalid column definitions: {schema_path}")
    names = [item["name"] for item in columns]
    if not names:
        try:
            if csv_path.read_bytes() not in {b"", b"\n"}:
                raise JointError(f"CSV values do not match schema: {csv_path}")
        except OSError as exc:
            raise JointError(f"Could not read CSV artifact {csv_path}: {exc}") from exc
        return pd.DataFrame()
    dtype_map: dict[str, str] = {}
    for item in columns:
        dtype = item["dtype"]
        if dtype in {"object", "string"}:
            dtype_map[item["name"]] = "string"
        else:
            dtype_map[item["name"]] = dtype
    try:
        table = pd.read_csv(csv_path, dtype=dtype_map, keep_default_na=True)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise JointError(f"Could not read CSV artifact {csv_path}: {exc}") from exc
    if list(table.columns) != names:
        raise JointError(f"CSV columns do not match schema: {csv_path}")
    try:
        for item in columns:
            name, dtype = item["name"], item["dtype"]
            if dtype == "object":
                kind = item["object_kind"]
                if kind in {"string", "empty"}:
                    table[name] = table[name].astype(object)
                elif kind == "integer":
                    table[name] = _parse_object_integers(table[name])
                elif kind == "floating":
                    table[name] = (
                        pd.to_numeric(table[name], errors="raise").astype(float).astype(object)
                    )
                elif kind == "boolean":
                    table[name] = table[name].astype("boolean").astype(object)
                else:
                    raise ValueError(f"unsupported object column kind {kind}")
            else:
                table[name] = table[name].astype(dtype)
    except (TypeError, ValueError, KeyError) as exc:
        raise JointError(f"CSV values do not match schema: {csv_path}") from exc
    return table


def _json_value(value: Any) -> Any:
    """Encode report mappings without losing their non-string keys in JSON."""
    if isinstance(value, Mapping):
        items = [[_json_value(key), _json_value(item)] for key, item in value.items()]
        return {
            "__joint_json_type__": "mapping",
            "items": sorted(
                items, key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":"))
            ),
        }
    if isinstance(value, np.generic):
        dtype = value.dtype
        if dtype.kind not in "biuf":
            raise TypeError(f"Unsupported numpy scalar dtype: {dtype}")
        return {"__joint_json_type__": "scalar", "dtype": dtype.str, "value": value.item()}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object dtype arrays are not supported in pipeline JSON")
        raw = np.ascontiguousarray(value).tobytes()
        return {
            "__joint_json_type__": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_save_json(payload: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            fd = -1
            raise
        fd = -1
        with handle:
            handle.write(
                json.dumps(
                    {"format": "joint-pipeline-json-v1", "value": _json_value(payload)},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        os.replace(temporary, destination)
    finally:
        if fd != -1:
            os.close(fd)
        temporary.unlink(missing_ok=True)
    return destination


def _restore_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__joint_json_type__", "items"}:
            if value["__joint_json_type__"] != "mapping" or not isinstance(value["items"], list):
                raise JointError("Pipeline JSON artifact has an invalid mapping encoding")
            restored: dict[Any, Any] = {}
            for item in value["items"]:
                if not isinstance(item, list) or len(item) != 2:
                    raise JointError("Pipeline JSON artifact has an invalid mapping item")
                key, mapped_value = (_restore_json(part) for part in item)
                try:
                    restored[key] = mapped_value
                except TypeError as exc:
                    raise JointError(
                        "Pipeline JSON artifact has an unhashable mapping key"
                    ) from exc
            return restored
        marker = value.get("__joint_json_type__")
        if marker == "scalar":
            if set(value) != {"__joint_json_type__", "dtype", "value"}:
                raise JointError("Pipeline JSON scalar has an invalid schema")
            try:
                dtype = np.dtype(value["dtype"])
                if dtype.kind not in "biuf" or dtype.hasobject:
                    raise TypeError
                with np.errstate(over="raise", invalid="raise"):
                    return np.asarray(value["value"], dtype=dtype)[()]
            except (TypeError, ValueError, OverflowError, MemoryError, FloatingPointError) as exc:
                raise JointError("Pipeline JSON scalar has an invalid dtype or value") from exc
        if marker == "ndarray":
            if set(value) != {"__joint_json_type__", "dtype", "shape", "data"}:
                raise JointError("Pipeline JSON ndarray has an invalid schema")
            try:
                dtype = np.dtype(value["dtype"])
                shape = tuple(value["shape"])
                if (
                    dtype.hasobject
                    or not isinstance(value["shape"], list)
                    or any(not isinstance(dim, int) or dim < 0 for dim in shape)
                ):
                    raise TypeError
                raw = base64.b64decode(value["data"], validate=True)
                expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if len(raw) != expected:
                    raise ValueError
                return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
            except (
                TypeError,
                ValueError,
                OverflowError,
                MemoryError,
                base64.binascii.Error,
            ) as exc:
                raise JointError("Pipeline JSON ndarray has invalid dtype, shape, or data") from exc
        return {key: _restore_json(item) for key, item in value.items()}
    return value


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise JointError(f"Could not read pipeline JSON artifact {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format", "value"}
        or payload["format"] != "joint-pipeline-json-v1"
    ):
        raise JointError(f"Pipeline JSON artifact has an invalid schema: {path}")
    restored = _restore_json(payload["value"])
    if not isinstance(restored, dict):
        raise JointError(f"Pipeline JSON artifact must contain a mapping: {path}")
    return restored


def _image_and_mask(config: JointConfig) -> tuple[np.ndarray, np.ndarray | None]:
    if config.input.laser_image is None:
        raise JointError("input.laser_image is required for segmentation")
    source = config.input.laser_image
    image = (
        np.load(source) if source.suffix.lower() == ".npy" else skio.imread(source, as_gray=True)
    )
    rectangle = config.laser_segmentation.parameters.get("include_rectangle")
    if rectangle is None:
        return np.asarray(image), None
    try:
        row0, col0, row1, col1 = map(int, rectangle)
    except (TypeError, ValueError) as exc:
        raise JointError(
            "laser_segmentation.parameters.include_rectangle must have four integers"
        ) from exc
    mask = np.zeros(np.asarray(image).shape[:2], dtype=bool)
    mask[row0 : row1 + 1, col0 : col1 + 1] = True
    return np.asarray(image), mask


def _segment_from_config(config: JointConfig, msi: anndata.AnnData) -> SegmentationResult:
    image, mask = _image_and_mask(config)
    parameters = dict(config.laser_segmentation.parameters)
    parameters.pop("include_rectangle", None)
    row_parameters = parameters.pop("row_detection", {})
    exclusion = parameters.pop("exclusion", {})
    segmented = segment_laser_marks(image, mask=mask, **parameters)
    rows = detect_rows(
        segmented.regions,
        expected_rows=config.laser_segmentation.expected_rows,
        **row_parameters,
    )
    if exclusion:
        rows = [exclude_regions(row, **exclusion) for row in rows]
    if config.laser_segmentation.method == "simulated":
        return simulate_laser_marks(
            msi,
            rows,
            image.shape[:2],
            radius=config.laser_segmentation.radius,
        )
    regions = pd.concat(rows, ignore_index=True)
    if config.laser_segmentation.merge_rules is not None:
        regions = merge_regions(regions, load_merge_rules(config.laser_segmentation.merge_rules))
    return SegmentationResult(segmented.labels, regions, segmented.diagnostics)


class JointPipeline:
    """Checkpointed orchestration for the JOINT spatial core stages."""

    def __init__(self, config: JointConfig):
        self.config = config
        self.output_dir = Path(config.project.output_dir).resolve()
        self.logger = None
        self.store: StageStore | None = None

    def _get_store(self) -> StageStore:
        if self.store is None:
            self.logger = configure_logging(self.output_dir)
            self.store = StageStore(self.output_dir, logger=self.logger)
            try:
                self.store.initialize(build_run_metadata(self.config))
            except Exception as exc:
                raise JointError(f"Could not initialize run provenance: {exc}") from exc
        return self.store

    @staticmethod
    def _matrix_matches(left: object, right: object) -> bool:
        if not hasattr(left, "shape") or not hasattr(right, "shape") or left.shape != right.shape:
            return False
        if sparse.issparse(left) and sparse.issparse(right):
            return bool((left != right).nnz == 0)
        left_values = left.toarray() if sparse.issparse(left) else np.asarray(left)
        right_values = right.toarray() if sparse.issparse(right) else np.asarray(right)
        return bool(np.array_equal(left_values, right_values, equal_nan=True))

    @classmethod
    def _values_match(cls, left: object, right: object) -> bool:
        if left is right:
            return True
        if left is None or right is None:
            return False
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(
                cls._values_match(left[key], right[key]) for key in left
            )
        if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
            return bool(left.equals(right))
        if isinstance(left, pd.Series) and isinstance(right, pd.Series):
            return bool(left.equals(right))
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple, np.ndarray)):
            try:
                right_values = right.tolist() if isinstance(right, np.ndarray) else right
                return len(left) == len(right_values) and all(
                    cls._values_match(left_item, right_item)
                    for left_item, right_item in zip(left, right_values, strict=True)
                )
            except (TypeError, ValueError):
                return False
        if isinstance(right, (list, tuple)) and isinstance(left, np.ndarray):
            return cls._values_match(right, left)
        if sparse.issparse(left) or sparse.issparse(right):
            return cls._matrix_matches(left, right)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                left_array = np.asarray(left)
                right_array = np.asarray(right)
                if left_array.dtype.names is not None or right_array.dtype.names is not None:
                    if (
                        left_array.shape != right_array.shape
                        or left_array.dtype.names != right_array.dtype.names
                    ):
                        return False
                    assert left_array.dtype.names is not None
                    return all(
                        cls._values_match(left_array[name], right_array[name])
                        for name in left_array.dtype.names
                    )
                try:
                    return bool(np.array_equal(left_array, right_array, equal_nan=True))
                except TypeError:
                    return bool(np.array_equal(left_array, right_array))
            except (TypeError, ValueError):
                return False
        try:
            if pd.isna(left) and pd.isna(right):
                return True
        except (TypeError, ValueError):
            pass
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    def _persisted_stage_input(self, caller: anndata.AnnData, input_path: Path) -> anndata.AnnData:
        if not isinstance(caller, anndata.AnnData):
            raise JointError("Downstream stage input must be an AnnData object")
        persisted = read_h5ad(input_path)
        matches = (
            caller.shape == persisted.shape
            and caller.obs_names.equals(persisted.obs_names)
            and caller.var_names.equals(persisted.var_names)
            and self._matrix_matches(caller.X, persisted.X)
            and caller.obs.equals(persisted.obs)
            and caller.var.equals(persisted.var)
            and set(caller.layers) == set(persisted.layers)
            and all(
                self._matrix_matches(caller.layers[name], persisted.layers[name])
                for name in caller.layers
            )
            and set(caller.obsm) == set(persisted.obsm)
            and all(
                self._matrix_matches(caller.obsm[name], persisted.obsm[name])
                for name in caller.obsm
            )
            and self._values_match(caller.uns, persisted.uns)
            and set(caller.obsp) == set(persisted.obsp)
            and all(
                self._matrix_matches(caller.obsp[name], persisted.obsp[name])
                for name in caller.obsp
            )
            and set(caller.varm) == set(persisted.varm)
            and all(
                self._matrix_matches(caller.varm[name], persisted.varm[name])
                for name in caller.varm
            )
            and set(caller.varp) == set(persisted.varp)
            and all(
                self._matrix_matches(caller.varp[name], persisted.varp[name])
                for name in caller.varp
            )
        )
        caller_raw = caller.raw.to_adata() if caller.raw is not None else None
        persisted_raw = persisted.raw.to_adata() if persisted.raw is not None else None
        if caller_raw is None or persisted_raw is None:
            matches = matches and caller_raw is persisted_raw
        else:
            matches = matches and (
                caller_raw.shape == persisted_raw.shape
                and caller_raw.obs_names.equals(persisted_raw.obs_names)
                and caller_raw.var_names.equals(persisted_raw.var_names)
                and self._matrix_matches(caller_raw.X, persisted_raw.X)
                and caller_raw.var.equals(persisted_raw.var)
                and set(caller_raw.varm) == set(persisted_raw.varm)
                and all(
                    self._values_match(caller_raw.varm[name], persisted_raw.varm[name])
                    for name in caller_raw.varm
                )
            )
        if not matches:
            raise JointError(
                f"Caller AnnData does not match persisted upstream artifact: {input_path}"
            )
        return persisted

    def _resolved_config_text(self) -> str:
        payload = self.config.model_dump(mode="json")
        return yaml.safe_dump(payload, sort_keys=False)

    def _sync_resolved_config(self, *, resume: bool, overwrite: bool) -> Path:
        del resume  # Root snapshot synchronization is content-addressed and idempotent.
        destination = self.output_dir / "resolved-config.yaml"
        expected = self._resolved_config_text()
        if destination.exists() or destination.is_symlink():
            try:
                if (
                    destination.is_file()
                    and not destination.is_symlink()
                    and destination.read_text(encoding="utf-8") == expected
                ):
                    return destination
            except (OSError, UnicodeError) as exc:
                if not overwrite:
                    raise JointError(f"Could not read resolved configuration: {exc}") from exc
            if not overwrite:
                raise JointError("resolved-config.yaml already exists; pass overwrite=True")
        try:
            _atomic_save_text(expected, destination)
        except OSError as exc:
            raise JointError(f"Could not write resolved configuration: {exc}") from exc
        return destination

    def _sync_diagnostic_figure(
        self, source: Path, name: str, *, resume: bool = True, overwrite: bool = True
    ) -> Path:
        figures_dir = self.output_dir / "figures"
        if figures_dir.is_symlink() or (figures_dir.exists() and not figures_dir.is_dir()):
            raise JointError(f"Diagnostic figures directory is unsafe: {figures_dir}")
        figures_dir.mkdir(parents=True, exist_ok=True)
        if not figures_dir.resolve().is_relative_to(self.output_dir.resolve()):
            raise JointError(f"Diagnostic figures directory is outside output root: {figures_dir}")
        destination = figures_dir / name
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            directory_fd = os.open(figures_dir, flags)
        except OSError as exc:
            raise JointError(f"Diagnostic figures directory is unsafe: {figures_dir}: {exc}") from exc
        temporary_name = f".{name}.tmp"
        try:
            try:
                destination_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                destination_stat = None
            if destination_stat is not None:
                if not stat.S_ISREG(destination_stat.st_mode):
                    raise JointError(
                        f"Diagnostic figure destination is not a regular file: {destination}"
                    )
                if not (resume or overwrite):
                    if file_signature(source)["sha256"] == file_signature(destination)["sha256"]:
                        return destination
                    raise JointError(
                        f"Diagnostic figure already exists; pass overwrite=True: {destination}"
                    )
            temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            try:
                os.stat(temporary_name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise JointError(f"Diagnostic figure temporary path is unsafe: {temporary_name}")
            temporary_fd = os.open(temporary_name, temporary_flags, 0o600, dir_fd=directory_fd)
            try:
                with source.open("rb") as src, os.fdopen(temporary_fd, "wb") as dst:
                    temporary_fd = None
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
                os.replace(
                    temporary_name,
                    name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            finally:
                if temporary_fd is not None:
                    os.close(temporary_fd)
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        except (OSError, ValueError) as exc:
            raise JointError(f"Could not synchronize diagnostic figure {destination}: {exc}") from exc
        finally:
            os.close(directory_fd)
        return destination

    def _snapshot_resolved_config(self) -> bytes | None:
        destination = self.output_dir / "resolved-config.yaml"
        if not destination.exists() and not destination.is_symlink():
            return None
        try:
            if destination.is_symlink() or not destination.is_file():
                raise JointError("resolved-config.yaml must be a regular file")
            return destination.read_bytes()
        except (OSError, UnicodeError) as exc:
            raise JointError(f"Could not snapshot resolved configuration: {exc}") from exc

    def _restore_resolved_config(self, previous: bytes | None) -> None:
        destination = self.output_dir / "resolved-config.yaml"
        try:
            if previous is None:
                if destination.exists() or destination.is_symlink():
                    _remove_path(destination)
            else:
                _atomic_save_bytes(previous, destination)
        except (OSError, UnicodeError) as exc:
            raise JointError(f"Could not restore resolved configuration: {exc}") from exc

    def _sync_final_root(self, stage_path: Path, *, resume: bool, overwrite: bool) -> Path:
        destination = self.output_dir / "joint-final.h5ad"
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file():
                if not (overwrite or resume):
                    raise JointError("joint-final.h5ad already exists; pass overwrite=True")
            else:
                try:
                    same = (
                        file_signature(destination)["sha256"]
                        == file_signature(stage_path)["sha256"]
                    )
                except JointError:
                    same = False
                if same:
                    return destination
                if not (overwrite or resume):
                    raise JointError("joint-final.h5ad already exists; pass overwrite=True")
        fd, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            try:
                with os.fdopen(fd, "wb") as destination_handle:
                    fd = -1
                    with stage_path.open("rb") as source_handle:
                        shutil.copyfileobj(source_handle, destination_handle)
            finally:
                if fd != -1:
                    os.close(fd)
            os.replace(temporary, destination)
        except OSError as exc:
            raise JointError(f"Could not synchronize joint-final.h5ad: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _preflight_final_root(self, signature: str, *, resume: bool, overwrite: bool) -> None:
        destination = self.output_dir / "joint-final.h5ad"
        if not destination.exists() and not destination.is_symlink():
            return
        if overwrite:
            return
        if resume:
            store = self._get_store()
            record = store._load()["stages"].get("final")
            if (
                isinstance(record, Mapping)
                and record.get("status") == "complete"
                and record.get("signature") == signature
                and (self.output_dir / "final").exists()
            ):
                artifacts = store._resume_artifacts("final", record)
                if artifacts is not None and "adata" in artifacts:
                    return
        raise JointError("joint-final.h5ad already exists; pass overwrite=True")

    def _run_stage(
        self,
        stage: str,
        signature: str,
        producer: Callable[[Path], Mapping[str, Path]],
        *,
        resume: bool,
        overwrite: bool,
    ) -> dict[str, Path]:
        previous_config = self._snapshot_resolved_config()
        self._sync_resolved_config(resume=resume, overwrite=overwrite)
        try:
            return self._get_store().run(
                stage, signature, producer, resume=resume, overwrite=overwrite
            )
        except BaseException as primary:
            try:
                self._restore_resolved_config(previous_config)
            except BaseException as restore_error:  # noqa: BLE001 - preserve primary failure
                primary.add_note(f"resolved configuration restore failed: {restore_error}")
            raise

    @classmethod
    def from_config(cls, path: str | Path) -> "JointPipeline":
        return cls(load_config(path))

    def preprocess(self, *, resume: bool, overwrite: bool) -> anndata.AnnData:
        source = self.config.input.imzml or self.config.input.h5ad
        if source is None:  # Defensive: JointConfig validation normally excludes this.
            raise JointError("An imzML or h5ad MSI input is required")
        source_inputs = [source]
        if source.suffix.lower() == ".imzml":
            source_inputs.append(source.with_suffix(".ibd"))
        matrix = self.config.preprocessing.matrix_removal
        if matrix.enabled and matrix.method == "reference":
            if matrix.reference_file is None:
                raise JointError("preprocessing.matrix_removal.reference_file is required")
            source_inputs.append(matrix.reference_file)
        signature = stage_signature(
            {
                "project_name": self.config.project.name,
                "preprocessing": self.config.preprocessing.model_dump(mode="json"),
            },
            source_inputs,
        )

        def producer(staging_dir: Path) -> Mapping[str, Path]:
            adata = preprocess_msi(
                source,
                config=self.config.preprocessing,
                dataset_id=self.config.project.name,
                cardinal_output_dir=staging_dir / "cardinal",
            )
            path = atomic_write_h5ad(adata, staging_dir / "msi.h5ad")
            return {"adata": path}

        artifacts = self._run_stage(
            "preprocessing", signature, producer, resume=resume, overwrite=overwrite
        )
        return read_h5ad(artifacts["adata"])

    def segment(self, msi: anndata.AnnData, *, resume: bool, overwrite: bool) -> SegmentationResult:
        inputs: list[Path] = []
        if self.config.input.laser_image is not None:
            inputs.append(self.config.input.laser_image)
        if (
            self.config.laser_segmentation.method == "real"
            and self.config.laser_segmentation.merge_rules is not None
        ):
            inputs.append(self.config.laser_segmentation.merge_rules)
        if self.config.laser_segmentation.method == "simulated":
            inputs.append(self.output_dir / "preprocessing" / "msi.h5ad")
        segmentation_configuration = self.config.laser_segmentation.model_dump(mode="json")
        if self.config.laser_segmentation.method == "simulated":
            segmentation_configuration.pop("merge_rules", None)
        signature = stage_signature(segmentation_configuration, inputs)

        def producer(staging_dir: Path) -> Mapping[str, Path]:
            result = _segment_from_config(self.config, msi)
            regions_path, regions_schema = _atomic_save_typed_csv(
                result.regions, staging_dir / "laser_regions.csv"
            )
            figure_path = staging_dir / "laser-segmentation.png"
            figure, _ = plot_segmentation(result.labels, save=figure_path)
            plt.close(figure)
            return {
                "labels": _atomic_save_npy(result.labels, staging_dir / "laser_labels.npy"),
                "regions": regions_path,
                "regions_schema": regions_schema,
                "diagnostics": _atomic_save_json(
                    result.diagnostics, staging_dir / "diagnostics.json"
                ),
                "figure": figure_path,
            }

        artifacts = self._run_stage(
            "segmentation", signature, producer, resume=resume, overwrite=overwrite
        )
        self._sync_diagnostic_figure(
            artifacts["figure"], "laser-segmentation.png", resume=resume, overwrite=overwrite
        )
        return SegmentationResult(
            labels=np.load(artifacts["labels"], allow_pickle=False),
            regions=_read_typed_csv(artifacts["regions"], artifacts["regions_schema"]),
            diagnostics=_read_json_mapping(artifacts["diagnostics"]),
        )

    def register(
        self,
        msi: anndata.AnnData,
        segmented: SegmentationResult,
        *,
        resume: bool,
        overwrite: bool,
    ) -> RegistrationResult:
        signature = stage_signature(
            self.config.registration.model_dump(mode="json"),
            [
                self.output_dir / "preprocessing" / "msi.h5ad",
                self.output_dir / "segmentation" / "laser_regions.csv",
            ],
        )

        def producer(staging_dir: Path) -> Mapping[str, Path]:
            result = register_laser_points(
                msi,
                segmented.regions,
                orientation=self.config.registration.orientation,
                edge_policy=self.config.registration.edge_policy,
            )
            mapping_path, mapping_schema = _atomic_save_typed_csv(
                result.mapping, staging_dir / "mapping.csv"
            )
            figure_path = staging_dir / "registration.png"
            figure, _ = plot_registration(result, save=figure_path)
            plt.close(figure)
            return {
                "adata": atomic_write_h5ad(result.adata, staging_dir / "laser.h5ad"),
                "mapping": mapping_path,
                "mapping_schema": mapping_schema,
                "transform": _atomic_save_json(result.transform, staging_dir / "transform.json"),
                "report": _atomic_save_json(result.report, staging_dir / "report.json"),
                "figure": figure_path,
            }

        artifacts = self._run_stage(
            "registration", signature, producer, resume=resume, overwrite=overwrite
        )
        self._sync_diagnostic_figure(
            artifacts["figure"], "registration.png", resume=resume, overwrite=overwrite
        )
        return RegistrationResult(
            read_h5ad(artifacts["adata"]),
            _read_typed_csv(artifacts["mapping"], artifacts["mapping_schema"]),
            _read_json_mapping(artifacts["transform"]),
            _read_json_mapping(artifacts["report"]),
        )

    def quantify(
        self,
        registered: RegistrationResult,
        segmented: SegmentationResult,
        *,
        resume: bool,
        overwrite: bool,
    ) -> QuantificationResult:
        if self.config.input.cell_segmentation is None:
            raise JointError("input.cell_segmentation is required for quantification")
        signature = stage_signature(
            {
                "quantification": self.config.quantification.model_dump(mode="json"),
                "trusted_pickle": self.config.input.trusted_pickle,
            },
            [
                self.output_dir / "registration" / "laser.h5ad",
                self.output_dir / "segmentation" / "laser_labels.npy",
                self.config.input.cell_segmentation,
            ],
        )

        def producer(staging_dir: Path) -> Mapping[str, Path]:
            result = quantify_cells(
                registered.adata,
                read_segmentation(
                    self.config.input.cell_segmentation,
                    trusted_pickle=self.config.input.trusted_pickle,
                ),
                segmented.labels,
                **self.config.quantification.model_dump(),
            )
            overlaps_path, overlaps_schema = _atomic_save_typed_csv(
                result.overlaps, staging_dir / "overlaps.csv"
            )
            accepted_path, accepted_schema = _atomic_save_typed_csv(
                result.accepted, staging_dir / "accepted.csv"
            )
            rejected_path, rejected_schema = _atomic_save_typed_csv(
                result.rejected, staging_dir / "rejected.csv"
            )
            coverage_path = _atomic_save_csv(
                coverage_report(result), staging_dir / "coverage.csv"
            )
            return {
                "adata": atomic_write_h5ad(result.adata, staging_dir / "cells.h5ad"),
                "overlaps": overlaps_path,
                "overlaps_schema": overlaps_schema,
                "accepted": accepted_path,
                "accepted_schema": accepted_schema,
                "rejected": rejected_path,
                "rejected_schema": rejected_schema,
                "report": _atomic_save_json(result.report, staging_dir / "report.json"),
                "coverage": coverage_path,
            }

        artifacts = self._run_stage(
            "quantification", signature, producer, resume=resume, overwrite=overwrite
        )
        return QuantificationResult(
            read_h5ad(artifacts["adata"]),
            _read_typed_csv(artifacts["overlaps"], artifacts["overlaps_schema"]),
            _read_typed_csv(artifacts["accepted"], artifacts["accepted_schema"]),
            _read_typed_csv(artifacts["rejected"], artifacts["rejected_schema"]),
            _read_json_mapping(artifacts["report"]),
        )

    def run_spatial_core(
        self, *, resume: bool = False, overwrite: bool = False
    ) -> QuantificationResult:
        msi = self.preprocess(resume=resume, overwrite=overwrite)
        segmented = self.segment(msi, resume=resume, overwrite=overwrite)
        registered = self.register(msi, segmented, resume=resume, overwrite=overwrite)
        return self.quantify(registered, segmented, resume=resume, overwrite=overwrite)

    def qc(
        self, adata: anndata.AnnData, *, resume: bool = False, overwrite: bool = False
    ) -> anndata.AnnData:
        input_path = self.output_dir / "quantification" / "cells.h5ad"
        persisted = self._persisted_stage_input(adata, input_path)
        signature = stage_signature(
            {
                "qc": self.config.qc.model_dump(mode="json"),
                "random_seed": self.config.project.random_seed,
            },
            [input_path],
        )

        def producer(staging_dir: Path) -> Mapping[str, Path]:
            report = spectral_qc(persisted).reset_index(names="obs_id")
            report_path, report_schema = _atomic_save_typed_csv(
                report, staging_dir / "spectral.csv"
            )
            result = persisted.copy()
            result.obs["outlier"] = detect_outliers(
                result,
                contamination=self.config.qc.outlier_contamination,
                random_seed=self.config.project.random_seed,
            )
            return {
                "adata": atomic_write_h5ad(result, staging_dir / "cells_qc.h5ad"),
                "spectral": report_path,
                "spectral_schema": report_schema,
            }

        artifacts = self._run_stage("qc", signature, producer, resume=resume, overwrite=overwrite)
        return read_h5ad(artifacts["adata"])

    def annotate(
        self, adata: anndata.AnnData, *, resume: bool = False, overwrite: bool = False
    ) -> anndata.AnnData:
        input_path = (
            self.output_dir / "qc" / "cells_qc.h5ad"
            if self.config.qc.enabled
            else self.output_dir / "quantification" / "cells.h5ad"
        )
        persisted = self._persisted_stage_input(adata, input_path)
        reference_paths = [
            path
            for path in (
                self.config.annotation.hmdb_reference,
                self.config.annotation.metaboscape_reference,
            )
            if path is not None
        ]
        signature = stage_signature(
            self.config.annotation.model_dump(mode="json"),
            [input_path, *reference_paths],
        )

        def producer(staging_dir: Path) -> Mapping[str, Path]:
            result = persisted.copy()
            if self.config.annotation.hmdb_reference is not None:
                result = annotate_hmdb(
                    result,
                    self.config.annotation.hmdb_reference,
                    mode=self.config.annotation.ion_mode,
                    ppm=self.config.annotation.hmdb_ppm,
                )
            if self.config.annotation.metaboscape_reference is not None:
                result = annotate_metaboscape(
                    result,
                    self.config.annotation.metaboscape_reference,
                    ppm=self.config.annotation.metaboscape_ppm,
                )
            return {"adata": atomic_write_h5ad(result, staging_dir / "cells_annotated.h5ad")}

        artifacts = self._run_stage(
            "annotation", signature, producer, resume=resume, overwrite=overwrite
        )
        return read_h5ad(artifacts["adata"])

    def analyze(
        self, adata: anndata.AnnData, *, resume: bool = False, overwrite: bool = False
    ) -> anndata.AnnData:
        if self.config.annotation.enabled:
            input_path = self.output_dir / "annotation" / "cells_annotated.h5ad"
        elif self.config.qc.enabled:
            input_path = self.output_dir / "qc" / "cells_qc.h5ad"
        else:
            input_path = self.output_dir / "quantification" / "cells.h5ad"
        persisted = self._persisted_stage_input(adata, input_path)
        signature = stage_signature(
            {
                "project_name": self.config.project.name,
                "random_seed": self.config.project.random_seed,
                "clustering": self.config.clustering.model_dump(mode="json"),
                "analysis": self.config.analysis.model_dump(mode="json"),
            },
            [input_path],
        )

        def producer(staging_dir: Path) -> Mapping[str, Path]:
            result = persisted.copy()
            produced: dict[str, Path] = {}
            if self.config.clustering.enabled:
                result = cluster_cells(
                    result,
                    n_neighbors=self.config.clustering.n_neighbors,
                    resolution=self.config.clustering.resolution,
                    n_top_features=self.config.clustering.n_top_features,
                    random_seed=self.config.project.random_seed,
                )
            if self.config.analysis.cosg.enabled:
                cosg = run_cosg(
                    result,
                    groupby=self.config.analysis.cosg.groupby,
                    n_genes=self.config.analysis.cosg.n_genes,
                )
                cosg_path, cosg_schema = _atomic_save_typed_csv(cosg, staging_dir / "cosg.csv")
                produced.update(cosg=cosg_path, cosg_schema=cosg_schema)
            if self.config.analysis.cnmf.enabled:
                cnmf_config = self.config.analysis.cnmf
                prepared = prepare_cnmf(
                    result,
                    staging_dir / "cnmf",
                    name=self.config.project.name,
                    components=cnmf_config.components,
                    seed=cnmf_config.seed,
                    num_highvar_genes=cnmf_config.num_highvar_genes,
                )
                run_cnmf(
                    prepared,
                    worker_index=0,
                    total_workers=1,
                    selected_k=cnmf_config.selected_k,
                )
                usage_candidates = sorted(
                    prepared.task_dir.glob(
                        f"{self.config.project.name}.usages.k_{cnmf_config.selected_k}*.txt"
                    )
                )
                if len(usage_candidates) != 1:
                    raise JointError(f"Expected one cNMF usage file, found: {usage_candidates}")
                usage_path = staging_dir / "cnmf-usage.tsv"
                shutil.copyfile(usage_candidates[0], usage_path)
                result = load_cnmf_results(result, usage_path)
                joint_settings = dict(result.uns.get("joint", {}))
                joint_settings["cnmf_usage_path"] = str(
                    (self.output_dir / "analysis" / usage_path.name).resolve()
                )
                result.uns["joint"] = joint_settings
                marker = _atomic_save_json(
                    {
                        "selected_k": cnmf_config.selected_k,
                        "usage_artifact": usage_path.name,
                    },
                    staging_dir / "cnmf-complete.json",
                )
                produced.update(cnmf=marker, cnmf_usage=usage_path)
            produced["adata"] = atomic_write_h5ad(result, staging_dir / "cells_analyzed.h5ad")
            return produced

        artifacts = self._run_stage(
            "analysis", signature, producer, resume=resume, overwrite=overwrite
        )
        return read_h5ad(artifacts["adata"])

    def trajectory(
        self, adata: anndata.AnnData, *, resume: bool = False, overwrite: bool = False
    ) -> anndata.AnnData:
        path_file = self.config.trajectory.path_file
        if path_file is None:
            raise JointError("trajectory.path_file is required when trajectory is enabled")
        input_path = self.output_dir / "analysis" / "cells_analyzed.h5ad"
        persisted = self._persisted_stage_input(adata, input_path)
        signature = stage_signature(
            self.config.trajectory.model_dump(mode="json"),
            [input_path, path_file],
        )

        def producer(staging_dir: Path) -> Mapping[str, Path]:
            try:
                table = pd.read_csv(path_file)
            except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
                raise JointError(f"Could not read trajectory path file {path_file}: {exc}") from exc
            if list(table.columns) != ["x", "y"]:
                raise JointError("trajectory path file must contain exactly columns x and y")
            try:
                values = (
                    table.loc[:, ["x", "y"]]
                    .apply(pd.to_numeric, errors="raise")
                    .to_numpy(dtype=float)
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise JointError("trajectory path x and y columns must be numeric") from exc
            if values.shape[0] < 2:
                raise JointError("trajectory path must contain at least two points")
            if not np.isfinite(values).all():
                raise JointError("trajectory path x and y values must be finite")
            result = fit_spatial_trajectory(persisted, values)
            trends = calculate_feature_trends(
                result,
                self.config.trajectory.features,
                points=self.config.trajectory.points,
            )
            trends_path, trends_schema = _atomic_save_typed_csv(trends, staging_dir / "trends.csv")
            return {
                "adata": atomic_write_h5ad(result, staging_dir / "cells_trajectory.h5ad"),
                "trends": trends_path,
                "trends_schema": trends_schema,
            }

        artifacts = self._run_stage(
            "trajectory", signature, producer, resume=resume, overwrite=overwrite
        )
        return read_h5ad(artifacts["adata"])

    def _finalize(
        self,
        adata: anndata.AnnData,
        input_path: Path,
        *,
        resume: bool,
        overwrite: bool,
    ) -> anndata.AnnData:
        persisted = self._persisted_stage_input(adata, input_path)
        signature = stage_signature({"stage": "final"}, [input_path])
        self._preflight_final_root(signature, resume=resume, overwrite=overwrite)

        def producer(staging_dir: Path) -> Mapping[str, Path]:
            return {"adata": atomic_write_h5ad(persisted, staging_dir / "joint-final.h5ad")}

        artifacts = self._run_stage(
            "final", signature, producer, resume=resume, overwrite=overwrite
        )
        self._sync_final_root(artifacts["adata"], resume=resume, overwrite=overwrite)
        return read_h5ad(artifacts["adata"])

    def run(self, *, resume: bool = False, overwrite: bool = False) -> anndata.AnnData:
        result = self.run_spatial_core(resume=resume, overwrite=overwrite).adata
        if self.config.qc.enabled:
            result = self.qc(result, resume=resume, overwrite=overwrite)
        if self.config.annotation.enabled:
            result = self.annotate(result, resume=resume, overwrite=overwrite)
        result = self.analyze(result, resume=resume, overwrite=overwrite)
        if self.config.trajectory.enabled:
            result = self.trajectory(result, resume=resume, overwrite=overwrite)
        if self.config.trajectory.enabled:
            final_input = self.output_dir / "trajectory" / "cells_trajectory.h5ad"
        else:
            final_input = self.output_dir / "analysis" / "cells_analyzed.h5ad"
        return self._finalize(
            result,
            final_input,
            resume=resume,
            overwrite=overwrite,
        )
