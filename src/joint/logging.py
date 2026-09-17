"""Run-scoped logging helpers."""

import logging
import os
import sys
from pathlib import Path

from joint.errors import JointError


def _handler_targets_log_path(handler: logging.Handler, log_path: Path) -> bool:
    stream = getattr(handler, "stream", None)
    if stream is None or stream.closed:
        return False
    try:
        handler_stat = os.fstat(stream.fileno())
        log_path_stat = os.stat(log_path, follow_symlinks=False)
    except OSError:
        return False
    return (handler_stat.st_dev, handler_stat.st_ino) == (
        log_path_stat.st_dev,
        log_path_stat.st_ino,
    )


def _close_fd_backed_handler(handler: logging.Handler) -> None:
    stream = getattr(handler, "stream", None)
    try:
        handler.close()
    finally:
        if stream is not None and not stream.closed:
            stream.close()


def configure_logging(output_dir: str | Path, *, verbose: bool = False) -> logging.Logger:
    destination = Path(output_dir).resolve()
    log_dir = destination / "logs"
    if log_dir.is_symlink() or (log_dir.exists() and not log_dir.is_dir()):
        raise JointError(f"Log directory is unsafe: {log_dir}")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise JointError(f"Log directory is unsafe: {log_dir}: {exc}") from exc
    if not log_dir.resolve().is_relative_to(destination):
        raise JointError(f"Log directory is unsafe: {log_dir}")
    log_path_raw = log_dir / "joint.log"
    logger = logging.getLogger(f"joint.{destination}")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    unsafe_log_path = log_path_raw.is_symlink() or (
        log_path_raw.exists() and not log_path_raw.is_file()
    )
    stale = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_joint_log_path", None) == str(log_path_raw)
        and (
            getattr(handler, "stream", None) is None
            or handler.stream.closed
            or not log_path_raw.exists()
            or unsafe_log_path
            or not _handler_targets_log_path(handler, log_path_raw)
        )
    ]
    for handler in stale:
        logger.removeHandler(handler)
        _close_fd_backed_handler(handler)
    if unsafe_log_path:
        raise JointError(f"Log destination is unsafe: {log_path_raw}")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if not any(getattr(handler, "_joint_log_path", None) == str(log_path_raw) for handler in logger.handlers):
        file_fd: int | None = None
        file_stream = None
        try:
            directory_fd = os.open(log_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                file_fd = os.open(
                    "joint.log",
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
                    0o644,
                    dir_fd=directory_fd,
                )
            finally:
                os.close(directory_fd)
            file_stream = os.fdopen(file_fd, "a", encoding="utf-8")
            file_fd = None
            file_handler = logging.StreamHandler(file_stream)
        except OSError as exc:
            if file_stream is not None:
                file_stream.close()
            elif file_fd is not None:
                os.close(file_fd)
            raise JointError(f"Log destination is unsafe: {log_path_raw}: {exc}") from exc
        except BaseException:
            if file_stream is not None:
                file_stream.close()
            elif file_fd is not None:
                os.close(file_fd)
            raise
        file_handler._joint_log_path = str(log_path_raw)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if not any(
        isinstance(handler, logging.StreamHandler)
        and not getattr(handler, "_joint_log_path", None)
        for handler in logger.handlers
    ):
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
