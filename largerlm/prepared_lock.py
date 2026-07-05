from __future__ import annotations

import errno
import fcntl
import os
from dataclasses import dataclass
from pathlib import Path


PREPARED_RUN_LOCK_FILE = ".largerlm-selected-replay.lock"
PREPARED_RUN_LOCK_ENV = "LARGERLM_PREPARED_RUN_LOCK_PATH"


class PreparedRunLockError(RuntimeError):
    """Raised when a prepared-package generation lock cannot be acquired."""


@dataclass
class PreparedRunLock:
    path: Path
    fd: int

    def close(self) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


@dataclass(frozen=True)
class PreparedRunLockStatus:
    path: Path
    available: bool
    busy: bool
    inherited_lock_marker_matches: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "available": self.available,
            "busy": self.busy,
            "inherited_lock_marker_matches": self.inherited_lock_marker_matches,
            "error": self.error,
        }


def same_lock_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left) == Path(right)


def prepared_run_lock_already_held(path: Path) -> bool:
    held = os.environ.get(PREPARED_RUN_LOCK_ENV)
    return bool(held) and same_lock_path(held, path)


def prepared_run_lock_path_for_manifest(manifest_path: str | Path) -> Path:
    return Path(manifest_path).parent / PREPARED_RUN_LOCK_FILE


def inspect_prepared_run_lock_path(path: str | Path) -> PreparedRunLockStatus:
    lock_path = Path(path)
    inherited = prepared_run_lock_already_held(lock_path)
    parent = lock_path.parent
    if not parent.exists():
        return PreparedRunLockStatus(
            path=lock_path,
            available=False,
            busy=False,
            inherited_lock_marker_matches=inherited,
            error=f"prepared run lock directory does not exist: {parent}",
        )
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        return PreparedRunLockStatus(
            path=lock_path,
            available=False,
            busy=False,
            inherited_lock_marker_matches=inherited,
            error=f"failed to open prepared run lock {lock_path}: {exc}",
        )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return PreparedRunLockStatus(
                path=lock_path,
                available=False,
                busy=True,
                inherited_lock_marker_matches=inherited,
            )
        return PreparedRunLockStatus(
            path=lock_path,
            available=False,
            busy=False,
            inherited_lock_marker_matches=inherited,
            error=f"failed to probe prepared run lock {lock_path}: {exc}",
        )
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return PreparedRunLockStatus(
        path=lock_path,
        available=True,
        busy=False,
        inherited_lock_marker_matches=inherited,
    )


def acquire_prepared_run_lock_path(
    path: str | Path,
    *,
    busy_message: str,
) -> PreparedRunLock:
    lock_path = Path(path)
    parent = lock_path.parent
    if not parent.exists():
        raise PreparedRunLockError(
            f"prepared run lock directory does not exist: {parent}"
        )
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise PreparedRunLockError(
            f"failed to open prepared run lock {lock_path}: {exc}"
        ) from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            raise PreparedRunLockError(
                f"{busy_message} (lock {lock_path}); wait for it to finish "
                "before starting another GLM replay"
            ) from exc
        raise PreparedRunLockError(
            f"failed to acquire prepared run lock {lock_path}: {exc}"
        ) from exc
    os.set_inheritable(fd, True)
    return PreparedRunLock(path=lock_path, fd=fd)
