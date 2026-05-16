"""Single-process guard for the single-stock tool.

Mirrors scripts/run.py's Windows-safe PID liveness check so the same
machine can host both the main bot and this tool simultaneously without
either stomping on the other's lock.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return str(pid) in result.stdout
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(lock_path: str | Path) -> Path:
    path = Path(lock_path)
    if path.exists():
        try:
            old_pid = int(path.read_text(encoding="utf-8").strip())
            if _is_process_running(old_pid):
                print(f"ERROR: singlestock already running (PID {old_pid}).")
                print(f"Kill it first, or delete {path} if stale.")
                sys.exit(1)
            print(f"Removing stale singlestock lock (PID {old_pid} not running)")
            path.unlink(missing_ok=True)
        except (ValueError, OSError):
            path.unlink(missing_ok=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")
    return path


def release_lock(lock_path: str | Path) -> None:
    path = Path(lock_path)
    try:
        if not path.exists():
            return
        lock_pid = int(path.read_text(encoding="utf-8").strip())
        if lock_pid != os.getpid():
            return
        path.unlink(missing_ok=True)
    except Exception:
        pass
