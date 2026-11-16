"""What the dashboard keeps on disk, all of it under ``results/ui/``.

Three things: completed results (so a report or a log entry can refer to one by
id), the prediction log, and the Lab notebook. Results and log entries are
written once and never changed; the notebook is rewritten as attempts finish.
"""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path

from lotto.core.files import write_json

_ID = re.compile(r"[0-9a-f]{16}")
#: How many result files to keep. Log entries carry their own copy, so pruning never loses one.
KEEP_RESULTS = 300


class DashboardInUse(RuntimeError):
    """Another dashboard owns this folder's mutable notebook."""


@contextmanager
def dashboard_lock(root: Path):
    """Own a dashboard folder until shutdown; the OS releases the lock after a crash.

    Acquire this before constructing Service: startup also changes the notebook.
    Keep the file in place so all launches lock the same inode.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".dashboard.lock").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError) as exc:
            raise DashboardInUse("A dashboard is already using results/ui. Stop it before starting another.") from exc
        yield
        # Closing the handle releases the lock, including on exceptional exits.


def valid_id(value) -> bool:
    """True for the 16-hex-digit ids this module hands out; nothing else reaches a path."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None


class Storage:
    """Files under one root folder, with a lock because the server is threaded."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    # -- results -------------------------------------------------------------
    def put_result(self, result: dict) -> dict:
        """Keep a completed result and return its immutable on-disk version."""
        if not valid_id(result.get("id")):
            raise ValueError("invalid result id")
        path = self.root / "results" / f"{result['id']}.json"
        with self._lock:
            if not path.exists():
                write_json(path, result)
                self._prune()
            return json.loads(path.read_text())

    def _prune(self) -> None:
        """Drop the oldest result files beyond `KEEP_RESULTS`. Call with the lock held."""
        files = sorted((self.root / "results").glob("*.json"), key=lambda p: p.stat().st_mtime)
        for old in files[:-KEEP_RESULTS]:
            old.unlink(missing_ok=True)

    def get_result(self, result_id: str) -> dict | None:
        """The result with this id, or None."""
        if not valid_id(result_id):
            return None
        with self._lock:
            path = self.root / "results" / f"{result_id}.json"
            return json.loads(path.read_text()) if path.is_file() else None

    # -- prediction log ------------------------------------------------------
    def log_entries(self) -> list[dict]:
        """Every saved prediction, oldest first."""
        folder = self.root / "log"
        if not folder.is_dir():
            return []
        entries = [json.loads(p.read_text()) for p in folder.glob("*.json") if valid_id(p.stem)]
        return sorted(entries, key=lambda e: (e["saved_utc"], e["id"]))

    def add_log_entry(self, entry: dict) -> dict:
        """Save an entry once. Saving the same id again returns the original, unchanged."""
        if not valid_id(entry.get("id")):
            raise ValueError("invalid log id")
        path = self.root / "log" / f"{entry['id']}.json"
        with self._lock:
            if path.exists():
                return json.loads(path.read_text())
            write_json(path, entry)
            return entry

    # -- Lab notebook --------------------------------------------------------
    def notebook(self) -> list[dict]:
        """Every Lab attempt recorded so far, oldest first."""
        path = self.root / "notebooks" / "notebook.json"
        return json.loads(path.read_text())["attempts"] if path.is_file() else []

    def save_notebook(self, attempts: list[dict]) -> None:
        """Replace the notebook file with these attempts."""
        with self._lock:
            write_json(self.root / "notebooks" / "notebook.json", {"schema": 1, "attempts": attempts})
