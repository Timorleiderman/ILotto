"""What the dashboard keeps on disk, all of it under ``results/ui/``.

Three things: completed results (so a report or a log entry can refer to one by
id), the prediction log, and the Lab notebook. Results and log entries are
written once and never changed; the notebook is rewritten as attempts finish.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from lotto.core.files import write_json

_ID = re.compile(r"[0-9a-f]{16}")


def valid_id(value) -> bool:
    """True for the 16-hex-digit ids this module hands out; nothing else reaches a path."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None


class Storage:
    """Files under one root folder, with a lock because the server is threaded."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._results: dict[str, dict] = {}

    # -- results -------------------------------------------------------------
    def put_result(self, result: dict) -> None:
        """Keep a completed result. An id already on disk is left as it is."""
        path = self.root / "results" / f"{result['id']}.json"
        with self._lock:
            self._results[result["id"]] = result
            if not path.exists():
                write_json(path, result)

    def get_result(self, result_id: str) -> dict | None:
        """The result with this id, or None."""
        if not valid_id(result_id):
            return None
        with self._lock:
            if result_id in self._results:
                return self._results[result_id]
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
