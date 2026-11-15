"""One dashboard owns the mutable notebook, including startup recovery."""

import subprocess
import sys

import pytest

from lotto.ui import server
from lotto.ui.storage import dashboard_lock


def other_process_lock(root):
    return subprocess.run(
        [sys.executable, "-c", """
import sys
from pathlib import Path
from lotto.ui.storage import DashboardInUse, dashboard_lock
try:
    with dashboard_lock(Path(sys.argv[1])):
        pass
except DashboardInUse:
    sys.exit(2)
""", str(root)], capture_output=True, text=True, timeout=10)


def test_lock_excludes_another_process_and_releases_after_error(tmp_path):
    with pytest.raises(RuntimeError, match="simulated stop"):
        with dashboard_lock(tmp_path):
            assert other_process_lock(tmp_path).returncode == 2
            raise RuntimeError("simulated stop")
    next_launch = other_process_lock(tmp_path)
    assert next_launch.returncode == 0, next_launch.stderr
    assert (tmp_path / ".dashboard.lock").exists()  # reuse the same inode on every launch


def test_second_launch_cannot_recover_or_change_the_first_notebook(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(server.paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(server, "Service", lambda *a, **k: pytest.fail("must lock before startup recovery"))
    with dashboard_lock(tmp_path / "results/ui"), pytest.raises(SystemExit) as error:
        server.main(["--offline", "--no-browser", "--port", "7000"])
    assert error.value.code == 2
    assert "already using results/ui" in capsys.readouterr().err
