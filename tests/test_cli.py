"""The catalog and the command that reads it."""

import importlib
from pathlib import Path

import pytest

from lotto import cli
from lotto.experiments import CATALOG

DOCS = Path(__file__).parent.parent / "docs"


def test_every_experiment_is_complete():
    assert len({e.id for e in CATALOG}) == len(CATALOG)
    for e in CATALOG:
        assert e.question.endswith("?"), e.id
        assert callable(importlib.import_module(f"lotto.experiments.{e.module}").main), e.id
        assert (DOCS / e.page).is_file(), f"{e.id}: docs/{e.page} is missing"


def test_list_shows_every_experiment(capsys):
    assert cli.main(["list"]) == 0
    shown = capsys.readouterr().out
    assert all(e.id in shown for e in CATALOG)


def test_unknown_experiment_is_refused(capsys):
    assert cli.main(["run", "no-such-experiment"]) == 2


def test_generated_index_is_current():
    """docs/reference/experiments.md is generated; regenerate it with
    `uv run lotto list --markdown` when the catalog changes."""
    page = (DOCS / "reference" / "experiments.md").read_text()
    assert cli.catalog_markdown() in page


@pytest.mark.parametrize("experiment_id", ["coverage-search", "strong-numbers", "thousand-draws"])
def test_protocol_experiments_support_help_without_running(experiment_id, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", experiment_id, "--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_list_and_run_help_and_unknown_flags(capsys):
    assert cli.main(["list", "--help"]) == 0
    assert "usage:" in capsys.readouterr().out
    assert cli.main(["run", "--help"]) == 0
    assert "usage:" in capsys.readouterr().out
    assert cli.main(["list", "--unknown"]) == 2
