"""One command for the whole project.

    uv run lotto list                      every experiment: id, question, a fast first run
    uv run lotto run coverage --help       the options of one experiment
    uv run lotto run coverage --count 10   run it
    uv run lotto report --quick            rebuild the benchmark report  (= lotto run report)
    uv run lotto plan                      which form to play next       (= lotto run plan)
    uv run lotto predict                   every model's 10-row form for the next draw

Everything after the experiment id is passed to that experiment unchanged.
"""

from __future__ import annotations

import importlib
import sys

from lotto.experiments import CATALOG, find

SHORTCUTS = ("report", "plan", "predict")


def catalog_text() -> str:
    """The catalog as plain text, grouped by chapter."""
    lines, chapter = [], None
    for e in CATALOG:
        if e.chapter != chapter:
            chapter = e.chapter
            lines += ["", chapter]
        lines.append(f"  {e.id:28s}{e.question}")
        if e.quick:
            lines.append(f"  {'':28s}try: uv run lotto run {e.id} {e.quick}")
        if e.needs:
            lines.append(f"  {'':28s}needs: {e.needs}")
    return "\n".join(lines).lstrip("\n")


def catalog_markdown() -> str:
    """The catalog as a Markdown table, for the generated docs index."""
    rows = ["| Chapter | Experiment | Question | Command | Write-up |", "|---|---|---|---|---|"]
    for e in CATALOG:
        command = f"`uv run lotto run {e.id} {e.quick}`".replace(" `", "`")
        rows.append(f"| {e.chapter} | `{e.id}` | {e.question} | {command} | [{e.page}](../{e.page}) |")
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "list":
        print(catalog_markdown() if rest == ["--markdown"] else catalog_text())
        return 0
    if command in SHORTCUTS:
        experiment_id = command
    elif command == "run" and rest:
        experiment_id, rest = rest[0], rest[1:]
    else:
        print(__doc__)
        return 2
    try:
        experiment = find(experiment_id)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    module = importlib.import_module(f"lotto.experiments.{experiment.module}")
    # Experiments parse their own options, so hand them a clean argument list.
    sys.argv = [f"lotto run {experiment.id}", *rest]
    return module.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
