"""Where the repository is, found once for the whole package.

Experiments read the committed archive and frozen evidence, so they need the
repository checkout, not just the installed package. With the usual editable
install (``uv sync``) the package lives inside the checkout. With a regular
install (``pip install .``) it lives in site-packages, and the checkout is
found from the working directory instead. ``LOTTO_ROOT`` overrides both.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The file that marks a checkout: the committed draw archive.
MARKER = Path("input") / "Orig_IL_lotto.csv"


def find_repo_root() -> Path:
    """The checkout that holds ``input/Orig_IL_lotto.csv``.

    Tried in order: ``$LOTTO_ROOT``; the folder above this package; the working
    directory and each of its parents. Falls back to the folder above this
    package, so error messages name a concrete missing file.
    """
    package_parent = Path(__file__).resolve().parents[2]
    if os.environ.get("LOTTO_ROOT"):
        return Path(os.environ["LOTTO_ROOT"]).resolve()
    if (package_parent / MARKER).is_file():
        return package_parent
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / MARKER).is_file():
            return folder
    return package_parent


REPO_ROOT = find_repo_root()
