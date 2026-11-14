"""The legacy split is wrong, and this test proves it stays visibly wrong.

This test documents a defect on purpose. If it fails, someone repaired the
exhibit: update legacy/README.md and docs/approaches/legacy-models.md to match.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEGACY = Path(__file__).resolve().parent.parent / "legacy"


def test_legacy_test_windows_do_not_precede_their_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # importing helpers creates a log file in the working directory
    monkeypatch.syspath_prepend(str(LEGACY))
    sys.modules.pop("helpers", None)
    import helpers

    # Row r holds the value r + 1 everywhere, so values identify rows after the split's "- 1".
    row_ids = pd.DataFrame(np.arange(100).repeat(7).reshape(100, 7) + 1)
    x_train, y_train, x_test, y_test = helpers.train_test_split(row_ids, test_size=10, w=3)

    # Test inputs come from the end of the file, their target from the start.
    assert x_test[0][:, 0].tolist() == [87, 88, 89]
    assert int(y_test[0][0]) == 0
    # The file is newest-first, so rows 10-12 "predicting" row 13 runs time backwards.
    assert x_train[0][:, 0].tolist() == [10, 11, 12]
    assert int(y_train[0][0]) == 13
