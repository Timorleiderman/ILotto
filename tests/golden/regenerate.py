"""Rewrite the golden files from the current code. Run only after an intended change.

    uv run python -m tests.golden.regenerate
"""

import json

from tests.test_golden import GOLDEN, coverage_numbers, quick_suite_numbers

(GOLDEN / "quick_suite_n100.json").write_text(json.dumps(quick_suite_numbers(), indent=2) + "\n")
(GOLDEN / "coverage.json").write_text(json.dumps(coverage_numbers(), indent=2) + "\n")
print("golden files rewritten")
