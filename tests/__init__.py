"""Test suite.

Tests read a pinned copy of the archive, never the live one. The live archive
grows after every draw; a test asserting a statistical property of it would
fail by chance about one refresh in twenty. Watching the live data is the
report's job, not the test suite's.
"""

from pathlib import Path

#: The 1,661 modern draws through 2026-11-03.
SNAPSHOT = str(Path(__file__).parent / "golden" / "archive_2026-11-03.csv")

#: The 1,005 published prize tables, draws 2982-3986 (2018-01-30 to 2026-11-19).
PRIZES_SNAPSHOT = str(Path(__file__).parent / "golden" / "prizes_through_3986.csv")
