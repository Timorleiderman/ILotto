"""Test suite.

Tests read a pinned copy of the archive, never the live one. The live archive
grows after every draw; a test asserting a statistical property of it would
fail by chance about one refresh in twenty. Watching the live data is the
report's job, not the test suite's.
"""

from pathlib import Path

#: The 1,661 modern draws through 2026-11-03.
SNAPSHOT = str(Path(__file__).parent / "golden" / "archive_2026-11-03.csv")
