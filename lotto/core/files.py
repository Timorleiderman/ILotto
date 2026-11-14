"""Small file helpers shared by the experiments and the seed search."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path: Path | str) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path | str, value) -> None:
    """Write JSON so that readers see either the old file or the complete new one.

    The data goes to a temporary file in the same folder, is flushed to disk,
    and is then renamed over the target; a crash midway leaves the old file in
    place. Each file must have a single writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as stream:
        staging = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            staging.unlink(missing_ok=True)
            raise
    os.replace(staging, path)
