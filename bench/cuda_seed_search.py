"""Persistent CUDA worker, using only the Python standard library on Linux."""

from __future__ import annotations

import json
from pathlib import Path
import selectors
import subprocess
import tempfile

from .native_seed_search import validate_native_result
from .seed_search import MAX_SEED, _prefix_hash


class CudaWorker:
    def __init__(self, binary: Path, timeout: float = 120):
        self.timeout = timeout
        self.errors = tempfile.TemporaryFile(mode="w+t")
        try:
            self.process = subprocess.Popen([str(Path(binary).resolve())], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.errors, text=True)
        except BaseException:
            self.errors.close()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def request(self, text):
        try:
            self.process.stdin.write(text)
            self.process.stdin.flush()
            if not self.selector.select(self.timeout):
                raise TimeoutError("CUDA worker timed out; batch was not committed")
            line = self.process.stdout.readline()
            if not line:
                self.errors.seek(0)
                raise RuntimeError(f"CUDA worker exited: {self.errors.read(2000)}")
            return json.loads(line)
        except BaseException:
            self.close()
            raise

    def vectors(self, pairs):
        if not 1 <= len(pairs) <= 10000:
            raise ValueError("Expected 1..10000 vector pairs")
        return self.request(f"V {len(pairs)}\n" + "".join(f"{s} {d}\n" for s, d in pairs))["outcomes"]

    def batch(self, prefix, start, stop, *, threads=128, blocks=656):
        _prefix_hash(prefix)
        if (not 0 <= start < stop <= MAX_SEED or stop - start > 1_000_000_000
                or threads not in (64, 128, 256, 512) or not 1 <= blocks <= 65535):
            raise ValueError("Invalid CUDA batch range or launch dimensions")
        result = self.request(
            f"B {start} {stop - start} {threads} {blocks} {len(prefix)}\n"
            + "".join(f"{p.date} {p.outcome}\n" for p in prefix))
        return validate_native_result(result, prefix, start, stop)

    def close(self):
        self.selector.close()
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()
        self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
