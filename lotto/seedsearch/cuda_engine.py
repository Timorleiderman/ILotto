"""Persistent CUDA worker, using only the Python standard library on Linux."""

from __future__ import annotations

import json
import selectors
import subprocess
import tempfile
from pathlib import Path

from lotto.seedsearch.generator import GENERATOR, MAX_SEED, prefix_hash
from lotto.seedsearch.rust_engine import validate_native_result

#: Largest batch the CUDA worker accepts (matches MAX_BATCH in main.cu). At
#: ~7e9 seeds/s on an RTX 3090 that is ~3 s of GPU work per Python round trip.
MAX_BATCH = 20_000_000_000
#: What workers built before the capability query ("C") accept. The worker
#: binary is not tracked by Git, so a pull does not upgrade it.
LEGACY_MAX_BATCH = 1_000_000_000


class CudaWorker:
    def __init__(self, binary: Path, timeout: float = 120, *, command=None):
        self.timeout = timeout
        self._argv = command or [str(Path(binary).resolve())]
        self._max_batch = None
        self.capabilities = None
        self._start()

    def _start(self):
        self.errors = tempfile.TemporaryFile(mode="w+t")
        try:
            self.process = subprocess.Popen(self._argv, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.errors, text=True)
        except BaseException:
            self.errors.close()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    @property
    def max_batch(self) -> int:
        """Largest batch this worker accepts, asked of the worker itself.

        A worker built before the query exits with "Unknown command"; it is
        restarted and limited to LEGACY_MAX_BATCH, so an un-rebuilt binary keeps
        working at its old batch size instead of rejecting every larger batch.
        """
        if self._max_batch is None:
            try:
                caps = self.request("C\n")
            except RuntimeError as exc:
                if "Unknown command" not in str(exc):
                    raise
                self._start()
                caps = {"generator": GENERATOR, "protocol": 1, "max_batch": LEGACY_MAX_BATCH}
            if (caps.get("generator") != GENERATOR or not isinstance(caps.get("max_batch"), int)
                    or not 1 <= caps["max_batch"] <= MAX_BATCH):
                self.close()
                raise RuntimeError(f"CUDA worker reported invalid capabilities: {caps}")
            self.capabilities, self._max_batch = caps, caps["max_batch"]
        return self._max_batch

    def request(self, text):
        """Send one command to the worker and return its JSON reply; kills the worker on any failure."""
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
        """Hash (seed, date) pairs on the GPU, for checking against the Python reference."""
        if not 1 <= len(pairs) <= 10000:
            raise ValueError("Expected 1..10000 vector pairs")
        return self.request(f"V {len(pairs)}\n" + "".join(f"{s} {d}\n" for s, d in pairs))["outcomes"]

    def batch(self, prefix, start, stop, *, threads=128, blocks=656):
        """Search seeds ``[start, stop)`` against ``prefix`` on the GPU and validate the result."""
        prefix_hash(prefix)
        if (not 0 <= start < stop <= MAX_SEED or stop - start > self.max_batch
                or threads not in (64, 128, 256, 512) or not 1 <= blocks <= 65535):
            raise ValueError("Invalid CUDA batch range or launch dimensions")
        result = self.request(
            f"B {start} {stop - start} {threads} {blocks} {len(prefix)}\n"
            + "".join(f"{p.date} {p.outcome}\n" for p in prefix))
        return validate_native_result(result, prefix, start, stop)

    def close(self):
        """Stop the worker process and release its pipes."""
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
