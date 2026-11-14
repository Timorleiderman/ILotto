"""Mac-owned reservations for remote workers; never allocate from copied ledgers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import uuid

from lotto.seedsearch.generator import GENERATOR, MAX_SEED, prefix_hash, write_json


def remote_ranges(root: Path) -> list[dict]:
    """Read while holding root/allocation.lock when making allocation decisions."""
    path = Path(root) / "remote_allocations.json"
    if not path.exists():
        if any((Path(root).parent / "remote_workers").glob("*/manifest.json")):
            raise ValueError("Remote allocation registry is missing but worker history exists")
        return []
    registry = json.loads(path.read_text())
    if registry.get("schema") != 1 or registry.get("generator") != GENERATOR:
        raise ValueError("Remote allocation registry identity mismatch")
    records = registry["ranges"]
    previous = 0
    ids = set()
    for record in records:
        if (not isinstance(record["start"], int) or not isinstance(record["stop"], int)
                or not previous <= record["start"] < record["stop"] <= MAX_SEED
                or record["id"] in ids
                or any(record["manifest"][key] != record[key] for key in ("id", "start", "stop"))
                or record["manifest"]["generator"] != GENERATOR):
            raise ValueError("Invalid remote allocation registry")
        previous = record["stop"]
        ids.add(record["id"])
    return records


def reserve_remote(prefix, *, root: Path, run_root: Path, candidates: int, worker: dict, source: dict):
    """Reserve beyond ALL local ranges, earlier GPU manifests, and remote leases.

    Local campaign writers use the same short-lived allocation lock. A legacy
    running campaign is safe only when its complete finite interval is already
    reserved, as with the original one-hour job. Its ledger is never edited here.
    """
    from lotto.seedsearch import campaign as seed_campaign

    if not isinstance(candidates, int) or candidates < 1:
        raise ValueError("Candidate count must be positive")
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    history_hash = prefix_hash(prefix)
    with seed_campaign.campaign_lock(root / "allocation.lock",
                                    wait=seed_campaign.ALLOCATION_WAIT_SECONDS):
        records = remote_ranges(root)
        ends = [r["stop"] for r in records]
        # Include other historical prefixes too; reusing a seed is unnecessary.
        for path in root.glob("*/ledger.json"):
            ledger = json.loads(path.read_text())
            if ledger["identity"]["generator"] != GENERATOR:
                continue
            for record in ledger["ranges"]:
                if not 0 <= record["start"] < record["stop"] <= MAX_SEED:
                    raise ValueError("Invalid local reservation")
                ends.append(record["stop"])
        # Earlier manually assigned GPU pilots remain reserved, including tails.
        for path in run_root.glob("*/allocation.json"):
            manifest = json.loads(path.read_text())
            if manifest["generator"] != GENERATOR:
                continue
            if not 0 <= manifest["start"] < manifest["stop"] <= MAX_SEED:
                raise ValueError("Invalid historical GPU allocation")
            ends.append(manifest["stop"])
        start = max(ends, default=0)
        stop = start + candidates
        if stop > MAX_SEED:
            raise ValueError("Requested allocation exceeds the unsigned seed space")
        allocation_id = uuid.uuid4().hex
        manifest = {"schema": 1, "id": allocation_id, "generator": GENERATOR,
                    "prefix_sha256": history_hash, "prefix": [asdict(x) for x in prefix],
                    "start": start, "stop": stop, "worker": worker, "source": source}
        record = {"id": allocation_id, "start": start, "stop": stop, "manifest": manifest}
        records.append(record)
        # Commit the reservation first. Even an interrupted export cannot lose it.
        write_json(root / "remote_allocations.json",
                   {"schema": 1, "generator": GENERATOR, "ranges": records})
        return manifest


def get_remote(root: Path, allocation_id: str):
    for record in remote_ranges(root):
        if record["id"] == allocation_id:
            return record["manifest"]
    raise ValueError("Unknown allocation ID; use the original coordinator and ledger")
