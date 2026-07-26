"""Loading and cleaning of the Mifal HaPayis Lotto archive.

The published archive spans 1968-today, but the *game* changed several times.
Mixing eras is the single largest source of bogus signal in this dataset, so
everything here is era-aware and the default is the modern 6/37 + strong 1-7
format only.

Era history recovered from the archive itself (see `docs/methodology`):

    1968-2004   pool up to 6/49, strong number up to 49
    2005-2008   6/34, strong 1-10
    2009-2011   6/37 transition, strong still reaching 8-9
    2012-now    6/37, strong 1-7   <- the only homogeneous block
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

CSV_URL = "https://pais.co.il/Lotto/lotto_resultsDownload.aspx"
RAW_CSV = "input/Orig_IL_lotto.csv"
#: Deliberately not `lotto_IL_filtered.csv`: `helpers.ILottoCSV` regenerates that
#: file with the legacy value-based filter, so the two would overwrite each
#: other with incompatible contents. Both readers get their own artifact.
CLEAN_CSV = "input/lotto_IL_modern.csv"

BALL_COLS = [f"Ball_{i}" for i in range(1, 7)]
ALL_COLS = BALL_COLS + ["Ball_Bonus"]

#: First draw date of the current game format.
MODERN_ERA_START = "2012-01-01"

#: Current game parameters.
N_NUMBERS = 37
N_DRAWN = 6
N_STRONG = 7


@dataclass(frozen=True)
class Draws:
    """A block of draws in *chronological* order (oldest first).

    Attributes:
        dates: datetime64 array, length ``n``.
        balls: int array ``(n, 6)``, 1-indexed, ascending within a row.
        strong: int array ``(n,)``, 1-indexed.
    """

    dates: np.ndarray
    balls: np.ndarray
    strong: np.ndarray

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def multi_hot(self) -> np.ndarray:
        """``(n, 37)`` float array; 1.0 where the number was drawn."""
        m = np.zeros((len(self), N_NUMBERS), dtype=np.float32)
        rows = np.repeat(np.arange(len(self)), N_DRAWN)
        m[rows, self.balls.ravel() - 1] = 1.0
        return m

    def slice(self, start: int | None = None, stop: int | None = None) -> "Draws":
        s = slice(start, stop)
        return Draws(self.dates[s], self.balls[s], self.strong[s])


def download(raw_csv: str = RAW_CSV, timeout: int = 15) -> bool:
    """Refresh the raw archive from pais.co.il. Returns True if updated.

    Failure is non-fatal: the committed CSV is used instead, which keeps CI
    green when the site is unreachable.
    """
    os.makedirs(os.path.dirname(raw_csv) or ".", exist_ok=True)
    try:
        res = requests.get(CSV_URL, timeout=timeout)
        res.raise_for_status()
        if len(res.content) < 10_000:
            logger.warning("Download suspiciously small (%d bytes); keeping local copy", len(res.content))
            return False
        with open(raw_csv, "wb") as fh:
            fh.write(res.content)
        logger.info("Downloaded %d bytes of draw history", len(res.content))
        return True
    except requests.RequestException as exc:
        logger.warning("Could not refresh archive (%s); using committed copy", exc)
        return False


def _read_raw(raw_csv: str) -> pd.DataFrame:
    """Parse the Hebrew-encoded raw archive into a typed frame.

    Columns are positional because the header is latin-1-mojibaked Hebrew:
    id, date, 6 balls, strong number, then winner counts.
    """
    df = pd.read_csv(raw_csv, encoding="latin-1")
    names = ["draw_id", "Date", *BALL_COLS, "Ball_Bonus", "winners_lotto", "winners_double"]
    df = df.iloc[:, : len(names)]
    df.columns = names

    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    for col in ALL_COLS + ["winners_lotto", "winners_double"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["date", *ALL_COLS])


def load(
    raw_csv: str = RAW_CSV,
    clean_csv: str | None = CLEAN_CSV,
    era_start: str | None = MODERN_ERA_START,
    refresh: bool = False,
) -> Draws:
    """Load draws in chronological order, restricted to one game format.

    Unlike the original pipeline this **never filters rows by ball value**.
    Doing so silently deletes every historical draw that happened to contain a
    high number, which biases the surviving frequency distribution downward.
    We filter by *date* (i.e. by game format) instead, which is unbiased.
    """
    if refresh:
        download(raw_csv)

    df = _read_raw(raw_csv).sort_values("date").reset_index(drop=True)

    if era_start is not None:
        df = df[df["date"] >= pd.Timestamp(era_start)].reset_index(drop=True)

    if df.empty:
        # Reachable if the upstream download succeeds but yields nothing parseable
        # for this era. Failing with a clear message beats an opaque numpy
        # "zero-size array to reduction" from the range check below.
        raise ValueError(
            f"No draws parsed from {raw_csv} on or after {era_start}. The archive is "
            "empty, truncated, or its column layout changed upstream."
        )

    balls = df[BALL_COLS].to_numpy(dtype=np.int64)
    strong = df["Ball_Bonus"].to_numpy(dtype=np.int64)

    # Guard: anything outside the declared game format means the era window is wrong.
    if era_start is not None:
        bad = (balls < 1) | (balls > N_NUMBERS)
        if bad.any() or strong.min() < 1 or strong.max() > N_STRONG:
            raise ValueError(
                f"Draws outside the 6/{N_NUMBERS} + strong 1-{N_STRONG} format found "
                f"after {era_start}; the era boundary needs revisiting."
            )
    balls = np.sort(balls, axis=1)

    draws = Draws(df["date"].to_numpy(), balls, strong)
    logger.info("Loaded %d draws (%s to %s)", len(draws), df["date"].min().date(), df["date"].max().date())

    if clean_csv:
        out = pd.DataFrame(balls, columns=BALL_COLS)
        out.insert(0, "Date", df["date"].dt.strftime("%d/%m/%Y").to_numpy())
        out["Ball_Bonus"] = strong
        os.makedirs(os.path.dirname(clean_csv) or ".", exist_ok=True)
        out.to_csv(clean_csv, index=False)

    return draws


def next_draw_date(draws: Draws, lookback: int = 200) -> np.datetime64:
    """The date of the next scheduled draw after the last one on record.

    Inferred from the recent cadence rather than hard-coded, so it keeps working
    if the operator changes the schedule. Draw days are currently Tuesday and
    Saturday, with an intermittent Thursday series; only weekdays carrying a
    meaningful share of the recent draws count as scheduled.
    """
    recent = draws.dates[-lookback:].astype("datetime64[D]")
    weekday = (recent.astype(int) + 3) % 7  # Monday == 0
    counts = np.bincount(weekday, minlength=7)
    scheduled = set(np.flatnonzero(counts >= 0.1 * len(recent)).tolist())
    if not scheduled:  # pathological history; fall back to a week later
        return draws.dates[-1].astype("datetime64[D]") + np.timedelta64(7, "D")

    day = draws.dates[-1].astype("datetime64[D]")
    for _ in range(1, 15):
        day = day + np.timedelta64(1, "D")
        if int((day.astype(int) + 3) % 7) in scheduled:
            return day
    return day


def era_table(raw_csv: str = RAW_CSV) -> pd.DataFrame:
    """Per-year summary used by the report to justify the era cutoff."""
    df = _read_raw(raw_csv)
    df["year"] = df["date"].dt.year
    grp = df.groupby("year").agg(
        draws=("draw_id", "size"),
        max_ball=(BALL_COLS[-1], "max"),
        max_strong=("Ball_Bonus", "max"),
    )
    # How much of each year the legacy value-filter would have thrown away.
    df["dropped"] = (df[BALL_COLS].max(axis=1) > N_NUMBERS) | (df["Ball_Bonus"] > N_STRONG)
    grp["dropped_by_legacy_filter"] = df.groupby("year")["dropped"].sum()
    grp["kept_pct"] = 100 * (1 - grp["dropped_by_legacy_filter"] / grp["draws"])
    return grp.reset_index()
