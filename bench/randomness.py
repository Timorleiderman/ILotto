"""Statistical tests for exploitable structure in the draw history.

Before trying to model a sequence it is worth asking whether it contains
anything to model. Each test returns a p-value under the null "the machine is
fair and draws are independent". If none of these fire, no architecture will
help, and the honest report says so.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .data import N_DRAWN, N_NUMBERS, N_STRONG, Draws


def uniformity(draws: Draws) -> dict:
    """Chi-square goodness-of-fit on the marginal frequency of each number."""
    counts = np.bincount(draws.balls.ravel(), minlength=N_NUMBERS + 1)[1:]
    res = stats.chisquare(counts)
    return {
        "test": "Ball frequency uniformity (chi-square)",
        "statistic": float(res.statistic),
        "dof": N_NUMBERS - 1,
        "p_value": float(res.pvalue),
        "detail": f"counts range {counts.min()}-{counts.max()}, expected {len(draws) * N_DRAWN / N_NUMBERS:.1f}",
    }


def strong_uniformity(draws: Draws) -> dict:
    counts = np.bincount(draws.strong, minlength=N_STRONG + 1)[1:]
    res = stats.chisquare(counts)
    return {
        "test": "Strong-number uniformity (chi-square)",
        "statistic": float(res.statistic),
        "dof": N_STRONG - 1,
        "p_value": float(res.pvalue),
        "detail": f"counts range {counts.min()}-{counts.max()}",
    }


def serial_independence(draws: Draws) -> dict:
    """Does a number appearing make it more or less likely to appear next time?

    Builds a 2x2 table of (in draw t) x (in draw t+1) pooled over all numbers
    and applies a chi-square test of independence. This is the cleanest test of
    the "hot number" and "due number" folklore at the same time: hot predicts
    positive association, due predicts negative.
    """
    m = draws.multi_hot.astype(bool)
    prev, nxt = m[:-1].ravel(), m[1:].ravel()
    table = np.array(
        [
            [np.sum(~prev & ~nxt), np.sum(~prev & nxt)],
            [np.sum(prev & ~nxt), np.sum(prev & nxt)],
        ]
    )
    chi2, p, _, _ = stats.chi2_contingency(table)
    p_repeat = table[1, 1] / table[1].sum()
    return {
        "test": "Lag-1 serial independence (hot/due folklore)",
        "statistic": float(chi2),
        "dof": 1,
        "p_value": float(p),
        "detail": f"P(drawn again | drawn) = {p_repeat:.4f} vs {N_DRAWN / N_NUMBERS:.4f} expected",
    }


def gap_distribution(draws: Draws) -> dict:
    """Are the gaps between successive appearances geometric, as fairness implies?

    A biased machine shows up as over- or under-dispersed gaps even when the
    marginal frequencies look fine, so this catches things `uniformity` misses.
    """
    m = draws.multi_hot.astype(bool)
    gaps: list[int] = []
    for num in range(N_NUMBERS):
        idx = np.flatnonzero(m[:, num])
        gaps.extend(np.diff(idx).tolist())
    gaps_arr = np.asarray(gaps)

    p_hit = N_DRAWN / N_NUMBERS
    # Bucket gaps and compare observed vs geometric expectation.
    max_bucket = 12
    obs = np.bincount(np.clip(gaps_arr, 1, max_bucket), minlength=max_bucket + 1)[1:]
    ks = np.arange(1, max_bucket + 1)
    exp_p = p_hit * (1 - p_hit) ** (ks - 1)
    exp_p[-1] = (1 - p_hit) ** (max_bucket - 1)  # tail mass
    exp = exp_p / exp_p.sum() * obs.sum()
    res = stats.chisquare(obs, exp)
    return {
        "test": "Appearance-gap distribution vs geometric",
        "statistic": float(res.statistic),
        "dof": max_bucket - 1,
        "p_value": float(res.pvalue),
        "detail": f"{len(gaps_arr)} gaps, mean {gaps_arr.mean():.2f} vs {1 / p_hit:.2f} expected",
    }


def pair_independence(draws: Draws) -> dict:
    """Do any *pairs* of numbers co-occur more than chance?

    Marginal uniformity would not reveal a machine that keeps two balls near
    each other. 666 pairs are tested at once, so the reported p-value is the
    smallest one after a Sidak correction for the whole family.
    """
    m = draws.multi_hot
    n = len(draws)
    co = m.T @ m
    iu = np.triu_indices(N_NUMBERS, k=1)
    observed = co[iu]

    # Each pair co-occurs with hypergeometric probability C(35,4)/C(37,6).
    p_pair = (N_DRAWN * (N_DRAWN - 1)) / (N_NUMBERS * (N_NUMBERS - 1))
    exp = n * p_pair
    sd = np.sqrt(n * p_pair * (1 - p_pair))
    z = (observed - exp) / sd
    worst = int(np.argmax(np.abs(z)))
    p_single = 2 * stats.norm.sf(abs(z[worst]))
    n_tests = len(observed)
    p_family = 1 - (1 - p_single) ** n_tests
    return {
        "test": f"Pairwise co-occurrence ({n_tests} pairs, Sidak-corrected)",
        "statistic": float(z[worst]),
        "dof": None,
        "p_value": float(min(p_family, 1.0)),
        "detail": (
            f"most extreme pair {iu[0][worst] + 1}&{iu[1][worst] + 1}: "
            f"{int(observed[worst])} co-occurrences vs {exp:.1f} expected (z={z[worst]:.2f})"
        ),
    }


def sum_distribution(draws: Draws) -> dict:
    """Kolmogorov-Smirnov of the draw sum against its exact null distribution.

    A single statistic that is sensitive to any drift in the overall level of
    the drawn numbers. The null is obtained by Monte Carlo over real 6/37 draws.
    """
    rng = np.random.default_rng(0)
    observed = draws.balls.sum(axis=1)
    n_sim = 100_000
    # argsort of uniforms gives an independent random permutation per row, which
    # is how you draw without replacement in a vectorised way.
    picks = rng.random((n_sim, N_NUMBERS)).argsort(axis=1)[:, :N_DRAWN] + 1
    sim = picks.sum(axis=1)
    res = stats.ks_2samp(observed, sim)
    return {
        "test": "Draw-sum distribution (two-sample KS vs simulated fair draws)",
        "statistic": float(res.statistic),
        "dof": None,
        "p_value": float(res.pvalue),
        "detail": f"observed mean sum {observed.mean():.1f} vs {sim.mean():.1f} simulated",
    }


def drift(draws: Draws) -> dict:
    """Split the history in half; do the frequency profiles agree?

    Machines are replaced and balls wear. If a bias existed it would most
    likely be non-stationary, which a whole-history chi-square would dilute.
    """
    half = len(draws) // 2
    a = np.bincount(draws.balls[:half].ravel(), minlength=N_NUMBERS + 1)[1:]
    b = np.bincount(draws.balls[half:].ravel(), minlength=N_NUMBERS + 1)[1:]
    chi2, p, _, _ = stats.chi2_contingency(np.vstack([a, b]))
    return {
        "test": "First half vs second half frequency drift",
        "statistic": float(chi2),
        "dof": N_NUMBERS - 1,
        "p_value": float(p),
        "detail": f"{half} vs {len(draws) - half} draws",
    }


def incompressibility(draws: Draws, n_sim: int = 200, seed: int = 0) -> dict:
    """Kolmogorov-flavoured test: does the archive contain a generating function?

    If the draws were produced by any function simple enough for a
    general-purpose compressor to model — periodicity, arithmetic structure,
    drifting rates — the canonically packed archive would compress below its
    entropy floor of log2(C(37,6) * 7) = 23.96 bits/draw. A control sequence
    actually generated by a simple arithmetic function compresses to ~10% of
    that floor, so the test has teeth; see `test_compression_detects_a_generator`.

    The p-value is Monte-Carlo: the observed compressed size ranked against the
    same compressor on simulated fair histories of the same length. Small p
    means "more compressible than fair", i.e. structure.
    """
    import lzma
    from math import comb, log2

    def pack(balls: np.ndarray, strong: np.ndarray) -> bytes:
        out = bytearray()
        for row, s_ in zip(balls, strong):
            r, prev = 0, 0
            for i, b in enumerate(row):
                for x in range(prev + 1, int(b)):
                    r += comb(N_NUMBERS - x, N_DRAWN - 1 - i)
                prev = int(b)
            out += (r * N_STRONG + int(s_) - 1).to_bytes(3, "big")
        return bytes(out)

    compress = lambda b: len(lzma.compress(b, preset=9))  # noqa: E731
    observed = compress(pack(draws.balls, draws.strong))

    rng = np.random.default_rng(seed)
    n = len(draws)
    sims = np.empty(n_sim)
    for k in range(n_sim):
        picks = rng.random((n, N_NUMBERS)).argsort(axis=1)[:, :N_DRAWN] + 1
        sims[k] = compress(pack(np.sort(picks, axis=1), rng.integers(1, N_STRONG + 1, n)))
    p = (1 + np.sum(sims <= observed)) / (1 + n_sim)

    floor_bits = log2(comb(N_NUMBERS, N_DRAWN)) + log2(N_STRONG)
    return {
        "test": "Incompressibility (no generating function within compressor reach)",
        "statistic": float(observed * 8 / n),
        "dof": None,
        "p_value": float(p),
        "detail": (
            f"lzma: {observed * 8 / n:.2f} bits/draw vs entropy floor {floor_bits:.2f}; "
            f"fair-sim mean {sims.mean() * 8 / n:.2f}"
        ),
    }


ALL_TESTS = (
    uniformity,
    strong_uniformity,
    serial_independence,
    gap_distribution,
    pair_independence,
    sum_distribution,
    drift,
    incompressibility,
)


def run_all(draws: Draws) -> list[dict]:
    return [test(draws) for test in ALL_TESTS]
