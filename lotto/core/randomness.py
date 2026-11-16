"""Statistical tests for exploitable structure in the draw history.

Before trying to model a sequence it is worth asking whether it contains
anything to model. Each test returns a p-value under the null "the machine is
fair and draws are independent". Nonsignificance is not proof of randomness:
power depends on the sample size and the particular alternative being tested.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from lotto.core.data import N_DRAWN, N_NUMBERS, N_STRONG, Draws


def uniformity(draws: Draws) -> dict:
    """Marginal uniformity, adjusted for drawing six balls without replacement.

    Pearson's statistic has asymptotic distribution (N-k)/(N-1) * chi2(N-1),
    not chi2(N-1): each draw's indicators have negative covariance. See Genest,
    Lockhart and Stephens (2002), https://doi.org/10.1111/1467-9884.00315.
    """
    counts = np.bincount(draws.balls.ravel(), minlength=N_NUMBERS + 1)[1:]
    pearson = stats.chisquare(counts).statistic
    corrected = pearson * (N_NUMBERS - 1) / (N_NUMBERS - N_DRAWN)
    return {
        "test": "Ball frequency uniformity (without-replacement chi-square)",
        "statistic": float(corrected),
        "dof": N_NUMBERS - 1,
        "p_value": float(stats.chi2.sf(corrected, N_NUMBERS - 1)),
        "detail": (f"counts range {counts.min()}-{counts.max()}, "
                   f"expected {len(draws) * N_DRAWN / N_NUMBERS:.1f}; "
                   f"Pearson statistic multiplied by {(N_NUMBERS - 1) / (N_NUMBERS - N_DRAWN):.6f}"),
    }


def strong_uniformity(draws: Draws) -> dict:
    """Are the seven strong numbers drawn equally often? Plain chi-square over 7 cells."""
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
    smallest exact doubled-tail binomial p-value with a Bonferroni correction.
    Shared-ball pairs are dependent; this correction does not need independence.
    """
    m = draws.multi_hot.astype(np.int64)
    n = len(draws)
    co = m.T @ m
    iu = np.triu_indices(N_NUMBERS, k=1)
    observed = co[iu]

    # Each pair co-occurs with hypergeometric probability C(35,4)/C(37,6).
    p_pair = (N_DRAWN * (N_DRAWN - 1)) / (N_NUMBERS * (N_NUMBERS - 1))
    exp = n * p_pair
    sd = np.sqrt(n * p_pair * (1 - p_pair))
    z = (observed - exp) / sd
    p_single = np.minimum(1.0, 2 * np.minimum(stats.binom.cdf(observed, n, p_pair),
                                             stats.binom.sf(observed - 1, n, p_pair)))
    worst = int(np.argmin(p_single))
    n_tests = len(observed)
    p_family = min(1.0, n_tests * p_single[worst])
    return {
        "test": f"Pairwise co-occurrence ({n_tests} pairs, exact-binomial Bonferroni)",
        "statistic": float(z[worst]),
        "dof": None,
        "p_value": float(p_family),
        "detail": (
            f"most extreme pair {iu[0][worst] + 1}&{iu[1][worst] + 1}: "
            f"{int(observed[worst])} co-occurrences vs {exp:.1f} expected (z={z[worst]:.2f})"
        ),
    }


def sum_pmf() -> np.ndarray:
    """Exact distribution of the sum of six distinct numbers from 1..37.

    A dynamic program over the 37 numbers: `ways[size, total]` counts the
    subsets of that size adding up to that total. Sizes are visited downward
    so each number is used at most once.
    """
    max_sum = sum(range(N_NUMBERS - N_DRAWN + 1, N_NUMBERS + 1))
    ways = np.zeros((N_DRAWN + 1, max_sum + 1))
    ways[0, 0] = 1.0
    for number in range(1, N_NUMBERS + 1):
        for size in range(N_DRAWN, 0, -1):
            ways[size, number:] += ways[size - 1, :-number]
    return ways[N_DRAWN] / ways[N_DRAWN].sum()


def sum_distribution(draws: Draws, n_bins: int = 20) -> dict:
    """Chi-square of the draw sum against its exact null distribution.

    A single statistic that is sensitive to any drift in the overall level of
    the drawn numbers. Sums are grouped into `n_bins` bins of equal probability
    under the exact null, so every bin expects the same count.
    """
    observed = draws.balls.sum(axis=1)
    cdf = np.cumsum(sum_pmf())
    # Bin edges at equal-probability quantiles of the exact distribution.
    edges = np.unique(np.searchsorted(cdf, np.arange(1, n_bins) / n_bins))
    # A sum equal to an edge belongs to the bin that ends at that edge, because
    # each bin's expected share below is cdf[edge] minus the previous cdf[edge].
    bins = np.searchsorted(edges, observed, side="left")
    counts = np.bincount(bins, minlength=len(edges) + 1)
    bin_p = np.diff(np.concatenate([[0.0], cdf[edges], [1.0]]))
    res = stats.chisquare(counts, bin_p * len(observed))
    mean = float((np.arange(len(cdf)) * sum_pmf()).sum())
    return {
        "test": "Draw-sum distribution (chi-square vs exact null)",
        "statistic": float(res.statistic),
        "dof": len(counts) - 1,
        "p_value": float(res.pvalue),
        "detail": f"observed mean sum {observed.mean():.1f} vs {mean:.1f} expected; {len(counts)} bins",
    }


def drift(draws: Draws) -> dict:
    """Split the history in half; do the frequency profiles agree?

    Machines are replaced and balls wear. If a bias existed it would most
    likely be non-stationary, which a whole-history chi-square would dilute.
    """
    half = len(draws) // 2
    if half < 1:
        raise ValueError("Frequency drift needs at least two draws")
    a = np.bincount(draws.balls[:half].ravel(), minlength=N_NUMBERS + 1)[1:]
    b = np.bincount(draws.balls[half:].ravel(), minlength=N_NUMBERS + 1)[1:]
    # Under independent fair draws, the difference of frequency vectors has
    # the same without-replacement covariance as uniformity(), scaled by
    # 1/n_a + 1/n_b. Its sum-zero subspace has 36 equal eigenvalues.
    n_b = len(draws) - half
    p_hit = N_DRAWN / N_NUMBERS
    variance = p_hit * (1 - p_hit) * N_NUMBERS / (N_NUMBERS - 1) * (1 / half + 1 / n_b)
    difference = a / half - b / n_b
    chi2 = float(difference @ difference / variance)
    p = stats.chi2.sf(chi2, N_NUMBERS - 1)
    return {
        "test": "First half vs second half frequency drift (without-replacement Wald)",
        "statistic": float(chi2),
        "dof": N_NUMBERS - 1,
        "p_value": float(p),
        "detail": f"{half} vs {n_b} draws; asymptotic covariance under independent uniform 6/37 draws",
    }


def incompressibility(draws: Draws, n_sim: int = 399, seed: int = 0) -> dict:
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

    zlib rather than lzma, deliberately: encoder setup dominates on a ~5 KB
    payload, and the 201 lzma preset-9 encoders this null needs cost minutes on
    a CI runner against ~10 ms for zlib. The teeth are unchanged — the
    function-generated control still collapses to 15.5% of the entropy floor
    under zlib (10.4% under lzma) while the real archive sits at 100.4% either
    way — and observed and null always use the same compressor, so the
    calibration is internally consistent.

    `n_sim` sets the smallest reportable p-value, 1 / (1 + n_sim); it must stay
    below alpha divided by the number of tests in `ALL_TESTS`.
    """
    import zlib
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

    compress = lambda b: len(zlib.compress(b, 9))  # noqa: E731
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
            f"zlib: {observed * 8 / n:.2f} bits/draw vs entropy floor {floor_bits:.2f}; "
            f"fair-sim mean {sims.mean() * 8 / n:.2f}"
        ),
    }


def surrogate_determinism(draws: Draws, n_sim: int = 399, n_eval: int = 120, seed: int = 0) -> dict:
    """The canonical chaos-vs-noise test: does the *ordering* of draws predict?

    Lorenz's method of analogues is scored on the real sequence, then on
    shuffled surrogates — permutation destroys any dynamics while preserving
    the composition exactly. If the draws were successive states of one
    evolving system, similar past windows would have similar futures and the
    real ordering would out-predict its shuffles. The negative control in
    `test_surrogate_test_detects_determinism` holds a deterministic cycle,
    which this test flags at the smallest reachable p.

    `n_sim` sets that smallest reportable p-value, 1 / (1 + n_sim). It must stay
    below alpha divided by the number of tests in `ALL_TESTS`, or Holm's
    correction could never flag this test whatever the data did.
    """
    from lotto.approaches.chaos import analogue_skill

    observed = analogue_skill(draws.multi_hot, n_eval=n_eval)
    rng = np.random.default_rng(seed)
    sims = np.empty(n_sim)
    for i in range(n_sim):
        sims[i] = analogue_skill(draws.multi_hot[rng.permutation(len(draws))], n_eval=n_eval)
    p = (1 + np.sum(sims >= observed)) / (1 + n_sim)
    return {
        "test": "Surrogate-data determinism (method of analogues vs shuffles)",
        "statistic": float(observed),
        "dof": None,
        "p_value": float(p),
        "detail": (
            f"real ordering {observed:.3f} matches vs {sims.mean():.3f} mean over "
            f"{n_sim} shuffled surrogates (chance {36 / 37:.3f})"
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
    surrogate_determinism,
)


def run_all(draws: Draws) -> list[dict]:
    """Run every test in ``ALL_TESTS`` and return their result dicts in that order."""
    return [test(draws) for test in ALL_TESTS]
