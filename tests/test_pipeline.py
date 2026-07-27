"""Guards on the parts that were previously wrong.

Each test here corresponds to a specific bug the rewrite fixed, so a regression
shows up as a named failure rather than as a quietly wrong number in a report.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import backtest, data, metrics, predictors, randomness


@pytest.fixture(scope="module")
def draws():
    return data.load(clean_csv=None)


def fake_draws(n=400, seed=0):
    """A synthetic fair 6/37 history, for tests that must not depend on the archive."""
    rng = np.random.default_rng(seed)
    balls = np.sort(rng.random((n, 37)).argsort(axis=1)[:, :6] + 1, axis=1)
    strong = rng.integers(1, 8, size=n)
    dates = np.datetime64("2012-01-01", "D") + np.arange(n) * 3
    return data.Draws(dates.astype("datetime64[ns]"), balls, strong)


def test_draws_are_chronological(draws):
    # The archive ships newest-first; loading it as-is trains models backwards.
    assert (np.diff(draws.dates.astype("datetime64[D]").astype(int)) > 0).all()


def test_only_current_game_format(draws):
    assert draws.balls.min() >= 1 and draws.balls.max() <= data.N_NUMBERS
    assert draws.strong.min() >= 1 and draws.strong.max() <= data.N_STRONG
    assert len(draws) > 1000


def test_no_outcome_based_filtering(draws):
    """Every number must be reachable; a value filter would truncate the top."""
    counts = np.bincount(draws.balls.ravel(), minlength=data.N_NUMBERS + 1)[1:]
    assert (counts > 0).all()
    # A value filter shows up as the highest numbers being rare. Under fair
    # draws no number should sit anywhere near half the expected count.
    assert counts.min() > 0.5 * counts.mean()


def test_multi_hot_round_trip():
    d = fake_draws(50)
    m = d.multi_hot
    assert m.shape == (50, 37)
    assert (m.sum(axis=1) == 6).all()
    assert set(np.flatnonzero(m[0]) + 1) == set(d.balls[0].tolist())


def test_baseline_constant_matches_theory():
    # E[matches] for a random ticket is the hypergeometric mean, 6*6/37.
    from scipy import stats

    assert metrics.BASELINE_MATCHES == pytest.approx(stats.hypergeom(37, 6, 6).mean())


def test_uniform_scores_hit_baseline_log_loss():
    uniform = np.full(37, 1 / 37)
    assert metrics.log_loss(uniform, np.array([1, 2, 3, 4, 5, 6])) == pytest.approx(
        metrics.BASELINE_LOGLOSS
    )


def test_metrics_are_order_invariant():
    """The old per-position metric rewarded knowing the sort order, not the numbers."""
    rng = np.random.default_rng(1)
    scores = rng.random(37)
    balls = np.array([3, 9, 14, 22, 30, 31])
    shuffled = rng.permutation(balls)
    assert metrics.matches(scores, balls) == metrics.matches(scores, shuffled)
    assert metrics.log_loss(scores, balls) == pytest.approx(metrics.log_loss(scores, shuffled))


def test_random_predictor_lands_on_baseline():
    d = fake_draws(900, seed=5)
    r = backtest.walk_forward(predictors.UniformRandom(seed=3), d, n_test=400, min_history=100)
    assert abs(r.summary["z_score"]) < 3.5


def test_walk_forward_never_sees_the_future():
    """A predictor that peeks would score far above chance; assert it cannot."""
    seen: list[int] = []

    class Spy(predictors.Predictor):
        name = "spy"

        def scores(self, history):
            seen.append(len(history))
            return np.zeros(37), np.zeros(7)

    d = fake_draws(300)
    backtest.walk_forward(Spy(), d, n_test=50, min_history=100)
    # Histories are strictly increasing and always stop before the target draw.
    assert seen == list(range(250, 300))


def test_holm_bonferroni_is_monotone():
    verdict = metrics.holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.4})
    assert verdict["a"] is True
    assert verdict["c"] is False
    # Once a hypothesis fails, every less-significant one must fail too.
    assert not (verdict["b"] is False and verdict["c"] is True)


def test_randomness_tests_pass_on_fair_synthetic_data():
    """The test suite must not fire on data that is random by construction."""
    d = fake_draws(1500, seed=11)
    results = randomness.run_all(d)
    assert len(results) == len(randomness.ALL_TESTS)
    corrected = metrics.holm_bonferroni({t["test"]: t["p_value"] for t in results})
    assert not any(corrected.values()), [t for t in results if corrected[t["test"]]]


@pytest.mark.parametrize("predictor", predictors.statistical_suite(), ids=lambda p: p.name)
def test_predictor_contract(predictor):
    d = fake_draws(300)
    ball_scores, strong_scores = predictor.scores(d)
    assert ball_scores.shape == (37,)
    assert strong_scores.shape == (7,)
    assert np.isfinite(strong_scores).all()
    picks = metrics.top_k(np.nan_to_num(ball_scores, neginf=-1e18))
    assert len(set(picks.tolist())) == 6


def test_log_loss_is_withheld_from_rank_only_predictors():
    """Normalising ranks or negated counts into a distribution is meaningless.

    The cold-frequency predictor is the sharp case: its scores are all negative,
    so clipping collapses them to the uniform prior and it would report exactly
    the baseline log loss regardless of how it ranked the numbers.
    """
    d = fake_draws(400, seed=4)
    cold = backtest.walk_forward(
        predictors.Frequency(cold=True), d, n_test=30, min_history=100
    )
    hot = backtest.walk_forward(predictors.Frequency(), d, n_test=30, min_history=100)
    assert cold.summary["mean_log_loss"] is None
    assert hot.summary["mean_log_loss"] is not None


def test_rank_only_predictors_declare_themselves():
    ranked = {"Overdue (longest absence)", "Rank ensemble", "Sum-targeted combination"}
    for p in predictors.statistical_suite():
        if p.name in ranked:
            assert not p.emits_probabilities, p.name


def test_verdict_covers_randomness_tests_too(tmp_path):
    """A significant randomness test must flip the headline even if no strategy wins."""
    from bench import report

    d = fake_draws(300, seed=6)
    results = backtest.run_suite([predictors.UniformRandom(seed=1)], d, n_test=30)
    assert not any(r.summary["significant_after_correction"] for r in results)

    rigged = [{"test": "rigged", "statistic": 99.0, "dof": 1, "p_value": 1e-9, "detail": ""}]
    html = report.build(
        d,
        rigged,
        results,
        {"n_trials": 10, "mean": 0.97, "std": 0.05, "p95": 1.05, "max": 1.1,
         "samples": np.full(10, 0.97)},
        [],
        __import__("pandas").DataFrame(
            {"year": [2012], "draws": [1], "max_ball": [37], "max_strong": [7],
             "dropped_by_legacy_filter": [0], "kept_pct": [100.0]}
        ),
        out_path=str(tmp_path / "report.html"),
    )
    assert "signal detected" in html and "no signal detected" not in html


def test_neural_predictor_end_to_end():
    from bench.nn import NeuralPredictor

    d = fake_draws(300, seed=2)
    p = NeuralPredictor(arch="mlp", epochs=3, refit_every=100, window=10)
    r = backtest.walk_forward(p, d, n_test=20, min_history=100)
    assert r.summary["n_draws"] == 20
    assert 0 <= r.summary["mean_matches"] <= 6


def test_nn_windows_are_causal():
    from bench.nn import make_windows

    d = fake_draws(60)
    X, y, s = make_windows(d, window=10)
    assert len(X) == 50
    # y[i] must be the draw immediately after the window that produced X[i],
    # and must not appear inside it.
    assert np.array_equal(y[0], d.multi_hot[10])
    assert not np.array_equal(X[0][-1][:37], y[0])


def test_empty_era_fails_with_a_clear_message(tmp_path):
    """A truncated or restructured upstream archive must not fail opaquely."""
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "id,date,1,2,3,4,5,6,strong,w1,w2\n"
        "1,03/09/1968,3,14,18,22,25,33,2,0,0\n",
        encoding="latin-1",
    )
    with pytest.raises(ValueError, match="No draws parsed"):
        data.load(raw_csv=str(raw), clean_csv=None)


def test_hostile_csv_content_never_reaches_the_report(tmp_path):
    """The archive is re-downloaded in CI, so its content is untrusted input.

    Rows carrying markup must be dropped by type coercion rather than rendered.
    """
    raw = tmp_path / "raw.csv"
    rows = ["id,date,1,2,3,4,5,6,strong,w1,w2"]
    rows.append("9,<script>alert(1)</script>,10,18,20,21,29,34,2,0,0")
    rows.append("8,11/10/2025,<img src=x onerror=alert(2)>,18,20,21,29,34,2,0,0")
    rng = np.random.default_rng(3)
    for i in range(400):
        balls = np.sort(rng.random(37).argsort()[:6] + 1)
        day, month, year = 1 + i % 28, 1 + i % 12, 2013 + i % 10
        rows.append(
            f"{100 + i},{day:02d}/{month:02d}/{year},"
            + ",".join(str(b) for b in balls)
            + f",{rng.integers(1, 8)},0,0"
        )
    raw.write_text("\n".join(rows) + "\n", encoding="latin-1")

    d = data.load(raw_csv=str(raw), clean_csv=None)
    assert len(d) == 400  # both hostile rows coerced to NaN/NaT and dropped
    assert d.balls.min() >= 1 and d.balls.max() <= data.N_NUMBERS


# --- perfect-fit interpolators (bench/spacetime.py) -------------------------


def test_perfect_fits_are_actually_perfect():
    """The premise is granted, not strawmanned: the fits must be exact."""
    from bench import spacetime

    d = fake_draws(400, seed=13)
    dft = spacetime.PerfectFitDFT()
    assert np.abs(dft.reconstruct(d) - d.multi_hot).max() < 1e-6

    # Spline: zero error at every knot.
    from scipy.interpolate import CubicSpline

    t = np.arange(len(d), dtype=float)
    for j in (0, 18, 36):
        cs = CubicSpline(t, d.multi_hot[:, j])
        assert np.abs(cs(t) - d.multi_hot[:, j]).max() < 1e-12


def test_dft_extrapolation_is_the_first_draw():
    """f(n) = f(0) by periodicity — computed from the coefficients, not assumed."""
    from bench import spacetime

    d = fake_draws(200, seed=14)
    n = len(d)
    coef = np.fft.rfft(d.multi_hot[:, 7])
    k = np.arange(len(coef))
    # Real signal: f(t) = Re[ c0 + 2*sum_{k>=1} c_k e^{2pi i k t / n} ] / n  (n even: Nyquist term once)
    weights = np.ones(len(coef)) * 2.0
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    f_n = np.real(np.sum(weights * coef * np.exp(2j * np.pi * k * n / n))) / n
    # multi_hot is float32, so the round-trip carries ~1e-9 of noise.
    assert f_n == pytest.approx(d.multi_hot[0, 7], abs=1e-6)

    ball_scores, _ = spacetime.PerfectFitDFT().scores(d)
    assert np.array_equal(np.sort(np.argsort(-ball_scores)[:6]) + 1, d.balls[0])


def test_perfect_fit_families_disagree_about_the_future():
    """Same history, three exact fits, three different next draws.

    This is the identifiability argument made constructive: if the data
    determined the continuation, exact fits would have to agree on it.
    """
    from bench import metrics, spacetime

    d = fake_draws(400, seed=15)
    tickets = []
    for p in spacetime.spacetime_suite():
        scores, _ = p.scores(d)
        tickets.append(frozenset((metrics.top_k(scores) + 1).tolist()))
    assert len(set(tickets)) >= 2, "at least two families must disagree"


def test_perfect_fit_walk_forward_is_chance():
    from bench import spacetime

    d = fake_draws(700, seed=16)
    for p in spacetime.spacetime_suite():
        r = backtest.walk_forward(p, d, n_test=150, min_history=400)
        se = np.sqrt(metrics.BASELINE_MATCH_VAR / 150)
        assert abs(r.summary["mean_matches"] - metrics.BASELINE_MATCHES) < 4 * se, p.name


def test_compression_detects_a_generator():
    """Negative control: the incompressibility test must fire on a sequence that
    actually has a generating function, else 'no structure found' means nothing."""
    n = 800
    balls = np.array([sorted(((np.arange(6) * 6 + t) % 37) + 1) for t in range(n)])
    strong = (np.arange(n) % 7) + 1
    dates = np.datetime64("2012-01-03", "D") + np.arange(n) * 3
    functional = data.Draws(dates.astype("datetime64[ns]"), balls, strong)

    t = randomness.incompressibility(functional, n_sim=40)
    assert t["p_value"] < 0.05, "a generated sequence must be flagged as compressible"
    assert t["statistic"] < 10, "bits/draw should collapse far below the 23.96 floor"


def test_compression_passes_fair_data():
    d = fake_draws(800, seed=17)
    t = randomness.incompressibility(d, n_sim=40)
    assert t["p_value"] > 0.05


# --- chaos theory (bench/chaos.py) ------------------------------------------


def test_correlation_dimension_separates_chaos_from_noise():
    """The instrument check: GP dimension must saturate on a chaotic series and
    keep climbing with embedding dimension on iid noise."""
    from bench import chaos

    rng = np.random.default_rng(4)
    lx = np.empty(1600)
    lx[0] = 0.61234
    for i in range(1, len(lx)):
        lx[i] = 4 * lx[i - 1] * (1 - lx[i - 1])  # logistic map, D2 = 1
    noise = rng.normal(size=1600)

    assert chaos.correlation_dimension(lx, m=4) < 1.8, "chaos must saturate near 1"
    assert chaos.correlation_dimension(noise, m=4) > 2.5, "noise must fill the embedding"


def test_draw_sums_have_no_attractor():
    """The real archive must look like the noise column, not the chaos column."""
    from bench import chaos

    d = data.load(clean_csv=None)
    sums = d.balls.sum(axis=1).astype(float)
    assert chaos.correlation_dimension(sums, m=4) > 2.5


def test_analogues_nail_a_deterministic_sequence():
    """Positive control: on a periodic sequence, similar pasts genuinely have
    similar futures, and the method must recover them perfectly."""
    from bench import chaos

    d = fake_draws(700, seed=18)
    cyc = np.tile(d.multi_hot[:7], (100, 1))[:700]
    assert chaos.analogue_skill(cyc, n_eval=100) == pytest.approx(6.0)


def test_analogues_walk_forward_is_chance_on_fair_draws():
    from bench.chaos import MethodOfAnalogues

    d = fake_draws(900, seed=19)
    r = backtest.walk_forward(MethodOfAnalogues(), d, n_test=150, min_history=500)
    se = np.sqrt(metrics.BASELINE_MATCH_VAR / 150)
    assert abs(r.summary["mean_matches"] - metrics.BASELINE_MATCHES) < 4 * se
    # Smoothed successor frequencies are genuine probabilities summing to 6.
    b, s = MethodOfAnalogues().scores(d)
    assert b.sum() == pytest.approx(6.0, abs=1e-9)
    assert s.sum() == pytest.approx(1.0, abs=1e-9)
    assert (b > 0).all() and (b < 1).all()


def test_surrogate_test_detects_determinism():
    """Negative control for the negative result: a deterministic cycle must be
    flagged at the smallest reachable p, and fair draws must not be."""
    d = fake_draws(800, seed=20)
    cyc_balls = np.tile(d.balls[:7], (120, 1))[:800]
    cyc = data.Draws(d.dates, cyc_balls, d.strong)

    flagged = randomness.surrogate_determinism(cyc, n_sim=30)
    assert flagged["p_value"] <= 1 / 31 + 1e-9

    fair = randomness.surrogate_determinism(d, n_sim=30)
    assert fair["p_value"] > 0.05
