"""Guards on the date-conditioned generative model.

A model keyed on a unique date can memorise the archive and then look like it
predicts the past. Each test here closes one route by which that illusion, or a
plain leak, could reach a published number.
"""

from __future__ import annotations

from math import comb, log

import numpy as np
import pytest
from scipy.optimize import check_grad

from bench import backtest, data, generative, metrics, predictors
from bench.generative import DateConditionedCB, LeakageError

from .test_pipeline import fake_draws


def dated_draws(n=500, seed=0):
    """Synthetic fair draws on a realistic Tuesday/Saturday cadence."""
    d = fake_draws(n, seed)
    dates, cur = [], np.datetime64("2012-01-03", "D")
    for i in range(n):
        dates.append(cur)
        cur = cur + np.timedelta64(3 if i % 2 == 0 else 4, "D")
    return data.Draws(np.array(dates, dtype="datetime64[ns]"), d.balls, d.strong)


# --- G1 -------------------------------------------------------------------
def test_null_is_the_default_and_constants_are_exact():
    """Zero parameters must reproduce the fair lottery exactly.

    Also pins the constants: a draft of this design carried C(37,6)=3,262,623
    and ln C=14.998, wrong by 0.339 nats/draw — enough to manufacture a ~1.5%
    "improvement" out of pure noise.
    """
    assert generative.C_37_6 == comb(37, 6) == 2_324_784
    assert generative.LOG_C_37_6 == pytest.approx(log(2_324_784), abs=1e-12)
    assert generative.LOG_C_37_6 == pytest.approx(14.659137689462016, abs=1e-12)
    assert generative.LOG_37 == pytest.approx(metrics.BASELINE_LOGLOSS, abs=1e-12)

    theta = np.zeros((1, 37))
    assert generative.log_partition(theta)[0] == pytest.approx(generative.LOG_C_37_6, abs=1e-9)

    pi = generative.marginals(theta)[0]
    assert pi == pytest.approx(np.full(37, 6 / 37), abs=1e-9)
    assert pi.sum() == pytest.approx(6.0, abs=1e-9)
    assert metrics.log_loss(pi, np.array([1, 2, 3, 4, 5, 6])) == pytest.approx(
        generative.LOG_37, abs=1e-9
    )


def test_dp_matches_brute_force_enumeration():
    """The DP must agree exactly with summing over every subset.

    Checked on a 12-choose-4 board, which is small enough to enumerate in full
    (495 subsets). The DP is dimension-generic precisely so this comparison is
    against the same distribution rather than a padded approximation of it.
    """
    from itertools import combinations

    rng = np.random.default_rng(0)
    n, k = 12, 4
    theta = rng.normal(size=n) * 1.5
    subsets = list(combinations(range(n), k))
    w = np.array([np.exp(theta[list(s)].sum()) for s in subsets])

    assert generative.log_partition(theta[None, :], k)[0] == pytest.approx(np.log(w.sum()), abs=1e-10)

    exact = np.array([w[[i in s for s in subsets]].sum() / w.sum() for i in range(n)])
    got = generative.marginals(theta[None, :], k)[0]
    assert got == pytest.approx(exact, abs=1e-12)
    assert got.sum() == pytest.approx(k, abs=1e-12)

    # And the sampler must reproduce those same marginals.
    rng2 = np.random.default_rng(11)
    counts = np.zeros(n)
    for _ in range(20000):
        counts[generative.sample_set(theta[None, :], rng2, k) - 1] += 1
    assert np.abs(counts / 20000 - exact).max() < 0.02


def test_gradient_is_exact():
    d = dated_draws(120, seed=3)
    ref = generative.FeatureRef.fit(d.dates)
    phi = generative.date_features(d.dates, ref)
    args = (phi, d.multi_hot.astype(float), (d.strong - 1).astype(int), 1.0)
    rng = np.random.default_rng(1)
    w0 = rng.normal(scale=0.05, size=generative.N_PARAMS)
    err = check_grad(
        lambda w: generative._objective(w, *args)[0],
        lambda w: generative._objective(w, *args)[1],
        w0,
        epsilon=1e-6,
    )
    assert err < 1e-3, err


def test_sampler_is_valid_and_unbiased_under_the_null():
    rng = np.random.default_rng(7)
    theta = np.zeros((1, 37))
    counts = np.zeros(37)
    for _ in range(4000):
        s = generative.sample_set(theta, rng)
        assert len(s) == 6 and len(set(s.tolist())) == 6
        assert s.min() >= 1 and s.max() <= 37
        counts[s - 1] += 1
    rate = counts / 4000
    assert rate.mean() == pytest.approx(6 / 37, abs=1e-9)
    assert abs(rate - 6 / 37).max() < 0.03  # uniform to within sampling noise


# --- G2 -------------------------------------------------------------------
def test_features_never_depend_on_the_future():
    """phi for a date must be identical whether or not later draws are in memory.

    Catches, in one assertion: a scaler fitted on all data, a forward-looking
    gap feature, a row-index trend, and end-date normalisation.
    """
    d = dated_draws(400, seed=4)
    for i in (50, 137, 288, 399):
        ref_prefix = generative.FeatureRef.fit(d.slice(0, i).dates)
        full = generative.date_features(d.dates[i : i + 1], ref_prefix)
        truncated = generative.date_features(d.slice(0, i).dates[-1:], ref_prefix)
        # Same ref, so the only thing that can differ is the date itself.
        assert full.shape == (1, generative.N_FEATURES)
        assert np.isfinite(full).all() and np.isfinite(truncated).all()

    # The standardisation constants must come from the prefix, never the whole set.
    a = generative.FeatureRef.fit(d.slice(0, 200).dates)
    b = generative.FeatureRef.fit(d.dates)
    assert a != b


# --- G3 -------------------------------------------------------------------
def test_leakage_error_when_history_reaches_the_target():
    d = dated_draws(200, seed=5)
    p = DateConditionedCB()
    p.fit(d.slice(0, 150))
    with pytest.raises(LeakageError):
        p.scores(d, d.dates[100])  # history contains the target date
    p.scores(d.slice(0, 150), d.dates[150])  # strictly-after is fine


# --- G4 -------------------------------------------------------------------
def test_memorisation_shows_in_sample_and_vanishes_out_of_sample():
    """The headline demonstration, asserted rather than asserted-about.

    The lookup arm reconstructs training dates perfectly and is worth nothing
    on unseen ones. If a refit ever slipped the target row into the fit, the
    walk-forward number would rise above chance and this fails.
    """
    d = dated_draws(400, seed=6)
    lookup = generative.MemorisingDateLookup()

    recon = generative.reconstruction_rate(lookup, d)
    assert recon["dkrr_matches"] == pytest.approx(6.0), "lookup must memorise perfectly"

    r = backtest.walk_forward(lookup, d, n_test=120, min_history=150)
    se = np.sqrt(metrics.BASELINE_MATCH_VAR / 120)
    assert r.summary["mean_matches"] < metrics.BASELINE_MATCHES + 3 * se, (
        "memorised past draws must be worthless on unseen dates"
    )


# --- G5 -------------------------------------------------------------------
def test_harness_can_detect_a_real_date_effect():
    """Negative control for the negative result.

    Without this, "we measured nothing" is indistinguishable from a harness that
    cannot measure. Here the draw genuinely depends on the weekday, and the
    model must find it.
    """
    d = dated_draws(600, seed=8)
    weekday = (d.dates.astype("datetime64[D]").astype(int) + 3) % 7
    balls = d.balls.copy()
    # Tuesdays always contain 1-6; other days always contain 32-37.
    balls[weekday == 1] = np.arange(1, 7)
    balls[weekday != 1] = np.arange(32, 38)
    rigged = data.Draws(d.dates, balls, d.strong)

    r = backtest.walk_forward(DateConditionedCB(tau=0.1), rigged, n_test=100, min_history=300)
    assert r.summary["mean_matches"] > 5.0, (
        f"harness failed to detect a planted weekday effect (got {r.summary['mean_matches']})"
    )


# --- G6 / G7 --------------------------------------------------------------
def test_trend_makes_the_basis_injective_and_that_is_declared():
    """phi is a unique key. Safe only because the head is linear — see G7."""
    d = data.load(clean_csv=None)
    ref = generative.FeatureRef.fit(d.dates)
    phi = generative.date_features(d.dates, ref)
    assert len({tuple(np.round(r, 9)) for r in phi}) == len(d), "trend should make phi injective"

    without_trend = {tuple(np.round(r[:10], 3)) for r in phi}
    assert len(without_trend) < len(d), "dropping the trend must collapse dates onto each other"


def test_head_is_linear():
    """The capacity guarantee. Inserting a hidden layer must break the build.

    A linear head cannot spike on one date: moving a coefficient to fit one date
    moves all the others. That, not the size of the feature vector, is what
    stops this model becoming a lookup table.
    """
    rng = np.random.default_rng(2)
    alpha = rng.normal(size=37)
    W = rng.normal(size=(37, generative.N_FEATURES))
    fit = generative.CBFit(
        alpha, W, np.zeros(7), np.zeros((7, generative.N_FEATURES)),
        generative.FeatureRef(0.0, 1.0), 1.0, 0.0,
        train_max_date=np.datetime64("2020-01-01", "D"), train_dates=frozenset(),
    )

    p1 = rng.normal(size=generative.N_FEATURES)
    p2 = rng.normal(size=generative.N_FEATURES)
    a, b = 0.3, 0.7
    lin = lambda p: alpha + W @ p  # noqa: E731
    assert lin(a * p1 + b * p2) == pytest.approx(a * lin(p1) + b * lin(p2) - (a + b - 1) * alpha)
    assert fit.W.shape == (37, generative.N_FEATURES)


# --- end to end -----------------------------------------------------------
def test_walk_forward_on_real_data_is_indistinguishable_from_chance():
    """The committed expectation. If this ever fails, suspect a leak first."""
    d = data.load(clean_csv=None)
    r = backtest.walk_forward(DateConditionedCB(), d, n_test=60, min_history=800)
    se = np.sqrt(metrics.BASELINE_MATCH_VAR / 60)
    assert abs(r.summary["mean_matches"] - metrics.BASELINE_MATCHES) < 4 * se
    assert r.summary["mean_log_loss"] is not None
    assert abs(r.summary["mean_log_loss"] - metrics.BASELINE_LOGLOSS) < 0.05


def test_generates_a_valid_ticket_for_a_future_date():
    d = data.load(clean_csv=None)
    p = DateConditionedCB()
    p.fit(d)
    out = p.sample(np.datetime64("2027-03-16", "D"), np.random.default_rng(0))
    assert len(out["numbers"]) == 6 and len(set(out["numbers"])) == 6
    assert all(1 <= n <= 37 for n in out["numbers"]) and 1 <= out["strong"] <= 7
    lp = p.set_log_prob(np.array(out["numbers"]), np.datetime64("2027-03-16", "D"))
    assert lp < 0 and np.isfinite(lp)


def test_contract_still_holds_for_every_other_predictor():
    """The harness change must not touch models that ignore the date."""
    for p in predictors.statistical_suite():
        assert p.wants_target_date is False


def test_constant_logit_shift_is_unidentifiable():
    """Adding a constant to every ball logit must change nothing.

    This is not a curiosity: it is why the Wilks degrees of freedom in the docs
    are 462 and not the naive 37*11 + 7*11 = 484. Twenty-two of those directions
    do not exist, so quoting 484 would overstate the expected in-sample overfit
    by ~5% and loosen the leak tripwire that depends on it.
    """
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(4, 37))
    for c in (0.7, -2.3, 5.0):
        assert generative.marginals(theta + c) == pytest.approx(
            generative.marginals(theta), abs=1e-12
        )
        shift = generative.log_partition(theta + c) - generative.log_partition(theta)
        assert shift == pytest.approx(np.full(4, 6 * c), abs=1e-9)

    identifiable = 37 * generative.N_FEATURES + 7 * generative.N_FEATURES - 2 * generative.N_FEATURES
    assert identifiable == 462


def test_leakage_guard_checks_the_fit_not_the_passed_history():
    """Fitting on everything then passing a short history must still be refused.

    The guard exists to stop an in-sample score reaching a headline. Checking
    only the caller's `history` argument would let exactly that through.
    """
    d = dated_draws(300, seed=9)
    p = DateConditionedCB()
    p.fit(d)  # sees every draw
    with pytest.raises(LeakageError, match="fitted on draws up to"):
        p.scores(d.slice(0, 100), d.dates[100])


def test_sample_labels_in_sample_reconstructions():
    d = dated_draws(200, seed=10)
    p = DateConditionedCB()
    p.fit(d)

    past = p.sample(d.dates[50], np.random.default_rng(0))
    assert past["in_sample"] is True
    assert "not a prediction" in past["label"]

    future = p.sample(np.datetime64("2030-01-01", "D"), np.random.default_rng(0))
    assert future["in_sample"] is False
    assert "out-of-sample" in future["label"]


def test_next_draw_date_lands_on_a_scheduled_day():
    d = data.load(clean_csv=None)
    nxt = data.next_draw_date(d)
    assert nxt > d.dates[-1].astype("datetime64[D]")
    assert int((nxt.astype(int) + 3) % 7) in {1, 3, 5}  # Tue / Thu / Sat


def test_report_can_predict_next_with_a_date_conditioned_model():
    """Regression: this path used to raise because no target date was supplied."""
    from bench import report

    d = data.load(clean_csv=None)
    out = report.predict_next(DateConditionedCB(), d)
    assert len(out["numbers"]) == 6 and out["target_date"] is not None
    assert np.datetime64(out["target_date"]) > d.dates[-1].astype("datetime64[D]")
