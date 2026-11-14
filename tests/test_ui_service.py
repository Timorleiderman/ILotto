"""The dashboard's registry and service functions, without the web server."""

import json

import numpy as np
import pytest

from lotto.approaches import quick_suite
from lotto.core import backtest, data, metrics
from lotto.tickets.coverage import portfolio_odds
from lotto.ui import registry
from lotto.ui.jobs import Job
from lotto.ui.registry import Invalid
from lotto.ui.service import Service
from lotto.ui.storage import Storage
from tests import SNAPSHOT

QUICK = [p for p in registry.PRESETS if p.model not in ("date_cb", "gru", "transformer")]


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    return Service(Storage(tmp_path_factory.mktemp("ui")), raw_csv=SNAPSHOT, offline=True)


def job():
    return Job("test", "test", lambda j: {})


# --- registry ----------------------------------------------------------------


def test_presets_are_the_scoreboard_suite(service):
    """Same strategies, same order, same scores: the dashboard cannot drift from the scoreboard."""
    draws, _ = service.snapshot()
    suite = quick_suite()
    assert [registry.build_preset(p.id).name for p in QUICK] == [s.name for s in suite]
    for preset, reference in zip(QUICK, suite):
        ours, theirs = registry.build_preset(preset.id).scores(draws), reference.scores(draws)
        np.testing.assert_array_equal(ours[0], theirs[0], err_msg=preset.id)
    assert len(registry.PRESETS) == len(QUICK) + 3  # plus date-conditioned, GRU, Transformer


@pytest.mark.parametrize("model,params,seed,field", [
    ("hot", {"window": 5}, None, "window"),            # below the range
    ("hot", {"window": 20.5}, None, "window"),         # not a whole number
    ("hot", {"window": True}, None, "window"),         # a boolean is not a number
    ("hot", {"windw": 50}, None, "windw"),             # unknown setting
    ("ewma", {"half_life": "50"}, None, "half_life"),  # a string is not a number
    ("ensemble", {"members": []}, None, "members"),
    ("ensemble", {"members": ["hot_all", "gru"]}, None, "members"),
    ("date_cb", {"ticket": "lucky"}, None, "ticket"),
    ("random", {}, -1, "seed"),
    ("random", {}, 1.5, "seed"),
    ("gru", {}, str(2**32), "seed"),
    ("nope", {}, None, "model"),
])
def test_bad_requests_are_refused_by_field(model, params, seed, field):
    with pytest.raises(Invalid) as error:
        registry.resolve(model, params, seed)
    assert field in error.value.fields


def test_large_seeds_travel_as_strings():
    _, _, seed = registry.resolve("seed_world", {}, str(2**64 - 1))
    assert seed == 2**64 - 1
    assert registry.resolve("hot", {}, "7")[2] is None  # a model without randomness ignores it


# --- predict -----------------------------------------------------------------


@pytest.mark.parametrize("preset", QUICK, ids=lambda p: p.id)
def test_every_quick_preset_predicts(service, preset):
    result = service.predict({"model": preset.model, "params": preset.params,
                              "seed": None if preset.seed is None else str(preset.seed), "rows": 10}, job())
    ticket = result["ticket"]
    assert len(set(ticket[:6])) == 6 and ticket[:6] == sorted(ticket[:6]) and 1 <= ticket[6] <= 7
    assert result["preset"] == preset.id and result["mode"] == "forecast"
    assert portfolio_odds(result["form"])["p_any_prize"] == result["odds"]["p_any_prize"]
    json.dumps(result, allow_nan=False)  # no NaN or Infinity reaches the browser


def test_same_seed_same_ticket_and_a_new_seed_changes_it(service):
    def ticket(seed):
        return service.predict({"model": "random", "seed": seed, "rows": 1}, job())["ticket"]

    assert ticket("5") == ticket("5")
    assert ticket("5") != ticket("6")


def test_changed_settings_lose_the_published_record(service):
    result = service.predict({"model": "hot", "params": {"window": 37}}, job())
    assert result["preset"] is None and result["reference"] is None


def test_a_single_row_is_the_ticket(service):
    result = service.predict({"model": "hot", "rows": 1}, job())
    assert result["form"] == [result["ticket"]] and result["odds"]["cost"] == 3


def test_looking_back_only_uses_earlier_draws(service):
    draws, _ = service.snapshot()
    target = str(draws.dates[-10].astype("datetime64[D]"))
    result = service.predict({"model": "date_cb", "date": target, "rows": 2}, job())
    assert result["mode"] == "retrospective" and result["trained_through"] < target
    assert result["actual"] == [*draws.balls[-10].tolist(), int(draws.strong[-10])]
    assert result["preset"] is None  # a look back never borrows the published record
    with pytest.raises(Invalid):
        service.predict({"model": "hot", "date": target}, job())  # only some models take a date


def test_sampled_ticket_follows_the_seed(service):
    def run(seed):
        return service.predict({"model": "date_cb", "params": {"ticket": "sampled"}, "seed": seed, "rows": 1}, job())

    assert run("1")["ticket_kind"] == "sampled" and run("1")["ticket"] == run("1")["ticket"]
    assert {tuple(run(str(s))["ticket"]) for s in range(6)} != {tuple(run("1")["ticket"])}


def test_seed_world_is_a_toy_outside_the_benchmark(service):
    result = service.predict({"model": "seed_world", "seed": "45123", "rows": 1}, job())
    assert result["benchmark"] is False and result["reference"] is None


# --- Lab ---------------------------------------------------------------------


def test_backtest_matches_the_core_harness_and_corrects_across_attempts(service):
    draws, _ = service.snapshot()
    for window in (100, 50):
        service.backtest({"model": "hot", "params": {"window": window}, "n_test": 300}, job())
    notebook = service.notebook()
    done = [a for a in notebook["attempts"] if a["state"] == "complete"]
    assert len(done) == 2 and all("<svg" in a["chart"] for a in done)
    direct = backtest.walk_forward(registry.build_preset("hot_100"), draws, n_test=300).summary
    assert done[0]["mean_matches"] == direct["mean_matches"] and done[0]["p_value"] == direct["p_value"]
    verdicts = metrics.holm_bonferroni({a["id"]: a["p_value"] for a in done})
    assert {a["id"]: a["significant"] for a in done} == verdicts
    service.backtest({"model": "hot", "params": {"window": 100}, "n_test": 300}, job())  # the same question again
    assert len(service.notebook()["attempts"]) == 2


# --- Log ---------------------------------------------------------------------


def test_only_forecasts_for_future_draws_can_be_saved(service):
    forecast = service.predict({"model": "hot", "rows": 2}, job())
    first = service.save_to_log(forecast["id"])
    assert service.save_to_log(forecast["id"]) == first  # saving twice keeps the original
    draws, _ = service.snapshot()
    past = service.predict({"model": "date_cb", "date": str(draws.dates[-3].astype("datetime64[D]"))}, job())
    with pytest.raises(Invalid):
        service.save_to_log(past["id"])
    with pytest.raises(Invalid):
        service.save_to_log("../../etc/passwd")


def test_log_scores_each_row_against_the_published_draw(tmp_path):
    service = Service(Storage(tmp_path), raw_csv=SNAPSHOT, offline=True)
    draws, _ = service.snapshot()
    actual = draws.balls[-1].tolist()
    winner = [*actual, int(draws.strong[-1])]
    near = [*actual[:3], *[n for n in range(1, 38) if n not in actual][:3], int(draws.strong[-1]) % 7 + 1]
    result = {"id": "0123456789abcdef", "kind": "prediction", "mode": "forecast", "model_label": "Test",
              "target_date": str(draws.dates[-1].astype("datetime64[D]")), "params": {}, "seed": None, "rows": 2,
              "ticket": winner, "form": [winner, sorted(near[:6]) + near[6:]], "odds": {"p_any_prize": 0.0833}}
    service.storage.add_log_entry({"id": result["id"], "saved_utc": "2026-11-01T00:00:00+00:00",
                                   "saved_local_date": "2026-11-01", "timing": "before_draw_day", "result": result})
    entry = service.log()["entries"][0]
    assert entry["state"] == "scored" and entry["prize"] and entry["best_matches"] == 6
    assert [s["tier"] for s in entry["row_scores"]] == ["1st (6+strong)", "8th (3)"]
    summary = service.log()["summary"]
    assert summary["scored"] == 1 and summary["with_prize"] == 1 and summary["expected_with_prize"] == 0.0833


def test_reading_never_writes_the_archive_folder(tmp_path):
    before = sorted(p.name for p in data.paths.REPO_ROOT.joinpath("tests/golden").iterdir())
    Service(Storage(tmp_path), raw_csv=SNAPSHOT, offline=True).state()
    assert sorted(p.name for p in data.paths.REPO_ROOT.joinpath("tests/golden").iterdir()) == before
