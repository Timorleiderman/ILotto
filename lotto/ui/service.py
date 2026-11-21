"""What the dashboard can do, as plain functions over the existing computation.

Nothing statistical is implemented here. Each function validates a request,
calls code in `lotto.core`, `lotto.approaches` and `lotto.tickets`, and packs
the answer into a plain dictionary the page, the reports and the log all share.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy import stats

from lotto.core import backtest, data, metrics, paths
from lotto.core.files import sha256_file
from lotto.economics import archive as prize_archive
from lotto.economics import value as ticket_value
from lotto.report import charts
from lotto.tickets.coverage import portfolio_odds
from lotto.tickets.draw_plan import CAP, FORMS, ROW_PRICE
from lotto.tickets.model_forms import best_ticket, model_form
from lotto.tickets.strong_assignment import strong_prize_odds
from lotto.ui import registry
from lotto.ui.jobs import Cancelled, Job
from lotto.ui.registry import Invalid
from lotto.ui.storage import Storage, valid_id

SCHEMA = 2
ISRAEL = ZoneInfo("Asia/Jerusalem")
ROW_CHOICES = (1, *sorted(FORMS))
MIN_HISTORY = 200
logger = logging.getLogger(__name__)


def algorithm_fingerprint() -> str:
    """Invalidate saved computations when their implementation or runtime changes."""
    from importlib.metadata import PackageNotFoundError, version

    package = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    for name in ("numpy", "scipy", "numba", "tensorflow"):
        try:
            digest.update(f"{name}={version(name)}".encode())
        except PackageNotFoundError:
            digest.update(f"{name}=absent".encode())
    return digest.hexdigest()


def _day(value) -> str:
    return str(np.datetime64(value, "D"))


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today_in_israel() -> np.datetime64:
    """Today's date where the draws happen."""
    return np.datetime64(datetime.now(ISRAEL).date().isoformat(), "D")


def _finite(values: np.ndarray) -> list[float | None]:
    """Scores as JSON numbers; a number a model excluded (-inf) becomes null."""
    return [float(v) if np.isfinite(v) else None for v in values]


def _ticket_row(numbers, strong) -> list[int]:
    return [*sorted(int(n) for n in numbers), int(strong)]


class Service:
    """The dashboard's state (the loaded archive, saved results) and its actions."""

    def __init__(self, storage: Storage, raw_csv: str = data.RAW_CSV, offline: bool = False,
                 prizes_csv: str = prize_archive.PRIZES_CSV):
        self.storage, self.raw_csv, self.offline = storage, raw_csv, offline
        self.prizes_csv = prizes_csv
        self.announcement: prize_archive.Announcement | None = None
        self._value_cache: dict = {}
        self.refresh_status = {"state": "not_checked", "at": None}
        self._lock = threading.Lock()
        self._odds: dict[int, dict] = {}
        self._reference_cache: tuple[int, dict] | None = None
        self.algorithm = algorithm_fingerprint()
        self._load()
        attempts = self.storage.notebook()
        changed = False
        for attempt in attempts:
            if attempt["state"] in ("queued", "running"):
                attempt.update(state="interrupted", finished_utc=_now_utc())
                changed = True
            elif (attempt["state"] == "complete" and attempt.get("model") == "date_cb"
                  and attempt.get("params", {}).get("ticket") == "sampled"):
                # Older dashboards labelled top-six evaluations as sampled tickets.
                # Retain that history, but never count it as evidence for the selected rule.
                attempt.update(state="invalid", invalidated_utc=_now_utc(),
                               message="An earlier dashboard tested top-six tickets for these sampled settings. "
                                       "Excluded from the summary; switch to top six and run a new backtest.")
                changed = True
        if changed:
            self.storage.save_notebook(attempts)

    # ------------------------------------------------------------------ archive
    def _load(self) -> None:
        draws = data.load(raw_csv=self.raw_csv, clean_csv=None)  # reading must not write input/
        with self._lock:
            self.draws = draws
            self.archive = {"draws": len(draws), "last_date": _day(draws.dates[-1]),
                            "sha256": sha256_file(self.raw_csv),
                            "last_draw": _ticket_row(draws.balls[-1], draws.strong[-1])}

    def snapshot(self) -> tuple[data.Draws, dict]:
        """The archive a job should use from start to finish."""
        with self._lock:
            return self.draws, self.archive

    def next_draw(self, draws: data.Draws) -> str:
        """The estimated date of the next draw: inferred from the recent rhythm, not an official schedule."""
        return _day(data.next_draw_date(draws, today=today_in_israel()))

    def refresh(self, job: Job) -> dict:
        """Check pais.co.il for new draws; a failure keeps the archive that is loaded."""
        if self.offline:
            raise Invalid({"refresh": "The dashboard was started with --offline."})
        job.step("Contacting pais.co.il")
        _, before_archive = self.snapshot()
        before = before_archive["draws"]
        state = data.check_for_new_draws(self.raw_csv)
        # The download has already replaced the file on disk, so the loaded
        # archive must follow it even if this job is cancelled before publishing:
        # a retry would report "unchanged" and never reload.
        if state == "updated":
            self._load()
        with self._lock:
            self.refresh_status = {"state": state, "at": _now_utc()}
        # Prize tables and the advertised jackpot feed the Value screen. Best effort:
        # a failure here keeps the committed tables and never fails the draw check.
        job.step("Reading prize tables")
        prize_archive.refresh(self.prizes_csv)
        announcement = prize_archive.fetch_announcement()
        if announcement is not None:
            with self._lock:
                self.announcement = announcement

        def publish() -> dict:
            _, archive = self.snapshot()
            return {"state": state, "added": archive["draws"] - before, "archive": archive}

        return job.finish(publish)

    def state(self) -> dict:
        """Everything the page needs to draw itself."""
        with self._lock:
            draws, archive, refresh_status = self.draws, self.archive, dict(self.refresh_status)
        return {"schema": SCHEMA, "archive": archive, "next_draw": self.next_draw(draws),
                "today": _day(today_in_israel()), "refresh": {**refresh_status, "offline": self.offline},
                "rows": list(ROW_CHOICES), "row_price": ROW_PRICE, "baseline_matches": metrics.BASELINE_MATCHES,
                **registry.describe(), "preset_names": self.preset_names()}

    # --------------------------------------------------------------------- odds
    def form_odds(self, rows: int) -> dict:
        """Exact fair-draw odds of a form of this size.

        They depend only on how the rows overlap, so every model's relabelled
        form shares them, and a single row has the same odds whatever its numbers.
        """
        with self._lock:
            cached = self._odds.get(rows)
        if cached is not None:
            return cached
        design = [list(r) for r in FORMS[rows]] if rows in FORMS else [[1, 2, 3, 4, 5, 6, 1]]
        odds, strong = portfolio_odds(design), strong_prize_odds(design)
        computed = {
            "rows": rows, "p_any_prize": odds["p_any_prize"],
            "p_at_least": {k: odds["p_at_least_matches"][k] for k in ("3", "4", "5", "6")},
            "p_three_plus_and_strong": strong["p_at_least_matches_and_strong"],
            "first_prize_ways": odds["jackpot_covered_outcomes"], "total_outcomes": odds["total_full_outcomes"],
            "p_first_prize": odds["p_jackpot"], "cost": ROW_PRICE * rows,
        }
        with self._lock:
            self._odds[rows] = computed
        return computed

    # ---------------------------------------------------------------- reference
    def preset_names(self) -> dict[str, str]:
        """Scoreboard name of every preset, e.g. ``hot_100`` → "Hot numbers (last 100 draws)"."""
        with self._lock:
            if not hasattr(self, "_preset_names"):
                self._preset_names = {p.id: registry.build_preset(p.id).name for p in registry.PRESETS}
            return self._preset_names

    def reference(self, preset_id: str | None) -> dict | None:
        """The published scoreboard record of a preset: a dated reference, not a live result."""
        path = paths.REPO_ROOT / "docs/benchmark/results.json"
        if preset_id is None or not path.is_file():
            return None
        mtime = path.stat().st_mtime_ns
        with self._lock:
            if self._reference_cache is not None and self._reference_cache[0] == mtime:
                report = self._reference_cache[1]
            else:
                report = json.loads(path.read_text(encoding="utf-8"))
                self._reference_cache = (mtime, report)
        name = self.preset_names()[preset_id]
        row = next((r for r in report["strategies"] if r["name"] == name), None)
        if row is None:
            return None
        return {"preset": preset_id, "name": name, "mean_matches": row["mean_matches"], "n_draws": row["n_draws"],
                "p_value": row["p_value"], "significant": bool(row.get("significant_after_correction")),
                "strategies_compared": len(report["strategies"]),
                "archive_last_date": report["last_draw_date"], "generated_utc": report["generated_utc"],
                "configuration_verified": False}

    def validate_request(self, kind: str, request: dict) -> None:
        """Reject malformed requests before queueing expensive work; services also validate directly."""
        allowed = {
            "predict": {"model", "params", "seed", "rows", "date"},
            "backtest": {"model", "params", "seed", "n_test"},
            "compare": set(),
        }[kind]
        if not isinstance(request, dict):
            raise Invalid({"request": "an object is required"})
        unknown = set(request) - allowed
        if unknown:
            raise Invalid({key: "unknown request field" for key in unknown})
        if kind == "compare":
            return
        spec, params, _ = registry.resolve(request.get("model"), request.get("params", {}), request.get("seed"))
        if spec.needs_tensorflow and not registry.tensorflow_available():
            raise Invalid({"model": "Needs TensorFlow: uv sync --extra neural"})
        if kind == "predict":
            rows = request.get("rows", 10)
            if isinstance(rows, bool) or not isinstance(rows, int) or rows not in ROW_CHOICES:
                raise Invalid({"rows": f"choose one of {', '.join(map(str, ROW_CHOICES))}"})
            self._target(spec, request.get("date"), self.snapshot()[0])
        else:
            if not spec.benchmark:
                raise Invalid({"model": "Toy generators outside the benchmark cannot be backtested in the Lab."})
            if spec.id == "date_cb" and params["ticket"] == "sampled":
                raise Invalid({"ticket": "The Lab evaluates top-six tickets. Choose 'top six' before backtesting."})
            n_test = request.get("n_test", 300)
            if isinstance(n_test, bool) or not isinstance(n_test, int) or not 100 <= n_test <= 1000:
                raise Invalid({"n_test": "a whole number of draws from 100 to 1,000"})

    # ------------------------------------------------------------------ predict
    def _target(self, spec: registry.Model, raw_date, draws: data.Draws) -> tuple[str, str]:
        """The draw date a request is about, and whether it is a forecast or a look back."""
        nxt = self.next_draw(draws)
        if raw_date in (None, "", nxt):
            return nxt, "forecast"
        try:
            if not isinstance(raw_date, str) or len(raw_date) != 10:
                raise ValueError
            target = date.fromisoformat(raw_date).isoformat()
            if target != raw_date:
                raise ValueError
        except (ValueError, TypeError) as exc:
            raise Invalid({"date": "a date like 2026-11-17 is required"}) from exc
        if not spec.any_date:
            raise Invalid({"date": "this model only forecasts the next draw"})
        if not "1968-01-01" <= target <= "2100-12-31":
            raise Invalid({"date": "choose a date between 1968 and 2100"})
        retrospective = target < _day(today_in_israel()) or target <= _day(draws.dates[-1])
        return target, "retrospective" if retrospective else "forecast"

    def _scores(self, spec, params, seed, draws, target, mode):
        """Run one model for one date. A look back only ever sees draws before that date."""
        history = draws
        if mode == "retrospective":
            history = draws.slice(0, int(np.searchsorted(draws.dates.astype("datetime64[D]"), np.datetime64(target))))
            if spec.benchmark and len(history) < MIN_HISTORY:  # the seed-world toy reads no history
                raise Invalid({"date": f"needs at least {MIN_HISTORY} earlier draws; choose a later date"})
        predictor = registry.build(spec, params, seed)
        _, balls, strong = backtest.next_draw_scores(predictor, history, target_date=np.datetime64(target))
        return predictor, history, balls, strong

    def predict(self, request: dict, job: Job) -> dict:
        """One model, one draw: its ticket, a form, exact odds and where the numbers came from."""
        self.validate_request("predict", request)
        spec, params, seed = registry.resolve(request.get("model"), request.get("params", {}), request.get("seed"))
        if spec.needs_tensorflow and not registry.tensorflow_available():
            raise Invalid({"model": "Needs TensorFlow: uv sync --extra neural"})
        rows = request.get("rows", 10)
        if isinstance(rows, bool) or not isinstance(rows, int) or rows not in ROW_CHOICES:
            raise Invalid({"rows": f"choose one of {', '.join(map(str, ROW_CHOICES))}"})
        draws, archive = self.snapshot()
        target, mode = self._target(spec, request.get("date"), draws)
        design = [list(r) for r in FORMS[rows]] if rows in FORMS else "single-ticket"
        identity = {"schema": SCHEMA, "algorithm": self.algorithm, "model": spec.id, "params": params,
                    "seed": None if seed is None else str(seed), "rows": rows, "target_date": target,
                    "mode": mode, "archive_sha256": archive["sha256"],
                    "design_sha256": hashlib.sha256(json.dumps(design).encode()).hexdigest()}
        result_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        cached = self.storage.get_result(result_id)
        if cached is not None:
            return job.finish(lambda: cached)

        job.step(f"Running {spec.label}" + (" (training)" if spec.needs_tensorflow else ""))
        predictor, history, balls, strong = self._scores(spec, params, seed, draws, target, mode)

        ticket, ticket_kind = best_ticket(balls, strong), "best"
        if spec.id == "date_cb" and params["ticket"] == "sampled":
            sample = predictor.sample(np.datetime64(target), np.random.default_rng(seed))
            ticket, ticket_kind = _ticket_row(sample["numbers"], sample["strong"]), "sampled"
        job.step("Building the form and counting its odds")
        form = [ticket] if rows == 1 else model_form(balls, strong, [list(r) for r in FORMS[rows]])

        actual = None
        if mode == "retrospective":
            hit = np.flatnonzero(draws.dates.astype("datetime64[D]") == np.datetime64(target))
            if len(hit):
                actual = _ticket_row(draws.balls[hit[0]], draws.strong[hit[0]])
        preset = registry.matching_preset(spec.id, params, seed)
        result = {
            "id": result_id,
            "kind": "prediction", **identity,
            "model_label": spec.label, "family": spec.family, "strategy_name": predictor.name,
            "mode": mode, "trained_through": _day(history.dates[-1]) if len(history) else None,
            "history_draws": len(history),
            "actual": actual, "ticket": ticket, "ticket_kind": ticket_kind, "form": form,
            "scores": _finite(balls), "strong_scores": _finite(strong),
            "score_kind": "relative_weight" if predictor.emits_probabilities else "score",
            "odds": self.form_odds(rows),
            "preset": None if preset is None or mode != "forecast" else preset.id,
            "reference": None if preset is None or mode != "forecast" else self.reference(preset.id),
            "archive": archive, "generated_utc": _now_utc(), "benchmark": spec.benchmark,
        }
        return job.finish(lambda: self.storage.put_result(result))

    # ------------------------------------------------------------------ compare
    def compare(self, request: dict, job: Job) -> dict:
        """Every scoreboard preset on the next draw, and which numbers they agree on."""
        self.validate_request("compare", request)
        draws, archive = self.snapshot()
        target = self.next_draw(draws)
        tensorflow = registry.tensorflow_available()
        rows = []
        for index, preset in enumerate(registry.PRESETS):
            spec, params, seed = registry.resolve(preset.model, preset.params, preset.seed)
            row = {"preset": preset.id, "model": spec.id, "name": self.preset_names()[preset.id], "family": spec.family}
            reference = self.reference(preset.id)
            row.update(params=params, seed=None if seed is None else str(seed), reference=reference)
            row["reference_mean"] = None if reference is None else reference["mean_matches"]
            if spec.needs_tensorflow and not tensorflow:
                rows.append({**row, "state": "unavailable", "message": "Needs TensorFlow: uv sync --extra neural"})
                continue
            job.step(f"{row['name']}", index, len(registry.PRESETS))
            try:
                _, _, balls, strong = self._scores(spec, params, seed, draws, target, "forecast")
                rows.append({**row, "state": "complete", "ticket": best_ticket(balls, strong)})
            except Cancelled:
                raise
            except Exception:  # one failing model must not lose the others
                logger.exception("Comparison failed for %s", preset.id)
                rows.append({**row, "state": "failed", "message": "Model failed; see the server terminal."})
        done = [r for r in rows if r["state"] == "complete"]
        consensus = np.zeros(data.N_NUMBERS, dtype=int)
        for r in done:
            consensus[np.array(r["ticket"][:6]) - 1] += 1
        identity = {"schema": SCHEMA, "algorithm": self.algorithm, "kind": "comparison",
                    "target_date": target, "archive_sha256": archive["sha256"],
                    "presets": [r["preset"] for r in done]}
        # Each comparison is a run: transient failures and availability can differ on a retry.
        result = {"id": uuid.uuid4().hex[:16], **identity,
                  "rows": rows, "consensus": consensus.tolist(), "counted": len(done), "archive": archive,
                  "generated_utc": _now_utc()}
        return job.finish(lambda: self.storage.put_result(result))

    # ---------------------------------------------------------------------- Lab
    def backtest(self, request: dict, job: Job) -> dict:
        """Replay one configuration over past draws, each forecast seeing only earlier draws."""
        self.validate_request("backtest", request)
        spec, params, seed = registry.resolve(request.get("model"), request.get("params", {}), request.get("seed"))
        if spec.needs_tensorflow and not registry.tensorflow_available():
            raise Invalid({"model": "Needs TensorFlow: uv sync --extra neural"})
        n_test = request.get("n_test", 300)
        if isinstance(n_test, bool) or not isinstance(n_test, int) or not 100 <= n_test <= 1000:
            raise Invalid({"n_test": "a whole number of draws from 100 to 1,000"})
        draws, archive = self.snapshot()
        window = backtest.evaluation_window(len(draws), n_test)
        config = {"model": spec.id, "params": params, "seed": None if seed is None else str(seed)}
        key = hashlib.sha256(json.dumps({**config, "n": len(window), "archive": archive["sha256"],
                                        "algorithm": self.algorithm, "schema": SCHEMA},
                                        sort_keys=True).encode()).hexdigest()[:16]
        attempts = self.storage.notebook()
        cached = next((a for a in attempts if a.get("comparison_id", a["id"]) == key
                       and a["state"] == "complete"), None)
        if cached is None:
            # Record the attempt before running it: an abandoned try is still a try.
            previous = [a for a in attempts if a.get("comparison_id", a["id"]) == key]
            attempt_id = key if not previous else uuid.uuid4().hex[:16]
            attempt = {"id": attempt_id, "comparison_id": key, "algorithm": self.algorithm,
                       "archive_sha256": archive["sha256"], **config,
                       "label": self._label(spec, params, seed), "state": "running",
                       "requested_draws": n_test, "started_utc": _now_utc(), "archive_last_date": archive["last_date"]}
            attempts.append(attempt)
            self.storage.save_notebook(attempts)
            try:
                run = backtest.walk_forward(
                    registry.build(spec, params, seed), draws, n_test=n_test,
                    on_step=lambda done, total: job.step(f"Replaying draw {done + 1} of {total}", done, total))
            except Cancelled:
                attempt.update(state="cancelled", finished_utc=_now_utc())
                self.storage.save_notebook(attempts)
                raise
            except Exception:
                attempt.update(state="failed", message="Run failed; see the server terminal.",
                               finished_utc=_now_utc())
                self.storage.save_notebook(attempts)
                raise
            summary = run.summary
            hist = np.bincount(run.match_counts, minlength=data.N_DRAWN + 1).tolist()
            pmf = stats.hypergeom(data.N_NUMBERS, data.N_DRAWN, data.N_DRAWN).pmf(np.arange(data.N_DRAWN + 1))
            completed_fields = {
                "state": "complete", "draws": len(window), "first_date": _day(draws.dates[window.start]),
                "last_date": _day(draws.dates[window.stop - 1]), "mean_matches": summary["mean_matches"],
                "p_value": summary["p_value"], "histogram": hist,
                "chance": [float(v) for v in pmf * len(window)],
                "seconds": round(run.seconds, 2), "finished_utc": _now_utc(),
            }

            def publish():
                attempt.update(completed_fields)
                self.storage.save_notebook(attempts)
                return {"attempt_id": attempt_id, "notebook": self.notebook()}

            try:
                return job.finish(publish)
            except Cancelled:
                attempt.update(state="cancelled", finished_utc=_now_utc())
                self.storage.save_notebook(attempts)
                raise
        return job.finish(lambda: {"attempt_id": cached["id"], "notebook": self.notebook()})

    @staticmethod
    def _label(spec: registry.Model, params: dict, seed: int | None) -> str:
        parts = [f"{p.label.lower()} {'all' if params[p.name] is None else params[p.name]}"
                 for p in spec.params if p.kind != "subset"]
        if "members" in params:
            parts.append("members " + ", ".join(params["members"]))
        if seed is not None:
            parts.append(f"seed {seed}")
        return spec.label + (" · " + " · ".join(parts) if parts else "")

    def notebook(self) -> dict:
        """Every Lab attempt, with verdicts corrected across all completed attempts.

        The correction is Holm-Bonferroni, recomputed whenever the notebook
        changes. It is exploratory by construction: settings chosen after
        seeing earlier results are not independent tests.
        """
        attempts = self.storage.notebook()
        done = {a["id"]: a["p_value"] for a in attempts if a["state"] == "complete"}
        verdicts = metrics.holm_bonferroni(done) if done else {}
        view = []
        for a in attempts:
            row = {**a, "significant": verdicts.get(a["id"])}
            if a["state"] == "complete":
                labels = [str(k) for k in range(data.N_DRAWN + 1)]
                row["chart"] = charts.grouped_bars(
                    labels, [("This configuration", a["histogram"]), ("Chance", a["chance"])],
                    x_title="Numbers matched in one draw", y_title="Draws")
            view.append(row)
        return {"attempts": view, "completed": len(done), "baseline": metrics.BASELINE_MATCHES}

    # -------------------------------------------------------------------- Value
    def _prize_history(self, taxed: bool) -> tuple[list, list[dict], dict]:
        """The prize tables and their valued history, recomputed only when the file changes."""
        path = Path(self.prizes_csv)
        stamp = path.stat().st_mtime_ns if path.is_file() else None
        with self._lock:
            cached = self._value_cache.get(taxed)
        if cached is not None and cached[0] == stamp:
            return cached[1], cached[2], cached[3]
        results = prize_archive.load(self.prizes_csv)
        if len(results) < 20:
            raise Invalid({"value": "No prize tables yet. Press 'Check for new draws' while online."})
        rows = ticket_value.history(results, taxed)
        summary = ticket_value.summarise(rows)
        with self._lock:
            self._value_cache[taxed] = (stamp, results, rows, summary)
        return results, rows, summary

    def announced_prizes(self, results: list) -> dict:
        """The next draw's prizes: as advertised if read this session, else inferred from the last result."""
        last = results[-1]
        with self._lock:
            announcement = self.announcement
        if announcement is not None and announcement.draw_id > last.draw_id:
            return {"source": "announced", "draw_id": announcement.draw_id, "sales_close": announcement.sales_close,
                    "first_prize": announcement.first_prize, "second_prize": announcement.second_prize}
        first, second, bonus_likely = prize_archive.inferred_next_prizes(results)
        return {"source": "inferred", "draw_id": last.draw_id + 1, "sales_close": None,
                "first_prize": first, "second_prize": second, "after_draw": last.draw_id,
                "bonus_likely": bonus_likely}

    def value(self, query: dict) -> dict:
        """What rows in the next draw cost and return, and what a row was worth in every past draw."""
        allowed = {"first", "second", "rows", "kind", "bonus", "tax"}
        unknown = set(query) - allowed
        if unknown:
            raise Invalid({key: "unknown request field" for key in unknown})

        def whole(name, low, high, default):
            raw = query.get(name)
            if raw in (None, ""):
                return default
            try:
                number = int(raw)
            except (TypeError, ValueError):
                number = None
            if number is None or str(number) != str(raw).strip() or not low <= number <= high:
                raise Invalid({name: f"a whole number from {low:,} to {high:,}"})
            return number

        def flag(name, default):
            raw = query.get(name)
            if raw in (None, ""):
                return default
            if raw not in ("0", "1", 0, 1, True, False):
                raise Invalid({name: "0 or 1"})
            return raw in ("1", 1, True)

        taxed = flag("tax", True)
        results, rows, summary = self._prize_history(taxed)
        announced = self.announced_prizes(results)
        first = whole("first", 100_000, 1_000_000_000, announced["first_prize"])
        second = whole("second", 10_000, 100_000_000, announced["second_prize"])
        n_rows = whole("rows", 1, ticket_value.FULL_COVERAGE_ROWS, 10)
        kind = query.get("kind") or "regular"
        if kind not in ticket_value.KINDS:
            raise Invalid({"kind": "regular or double"})
        bonus = flag("bonus", bool(announced.get("bonus_likely")))
        plan = ticket_value.next_draw(results, first, second, bonus=bonus, rows=n_rows, kind=kind, taxed=taxed)
        dates = [r["date"] for r in rows]
        chart = charts.timeline(
            # A Double row's return is within a cent of a regular row's, so one line carries both.
            dates, [("Return per NIS, regular row", [r["regular"]["return_per_shekel"] for r in rows])],
            reference=1.0, reference_label="break-even: 1 NIS back per NIS spent",
            y_title="Expected return per NIS", title="Expected return per shekel, every draw since 2018")
        best = sorted(rows, key=lambda r: -r["regular"]["return_per_shekel"])[:10]
        untaxed_rows = self._prize_history(False)[1] if taxed else rows
        before, after = (ticket_value.summarise(untaxed_rows), summary) if taxed else (summary,
                                                                                       self._prize_history(True)[2])
        at_cap = sum(r["positive"] and r["first_prize"] >= CAP for r in untaxed_rows)
        return {"announced": announced, "plan": plan, "summary": summary, "chart": chart,
                "positive_before_tax": before["positive_draws"], "positive_after_tax": after["positive_draws"],
                "positive_before_tax_at_cap": at_cap,
                "best": [{k: r[k] for k in ("draw_id", "date", "first_prize", "second_prize", "bonus", "first_won",
                                            "rows_sold", "positive")}
                         | {"regular": r["regular"]["return_per_shekel"], "double": r["double"]["return_per_shekel"]}
                         for r in best],
                "latest": [{"draw_id": r["draw_id"], "date": r["date"], "first_prize": r["first_prize"],
                            "bonus": r["bonus"], "first_won": r["first_won"], "rows_sold": r["rows_sold"],
                            "regular": r["regular"]["return_per_shekel"]} for r in rows[-12:][::-1]],
                "prize_tables": {"draws": len(results), "first_draw": results[0].draw_id,
                                 "last_draw": results[-1].draw_id, "last_date": results[-1].date},
                "tax_year": ticket_value.TAX_TABLE_YEAR, "prices": ticket_value.PRICES, "offline": self.offline}

    # ---------------------------------------------------------------------- Log
    def save_to_log(self, result_id) -> dict:
        """Save a completed forecast for a draw that has not happened yet."""
        result = self.storage.get_result(result_id) if valid_id(result_id) else None
        if result is None or result.get("kind") != "prediction":
            raise Invalid({"result": "that prediction is no longer available; generate it again"})
        if result["mode"] != "forecast":
            raise Invalid({"result": "a look back at a past date cannot be saved as a prediction"})
        draws, _ = self.snapshot()
        if result["target_date"] <= _day(draws.dates[-1]):
            raise Invalid({"result": "the result for that date is already in the archive"})
        today = _day(today_in_israel())
        if result["target_date"] < today:
            raise Invalid({"result": "that draw date has passed"})
        entry = {"id": result["id"], "saved_utc": _now_utc(), "saved_local_date": today,
                 # Saved on the draw day itself, we cannot tell whether it was before the draw.
                 "timing": "before_draw_day" if today < result["target_date"] else "unverified",
                 "result": result}
        return self.storage.add_log_entry(entry)

    def log(self) -> dict:
        """Saved predictions, each scored against the archive if its draw is in it."""
        draws, _ = self.snapshot()
        day_index = {str(d): i for i, d in enumerate(draws.dates.astype("datetime64[D]"))}
        entries = []
        for entry in reversed(self.storage.log_entries()):
            result = entry["result"]
            view = {"id": entry["id"], "saved_utc": entry["saved_utc"], "timing": entry["timing"],
                    "target_date": result["target_date"], "model_label": result["model_label"],
                    "params": result["params"], "seed": result["seed"], "rows": result["rows"],
                    "ticket": result["ticket"], "form": result["form"], "p_any_prize": result["odds"]["p_any_prize"],
                    "state": "waiting"}
            idx = day_index.get(result["target_date"])
            if idx is not None:
                balls, strong = set(draws.balls[idx].tolist()), int(draws.strong[idx])
                scored = [{"matches": len(balls & set(row[:6])), "strong": row[6] == strong} for row in result["form"]]
                for s in scored:
                    s["tier"] = metrics.prize_tier(s["matches"], s["strong"])
                view.update(state="scored", actual=_ticket_row(balls, strong), row_scores=scored,
                            best_matches=max(s["matches"] for s in scored),
                            prize=any(s["tier"] != "no prize" for s in scored),
                            ticket_matches=len(balls & set(result["ticket"][:6])))
            entries.append(view)
        counted = [e for e in entries if e["state"] == "scored" and e["timing"] == "before_draw_day"]
        return {"entries": entries,
                "summary": {"saved": len(entries), "waiting": sum(e["state"] == "waiting" for e in entries),
                            "scored": len(counted), "with_prize": sum(e["prize"] for e in counted),
                            "expected_with_prize": sum(e["p_any_prize"] for e in counted),
                            "unverified": sum(e["timing"] == "unverified" for e in entries)}}
