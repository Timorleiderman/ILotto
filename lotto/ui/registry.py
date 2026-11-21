"""Every model the dashboard offers, its settings, and how to build it.

This is the single list the page's controls are generated from and every
request is checked against. Two ideas are kept apart:

* a **model** is a method with editable settings (``hot`` has a window);
* a **preset** is one exact configuration that the published scoreboard
  measured (``hot_100`` is ``hot`` with window 100).

The presets must equal ``lotto.approaches.full_suite()``; a test compares them,
so the scoreboard and the dashboard cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from dataclasses import dataclass

from lotto.approaches import chaos, date_conditioned, golden_ratio, perfect_fit, statistical
from lotto.approaches.seed_world import SeedWorld
from lotto.approaches.statistical import Predictor

MAX_SEED_32 = 2**32 - 1
MAX_SEED_64 = 2**64 - 1


class Invalid(ValueError):
    """A request did not pass validation. ``fields`` maps a field name to what is wrong."""

    def __init__(self, fields: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in fields.items()))
        self.fields = fields


@dataclass(frozen=True)
class Param:
    """One editable setting of a model."""

    name: str
    label: str
    kind: str  #: "int", "float", "int_or_all", "choice" or "subset"
    default: object
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""

    def parse(self, value):
        """The validated value, or raise ValueError saying what is expected."""
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(f"choose one of: {', '.join(self.choices)}")
            return value
        if self.kind == "subset":
            if (not isinstance(value, list) or not value
                    or any(not isinstance(v, str) or v not in self.choices for v in value)
                    or len(set(value)) != len(value)):
                raise ValueError("choose at least one of the listed models, each once")
            return [v for v in self.choices if v in value]  # canonical order
        if self.kind == "int_or_all" and value in (None, "all"):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("a number is required")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("a finite number is required")
        if self.kind in ("int", "int_or_all"):
            if value != int(value):
                raise ValueError("a whole number is required")
            value = int(value)
        else:
            value = float(value)
        if not self.low <= value <= self.high:
            raise ValueError(f"must be between {self.low:g} and {self.high:g}")
        return value


@dataclass(frozen=True)
class Model:
    """One method the dashboard can run."""

    id: str
    label: str
    family: str
    description: str
    build: Callable[[dict, int | None], Predictor]
    params: tuple[Param, ...] = ()
    seed_default: int | None = None  #: None means the model uses no randomness
    seed_max: int = MAX_SEED_32
    any_date: bool = False  #: accepts a draw date other than the next one
    needs_tensorflow: bool = False
    benchmark: bool = True  #: False for toys that are never compared with the suites

    def seeded(self, params: dict) -> bool:
        """Whether these settings use the seed."""
        if self.id == "date_cb":
            return params.get("ticket") == "sampled"
        return self.seed_default is not None


@dataclass(frozen=True)
class Preset:
    """One exact configuration measured on the published scoreboard."""

    id: str
    model: str
    params: dict
    seed: int | None = None


#: The seven members of the scoreboard's rank ensemble, as preset ids.
ENSEMBLE_MEMBERS = ("hot_all", "hot_100", "cold_all", "ewma", "overdue", "pairwise", "lag1")

_WINDOW = dict(kind="int_or_all", low=10, high=1500, help="Number of recent draws to count; blank for all of them.")


def _neural(arch: str):
    def build(p: dict, seed: int | None) -> Predictor:
        from lotto.approaches.neural import NeuralPredictor

        return NeuralPredictor(arch=arch, window=p["window"], epochs=p["epochs"], refit_every=50, seed=seed)

    return build


def _ensemble(p: dict, seed: int | None) -> Predictor:
    return statistical.Ensemble([build_preset(member) for member in p["members"]])


_NEURAL_PARAMS = (
    Param("window", "History window (draws)", "int", 20, 5, 60, help="How many past draws the network reads."),
    Param("epochs", "Training epochs", "int", 60, 1, 200, help="Upper limit; training stops early when it stops improving."),
)

MODELS: tuple[Model, ...] = (
    Model("hot", "Hot numbers", "Counting", "The numbers drawn most often.",
          lambda p, s: statistical.Frequency(window=p["window"]),
          (Param("window", "Window (draws)", default=None, **_WINDOW),)),
    Model("cold", "Cold numbers", "Counting", "The numbers drawn least often.",
          lambda p, s: statistical.Frequency(window=p["window"], cold=True),
          (Param("window", "Window (draws)", default=None, **_WINDOW),)),
    Model("ewma", "EWMA frequency", "Counting", "Frequency where recent draws count for more.",
          lambda p, s: statistical.ExponentialRecency(half_life=p["half_life"]),
          (Param("half_life", "Half-life (draws)", "int", 50, 5, 500,
                 help="A draw this many draws ago counts half as much as the latest."),)),
    Model("overdue", "Overdue", "Counting", "The numbers absent for longest.",
          lambda p, s: statistical.Overdue()),
    Model("pairwise", "Pairwise co-occurrence", "Counting",
          "Numbers that have appeared alongside the last draw's numbers.",
          lambda p, s: statistical.PairwiseMarkov(alpha=p["alpha"]),
          (Param("alpha", "Smoothing", "float", 1.0, 0, 10, help="Added to every pair count."),)),
    Model("lag1", "Lag-1 persistence", "Counting",
          "Each number's chance given whether it was drawn last time.",
          lambda p, s: statistical.SequentialMarkov(alpha=p["alpha"]),
          (Param("alpha", "Smoothing", "float", 2.0, 0.1, 10, help="Pulls each estimate toward one half."),)),
    Model("sum", "Sum-targeted", "Counting", "Six numbers whose total lands on a target.",
          lambda p, s: statistical.SumConstrained(target=p["target"], seed=s),
          (Param("target", "Target sum", "int", 114, 21, 207, help="114 is the average total of six numbers."),),
          seed_default=0),
    Model("ensemble", "Rank ensemble", "Counting", "Average rank across the counting models.",
          _ensemble,
          (Param("members", "Models to average", "subset", list(ENSEMBLE_MEMBERS), choices=ENSEMBLE_MEMBERS),)),
    Model("random", "Random ticket", "Counting", "Six numbers at random: the baseline.",
          lambda p, s: statistical.UniformRandom(seed=s), seed_default=12345),
    Model("dft", "Perfect fit: DFT", "Perfect fit", "The smoothest wave through every past draw, continued one step.",
          lambda p, s: perfect_fit.PerfectFitDFT()),
    Model("spline", "Perfect fit: spline", "Perfect fit", "A cubic spline through every past draw, continued one step.",
          lambda p, s: perfect_fit.PerfectFitSpline()),
    Model("poly", "Perfect fit: polynomial", "Perfect fit", "A polynomial through the last six draws, continued one step.",
          lambda p, s: perfect_fit.PerfectFitPolynomial()),
    Model("analogues", "Method of analogues", "Chaos",
          "Find past stretches like the latest one and average what followed them.",
          lambda p, s: chaos.MethodOfAnalogues(m=p["m"], k=p["k"], prior_strength=p["prior_strength"]),
          (Param("m", "Window length (draws)", "int", 3, 1, 6, help="How many consecutive draws make one stretch."),
           Param("k", "Neighbours", "int", 20, 5, 100, help="How many similar stretches to average."),
           Param("prior_strength", "Prior strength", "float", 5.0, 0, 50, help="Pull toward equal chances."))),
    Model("golden", "Golden ratio schedule", "Golden ratio", "Six numbers placed by the golden angle.",
          lambda p, s: golden_ratio.GoldenRatio()),
    Model("fibonacci", "Fibonacci ticket", "Golden ratio", "The fixed ticket 2-3-5-8-13-21, a control.",
          lambda p, s: golden_ratio.FibonacciTicket()),
    Model("date_cb", "Date-conditioned generative", "Generative",
          "A probability model of the draw given its calendar date. Takes any date.",
          lambda p, s: date_conditioned.DateConditionedCB(tau=p["tau"]),
          (Param("tau", "Regularisation", "float", 10.0, 0.1, 1000, help="Higher keeps the model closer to equal chances."),
           Param("ticket", "Ticket", "choice", "top six", choices=("top six", "sampled"),
                 help="The six likeliest numbers, or one ticket drawn from the model using the seed.")),
          seed_default=0, any_date=True),
    Model("gru", "Neural: GRU", "Neural", "A recurrent network over recent draws, predicting the set of numbers.",
          _neural("gru"), _NEURAL_PARAMS, seed_default=42, needs_tensorflow=True),
    Model("transformer", "Neural: Transformer", "Neural", "Self-attention over recent draws, predicting the set of numbers.",
          _neural("transformer"), _NEURAL_PARAMS, seed_default=42, needs_tensorflow=True),
    Model("seed_world", "Seed world (toy)", "Side quest",
          "An artificial generator: a seed and a date give one ticket. Not a model of the real lottery.",
          lambda p, s: SeedWorld(seed=s), seed_default=45123, seed_max=MAX_SEED_64, any_date=True, benchmark=False),
)

#: In scoreboard order: must match `lotto.approaches.full_suite()`.
PRESETS: tuple[Preset, ...] = (
    Preset("random", "random", {}, 12345),
    Preset("hot_all", "hot", {"window": None}),
    Preset("hot_100", "hot", {"window": 100}),
    Preset("cold_all", "cold", {"window": None}),
    Preset("ewma", "ewma", {"half_life": 50}),
    Preset("overdue", "overdue", {}),
    Preset("pairwise", "pairwise", {"alpha": 1.0}),
    Preset("lag1", "lag1", {"alpha": 2.0}),
    Preset("sum", "sum", {"target": 114}, 0),
    Preset("ensemble", "ensemble", {"members": list(ENSEMBLE_MEMBERS)}),
    Preset("dft", "dft", {}),
    Preset("spline", "spline", {}),
    Preset("poly", "poly", {}),
    Preset("analogues", "analogues", {"m": 3, "k": 20, "prior_strength": 5.0}),
    Preset("golden", "golden", {}),
    Preset("fibonacci", "fibonacci", {}),
    Preset("date_cb", "date_cb", {"tau": 10.0, "ticket": "top six"}),
    Preset("gru", "gru", {"window": 20, "epochs": 60}, 42),
    Preset("transformer", "transformer", {"window": 20, "epochs": 60}, 42),
)

_MODELS = {m.id: m for m in MODELS}
_PRESETS = {p.id: p for p in PRESETS}


def model(model_id: str) -> Model:
    """The model with this id; `Invalid` if there is none."""
    if not isinstance(model_id, str) or model_id not in _MODELS:
        raise Invalid({"model": "unknown model"})
    return _MODELS[model_id]


def preset(preset_id: str) -> Preset:
    """The preset with this id; `Invalid` if there is none."""
    if not isinstance(preset_id, str) or preset_id not in _PRESETS:
        raise Invalid({"preset": "unknown preset"})
    return _PRESETS[preset_id]


def tensorflow_available() -> bool:
    """Whether TensorFlow is installed, without importing it (the import takes seconds)."""
    return importlib.util.find_spec("tensorflow") is not None


def parse_seed(value, maximum: int) -> int:
    """A seed sent as a decimal string or an integer.

    Strings are accepted because JavaScript numbers cannot hold every 64-bit
    integer exactly.
    """
    if isinstance(value, str) and value.isascii() and value.isdigit() and len(value) <= 20:
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"a whole number from 0 to {maximum}")
    return value


def resolve(model_id: str, raw_params, raw_seed) -> tuple[Model, dict, int | None]:
    """Validate a request's model, settings and seed. Missing settings take their defaults.

    Unknown settings are refused, so a typo cannot silently run the default.
    """
    spec = model(model_id)
    if not isinstance(raw_params, dict):
        raise Invalid({"params": "an object of settings is required"})
    errors: dict[str, str] = {}
    known = {p.name for p in spec.params}
    for name in raw_params:
        if name not in known:
            errors[name] = "this model has no such setting"
    params = {}
    for p in spec.params:
        try:
            params[p.name] = p.parse(raw_params.get(p.name, p.default))
        except ValueError as exc:
            errors[p.name] = str(exc)
    seed = None
    if spec.seeded(params):
        try:
            seed = parse_seed(spec.seed_default if raw_seed is None else raw_seed, spec.seed_max)
        except ValueError as exc:
            errors["seed"] = str(exc)
    if errors:
        raise Invalid(errors)
    return spec, params, seed


def build(spec: Model, params: dict, seed: int | None) -> Predictor:
    """A fresh predictor for validated settings. Never reuse one between jobs."""
    return spec.build(params, seed)


def build_preset(preset_id: str) -> Predictor:
    """A fresh predictor for a scoreboard preset."""
    p = preset(preset_id)
    spec, params, seed = resolve(p.model, p.params, p.seed)
    return build(spec, params, seed)


def matching_preset(model_id: str, params: dict, seed: int | None) -> Preset | None:
    """The scoreboard preset with exactly these settings, if there is one."""
    for p in PRESETS:
        if p.model == model_id and p.params == params and p.seed == seed:
            return p
    return None


def describe() -> dict:
    """The registry as plain data for the page."""
    tf = tensorflow_available()
    return {
        "models": [{
            "id": m.id, "label": m.label, "family": m.family, "description": m.description,
            "params": [{"name": p.name, "label": p.label, "kind": p.kind, "default": p.default, "low": p.low,
                        "high": p.high, "choices": list(p.choices), "help": p.help} for p in m.params],
            "seed_default": None if m.seed_default is None else str(m.seed_default),
            "seed_max": str(m.seed_max), "seed_only_when": {"ticket": "sampled"} if m.id == "date_cb" else None,
            "any_date": m.any_date, "benchmark": m.benchmark,
            "available": tf or not m.needs_tensorflow,
            "unavailable_reason": "" if tf or not m.needs_tensorflow else "Needs TensorFlow: uv sync --extra neural",
        } for m in MODELS],
        "presets": [{"id": p.id, "model": p.model, "params": p.params,
                     "seed": None if p.seed is None else str(p.seed)} for p in PRESETS],
    }
