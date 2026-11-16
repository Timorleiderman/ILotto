"""Every idea that tries to predict the draw, from folklore to neural networks.

Each module is one family of strategies behind the same contract,
``scores(history) -> (ball_scores[37], strong_scores[7])``. Read
``statistical.py`` first; ``neural.py`` and ``date_conditioned.py`` build on it.

``quick_suite()`` and ``full_suite()`` list every strategy, in scoreboard order;
the benchmark report and ``lotto predict`` both use them.
"""

from __future__ import annotations


def quick_suite() -> list:
    """Every strategy that needs no training: seconds to run, no TensorFlow."""
    from lotto.approaches.chaos import chaos_suite
    from lotto.approaches.golden_ratio import golden_suite
    from lotto.approaches.perfect_fit import spacetime_suite
    from lotto.approaches.statistical import statistical_suite

    return statistical_suite() + spacetime_suite() + chaos_suite() + golden_suite()


def full_suite() -> list:
    """``quick_suite()`` plus the date-conditioned model and the two neural networks."""
    from lotto.approaches.neural import NeuralPredictor
    from lotto.approaches.statistical import generative_suite

    return quick_suite() + generative_suite() + [
        NeuralPredictor(arch="gru", epochs=60, refit_every=50),
        NeuralPredictor(arch="transformer", epochs=60, refit_every=50),
    ]
