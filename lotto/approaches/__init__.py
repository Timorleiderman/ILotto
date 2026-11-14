"""Every idea that tries to predict the draw, from folklore to neural networks.

Each module is one family of strategies behind the same contract,
``scores(history) -> (ball_scores[37], strong_scores[7])``. Read
``statistical.py`` first; ``neural.py`` and ``date_conditioned.py`` build on it.
"""
