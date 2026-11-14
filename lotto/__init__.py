"""ILotto: a reproducible study of the Israeli Lotto (6/37 + strong 1-7).

The package is organised by subject. Read them in this order:

    core/         the measuring apparatus: data, metrics, backtest, randomness tests
    approaches/   every idea that tries to predict the draw, simplest first
    forecasting/  probability forecasts of the complete draw
    tickets/      what a buyer controls: coverage, strong numbers, which form to play
    seedsearch/   a side quest: recovering the seed of an artificial generator
    report/       rendering of the published benchmark report
    experiments/  one runnable module per experiment, listed in a catalog

Run anything with ``uv run lotto``.
"""
