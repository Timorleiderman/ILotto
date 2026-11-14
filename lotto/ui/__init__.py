"""The local dashboard: `uv run lotto ui`.

Presentation only. `registry` lists the models and their settings, `service`
calls the existing computation in `lotto.core`, `lotto.approaches` and
`lotto.tickets`, `jobs` runs one request at a time in the background,
`storage` keeps results under ``results/ui/``, and `server` serves the page on
this machine only. No statistics are implemented here.
"""
