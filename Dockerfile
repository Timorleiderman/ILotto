FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies come from pyproject.toml + uv.lock, the same pair CI installs
# from. One source of truth: a hand-maintained requirements.txt used to live
# here and drifted until the image could no longer refresh the draw archive.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

# The default entry point is the benchmark report: it is the thing that is
# tested, and it writes the published site. Override to run anything else, e.g.
#   docker run --rm ilotto uv run lotto list
CMD ["uv", "run", "lotto", "report", "--quick"]
