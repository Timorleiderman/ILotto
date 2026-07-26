FROM python:3.12-slim

# graphviz is for model_viz.py's architecture diagrams; git so the build can be
# run against a checkout rather than a copy.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies come from pyproject.toml + uv.lock, the same pair CI installs
# from. There used to be a hand-maintained requirements.txt here; it drifted, and
# by the time anyone noticed it was missing `requests`, which both helpers.py and
# bench/data.py import — so the image could not refresh the draw archive at all.
# One source of truth avoids repeating that.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY . .

# The default entry point is the benchmark report rather than the legacy
# evaluate.py: it is the thing that is tested, and it writes the published site.
# Override to run anything else, e.g.
#   docker run --rm ilotto uv run python predict.py
ENV TF_CPP_MIN_LOG_LEVEL=3
CMD ["uv", "run", "python", "scripts/build_report.py", "--quick"]
