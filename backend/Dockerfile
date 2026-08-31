FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# `--no-dev` in both layers: without it the production image ships pytest, mypy,
# ruff and import-linter. They are build-time tools, and every one of them is
# extra bytes to pull, extra packages to audit, and an interpreter that can run
# a test suite against production credentials.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.13-slim-bookworm

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app

# Create the mount point while still root, then hand it to appuser. Docker
# initialises an empty named volume from whatever the image has at that path,
# ownership included - so without this the volume arrives owned by root, the
# container runs as appuser, and every upload fails with EACCES. `uploads/` is
# gitignored and empty, so it is not in the build context to be copied.
RUN mkdir -p /app/uploads && chown appuser:appuser /app/uploads

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8000

# Liveness only, matching the endpoint's own contract: /health is static and
# touches no dependency, so a database blip cannot cascade into the container
# being killed. urllib rather than curl because the slim base has no curl and
# adding one for a healthcheck is a whole package of attack surface.
# PORT rather than a literal 8000: compose leaves it unset and the default
# applies, while a platform that assigns the port sets it and the check follows
# the server instead of probing a closed one. Shell form on purpose, because the
# exec form does no variable expansion.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4)"

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
