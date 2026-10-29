FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# `--no-dev`, so test and lint tools stay out of the production image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.13-slim-bookworm

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app

# Create the uploads dir owned by appuser, or a mounted volume is root-owned
# and every upload fails with EACCES.
RUN mkdir -p /app/uploads && chown appuser:appuser /app/uploads

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8000

# Liveness only, so a database blip does not kill the container. urllib, since
# the slim image has no curl. Reads PORT, which needs the shell form.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4)"

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
