FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

FROM python:3.13-slim-bookworm

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8000

CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
