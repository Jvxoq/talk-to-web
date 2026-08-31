#!/bin/sh
# Render's Docker Command is not a shell. It expands variables and strips
# quotes, then execs the argv directly, so `&&` reached alembic as an argument
# and `'*'` lost its quotes. Two commands therefore need a file, not a line.
#
# Migrations run here rather than in a pre-deploy hook because that hook needs a
# paid instance. Alembic is idempotent, so a boot with nothing to apply costs
# one query. This is only safe while the service runs a single instance.
set -e

alembic upgrade head

# exec replaces this shell, so Render's stop signal reaches uvicorn itself.
# --forwarded-allow-ips '*' trusts X-Forwarded-For, which is safe only because
# Render's proxy is the one thing that can reach this port.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --limit-concurrency 5 \
    --forwarded-allow-ips '*'
