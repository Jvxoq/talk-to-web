#!/bin/sh
# A script, because Render's Docker Command is not run through a shell.
# Migrations run on every boot, since the pre-deploy hook needs a paid plan.
# Safe only while there is one instance.
set -e

alembic upgrade head

# exec, so stop signals reach uvicorn. Trusting X-Forwarded-For is safe
# because only Render's proxy can reach this port.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --limit-concurrency 5 \
    --forwarded-allow-ips '*'
