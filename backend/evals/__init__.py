"""Eval harness for the chat agent.

Outside `app/`, so the import linter ignores it and it may import adapters
and settings directly. Production does not ship it.

Run with:

    uv run python -m evals --suite tools --limit 10 --concurrency 4 \\
        --out evals/results/latest.json
"""
