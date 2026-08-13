"""Eval harness for the chat agent.

Lives beside `app/`, not inside it: `root_package = "app"` in the
import-linter config makes `app/` the only package the layering rule sees, so
an eval driver here is invisible to it - the same altitude as `app/main.py`,
free to import `app.composition`, `app.settings` and any adapter directly.
That freedom is the reason this package exists outside `app/` rather than as
another application context: production must not carry code it never calls,
and an eval run needs real API keys and a real (if scratch) vector store,
neither of which belongs anywhere near `app.domain` or `app.application`.

Run with:

    uv run python -m evals --suite tools --limit 10 --concurrency 4 \\
        --out evals/results/latest.json
"""
