"""The security evaluation suite.

Run with:

    uv run python -m evals.security --mode gate

The design, the terms and the gates are in `security-evals/security-eval-design.md`.

The package splits in two, so the next agent reuses most of it:

- Core, agent-agnostic: `tasks`, `graders`, `metrics`, `gates`, `report`. These
  never import `app` or `evals.harness`. A test enforces it.
- Adapter, one per agent: `adapter` and `cost`. They know this agent's tools,
  its web fetcher and its thread store.
"""
