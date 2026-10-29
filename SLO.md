# SLO

An SLO (Service Level Objective) is a target for one measured number.
The user wants three things: a correct answer from their file, a fast reply,
and a safe agent. Every metric below checks one of those.

| Metric | Target | Current | Source |
|---|---|---|---|
| **Correct** | | | |
| retrieval_hit_rate@3 | >= 90% | 33% (n=3, gpt-oss-120b, 2026-08-13) | `--suite rag` |
| groundedness | >= 90% | 67% (n=3, gpt-oss-120b, 2026-08-13) | `--suite rag`, judge |
| tool_routing_exact_match | >= 80% | 36% (n=11, DeepSeek V4 Flash, 2026-08-22) | `--suite tools` |
| **Fast** | | | |
| ttft_p95 | < 3 s | not measured | not built yet |
| reply_latency_p95 | < 20 s | 17.8 s (n=11, 2026-08-22) | `--suite tools` |
| reply_error_rate | < 1% | not measured | not built yet |
| cost_per_reply | < $0.01 | $0.0004 (n=3, 2026-08-13) | `--suite rag` |
| **Safe** | | | |
| prompt_injection_success_rate | 0 of >= 60 | 0 of 10 (2026-10-20) | `--suite injection_pages` |
| jailbreak_success_rate | 0 of >= 60 | 0 of 10 (2026-10-20) | `--suite jailbreak` |
| over_refusal_rate | < 10% | 20%, 1 of 5 (2026-10-20) | `--suite jailbreak`, benign set |

## Notes

- **ttft** is time to the first streamed word. It excludes a cold start.
- **groundedness** is a judge model's score. It is reported, never a gate.
- **over_refusal_rate** is the share of harmless questions the agent refused.
  It keeps the jailbreak number honest.
- Zero attacks out of 10 still allows a true rate up to 26%. 60 cases bring
  that bound under 5%.
- Nothing here blocks a merge today. CI runs only the `ci`-tagged cases.
- Update a row only from a saved run in `backend/evals/results/`.
