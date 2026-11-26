# Security eval research (2026-11-26)

Research notes, not committed. Spot-checked sources are marked CHECKED.

## Sources

- OWASP-LLM: OWASP Top 10 for LLM Applications 2025. https://genai.owasp.org/llm-top-10/
- OWASP-AGENT: OWASP Top 10 for Agentic Applications. CHECKED.
  https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/
- OAI: OpenAI ChatGPT agent system card. CHECKED (exfiltration and system prompt extraction tests).
  https://deploymentsafety.openai.com/chatgpt-agent
- GDM: Lessons from Defending Gemini Against Indirect Prompt Injections. CHECKED.
  https://arxiv.org/abs/2505.14534
- ANTH-PI: Anthropic, Mitigating prompt injections in browser use. CHECKED (adaptive Best-of-N attacker).
  https://www.anthropic.com/research/prompt-injection-defenses
- ANTH-EVALS: Anthropic, Demystifying evals for AI agents.
  https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Anthropic system cards (only the table of contents was read):
  Sonnet 4.5 https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf
  Opus 4.5 https://www-cdn.anthropic.com/bf10f64990cfda0ba858290be7b8cc6317685f47.pdf

## Behaviors to evaluate, and where this agent stands

| # | Behavior | Named by | Status here | Public dataset |
|---|---|---|---|---|
| 1 | Injection hidden in a fetched page or tool result | OWASP-LLM LLM01, OWASP-AGENT ASI01, OAI, GDM, ANTH-PI | DONE (BIPIA + canary pages) | AgentDojo, InjecAgent, BIPIA |
| 2 | Injection hidden in an uploaded document (RAG) | OWASP-LLM LLM01/LLM08, OWASP-AGENT ASI06 | MISSING | none named |
| 3 | Data exfiltration after an injection (link, image, search query) | OAI, GDM, OWASP-LLM LLM02 | HALF DONE (canaries only) | none public, labs use internal sets |
| 4 | Direct jailbreak | OAI, Anthropic cards | DONE (JailbreakBench) | StrongREJECT, JailbreakBench |
| 5 | Over-refusal on harmless questions | XSTest, OR-Bench papers | DONE (benign twins) | XSTest, OR-Bench |
| 6 | System prompt leak and instruction hierarchy | OAI, OWASP-LLM LLM07 | MISSING | none named |
| 7 | One user's documents leaking to another user | OWASP-LLM LLM02 | MISSING | none, write it here |
| 8 | Unsafe output (markdown, HTML, links in the stream) | OWASP-LLM LLM05 | MISSING | none named |
| 9 | Harmful task done with the tools | Anthropic cards, OAI | MISSING | AgentHarm |
| 10 | Groundedness and hallucination | ANTH-EVALS, OAI, OWASP-LLM LLM09 | MISSING as a security check | SimpleQA |
| 11 | Wrong or extra tool calls | ANTH-EVALS, OWASP-LLM LLM06, OWASP-AGENT ASI02 | partly, in the tools suite | none named |
| 12 | Unbounded cost (loops, huge pages, long uploads) | OWASP-LLM LLM10 | MISSING | none named |

Not relevant here: confirmation before risky actions (no side-effect tools),
supply chain, data poisoning, multi-agent and code-execution risks.

Suggested order: 2, 3, 6, 7, 8, then 9 and 10.

UNBACKED: NIST AI 600-1 and MITRE ATLAS were not read. Anthropic card details
beyond the table of contents were not read. No lab source found for voice-mode
injection evals.

## Guardrail packages

| Package | Covers | Status |
|---|---|---|
| NVIDIA NeMo Guardrails | framework: input, output, tool-result, topic, hallucination rails | v0.25.0, 2026-11-19 (CHECKED), Apache-2.0, Python <3.14 |
| Microsoft Presidio | PII only, runs locally | MIT, maintained |
| Meta Prompt Guard 2 (22M/86M) | injection and jailbreak, small local model, 512-token input | Llama license |
| Meta Llama Guard 4 (12B) | input and output moderation, 14 harm categories | not on Together serverless |
| IBM Granite Guardian | harm, jailbreak, RAG groundedness | needs your own GPU |
| Guardrails AI | validators hub, incl. topic and provenance | v0.11.0, hosted inference ended 2026-08-25 |
| Protect AI LLM Guard | wide scanner list | ARCHIVED (CHECKED), skip |

Recommended combo: NeMo Guardrails as the runner, Presidio for PII,
Prompt Guard 2 for injection in user input and tool results, one hosted safety
classifier for moderation and output, NeMo topic rail for banned topics.

Keep from the current code: the tool-output fence, the input guard as the one
decision point, and the regex detectors as a cheap first filter. Replace regex
PII with Presidio. Add the output guard, which does not exist yet.

Not verified: real latency of the model-based rails, and whether NeMo works
with the LangGraph plus Together setup.
