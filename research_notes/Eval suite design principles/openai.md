# OpenAI guidance on whole-suite eval design (as of Nov 2026)

Note on access: platform.openai.com URLs 301-redirect to developers.openai.com/api/docs/...; all content below was read there. The fetch tool summarizes pages with a small model, so quotes are as returned by that tool and not verified against raw page text. Quotes under 15 words.
NOT opened / failed: the OpenAI "From experiments to deployments" PDF (too large to fetch). No OpenAI Agents SDK page on evals found (tracing page has none). No SimpleQA page opened.

## 1. What a suite must contain (dataset sourcing, edge cases, production-like data, size)

### Takeaway
OpenAI says: build the dataset from mixed sources, include typical, edge and adversarial cases, and make it match real production traffic. It gives no target size; it says start small and grow.

### Cited Findings
- Five-step process: define objective, collect dataset, define metrics, run and compare, evaluate continuously. — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- "Ensure your test data includes typical cases, edge cases, and adversarial cases" — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Named pitfall: "Creating eval datasets that don't faithfully reproduce production traffic patterns." — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Other named pitfalls: relying on generic metrics (BLEU, perplexity), vibe-based evals (skipping evals until launch), and "Not calibrating your automated metrics against human evals." — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Sources to mix: synthetic, domain-specific, purchased, human-curated, production, historical data; use expert labelers. — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- "Log as you develop so you can mine your logs for good eval cases." — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Page gives no recommended dataset size (only an illustrative "1000 reference transcripts"). — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Cookbook: "Begin with a very small set of labeled data"; use an early system version to generate draft truth data, then experts correct it; "aiming for breadth rather than depth at first"; treat labeling as a scarce resource. — [Eval-Driven System Design](https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection)
- Cookbook: study the examples that failed; "not all evals matter" (check that an eval actually affects outcomes). — [Eval-Driven System Design](https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection)
- Flywheel cookbook: start with about 50 failing traces for open coding, then group into failure categories (axial coding) to see which dominate; synthetic data can supplement when real examples are scarce. — [Evaluation flywheel](https://developers.openai.com/cookbook/examples/evaluation/building_resilient_prompts_using_an_evaluation_flywheel)
- Evals API structure: two parts, a data schema (`data_source_config`) and graders (`testing_criteria`); data is JSONL with inputs and ground-truth labels. — [Evals guide](https://developers.openai.com/api/docs/guides/evals)

### Inferences
- A suite should be tagged by case type (typical / edge / adversarial) so coverage of each is visible.
- Cases should come from real or realistic traffic for this app (uploaded files, real chats), not only hand-invented prompts.

### Gaps
- No OpenAI statement on a minimum number of tasks or on per-task pass-rate targets or number of trials.

## 2. Eval-driven development, CI, and tying evals to changes and decisions

### Takeaway
OpenAI frames it as "evaluate early and often", with continuous evaluation (CE) that runs evals on every change, and a loop where failures feed new eval cases. They never say "CI" by name on the best-practices page.

### Cited Findings
- "Evaluate early and often." (eval-driven development) — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- "Set up continuous evaluation (CE) to run evals on every change"; also watch production for new nondeterminism and expand the eval set over time. — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Evals guide: workflow is describe task as eval, run against test inputs with a prompt, analyze and iterate; used to "keep tabs on the performance of your prompts as you iterate". — [Evals guide](https://developers.openai.com/api/docs/guides/evals)
- Cookbook: "The evals act as an objective guide"; "We're not done once system performs well according to our evals" (set keeps growing); after launch "prioritize continuous monitoring"; "Production data is your most authentic source." — [Eval-Driven System Design](https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection)
- Cookbook ties evals to business: build a cost model, map evals to KPIs and dollar impact, weigh performance against cost to develop and run (search-result summary plus fetched page). — [Eval-Driven System Design](https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection)
- Flywheel: three phases analyze, measure, improve as a "continuous cycle"; graders let you "easily evaluate our performance on any change to our system". — [Evaluation flywheel](https://developers.openai.com/cookbook/examples/evaluation/building_resilient_prompts_using_an_evaluation_flywheel)
- Trace evals: benchmark changes, catch regressions, confirm improvements. — [Trace grading](https://developers.openai.com/api/docs/guides/trace-grading)
- openai/evals README: "creating high quality evals is one of the most impactful things you can do." Without evals it is hard to see how model versions affect your use case. — [openai/evals](https://github.com/openai/evals)
- Platform note: the Evals platform and graders-in-evals are marked as being deprecated; check deprecations page. — [Evals guide](https://developers.openai.com/api/docs/guides/evals), [Graders](https://developers.openai.com/api/docs/guides/graders)

### Inferences
- "Run on every change" supports a cheap local run plus a heavier CI run, which fits the user's 3-trial local / 10-trial CI plan. OpenAI does not state those numbers.
- Release decisions should be tied to a stated success criterion set before running ("Define eval objective").

### Gaps
- No OpenAI text found on pass-rate gates, flaky-eval handling, or multiple trials per task. The whitepaper PDF might cover gating; not opened.

## 3. Grader types and validating LLM graders against humans

### Takeaway
Prefer cheap deterministic checks, use LLM judges where needed, and prove the judge agrees with expert humans before trusting it. Use pass/fail rather than fine scores.

### Cited Findings
- Grader types: string check, text similarity, score model (LLM), Python. — [Graders](https://developers.openai.com/api/docs/guides/graders)
- Metric-based evals suit regression testing but "may miss nuance"; human evals are best but slow and costly. — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- LLM-as-judge has "Position bias (response order), verbosity bias (preferring longer responses)"; use pairwise or pass/fail grading and control for length. — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- "Include a pass/fail threshold in addition to the numerical score." — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- "Use human feedback to calibrate automated scoring"; scale an LLM judge once it "consistently agrees with human annotations". — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Validate a model grader by building an eval with expert answers and ground-truth grades; check higher-quality answers score higher; add edge cases as grader errors are found. — [Graders](https://developers.openai.com/api/docs/guides/graders)
- Flywheel: measure the judge "against a human subject-matter expert"; track true positive and true negative rate, not raw accuracy, because sets are imbalanced; split roughly 20% train / 40% validation / 40% test. — [Evaluation flywheel](https://developers.openai.com/cookbook/examples/evaluation/building_resilient_prompts_using_an_evaluation_flywheel)
- Cookbook: use several narrow graders, "Each grader evaluates some portion of a predicted output", and measure both end-to-end and per step. — [Eval-Driven System Design](https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection)

### Inferences
- Each LLM judge in the suite needs its own labeled validation set and a recorded TPR/TNR.

### Gaps
- No OpenAI-stated required agreement level (e.g. kappa) found.

## 4. Agent-specific evals (tools, traces, multi-step)

### Takeaway
OpenAI says agents add nondeterminism in tool choice and handoffs, so test tool selection and arguments, and grade the whole trace, not only the final answer.

### Cited Findings
- Single-agent: "This introduces a new opportunity for nondeterminism"; test tool selection and argument accuracy. — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Workflows: evaluate each model call in isolation. — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Multi-agent: test handoff accuracy; architecture choice "should be driven by your evals". — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- Trace = "end-to-end log of decisions, tool calls, and reasoning steps"; trace evals give insight into "why an agent succeeds or fails" vs black-box testing. — [Trace grading](https://developers.openai.com/api/docs/guides/trace-grading)
- Trace graders can ask "Did the agent pick the right tool?" and check handoffs and policy compliance; move from single traces to datasets and eval runs for repeatability. — [Agent evals](https://developers.openai.com/api/docs/guides/agent-evals)
- openai/evals README points to a Completion Function Protocol for prompt chains or tool-using agents. — [openai/evals](https://github.com/openai/evals)

### Inferences
- For this app, the suite should have tasks per tool decision (RAG vs search vs fetch vs none), argument checks, and trace-level checks on multi-step runs, plus a summarization task set.

### Gaps
- Agents SDK tracing page says nothing about evals (opened: [Agents SDK tracing](https://openai.github.io/openai-agents-python/tracing/)). Nothing found on long-conversation or RAG-specific suite design from OpenAI.

## 5. Suite-level principles from OpenAI papers and benchmarks

### Takeaway
Validate the tasks themselves with several humans, keep tasks realistic and rubric-graded, and design scoring so it does not reward guessing.

### Cited Findings
- SWE-bench Verified: "Invest in deeply understanding our benchmarks"; vague issues and overly specific tests can wrongly fail valid solutions; multiple annotators per sample, keep the most severe label, accept over-filtering; environment faults must be separated from solution quality; static public benchmarks need frequent re-evaluation. — [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)
- GDPval: tasks based on real work products, each had "5 rounds of expert review", recruited breadth of experts, written rubrics, blind expert grading; automated grader "not yet as reliable as expert graders"; limited to one-shot tasks and misses iterative context-building. — [GDPval](https://openai.com/index/gdpval/)
- Why LMs hallucinate: "Penalize confident errors more than you penalize uncertainty"; give partial credit for abstention; update main metrics, not only side tests. — [Why language models hallucinate](https://openai.com/index/why-language-models-hallucinate/)

### Inferences
- Each task's grader should be audited for unfair failures (over-specific assertions, environment flakiness) before the task counts toward the 97% target.
- Where the app should say "I don't know" (RAG with no answer), the suite should score a confident wrong answer worse than a refusal.

### Gaps
- SimpleQA not opened. No OpenAI statement on trial counts or 97% targets.
