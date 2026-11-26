# Eval suite design principles: practitioners and production teams

Method note: every URL below was opened with WebFetch unless marked. WebFetch returns a model-written summary of the page, not raw text, so wording is paraphrase except where a short quote is shown. Items from search snippets only are in Gaps.

## 1. Where suite tasks come from (error analysis first, generic metrics)

### Takeaway
Practitioners agree: derive tasks from reading real traces and failures, not from imagined risks or off-the-shelf metrics. Start by reading about 30-100 traces, name the failure modes (keep under about 10), then write one check per observed failure mode.

### Cited Findings
- Hamel/Shreya FAQ calls error analysis "the most important activity in evals". Review at least 100 diverse traces; keep going until new traces reveal no new failure modes (theoretical saturation). Annotate at least 30 yourself before asking an agent for suggestions. — [Hamel evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Same FAQ on generic metrics: "Generic evaluations waste time and create false confidence". BERTScore/ROUGE are generally not useful for LLM outputs. Generic scores are fine only as a way to find traces worth reading. — [Hamel evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Field guide: error analysis is "the single most valuable activity in AI development". Method is bottom-up: free-form notes per conversation, LLM drafts a taxonomy, then label and count. Generic metrics called "worse than useless". Example: three failure types caused over 60% of problems. — [Hamel field guide](https://hamel.dev/blog/posts/field-guide/)
- Husain and Shankar (Lenny's Newsletter, partly paywalled): one principal domain expert reviews about 100 interactions; open coding then axial coding; aim for under 10 primary failure modes; humans must validate any LLM-assisted categorization. Avoid dashboards of hallucination/toxicity scores. Mistakes listed: starting with ready-made metrics, too many failure categories, terse critiques, Likert scales, accuracy as the only judge metric. — [Lenny's: building eval systems](https://www.lennysnewsletter.com/p/building-eval-systems-that-improve)
- Hamel "Your AI product needs evals": three levels (unit tests, human/model eval, A/B). Derive assertions by examining traces and failure modes. Organize tests by feature and scenario, plus generic checks. Make synthetic inputs challenging but realistic. A model that struggles with a test shows the test is useful. — [Hamel: your AI product needs evals](https://hamel.dev/blog/posts/evals/)
- FAQ common mistakes: writing a rubric before reviewing examples (criteria drift); one judge for many error types; outsourcing error analysis to non-experts; Likert scales; composite weighted scores that hide what changed; writing evaluators for failures you imagine instead of observe. — [Hamel evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Hamel judge post: after each fix, write a test case for that error (assertion or specialized judge). Re-run the whole process after material changes such as a model update. — [Hamel: LLM-as-judge](https://hamel.dev/blog/posts/llm-judge/)
- Anthropic: start with 20-50 simple tasks drawn from real failures (manual pre-release checks, bug reports, support tickets), prioritized by user impact. — [Anthropic: demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Eugene Yan: prefer organic failures from production or smaller models over synthetic defects, which are often out-of-distribution. Aim for 50-100 fail cases among 200+ samples for judge alignment. — [Eugene Yan: product evals](https://eugeneyan.com/writing/product-evals/)
- Eugene Yan: expect criteria drift, review at least 20 examples first, add criteria as new failure types appear, include hard negatives, practitioners found code assertions easier to trust than judge prompts. — [Eugene Yan: LLM evaluators](https://eugeneyan.com/writing/llm-evaluators/)
- Shankar et al.: users need criteria to grade, but grading outputs changes the criteria (criteria drift); alignment of evaluators is iterative, not one-time. — [arXiv 2404.12272](https://arxiv.org/abs/2404.12272)
- LangChain sources of evals: hand-curated, production traces (a user correction is "even better"), curated external datasets, dogfooding reports with trace links. — [LangChain: Better Harness](https://www.langchain.com/blog/better-harness-a-recipe-for-harness-hill-climbing-with-evals)
- Notion: evals start from real failures; first step was reviewing worst customer experiences; built a dataset of every case where language adherence failed (mixed Japanese/Korean/English workspaces), guarded by an LLM judge. Small-segment, high-priority cases ("needle-in-a-haystack") get their own tasks. (Vendor customer page, Braintrust.) — [Braintrust: Notion](https://braintrust.dev/customers/notion)
- GitHub (offline eval post): build the set to mirror the production task including context; "a cleaner dataset may exclude ambiguous cases" and inflate scores. Sources: production data (labels as workflow signals, not truth), targeted synthetic cases for hard patterns, adapted public benchmarks. — [GitHub: how to evaluate LLMs before production](https://github.blog/2026-08-25-how-to-evaluate-llms-before-production/)
- Airbnb: teams read roughly 100 outputs first, categorize failures; golden labels from experts; new failure modes found in production sampling lead to new evals. — [Airbnb: eval-driven development](https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/)

### Inferences
- For a 28-task suite, the defensible origin story is: tasks map to named failure modes from reading their own traces, each task tagged to a mode. Tasks with no observed-failure origin are the weak ones.
- Hamel's tension with Anthropic: Hamel wants hard tests with pass rate maybe near 70%; Anthropic wants regression tasks near 100%. Both fit if the suite is split into capability and regression groups (see section 3).

### Gaps
- Lenny's post Phase 3 (flywheel, CI vs production, eval counts) is paywalled; not read.
- Eugene Yan's page did not use the term "error analysis"; I did not find an Eugene Yan post specifically on deriving suites from traces beyond the two above.
- Shreya Shankar's own site (sh-reya.com) not opened.

## 2. Offline vs online: how production teams split it

### Takeaway
Teams use offline evals as a pre-release gate for named regressions, and online sampling/A-B tests/monitoring for what the suite cannot see. The link is a loop: new production failures become offline tasks.

### Cited Findings
- Hamel FAQ: "CI evals protect against known regressions before deployment". Production monitoring finds failures in live traffic and estimates frequency. When reviews reveal new failure patterns, add examples to the CI dataset. Online traces usually lack reference outputs, so reference-free judges are common. One approach: cheap checks on every trace, expensive judge on a nightly sample. Track confidence intervals. — [Hamel evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Anthropic: automated evals catch problems before release; production monitoring finds drift and real failures that synthetic suites miss; feed production failures back; use periodic human review to calibrate graders. — [Anthropic: demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Intercom Fin: offline backtests replay a few thousand real past conversations through production and candidate with identical inputs; LLM judges grade groundedness, guidance adherence, and correct escalation. Online A/B has one declared primary metric (resolution rate), power analysis, whole-week durations. Rollouts start dark at 0%, then 1%, with health checks. Standing rule: never knowingly ship a resolution regression or more hallucinations. Failure modes found in monitoring become cases in the next backtest. — [Fin: why not just ship it](https://ideas.fin.ai/p/why-not-just-ship-it)
- GitHub Copilot: more than 4,000 offline tests, most in CI; chat has 1,000+ technical questions. Separate live internal evaluation switches internal users to a new model (canary-like). Chat tests run daily against production models, and degradation triggers audits. Completions metrics: share of broken unit tests fixed; chat: share of questions answered correctly; both track token use. Complex chat answers graded by LLM, audited against human reviewers. — [GitHub: how we evaluate models for Copilot](https://github.blog/ai-and-ml/generative-ai/how-we-evaluate-models-for-github-copilot/)
- GitHub (2026 post): offline passing gives "enough structured evidence to justify moving to online experimentation"; experiments advance only if primary gains hold while recall stays inside a guardrail. Latency, cost, reliability are guardrails alongside quality. — [GitHub: how to evaluate LLMs before production](https://github.blog/2026-08-25-how-to-evaluate-llms-before-production/)
- GitHub MCP Server offline eval: each case is a user request, expected tool(s), expected arguments. Metrics: tool-selection accuracy/precision/recall/F1 plus confusion matrix; argument checks (hallucinated names, missing required args, exact values). Admitted limits: thin per-tool volume, single tool calls only. Cadence not stated. — [GitHub: MCP Server offline eval](https://github.blog/ai-and-ml/github-copilot/measuring-what-matters-how-offline-evaluation-of-github-mcp-server-works/)
- Airbnb: offline evals run on the golden set before release; online evals sample live de-identified traffic (example: about 5% daily), run programmatic checks and judges, send flagged outputs to human review. — [Airbnb](https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/)
- Discord (2023-24 post): offline stage uses a stronger critic LLM with yes/no prompts plus traditional metrics; online stage is limited release/A-B with satisfaction and system-health metrics (latency, errors, output-format adherence). Post does not describe how the eval set is built. — [Discord: developing rapidly with generative AI](https://discord.com/blog/developing-rapidly-with-generative-ai)
- Notion: regression evals (check breakage; new models may score about the same) versus frontier evals (show how a new model differs). About 80% of AI work grounded in Braintrust evals and traces; advice to rebuild eval systems about every six months. Vendor page. — [Braintrust: Notion](https://braintrust.dev/customers/notion)

### Inferences
- Common shape: offline = named-failure regression gate; online = sampled judge + human review + A/B on one primary metric; guardrail metrics (hallucination, cost, latency) sit beside quality in both.
- For this user's suite, "tool-selection and argument correctness" at GitHub MCP is the closest published analogue to their tool tasks, and GitHub itself says volume per tool is the weak point.

### Gaps
- No primary eval posts found for Duolingo, Stripe, Replit, Ramp, DoorDash, Microsoft (not searched deeply beyond one combined query), Google, or LinkedIn. A LinkedIn search returned only secondary coverage (The New Stack, ByteByteGo); not opened, so not cited.
- Copilot ZenML "production-first" talk summary appeared in search; not opened, and flagged by the search itself as secondhand and dated 2023. Not used.

## 3. Keeping the suite fresh (add failures, retire saturated tasks, versioning)

### Takeaway
The suite is a living artifact: add production failures continuously, graduate or retire saturated tasks, version datasets and configs, and re-check judges on a schedule.

### Cited Findings
- Anthropic: split capability evals (start low pass rate) from regression evals (stay near 100%). Capability tasks that saturate "graduate" into regression. Saturation means no more visible improvement, so add harder tasks. Treat the suite as living with clear ownership. If a task scores 0% over many trials, suspect a broken task or grader first. — [Anthropic: demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Hamel FAQ: a 100% pass rate likely means you are not challenging the system; suggests around 70% may be more meaningful. If an eval passes every example it gives no new information; retire it or run less often, but first try making it harder. High-stakes evals can stay frequent. — [Hamel evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Hamel: 100% pass is not the goal; threshold is a product decision about tolerable failures. Update tests as production failures appear. — [Hamel: your AI product needs evals](https://hamel.dev/blog/posts/evals/)
- LangChain: tag every eval to a behavior category; the suite "should not grow monotonically"; periodic spring cleaning removes saturated evals; keep a must-always-pass subset; use per-category optimization and holdout sets because hill-climbing overfits; quality over quantity. — [LangChain: Better Harness](https://www.langchain.com/blog/better-harness-a-recipe-for-harness-hill-climbing-with-evals)
- LangChain: each production failure becomes a candidate eval. — [LangChain: improving agents is a data mining problem](https://www.langchain.com/blog/improving-agents-is-a-data-mining-problem)
- GitHub: every run records dataset version, prompt, model and system config; prompts and eval configs treated like code, versioned and rollback-capable. Test one major variable at a time. — [GitHub: how to evaluate LLMs before production](https://github.blog/2026-08-25-how-to-evaluate-llms-before-production/)
- Hamel FAQ cadence: each review cycle at least 100 fresh traces, typical cycle 2-4 weeks; between cycles review 10-20 traces weekly, focusing on outliers. — [Hamel evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Airbnb: recalibrate judges regularly because failure modes change. — [Airbnb](https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/)
- Eugene Yan: guard against contamination (one cited study used posts after model cutoff). — [Eugene Yan: LLM evaluators](https://eugeneyan.com/writing/llm-evaluators/)

### Inferences
- Minimum suite metadata a reviewer can check: failure-mode tag, origin (trace/bug/synthetic), group (regression vs capability), date added, last-time-failed, dataset version recorded per run.
- A 28-task suite with all tasks passing is a red flag by both Hamel and Anthropic, unless it is labeled regression-only.

### Gaps
- No source gave a numeric retirement rule (for example "retire after N green runs"). The "mark stale after 90 days" claim came from a search snippet of a DEV Community article that cited Statsig; I could not verify it, so it is not used.

## 4. Validating LLM judges

### Takeaway
Treat the judge as a model that needs its own labeled test set. Use binary verdicts, one judge per failure mode, report true positive and true negative rates separately, and keep a held-out test split.

### Cited Findings
- Hamel FAQ: split labeled data train 10-20%, dev 40-45%, test 40-45%; keep test untouched until the end; report TPR and TNR; about 30-50 pass and 30-50 fail in dev and test, minimum 50 and 50 for debugging disagreements; plan 100-200 labeled examples per judge failure mode; separate judge per failure type. — [Hamel evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Hamel judge post: critique shadowing by one principal domain expert; binary pass/fail with detailed critique; raw agreement misleads when failures are rare; "Below 60 examples, the confidence intervals are often too wide". — [Hamel: LLM-as-judge](https://hamel.dev/blog/posts/llm-judge/)
- Eugene Yan: dev about 75% / test about 25%; one evaluator per dimension, no "God Evaluator"; measure precision, recall on the fail class, Cohen's kappa; run pairwise comparisons in both orders for position bias; benchmark is human performance, not perfection. — [Eugene Yan: product evals](https://eugeneyan.com/writing/product-evals/)
- Eugene Yan: check both error types (one summarization evaluator caught over 95% of consistent summaries but 30-60% of inconsistent); biases listed: position, verbosity, self-enhancement, prompt sensitivity. — [Eugene Yan: LLM evaluators](https://eugeneyan.com/writing/llm-evaluators/)
- Shankar et al.: LLM evaluators inherit flaws of the models they judge and need human checking; EvalGen selects evaluators by matching user grades on a subset. — [arXiv 2404.12272](https://arxiv.org/abs/2404.12272)
- Airbnb: golden set 50-100 examples that MUST include bad examples; agreement via Cohen's kappa or Krippendorff's alpha; target high 80s to 90s percent; judges use a different model than the generator; 3-5 judges each scoring one dimension. (Note: percent agreement target conflicts with Hamel/Yan's advice to prefer TPR/TNR on imbalanced data.) — [Airbnb](https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/)
- Intercom Fin: ensemble across model families, average rather than trust single verdicts, humans periodically label samples to calibrate. — [Fin](https://ideas.fin.ai/p/why-not-just-ship-it)
- GitHub: judge output is "another prediction rather than ground truth"; route low-confidence/high-impact cases to humans; sample high-confidence judgments; version the judge prompt like code. — [GitHub: how to evaluate LLMs before production](https://github.blog/2026-08-25-how-to-evaluate-llms-before-production/)
- Copilot: LLM-graded chat answers are audited against human reviewers. — [GitHub: how we evaluate models for Copilot](https://github.blog/ai-and-ml/generative-ai/how-we-evaluate-models-for-github-copilot/)
- Vercel primer: judges vary; position bias ranged 0.002 to 0.192 across 21 judges; same-verdict rate dropped from above 95% at temperature 0 to as low as 70% at temperature 1. Secondhand figures from a vendor primer. — [Vercel primer](https://vercel.com/i/ai-agent-evaluation-frameworks-production)
- Anthropic: two domain experts should agree on each task verdict; reference solution should pass all graders; read transcripts to confirm failures are the agent's fault, not the grader's. — [Anthropic: demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

### Inferences
- With about 28 tasks, a full judge validation (100+ labels per mode) is not realistic per task. A reviewer should at least expect: each judge-graded task has a labeled pass/fail set, TPR/TNR reported, and code assertions used wherever possible (Hamel, Yan, Vercel all prefer them).

### Gaps
- Judge validation numbers for very small suites are not covered by any source opened.

## 5. Cost, speed, and trials

### Takeaway
Order checks by cost, run cheap deterministic checks on every change, and run judge-heavy or many-trial runs less often. Run multiple trials per task and report consistency, not a single score.

### Cited Findings
- Hamel: cheap unit-level tests run on every change; human/model eval on a schedule; A/B only after major changes. Cost grows by level. — [Hamel: your AI product needs evals](https://hamel.dev/blog/posts/evals/)
- Hamel FAQ: CI datasets are small and purpose-built; "Favor assertions or other deterministic checks over LLM-as-judge evaluators". Expensive evals must justify frequent runs. — [Hamel evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Vercel primer: start with the cheapest check that can catch the failure; ladder from deterministic assertions to exact match to similarity to judge; a presence check on a required tool call often answers a trajectory question without a judge; expensive-first ordering makes the suite too slow to run consistently; CI gate fails the merge on failure. — [Vercel primer](https://vercel.com/i/ai-agent-evaluation-frameworks-production)
- Airbnb: run cheap programmatic checks first, then judges, humans for ground truth and edge cases. — [Airbnb](https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/)
- LangChain: tags let you run subsets; stratified sampling for large or costly sets. — [LangChain: Better Harness](https://www.langchain.com/blog/better-harness-a-recipe-for-harness-hill-climbing-with-evals)
- Anthropic: run several trials per task; pass@k when one success is enough, pass^k when consistency matters (customer-facing); keep trials isolated so shared state does not correlate failures. — [Anthropic: demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Eugene Yan: size sample to the decision; halving the margin of error needs about 4x samples; example: 200 samples at a 3% defect rate gives about +/-2.4%. Avoid demanding 90%+ judge accuracy without a human-agreement baseline. — [Eugene Yan: product evals](https://eugeneyan.com/writing/product-evals/)
- Eugene Yan: dev can tolerate API cost on a few hundred samples; high-throughput guardrails may need a finetuned classifier; a panel of smaller judges reached higher correlation than GPT-4 at about one-seventh the cost (cited study). — [Eugene Yan: LLM evaluators](https://eugeneyan.com/writing/llm-evaluators/)
- Discord: used token counts from limited release to project full-launch cost; cost and latency tracked beside quality. — [Discord](https://discord.com/blog/developing-rapidly-with-generative-ai)
- Intercom Fin and GitHub both put cost and latency in the release scorecard, not only quality. — [Fin](https://ideas.fin.ai/p/why-not-just-ship-it); [GitHub](https://github.blog/2026-08-25-how-to-evaluate-llms-before-production/)

### Inferences
- A "fast tier (3 trials, local, mostly deterministic) and slow tier (10 trials, judges, manual CI)" split matches Hamel, Vercel and Anthropic. No source prescribes exact trial counts; 3 and 10 are the user's choice. At 3 trials pass^k and pass@k are coarse, so a reviewer should check that the report shows per-task trial results, not only an average.

### Gaps
- No source gave a concrete wall-clock or dollar budget for a suite of about 30 tasks.
- "Fast tier / slow tier" is not an official term in these sources; Vercel's page does not name tiers (I inferred it).

## 6. What teams get wrong (summary list with sources)

### Takeaway
Most-cited mistakes: skipping error analysis, generic metrics, uncalibrated judges, Likert/composite scores, a suite that only ever passes, and no link to production.

### Cited Findings
- Tools first, skipping error analysis, generic dashboards, over-automating, trusting unvalidated judges — [Hamel field guide](https://hamel.dev/blog/posts/field-guide/)
- Criteria written before looking at outputs; one judge for many errors; Likert; composite scores; eval-driven development on imagined failures — [Hamel evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Synthetic-only data, God evaluator, too few failure cases, position bias ignored, too few samples to see a real change — [Eugene Yan: product evals](https://eugeneyan.com/writing/product-evals/)
- Suite saturates and stops showing progress; single run hides variance; broken grader blamed on agent — [Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Overly clean offline dataset inflates scores vs production — [GitHub](https://github.blog/2026-08-25-how-to-evaluate-llms-before-production/)
- Evals treated as one-time offline activity — [Vercel primer](https://vercel.com/i/ai-agent-evaluation-frameworks-production)
- Thin per-tool coverage (GitHub admits this itself) — [GitHub MCP offline eval](https://github.blog/ai-and-ml/github-copilot/measuring-what-matters-how-offline-evaluation-of-github-mcp-server-works/)

### Inferences
- Source conflict to flag: Hamel suggests about 70% pass rate may be healthier; Anthropic says regression tasks should be near 100%. Different groups, so not a true conflict.
- Source conflict: Airbnb's percent-agreement target vs Hamel/Yan's TPR/TNR preference.

### Gaps
- Google "Rules of Machine Learning" not opened (out of time budget); no suite-level principle taken from it.
- Shankar et al. full paper not read, only the abstract page summary.
