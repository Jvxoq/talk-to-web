# Anthropic guidance on whole-suite eval design

Method note: pages were read with WebFetch (a small model summarizes the page). Quotes below were returned as verbatim by the fetch, but the second pass was "verbatim" requested; a few were cut off mid-sentence ("truncated"). Re-check wording on the page before quoting in public. Not opened: building-effective-agents, effective-context-engineering (not needed for suite design; no fetch made). The two docs pages were only read in the first 100k of 151k chars (grading-methods section unread).
Primary: [Demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

## 1. Capability vs regression suites, graduation, saturation

### Takeaway
Two suites with different targets: capability starts low, regression stays near 100%. High-scoring capability tasks graduate to regression. A suite at 100% only guards, it gives no improvement signal.

### Cited Findings
- Capability evals "should start at a low pass rate"; regression evals "should have a nearly 100% pass rate" (fetch paraphrase, wording not confirmed verbatim) — [Demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Graduation: "capability evals with high pass rates can “graduate” to become a regression suite" — same page
- Saturation: "Eval saturation occurs when an agent passes all of the solvable tasks, leaving no room for improvement." — same page
- "An eval at 100% tracks regressions but provides no signal for improvement." (first-pass fetch; not re-confirmed verbatim) — same page
- Regression suite runs continuously to catch drift — same page

### Inferences
- The user's 97% per-trial target fits a regression suite, not a capability suite. A mix needs two labelled groups of tasks.

### Gaps
- No Anthropic number for when to graduate a task (no threshold given in what I read).

## 2. Task quality, balance, real failures, size, growth

### Takeaway
Few tasks, from real failures, unambiguous, each with a reference solution, balanced both ways, and the suite is maintained like unit tests.

### Cited Findings
- "20-50 simple tasks drawn from real failures is a great start." — [Demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Convert user-reported failures into tasks so the suite reflects real usage (paraphrase) — same page
- "Evals get harder to build the longer you wait" (fetch paraphrase) — same page
- "A good task is one where two domain experts would independently reach the same pass/fail verdict." — same page
- "For each task, it’s useful to create a reference solution: a known working output that passes all graders." — same page
- "a 0% pass rate across many trials (i.e. 0% pass@100) is most often a signal of a broken task" — same page. Action: check spec and graders.
- "One-sided evals create one-sided optimization." (test both when the behavior should and should not happen) — same page
- "An eval suite is a living artifact that needs ongoing attention and clear ownership to remain useful." — same page
- "owning and iterating on evaluations should be as routine as maintaining unit tests." — same page
- Dedicated team owns infrastructure, domain experts and product teams contribute tasks (paraphrase) — same page
- Eval-driven development: build evals for planned capabilities before the agent can do them (paraphrase) — same page
- Docs: "Design evals that mirror your real-world task distribution." — [Develop tests](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)
- Docs edge cases: "Irrelevant or nonexistent input data"; "Overly long input data or user input"; ambiguous cases — same page
- Docs: "More questions with slightly lower signal automated grading is better than fewer questions" — same page (note: a docs-page view for model-output evals; tension with the 20-50 start. Different context.)

### Inferences
- 28 tasks is inside Anthropic's 20-50 range. Review should check that each task came from a real failure or real usage, and that positive and negative twins exist.

### Gaps
- No guidance found on how often to add tasks or a target growth rate.

## 3. Graders, calibration, transcripts

### Takeaway
Mix code, model, and human graders. Prefer code where possible. Grade the outcome, not the path. Calibrate model judges against humans. Read transcripts.

### Cited Findings
- Three grader types: code-based, model-based, human; deterministic preferred "where possible" (paraphrase) — [Demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Grade what the agent produced, not the path it took (paraphrase); partial credit for multi-part tasks — same page
- Graders must resist bypass or hacks (paraphrase) — same page
- "LLM-as-judge graders should be closely calibrated with human experts" — same page
- "give the LLM a way out, like providing an instruction to return “Unknown”" — same page
- "Read the transcripts!" and graders cannot be trusted unless transcripts of many trials are read; failures should look fair (paraphrase) — same page
- Docs: use "a different model from the one used to generate the evaluated output" (code comment) — [Develop tests](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)

### Gaps
- Docs grading-methods ranking (code > LLM > human) lives past char 100000 of the docs page; not read.

## 4. Isolation and harnesses

### Takeaway
Every trial starts clean. The eval harness runs the tasks. The agent harness is the thing under test. You evaluate both together.

### Cited Findings
- "Each trial should be “isolated” by starting from a clean environment." — [Demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). Shared state causes correlated failures (paraphrase).
- "An evaluation harness is the infrastructure that runs evals end-to-end." — same page
- "An agent harness (or scaffold) is the system that enables a model to act as an agent" — same page
- Evaluating "an agent" means evaluating harness and model together (paraphrase) — same page

## 5. Reading results with multiple trials

### Takeaway
Pick the metric by what the user needs. For consistency, use pass^k.

### Cited Findings
- pass@k: chance of at least one correct solution in k tries (paraphrase). "pass^k measures the probability that all k trials succeed." — [Demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Suite-structure points from the stats post: resample answers several times and average per question; paired-differences tests when comparing; cluster standard errors for grouped questions; power analysis picks question count — [Statistical approach](https://www.anthropic.com/research/statistical-approach-to-model-evals). That post targets evals of "hundreds or thousands" of questions.

### Inferences
- A 97% per-trial target over 10 trials is a pass^k style claim only if all trials must pass. Review should say which. With 28 tasks the stats post's large-sample assumptions are weak.

### Gaps
- Demystifying evals gave no per-task pass-rate target like 97% in what I read.

## 6. Success criteria

### Takeaway
Criteria must be specific, measurable, achievable, relevant, and usually several at once.

### Cited Findings
- Four traits, [Define success](https://platform.claude.com/docs/en/test-and-evaluate/define-success): "Clearly define what you want to achieve."; "Use quantitative metrics or well-defined qualitative scales."; achievable vs frontier capability; "Align your criteria with your application's purpose and user needs."
- Bad: "Safe outputs". Good: "Less than 0.1% of outputs out of 10,000 trials flagged for toxicity" — same page
- Bad: "The model should classify sentiments well". Good: "F1 score of at least 0.85" — same page
- "Most use cases need multidimensional evaluation along several success criteria." — same page

### Gaps
- Docs page was read only to char 100000 of 151306; later content unread.
