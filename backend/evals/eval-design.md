# Chat agent eval: metrics per behavior and redesign

Date: 2026-11-18. Built with the `write-eval` skill. Security is out of scope here.
It gets its own reusable suite later.

Every source below was opened and read on 2026-11-18, unless marked otherwise.

## Terms

This doc uses the terms from Anthropic's "Demystifying evals for AI agents" [1].

- **Task**: one test with defined inputs and success criteria. In code, one
  `EvalCase` line in a `.jsonl` file.
- **Trial**: one attempt at a task. We run several trials per task, because
  model outputs vary. In code, `--repeat N` gives N trials.
- **Grader**: logic that scores one part of the agent's work. A task can have
  several graders. Each grader holds one or more **assertions** (checks).
- **Transcript**: the full record of one trial: outputs, tool calls and
  intermediate results.
- **Outcome**: the final state at the end of a trial, not what the agent says
  about it.
- **Evaluation harness**: the code that runs tasks, records each trial, grades
  it and adds up the results. Here, `backend/evals/`.
- **Agent harness**: the code that lets the model act as an agent. Here, the
  LangGraph graph, its tools and its guards. An eval scores the model and the
  agent harness together.
- **Suite**: a set of tasks with one shared goal.
- **Behavior**: one thing the agent must do (B1 to B6 below). Each behavior has
  many tasks. This term is ours, not the blog's.

## The behaviors

Your five "stay on track" points, minus the security one, plus behaviors 1 and 2:

| ID | Behavior |
|---|---|
| B1 | Answer about the uploaded file. Search the file first. |
| B2 | Read a link the user pastes. Search the web for current events. |
| B3 | Never search files on an account with no files. |
| B4 | Say "not in your file" instead of making up an answer. |
| B5 | In a long chat, summarize old messages and keep going. |
| B6 | When the loop limit hits, still answer in text. |

## Two layers (from the skill)

- **System layer**: a scripted fake model tries the bad move on purpose. The
  agent harness must stop it. Exact, free, runs on every commit.
- **Model layer**: the real model runs k trials of each task. Measures how often
  it does the right thing.

B3 and B6 have code guards (`tools/base.py` checks `has_documents`,
`agent/nodes.py` drops tools on the last lap). So they get a system-layer test
first. The rest are model behavior and live in the model layer.

## Metrics per behavior

### B1: answer about the file

| Metric | What it measures | Grader | Source |
|---|---|---|---|
| Retrieval hit rate@k, MRR | The right file chunk came back, and how high | Code (keep current) | Ragas docs list context recall / precision as the standard RAG retrieval metrics [5] |
| Faithfulness (groundedness) | Every claim in the answer is backed by the file | Judge, pass/fail | Ragas paper: split answer into claims, check each against the context [4]. FACTS Grounding: an answer passes only if fully grounded [6] |
| Eligibility | The answer actually answers the question | Judge, pass/fail | FACTS Grounding phase 1: off-request answers are disqualified [6] |
| Key-fact correctness | The gold fact (name, number) is in the answer | Code, with number tolerance | Anthropic: use code graders where they can decide [1] |
| File-first order | `retrieve_documents` runs before any web search | Code, on the transcript's tool calls | Ragas tool call accuracy, flexible-order mode [5]. Anthropic: grade outcomes, avoid rigid step order [1] |

Change from today: the judge returns **pass/fail per assertion**, not a 0.0 to
1.0 score. Hamel Husain says to start with binary pass/fail and calls 1 to 5
scales arbitrary [7].

Tone: leave it out for now. Add a tone judge only if reading transcripts shows a
real tone failure. Anthropic lists a tone rubric as an example, not a must [1].

### B2: read a link, search the web

| Metric | What it measures | Grader | Source |
|---|---|---|---|
| Tool call accuracy | Right tool, right argument (the pasted URL) | Code, on the transcript's tool calls | BFCL checks function name and arguments by parsing the call [9]. Ragas tool call accuracy [5] |
| Goal accuracy | The answer uses the fetched page or search result | Judge, pass/fail | Ragas agent goal accuracy is binary [5] |
| Citation precision | Every cited URL is one the agent actually fetched | Code | ALCE defines citation recall and precision [8] |
| Failed tool reported | Page 404 or timeout: the answer says so, no invented content | Code + judge | Skill line "a failed tool is reported, not hidden" |

### B3: no file search on an empty account

| Metric | What it measures | Grader | Source |
|---|---|---|---|
| Guard holds | Fake model calls `retrieve_documents` on an empty account: refused | System layer, code | Skill: every promise with a code guard gets a system-layer test |
| Wrong-call rate | Real model on empty account never asks for file search | Code, on the transcript's tool calls, pass^k | BFCL "relevance detection": no call when no tool fits [9] |
| Benign twin | Same question on an account with files: file search runs | Code | Skill: every "must not" task gets a twin |

### B4: say "not in your file"

| Metric | What it measures | Grader | Source |
|---|---|---|---|
| Abstention recall | Answer is not in the file: agent says so | Judge, pass/fail | Abstention survey (Wen et al.) [10]* |
| Over-abstention rate | Answer IS in the file: agent wrongly says it is not | Judge, pass/fail | Same survey, "over-abstention" [10]* |
| No invented file fact | Agent never presents an unsupported fact as coming from the file | Judge (faithfulness, reused from B1) | Ragas faithfulness [4] |

LongMemEval also tests abstention as one of its five memory skills [12].

*The survey's metric names came from a model summary of section 4.2. Check the
paper before you quote them.

Do not grade this by searching for the words "not in your file". The skill bans
it: the agent can say the words and still invent a fact.

### B5: long chats

| Metric | What it measures | Grader | Source |
|---|---|---|---|
| Summary fires | Thread passes 12k tokens: a summarize event shows up and the reply still finishes | System layer, code | Anthropic context engineering: compaction summarizes near the limit and continues [3] |
| Fact recall after summary | A fact from turn 1 ("my budget is $40k") is still used after the summary | Code, key-fact assertion | LongMemEval "information extraction" [12]. Anthropic: tune compaction for recall first [3] |
| Knowledge update | User changes a fact mid-chat: the latest value wins | Code | LongMemEval "knowledge updates" [12] |
| Multi-turn reliability | Same long chat, k trials, all pass | pass^k | Laban et al.: a 39% average drop in multi-turn, mostly from unreliability [11] |

Cheap trick: seed a long history directly into the thread instead of playing 20
real turns. Only the last 2 or 3 turns hit the real model.

### B6: loop limit

| Metric | What it measures | Grader | Source |
|---|---|---|---|
| Trial ends with text | Fake model asks for tools forever: the agent stops at 8 laps, last lap has no tools, reply is non-empty text | System layer, code | Anthropic "Building effective agents": use stopping conditions like max iterations [2] |
| Answer at the cap | Real model on a question that tempts many searches: still gives a useful answer | Judge, pass/fail | Skill line "the run ends" |

## Where new tasks come from

Now, while the agent is not live:

- Bugs you find while testing it yourself become tasks.
- Every failed CI gate that turns out to be a real agent bug becomes a task.

Later, once real users chat with it, sample their chats and read them. Name
the failure modes you see. Turn each one into a task. This follows Hamel [15]
and Anthropic [1]. The real-chat loop is out of scope until the agent is live.

## Rules for every behavior

- **Never trust one trial.** pass^k = all k trials of a task pass. Anthropic:
  pass^k fits agents where users need consistency [1]. tau-bench uses it too
  [13]. Target: the whole regression suite passes 97% of its trials, and no
  task is broken. The CI gate gives a failed task 2 more trials before it
  blocks. The release check runs 10 trials per task. See the Gates item below
  for the exact bars.
- **Clean is not zero.** With n clean trials, the true failure rate can still be
  about 3/n. Print that next to every clean result.
- **Judges get checked before they gate.** One judge per assertion, pass/fail,
  plus "unknown". Label about 100 transcripts by hand. Report true positive rate
  and true negative rate, not plain agreement [7].
- **Judge model is not the agent model.** MT-Bench found judges favor longer
  answers and their own outputs [14].
- **Two suites.** Regression suite sits near 100% and gates. Capability suite
  starts low and only reports. A task moves over once it passes steadily [1].
- **Save every transcript.** Each trial's transcript (final answer, every tool
  call with its arguments, any error) goes into the results file. People read
  transcripts to see why a task failed. The judge check needs about 100
  hand-labeled transcripts. Anthropic: "read the transcripts" [1].
- **Every trial starts clean.** Each trial runs in its own chat thread, with no
  state from another trial. The shared document shelf is the one exception, on
  purpose (see "What evals cannot prove" in the `evals` skill). Keep trials
  independent [1].

## Redesign: what changes in `backend/evals/`

1. **Group by behavior, not by tool.** Each task gets a behavior tag (`b1` to
   `b6`). Old `tools.jsonl` and `rag.jsonl` tasks move under their behavior. The
   report shows one block per behavior.
2. **New system-layer tests (pytest, free).** A scripted fake model drives the
   real agent harness. Covers B3 guard, B5 summary fires, B6 loop cap. Runs
   every commit.
3. **Judge rewrite.** Split into three binary judges: grounded, eligible,
   abstained. Each one gets a labeled set and its two rates in the repo.
   These three cover every judge metric in the tables above:
   - **grounded**: B1 faithfulness, B2 goal accuracy (the source is the
     fetched page or search result), B4 no invented file fact.
   - **eligible**: B1 eligibility, B6 answer at the cap.
   - **abstained**: B4 abstention recall and over-abstention, B2 failed tool
     reported.
   Code graders need no labels. Only these three judges do.
4. **Recorded web fixtures for B2.** A fake `fetch_web_pages` and `search_web`
   return saved pages. B2 tasks then run in CI with no live internet. Live web
   tasks stay untagged and run locally only, with a real `TAVILY_API_KEY`.
5. **Multi-turn task type for B5.** A task holds a seeded history plus the final
   turns. Reuses the existing evaluation harness.
6. **Gates.** The whole token budget is about $5, so every bar below is sized
   to fit it. Judges gate only after their rates are measured.
   - **System layer: 100%, every commit, free.** Plain pytest, no model.
   - **Target: the whole regression suite passes 97% of its trials, and no
     task is broken.** Not 97% per task. Proving 97% for one task takes about
     100 trials (the 3/n rule above). For 28 tasks that is 2,800 trials, about
     $5.30, the whole budget. A number for the whole suite needs 280.
   - **CI gate, before merge, started by hand.** Each task gets 1 trial, graded
     by code graders only. Hamel favors assertions over judges in CI [15]. A
     task that fails gets 2 more trials. Block only if one of those fails too.
     About $0.05 per run. A healthy agent, missing 3 answers in 100, gets
     blocked by bad luck about 1 run in 20. A task that breaks fully is always
     caught. A task that fails half its trials is caught about 4 runs in 10.
     Run the same gate locally before you start CI.
   - **Release check, before the agent goes out, and after any model or
     prompt change.** Each task gets 10 trials, 280 in all, with judges on.
     Ship when the suite fails 8 or fewer of 280 (that is 97%) and no task
     fails 3 or more of its 10. About $0.53. An agent that misses 2 answers in
     100 passes about 9 times in 10. One that misses 5 in 100 almost never
     passes. An agent right at 97% passes about half the time, because it sits
     on the line. Never rerun the check until it passes.
   - **New tasks:** run `--repeat 3` locally while you write them (see the
     `evals` skill).
   - Errored trials (rate limit, provider outage) count for nothing. They are
     neither passes nor failures. Rerun them.
   - **Budget:** 10 CI runs plus one release check is about $1. Costs use the
     measured 4,200 input and 300 output tokens per trial, at Together's
     DeepSeek V4.1 Flash price ($0.30 / $1.20 per million) [16]. Together no
     longer lists the configured V4 Flash 0731 for pay-per-use, so recheck the
     price when the model changes.
7. **Per-task counts in the results file.** Today `report.py` only lists tasks
   that failed every trial (`regressions`) or some trials (`flaky`), as text.
   Add one record per task with its passed, failed and errored trial counts,
   plus suite totals. Both gates read these counts.
   Each run also records the model, a prompt version, a dataset version and
   the estimated cost. Two runs can then be compared fairly [17].
8. **Two run modes in the evals command.** `gate`: 1 trial per task, no judge
   calls, a failed task gets 2 more trials, exit 1 if any of those fail.
   `release`: 10 trials per task, judges on, exit 1 unless the suite fails 8
   or fewer and no task fails 3 or more. Both print one line per task.
9. **Manual-only CI.** Change `.github/workflows/evals.yml`: remove the
   `pull_request` trigger, keep `workflow_dispatch`, and add a `mode` input
   (`gate` or `release`, default `gate`). Keep the results file upload.
10. **Hand labels live in Langfuse, for development only.** Decided
    2026-11-25. The app already traces to Langfuse Cloud, so labeling uses the
    free Hobby plan: 50k units a month, 2 users, 1 annotation queue, 30 days
    of data [18]. Production tracing and labeling are decided later.
    - **One trace per trial.** Today the evaluation harness opens no span per
      trial, and the judge's span lands in a trace of its own. Open one root
      span per trial, run the judge inside it, and save the trace id in the
      results file.
    - **The judge writes scores.** Each verdict (grounded, eligible,
      abstained) goes onto the trial's trace as a boolean score, through the
      SDK's `create_score` [19]. The `Tracer` port gets a score method.
      `NullTracer` ignores it.
    - **The labeler can see the source.** Tool spans record arguments but not
      the result text. Grounded cannot be labeled without the file chunk or
      page the agent read, so record that text on the tool span, capped.
    - **Labeling.** Create one human score config per judge and one annotation
      queue. Add about 100 trial traces with a mix of passes and failures.
      Label each one without looking at the judge's score first. Finish
      within 30 days of the run, before the data leaves the plan's window.
    - **Agreement.** Langfuse Score Analytics compares a human score with an
      automated one on the same traces, with a confusion matrix and Cohen's
      kappa [20]. It is in beta, and its plan is not stated. If the Hobby plan
      lacks it, a small script pulls both scores through the API and reports
      the true positive and true negative rates.

Moved out: `injection.jsonl`, `benign.jsonl`, `redteam.jsonl`, the planted
attack memo, and the guardrail golden-file test go to the separate security
suite, kept local in `security-evals/` (gitignored).

## Sources

1. Anthropic, "Demystifying evals for AI agents", 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
2. Anthropic, "Building effective agents", 2024. https://www.anthropic.com/engineering/building-effective-agents
3. Anthropic, "Effective context engineering for AI agents", 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
4. Es et al., "Ragas: Automated Evaluation of Retrieval Augmented Generation", 2023. https://arxiv.org/abs/2309.15217
5. Ragas docs, "List of available metrics". https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
6. Google DeepMind, "FACTS Grounding", 2024. https://deepmind.google/discover/blog/facts-grounding-a-new-benchmark-for-evaluating-the-factuality-of-large-language-models/
7. Hamel Husain, "Using LLM-as-a-Judge For Evaluation: A Complete Guide", 2024. https://hamel.dev/blog/posts/llm-judge/
8. Gao et al., "Enabling Large Language Models to Generate Text with Citations" (ALCE), 2023. https://arxiv.org/abs/2305.14627
9. UC Berkeley, "Berkeley Function-Calling Leaderboard", 2024. https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
10. Wen et al., "Know Your Limits: A Survey of Abstention in LLMs", 2024. https://arxiv.org/abs/2407.18418
11. Laban et al., "LLMs Get Lost In Multi-Turn Conversation", 2025. https://arxiv.org/abs/2505.06120
12. Wu et al., "LongMemEval", 2024. https://arxiv.org/abs/2410.10813
13. Yao et al., "tau-bench", 2024. https://arxiv.org/abs/2406.12045
14. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", 2023. https://arxiv.org/abs/2306.05685
15. Hamel Husain, "LLM Evals FAQ". https://hamel.dev/blog/posts/evals-faq/
16. Together AI pricing, read 2026-11-24. https://www.together.ai/pricing
17. GitHub, "How to evaluate LLMs before production", 2026. https://github.blog/2026-08-25-how-to-evaluate-llms-before-production/
18. Langfuse pricing, read 2026-11-25. https://langfuse.com/pricing
19. Langfuse, "Custom scores", read 2026-11-25. https://langfuse.com/docs/evaluation/evaluation-methods/custom-scores
20. Langfuse, "Score Analytics", read 2026-11-25. https://langfuse.com/docs/evaluation/evaluation-methods/score-analytics

Not used: two OpenAI pages (hallucination scoring, SimpleQA) blocked the fetch.
Nothing here leans on them.
