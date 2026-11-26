# Statistical and reliability properties of a trustworthy agent eval suite

Note on method: pages were read through WebFetch, which returns a model-written summary of the page, not raw text. Formulas and numbers below are as reported by that read. The two arXiv PDFs would not parse, so the arXiv HTML versions were used. Items marked (MY COMPUTATION) are my own math, not source claims. All dates are as of Nov 2026.

## 1. What Evan Miller's paper recommends (standard errors, clustering, pairing, resampling, power)

### Takeaway
The paper gives five recommendations: report standard errors, cluster them, reduce within-question variance by resampling, compare models with paired differences, and run power analysis before the eval. It treats questions as a sample from a super-population, so the number of questions is what sets the error bar.

### Cited Findings
- The paper's framing: eval questions are samples from an unseen super-population, with formulas for analysis, model comparison and experiment planning — [arXiv abstract 2411.00640](https://arxiv.org/abs/2411.00640)
- Rec 1: report the SE of each mean (CLT) beside the score, plus the number of questions. SE = sqrt( sum (s_i - mean)^2 / (n(n-1)) ). For 0/1 scores SE = sqrt(p(1-p)/n). 95% CI = mean +/- 1.96 SE. Bootstrapping is unnecessary unless the sampling scheme is complicated — [arXiv HTML 2411.00640](https://arxiv.org/html/2411.00640)
- Rec 2: when questions come in related groups, use clustered SEs and report the cluster count. In the paper's examples clustered SE was up to about 3x the naive SE. If scores in a cluster are perfectly correlated, each cluster counts as one observation — [arXiv HTML 2411.00640](https://arxiv.org/html/2411.00640); the Anthropic page says clustered errors can exceed 3x naive ones — [Anthropic](https://www.anthropic.com/research/statistical-approach-to-model-evals)
- Rec 3: reduce variance by resampling each question K times and using question-level averages. Variance = (Var(x) + E[sigma_i^2]) / n, where Var(x) is between-question (cannot be reduced by resampling) and sigma_i^2 is within-question (can). Pooling all K*N answers into one SE is wrong because repeats of one question are not independent. The paper says not to lower the temperature to cut variance, since that can add bias — [arXiv HTML 2411.00640](https://arxiv.org/html/2411.00640)
- Resampling has a ceiling: in the paper's uniform binary example the maximum variance reduction is 2/3 — [arXiv HTML 2411.00640](https://arxiv.org/html/2411.00640)
- Rec 4: compare models with paired differences on the same questions. Paired SE = sqrt( sum (d_i - dbar)^2 / (n(n-1)) ) with d_i = s_A,i - s_B,i. Equivalent form: sqrt(SE_A^2 + SE_B^2 - 2 SE_A SE_B Corr). Report mean difference, its SE and CI, and the score correlation. Unpaired is sqrt(SE_A^2 + SE_B^2) — [arXiv HTML 2411.00640](https://arxiv.org/html/2411.00640)
- Anthropic's page says frontier-model score correlations run 0.3 to 0.7, which makes pairing a real variance reduction — [Anthropic](https://www.anthropic.com/research/statistical-approach-to-model-evals)
- Rec 5: do power analysis first. n = (z_{alpha/2} + z_beta)^2 (omega^2 + sigma_A^2/K_A + sigma_B^2/K_B) / delta^2, where omega^2 = Var(x_A)+Var(x_B)-2Cov. Example in the paper: omega^2 = 1/9, delta = 3 points, alpha 0.05, beta 0.20 gives n about 969 questions. The paper suggests new evals have at least about 1,000 questions — [arXiv HTML 2411.00640](https://arxiv.org/html/2411.00640)

### Inferences
- The user's per-task trials are the "K" (resamples). Per the paper, extra trials only shrink the within-task term. The between-task term stays, so 28 tasks is the real limit on suite-level precision.
- The paper's design is for many independent questions. A 28-task suite is far below its ~1,000 suggestion, so only large regressions are detectable at suite level (see section 6).

### Gaps
- The paper's Appendix C cluster-adjusted power formulas were not read.
- The full text was read via an HTML summary, not line by line.

## 2. tau-bench: pass^k definition and what it showed

### Takeaway
pass^k is the chance that all k trials of a task succeed, averaged over tasks. tau-bench showed agents that look fine on one trial are much worse when consistency is required.

### Cited Findings
- Definition: pass^k = E_task[ C(c,k) / C(n,k) ], with n trials and c successes per task. For comparison pass@k = 1 - E_task[ C(n-c,k)/C(n,k) ] — [arXiv HTML 2406.12045](https://arxiv.org/html/2406.12045)
- Main results use at least 3 trials per task. The paper ran over 40 gpt-4-turbo trials per retail task only for task-difficulty analysis — [arXiv HTML 2406.12045](https://arxiv.org/html/2406.12045)
- gpt-4o retail pass^1 is 61.2 and airline pass^1 is 35.2. The paper says retail pass^8 drops below 25% — [arXiv HTML 2406.12045](https://arxiv.org/html/2406.12045); abstract: agents "quite inconsistent," pass^8 under 25% in retail — [arXiv abstract 2406.12045](https://arxiv.org/abs/2406.12045)
- Anthropic's definitions: pass@k is "at least one correct solution in k attempts" and rises with k. pass^k is "all k trials succeed" and falls with k. pass@k suits cases where one success matters, pass^k suits cases needing consistency — [Anthropic engineering](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- tau2-bench: search results say it adds a setting where the simulated user also uses tools. I found no tau2-bench paper in results; ReliabilityBench (arXiv 2601.06112) builds on pass^k — [search result list](https://arxiv.org/html/2601.06112v1) (page not opened)

### Inferences
- (MY COMPUTATION) If a task truly passes 97% per trial, p^k gives pass^3 = 0.913, pass^10 = 0.737, pass^100 = 0.048. So pass^k for large k is almost always low, even at the 97% target. The k must match the user's real need.
- (MY COMPUTATION) With n=10 trials and 9 passes, the estimator for k=3 is C(9,3)/C(10,3) = 0.70. With n=3 trials, pass^k can only be estimated for k up to 3.
- The estimator needs k <= n. A "last 100 trials" release check supports k up to 100, but the numbers at large k are driven by the worst tasks.

### Gaps
- The full Figure 4 values (pass^k for k=2..8) were not available as text.
- Airline pass^k at k>1 not found.
- tau2-bench primary paper not opened.

## 3. Rule of three and trials needed to support a 97% claim

### Takeaway
With zero failures in n trials, the 95% upper bound on the failure rate is about 3/n. To back "failure rate at most 3%" with 95% confidence you need about 100 straight passes, and any failure raises the bar.

### Cited Findings
- Rule: if an event never occurs in n subjects, 0 to 3/n is a 95% CI for its rate. Derivation: (1-p)^n = 0.05, n ln(1-p) = ln 0.05 ≈ -2.996 — [Wikipedia: Rule of three](https://en.wikipedia.org/wiki/Rule_of_three_(statistics))
- General form: upper bound = -ln(alpha)/n. Numerators: 95% = 3, 97% = 3.51, 99% = 4.61, 99.5% = 5.3. Good approximation when n > 30 — [Wikipedia: Rule of three](https://en.wikipedia.org/wiki/Rule_of_three_(statistics))
- Original source: Hanley & Lippman-Hand 1983, JAMA 249(13):1743 — cited via [Wikipedia](https://en.wikipedia.org/wiki/Rule_of_three_(statistics)) and a search listing; **I did not open the 1983 paper itself.** Another search source says the approximation is close even at n=20 (15% vs 14%) — [search results, BMJ 1995 reminder](https://doi.org/10.1136/bmj.311.7005.619) (not opened)

### Inferences (all MY COMPUTATION)
- Exact zero-failure 95% bound is 1 - 0.05^(1/n): n=3 gives 0.63, n=10 gives 0.26, n=30 gives 0.095, n=100 gives 0.0295.
- Trials needed with zero failures so the upper failure bound is at most 3%: ln(0.05)/ln(0.97) = 99 trials (95% conf); 76 trials for 90% conf.
- So 3 local trials or 10 CI trials say almost nothing about a 97% target: 3/3 passes is consistent with a true pass rate as low as about 37% (Clopper-Pearson lower bound 0.368). 10/10 gives lower bound 0.741.
- With 100 trials: 100/100 gives one-sided 95% lower bound 0.970. 99/100 gives 0.953, 98/100 gives 0.938, 97/100 gives 0.924. So observing exactly 97/100 does not demonstrate a 97% rate. It is just the point estimate.
- A task that is truly 97% passes 100/100 only 4.8% of the time. It scores 97 or more of 100 with probability 0.65. A task truly at 95% still scores 97 or more of 100 with probability 0.26. A single-task 97% gate on 100 trials has weak power to tell 95% from 97%.
- P(3/3 pass) is 0.73 even if true rate is 90%. P(all 28 tasks pass 3/3) is only 0.077 if each is truly 97%, so "all green" on 3 trials is unlikely even for a healthy suite.
- Sample size to tell 97% from 90% on one task (two-sample, alpha .05, power .8): about 191 per version. 97% vs 94%: about 746 per version.
- Trials must be independent: Anthropic warns shared state between trials (leftover files) causes correlated failures, so use a clean environment per trial — [Anthropic engineering](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). "Last 100 trials" mixing different code versions would not be i.i.d. draws from one pass rate (my inference).

### Gaps
- No source found that states a required trial count for a "97% per task" claim. The numbers above are my math.
- Source on Wilson/Clopper-Pearson for agent evals not found.

## 4. Run-to-run variance, non-determinism, reporting variance

### Takeaway
Even temperature 0 does not give identical outputs on shared servers, so one run is one draw. Report variance and do not rely on one trial.

### Cited Findings
- Thinking Machines: Qwen3-235B-A22B-Instruct-2507 at temperature 0, 1,000 completions of 1,000 tokens each gave 80 unique completions. The most common appeared 78 times. They diverged at token 103. Cause: kernels not batch-invariant, so results depend on server load. With batch-invariant kernels all 1,000 matched, at a speed cost (26s vs 55s, improved to 42s on Qwen3-8B) — [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- Anthropic: a task that passed on one eval run might fail on the next, so run multiple trials per task. No trial number is given — [Anthropic engineering](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Miller: do not lower temperature to cut variance (bias risk); resample instead — [arXiv HTML 2411.00640](https://arxiv.org/html/2411.00640)
- "Quantifying Variance in Evaluation Benchmarks": defines seed-variance and monotonicity metrics, says variance determines if a difference is meaningful and is rarely quantified, urges factoring in variance when comparing models. Reports no specific seed-variance numbers on the abstract page — [arXiv 2406.10229](https://arxiv.org/abs/2406.10229)

### Inferences
- Agent trials add more variance than a single LLM call, since many steps and tool results compound (my inference, no source opened that quantifies it).
- Closed API models are not under the user's control, so batch-invariance fixes are unavailable. Treat non-determinism as permanent and measure it.

### Gaps
- Magnitudes from 2406.10229 not read (full paper not opened).
- No source found quantifying agent-level (multi-step) run-to-run variance beyond tau-bench's pass^k decay.

## 5. Flaky tasks and LLM judge noise

### Takeaway
A task that always fails usually has a broken task or grader. Judge verdicts also vary run to run, so judge noise adds to the error bar and needs its own check.

### Cited Findings
- Anthropic: 0% pass over many trials usually means a broken task or grader, not an incapable agent. Write a reference solution to prove a task is solvable. Start each trial from a clean environment. The article does not use the word "flaky" — [Anthropic engineering](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Anthropic on LLM judges: calibrate closely to human experts, give an escape such as "Unknown", grade each dimension with a separate judge — [Anthropic engineering](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Within-judge variance preprint (arXiv 2609.33044, Sep 2026, v2, single preprint): at temperature 0 and a pinned model, identical inputs still gave different verdicts. About 5% of items flipped on average, about 40% of close-call items flipped, per-judge flip rates 0.13% to nearly 10%. Aggregate rank noise was mostly from prompt sampling. It recommends several judge re-runs, intervals over re-runs, and at least two judges from different families — [arXiv 2609.33044](https://arxiv.org/abs/2609.33044)
- Standard CIs miss judge/prompt/model-choice variation. A Terminal-Bench 2.0 reanalysis reported: rerun-only intervals separated 4 neighboring model pairs, wider intervals overlapped on all 16 pairs — [arXiv 2604.11581 via search summary](https://arxiv.org/html/2604.11581v3) (page not opened; secondhand)
- Judge sensitivity/specificity errors bias naive scores. A paper proposes a bias-corrected estimator with CIs that count both test and human-calibration-set uncertainty — [arXiv 2511.21140](https://arxiv.org/abs/2511.21140)
- Prompt rewording can move a judge's verdict more than decoding noise (JudgeSense) — [search summary of arXiv 2604.23478](https://arxiv.org/pdf/2604.23478) (not opened)

### Inferences
- For a binary pass/fail judge, judge noise acts like extra within-task variance. If the judge flips 5% of items, a true 97% task can read much lower. Judge flip rate should be measured on the user's own judged tasks (same input, judge run 5+ times).
- Code-graded tasks (exact tool-call checks) avoid this noise. Report them separately from judged tasks.

### Gaps
- None of the judge-noise papers were checked against a chat-agent pass/fail setting. Most are about pairwise leaderboards (Arena-Hard etc.).
- The 2609.33044 arXiv ID is a September 2026 ID; I confirmed the page loads and dates match, but it is a recent preprint and not peer reviewed as far as I could tell.

## 6. Suite size (number of tasks) vs. size of regression detectable

### Takeaway
No source gives a rule for 28 tasks. Anthropic says 20 to 50 tasks is a fine start when effects are large. Miller's math shows small effects need on the order of 1,000 questions.

### Cited Findings
- Anthropic: start with about 20 to 50 simple tasks drawn from real failures. Early changes often have large effects, so small samples can detect them. Mature agents may need larger, harder suites — [Anthropic engineering](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Miller: detect a 3-point difference needs about 969 questions in his example. New evals should have at least about 1,000 questions — [arXiv HTML 2411.00640](https://arxiv.org/html/2411.00640)
- Minimum detectable effect for fixed n: delta = (z_{alpha/2} + z_beta) sqrt( (omega^2 + sigma_A^2/K_A + sigma_B^2/K_B) / n ) — [arXiv HTML 2411.00640](https://arxiv.org/html/2411.00640)

### Inferences (MY COMPUTATION, paired, alpha .05, power .8, z sum 2.80)
- Suite-level MDE = 2.80 * sd_d / sqrt(28), where sd_d is the SD of per-task differences between versions. If sd_d = 0.05, MDE is 2.6 points. If 0.10, 5.3 points. If 0.20, 10.6 points.
- Within-task noise alone (p=.97, independent trials): suite-mean SE is 1.9 points with 3 trials per task and 1.0 point with 10 trials. This ignores between-task spread and clustering, so it understates real uncertainty.
- Practical read: 28 tasks can catch a few tasks collapsing (large per-task drops), not a uniform 1 to 2 point slide. Per-task 97% gates on 100 trials have the weak-power problem shown in section 3.
- Pair by task: compare version A and B on the same tasks, with per-task differences, as Miller recommends. Tasks that are 100% in both versions give d=0 and add no noise.

### Gaps
- No source gives suite-size rules for agent evals with binary per-task pass rates. The sd_d values above are assumed, not measured. The user's own run data should supply sd_d.
- Cluster definition for the user's tasks (e.g., tasks from the same fixture or feature) not decided here.
