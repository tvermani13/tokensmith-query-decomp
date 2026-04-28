# TokenSmith Query Planner — Implementation + Evaluation Report

## What was implemented (high level)

A **query planner** layer was added to TokenSmith to improve multi-part / multi-hop questions by:

- **Classifying** queries as *simple* vs *complex* (heuristics first, LLM JSON fallback in uncertain cases).
- **Decomposing** complex questions into 2–5 sub-queries with a small dependency graph (executed in waves).
- **Retrieving** evidence per sub-query using TokenSmith’s existing retriever ensemble + RRF rank fusion.
- **Merging** retrieved chunks, deduplicating, re-ranking against the original question, and selecting a final chunk set.
- **Answering** with TokenSmith’s native generator (`src.generator.answer`) to match the baseline answer style (important for “LLM-as-judge” quality).

The planner is **optional** in the API via `use_planner: true` on `POST /api/chat` and `POST /api/chat/stream`.

## Key files added

### Planner package
- `src/planner/classifier.py`
- `src/planner/decomposer.py`
- `src/planner/synthesizer.py` (dedup + rerank + final answer path)
- `src/planner/pipeline.py`
- `src/planner/__init__.py`

### Evaluation
- `eval/benchmark_questions.json` (19-question set)
- `eval/run_planner_eval.py` (single pass baseline vs planner + optional judge)
- `eval/run_multi_pass_eval.py` (N independent full passes + aggregation + “mean judge movers”)
- `eval/PLANNER_IMPLEMENTATION_REPORT.md` (this document)

## Key files changed

### API integration
- `src/api_server.py`
  - builds `PlannerPipeline` on startup
  - routes `/api/chat` and `/api/chat/stream` when `use_planner` is set
  - uses larger LLM `n_ctx` for planner *helper* calls (classify/decompose/intermediate) to avoid context overflow
  - provides `final_answer_fn` to generate the final user-facing answer using `answer(...)` (TokenSmith style)

## How to run evaluation

Prereq: start the server:

```bash
conda activate tokensmith
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

### Single pass (writes `eval/results.json`)

```bash
python -m eval.run_planner_eval --use-judge
```

`eval/results.json` also includes `judge_mean_shift_if_removed`, a **leave-one-out** table that shows which questions most change the *mean* judge score if you remove that question (computed separately for baseline vs planner), plus the script prints the **top 5** movers.

### 3 full passes (recommended for reporting)

```bash
python -m eval.run_multi_pass_eval --use-judge --passes 3
```

If you already have `pass_*.json` files (e.g. a long run finished but aggregation crashed), you can **re-aggregate without calling the API**:

```bash
python -m eval.run_multi_pass_eval --use-judge --from-dir eval/multi_run
```

This writes to `eval/multi_run/`:
- `pass_01.json` … `pass_03.json` (one full run each)
- `aggregate.json` (mean/stdev/min/max of summary metrics)
- `judge_influence.json` (see below)

## “Which questions move the mean judge score?” (leave-one-out)

`eval/run_multi_pass_eval.py` identifies questions that *most influence* the **mean** judge score by computing, within each pass:

\[
\Delta_q = \mathrm{mean\_judge\_without\_q} - \mathrm{mean\_judge\_with\_all}
\]

and then taking **\(| \Delta_q |\)** and averaging it across the 3 passes (so it’s not a one-off fluke).

Interpretation:
- A large \(| \Delta_q |\) means that question is a **lever** on the mean: removing it would noticeably change the aggregate score.
- It does *not* mean the question is “good” or “bad” by itself—just that it is influential.

## Performance snapshot (from your most recent single-pass run)

This is a representative “good” outcome after the final-answer integration:

- **Keyword recall (all)**: planner **0.586** vs baseline **0.578** (planner slightly higher)
- **Keyword recall (complex)**: planner **0.530** vs baseline **0.532** (effectively tied)
- **LLM-judge (1–5)**: planner **4.00** vs baseline **3.89** (planner higher)
- **Mean latency (ms)**: planner **~25,329** vs baseline **~14,556** (planner ~1.7× slower on average; expected for multi-step planning)

(Exact numbers are printed by `run_planner_eval` / stored in `eval/results.json`.)

## Classifier reporting (accuracy + coverage)

For each planner answer, `/api/chat` returns a `planner` object containing:
- `classification`: `simple` | `complex`
- `confidence`: heuristic/LLM confidence scalar
- `used_llm`: whether the LLM JSON fallback was used
- `reason`: short explanation string

`eval/run_multi_pass_eval.py` stores these per-pass and aggregates:
- **accuracy** over benchmark labels (`simple`/`complex`)
- **coverage** (how many questions emitted classifier metadata)
- **used_llm_rate** (fraction routed through LLM fallback)
- simple vs complex accuracy

## Notes / known limitations

- The benchmark’s **keyword recall** is a weak metric (string contains checks). A question can be answered well while missing exact substrings; judge score can disagree with keyword recall.
- Latency is measured **client-side** in the eval scripts, so it includes request overhead and is not a pure “model-only” time.
- **Judge limitation**: the judge is a small local model (e.g. Qwen2.5-3B Instruct, 8-bit). This improves reproducibility/cost but can be noisier than larger judges; interpret absolute scores cautiously and emphasize relative comparisons + multi-pass stability.

## Qualitative example (include at least one)

In addition to aggregate metrics, include one short “before vs after” example using a single question id from `eval/multi_run/pass_01.json`:
- state the question id + text
- show baseline vs planner judged score
- 2–4 sentences on *what specifically improved* (coverage of key concepts, fewer errors, better structure)

## Bugfix note: judge “mean” computation

`eval/run_planner_eval.py` was updated to compute average judge using **all questions**, not a filtered “>0” subset (a rare edge case, but the correct definition for a mean over N questions).
