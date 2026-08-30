# TokenSmith Query Decomposition

An optional planning layer for multi-hop retrieval: classify a question, decompose complex requests into dependency-aware sub-queries, retrieve and merge evidence, then synthesize the final answer through TokenSmith's native generation path.

The repository contains the planner implementation snapshot and a reproducible baseline-vs-planner evaluation harness.

## System design

```text
question
   │
   ▼
classify ── simple ───────────────► baseline retrieval + generation
   │
 complex
   ▼
decompose (2–5 sub-queries)
   │
   ▼
retrieve in dependency waves
   │
   ▼
deduplicate + rerank evidence
   │
   ▼
TokenSmith final-answer generator
```

The planner is opt-in through `use_planner: true` on TokenSmith's `POST /api/chat` and `POST /api/chat/stream` endpoints.

## Evaluation snapshot

The recommended aggregate is three full passes over 19 questions using the same local judge model.

| Metric | Baseline | Planner | Read |
| --- | ---: | ---: | --- |
| Mean judge score (1–5) | 3.93 | 4.04 | Small improvement |
| Keyword recall | 0.579 | 0.580 | Effectively flat |
| Mean latency | 14.7 s | 23.3 s | ~58% slower |
| Classifier accuracy | — | 68.4% | Stronger on simple (87.5%) than complex (54.5%) |

The result is intentionally reported as a tradeoff: planning improved the multi-pass judge mean, but added substantial latency and did not materially change keyword recall. That argues for selective routing and a better complex-query classifier rather than enabling decomposition for every request.

See [`eval/multi_run/aggregate.json`](eval/multi_run/aggregate.json) for the raw aggregate and [`eval/PLANNER_IMPLEMENTATION_REPORT.md`](eval/PLANNER_IMPLEMENTATION_REPORT.md) for methodology and limitations. The single-run `results.json` is retained as an iteration artifact, not the headline result.

## Repository map

- `planner/classifier.py` — heuristic-first simple/complex routing with an LLM fallback
- `planner/decomposer.py` — dependency-aware sub-query generation
- `planner/pipeline.py` — orchestration and execution waves
- `planner/synthesizer.py` — evidence merge, deduplication, reranking, and answer handoff
- `eval/run_planner_eval.py` — one-pass baseline comparison
- `eval/run_multi_pass_eval.py` — repeated runs, aggregate statistics, and judge-influence analysis
- `eval/benchmark_questions.json` — 19-question evaluation set

## Run the evaluation

Start a local TokenSmith backend, then run:

```bash
python -m eval.run_multi_pass_eval --use-judge --passes 3
```

To re-aggregate completed pass files without calling the backend again:

```bash
python -m eval.run_multi_pass_eval --use-judge --from-dir eval/multi_run
```

## Limitations

- The benchmark is small, and its keyword-recall metric is based on string matching.
- The judge is a small local model, so absolute scores should be interpreted cautiously.
- Latency includes request overhead and varies with local inference conditions.
- The current classifier is materially weaker on complex questions, which is the main routing bottleneck.
