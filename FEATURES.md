# TokenSmith Query Decomposition — Feature Catalog

Last reviewed: 2026-09-28

README: [`README.md`](README.md). Runbook: [`RUNBOOK.md`](RUNBOOK.md). Methodology:
[`eval/PLANNER_IMPLEMENTATION_REPORT.md`](eval/PLANNER_IMPLEMENTATION_REPORT.md).
File guide: [`IMPLEMENTATION_FILES.md`](IMPLEMENTATION_FILES.md).
Ecosystem: [`../ECOSYSTEM.md`](../ECOSYSTEM.md).

## Purpose

A public, grader-facing snapshot of an opt-in query-planning layer for the
TokenSmith RAG backend, plus a reproducible baseline-vs-planner evaluation. The
TokenSmith backend itself is **not** in this repository, so the planner can't
run on its own here.

## Planner (`planner/`)

| Module | Capability |
| --- | --- |
| `classifier.py` | Heuristic-first simple/complex routing with an LLM fallback (`QueryClassifier`, `QueryType`). |
| `decomposer.py` | Splits complex questions into 2–5 dependency-aware sub-queries (`QueryDecomposer`, `SubQuery`). |
| `pipeline.py` | Runs retrieval in dependency waves and hands off to TokenSmith's generator (`PlannerPipeline`). |
| `synthesizer.py` | Merges, deduplicates (`difflib`), and reranks evidence chunks (`Synthesizer`, `RankedChunk`). |

Upstream integration point: `use_planner: true` on TokenSmith's `POST /api/chat`
and `POST /api/chat/stream`.

## Evaluation (`eval/`)

| Artifact | Role |
| --- | --- |
| `run_planner_eval.py` | One pass, baseline vs planner, optional local llama-cpp judge. |
| `run_multi_pass_eval.py` | Repeated passes, aggregation, judge-influence analysis; can re-aggregate saved passes offline. |
| `benchmark_questions.json` | 19-question set. |
| `multi_run/aggregate.json` | Headline aggregate (3 passes). |
| `multi_run/judge_influence.json` | Judge sensitivity analysis. |
| `../results.json` | Single-run iteration artifact; not the headline result. |

## Headline result (3 passes × 19 questions)

| Metric | Baseline | Planner |
| --- | ---: | ---: |
| Mean judge score (1–5) | 3.93 | 4.04 |
| Keyword recall | 0.579 | 0.580 |
| Mean latency | 14.7 s | 23.3 s (about 58% slower) |
| Classifier accuracy | — | 68.4% overall; 87.5% simple, 54.5% complex |

The takeaway is selective routing, not blanket decomposition. The complex-query
classifier is the bottleneck.

## Status and gaps

- Public GitHub repo (`tvermani13/tokensmith-query-decomp`), linked from the portfolio site.
- No `requirements` file or tests. The eval scripts import `src.config` from
  the external backend, so they only run inside a backend checkout. Judging
  also needs `llama-cpp-python` plus a local GGUF model.
- Small textbook-domain benchmark with string-match recall. Transfer to SEC or
  finance questions (for example Deep Research Engine) is unmeasured.
