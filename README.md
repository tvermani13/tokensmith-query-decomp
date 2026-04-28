## TokenSmith Query Decomposition (CS 6423)

This is the grader-facing repository for my project:
**Query Decomposition and Planning for Multi-Hop Retrieval in TokenSmith**.

The baseline TokenSmith flow is single-pass retrieval + generation.  
My planner layer adds:

- **Query classification** (`simple` vs `complex`)
- **Query decomposition** for complex questions (2–5 sub-queries, dependency-aware)
- **Evidence synthesis** (merge, dedupe, rerank, final answer generation)

## What Is Included Here

### Implementation snapshot

- `planner/` — copied planner implementation files from backend work
- `eval/` — copied evaluation scripts + benchmark + key aggregate outputs
- `IMPLEMENTATION_FILES.md` — exact file map for graders

### Course/report artifacts

- `benchmark/questions.json` — learning episode question set used in report appendix work
- `reports/appendix_learning_episode.md` — appendix source
- `results.json` — run output captured during project iteration

## Quick Navigation for Graders

1. Start with `IMPLEMENTATION_FILES.md`.
2. Inspect planner logic in:
   - `planner/classifier.py`
   - `planner/decomposer.py`
   - `planner/synthesizer.py`
   - `planner/pipeline.py`
3. Inspect eval harness in:
   - `eval/run_planner_eval.py`
   - `eval/run_multi_pass_eval.py`
   - `eval/benchmark_questions.json`
4. Inspect reported aggregate outcomes in:
   - `eval/multi_run/aggregate.json`
   - `eval/multi_run/judge_influence.json`
   - `eval/PLANNER_IMPLEMENTATION_REPORT.md`

## Repro Notes

The scripts in `eval/` are designed to run against a local TokenSmith backend API (`/api/chat`) with the TokenSmith conda environment and local GGUF models configured.

Typical flow (from backend repo):

```bash
conda activate tokensmith
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
python -m eval.run_multi_pass_eval --use-judge --passes 3
```

This repo intentionally stays focused on the planner/eval artifacts rather than mirroring the entire backend repository.

