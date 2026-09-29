# TokenSmith Query Decomposition runbook

Last reviewed: 2026-09-28

This is a static research snapshot: nothing is deployed and there are no
services to operate. The page covers reproducing the evaluation.

## Prerequisites

- A TokenSmith backend checkout (separate repository, not included here). The
  eval scripts import `src.config.RAGConfig` from that tree, so copy `planner/`
  and `eval/` into the backend checkout and run the commands from its root.
- The backend running and reachable at `http://localhost:8000`, with the
  planner package installed.
- Python 3.10+. HTTP uses the standard library. `--use-judge` needs
  `llama-cpp-python` and a local judge model.

## Reproduce

```bash
python -m eval.run_multi_pass_eval --use-judge --passes 3 --base-url http://localhost:8000
python -m eval.run_multi_pass_eval --use-judge --from-dir eval/multi_run     # re-aggregate saved passes; no running server needed
python -m eval.run_planner_eval --base-url http://localhost:8000 --limit 5   # quick single-pass smoke run
```

Outputs land in `eval/multi_run/` (per-pass files, `aggregate.json`,
`judge_influence.json`).

## Maintenance notes

- The repository is public. Keep it free of backend credentials and course-only
  material you don't want published.
- If you revisit the work, add a `requirements.txt` and a small unit test for
  `classifier.py` routing so the snapshot is runnable without the backend.
