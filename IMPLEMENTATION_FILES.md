# Implementation Files (Grader Guide)

This repository is the grader-facing companion for my TokenSmith query decomposition/planner work.
It includes the key implementation and evaluation artifacts copied from the backend work tree.

## Copied Planner Implementation

- `planner/classifier.py`
- `planner/decomposer.py`
- `planner/synthesizer.py`
- `planner/pipeline.py`
- `planner/__init__.py`

These correspond to the planner package that performs:
- simple/complex classification
- complex-query decomposition into sub-queries
- chunk merging, de-duplication, re-ranking, and final synthesis

## Copied Evaluation Harness

- `eval/run_planner_eval.py`
- `eval/run_multi_pass_eval.py`
- `eval/benchmark_questions.json`

These scripts run baseline vs planner evaluation and aggregate metrics across multiple passes.

## Copied Evaluation Outputs / Report Artifacts

- `eval/PLANNER_IMPLEMENTATION_REPORT.md`
- `eval/multi_run/aggregate.json`
- `eval/multi_run/judge_influence.json`

These are the main artifacts cited in the report tables/discussion.

## Existing Course Artifacts In This Repo

- `benchmark/questions.json`
- `reports/appendix_learning_episode.md`
- `results.json`

## Notes

- This repo is intentionally a focused snapshot for grading/reproducibility, not a full fork of the entire `TokenSmith` backend repository.
- Paths here are organized to keep planner/eval logic easy to locate from the report link.
