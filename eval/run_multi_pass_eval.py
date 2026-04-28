"""
Run N independent full benchmark passes (default: 3) and aggregate:

- mean ± stdev of summary metrics (keyword recall, latency, judge)
- "leave-one-out" influence: which single questions most move the *mean* judge
  if removed (per pass + averaged)

Usage (API must be running):
  python -m eval.run_multi_pass_eval --use-judge --passes 3

Aggregate from existing pass files (no API calls):
  python -m eval.run_multi_pass_eval --use-judge --from-dir eval/multi_run

Outputs (default out dir):
  eval/multi_run/aggregate.json
  eval/multi_run/pass_01.json ... pass_03.json
  eval/multi_run/judge_influence.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.config import RAGConfig

from .run_planner_eval import (
    OneRun,
    QuestionResult,
    Summary,
    _summarize,
    run_single_pass,
)


def _stdev(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return math.sqrt(v) if len(xs) > 1 else 0.0


def _one_run_from_dict(d: dict[str, Any]) -> OneRun:
    return OneRun(
        answer=str(d.get("answer", "") or ""),
        latency_ms=float(d.get("latency_ms", 0.0) or 0.0),
        keyword_recall=float(d.get("keyword_recall", 0.0) or 0.0),
        judge_score=float(d.get("judge_score", 0.0) or 0.0),
        chunks_used=d.get("chunks_used"),
        meta=d.get("meta"),
    )


def _question_result_from_dict(d: dict[str, Any]) -> QuestionResult:
    baseline = d.get("baseline", {}) or {}
    planner = d.get("planner", {}) or {}
    if not isinstance(baseline, dict) or not isinstance(planner, dict):
        raise TypeError("Expected baseline/planner dicts in results JSON")
    return QuestionResult(
        id=str(d["id"]),
        type=str(d["type"]),
        topic=str(d["topic"]),
        question=str(d["question"]),
        gold_keywords=list(d.get("gold_keywords", []) or []),
        baseline=_one_run_from_dict(baseline),
        planner=_one_run_from_dict(planner),
    )


def _classifier_metrics(results: list[QuestionResult]) -> dict[str, Any]:
    """
    Compute classifier accuracy/coverage from the planner response metadata emitted
    by the API (/api/chat includes `planner` on planner responses).
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        meta = r.planner.meta or {}
        planner = meta.get("planner") if isinstance(meta, dict) else None
        if not isinstance(planner, dict):
            continue
        pred = planner.get("classification")
        rows.append(
            {
                "gold": r.type,
                "pred": pred,
                "correct": (pred == r.type),
                "used_llm": bool(planner.get("used_llm")),
                "confidence": planner.get("confidence"),
            }
        )

    covered = len(rows)
    correct = sum(1 for x in rows if x["correct"])
    used_llm = sum(1 for x in rows if x["used_llm"])
    simple = [x for x in rows if x["gold"] == "simple"]
    complex_ = [x for x in rows if x["gold"] == "complex"]
    simple_correct = sum(1 for x in simple if x["correct"])
    complex_correct = sum(1 for x in complex_ if x["correct"])

    return {
        "total": len(results),
        "covered": covered,
        "accuracy": (correct / covered) if covered else None,
        "used_llm_rate": (used_llm / covered) if covered else None,
        "simple": {
            "covered": len(simple),
            "accuracy": (simple_correct / len(simple)) if simple else None,
        },
        "complex": {
            "covered": len(complex_),
            "accuracy": (complex_correct / len(complex_)) if complex_ else None,
        },
    }


def _leave_one_out_deltas(
    results: list[QuestionResult], *, which: str
) -> dict[str, float]:
    """
    Return {qid: mean_without_q - mean_with_all} for either baseline or planner judge.
    which: "baseline" | "planner"
    """
    n = len(results)
    if n == 0:
        return {}
    if which == "baseline":
        scores = [r.baseline.judge_score for r in results]
    else:
        scores = [r.planner.judge_score for r in results]

    mean_all = sum(scores) / n
    out: dict[str, float] = {}
    s = sum(scores)
    for r, s_i in zip(results, scores, strict=True):
        mean_wo = (s - s_i) / (n - 1) if n > 1 else 0.0
        out[r.id] = mean_wo - mean_all
    return out


def _aggregate_influence(
    per_pass: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """
    Average abs influence per question id across passes.
    """
    ids = set()
    for d in per_pass:
        ids |= set(d.keys())
    rows: list[dict[str, Any]] = []
    for qid in sorted(ids):
        vals = [abs(d[qid]) for d in per_pass if qid in d]
        rows.append(
            {
                "id": qid,
                "abs_mean_shift_mean_judge__avg_across_passes": sum(vals)
                / max(len(vals), 1),
                "passes": len(vals),
            }
        )
    rows.sort(
        key=lambda r: r["abs_mean_shift_mean_judge__avg_across_passes"],
        reverse=True,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/benchmark_questions.json")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--use-judge", action="store_true")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out-dir", default="eval/multi_run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--from-dir",
        default=None,
        help=(
            "If set, skip running the benchmark and instead aggregate from existing "
            "`pass_*.json` files in this directory (writes aggregate.json + judge_influence.json)."
        ),
    )
    ap.add_argument(
        "--sleep-s-between-passes",
        type=float,
        default=0.0,
        help="Optional cool-down between passes (e.g. thermal throttling).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_pass: list[dict[str, Any]] = []
    summaries: list[Summary] = []

    if args.from_dir:
        src_dir = Path(args.from_dir)
        pass_files = sorted(src_dir.glob("pass_*.json"))
        if args.limit:
            pass_files = pass_files[: args.limit]
        if not pass_files:
            raise FileNotFoundError(f"No pass_*.json files found under {src_dir}")

        print(f"\n=== AGGREGATING FROM {len(pass_files)} EXISTING PASS FILES ===\n", flush=True)
        for pass_path in pass_files:
            p = json.loads(pass_path.read_text())
            rs = [_question_result_from_dict(r) for r in p.get("results", [])]
            s = Summary(**p["summary"]) if "summary" in p else _summarize(
                rs, use_judge=bool(args.use_judge)
            )
            summaries.append(s)
            per_pass.append(
                {
                    "pass_index": p.get("pass_index"),
                    "file": str(pass_path),
                    "elapsed_s": None,
                }
            )
        p0 = json.loads(Path(per_pass[0]["file"]).read_text())
        judge_model = args.judge_model or p0.get("judge_model")
        base_url = p0.get("base_url", args.base_url)
        questions_count = len(p0.get("results", []))
    else:
        questions_path = Path(args.questions)
        data = json.loads(questions_path.read_text())
        questions = data["questions"]
        if args.limit:
            questions = questions[: args.limit]

        cfg = RAGConfig.from_yaml(Path("config/config.yaml"))
        judge_model = args.judge_model or cfg.gen_model
        base_url = args.base_url
        questions_count = len(questions)

        for i in range(int(args.passes)):
            t0 = time.time()
            print(f"\n=== PASS {i+1}/{int(args.passes)} ===\n", flush=True)
            results = run_single_pass(
                questions,
                base_url=args.base_url,
                timeout_s=args.timeout_s,
                use_judge=bool(args.use_judge),
                judge_model=judge_model,
            )
            s = _summarize(results, use_judge=bool(args.use_judge))
            summaries.append(s)

            pass_path = out_dir / f"pass_{i+1:02d}.json"
            pass_payload = {
                "pass_index": i + 1,
                "base_url": args.base_url,
                "judge_model": judge_model if args.use_judge else None,
                "classifier": _classifier_metrics(results),
                "summary": asdict(s),
                "results": [asdict(r) for r in results],
            }
            pass_path.write_text(json.dumps(pass_payload, indent=2))
            per_pass.append(
                {
                    "pass_index": i + 1,
                    "file": str(pass_path),
                    "elapsed_s": time.time() - t0,
                }
            )
            if args.sleep_s_between_passes and i != int(args.passes) - 1:
                time.sleep(float(args.sleep_s_between_passes))

    # Aggregate means across passes
    def collect(getter) -> dict[str, float]:
        return {
            "mean": sum(getter(s) for s in summaries) / max(len(summaries), 1),
            "stdev": _stdev([getter(s) for s in summaries]),
            "min": min(getter(s) for s in summaries),
            "max": max(getter(s) for s in summaries),
        }

    agg = {
        "passes": len(summaries),
        "questions": questions_count,
        "base_url": base_url,
        "judge_model": judge_model if args.use_judge else None,
        "per_pass_files": [p["file"] for p in per_pass],
        "per_pass": per_pass,
        "metrics": {
            "planner_avg_judge_score": collect(
                lambda s: s.planner_avg_judge_score
            )
            if args.use_judge
            else None,
            "baseline_avg_judge_score": collect(
                lambda s: s.baseline_avg_judge_score
            )
            if args.use_judge
            else None,
            "planner_avg_keyword_recall_all": collect(
                lambda s: s.planner_avg_keyword_recall_all
            ),
            "baseline_avg_keyword_recall_all": collect(
                lambda s: s.baseline_avg_keyword_recall_all
            ),
            "planner_avg_latency_ms": collect(lambda s: s.planner_avg_latency_ms),
            "baseline_avg_latency_ms": collect(lambda s: s.baseline_avg_latency_ms),
        },
    }

    # Aggregate classifier metrics (mean ± stdev over passes), if available.
    cls_by_pass: list[dict[str, Any]] = []
    for pf in per_pass:
        try:
            p = json.loads(Path(pf["file"]).read_text())
            if isinstance(p.get("classifier"), dict):
                cls_by_pass.append(p["classifier"])
        except Exception:
            continue

    def _collect_cls(path: str) -> list[float]:
        vals: list[float] = []
        for d in cls_by_pass:
            v: Any = d
            for part in path.split("."):
                if not isinstance(v, dict):
                    v = None
                    break
                v = v.get(part)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    def _cls_stats(path: str) -> dict[str, Any] | None:
        vals = _collect_cls(path)
        if not vals:
            return None
        return {
            "mean": sum(vals) / len(vals),
            "stdev": _stdev(vals),
            "min": min(vals),
            "max": max(vals),
        }

    if cls_by_pass:
        agg["classifier"] = {
            "accuracy": _cls_stats("accuracy"),
            "used_llm_rate": _cls_stats("used_llm_rate"),
            "simple_accuracy": _cls_stats("simple.accuracy"),
            "complex_accuracy": _cls_stats("complex.accuracy"),
            "covered": _cls_stats("covered"),
        }

    influence = None
    if args.use_judge:
        # Re-read passes from disk to compute influence (keeps this script simple)
        baseline_los: list[dict[str, float]] = []
        planner_los: list[dict[str, float]] = []
        for pf in per_pass:
            p = json.loads(Path(pf["file"]).read_text())
            rs = [_question_result_from_dict(r) for r in p["results"]]
            baseline_los.append(_leave_one_out_deltas(rs, which="baseline"))
            planner_los.append(_leave_one_out_deltas(rs, which="planner"))
        influence = {
            "description": (
                "For each question, compute (mean_judge_without_q - mean_judge_with_all) "
                "within a pass, then take absolute value and average across passes. "
                "Large values mean the question strongly influences the reported mean."
            ),
            "baseline_top_movers": _aggregate_influence(baseline_los)[:10],
            "planner_top_movers": _aggregate_influence(planner_los)[:10],
        }
        (out_dir / "judge_influence.json").write_text(
            json.dumps(influence, indent=2)
        )

    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print("\n=== AGGREGATE (mean ± stdev over passes) ===")
    print(json.dumps(agg, indent=2))
    if influence is not None:
        print(
            "\nTop planner mean-judge movers (by avg abs mean-shift) "
            f"(see {out_dir / 'judge_influence.json'}):"
        )
        for row in influence["planner_top_movers"][:5]:
            print(
                f"  {row['id']}: {row['abs_mean_shift_mean_judge__avg_across_passes']:.4f}"
            )
    print(f"\nWrote: {out_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
