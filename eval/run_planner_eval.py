"""
Run baseline vs planner evaluation against the local TokenSmith FastAPI.

Requirements:
- TokenSmith API is running (uvicorn src.api_server:app ...)
- config/config.yaml exists (used to load judge model path)

Example:
  python -m eval.run_planner_eval \
    --questions eval/benchmark_questions.json \
    --output eval/results.json \
    --base-url http://localhost:8000 \
    --use-judge
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from src.config import RAGConfig


def _post_json(url: str, payload: dict, timeout_s: float = 600.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {raw}") from e


def keyword_recall(text: str, gold_keywords: list[str]) -> float:
    if not gold_keywords:
        return 1.0
    t = (text or "").lower()
    found = sum(1 for kw in gold_keywords if kw.lower() in t)
    return found / len(gold_keywords)


_JUDGE_PROMPT = """\
You are an expert database instructor grading a student's answer.

Question: {question}

Student's answer: {answer}

Rate the answer on a scale of 1–5:
  1 = completely wrong or off-topic
  2 = partially correct but missing key concepts
  3 = mostly correct but imprecise or incomplete
  4 = correct with minor omissions
  5 = comprehensive, accurate, and well-grounded

Respond with ONLY a single integer (1–5)."""


def llm_judge(model_path: str, question: str, answer: str) -> float:
    # Lazy import so `--help` works even if llama-cpp isn't installed
    try:
        from src.generator import get_llama_model
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "LLM judge requested, but llama-cpp-python isn't available in this environment. "
            "Run this script inside the TokenSmith conda env (make build / conda activate tokensmith)."
        ) from e

    model = get_llama_model(model_path)
    prompt = _JUDGE_PROMPT.format(question=question, answer=answer)
    out = model.create_completion(
        prompt,
        max_tokens=16,
        temperature=0.0,
        stream=False,
    )
    text = (out["choices"][0]["text"] or "").strip()
    for ch in text:
        if ch in "12345":
            return float(ch)
    return 3.0


@dataclass
class OneRun:
    answer: str
    latency_ms: float
    keyword_recall: float
    judge_score: float = 0.0
    chunks_used: Optional[list[int]] = None
    meta: Optional[dict[str, Any]] = None


@dataclass
class QuestionResult:
    id: str
    type: str
    topic: str
    question: str
    gold_keywords: list[str]
    baseline: OneRun
    planner: OneRun


@dataclass
class Summary:
    total_questions: int
    simple_count: int
    complex_count: int
    baseline_avg_keyword_recall_all: float
    planner_avg_keyword_recall_all: float
    baseline_avg_keyword_recall_complex: float
    planner_avg_keyword_recall_complex: float
    baseline_avg_latency_ms: float
    planner_avg_latency_ms: float
    baseline_avg_judge_score: float
    planner_avg_judge_score: float


def _avg(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def _summarize(results: list[QuestionResult], use_judge: bool) -> Summary:
    complex_only = [r for r in results if r.type == "complex"]
    if use_judge:
        baseline_judge_vals = [r.baseline.judge_score for r in results]
        planner_judge_vals = [r.planner.judge_score for r in results]
    else:
        baseline_judge_vals = [0.0] * len(results)
        planner_judge_vals = [0.0] * len(results)
    return Summary(
        total_questions=len(results),
        simple_count=sum(1 for r in results if r.type == "simple"),
        complex_count=sum(1 for r in results if r.type == "complex"),
        baseline_avg_keyword_recall_all=_avg(
            [r.baseline.keyword_recall for r in results]
        ),
        planner_avg_keyword_recall_all=_avg(
            [r.planner.keyword_recall for r in results]
        ),
        baseline_avg_keyword_recall_complex=_avg(
            [r.baseline.keyword_recall for r in complex_only]
        ),
        planner_avg_keyword_recall_complex=_avg(
            [r.planner.keyword_recall for r in complex_only]
        ),
        baseline_avg_latency_ms=_avg([r.baseline.latency_ms for r in results]),
        planner_avg_latency_ms=_avg([r.planner.latency_ms for r in results]),
        baseline_avg_judge_score=_avg(baseline_judge_vals) if use_judge else 0.0,
        planner_avg_judge_score=_avg(planner_judge_vals) if use_judge else 0.0,
    )


def _leave_one_out_judge_deltas(
    results: list[QuestionResult], *, which: str
) -> list[dict[str, Any]]:
    """
    For each question id, compute:
      mean_without_q - mean_with_all
    for either baseline or planner judge score.
    """
    n = len(results)
    if n == 0:
        return []
    if which == "baseline":
        scores = [r.baseline.judge_score for r in results]
    else:
        scores = [r.planner.judge_score for r in results]
    mean_all = sum(scores) / n
    s = sum(scores)
    out: list[dict[str, Any]] = []
    for r, s_i in zip(results, scores, strict=True):
        mean_wo = (s - s_i) / (n - 1) if n > 1 else 0.0
        out.append(
            {
                "id": r.id,
                "type": r.type,
                "score": float(s_i),
                "mean_judge": float(mean_all),
                "mean_judge_without_this_question": float(mean_wo),
                "mean_shift_if_removed": float(mean_wo - mean_all),
            }
        )
    out.sort(key=lambda d: abs(d["mean_shift_if_removed"]), reverse=True)
    return out


def run_single_pass(
    questions: list[dict],
    *,
    base_url: str,
    timeout_s: float,
    use_judge: bool,
    judge_model: str,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for q in questions:
        q_text = q["question"]
        gold_kw = q.get("gold_keywords", [])

        t0 = time.perf_counter()
        baseline_resp = _post_json(
            f"{base_url}/api/chat",
            {"query": q_text},
            timeout_s=timeout_s,
        )
        t1 = time.perf_counter()
        baseline_ans = baseline_resp.get("answer", "")
        baseline_run = OneRun(
            answer=baseline_ans,
            latency_ms=(t1 - t0) * 1000.0,
            keyword_recall=keyword_recall(baseline_ans, gold_kw),
            chunks_used=baseline_resp.get("chunks_used"),
        )

        t0 = time.perf_counter()
        planner_resp = _post_json(
            f"{base_url}/api/chat",
            {"query": q_text, "use_planner": True},
            timeout_s=timeout_s,
        )
        t1 = time.perf_counter()
        planner_ans = planner_resp.get("answer", "")
        planner_run = OneRun(
            answer=planner_ans,
            latency_ms=(t1 - t0) * 1000.0,
            keyword_recall=keyword_recall(planner_ans, gold_kw),
            chunks_used=planner_resp.get("chunks_used"),
            meta={"planner": planner_resp.get("planner")},
        )

        if use_judge:
            baseline_run.judge_score = llm_judge(judge_model, q_text, baseline_ans)
            planner_run.judge_score = llm_judge(judge_model, q_text, planner_ans)

        results.append(
            QuestionResult(
                id=q["id"],
                type=q["type"],
                topic=q["topic"],
                question=q_text,
                gold_keywords=gold_kw,
                baseline=baseline_run,
                planner=planner_run,
            )
        )

        print(
            f"[{q['id']}] {q['type']:<7} "
            f"kw_recall baseline={baseline_run.keyword_recall:.2f} "
            f"planner={planner_run.keyword_recall:.2f} "
            f"lat_ms baseline={baseline_run.latency_ms:.0f} "
            f"planner={planner_run.latency_ms:.0f}"
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/benchmark_questions.json")
    ap.add_argument("--output", default="eval/results.json")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--use-judge", action="store_true")
    ap.add_argument("--judge-model", default=None, help="Override judge model path.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    questions_path = Path(args.questions)
    data = json.loads(questions_path.read_text())
    questions = data["questions"]
    if args.limit:
        questions = questions[: args.limit]

    cfg = RAGConfig.from_yaml(Path("config/config.yaml"))
    judge_model = args.judge_model or cfg.gen_model

    results = run_single_pass(
        questions,
        base_url=args.base_url,
        timeout_s=args.timeout_s,
        use_judge=bool(args.use_judge),
        judge_model=judge_model,
    )
    summary = _summarize(results, use_judge=bool(args.use_judge))
    loo = None
    if args.use_judge:
        loo = {
            "baseline": _leave_one_out_judge_deltas(results, which="baseline"),
            "planner": _leave_one_out_judge_deltas(results, which="planner"),
        }

    out = {
        "metadata": data.get("metadata", {}),
        "base_url": args.base_url,
        "judge_model": judge_model if args.use_judge else None,
        "summary": asdict(summary),
        "judge_mean_shift_if_removed": loo,
        "results": [asdict(r) for r in results],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(asdict(summary), indent=2))
    if loo is not None:
        print("\nTop movers of MEAN judge (|mean_shift_if_removed|), single pass")
        print(
            "Planner top 5:",
        )
        for row in loo["planner"][:5]:
            print(
                f"  {row['id']}: score={row['score']:.1f} "
                f"shift={row['mean_shift_if_removed']:+.4f}"
            )
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
