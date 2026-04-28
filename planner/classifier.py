"""
classifier.py — Heuristic-first binary classifier for query routing.

Determines whether a user query is SIMPLE (single retrieval pass likely
suffices) or COMPLEX (benefits from decomposition into sub-queries).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class QueryType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass
class ClassificationResult:
    query: str
    query_type: QueryType
    confidence: float  # 0.0–1.0
    reason: str
    used_llm: bool


_CONJUNCTIONS = re.compile(
    r"\b(and\s+(?:how|what|why|when|where|which|who))\b"
    r"|\b(but\s+(?:how|what|why|when|where|which|who))\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(compare|contrast|differ(?:ence|ent)?|vs\.?|versus"
    r"|similarity|similarities|tradeoff|trade-off)\b",
    re.IGNORECASE,
)
_MULTI_PART = re.compile(
    r"\b(first.*then|step\s*\d|part\s*\d|also|additionally|furthermore)\b",
    re.IGNORECASE,
)
_QUESTION_MARKS = re.compile(r"\?")
_CAUSAL_CHAIN = re.compile(
    r"\b(why\s+does.*and\s+what|how\s+does.*affect|what\s+happens\s+.*when"
    r"|if.*then\s+what)\b",
    re.IGNORECASE,
)


def _heuristic_score(query: str) -> tuple[float, str]:
    """Return (complexity_score, reason). Score > 0.5 → likely complex."""
    score = 0.0
    reasons: list[str] = []

    n_qmarks = len(_QUESTION_MARKS.findall(query))
    if n_qmarks >= 2:
        score += 0.45
        reasons.append(f"{n_qmarks} question marks")

    if _COMPARISON.search(query):
        score += 0.45
        reasons.append("comparison keyword")

    if _CONJUNCTIONS.search(query):
        score += 0.30
        reasons.append("conjunction + interrogative")

    if _MULTI_PART.search(query):
        score += 0.20
        reasons.append("multi-part signal")

    if _CAUSAL_CHAIN.search(query):
        score += 0.30
        reasons.append("causal chain")

    word_count = len(query.split())
    if word_count > 25:
        score += 0.15
        reasons.append(f"long query ({word_count} words)")

    score = min(score, 1.0)
    return score, "; ".join(reasons) if reasons else "no complexity signals"


_CLASSIFY_PROMPT = """\
You are a query complexity classifier for a database-textbook QA system.

Classify the following query as either SIMPLE or COMPLEX.

SIMPLE: The query asks for a single fact, definition, or concept that can
be answered from one contiguous section of the textbook.
  Examples: "What is a B+ tree?", "Define serializability."

COMPLEX: The query requires information from multiple sections, involves
comparison, multi-hop reasoning, or has multiple sub-questions.
  Examples: "How does 2PL compare to OCC?",
            "Why does 2PC block when the coordinator fails, and what do
             participants wait for?"

Respond with ONLY a JSON object (no markdown fences):
{{"type": "SIMPLE" or "COMPLEX", "reason": "<one-sentence justification>"}}

Query: {query}"""


class QueryClassifier:
    """Classify queries as simple or complex for routing decisions."""

    def __init__(
        self,
        llm_generate: Callable[[str], str],
        heuristic_threshold: float = 0.55,
    ):
        self.llm_generate = llm_generate
        self.threshold = float(heuristic_threshold)

    def classify(self, query: str) -> ClassificationResult:
        score, reason = _heuristic_score(query)

        if score >= self.threshold:
            return ClassificationResult(
                query=query,
                query_type=QueryType.COMPLEX,
                confidence=score,
                reason=f"heuristic: {reason}",
                used_llm=False,
            )
        if score <= (1 - self.threshold):
            return ClassificationResult(
                query=query,
                query_type=QueryType.SIMPLE,
                confidence=1 - score,
                reason=f"heuristic: {reason}",
                used_llm=False,
            )

        return self._llm_classify(query)

    def _llm_classify(self, query: str) -> ClassificationResult:
        prompt = _CLASSIFY_PROMPT.format(query=query)
        raw = self.llm_generate(prompt)

        try:
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            obj = json.loads(cleaned)
            qtype = (
                QueryType.COMPLEX
                if obj.get("type", "").upper() == "COMPLEX"
                else QueryType.SIMPLE
            )
            reason = obj.get("reason", "LLM classification")
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "LLM classifier JSON parse failed; defaulting to COMPLEX"
            )
            qtype = QueryType.COMPLEX
            reason = "LLM output unparseable — defaulting to COMPLEX"

        return ClassificationResult(
            query=query,
            query_type=qtype,
            confidence=0.70,
            reason=f"llm: {reason}",
            used_llm=True,
        )
