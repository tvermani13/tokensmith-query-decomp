"""
decomposer.py — LLM-driven query decomposition with dependency graphs.

Given a complex query, produces a list of SubQuery objects arranged in a
dependency DAG, topologically sorted into execution waves.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    id: str  # e.g. "sq1", "sq2"
    text: str
    depends_on: list[str] = field(default_factory=list)
    answer: Optional[str] = None
    chunks: list[dict] = field(default_factory=list)


@dataclass
class DecompositionResult:
    original_query: str
    sub_queries: list[SubQuery]
    parse_succeeded: bool
    raw_llm_output: str


_DECOMPOSE_PROMPT = """\
You are a query planner for a database-textbook QA system.

Break the following COMPLEX query into 2–5 simpler sub-queries that,
when answered independently, provide all the evidence needed to answer
the original question.

Rules:
1. Each sub-query should target a single concept or fact.
2. If a sub-query depends on the answer to a previous sub-query, list
   that dependency. Independent sub-queries have an empty depends_on.
3. Preserve the user's intent — do not add questions about topics the
   user did not ask about.

Respond with ONLY a JSON array (no markdown fences, no prose):
[
  {{"id": "sq1", "text": "...", "depends_on": []}},
  {{"id": "sq2", "text": "...", "depends_on": ["sq1"]}},
  ...
]

Original query: {query}"""


class QueryDecomposer:
    def __init__(self, llm_generate: Callable[[str], str], max_retries: int = 2):
        self.llm_generate = llm_generate
        self.max_retries = int(max_retries)

    def decompose(self, query: str) -> DecompositionResult:
        prompt = _DECOMPOSE_PROMPT.format(query=query)
        raw = ""

        for attempt in range(self.max_retries):
            raw = self.llm_generate(prompt)
            sub_queries = self._parse(raw)
            if sub_queries is not None:
                logger.info(
                    "Decomposed into %d sub-queries on attempt %d",
                    len(sub_queries),
                    attempt + 1,
                )
                return DecompositionResult(
                    original_query=query,
                    sub_queries=sub_queries,
                    parse_succeeded=True,
                    raw_llm_output=raw,
                )
            logger.warning("Parse attempt %d failed, retrying...", attempt + 1)

        logger.warning(
            "All decomposition attempts failed; falling back to single query"
        )
        fallback = SubQuery(id="sq1", text=query, depends_on=[])
        return DecompositionResult(
            original_query=query,
            sub_queries=[fallback],
            parse_succeeded=False,
            raw_llm_output=raw,
        )

    @staticmethod
    def execution_order(sub_queries: list[SubQuery]) -> list[list[SubQuery]]:
        completed: set[str] = set()
        remaining = {sq.id: sq for sq in sub_queries}
        waves: list[list[SubQuery]] = []

        while remaining:
            wave = [
                sq
                for sq in remaining.values()
                if all(dep in completed for dep in sq.depends_on)
            ]
            if not wave:
                logger.warning(
                    "Circular dependency detected; forcing remaining sub-queries"
                )
                wave = list(remaining.values())

            waves.append(wave)
            for sq in wave:
                completed.add(sq.id)
                del remaining[sq.id]

        return waves

    def _parse(self, raw: str) -> Optional[list[SubQuery]]:
        try:
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start == -1 or end == -1:
                return None

            arr = json.loads(cleaned[start : end + 1])
            if not isinstance(arr, list) or len(arr) == 0:
                return None

            sub_queries: list[SubQuery] = []
            for item in arr:
                sq = SubQuery(
                    id=item.get("id", f"sq{len(sub_queries)+1}"),
                    text=item.get("text", ""),
                    depends_on=item.get("depends_on", []),
                )
                if sq.text:
                    sub_queries.append(sq)

            return sub_queries if sub_queries else None
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.debug("JSON parse error: %s", exc)
            return None
