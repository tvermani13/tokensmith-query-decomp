"""
pipeline.py — End-to-end planner pipeline for TokenSmith.

Orchestrates: classify → decompose → retrieve → synthesize.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .classifier import ClassificationResult, QueryClassifier, QueryType
from .decomposer import DecompositionResult, QueryDecomposer, SubQuery
from .synthesizer import SynthesisResult, Synthesizer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    answer: str
    classification: ClassificationResult
    decomposition: Optional[DecompositionResult]
    synthesis: SynthesisResult
    latency_ms: float
    sub_query_latencies_ms: dict[str, float] = field(default_factory=dict)


class PlannerPipeline:
    def __init__(
        self,
        llm_generate: Callable[[str], str],
        retrieve: Callable[[str, int], list[dict]],
        embed: Optional[Callable[[str], list[float]]] = None,
        final_answer_fn: Optional[Callable[[str, list[str]], str]] = None,
        top_k: int = 10,
        use_llm_rerank: bool = False,
        classifier_threshold: float = 0.55,
    ):
        self.llm_generate = llm_generate
        self.retrieve = retrieve
        self.top_k = int(top_k)

        self.classifier = QueryClassifier(
            llm_generate, heuristic_threshold=classifier_threshold
        )
        self.decomposer = QueryDecomposer(llm_generate)
        self.synthesizer = Synthesizer(
            llm_generate,
            embed_fn=embed,
            top_k_reranked=self.top_k,
            use_llm_rerank=use_llm_rerank,
            final_answer_fn=final_answer_fn,
        )

    def answer(self, query: str) -> PipelineResult:
        t0 = time.perf_counter()

        classification = self.classifier.classify(query)
        logger.info(
            "Query classified as %s (confidence=%.2f, llm=%s)",
            classification.query_type.value,
            classification.confidence,
            classification.used_llm,
        )

        if classification.query_type == QueryType.SIMPLE:
            return self._simple_path(query, classification, t0)
        return self._complex_path(query, classification, t0)

    def _simple_path(
        self,
        query: str,
        classification: ClassificationResult,
        t0: float,
    ) -> PipelineResult:
        chunks = self.retrieve(query, self.top_k)
        synthesis = self.synthesizer.synthesize(
            original_query=query,
            sub_query_chunks={"sq1": chunks},
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            answer=synthesis.answer,
            classification=classification,
            decomposition=None,
            synthesis=synthesis,
            latency_ms=elapsed,
        )

    def _complex_path(
        self,
        query: str,
        classification: ClassificationResult,
        t0: float,
    ) -> PipelineResult:
        decomp = self.decomposer.decompose(query)
        waves = self.decomposer.execution_order(decomp.sub_queries)

        sub_query_chunks: dict[str, list[dict]] = {}
        sq_latencies: dict[str, float] = {}

        for wave in waves:
            for sq in wave:
                sq_t0 = time.perf_counter()

                effective_query = sq.text
                if sq.depends_on:
                    context_parts = []
                    for dep_id in sq.depends_on:
                        dep = next(
                            (s for s in decomp.sub_queries if s.id == dep_id),
                            None,
                        )
                        if dep and dep.answer:
                            context_parts.append(
                                f"Context ({dep.text}): {dep.answer}"
                            )
                    if context_parts:
                        effective_query = "\n".join(context_parts) + "\n\n" + sq.text

                chunks = self.retrieve(effective_query, self.top_k)
                sq.chunks = chunks
                sub_query_chunks[sq.id] = chunks

                # Generate an intermediate answer only if another sub-query depends on it.
                if self._has_dependents(sq.id, decomp.sub_queries):
                    chunks_text = "\n\n".join(c.get("text", "") for c in chunks[:5])
                    intermediate_prompt = (
                        f"Based on the following excerpts, briefly answer: {sq.text}\n\n"
                        f"{chunks_text}\n\nAnswer:"
                    )
                    sq.answer = self.llm_generate(intermediate_prompt)

                sq_latencies[sq.id] = (time.perf_counter() - sq_t0) * 1000

        synthesis = self.synthesizer.synthesize(query, sub_query_chunks)
        elapsed = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            answer=synthesis.answer,
            classification=classification,
            decomposition=decomp,
            synthesis=synthesis,
            latency_ms=elapsed,
            sub_query_latencies_ms=sq_latencies,
        )

    @staticmethod
    def _has_dependents(sq_id: str, all_sqs: list[SubQuery]) -> bool:
        return any(sq_id in sq.depends_on for sq in all_sqs)
