"""
synthesizer.py — Chunk deduplication, re-ranking, and cited answer generation.

TokenSmith integration note:
We preserve TokenSmith chunk identity via `chunk_id` so the FastAPI layer
can map planner-selected chunks back to pages/sources.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RankedChunk:
    text: str
    source: str
    score: float
    sub_query_ids: list[str] = field(default_factory=list)
    chunk_id: Optional[int] = None  # TokenSmith global chunk index (if known)


@dataclass
class SynthesisResult:
    answer: str
    chunks_used: list[RankedChunk]
    total_chunks_before_dedup: int
    total_chunks_after_dedup: int


_RERANK_PROMPT = """\
You are a relevance judge.  Given a QUESTION and a CHUNK of text from a
database textbook, rate the chunk's relevance on a scale of 0–10.

Respond with ONLY a single integer (0–10), nothing else.

QUESTION: {question}

CHUNK: {chunk}"""


_SYNTHESIZE_PROMPT = """\
You are a knowledgeable database tutor.  Answer the student's question
using ONLY the provided chunks.

Strict rules:
- Write a concise, well-structured answer (prefer 3–8 bullet points or short paragraphs).
- Every sentence that makes a factual claim must end with 1–2 citations like [1] or [2][3].
- Do NOT repeat the same citation over and over.
- Do NOT output a trailing "citation list" like "[1] [1] [1] ...".
- If the chunks do not contain enough information, say so explicitly (do not guess).

If the chunks do not contain enough information, say so explicitly
rather than guessing.

Chunks:
{chunks_block}

Question: {question}

Answer:"""


class Synthesizer:
    def __init__(
        self,
        llm_generate: Callable[[str], str],
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        dedup_threshold: float = 0.85,
        top_k_reranked: int = 10,
        use_llm_rerank: bool = False,
        max_chunk_chars: int = 1200,
        max_total_context_chars: int = 18000,
        final_answer_fn: Optional[Callable[[str, list[str]], str]] = None,
    ):
        self.llm_generate = llm_generate
        self.embed_fn = embed_fn
        self.dedup_threshold = float(dedup_threshold)
        self.top_k = int(top_k_reranked)
        self.use_llm_rerank = bool(use_llm_rerank)
        self.max_chunk_chars = int(max_chunk_chars)
        self.max_total_context_chars = int(max_total_context_chars)
        self.final_answer_fn = final_answer_fn

    def synthesize(
        self,
        original_query: str,
        sub_query_chunks: dict[str, list[dict]],
    ) -> SynthesisResult:
        all_chunks = self._flatten(sub_query_chunks)
        total_before = len(all_chunks)
        logger.info("Total chunks before dedup: %d", total_before)

        deduped = self._deduplicate(all_chunks)
        total_after = len(deduped)
        logger.info("Total chunks after dedup: %d", total_after)

        ranked = self._rerank(original_query, deduped)
        top_chunks = ranked[: self.top_k]

        answer = self._generate_answer(original_query, top_chunks)

        return SynthesisResult(
            answer=answer,
            chunks_used=top_chunks,
            total_chunks_before_dedup=total_before,
            total_chunks_after_dedup=total_after,
        )

    @staticmethod
    def _flatten(sub_query_chunks: dict[str, list[dict]]) -> list[RankedChunk]:
        flat: list[RankedChunk] = []
        for sq_id, chunks in sub_query_chunks.items():
            for c in chunks:
                flat.append(
                    RankedChunk(
                        text=c.get("text", ""),
                        source=c.get("source", "unknown"),
                        score=float(c.get("score", 0.0) or 0.0),
                        sub_query_ids=[sq_id],
                        chunk_id=(
                            int(c["chunk_id"]) if "chunk_id" in c and c["chunk_id"] is not None else None
                        ),
                    )
                )
        return flat

    def _deduplicate(self, chunks: list[RankedChunk]) -> list[RankedChunk]:
        unique: list[RankedChunk] = []
        for candidate in chunks:
            is_dup = False
            for existing in unique:
                ratio = SequenceMatcher(
                    None, candidate.text[:500], existing.text[:500]
                ).ratio()
                if ratio >= self.dedup_threshold:
                    existing.sub_query_ids.extend(candidate.sub_query_ids)
                    existing.score = max(existing.score, candidate.score)
                    # Prefer to keep a real chunk_id if one side has it.
                    if existing.chunk_id is None and candidate.chunk_id is not None:
                        existing.chunk_id = candidate.chunk_id
                    is_dup = True
                    break
            if not is_dup:
                unique.append(candidate)
        return unique

    def _rerank(self, query: str, chunks: list[RankedChunk]) -> list[RankedChunk]:
        if self.use_llm_rerank:
            return self._llm_rerank(query, chunks)

        if self.embed_fn is not None:
            return self._embedding_rerank(query, chunks)

        return self._keyword_rerank(query, chunks)

    def _embedding_rerank(self, query: str, chunks: list[RankedChunk]) -> list[RankedChunk]:
        q_vec = self.embed_fn(query)
        for chunk in chunks:
            c_vec = self.embed_fn(chunk.text[:512])
            chunk.score = self._cosine(q_vec, c_vec)
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def _llm_rerank(self, query: str, chunks: list[RankedChunk]) -> list[RankedChunk]:
        for chunk in chunks:
            prompt = _RERANK_PROMPT.format(question=query, chunk=chunk.text[:800])
            raw = self.llm_generate(prompt).strip()
            try:
                chunk.score = float(re.search(r"\d+", raw).group()) / 10.0
            except (AttributeError, ValueError):
                chunk.score = 0.5
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    @staticmethod
    def _keyword_rerank(query: str, chunks: list[RankedChunk]) -> list[RankedChunk]:
        import string

        strip = str.maketrans("", "", string.punctuation)
        q_tokens = set(query.lower().translate(strip).split())
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "has",
            "have",
            "had",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "and",
            "or",
            "but",
            "not",
            "how",
            "what",
            "why",
            "when",
            "where",
            "which",
            "who",
            "its",
            "this",
            "that",
        }
        q_tokens -= stop
        for chunk in chunks:
            c_tokens = set(chunk.text.lower().translate(strip).split()) - stop
            overlap = len(q_tokens & c_tokens)
            chunk.score = overlap / max(len(q_tokens), 1)
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / max(norm_a * norm_b, 1e-9)

    def _generate_answer(self, query: str, chunks: list[RankedChunk]) -> str:
        # Hard cap prompt size to avoid exceeding the model context window.
        # Also try to preserve sub-query coverage: include at least one chunk
        # from each sub-query id (if present), then fill by rank.
        def _fmt(i: int, c: RankedChunk) -> str:
            text = (c.text or "").strip()
            if self.max_chunk_chars > 0 and len(text) > self.max_chunk_chars:
                text = text[: self.max_chunk_chars].rstrip() + "…"
            return f"[{i+1}] (source: {c.source})\n{text}"

        chosen: list[RankedChunk] = []
        seen_sq: set[str] = set()
        # First pass: take first occurrence per sub_query id (in ranked order)
        for c in chunks:
            sq_ids = c.sub_query_ids or []
            pick = False
            for sq in sq_ids:
                if sq not in seen_sq:
                    pick = True
                    break
            if pick:
                chosen.append(c)
                for sq in sq_ids:
                    seen_sq.add(sq)
        # Second pass: fill remaining budget with the rest (still ranked order)
        for c in chunks:
            if c in chosen:
                continue
            chosen.append(c)

        parts: list[str] = []
        used_chars = 0
        for i, c in enumerate(chosen):
            block = _fmt(i, c)
            projected = used_chars + len(block) + (2 if parts else 0)
            if self.max_total_context_chars > 0 and projected > self.max_total_context_chars:
                break
            parts.append(block)
            used_chars = projected

        chunks_block = "\n\n".join(parts)
        # Prefer TokenSmith's native generation formatting if provided (usually higher-quality).
        if self.final_answer_fn is not None:
            # The baseline generator expects just chunk texts; it will add its own system prompt.
            chunk_texts = []
            for block in parts:
                # Strip the header line "[i] (source: ...)" to leave mostly raw excerpt text.
                # We keep it simple: drop the first line.
                lines = block.splitlines()
                chunk_texts.append("\n".join(lines[1:]).strip())
            return self.final_answer_fn(query, chunk_texts)

        prompt = _SYNTHESIZE_PROMPT.format(question=query, chunks_block=chunks_block)
        raw = self.llm_generate(prompt)
        return self._cleanup_citations(raw)

    @staticmethod
    def _cleanup_citations(text: str) -> str:
        """
        Reduce common failure mode: runaway repeated citations like
        "[10] [10] [10] ..." at the end of the answer.
        """
        if not text:
            return text

        # Collapse repeated adjacent identical citations: "[3] [3] [3]" -> "[3]"
        cleaned = re.sub(r"(\[\d+\])(?:\s+\1)+", r"\1", text)

        # Remove very long trailing citation-only tails.
        cleaned = re.sub(r"(?:\s*\[\d+\]){8,}\s*$", "", cleaned).rstrip()
        return cleaned
