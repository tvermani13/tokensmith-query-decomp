"""
Planner package: query classification, decomposition, and synthesis.

This is integrated into TokenSmith's API/CLI as an optional routing layer
for complex (multi-hop / multi-part) queries.
"""

from .classifier import QueryClassifier, QueryType, ClassificationResult
from .decomposer import QueryDecomposer, DecompositionResult, SubQuery
from .synthesizer import Synthesizer, SynthesisResult, RankedChunk
from .pipeline import PlannerPipeline, PipelineResult

__all__ = [
    "QueryClassifier",
    "QueryType",
    "ClassificationResult",
    "QueryDecomposer",
    "DecompositionResult",
    "SubQuery",
    "Synthesizer",
    "SynthesisResult",
    "RankedChunk",
    "PlannerPipeline",
    "PipelineResult",
]
