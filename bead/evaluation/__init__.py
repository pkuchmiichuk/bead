"""Evaluation module for model and human performance assessment.

Provides cross-validation, inter-annotator agreement metrics, model
performance metrics, convergence detection for active learning, and
per-annotator reliability summaries.
"""

from bead.evaluation.convergence import ConvergenceDetector
from bead.evaluation.interannotator import InterAnnotatorMetrics
from bead.evaluation.reliability import (
    AnnotationRecord,
    AnnotatorReliability,
    annotator_reliability,
    low_entropy_annotators,
)

__all__ = [
    "AnnotationRecord",
    "AnnotatorReliability",
    "ConvergenceDetector",
    "InterAnnotatorMetrics",
    "annotator_reliability",
    "low_entropy_annotators",
]
