"""Base class for model adapters used in item construction.

This module defines the abstract ModelAdapter interface that all model adapters
must implement to support judgment prediction operations during Stage 3
(Item Construction).

This is SEPARATE from template filling model adapters
(bead.templates.models.adapter), which are used in Stage 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from bead.items.cache import ModelOutputCache


@runtime_checkable
class TextGenerator(Protocol):
    """A model that generates text from a prompt.

    Implemented by API adapters that can produce completions (e.g. OpenAI,
    Anthropic). Used by ``CompletionCorpusSource`` to treat a language model as
    a corpus source. Kept separate from ``ModelAdapter`` because most adapters
    only score text, not generate it.
    """

    model_name: str

    def generate_completion(
        self, prompt: str, *, max_tokens: int = 256, temperature: float = 1.0
    ) -> str:
        """Generate a text completion for *prompt*."""
        ...


class ModelAdapter(ABC):
    """Base class for model adapters used in item construction.

    All model adapters must implement this interface to support
    judgment prediction operations during Stage 3 (Item Construction).

    This is SEPARATE from template filling model adapters
    (bead.templates.models.adapter), which are used in Stage 2.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g., "gpt2", "roberta-large-mnli").
    cache : ModelOutputCache
        Cache instance for storing model outputs.
    model_version : str
        Version of the model for cache tracking.

    Attributes
    ----------
    model_name : str
        Model identifier (e.g., "gpt2", "roberta-large-mnli").
    model_version : str
        Version of the model.
    cache : ModelOutputCache
        Cache for model outputs.
    """

    def __init__(
        self, model_name: str, cache: ModelOutputCache, model_version: str = "unknown"
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.cache = cache

    @abstractmethod
    def compute_log_probability(self, text: str) -> float:
        """Compute log probability of text under language model.

        Required for language model constraints. Should raise NotImplementedError
        if not supported by model type.

        Parameters
        ----------
        text : str
            Text to compute log probability for.

        Returns
        -------
        float
            Log probability of the text.

        Raises
        ------
        NotImplementedError
            If this operation is not supported by the model type.
        """
        pass

    @abstractmethod
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text.

        Required for complexity-based filtering. Should raise NotImplementedError
        if not supported by model type.

        Parameters
        ----------
        text : str
            Text to compute perplexity for.

        Returns
        -------
        float
            Perplexity of the text (must be positive).

        Raises
        ------
        NotImplementedError
            If this operation is not supported by the model type.
        """
        pass

    @abstractmethod
    def get_embedding(
        self, text: str
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Get embedding vector for text.

        Required for similarity computations and semantic clustering.
        Should raise NotImplementedError if not supported by model type.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector for the text.

        Raises
        ------
        NotImplementedError
            If this operation is not supported by the model type.
        """
        pass

    @abstractmethod
    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute natural language inference scores.

        Must return dict with keys: "entailment", "neutral", "contradiction".
        Required for inference-based constraints. Should raise NotImplementedError
        if not supported by model type.

        Parameters
        ----------
        premise : str
            Premise text.
        hypothesis : str
            Hypothesis text.

        Returns
        -------
        dict[str, float]
            Dictionary with keys "entailment", "neutral", "contradiction"
            mapping to probability scores that sum to ~1.0.

        Raises
        ------
        NotImplementedError
            If this operation is not supported by the model type.
        """
        pass

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.

        Default implementation using cosine similarity of embeddings.
        Can be overridden for specialized similarity computation.

        Parameters
        ----------
        text1 : str
            First text.
        text2 : str
            Second text.

        Returns
        -------
        float
            Similarity score in [-1, 1] (cosine similarity).

        Raises
        ------
        NotImplementedError
            If embeddings are not supported by the model type.
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_nli_label(self, premise: str, hypothesis: str) -> str:
        """Get predicted NLI label (max score).

        Default implementation using argmax over compute_nli() scores.

        Parameters
        ----------
        premise : str
            Premise text.
        hypothesis : str
            Hypothesis text.

        Returns
        -------
        str
            Predicted label: "entailment", "neutral", or "contradiction".

        Raises
        ------
        NotImplementedError
            If NLI is not supported by the model type.
        """
        scores = self.compute_nli(premise, hypothesis)
        return max(scores, key=scores.get)  # type: ignore[arg-type, return-value]
