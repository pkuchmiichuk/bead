"""Abstract base classes for item scoring with language models.

This module provides language-agnostic base classes for scoring items
using various metrics (log probability, perplexity, embeddings).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bead.active_learning.models.forced_choice import ForcedChoiceModel
    from bead.items.adapters.huggingface import HuggingFaceLanguageModel
from collections.abc import Callable
from uuid import UUID, uuid4

import numpy as np

from bead.items.cache import ModelOutputCache
from bead.items.item import Item


class ItemScorer(ABC):
    """Abstract base class for item scoring.

    ItemScorer provides a framework for assigning numeric scores to items
    based on various criteria (language model probability, acceptability,
    similarity, etc.).

    Examples
    --------
    Implementing a custom scorer:
    >>> class AcceptabilityScorer(ItemScorer):
    ...     def score(self, item):
    ...         # Score based on some acceptability metric
    ...         text = item.rendered_elements.get("text", "")
    ...         return self._compute_acceptability(text)
    ...
    ...     def score_batch(self, items):
    ...         return [self.score(item) for item in items]
    """

    @abstractmethod
    def score(self, item: Item) -> float:
        """Compute score for a single item.

        Parameters
        ----------
        item : Item
            Item to score.

        Returns
        -------
        float
            Numeric score for the item.
        """
        ...

    def score_batch(self, items: list[Item]) -> list[float]:
        """Compute scores for multiple items.

        Default implementation calls score() for each item sequentially.
        Subclasses can override for batch processing optimization.

        Parameters
        ----------
        items : list[Item]
            Items to score.

        Returns
        -------
        list[float]
            Scores for each item.

        Examples
        --------
        >>> scorer = ConcreteScorer()
        >>> items = [item1, item2, item3]
        >>> scores = scorer.score_batch(items)  # doctest: +SKIP
        >>> len(scores) == len(items)
        True
        """
        return [self.score(item) for item in items]

    def score_with_metadata(
        self, items: list[Item]
    ) -> dict[UUID, dict[str, float | str]]:
        """Score items and return results with metadata.

        Parameters
        ----------
        items
            Items to score.

        Returns
        -------
        dict[UUID, dict[str, float | str]]
            Dictionary mapping item UUIDs to score dictionaries.
            Each score dict contains at least a "score" key.

        Examples
        --------
        >>> scorer = ConcreteScorer()
        >>> results = scorer.score_with_metadata([item1, item2])  # doctest: +SKIP
        >>> results[item1.id]["score"]  # doctest: +SKIP
        -42.5
        """
        scores = self.score_batch(items)

        results: dict[UUID, dict[str, float | str]] = {}
        for item, score in zip(items, scores, strict=True):
            results[item.id] = {"score": score}

        return results


class LanguageModelScorer(ItemScorer):
    """Scorer using language model log probabilities.

    Scores items based on their log probability under a language model.
    Uses HuggingFace adapters for model inference and supports caching.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., "gpt2", "gpt2-medium").
    cache_dir : Path | str | None
        Directory for caching model outputs. If None, no caching.
    device : str
        Device to run model on ("cpu", "cuda", "mps").
    text_key : str
        Key in item.rendered_elements to use as text (default: "text").
    model_version : str
        Version string for cache tracking.
    dtype : str
        Torch dtype to load the weights in, such as ``"bfloat16"``. Defaults to
        ``"auto"``, keeping the dtype the checkpoint was saved in.

    Examples
    --------
    >>> from pathlib import Path
    >>> scorer = LanguageModelScorer(
    ...     model_name="gpt2",
    ...     cache_dir=Path(".cache"),
    ...     device="cpu"
    ... )  # doctest: +SKIP
    >>> score = scorer.score(item)  # doctest: +SKIP
    >>> score < 0  # Log probabilities are negative  # doctest: +SKIP
    True
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | str | None = None,
        device: str = "cpu",
        text_key: str = "text",
        model_version: str = "unknown",
        dtype: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.text_key = text_key
        self.model_version = model_version
        self.dtype = dtype

        # lazy loading of model and cache
        self._model: HuggingFaceLanguageModel | None = None
        self._cache: ModelOutputCache | None = None

    @property
    def model(self) -> HuggingFaceLanguageModel:
        """Get the model, loading if necessary.

        Returns
        -------
        HuggingFaceLanguageModel
            The language model adapter.
        """
        if self._model is None:
            # import here to avoid circular dependency
            from bead.items.adapters.huggingface import (  # noqa: PLC0415
                HuggingFaceLanguageModel,
            )

            # set up cache
            if self.cache_dir:
                self._cache = ModelOutputCache(cache_dir=self.cache_dir)
            else:
                # create a no-op cache
                self._cache = ModelOutputCache(cache_dir=Path(".cache/temp"))

            self._model = HuggingFaceLanguageModel(
                model_name=self.model_name,
                cache=self._cache,
                device=self.device,  # type: ignore[arg-type]
                model_version=self.model_version,
                dtype=self.dtype,
            )

        return self._model

    def score(self, item: Item) -> float:
        """Compute log probability score for an item.

        Parameters
        ----------
        item : Item
            Item to score.

        Returns
        -------
        float
            Log probability of the item's text under the language model.

        Raises
        ------
        KeyError
            If text_key not found in item.rendered_elements.
        """
        text = item.rendered_elements.get(self.text_key)
        if text is None:
            raise KeyError(f"Key '{self.text_key}' not found in item.rendered_elements")

        return self.model.compute_log_probability(text)

    def score_batch(
        self, items: list[Item], batch_size: int | None = None
    ) -> list[float]:
        """Compute scores for multiple items efficiently using batched inference.

        Parameters
        ----------
        items : list[Item]
            Items to score.
        batch_size : int | None, default=None
            Number of items to process in each batch. If None, automatically
            infers optimal batch size based on available resources.

        Returns
        -------
        list[float]
            Log probabilities for each item.
        """
        # Extract texts
        texts: list[str] = []
        for item in items:
            text_val = item.rendered_elements.get(self.text_key)
            if text_val is None:
                msg = (
                    f"Key '{self.text_key}' not found in "
                    f"item {item.id}.rendered_elements"
                )
                raise KeyError(msg)
            # Type narrowing - text_val is now known to be str after this check
            assert isinstance(text_val, str), f"Expected str, got {type(text_val)}"
            texts.append(text_val)

        # Use batched scoring if available, otherwise fall back to sequential
        if hasattr(self.model, "compute_log_probability_batch"):
            scores = self.model.compute_log_probability_batch(
                texts, batch_size=batch_size
            )
        else:
            # Fallback for models without batch support
            scores = [self.model.compute_log_probability(text) for text in texts]

        return scores

    def score_with_metadata(
        self, items: list[Item]
    ) -> dict[UUID, dict[str, float | str]]:
        """Score items and return results with additional metrics.

        Returns log probability and perplexity for each item.

        Parameters
        ----------
        items
            Items to score.

        Returns
        -------
        dict[UUID, dict[str, float | str]]
            Dictionary with "score" (log prob) and "perplexity" for each item.
        """
        scores = self.score_batch(items)

        results: dict[UUID, dict[str, float | str]] = {}
        for item, score in zip(items, scores, strict=True):
            # compute perplexity from log probability
            # perplexity = exp(-log_prob / num_tokens)
            # for now, just include log_prob; perplexity computation
            # requires token count which we'd need to get from the model
            results[item.id] = {
                "score": score,
                "log_probability": score,
                "model": self.model_name,
            }

        return results


class ForcedChoiceScorer(ItemScorer):
    """Scorer for N-AFC (forced-choice) items with multiple options.

    Computes comparison scores for forced-choice items by scoring each
    option and applying a comparison function (e.g., max difference,
    variance, entropy).

    Parameters
    ----------
    base_scorer : ItemScorer
        Base scorer to use for individual options.
    comparison_fn : callable | None
        Function that takes list of scores and returns comparison metric.
        Default is standard deviation (variance in scores).
    option_prefix : str
        Prefix for option names in rendered_elements (default: "option").

    Examples
    --------
    >>> base = LanguageModelScorer("gpt2", device="cpu")  # doctest: +SKIP
    >>> fc_scorer = ForcedChoiceScorer(
    ...     base_scorer=base,
    ...     comparison_fn=lambda scores: max(scores) - min(scores)  # Range
    ... )  # doctest: +SKIP
    >>> # Item with option_a, option_b, option_c, ...
    >>> score = fc_scorer.score(forced_choice_item)  # doctest: +SKIP
    """

    def __init__(
        self,
        base_scorer: ItemScorer,
        comparison_fn: Callable[[list[float]], float] | None = None,
        option_prefix: str = "option",
    ) -> None:
        self.base_scorer = base_scorer
        self.option_prefix = option_prefix

        if comparison_fn is None:
            # default: standard deviation of scores
            self.comparison_fn: Callable[[list[float]], float] = (
                self._default_comparison
            )
        else:
            self.comparison_fn = comparison_fn

    @staticmethod
    def _default_comparison(scores: list[float]) -> float:
        """Compute standard deviation of scores."""
        return float(np.std(scores))

    def score(self, item: Item) -> float:
        """Score a forced-choice item.

        Extracts all options from item.rendered_elements (option_a, option_b, ...),
        scores each option, and applies comparison function.

        Parameters
        ----------
        item : Item
            Forced-choice item with multiple options.

        Returns
        -------
        float
            Comparison score across all options.

        Raises
        ------
        ValueError
            If item doesn't contain option elements or has precomputed scores.
        """
        # try to get precomputed scores from metadata first
        # look for lm_score_0, lm_score_1, ... or lm_score_a, lm_score_b, ...
        precomputed_scores = self._extract_precomputed_scores(item)
        if precomputed_scores:
            return self.comparison_fn(precomputed_scores)

        # otherwise score each option element
        option_scores: list[float] = []
        letters = "abcdefghijklmnopqrstuvwxyz"

        for letter in letters:
            option_name = f"{self.option_prefix}_{letter}"
            if option_name not in item.rendered_elements:
                break  # no more options

            # create temporary item for scoring this option
            option_text = item.rendered_elements[option_name]
            temp_item = Item(
                item_template_id=uuid4(),
                rendered_elements={"text": option_text},
            )
            score: float = self.base_scorer.score(temp_item)
            option_scores.append(score)

        if not option_scores:
            raise ValueError(
                f"Item has no options with prefix '{self.option_prefix}_' "
                "in rendered_elements"
            )

        return self.comparison_fn(option_scores)

    def _extract_precomputed_scores(self, item: Item) -> list[float] | None:
        """Extract precomputed option scores from item metadata if available.

        Looks for keys like: lm_score_0, lm_score_1, ... or
        lm_score_a, lm_score_b, ...

        Parameters
        ----------
        item : Item
            Item to extract scores from.

        Returns
        -------
        list[float] | None
            List of scores if found, None otherwise.
        """
        scores: list[float] = []
        letters = "abcdefghijklmnopqrstuvwxyz"

        # try numeric indices first (lm_score_0, lm_score_1, ...)
        for i in range(26):  # max 26 options
            key = f"lm_score_{i}"
            if key in item.item_metadata:
                metadata_val = item.item_metadata[key]
                if not isinstance(metadata_val, int | float | str):
                    raise TypeError(f"Expected numeric type, got {type(metadata_val)}")
                scores.append(float(metadata_val))
            else:
                break

        if scores:
            return scores

        # try letter indices (lm_score_a, lm_score_b, ...)
        scores = []
        for letter in letters:
            key = f"lm_score_{letter}"
            if key in item.item_metadata:
                metadata_val = item.item_metadata[key]
                if not isinstance(metadata_val, int | float | str):
                    raise TypeError(f"Expected numeric type, got {type(metadata_val)}")
                scores.append(float(metadata_val))
            else:
                break

        return scores if scores else None


class AcceptabilityScorer(ItemScorer):
    """Scorer wrapping a trained forced-choice acceptability model.

    Scores a forced-choice item by the model's predicted preference margin
    ``|2 * P(prefer first option) - 1|``, a value in ``[0, 1]`` where 0 is a
    near-tie (the model has no preference) and 1 is a clear winner. The margin
    is the stratification signal that replaces a raw language-model score
    difference.

    Parameters
    ----------
    model : ForcedChoiceModel
        A trained 2AFC model exposing ``predict_proba``.

    Examples
    --------
    >>> from bead.items.scoring import AcceptabilityScorer
    >>> scorer = AcceptabilityScorer(model)  # doctest: +SKIP
    >>> margin = scorer.score(item)  # doctest: +SKIP
    >>> 0.0 <= margin <= 1.0  # doctest: +SKIP
    True
    """

    def __init__(self, model: ForcedChoiceModel) -> None:
        self._model = model

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> AcceptabilityScorer:
        """Load a trained forced-choice model from a checkpoint directory.

        Parameters
        ----------
        path : str | Path
            Directory written by ``ForcedChoiceModel.save``.

        Returns
        -------
        AcceptabilityScorer
            Scorer wrapping the loaded model.
        """
        from bead.active_learning.models.forced_choice import (  # noqa: PLC0415
            ForcedChoiceModel,
        )
        from bead.config.active_learning import (  # noqa: PLC0415
            ForcedChoiceModelConfig,
        )

        model = ForcedChoiceModel(ForcedChoiceModelConfig())
        model.load(str(path))
        return cls(model)

    def _participant_ids(self, n_items: int) -> list[str] | None:
        """Population-level participant ids for prediction.

        Fixed-effects models take ``None``; mixed-effects models take a constant
        unknown id so prediction falls back to the population mean.
        """
        mode = self._model.config.mixed_effects.mode
        if mode == "fixed":
            return None
        return ["__population__"] * n_items

    def _margins(self, items: list[Item]) -> list[float]:
        """Predicted preference margins for a batch of forced-choice items."""
        proba = self._model.predict_proba(items, self._participant_ids(len(items)))
        return [float(2.0 * np.max(row) - 1.0) for row in proba]

    def score(self, item: Item) -> float:
        """Return the predicted preference margin for a single item."""
        return self._margins([item])[0]

    def score_batch(self, items: list[Item]) -> list[float]:
        """Return predicted preference margins for multiple items."""
        if not items:
            return []
        return self._margins(items)

    def score_with_metadata(
        self, items: list[Item]
    ) -> dict[UUID, dict[str, float | str]]:
        """Score items and return the margin plus per-option probabilities.

        Parameters
        ----------
        items
            Forced-choice items to score.

        Returns
        -------
        dict[UUID, dict[str, float | str]]
            For each item: ``"score"`` and ``"acceptability_margin"`` (the
            preference margin), ``"p_first"`` (probability of the first option),
            and ``"predicted_option"`` (argmax option index).
        """
        if not items:
            return {}

        proba = self._model.predict_proba(items, self._participant_ids(len(items)))
        results: dict[UUID, dict[str, float | str]] = {}
        for item, row in zip(items, proba, strict=True):
            margin = float(2.0 * np.max(row) - 1.0)
            results[item.id] = {
                "score": margin,
                "acceptability_margin": margin,
                "p_first": float(row[0]),
                "predicted_option": int(np.argmax(row)),
            }
        return results
