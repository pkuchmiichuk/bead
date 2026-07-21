"""Masked-language-model scorer.

bead ships a causal-LM scorer (:class:`bead.items.scoring.LanguageModelScorer`)
but no masked-LM equivalent. This mirrors it for masked models, scoring each item
by the pseudo-log-likelihood of its text (mask each token in turn, sum the
predicted log-probabilities).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bead.items.cache import ModelOutputCache
from bead.items.scoring import ItemScorer

if TYPE_CHECKING:
    from bead.items.adapters.huggingface import HuggingFaceMaskedLanguageModel
    from bead.items.item import Item


class MaskedLanguageModelScorer(ItemScorer):
    """Score items by masked-LM pseudo-log-likelihood.

    Parameters
    ----------
    model_name : str
        HuggingFace masked-LM identifier (e.g. a RoBERTa checkpoint).
    cache_dir : Path | str | None
        Directory for caching model outputs. Caching matters here because
        pseudo-log-likelihood runs one forward pass per token.
    device : str
        Device to run the model on ("cpu", "cuda", "mps").
    text_key : str
        Key in ``item.rendered_elements`` holding the text to score.
    model_version : str
        Version string for cache tracking.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | str | None = None,
        device: str = "cpu",
        text_key: str = "text",
        model_version: str = "unknown",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.text_key = text_key
        self.model_version = model_version
        self._model: HuggingFaceMaskedLanguageModel | None = None

    @property
    def model(self) -> HuggingFaceMaskedLanguageModel:
        """Return the masked-LM adapter, loading it on first use.

        Returns
        -------
        HuggingFaceMaskedLanguageModel
            The cached masked-language-model adapter.
        """
        if self._model is None:
            from bead.items.adapters.huggingface import HuggingFaceMaskedLanguageModel

            cache = ModelOutputCache(cache_dir=self.cache_dir or Path(".cache/temp"))
            self._model = HuggingFaceMaskedLanguageModel(
                model_name=self.model_name,
                cache=cache,
                device=self.device,
                model_version=self.model_version,
            )
        return self._model

    def score(self, item: Item) -> float:
        """Compute the pseudo-log-likelihood of an item's text.

        Parameters
        ----------
        item : Item
            Item to score.

        Returns
        -------
        float
            Pseudo-log-likelihood of the item's text under the model.

        Raises
        ------
        KeyError
            If ``text_key`` is absent from ``item.rendered_elements``.
        """
        text = item.rendered_elements.get(self.text_key)
        if text is None:
            raise KeyError(f"Key '{self.text_key}' not found in item.rendered_elements")
        return self.model.compute_log_probability(text)
