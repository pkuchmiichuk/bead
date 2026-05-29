"""Anthropic API adapter for item construction.

This module provides a ModelAdapter implementation for Anthropic's Claude API,
supporting natural language inference via prompting. Note that Claude API does
not provide direct access to log probabilities or embeddings.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import anthropic
except ImportError as e:
    raise ImportError(
        "anthropic package is required for Anthropic adapter. "
        "Install it with: pip install anthropic"
    ) from e

from bead.items.adapters.api_utils import rate_limit, retry_with_backoff
from bead.items.adapters.base import ModelAdapter
from bead.items.cache import ModelOutputCache


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude API models.

    Provides access to Claude models for prompted natural language inference.
    Note that Claude API does not support log probability computation or
    embeddings, so those methods will raise NotImplementedError.

    Parameters
    ----------
    model_name : str
        Claude model identifier (default: "claude-3-5-sonnet-20241022").
    api_key : str | None
        Anthropic API key. If None, uses ANTHROPIC_API_KEY environment variable.
    cache : ModelOutputCache | None
        Cache for model outputs. If None, creates in-memory cache.
    model_version : str
        Model version for cache tracking (default: "latest").

    Attributes
    ----------
    model_name : str
        Claude model identifier (e.g., "claude-3-5-sonnet-20241022").
    client : anthropic.Anthropic
        Anthropic API client.

    Raises
    ------
    ValueError
        If no API key is provided and ANTHROPIC_API_KEY is not set.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        cache: ModelOutputCache | None = None,
        model_version: str = "latest",
    ) -> None:
        if cache is None:
            cache = ModelOutputCache(backend="memory")

        super().__init__(
            model_name=model_name, cache=cache, model_version=model_version
        )

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Anthropic API key must be provided via api_key parameter "
                    "or ANTHROPIC_API_KEY environment variable"
                )

        self.client = anthropic.Anthropic(api_key=api_key)

    def compute_log_probability(self, text: str) -> float:
        """Compute log probability of text.

        Not supported by Anthropic API.

        Raises
        ------
        NotImplementedError
            Always raised - Claude API does not provide log probabilities.
        """
        raise NotImplementedError(
            "Log probability computation is not supported by Anthropic Claude API. "
            "Claude does not provide access to token-level probabilities."
        )

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text.

        Not supported by Anthropic API (requires log probabilities).

        Raises
        ------
        NotImplementedError
            Always raised - requires log probability support.
        """
        raise NotImplementedError(
            "Perplexity computation is not supported by Anthropic Claude API. "
            "This operation requires log probabilities, which Claude does not provide."
        )

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Not supported by Anthropic API.

        Raises
        ------
        NotImplementedError
            Always raised - Claude API does not provide embeddings.
        """
        raise NotImplementedError(
            "Embedding computation is not supported by Anthropic Claude API. "
            "Claude does not provide embedding vectors. "
            "Consider using OpenAI's text-embedding models or sentence transformers."
        )

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(
            anthropic.APIError,
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
        ),
    )
    @rate_limit(calls_per_minute=60)
    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute natural language inference scores via prompting.

        Uses Claude's messages API with a prompt to classify the relationship
        between premise and hypothesis.

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
            mapping to probability scores.
        """
        # Check cache
        cached = self.cache.get(
            model_name=self.model_name,
            operation="nli",
            premise=premise,
            hypothesis=hypothesis,
        )
        if cached is not None:
            return dict(cached)

        # Construct prompt
        prompt = (
            "Given the following premise and hypothesis, "
            "determine the relationship between them.\n\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n\n"
            "Choose one of the following:\n"
            "- entailment: The hypothesis is definitely true given the premise\n"
            "- neutral: The hypothesis might be true given the premise\n"
            "- contradiction: The hypothesis is definitely false given the premise\n\n"
            "Respond with only one word: entailment, neutral, or contradiction."
        )

        # Call API
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        if not response.content or len(response.content) == 0:
            raise ValueError("API response did not include content")

        # Get text from first content block
        answer = response.content[0].text.strip().lower()

        # Map to scores
        scores: dict[str, float] = {
            "entailment": 0.0,
            "neutral": 0.0,
            "contradiction": 0.0,
        }

        if "entailment" in answer:
            scores["entailment"] = 1.0
        elif "neutral" in answer:
            scores["neutral"] = 1.0
        elif "contradiction" in answer:
            scores["contradiction"] = 1.0
        else:
            # Default to neutral if unclear
            scores["neutral"] = 1.0

        # Cache result
        self.cache.set(
            model_name=self.model_name,
            operation="nli",
            result=scores,
            model_version=self.model_version,
            premise=premise,
            hypothesis=hypothesis,
        )

        return scores

    def generate_completion(
        self, prompt: str, *, max_tokens: int = 256, temperature: float = 1.0
    ) -> str:
        """Generate a text completion for *prompt* via the messages API.

        Parameters
        ----------
        prompt : str
            The prompt to complete.
        max_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature.

        Returns
        -------
        str
            The concatenated text of the response (empty if none).
        """
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [block.text for block in response.content if block.type == "text"]
        return "".join(parts).strip()
