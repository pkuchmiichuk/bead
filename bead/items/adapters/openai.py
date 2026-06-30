"""OpenAI API adapter for item construction.

This module provides a ModelAdapter implementation for OpenAI's API,
supporting GPT models for various NLP tasks including log probability
computation, embeddings, and natural language inference via prompting.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import openai
except ImportError as e:
    raise ImportError(
        "openai package is required for OpenAI adapter. "
        "Install it with: pip install openai"
    ) from e

from bead.items.adapters.api_utils import rate_limit, retry_with_backoff
from bead.items.adapters.base import ModelAdapter
from bead.items.cache import ModelOutputCache


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI API models.

    Provides access to OpenAI's GPT models for language model operations,
    embeddings, and prompted natural language inference.

    Parameters
    ----------
    model_name : str
        OpenAI model identifier (default: "gpt-3.5-turbo").
    api_key : str | None
        OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
    cache : ModelOutputCache | None
        Cache for model outputs. If None, creates in-memory cache.
    model_version : str
        Model version for cache tracking (default: "latest").
    embedding_model : str
        Model to use for embeddings (default: "text-embedding-ada-002").

    Attributes
    ----------
    model_name : str
        OpenAI model identifier (e.g., "gpt-3.5-turbo", "gpt-4").
    client : openai.OpenAI
        OpenAI API client.
    embedding_model : str
        Model to use for embeddings (default: "text-embedding-ada-002").

    Raises
    ------
    ValueError
        If no API key is provided and OPENAI_API_KEY is not set.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        cache: ModelOutputCache | None = None,
        model_version: str = "latest",
        embedding_model: str = "text-embedding-ada-002",
    ) -> None:
        if cache is None:
            cache = ModelOutputCache(backend="memory")

        super().__init__(
            model_name=model_name, cache=cache, model_version=model_version
        )

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key must be provided via api_key parameter "
                    "or OPENAI_API_KEY environment variable"
                )

        self.client = openai.OpenAI(api_key=api_key)
        self.embedding_model = embedding_model

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(openai.APIError, openai.APIConnectionError, openai.RateLimitError),
    )
    @rate_limit(calls_per_minute=60)
    def compute_log_probability(self, text: str) -> float:
        """Compute log probability of text using OpenAI completions API.

        Uses the completions API with logprobs to get token-level log probabilities
        and sums them to get the total log probability.

        Parameters
        ----------
        text : str
            Text to compute log probability for.

        Returns
        -------
        float
            Log probability of the text (sum of token log probabilities).
        """
        # Check cache
        cached = self.cache.get(
            model_name=self.model_name, operation="log_probability", text=text
        )
        if cached is not None:
            return float(cached)

        # Call API
        response = self.client.completions.create(
            model=self.model_name,
            prompt=text,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        # Sum token log probabilities
        logprobs = response.choices[0].logprobs
        if logprobs is None or logprobs.token_logprobs is None:
            raise ValueError("API response did not include logprobs")

        # Filter out None values (first token may have None)
        token_logprobs = [lp for lp in logprobs.token_logprobs if lp is not None]
        total_log_prob = sum(token_logprobs)

        # Cache result
        self.cache.set(
            model_name=self.model_name,
            operation="log_probability",
            result=total_log_prob,
            model_version=self.model_version,
            text=text,
        )

        return float(total_log_prob)

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text.

        Perplexity is computed as exp(-log_prob / num_tokens).

        Parameters
        ----------
        text : str
            Text to compute perplexity for.

        Returns
        -------
        float
            Perplexity of the text (must be positive).
        """
        # Check cache
        cached = self.cache.get(
            model_name=self.model_name, operation="perplexity", text=text
        )
        if cached is not None:
            return float(cached)

        # Get log probability
        log_prob = self.compute_log_probability(text)

        # Estimate number of tokens (rough approximation: 1 token ~ 4 chars)
        num_tokens = max(1, len(text) // 4)

        # Compute perplexity: exp(-log_prob / num_tokens)
        perplexity = np.exp(-log_prob / num_tokens)

        # Cache result
        self.cache.set(
            model_name=self.model_name,
            operation="perplexity",
            result=float(perplexity),
            model_version=self.model_version,
            text=text,
        )

        return float(perplexity)

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(openai.APIError, openai.APIConnectionError, openai.RateLimitError),
    )
    @rate_limit(calls_per_minute=60)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using OpenAI embeddings API.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector for the text.
        """
        # Check cache
        cached = self.cache.get(
            model_name=self.embedding_model, operation="embedding", text=text
        )
        if cached is not None:
            return np.array(cached)

        # Call API
        response = self.client.embeddings.create(model=self.embedding_model, input=text)

        embedding = np.array(response.data[0].embedding)

        # Cache result
        self.cache.set(
            model_name=self.embedding_model,
            operation="embedding",
            result=embedding.tolist(),
            model_version=self.model_version,
            text=text,
        )

        return embedding

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(openai.APIError, openai.APIConnectionError, openai.RateLimitError),
    )
    @rate_limit(calls_per_minute=60)
    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute natural language inference scores via prompting.

        Uses chat completions API with a prompt to classify the relationship
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
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        # Parse response
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError("API response did not include content")

        answer = answer.strip().lower()

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
        """Generate a text completion for *prompt* via the chat API.

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
            The generated text (empty if the API returns no content).
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
