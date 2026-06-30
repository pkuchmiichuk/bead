"""Tests for Together AI adapter."""

from __future__ import annotations

import os

import didactic.api as dx
import numpy as np
import pytest
from pytest_mock import MockerFixture

from bead.items.cache import ModelOutputCache


@pytest.fixture
def mock_openai(mocker: MockerFixture):
    """Mock the openai module for TogetherAI."""
    mock_client = mocker.MagicMock()
    mock_openai_module = mocker.MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client
    mock_openai_module.APIError = Exception
    mock_openai_module.APIConnectionError = Exception
    mock_openai_module.RateLimitError = Exception
    mock_openai_module.BadRequestError = Exception
    # Set __spec__ to avoid ValueError when transformers checks if openai is available
    mock_openai_module.__spec__ = mocker.MagicMock()

    # Patch both sys.modules and the module-level import in the adapter
    mocker.patch.dict("sys.modules", {"openai": mock_openai_module})
    mocker.patch("bead.items.adapters.togetherai.openai", mock_openai_module)

    return mock_client


@pytest.fixture
def togetherai_adapter(mock_openai, mocker: MockerFixture):
    """Create TogetherAI adapter instance for testing."""
    mocker.patch.dict(os.environ, {"TOGETHER_API_KEY": "test-key"})

    from bead.items.adapters.togetherai import TogetherAIAdapter  # noqa: PLC0415

    cache = ModelOutputCache(backend="memory")
    return TogetherAIAdapter(
        model_name="meta-llama/Llama-3-70b-chat-hf", cache=cache, api_key="test-key"
    )


class TestTogetherAIAdapterInitialization:
    """Tests for TogetherAI adapter initialization."""

    def test_initialization_with_api_key(self, mocker: MockerFixture) -> None:
        """Test initialization with explicit API key."""
        mock_client = mocker.MagicMock()
        mock_openai_module = mocker.MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_openai_module.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai_module})
        mocker.patch("bead.items.adapters.togetherai.openai", mock_openai_module)

        from bead.items.adapters.togetherai import TogetherAIAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")
        adapter = TogetherAIAdapter(
            model_name="meta-llama/Llama-3-70b-chat-hf",
            cache=cache,
            api_key="explicit-key",
        )

        assert adapter.model_name == "meta-llama/Llama-3-70b-chat-hf"
        assert adapter.cache is cache
        mock_openai_module.OpenAI.assert_called_once_with(
            api_key="explicit-key", base_url="https://api.together.xyz/v1"
        )

    def test_initialization_with_env_var(self, mocker: MockerFixture) -> None:
        """Test initialization using environment variable."""
        mock_client = mocker.MagicMock()
        mock_openai_module = mocker.MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_openai_module.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai_module})
        mocker.patch("bead.items.adapters.togetherai.openai", mock_openai_module)
        mocker.patch.dict(os.environ, {"TOGETHER_API_KEY": "env-key"})

        from bead.items.adapters.togetherai import TogetherAIAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")
        adapter = TogetherAIAdapter(
            model_name="meta-llama/Llama-3-70b-chat-hf", cache=cache
        )

        assert adapter.model_name == "meta-llama/Llama-3-70b-chat-hf"
        mock_openai_module.OpenAI.assert_called_once()

    def test_initialization_without_api_key_raises_error(
        self, mocker: MockerFixture
    ) -> None:
        """Test that initialization fails without API key."""
        mock_openai_module = mocker.MagicMock()
        mock_openai_module.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai_module})
        mocker.patch("bead.items.adapters.togetherai.openai", mock_openai_module)
        mocker.patch.dict(os.environ, {}, clear=True)

        from bead.items.adapters.togetherai import TogetherAIAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")

        with pytest.raises(
            (ValueError, dx.ValidationError),
            match="Together AI API key must be provided",
        ):
            TogetherAIAdapter(model_name="meta-llama/Llama-3-70b-chat-hf", cache=cache)


class TestTogetherAIComputeLogProbability:
    """Tests for compute_log_probability method."""

    def test_compute_log_probability_success(
        self, togetherai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test successful log probability computation."""
        # Mock API response
        mock_response = mocker.MagicMock()
        mock_logprobs = mocker.MagicMock()
        mock_logprobs.token_logprobs = [-0.5, -0.3, -0.2]
        mock_response.choices = [mocker.MagicMock(logprobs=mock_logprobs)]

        mock_openai.completions.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = togetherai_adapter.compute_log_probability("test text")

        assert isinstance(result, float)
        assert result == -1.0  # Sum of -0.5, -0.3, -0.2

    def test_compute_log_probability_filters_none(
        self, togetherai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test that None values in logprobs are filtered out."""
        mock_response = mocker.MagicMock()
        mock_logprobs = mocker.MagicMock()
        mock_logprobs.token_logprobs = [None, -0.5, -0.3, None, -0.2]
        mock_response.choices = [mocker.MagicMock(logprobs=mock_logprobs)]

        mock_openai.completions.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = togetherai_adapter.compute_log_probability("test text")

        assert result == -1.0  # Sum of -0.5, -0.3, -0.2 (None filtered)

    def test_compute_log_probability_uses_cache(
        self, togetherai_adapter, mocker: MockerFixture
    ) -> None:
        """Test that results are cached and retrieved."""
        mocker.patch("time.sleep")

        # Pre-populate cache
        togetherai_adapter.cache.set(
            model_name="meta-llama/Llama-3-70b-chat-hf",
            operation="log_probability",
            result=-2.5,
            model_version="latest",
            text="cached text",
        )

        result = togetherai_adapter.compute_log_probability("cached text")

        # Should get cached value without API call
        assert result == -2.5

    def test_compute_log_probability_unsupported_model_raises_error(
        self, togetherai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test that unsupported models raise NotImplementedError."""
        # Simulate BadRequestError for unsupported model
        # Create a proper exception instance without keyword arguments
        # Use AttributeError as stand-in
        bad_request_error = AttributeError("Bad request")
        mock_openai.completions.create.side_effect = bad_request_error
        mocker.patch("time.sleep")

        with pytest.raises(NotImplementedError, match="not supported for model"):
            togetherai_adapter.compute_log_probability("test text")


class TestTogetherAIComputePerplexity:
    """Tests for compute_perplexity method."""

    def test_compute_perplexity(
        self, togetherai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test perplexity computation."""
        # Mock log probability response
        mock_response = mocker.MagicMock()
        mock_logprobs = mocker.MagicMock()
        mock_logprobs.token_logprobs = [-1.0, -1.0, -1.0, -1.0]  # -4.0 total
        mock_response.choices = [mocker.MagicMock(logprobs=mock_logprobs)]

        mock_openai.completions.create.return_value = mock_response
        mocker.patch("time.sleep")

        # Test text of 16 characters (approx 4 tokens)
        result = togetherai_adapter.compute_perplexity("test text here!!")

        assert isinstance(result, float)
        assert result > 0  # Perplexity must be positive
        # perplexity = exp(-(-4.0) / 4) = exp(1.0) ≈ 2.718
        assert np.isclose(result, np.exp(1.0), rtol=0.01)


class TestTogetherAIGetEmbedding:
    """Tests for get_embedding method."""

    def test_get_embedding_not_implemented(self, togetherai_adapter) -> None:
        """Test that embedding computation raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="Embedding computation is not supported",
        ):
            togetherai_adapter.get_embedding("test text")


class TestTogetherAIComputeNLI:
    """Tests for compute_nli method."""

    def test_compute_nli_entailment(
        self, togetherai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test NLI with entailment prediction."""
        # Mock API response
        mock_message = mocker.MagicMock()
        mock_message.content = "entailment"
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai.chat.completions.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = togetherai_adapter.compute_nli("premise", "hypothesis")

        assert isinstance(result, dict)
        assert result["entailment"] == 1.0
        assert result["neutral"] == 0.0
        assert result["contradiction"] == 0.0

    def test_compute_nli_neutral(
        self, togetherai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test NLI with neutral prediction."""
        mock_message = mocker.MagicMock()
        mock_message.content = "neutral"
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai.chat.completions.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = togetherai_adapter.compute_nli("premise", "hypothesis")

        assert result["neutral"] == 1.0
        assert result["entailment"] == 0.0

    def test_compute_nli_contradiction(
        self, togetherai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test NLI with contradiction prediction."""
        mock_message = mocker.MagicMock()
        mock_message.content = "contradiction"
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai.chat.completions.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = togetherai_adapter.compute_nli("premise", "hypothesis")

        assert result["contradiction"] == 1.0
        assert result["entailment"] == 0.0

    def test_compute_nli_unclear_defaults_to_neutral(
        self, togetherai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test that unclear response defaults to neutral."""
        mock_message = mocker.MagicMock()
        mock_message.content = "unclear response"
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai.chat.completions.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = togetherai_adapter.compute_nli("premise", "hypothesis")

        assert result["neutral"] == 1.0

    def test_compute_nli_uses_cache(
        self, togetherai_adapter, mocker: MockerFixture
    ) -> None:
        """Test that NLI results are cached."""
        mocker.patch("time.sleep")

        # Pre-populate cache
        cached_result = {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}
        togetherai_adapter.cache.set(
            model_name="meta-llama/Llama-3-70b-chat-hf",
            operation="nli",
            result=cached_result,
            model_version="latest",
            premise="cached premise",
            hypothesis="cached hypothesis",
        )

        result = togetherai_adapter.compute_nli("cached premise", "cached hypothesis")

        assert result == cached_result


class TestTogetherAIRetryLogic:
    """Tests for retry and error handling."""

    def test_retry_on_api_error(
        self, togetherai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test retry on API errors."""
        import sys  # noqa: PLC0415

        openai_module = sys.modules["openai"]

        # Mock to fail twice, then succeed
        mock_response = mocker.MagicMock()
        mock_logprobs = mocker.MagicMock()
        mock_logprobs.token_logprobs = [-0.5]
        mock_response.choices = [mocker.MagicMock(logprobs=mock_logprobs)]

        mock_openai.completions.create.side_effect = [
            openai_module.APIError("error"),
            openai_module.APIError("error"),
            mock_response,
        ]

        mocker.patch("time.sleep")

        result = togetherai_adapter.compute_log_probability("test")

        assert result == -0.5
        assert mock_openai.completions.create.call_count == 3
