"""Tests for OpenAI adapter."""

from __future__ import annotations

import os

import didactic.api as dx
import numpy as np
import pytest
from pytest_mock import MockerFixture

from bead.items.cache import ModelOutputCache


@pytest.fixture
def mock_openai(mocker: MockerFixture):
    """Mock the openai module."""
    # Mock the openai module before importing the adapter
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
    mocker.patch("bead.items.adapters.openai.openai", mock_openai_module)

    return mock_client


@pytest.fixture
def openai_adapter(mock_openai, mocker: MockerFixture):
    """Create OpenAI adapter instance for testing."""
    # Mock environment variable
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})

    # Import after mocking
    from bead.items.adapters.openai import OpenAIAdapter  # noqa: PLC0415

    cache = ModelOutputCache(backend="memory")
    return OpenAIAdapter(model_name="gpt-4", cache=cache, api_key="test-key")


class TestOpenAIAdapterInitialization:
    """Tests for OpenAI adapter initialization."""

    def test_initialization_with_api_key(self, mocker: MockerFixture) -> None:
        """Test initialization with explicit API key."""
        mock_client = mocker.MagicMock()
        mock_openai_module = mocker.MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_openai_module.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai_module})
        mocker.patch("bead.items.adapters.openai.openai", mock_openai_module)

        from bead.items.adapters.openai import OpenAIAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")
        adapter = OpenAIAdapter(model_name="gpt-4", cache=cache, api_key="explicit-key")

        assert adapter.model_name == "gpt-4"
        assert adapter.cache is cache
        mock_openai_module.OpenAI.assert_called_once_with(api_key="explicit-key")

    def test_initialization_with_env_var(self, mocker: MockerFixture) -> None:
        """Test initialization using environment variable."""
        mock_client = mocker.MagicMock()
        mock_openai_module = mocker.MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_openai_module.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai_module})
        mocker.patch("bead.items.adapters.openai.openai", mock_openai_module)
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})

        from bead.items.adapters.openai import OpenAIAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")
        adapter = OpenAIAdapter(model_name="gpt-4", cache=cache)

        assert adapter.model_name == "gpt-4"
        mock_openai_module.OpenAI.assert_called_once_with(api_key="env-key")

    def test_initialization_without_api_key_raises_error(
        self, mocker: MockerFixture
    ) -> None:
        """Test that initialization fails without API key."""
        mock_openai_module = mocker.MagicMock()
        mock_openai_module.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai_module})
        mocker.patch("bead.items.adapters.openai.openai", mock_openai_module)
        mocker.patch.dict(os.environ, {}, clear=True)

        from bead.items.adapters.openai import OpenAIAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="OpenAI API key must be provided"
        ):
            OpenAIAdapter(model_name="gpt-4", cache=cache)


class TestOpenAIComputeLogProbability:
    """Tests for compute_log_probability method."""

    def test_compute_log_probability_success(
        self, openai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test successful log probability computation."""
        # Mock API response
        mock_response = mocker.MagicMock()
        mock_logprobs = mocker.MagicMock()
        mock_logprobs.token_logprobs = [-0.5, -0.3, -0.2]
        mock_response.choices = [mocker.MagicMock(logprobs=mock_logprobs)]

        mock_openai.completions.create.return_value = mock_response

        # Disable rate limiting and retry for test
        mocker.patch("time.sleep")

        result = openai_adapter.compute_log_probability("test text")

        assert isinstance(result, float)
        assert result == -1.0  # Sum of -0.5, -0.3, -0.2

    def test_compute_log_probability_filters_none(
        self, openai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test that None values in logprobs are filtered out."""
        mock_response = mocker.MagicMock()
        mock_logprobs = mocker.MagicMock()
        mock_logprobs.token_logprobs = [None, -0.5, -0.3, None, -0.2]
        mock_response.choices = [mocker.MagicMock(logprobs=mock_logprobs)]

        mock_openai.completions.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = openai_adapter.compute_log_probability("test text")

        assert result == -1.0  # Sum of -0.5, -0.3, -0.2 (None filtered)

    def test_compute_log_probability_uses_cache(
        self, openai_adapter, mocker: MockerFixture
    ) -> None:
        """Test that results are cached and retrieved."""
        mocker.patch("time.sleep")

        # Pre-populate cache
        openai_adapter.cache.set(
            model_name="gpt-4",
            operation="log_probability",
            result=-2.5,
            model_version="latest",
            text="cached text",
        )

        result = openai_adapter.compute_log_probability("cached text")

        # Should get cached value without API call
        assert result == -2.5

    def test_compute_log_probability_no_logprobs_raises_error(
        self, openai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test that missing logprobs raises error."""
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock(logprobs=None)]

        mock_openai.completions.create.return_value = mock_response
        mocker.patch("time.sleep")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="did not include logprobs"
        ):
            openai_adapter.compute_log_probability("test text")


class TestOpenAIComputePerplexity:
    """Tests for compute_perplexity method."""

    def test_compute_perplexity(
        self, openai_adapter, mock_openai, mocker: MockerFixture
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
        result = openai_adapter.compute_perplexity("test text here!!")

        assert isinstance(result, float)
        assert result > 0  # Perplexity must be positive
        # perplexity = exp(-(-4.0) / 4) = exp(1.0) ≈ 2.718
        assert np.isclose(result, np.exp(1.0), rtol=0.01)


class TestOpenAIGetEmbedding:
    """Tests for get_embedding method."""

    def test_get_embedding_success(
        self, openai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test successful embedding retrieval."""
        # Mock API response
        embedding_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_data = mocker.MagicMock()
        mock_data.embedding = embedding_vec
        mock_response = mocker.MagicMock()
        mock_response.data = [mock_data]

        mock_openai.embeddings.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = openai_adapter.get_embedding("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        np.testing.assert_array_equal(result, embedding_vec)

    def test_get_embedding_uses_cache(
        self, openai_adapter, mocker: MockerFixture
    ) -> None:
        """Test that embeddings are cached."""
        mocker.patch("time.sleep")

        # Pre-populate cache
        cached_embedding = [0.9, 0.8, 0.7]
        openai_adapter.cache.set(
            model_name="text-embedding-ada-002",
            operation="embedding",
            result=cached_embedding,
            model_version="latest",
            text="cached text",
        )

        result = openai_adapter.get_embedding("cached text")

        np.testing.assert_array_equal(result, cached_embedding)


class TestOpenAIComputeNLI:
    """Tests for compute_nli method."""

    def test_compute_nli_entailment(
        self, openai_adapter, mock_openai, mocker: MockerFixture
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

        result = openai_adapter.compute_nli("premise", "hypothesis")

        assert isinstance(result, dict)
        assert result["entailment"] == 1.0
        assert result["neutral"] == 0.0
        assert result["contradiction"] == 0.0

    def test_compute_nli_neutral(
        self, openai_adapter, mock_openai, mocker: MockerFixture
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

        result = openai_adapter.compute_nli("premise", "hypothesis")

        assert result["neutral"] == 1.0
        assert result["entailment"] == 0.0

    def test_compute_nli_contradiction(
        self, openai_adapter, mock_openai, mocker: MockerFixture
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

        result = openai_adapter.compute_nli("premise", "hypothesis")

        assert result["contradiction"] == 1.0
        assert result["entailment"] == 0.0

    def test_compute_nli_unclear_defaults_to_neutral(
        self, openai_adapter, mock_openai, mocker: MockerFixture
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

        result = openai_adapter.compute_nli("premise", "hypothesis")

        assert result["neutral"] == 1.0

    def test_compute_nli_uses_cache(
        self, openai_adapter, mocker: MockerFixture
    ) -> None:
        """Test that NLI results are cached."""
        mocker.patch("time.sleep")

        # Pre-populate cache
        cached_result = {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}
        openai_adapter.cache.set(
            model_name="gpt-4",
            operation="nli",
            result=cached_result,
            model_version="latest",
            premise="cached premise",
            hypothesis="cached hypothesis",
        )

        result = openai_adapter.compute_nli("cached premise", "cached hypothesis")

        assert result == cached_result


class TestOpenAIRetryLogic:
    """Tests for retry and error handling."""

    def test_retry_on_api_error(
        self, openai_adapter, mock_openai, mocker: MockerFixture
    ) -> None:
        """Test retry on API errors."""
        # Import after mocking
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

        result = openai_adapter.compute_log_probability("test")

        assert result == -0.5
        assert mock_openai.completions.create.call_count == 3
