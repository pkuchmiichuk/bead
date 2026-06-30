"""Tests for Anthropic adapter."""

from __future__ import annotations

import os

import didactic.api as dx
import pytest
from pytest_mock import MockerFixture

from bead.items.cache import ModelOutputCache


@pytest.fixture
def mock_anthropic(mocker: MockerFixture):
    """Mock the anthropic module."""
    mock_client = mocker.MagicMock()
    mock_anthropic_module = mocker.MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client
    mock_anthropic_module.APIError = Exception
    mock_anthropic_module.APIConnectionError = Exception
    mock_anthropic_module.RateLimitError = Exception
    # Set __spec__ to avoid ValueError when transformers checks module availability
    mock_anthropic_module.__spec__ = mocker.MagicMock()

    # Patch both sys.modules and the module-level import in the adapter
    mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_module})
    mocker.patch("bead.items.adapters.anthropic.anthropic", mock_anthropic_module)

    return mock_client


@pytest.fixture
def anthropic_adapter(mock_anthropic, mocker: MockerFixture):
    """Create Anthropic adapter instance for testing."""
    mocker.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})

    from bead.items.adapters.anthropic import AnthropicAdapter  # noqa: PLC0415

    cache = ModelOutputCache(backend="memory")
    return AnthropicAdapter(
        model_name="claude-3-5-sonnet-20241022", cache=cache, api_key="test-key"
    )


class TestAnthropicAdapterInitialization:
    """Tests for Anthropic adapter initialization."""

    def test_initialization_with_api_key(self, mocker: MockerFixture) -> None:
        """Test initialization with explicit API key."""
        mock_client = mocker.MagicMock()
        mock_anthropic_module = mocker.MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_anthropic_module.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_module})
        mocker.patch("bead.items.adapters.anthropic.anthropic", mock_anthropic_module)

        from bead.items.adapters.anthropic import AnthropicAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")
        adapter = AnthropicAdapter(
            model_name="claude-3-5-sonnet-20241022",
            cache=cache,
            api_key="explicit-key",
        )

        assert adapter.model_name == "claude-3-5-sonnet-20241022"
        assert adapter.cache is cache
        mock_anthropic_module.Anthropic.assert_called_once_with(api_key="explicit-key")

    def test_initialization_with_env_var(self, mocker: MockerFixture) -> None:
        """Test initialization using environment variable."""
        mock_client = mocker.MagicMock()
        mock_anthropic_module = mocker.MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_anthropic_module.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_module})
        mocker.patch("bead.items.adapters.anthropic.anthropic", mock_anthropic_module)
        mocker.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"})

        from bead.items.adapters.anthropic import AnthropicAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")
        adapter = AnthropicAdapter(model_name="claude-3-5-sonnet-20241022", cache=cache)

        assert adapter.model_name == "claude-3-5-sonnet-20241022"
        mock_anthropic_module.Anthropic.assert_called_once_with(api_key="env-key")

    def test_initialization_without_api_key_raises_error(
        self, mocker: MockerFixture
    ) -> None:
        """Test that initialization fails without API key."""
        mock_anthropic_module = mocker.MagicMock()
        mock_anthropic_module.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_module})
        mocker.patch("bead.items.adapters.anthropic.anthropic", mock_anthropic_module)
        mocker.patch.dict(os.environ, {}, clear=True)

        from bead.items.adapters.anthropic import AnthropicAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="Anthropic API key must be provided"
        ):
            AnthropicAdapter(model_name="claude-3-5-sonnet-20241022", cache=cache)


class TestAnthropicNotImplementedMethods:
    """Tests for methods that are not supported by Anthropic API."""

    def test_compute_log_probability_not_implemented(self, anthropic_adapter) -> None:
        """Test that log probability computation raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="Log probability computation is not supported",
        ):
            anthropic_adapter.compute_log_probability("test text")

    def test_compute_perplexity_not_implemented(self, anthropic_adapter) -> None:
        """Test that perplexity computation raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="Perplexity computation is not supported",
        ):
            anthropic_adapter.compute_perplexity("test text")

    def test_get_embedding_not_implemented(self, anthropic_adapter) -> None:
        """Test that embedding computation raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="Embedding computation is not supported",
        ):
            anthropic_adapter.get_embedding("test text")


class TestAnthropicComputeNLI:
    """Tests for compute_nli method."""

    def test_compute_nli_entailment(
        self, anthropic_adapter, mock_anthropic, mocker: MockerFixture
    ) -> None:
        """Test NLI with entailment prediction."""
        # Mock API response
        mock_content = mocker.MagicMock()
        mock_content.text = "entailment"
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_content]

        mock_anthropic.messages.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = anthropic_adapter.compute_nli("premise", "hypothesis")

        assert isinstance(result, dict)
        assert result["entailment"] == 1.0
        assert result["neutral"] == 0.0
        assert result["contradiction"] == 0.0

    def test_compute_nli_neutral(
        self, anthropic_adapter, mock_anthropic, mocker: MockerFixture
    ) -> None:
        """Test NLI with neutral prediction."""
        mock_content = mocker.MagicMock()
        mock_content.text = "neutral"
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_content]

        mock_anthropic.messages.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = anthropic_adapter.compute_nli("premise", "hypothesis")

        assert result["neutral"] == 1.0
        assert result["entailment"] == 0.0

    def test_compute_nli_contradiction(
        self, anthropic_adapter, mock_anthropic, mocker: MockerFixture
    ) -> None:
        """Test NLI with contradiction prediction."""
        mock_content = mocker.MagicMock()
        mock_content.text = "contradiction"
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_content]

        mock_anthropic.messages.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = anthropic_adapter.compute_nli("premise", "hypothesis")

        assert result["contradiction"] == 1.0
        assert result["entailment"] == 0.0

    def test_compute_nli_unclear_defaults_to_neutral(
        self, anthropic_adapter, mock_anthropic, mocker: MockerFixture
    ) -> None:
        """Test that unclear response defaults to neutral."""
        mock_content = mocker.MagicMock()
        mock_content.text = "unclear response"
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_content]

        mock_anthropic.messages.create.return_value = mock_response
        mocker.patch("time.sleep")

        result = anthropic_adapter.compute_nli("premise", "hypothesis")

        assert result["neutral"] == 1.0

    def test_compute_nli_empty_content_raises_error(
        self, anthropic_adapter, mock_anthropic, mocker: MockerFixture
    ) -> None:
        """Test that empty content raises error."""
        mock_response = mocker.MagicMock()
        mock_response.content = []

        mock_anthropic.messages.create.return_value = mock_response
        mocker.patch("time.sleep")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="did not include content"
        ):
            anthropic_adapter.compute_nli("premise", "hypothesis")

    def test_compute_nli_uses_cache(
        self, anthropic_adapter, mocker: MockerFixture
    ) -> None:
        """Test that NLI results are cached."""
        mocker.patch("time.sleep")

        # Pre-populate cache
        cached_result = {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}
        anthropic_adapter.cache.set(
            model_name="claude-3-5-sonnet-20241022",
            operation="nli",
            result=cached_result,
            model_version="latest",
            premise="cached premise",
            hypothesis="cached hypothesis",
        )

        result = anthropic_adapter.compute_nli("cached premise", "cached hypothesis")

        assert result == cached_result

    def test_compute_nli_calls_api_correctly(
        self, anthropic_adapter, mock_anthropic, mocker: MockerFixture
    ) -> None:
        """Test that NLI calls API with correct parameters."""
        mock_content = mocker.MagicMock()
        mock_content.text = "entailment"
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_content]

        mock_anthropic.messages.create.return_value = mock_response
        mocker.patch("time.sleep")

        anthropic_adapter.compute_nli("test premise", "test hypothesis")

        # Verify API was called
        mock_anthropic.messages.create.assert_called_once()
        call_kwargs = mock_anthropic.messages.create.call_args[1]

        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["max_tokens"] == 10
        assert call_kwargs["temperature"] == 0.0
        assert len(call_kwargs["messages"]) == 1
        assert "test premise" in call_kwargs["messages"][0]["content"]
        assert "test hypothesis" in call_kwargs["messages"][0]["content"]


class TestAnthropicRetryLogic:
    """Tests for retry and error handling."""

    def test_retry_on_api_error(
        self, anthropic_adapter, mock_anthropic, mocker: MockerFixture
    ) -> None:
        """Test retry on API errors."""
        import sys  # noqa: PLC0415

        anthropic_module = sys.modules["anthropic"]

        # Mock to fail twice, then succeed
        mock_content = mocker.MagicMock()
        mock_content.text = "entailment"
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_content]

        mock_anthropic.messages.create.side_effect = [
            anthropic_module.APIError("error"),
            anthropic_module.APIError("error"),
            mock_response,
        ]

        mocker.patch("time.sleep")

        result = anthropic_adapter.compute_nli("premise", "hypothesis")

        assert result["entailment"] == 1.0
        assert mock_anthropic.messages.create.call_count == 3
