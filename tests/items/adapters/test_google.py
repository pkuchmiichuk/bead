"""Tests for Google Generative AI adapter."""

from __future__ import annotations

import os

import didactic.api as dx
import numpy as np
import pytest
from pytest_mock import MockerFixture

from bead.items.cache import ModelOutputCache


@pytest.fixture(autouse=True)
def mock_transformers_import(mocker: MockerFixture) -> None:
    """Mock transformers to avoid PyO3 errors when importing google adapter."""
    # Mock transformers and sentence_transformers if not already imported
    # This prevents the __init__.py from triggering PyO3 errors
    mocker.patch.dict(
        "sys.modules",
        {
            "transformers": mocker.MagicMock(),
            "sentence_transformers": mocker.MagicMock(),
        },
    )


@pytest.fixture
def mock_genai(mocker: MockerFixture):
    """Mock the google.generativeai module."""
    # Create mock objects
    mock_genai = mocker.MagicMock()
    mock_model = mocker.MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    mock_genai.types = mocker.MagicMock()

    # Mock sys.modules BEFORE any import
    mocker.patch.dict(
        "sys.modules",
        {
            "google": mocker.MagicMock(),
            "google.generativeai": mock_genai,
        },
    )

    # Import directly from google module
    import bead.items.adapters.google as google_adapter_module  # noqa: PLC0415

    mocker.patch.object(google_adapter_module, "genai", mock_genai)

    return mock_genai, mock_model


@pytest.fixture
def google_adapter(mock_genai, mocker: MockerFixture):
    """Create Google adapter instance for testing."""
    mock_genai_module, mock_model = mock_genai
    mocker.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})

    from bead.items.adapters.google import GoogleAdapter  # noqa: PLC0415

    cache = ModelOutputCache(backend="memory")
    return GoogleAdapter(model_name="gemini-pro", cache=cache, api_key="test-key")


class TestGoogleAdapterInitialization:
    """Tests for Google adapter initialization."""

    def test_initialization_with_api_key(self, mocker: MockerFixture) -> None:
        """Test initialization with explicit API key."""
        # Mock genai in sys.modules first
        mock_genai = mocker.MagicMock()
        mock_model = mocker.MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.types = mocker.MagicMock()

        mocker.patch.dict(
            "sys.modules",
            {
                "google": mocker.MagicMock(),
                "google.generativeai": mock_genai,
            },
        )

        # Import directly from google module to avoid __init__.py
        import bead.items.adapters.google as google_adapter_module  # noqa: PLC0415

        mocker.patch.object(google_adapter_module, "genai", mock_genai)

        from bead.items.adapters.google import GoogleAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")
        adapter = GoogleAdapter(
            model_name="gemini-pro", cache=cache, api_key="explicit-key"
        )

        assert adapter.model_name == "gemini-pro"
        assert adapter.cache is cache
        mock_genai.configure.assert_called_once_with(api_key="explicit-key")
        mock_genai.GenerativeModel.assert_called_once_with("gemini-pro")

    def test_initialization_with_env_var(self, mocker: MockerFixture) -> None:
        """Test initialization using environment variable."""
        mock_genai = mocker.MagicMock()
        mock_model = mocker.MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.types = mocker.MagicMock()

        mocker.patch.dict(
            "sys.modules",
            {
                "google": mocker.MagicMock(),
                "google.generativeai": mock_genai,
            },
        )

        # Import directly from google module to avoid __init__.py
        import bead.items.adapters.google as google_adapter_module  # noqa: PLC0415

        mocker.patch.object(google_adapter_module, "genai", mock_genai)
        mocker.patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"})

        from bead.items.adapters.google import GoogleAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")
        adapter = GoogleAdapter(model_name="gemini-pro", cache=cache)

        assert adapter.model_name == "gemini-pro"
        mock_genai.configure.assert_called_once_with(api_key="env-key")

    def test_initialization_without_api_key_raises_error(
        self, mocker: MockerFixture
    ) -> None:
        """Test that initialization fails without API key."""
        mock_genai = mocker.MagicMock()
        mock_genai.types = mocker.MagicMock()

        mocker.patch.dict(
            "sys.modules",
            {
                "google": mocker.MagicMock(),
                "google.generativeai": mock_genai,
            },
        )

        # Import directly from google module to avoid __init__.py
        import bead.items.adapters.google as google_adapter_module  # noqa: PLC0415

        mocker.patch.object(google_adapter_module, "genai", mock_genai)
        mocker.patch.dict(os.environ, {}, clear=True)

        from bead.items.adapters.google import GoogleAdapter  # noqa: PLC0415

        cache = ModelOutputCache(backend="memory")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="Google API key must be provided"
        ):
            GoogleAdapter(model_name="gemini-pro", cache=cache)


class TestGoogleNotImplementedMethods:
    """Tests for methods that are not supported by Google API."""

    def test_compute_log_probability_not_implemented(self, google_adapter) -> None:
        """Test that log probability computation raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="Log probability computation is not supported",
        ):
            google_adapter.compute_log_probability("test text")

    def test_compute_perplexity_not_implemented(self, google_adapter) -> None:
        """Test that perplexity computation raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="Perplexity computation is not supported",
        ):
            google_adapter.compute_perplexity("test text")


class TestGoogleGetEmbedding:
    """Tests for get_embedding method."""

    def test_get_embedding_success(
        self, google_adapter, mock_genai, mocker: MockerFixture
    ) -> None:
        """Test successful embedding retrieval."""
        mock_genai_module, _ = mock_genai

        # Mock API response
        embedding_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_genai_module.embed_content.return_value = {"embedding": embedding_vec}
        mocker.patch("time.sleep")

        result = google_adapter.get_embedding("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        np.testing.assert_array_equal(result, embedding_vec)

        # Verify API was called correctly
        assert mock_genai_module.embed_content.call_count >= 1

    def test_get_embedding_uses_cache(
        self, google_adapter, mocker: MockerFixture
    ) -> None:
        """Test that embeddings are cached."""
        mocker.patch("time.sleep")

        # Pre-populate cache
        cached_embedding = [0.9, 0.8, 0.7]
        google_adapter.cache.set(
            model_name="models/embedding-001",
            operation="embedding",
            result=cached_embedding,
            model_version="latest",
            text="cached text",
        )

        result = google_adapter.get_embedding("cached text")

        np.testing.assert_array_equal(result, cached_embedding)


class TestGoogleComputeNLI:
    """Tests for compute_nli method."""

    def test_compute_nli_entailment(
        self, google_adapter, mock_genai, mocker: MockerFixture
    ) -> None:
        """Test NLI with entailment prediction."""
        _, mock_model = mock_genai

        # Mock API response
        mock_response = mocker.MagicMock()
        mock_response.text = "entailment"

        mock_model.generate_content.return_value = mock_response
        mocker.patch("time.sleep")

        result = google_adapter.compute_nli("premise", "hypothesis")

        assert isinstance(result, dict)
        assert result["entailment"] == 1.0
        assert result["neutral"] == 0.0
        assert result["contradiction"] == 0.0

    def test_compute_nli_neutral(
        self, google_adapter, mock_genai, mocker: MockerFixture
    ) -> None:
        """Test NLI with neutral prediction."""
        _, mock_model = mock_genai

        mock_response = mocker.MagicMock()
        mock_response.text = "neutral"

        mock_model.generate_content.return_value = mock_response
        mocker.patch("time.sleep")

        result = google_adapter.compute_nli("premise", "hypothesis")

        assert result["neutral"] == 1.0
        assert result["entailment"] == 0.0

    def test_compute_nli_contradiction(
        self, google_adapter, mock_genai, mocker: MockerFixture
    ) -> None:
        """Test NLI with contradiction prediction."""
        _, mock_model = mock_genai

        mock_response = mocker.MagicMock()
        mock_response.text = "contradiction"

        mock_model.generate_content.return_value = mock_response
        mocker.patch("time.sleep")

        result = google_adapter.compute_nli("premise", "hypothesis")

        assert result["contradiction"] == 1.0
        assert result["entailment"] == 0.0

    def test_compute_nli_unclear_defaults_to_neutral(
        self, google_adapter, mock_genai, mocker: MockerFixture
    ) -> None:
        """Test that unclear response defaults to neutral."""
        _, mock_model = mock_genai

        mock_response = mocker.MagicMock()
        mock_response.text = "unclear response"

        mock_model.generate_content.return_value = mock_response
        mocker.patch("time.sleep")

        result = google_adapter.compute_nli("premise", "hypothesis")

        assert result["neutral"] == 1.0

    def test_compute_nli_empty_text_raises_error(
        self, google_adapter, mock_genai, mocker: MockerFixture
    ) -> None:
        """Test that empty text raises error."""
        _, mock_model = mock_genai

        mock_response = mocker.MagicMock()
        mock_response.text = None

        mock_model.generate_content.return_value = mock_response
        mocker.patch("time.sleep")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="did not include text"
        ):
            google_adapter.compute_nli("premise", "hypothesis")

    def test_compute_nli_uses_cache(
        self, google_adapter, mocker: MockerFixture
    ) -> None:
        """Test that NLI results are cached."""
        mocker.patch("time.sleep")

        # Pre-populate cache
        cached_result = {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}
        google_adapter.cache.set(
            model_name="gemini-pro",
            operation="nli",
            result=cached_result,
            model_version="latest",
            premise="cached premise",
            hypothesis="cached hypothesis",
        )

        result = google_adapter.compute_nli("cached premise", "cached hypothesis")

        assert result == cached_result

    def test_compute_nli_calls_api_correctly(
        self, google_adapter, mock_genai, mocker: MockerFixture
    ) -> None:
        """Test that NLI calls API with correct parameters."""
        mock_genai_module, mock_model = mock_genai

        mock_response = mocker.MagicMock()
        mock_response.text = "entailment"

        mock_model.generate_content.return_value = mock_response
        mocker.patch("time.sleep")

        google_adapter.compute_nli("test premise", "test hypothesis")

        # Verify API was called
        assert mock_model.generate_content.call_count >= 1

        # Just verify the model was called with something
        # (we can't easily inspect genai.types.GenerationConfig)
        call_args = mock_model.generate_content.call_args
        if call_args and len(call_args[0]) > 0:
            prompt = call_args[0][0]
            assert "test premise" in prompt
            assert "test hypothesis" in prompt


class TestGoogleRetryLogic:
    """Tests for retry and error handling."""

    def test_retry_on_exception(
        self, google_adapter, mock_genai, mocker: MockerFixture
    ) -> None:
        """Test retry on exceptions."""
        _, mock_model = mock_genai

        # Mock to fail twice, then succeed
        mock_response = mocker.MagicMock()
        mock_response.text = "entailment"

        mock_model.generate_content.side_effect = [
            Exception("error"),
            Exception("error"),
            mock_response,
        ]

        mocker.patch("time.sleep")

        result = google_adapter.compute_nli("premise", "hypothesis")

        assert result["entailment"] == 1.0
        assert mock_model.generate_content.call_count == 3
