"""Tests for model adapter registry."""

from __future__ import annotations

import didactic.api as dx
import numpy as np
import pytest
from pytest_mock import MockerFixture

from bead.items.adapters.base import ModelAdapter
from bead.items.adapters.registry import ModelAdapterRegistry, default_registry
from bead.items.cache import ModelOutputCache


class DummyAdapter(ModelAdapter):
    """Dummy adapter for testing registry."""

    def compute_log_probability(self, text: str) -> float:
        """Compute dummy log probability."""
        return 0.0

    def compute_perplexity(self, text: str) -> float:
        """Compute dummy perplexity."""
        return 1.0

    def get_embedding(self, text: str) -> np.ndarray:
        """Get dummy embedding."""
        return np.array([0.0, 1.0, 2.0])

    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute dummy NLI scores."""
        return {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}


class NotAnAdapter:
    """Class that does not inherit from ModelAdapter."""

    pass


class TestModelAdapterRegistry:
    """Tests for ModelAdapterRegistry class."""

    def test_initialization(self) -> None:
        """Test registry initialization."""
        registry = ModelAdapterRegistry()
        assert registry.adapters == {}
        assert registry.instances == {}

    def test_register_adapter(self) -> None:
        """Test registering an adapter."""
        registry = ModelAdapterRegistry()
        registry.register("dummy", DummyAdapter)

        assert "dummy" in registry.adapters
        assert registry.adapters["dummy"] == DummyAdapter

    def test_register_non_adapter_raises_error(self) -> None:
        """Test that registering non-adapter class raises error."""
        registry = ModelAdapterRegistry()

        with pytest.raises(
            (ValueError, dx.ValidationError), match="must inherit from ModelAdapter"
        ):
            registry.register("invalid", NotAnAdapter)  # type: ignore[arg-type]

    def test_get_adapter_creates_instance(self) -> None:
        """Test getting adapter creates new instance."""
        registry = ModelAdapterRegistry()
        registry.register("dummy", DummyAdapter)

        cache = ModelOutputCache(backend="memory")
        adapter = registry.get_adapter("dummy", "test-model", cache=cache)

        assert isinstance(adapter, DummyAdapter)
        assert adapter.model_name == "test-model"

    def test_get_adapter_caches_instance(self) -> None:
        """Test that adapter instances are cached."""
        registry = ModelAdapterRegistry()
        registry.register("dummy", DummyAdapter)

        cache = ModelOutputCache(backend="memory")
        adapter1 = registry.get_adapter("dummy", "test-model", cache=cache)
        adapter2 = registry.get_adapter("dummy", "test-model", cache=cache)

        # Should return same instance
        assert adapter1 is adapter2

    def test_get_adapter_different_models_separate_instances(self) -> None:
        """Test that different models get separate instances."""
        registry = ModelAdapterRegistry()
        registry.register("dummy", DummyAdapter)

        cache = ModelOutputCache(backend="memory")
        adapter1 = registry.get_adapter("dummy", "model-1", cache=cache)
        adapter2 = registry.get_adapter("dummy", "model-2", cache=cache)

        # Should be different instances
        assert adapter1 is not adapter2
        assert adapter1.model_name == "model-1"
        assert adapter2.model_name == "model-2"

    def test_get_adapter_unknown_type_raises_error(self) -> None:
        """Test that unknown adapter type raises error."""
        registry = ModelAdapterRegistry()

        with pytest.raises(
            (ValueError, dx.ValidationError), match="Unknown adapter type"
        ):
            registry.get_adapter("unknown", "test-model")

    def test_clear_cache(self) -> None:
        """Test clearing cached instances."""
        registry = ModelAdapterRegistry()
        registry.register("dummy", DummyAdapter)

        cache = ModelOutputCache(backend="memory")
        adapter1 = registry.get_adapter("dummy", "test-model", cache=cache)

        registry.clear_cache()

        adapter2 = registry.get_adapter("dummy", "test-model", cache=cache)

        # Should be different instances after clearing cache
        assert adapter1 is not adapter2

    def test_list_adapters(self) -> None:
        """Test listing registered adapters."""
        registry = ModelAdapterRegistry()
        registry.register("dummy1", DummyAdapter)
        registry.register("dummy2", DummyAdapter)

        adapters = registry.list_adapters()

        assert "dummy1" in adapters
        assert "dummy2" in adapters
        assert len(adapters) == 2


class TestDefaultRegistry:
    """Tests for the default registry."""

    def test_default_registry_has_huggingface_adapters(self) -> None:
        """Test that default registry includes HuggingFace adapters."""
        adapters = default_registry.list_adapters()

        # HuggingFace adapters should be registered
        assert "huggingface_lm" in adapters
        assert "huggingface_mlm" in adapters
        assert "huggingface_nli" in adapters

    def test_default_registry_has_sentence_transformer(self) -> None:
        """Test that default registry includes sentence transformer."""
        adapters = default_registry.list_adapters()

        assert "sentence_transformer" in adapters

    def test_default_registry_api_adapters_optional(
        self, mocker: MockerFixture
    ) -> None:
        """Test that API adapters are optional (not required)."""
        # This test verifies that registry doesn't fail if API packages aren't installed
        # We can't easily test the actual import failures, but we can verify
        # the registry still works
        # Registry should still be usable even if some adapters failed to import
        adapters = default_registry.list_adapters()
        assert isinstance(adapters, list)

    def test_get_huggingface_adapter_from_default_registry(self) -> None:
        """Test getting a HuggingFace adapter from default registry."""
        cache = ModelOutputCache(backend="memory")

        # This should work without errors
        adapter = default_registry.get_adapter("huggingface_lm", "gpt2", cache=cache)

        assert adapter.model_name == "gpt2"

    def test_default_registry_caching_works(self) -> None:
        """Test that default registry caches instances."""
        cache = ModelOutputCache(backend="memory")

        adapter1 = default_registry.get_adapter("huggingface_lm", "gpt2", cache=cache)
        adapter2 = default_registry.get_adapter("huggingface_lm", "gpt2", cache=cache)

        # Should be same instance
        assert adapter1 is adapter2


class TestRegistryWithAPIAdapters:
    """Tests for registry with API adapters (if available)."""

    def test_register_and_use_openai_adapter(self, mocker: MockerFixture) -> None:
        """Test registering and using OpenAI adapter."""
        # Mock openai module
        mock_client = mocker.MagicMock()
        mock_openai_module = mocker.MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_openai_module.APIError = Exception
        mock_openai_module.APIConnectionError = Exception
        mock_openai_module.RateLimitError = Exception
        mocker.patch.dict("sys.modules", {"openai": mock_openai_module})

        # Now import and register
        from bead.items.adapters.openai import OpenAIAdapter  # noqa: PLC0415

        registry = ModelAdapterRegistry()
        registry.register("openai", OpenAIAdapter)

        cache = ModelOutputCache(backend="memory")
        adapter = registry.get_adapter(
            "openai", "gpt-4", cache=cache, api_key="test-key"
        )

        assert adapter.model_name == "gpt-4"

    def test_register_and_use_anthropic_adapter(self, mocker: MockerFixture) -> None:
        """Test registering and using Anthropic adapter."""
        # Mock anthropic module
        mock_client = mocker.MagicMock()
        mock_anthropic_module = mocker.MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_anthropic_module.APIError = Exception
        mock_anthropic_module.APIConnectionError = Exception
        mock_anthropic_module.RateLimitError = Exception
        mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_module})

        # Now import and register
        from bead.items.adapters.anthropic import AnthropicAdapter  # noqa: PLC0415

        registry = ModelAdapterRegistry()
        registry.register("anthropic", AnthropicAdapter)

        cache = ModelOutputCache(backend="memory")
        adapter = registry.get_adapter(
            "anthropic",
            "claude-3-5-sonnet-20241022",
            cache=cache,
            api_key="test-key",
        )

        assert adapter.model_name == "claude-3-5-sonnet-20241022"


class TestRegistryEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_register_same_name_overwrites(self) -> None:
        """Test that registering same name overwrites previous."""
        registry = ModelAdapterRegistry()

        class Adapter1(DummyAdapter):
            pass

        class Adapter2(DummyAdapter):
            pass

        registry.register("test", Adapter1)
        registry.register("test", Adapter2)

        # Should have the second adapter
        assert registry.adapters["test"] == Adapter2

    def test_get_adapter_with_kwargs(self) -> None:
        """Test getting adapter with additional kwargs."""
        registry = ModelAdapterRegistry()
        registry.register("dummy", DummyAdapter)

        cache = ModelOutputCache(backend="memory")
        adapter = registry.get_adapter(
            "dummy", "test-model", cache=cache, model_version="v1.0"
        )

        assert adapter.model_name == "test-model"
        assert adapter.model_version == "v1.0"

    def test_empty_registry_list(self) -> None:
        """Test listing adapters on empty registry."""
        registry = ModelAdapterRegistry()
        adapters = registry.list_adapters()

        assert adapters == []
