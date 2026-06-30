"""Test AdapterRegistry."""

from __future__ import annotations

import didactic.api as dx
import pytest

from bead.resources.adapters.glazing import GlazingAdapter
from bead.resources.adapters.registry import AdapterRegistry
from bead.resources.adapters.unimorph import UniMorphAdapter


def test_registry_initialization(adapter_registry: AdapterRegistry) -> None:
    """Test registry initializes empty."""
    assert adapter_registry.list_available() == []


def test_registry_register_adapter(adapter_registry: AdapterRegistry) -> None:
    """Test registering an adapter."""
    adapter_registry.register("glazing", GlazingAdapter)
    assert "glazing" in adapter_registry.list_available()


def test_registry_register_multiple_adapters(
    adapter_registry: AdapterRegistry,
) -> None:
    """Test registering multiple adapters."""
    adapter_registry.register("glazing", GlazingAdapter)
    adapter_registry.register("unimorph", UniMorphAdapter)
    available = adapter_registry.list_available()
    assert "glazing" in available
    assert "unimorph" in available


def test_registry_get_adapter(adapter_registry: AdapterRegistry) -> None:
    """Test getting adapter by name."""
    adapter_registry.register("glazing", GlazingAdapter)
    adapter = adapter_registry.get("glazing", resource="verbnet")
    assert isinstance(adapter, GlazingAdapter)
    assert adapter.resource == "verbnet"


def test_registry_get_nonexistent_adapter(adapter_registry: AdapterRegistry) -> None:
    """Test getting nonexistent adapter raises KeyError."""
    with pytest.raises(KeyError, match="not registered"):
        adapter_registry.get("nonexistent")


def test_registry_register_empty_name(adapter_registry: AdapterRegistry) -> None:
    """Test registering with empty name raises ValueError."""
    with pytest.raises((ValueError, dx.ValidationError), match="must be non-empty"):
        adapter_registry.register("", GlazingAdapter)


def test_registry_register_non_adapter_class(
    adapter_registry: AdapterRegistry,
) -> None:
    """Test registering non-adapter class raises ValueError."""

    class NotAnAdapter:
        """Not an adapter."""

        pass

    with pytest.raises((ValueError, dx.ValidationError), match="must be a subclass"):
        adapter_registry.register("invalid", NotAnAdapter)  # type: ignore[arg-type]


def test_registry_list_available_sorted(adapter_registry: AdapterRegistry) -> None:
    """Test that list_available returns sorted names."""
    adapter_registry.register("zebra", GlazingAdapter)
    adapter_registry.register("aardvark", UniMorphAdapter)
    available = adapter_registry.list_available()
    assert available == ["aardvark", "zebra"]
