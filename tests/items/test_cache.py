"""Test model output cache."""

from __future__ import annotations

from pathlib import Path

import didactic.api as dx
import numpy as np
import pytest

from bead.items.cache import (
    FilesystemBackend,
    InMemoryBackend,
    ModelOutputCache,
)

# Fixtures


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Temporary cache directory.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def fs_backend(temp_cache_dir: Path) -> FilesystemBackend:
    """Create filesystem backend.

    Parameters
    ----------
    temp_cache_dir : Path
        Temporary cache directory.

    Returns
    -------
    FilesystemBackend
        Filesystem backend instance.
    """
    return FilesystemBackend(temp_cache_dir)


@pytest.fixture
def memory_backend() -> InMemoryBackend:
    """Create in-memory backend.

    Returns
    -------
    InMemoryBackend
        In-memory backend instance.
    """
    return InMemoryBackend()


@pytest.fixture
def fs_cache(temp_cache_dir: Path) -> ModelOutputCache:
    """Create cache with filesystem backend.

    Parameters
    ----------
    temp_cache_dir : Path
        Temporary cache directory.

    Returns
    -------
    ModelOutputCache
        Cache instance with filesystem backend.
    """
    return ModelOutputCache(cache_dir=temp_cache_dir, backend="filesystem")


@pytest.fixture
def memory_cache() -> ModelOutputCache:
    """Create cache with in-memory backend.

    Returns
    -------
    ModelOutputCache
        Cache instance with in-memory backend.
    """
    return ModelOutputCache(backend="memory")


# FilesystemBackend Tests


def test_filesystem_backend_initialization(temp_cache_dir: Path) -> None:
    """Test filesystem backend initialization creates directory."""
    cache_dir = temp_cache_dir / "new_cache"
    _ = FilesystemBackend(cache_dir)
    assert cache_dir.exists()
    assert cache_dir.is_dir()


def test_filesystem_backend_get_miss(fs_backend: FilesystemBackend) -> None:
    """Test get returns None for missing key."""
    result = fs_backend.get("nonexistent_key")
    assert result is None


def test_filesystem_backend_set_and_get(fs_backend: FilesystemBackend) -> None:
    """Test set and get operations."""
    data = {"result": 42, "metadata": "test"}
    fs_backend.set("test_key", data)

    retrieved = fs_backend.get("test_key")
    assert retrieved == data


def test_filesystem_backend_delete(fs_backend: FilesystemBackend) -> None:
    """Test delete operation."""
    data = {"result": 100}
    fs_backend.set("key_to_delete", data)
    assert fs_backend.get("key_to_delete") is not None

    fs_backend.delete("key_to_delete")
    assert fs_backend.get("key_to_delete") is None


def test_filesystem_backend_delete_nonexistent(fs_backend: FilesystemBackend) -> None:
    """Test delete on nonexistent key doesn't raise error."""
    fs_backend.delete("nonexistent_key")  # Should not raise


def test_filesystem_backend_clear(fs_backend: FilesystemBackend) -> None:
    """Test clear removes all entries."""
    fs_backend.set("key1", {"value": 1})
    fs_backend.set("key2", {"value": 2})
    fs_backend.set("key3", {"value": 3})

    fs_backend.clear()

    assert fs_backend.get("key1") is None
    assert fs_backend.get("key2") is None
    assert fs_backend.get("key3") is None


def test_filesystem_backend_keys(fs_backend: FilesystemBackend) -> None:
    """Test keys returns all cache keys."""
    fs_backend.set("key1", {"value": 1})
    fs_backend.set("key2", {"value": 2})
    fs_backend.set("key3", {"value": 3})

    keys = fs_backend.keys()
    assert set(keys) == {"key1", "key2", "key3"}


def test_filesystem_backend_corrupted_file(
    fs_backend: FilesystemBackend, temp_cache_dir: Path
) -> None:
    """Test handling of corrupted cache file."""
    # Create corrupted JSON file
    corrupted_file = temp_cache_dir / "corrupted.json"
    corrupted_file.write_text("this is not valid json{")

    result = fs_backend.get("corrupted")
    assert result is None  # Should gracefully return None


# InMemoryBackend Tests


def test_memory_backend_initialization() -> None:
    """Test in-memory backend initialization."""
    backend = InMemoryBackend()
    assert backend._cache == {}


def test_memory_backend_get_miss(memory_backend: InMemoryBackend) -> None:
    """Test get returns None for missing key."""
    result = memory_backend.get("nonexistent_key")
    assert result is None


def test_memory_backend_set_and_get(memory_backend: InMemoryBackend) -> None:
    """Test set and get operations."""
    data = {"result": 42, "metadata": "test"}
    memory_backend.set("test_key", data)

    retrieved = memory_backend.get("test_key")
    assert retrieved == data


def test_memory_backend_delete(memory_backend: InMemoryBackend) -> None:
    """Test delete operation."""
    data = {"result": 100}
    memory_backend.set("key_to_delete", data)
    assert memory_backend.get("key_to_delete") is not None

    memory_backend.delete("key_to_delete")
    assert memory_backend.get("key_to_delete") is None


def test_memory_backend_delete_nonexistent(memory_backend: InMemoryBackend) -> None:
    """Test delete on nonexistent key doesn't raise error."""
    memory_backend.delete("nonexistent_key")  # Should not raise


def test_memory_backend_clear(memory_backend: InMemoryBackend) -> None:
    """Test clear removes all entries."""
    memory_backend.set("key1", {"value": 1})
    memory_backend.set("key2", {"value": 2})
    memory_backend.set("key3", {"value": 3})

    memory_backend.clear()

    assert memory_backend.get("key1") is None
    assert memory_backend.get("key2") is None
    assert memory_backend.get("key3") is None
    assert memory_backend._cache == {}


def test_memory_backend_keys(memory_backend: InMemoryBackend) -> None:
    """Test keys returns all cache keys."""
    memory_backend.set("key1", {"value": 1})
    memory_backend.set("key2", {"value": 2})
    memory_backend.set("key3", {"value": 3})

    keys = memory_backend.keys()
    assert set(keys) == {"key1", "key2", "key3"}


# ModelOutputCache Tests


def test_cache_initialization_filesystem(temp_cache_dir: Path) -> None:
    """Test cache initialization with filesystem backend."""
    cache = ModelOutputCache(cache_dir=temp_cache_dir, backend="filesystem")
    assert cache.enabled is True
    assert isinstance(cache._backend, FilesystemBackend)


def test_cache_initialization_memory() -> None:
    """Test cache initialization with in-memory backend."""
    cache = ModelOutputCache(backend="memory")
    assert cache.enabled is True
    assert isinstance(cache._backend, InMemoryBackend)


def test_cache_initialization_default_dir() -> None:
    """Test cache initialization with default directory."""
    cache = ModelOutputCache(backend="filesystem")
    # Verify the backend was created
    assert isinstance(cache._backend, FilesystemBackend)


def test_cache_initialization_unknown_backend() -> None:
    """Test cache initialization with unknown backend raises error."""
    with pytest.raises((ValueError, dx.ValidationError), match="Unknown backend"):
        ModelOutputCache(backend="invalid")  # type: ignore


def test_cache_disabled(memory_cache: ModelOutputCache) -> None:
    """Test cache operations when disabled."""
    cache = ModelOutputCache(backend="memory", enabled=False)

    # Set should be no-op
    cache.set("model", "op", 42, text="hello")

    # Get should return None
    result = cache.get("model", "op", text="hello")
    assert result is None


def test_cache_key_generation_deterministic(memory_cache: ModelOutputCache) -> None:
    """Test cache key generation is deterministic."""
    key1 = memory_cache.generate_cache_key("model", "op", text="hello", value=42)
    key2 = memory_cache.generate_cache_key("model", "op", text="hello", value=42)
    assert key1 == key2


def test_cache_key_generation_order_independent(
    memory_cache: ModelOutputCache,
) -> None:
    """Test cache key generation is independent of kwarg order."""
    key1 = memory_cache.generate_cache_key(
        "model", "op", text="hello", value=42, flag=True
    )
    key2 = memory_cache.generate_cache_key(
        "model", "op", flag=True, value=42, text="hello"
    )
    assert key1 == key2


def test_cache_key_generation_different_inputs(
    memory_cache: ModelOutputCache,
) -> None:
    """Test different inputs produce different keys."""
    key1 = memory_cache.generate_cache_key("model", "op", text="hello")
    key2 = memory_cache.generate_cache_key("model", "op", text="world")
    assert key1 != key2


def test_cache_get_miss(memory_cache: ModelOutputCache) -> None:
    """Test get returns None for cache miss."""
    result = memory_cache.get("model", "op", text="hello")
    assert result is None


def test_cache_set_and_get_float(memory_cache: ModelOutputCache) -> None:
    """Test caching float value."""
    memory_cache.set("gpt2", "log_probability", -2.5, text="Hello world")
    result = memory_cache.get("gpt2", "log_probability", text="Hello world")
    assert result == -2.5


def test_cache_set_and_get_dict(memory_cache: ModelOutputCache) -> None:
    """Test caching dict value."""
    nli_scores = {"entailment": 0.9, "neutral": 0.08, "contradiction": 0.02}
    memory_cache.set(
        "roberta-nli",
        "nli",
        nli_scores,
        premise="Mary loves books",
        hypothesis="Mary enjoys reading",
    )

    result = memory_cache.get(
        "roberta-nli",
        "nli",
        premise="Mary loves books",
        hypothesis="Mary enjoys reading",
    )
    assert result == nli_scores


def test_cache_set_and_get_numpy_array(memory_cache: ModelOutputCache) -> None:
    """Test caching numpy array."""
    embedding = np.array([1.0, 2.0, 3.0, 4.0])
    memory_cache.set("bert-base", "embedding", embedding, text="Hello")

    result = memory_cache.get("bert-base", "embedding", text="Hello")
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, embedding)


def test_cache_set_and_get_numpy_array_dtype(memory_cache: ModelOutputCache) -> None:
    """Test caching numpy array preserves dtype."""
    embedding = np.array([1, 2, 3, 4], dtype=np.int32)
    memory_cache.set("model", "embedding", embedding, text="test")

    result = memory_cache.get("model", "embedding", text="test")
    assert result.dtype == np.int32


def test_cache_set_with_model_version(memory_cache: ModelOutputCache) -> None:
    """Test caching with model version."""
    memory_cache.set("gpt2", "log_prob", -1.5, model_version="1.0.0", text="test")

    # Verify metadata is stored
    cache_key = memory_cache.generate_cache_key("gpt2", "log_prob", text="test")
    entry = memory_cache._backend.get(cache_key)
    assert entry is not None
    assert entry["model_version"] == "1.0.0"


def test_cache_metadata_fields(memory_cache: ModelOutputCache) -> None:
    """Test cache entry metadata fields."""
    memory_cache.set("model", "op", 42, model_version="2.0", text="hello")

    cache_key = memory_cache.generate_cache_key("model", "op", text="hello")
    entry = memory_cache._backend.get(cache_key)

    assert entry is not None
    assert "cache_key" in entry
    assert "timestamp" in entry
    assert "model_name" in entry
    assert "model_version" in entry
    assert "operation" in entry
    assert "inputs" in entry
    assert "result" in entry

    assert entry["model_name"] == "model"
    assert entry["operation"] == "op"
    assert entry["model_version"] == "2.0"


def test_cache_invalidate(memory_cache: ModelOutputCache) -> None:
    """Test invalidate removes specific entry."""
    memory_cache.set("model", "op", 42, text="hello")
    memory_cache.set("model", "op", 100, text="world")

    memory_cache.invalidate("model", "op", text="hello")

    assert memory_cache.get("model", "op", text="hello") is None
    assert memory_cache.get("model", "op", text="world") == 100


def test_cache_clear_model(memory_cache: ModelOutputCache) -> None:
    """Test clear_model removes all entries for a model."""
    memory_cache.set("gpt2", "log_prob", -1.5, text="a")
    memory_cache.set("gpt2", "perplexity", 10.0, text="b")
    memory_cache.set("bert", "embedding", np.array([1, 2]), text="c")

    memory_cache.clear_model("gpt2")

    assert memory_cache.get("gpt2", "log_prob", text="a") is None
    assert memory_cache.get("gpt2", "perplexity", text="b") is None
    assert memory_cache.get("bert", "embedding", text="c") is not None


def test_cache_clear(memory_cache: ModelOutputCache) -> None:
    """Test clear removes all entries."""
    memory_cache.set("model1", "op1", 1, text="a")
    memory_cache.set("model2", "op2", 2, text="b")
    memory_cache.set("model3", "op3", 3, text="c")

    memory_cache.clear()

    assert memory_cache.get("model1", "op1", text="a") is None
    assert memory_cache.get("model2", "op2", text="b") is None
    assert memory_cache.get("model3", "op3", text="c") is None


def test_cache_complex_nested_data(memory_cache: ModelOutputCache) -> None:
    """Test caching complex nested data structures."""
    data = {
        "scores": [0.1, 0.2, 0.3],
        "metadata": {
            "model": "test",
            "params": {"temperature": 0.7, "top_k": 50},
        },
        "flags": [True, False, True],
    }

    memory_cache.set("model", "op", data, text="test")
    result = memory_cache.get("model", "op", text="test")
    assert result == data


def test_cache_nested_numpy_arrays(memory_cache: ModelOutputCache) -> None:
    """Test caching nested structures with numpy arrays."""
    data = {
        "embedding": np.array([1.0, 2.0, 3.0]),
        "score": 0.95,
        "hidden_states": [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
        ],
    }

    memory_cache.set("model", "op", data, text="test")
    result = memory_cache.get("model", "op", text="test")

    assert result["score"] == 0.95
    np.testing.assert_array_equal(result["embedding"], data["embedding"])
    np.testing.assert_array_equal(result["hidden_states"][0], data["hidden_states"][0])
    np.testing.assert_array_equal(result["hidden_states"][1], data["hidden_states"][1])


def test_cache_filesystem_persistence(temp_cache_dir: Path) -> None:
    """Test filesystem cache persists across instances."""
    cache1 = ModelOutputCache(cache_dir=temp_cache_dir, backend="filesystem")
    cache1.set("model", "op", 42, text="hello")

    # Create new cache instance with same directory
    cache2 = ModelOutputCache(cache_dir=temp_cache_dir, backend="filesystem")
    result = cache2.get("model", "op", text="hello")

    assert result == 42


def test_cache_memory_no_persistence() -> None:
    """Test memory cache doesn't persist across instances."""
    cache1 = ModelOutputCache(backend="memory")
    cache1.set("model", "op", 42, text="hello")

    # Create new cache instance
    cache2 = ModelOutputCache(backend="memory")
    result = cache2.get("model", "op", text="hello")

    assert result is None  # Different instance, no persistence


def test_cache_unicode_text(memory_cache: ModelOutputCache) -> None:
    """Test caching with unicode text."""
    texts = [
        "Hello 世界",
        "Привет мир",
        "مرحبا بالعالم",
        "😀🎉🌍",
    ]

    for i, text in enumerate(texts):
        memory_cache.set("model", "op", i, text=text)

    for i, text in enumerate(texts):
        result = memory_cache.get("model", "op", text=text)
        assert result == i


def test_cache_empty_string(memory_cache: ModelOutputCache) -> None:
    """Test caching with empty string input."""
    memory_cache.set("model", "op", 42, text="")
    result = memory_cache.get("model", "op", text="")
    assert result == 42


def test_cache_large_numpy_array(memory_cache: ModelOutputCache) -> None:
    """Test caching large numpy array."""
    large_array = np.random.rand(1000, 768)
    memory_cache.set("model", "embedding", large_array, text="test")

    result = memory_cache.get("model", "embedding", text="test")
    np.testing.assert_array_equal(result, large_array)


def test_cache_serialize_for_hash_numpy(memory_cache: ModelOutputCache) -> None:
    """Test serialize_for_hash with numpy array."""
    arr = np.array([1, 2, 3])
    serialized = memory_cache._serialize_for_hash(arr)
    assert serialized == [1, 2, 3]


def test_cache_serialize_for_hash_nested_dict(memory_cache: ModelOutputCache) -> None:
    """Test serialize_for_hash with nested dict."""
    data = {"b": 2, "a": 1, "c": {"z": 3, "y": 2}}
    serialized = memory_cache._serialize_for_hash(data)

    # Should be sorted
    assert list(serialized.keys()) == ["a", "b", "c"]
    assert list(serialized["c"].keys()) == ["y", "z"]


def test_cache_serialize_result_numpy(memory_cache: ModelOutputCache) -> None:
    """Test serialize_result with numpy array."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    serialized = memory_cache._serialize_result(arr)

    assert serialized["__type__"] == "ndarray"
    assert serialized["data"] == [1.0, 2.0, 3.0]
    assert serialized["dtype"] == "float32"


def test_cache_deserialize_result_numpy(memory_cache: ModelOutputCache) -> None:
    """Test deserialize_result with numpy array."""
    serialized = {
        "__type__": "ndarray",
        "data": [1.0, 2.0, 3.0],
        "dtype": "float64",
    }
    result = memory_cache._deserialize_result(serialized)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    assert result.dtype == np.float64
