"""Content-addressable cache for judgment model outputs.

This module provides caching infrastructure for model outputs during item
construction. It supports multiple backends (filesystem, in-memory) and various
operation types including log probabilities, NLI scores, embeddings, and
similarity metrics.

Note: This cache is distinct from bead.templates.adapters.cache, which handles
MLM predictions for template filling. This module caches judgment model outputs
used in item construction.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends.

    Defines the interface that all cache backends must implement.
    """

    @abstractmethod
    def get(self, key: str) -> dict[str, object] | None:
        """Retrieve cache entry by key.

        Parameters
        ----------
        key
            Cache key to retrieve.

        Returns
        -------
        dict[str, object] | None
            Cache entry data if found, None otherwise.
        """
        pass

    @abstractmethod
    def set(self, key: str, data: dict[str, object]) -> None:
        """Store cache entry with key.

        Parameters
        ----------
        key
            Cache key.
        data
            Cache entry data to store.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete cache entry by key.

        Parameters
        ----------
        key
            Cache key to delete.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def keys(self) -> list[str]:
        """Return all cache keys.

        Returns
        -------
        list[str]
            List of all cache keys in the backend.
        """
        pass


class FilesystemBackend(CacheBackend):
    """Filesystem-based cache backend.

    Stores each cache entry as a separate JSON file with the cache key as
    the filename.

    Parameters
    ----------
    cache_dir : Path
        Directory for cache storage.

    Attributes
    ----------
    cache_dir : Path
        Directory where cache files are stored.

    Examples
    --------
    >>> from pathlib import Path
    >>> backend = FilesystemBackend(cache_dir=Path(".cache"))
    >>> backend.set("abc123", {"result": 42})
    >>> backend.get("abc123")
    {'result': 42}
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, object] | None:
        """Retrieve cache entry from filesystem.

        Parameters
        ----------
        key
            Cache key.

        Returns
        -------
        dict[str, object] | None
            Cache entry data if found, None otherwise.
        """
        cache_file = self.cache_dir / f"{key}.json"
        try:
            if cache_file.exists():
                with open(cache_file, encoding="utf-8") as f:
                    return json.load(f)
            return None
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            return None

    def set(self, key: str, data: dict[str, object]) -> None:
        """Store cache entry to filesystem.

        Parameters
        ----------
        key
            Cache key.
        data
            Cache entry data.
        """
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")

    def delete(self, key: str) -> None:
        """Delete cache entry from filesystem.

        Parameters
        ----------
        key
            Cache key to delete.
        """
        cache_file = self.cache_dir / f"{key}.json"
        try:
            if cache_file.exists():
                cache_file.unlink()
        except OSError as e:
            logger.warning(f"Failed to delete cache file {cache_file}: {e}")

    def clear(self) -> None:
        """Clear all cache entries from filesystem."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        except OSError as e:
            logger.warning(f"Failed to clear cache directory {self.cache_dir}: {e}")

    def keys(self) -> list[str]:
        """Return all cache keys from filesystem.

        Returns
        -------
        list[str]
            List of cache keys (filenames without .json extension).
        """
        try:
            return [f.stem for f in self.cache_dir.glob("*.json")]
        except OSError as e:
            logger.warning(f"Failed to list cache keys in {self.cache_dir}: {e}")
            return []


class InMemoryBackend(CacheBackend):
    """In-memory cache backend.

    Stores cache entries in a dictionary. No persistence across program runs.
    Useful for testing and temporary caching scenarios.

    Examples
    --------
    >>> backend = InMemoryBackend()
    >>> backend.set("xyz789", {"result": 3.14})
    >>> backend.get("xyz789")
    {'result': 3.14}
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, object]] = {}

    def get(self, key: str) -> dict[str, object] | None:
        """Retrieve cache entry from memory.

        Parameters
        ----------
        key
            Cache key.

        Returns
        -------
        dict[str, object] | None
            Cache entry data if found, None otherwise.
        """
        return self._cache.get(key)

    def set(self, key: str, data: dict[str, object]) -> None:
        """Store cache entry in memory.

        Parameters
        ----------
        key
            Cache key.
        data
            Cache entry data.
        """
        self._cache[key] = data

    def delete(self, key: str) -> None:
        """Delete cache entry from memory.

        Parameters
        ----------
        key
            Cache key to delete.
        """
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries from memory."""
        self._cache.clear()

    def keys(self) -> list[str]:
        """Return all cache keys from memory.

        Returns
        -------
        list[str]
            List of cache keys.
        """
        return list(self._cache.keys())


class ModelOutputCache:
    """Content-addressable cache for judgment model outputs.

    Caches results from various model operations to avoid redundant computation.
    Supports multiple operation types including log probabilities, perplexity,
    NLI scores, embeddings, and similarity metrics.

    Cache keys are automatically generated using SHA-256 hashing of the model
    name, operation type, and all input parameters, ensuring deterministic
    cache hits for identical inputs.

    Parameters
    ----------
    cache_dir : Path | None
        Directory for cache files (filesystem backend only).
        Defaults to ~/.cache/bead/models if not specified.
    backend : {"filesystem", "memory"}
        Cache backend type. "filesystem" persists across runs,
        "memory" is ephemeral.
    enabled : bool
        Whether caching is enabled.

    Attributes
    ----------
    enabled : bool
        Whether caching is enabled. When False, all operations are no-ops.

    Examples
    --------
    Basic usage with filesystem backend:

    >>> from pathlib import Path
    >>> cache = ModelOutputCache(cache_dir=Path(".cache"))
    >>> result = cache.get("gpt2", "log_probability", text="Hello world")
    >>> if result is None:
    ...     result = -2.5
    ...     cache.set("gpt2", "log_probability", result, text="Hello world")

    Caching NLI scores:

    >>> nli_scores = cache.get("roberta-nli", "nli",
    ...                        premise="Mary loves books",
    ...                        hypothesis="Mary enjoys reading")
    >>> if nli_scores is None:
    ...     nli_scores = {"entailment": 0.9, "neutral": 0.08, "contradiction": 0.02}
    ...     cache.set("roberta-nli", "nli", nli_scores,
    ...              premise="Mary loves books", hypothesis="Mary enjoys reading")

    Caching embeddings:

    >>> import numpy as np
    >>> embedding = cache.get("bert-base", "embedding", text="Hello")
    >>> if embedding is None:
    ...     embedding = np.random.rand(768)
    ...     cache.set("bert-base", "embedding", embedding, text="Hello")
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        backend: Literal["filesystem", "memory"] = "filesystem",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled

        if backend == "filesystem":
            if cache_dir is None:
                cache_dir = Path.home() / ".cache" / "bead" / "models"
            self._backend: CacheBackend = FilesystemBackend(cache_dir)
        elif backend == "memory":
            self._backend = InMemoryBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def generate_cache_key(
        self, model_name: str, operation: str, **inputs: str | int | float | bool | None
    ) -> str:
        """Generate deterministic cache key from inputs.

        Parameters
        ----------
        model_name
            Model identifier.
        operation
            Operation type (e.g., "log_probability", "embedding").
        **inputs
            Input parameters for the operation (text, premise, hypothesis).

        Returns
        -------
        str
            SHA-256 hex digest as cache key.
        """
        # create deterministic dict with sorted keys
        key_data = {
            "model_name": model_name,
            "operation": operation,
            "inputs": self._serialize_for_hash(inputs),
        }

        # json with sorted keys for determinism
        key_json = json.dumps(key_data, sort_keys=True)

        # sha-256 hash
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def _serialize_for_hash(self, obj: object) -> object:
        """Serialize object for deterministic hashing.

        Converts numpy arrays to lists and sorts dict keys.

        Parameters
        ----------
        obj
            Object to serialize. Accepts numpy arrays, dicts, lists, tuples,
            and primitive types.

        Returns
        -------
        object
            JSON-serializable version of the object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._serialize_for_hash(v) for k, v in sorted(obj.items())}  # type: ignore[misc]
        elif isinstance(obj, list | tuple):
            return [self._serialize_for_hash(item) for item in obj]  # type: ignore[misc]
        else:
            return obj

    def _serialize_result(self, result: object) -> object:
        """Serialize result for storage.

        Parameters
        ----------
        result
            Result to serialize. Accepts numpy arrays, dicts, lists, tuples,
            and primitive types.

        Returns
        -------
        object
            JSON-serializable version of result.
        """
        if isinstance(result, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": result.tolist(),
                "dtype": str(result.dtype),  # type: ignore[arg-type]
            }
        elif isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in result.items()}  # type: ignore[misc]
        elif isinstance(result, list | tuple):
            return [self._serialize_result(item) for item in result]  # type: ignore[misc]
        else:
            return result

    def _deserialize_result(self, result: Any) -> Any:
        """Deserialize result from storage.

        Parameters
        ----------
        result
            Serialized result from cache storage.

        Returns
        -------
        Any
            Deserialized result with numpy arrays restored.
        """
        if isinstance(result, dict):
            if result.get("__type__") == "ndarray":  # type: ignore[union-attr]
                return np.array(result["data"], dtype=result["dtype"])  # type: ignore[arg-type]
            else:
                return {k: self._deserialize_result(v) for k, v in result.items()}  # type: ignore[misc]
        elif isinstance(result, list):
            return [self._deserialize_result(item) for item in result]  # type: ignore[misc]
        else:
            return result

    def get(
        self, model_name: str, operation: str, **inputs: str | int | float | bool | None
    ) -> Any:
        """Retrieve cached result.

        Parameters
        ----------
        model_name
            Model identifier.
        operation
            Operation type (e.g., "log_probability", "nli", "embedding").
        **inputs
            Input parameters for the operation (text, premise, hypothesis).

        Returns
        -------
        Any
            Cached result if found, None otherwise.
        """
        if not self.enabled:
            return None

        cache_key = self.generate_cache_key(model_name, operation, **inputs)
        entry = self._backend.get(cache_key)

        if entry is None:
            return None

        # deserialize and return result
        return self._deserialize_result(entry["result"])

    def set(
        self,
        model_name: str,
        operation: str,
        result: str | float | dict[str, float] | list[float] | np.ndarray,
        model_version: str | None = None,
        **inputs: str | int | float | bool | None,
    ) -> None:
        """Store result in cache.

        Parameters
        ----------
        model_name
            Model identifier.
        operation
            Operation type (e.g., "log_probability", "nli", "embedding",
            "lm_completion").
        result
            Result to cache. Strings (LM completions), floats (log
            probabilities), float dicts (NLI scores), float lists,
            and numpy arrays (embeddings) are supported.
        model_version
            Optional model version string for tracking.
        **inputs
            Input parameters for the operation (text, premise, hypothesis).
        """
        if not self.enabled:
            return

        cache_key = self.generate_cache_key(model_name, operation, **inputs)

        # create cache entry with metadata
        entry = {
            "cache_key": cache_key,
            "timestamp": datetime.now(UTC).isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "operation": operation,
            "inputs": self._serialize_for_hash(inputs),
            "result": self._serialize_result(result),
        }

        self._backend.set(cache_key, entry)

    def invalidate(
        self, model_name: str, operation: str, **inputs: str | int | float | bool | None
    ) -> None:
        """Invalidate specific cache entry.

        Parameters
        ----------
        model_name
            Model identifier.
        operation
            Operation type.
        **inputs
            Input parameters for the operation.
        """
        cache_key = self.generate_cache_key(model_name, operation, **inputs)
        self._backend.delete(cache_key)

    def clear_model(self, model_name: str) -> None:
        """Clear all cache entries for a specific model.

        Parameters
        ----------
        model_name : str
            Model identifier.
        """
        # get all keys and filter by model name
        keys_to_delete: list[str] = []
        for key in self._backend.keys():
            entry = self._backend.get(key)
            if entry and entry.get("model_name") == model_name:
                keys_to_delete.append(key)

        # delete matching entries
        for key in keys_to_delete:
            self._backend.delete(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._backend.clear()
