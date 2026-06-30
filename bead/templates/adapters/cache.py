"""Content-addressable cache for model predictions.

This module implements caching for template filling model predictions
using SHA256-based content addressing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ModelOutputCache:
    """Content-addressable cache for model predictions.

    Uses SHA256 hashing to create deterministic cache keys based on:
    - Model name
    - Input text
    - Mask position
    - Top-K parameter

    Parameters
    ----------
    cache_dir : Path
        Directory for cache storage
    enabled : bool
        Enable/disable caching

    Examples
    --------
    >>> cache = ModelOutputCache(cache_dir=Path("/tmp/cache"), enabled=True)
    >>> key_args = ("bert-base-uncased", "The cat [MASK]", 2, 10)
    >>> predictions = cache.get(*key_args)
    >>> if predictions is None:
    ...     predictions = model.predict(...)
    ...     cache.set(*key_args, predictions)
    """

    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.cache_dir = cache_dir
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(
        self,
        model_name: str,
        input_text: str,
        mask_position: int,
        top_k: int,
    ) -> str:
        """Compute cache key from inputs.

        Parameters
        ----------
        model_name : str
            Model identifier
        input_text : str
            Input text with mask
        mask_position : int
            Position of mask token
        top_k : int
            Number of predictions

        Returns
        -------
        str
            SHA256 hex digest
        """
        # create deterministic key
        key_data = {
            "model_name": model_name,
            "input_text": input_text,
            "mask_position": mask_position,
            "top_k": top_k,
        }

        # serialize to JSON with sorted keys for determinism
        key_json = json.dumps(key_data, sort_keys=True)

        # hash with SHA256
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def get(
        self,
        model_name: str,
        input_text: str,
        mask_position: int,
        top_k: int,
    ) -> list[tuple[str, float]] | None:
        """Get cached predictions.

        Parameters
        ----------
        model_name : str
            Model identifier
        input_text : str
            Input text
        mask_position : int
            Mask position
        top_k : int
            Number of predictions

        Returns
        -------
        list[tuple[str, float]] | None
            Cached predictions or None if not found
        """
        if not self.enabled:
            return None

        cache_key = self._compute_key(model_name, input_text, mask_position, top_k)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)
                return [(item["token"], item["log_prob"]) for item in data]
        except json.JSONDecodeError, KeyError, OSError:
            # cache corruption; return None
            return None

    def set(
        self,
        model_name: str,
        input_text: str,
        mask_position: int,
        top_k: int,
        predictions: list[tuple[str, float]],
    ) -> None:
        """Store predictions in cache.

        Parameters
        ----------
        model_name : str
            Model identifier
        input_text : str
            Input text
        mask_position : int
            Mask position
        top_k : int
            Number of predictions
        predictions : list[tuple[str, float]]
            Predictions to cache
        """
        if not self.enabled:
            return

        cache_key = self._compute_key(model_name, input_text, mask_position, top_k)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # convert to serializable format
        data = [
            {"token": token, "log_prob": log_prob} for token, log_prob in predictions
        ]

        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            # silently fail on cache write errors
            pass

    def clear(self) -> None:
        """Clear all cached predictions."""
        if not self.enabled or not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except OSError:
                pass
