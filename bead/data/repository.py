"""Repository pattern for didactic Models with optional caching.

A generic repository over a JSONLines file plus an optional in-memory cache.
Mutations to a stored Model produce a new instance (didactic Models are
frozen); the repository writes the new instance back via ``update``.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from bead.data.base import BeadBaseModel
from bead.data.serialization import (
    append_jsonlines,
    read_jsonlines,
    write_jsonlines,
)


class Repository[T: BeadBaseModel]:
    """Generic CRUD repository for didactic Models persisted as JSONLines.

    Parameters
    ----------
    model_class
        The didactic Model class this repository manages.
    storage_path
        Path to the JSONLines file for persistent storage.
    use_cache
        Whether to use the in-memory cache.
    """

    def __init__(
        self, model_class: type[T], storage_path: Path, use_cache: bool = True
    ) -> None:
        self.model_class = model_class
        self.storage_path = storage_path
        self.use_cache = use_cache
        self.cache: dict[UUID, T] = {}
        if self.use_cache and self.storage_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        objects = read_jsonlines(self.storage_path, self.model_class)
        self.cache = {obj.id: obj for obj in objects}

    def get(self, object_id: UUID) -> T | None:
        """Return the object with *object_id* if present, else ``None``."""
        if self.use_cache:
            return self.cache.get(object_id)
        if not self.storage_path.exists():
            return None
        for obj in read_jsonlines(self.storage_path, self.model_class):
            if obj.id == object_id:
                return obj
        return None

    def get_all(self) -> list[T]:
        """Return every object in the repository."""
        if self.use_cache:
            return list(self.cache.values())
        if not self.storage_path.exists():
            return []
        return read_jsonlines(self.storage_path, self.model_class)

    def add(self, obj: T) -> None:
        """Append *obj* to storage and update the cache."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonlines([obj], self.storage_path)
        if self.use_cache:
            self.cache[obj.id] = obj

    def add_many(self, objects: list[T]) -> None:
        """Append every object in *objects* to storage and update the cache."""
        if not objects:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonlines(objects, self.storage_path)
        if self.use_cache:
            for obj in objects:
                self.cache[obj.id] = obj

    def update(self, obj: T) -> None:
        """Replace the stored object with the same id by *obj*."""
        if self.use_cache:
            self.cache[obj.id] = obj
        objects = list(self.cache.values()) if self.use_cache else self.get_all()
        objects = [o if o.id != obj.id else obj for o in objects]
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines(objects, self.storage_path)

    def delete(self, object_id: UUID) -> None:
        """Remove the object with *object_id* from storage."""
        if self.use_cache:
            self.cache.pop(object_id, None)
        objects = list(self.cache.values()) if self.use_cache else self.get_all()
        objects = [o for o in objects if o.id != object_id]
        if objects:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonlines(objects, self.storage_path)
        elif self.storage_path.exists():
            self.storage_path.unlink()

    def exists(self, object_id: UUID) -> bool:
        """Return whether an object with *object_id* exists."""
        return self.get(object_id) is not None

    def count(self) -> int:
        """Return the number of objects in the repository."""
        if self.use_cache:
            return len(self.cache)
        if not self.storage_path.exists():
            return 0
        return len(read_jsonlines(self.storage_path, self.model_class))

    def clear(self) -> None:
        """Drop every object and delete the storage file."""
        self.cache.clear()
        if self.storage_path.exists():
            self.storage_path.unlink()

    def rebuild_cache(self) -> None:
        """Reload the cache from storage."""
        if not self.storage_path.exists():
            self.cache.clear()
        else:
            self._load_cache()
