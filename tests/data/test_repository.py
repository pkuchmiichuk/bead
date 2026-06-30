"""Tests for repository pattern implementation."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from bead.data.base import BeadBaseModel
from bead.data.repository import Repository
from bead.data.serialization import read_jsonlines, write_jsonlines


class SimpleModel(BeadBaseModel):
    """Simple test model for repository tests."""

    name: str
    value: int = 0


# Basic CRUD tests
def test_repository_creation(tmp_path: Path) -> None:
    """Test creating a repository."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](
        model_class=SimpleModel, storage_path=storage_path, use_cache=True
    )

    assert repo.model_class == SimpleModel
    assert repo.storage_path == storage_path
    assert repo.use_cache is True
    assert repo.cache == {}


def test_repository_add_object(tmp_path: Path) -> None:
    """Test adding a single object."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    obj = SimpleModel(name="test", value=42)
    repo.add(obj)

    assert repo.exists(obj.id)
    assert repo.count() == 1
    assert storage_path.exists()


def test_repository_get_object(tmp_path: Path) -> None:
    """Test retrieving an object by ID."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    obj = SimpleModel(name="test", value=42)
    repo.add(obj)

    loaded = repo.get(obj.id)
    assert loaded is not None
    assert loaded.id == obj.id
    assert loaded.name == obj.name
    assert loaded.value == obj.value


def test_repository_get_nonexistent(tmp_path: Path) -> None:
    """Test getting a nonexistent object returns None."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    result = repo.get(uuid4())
    assert result is None


def test_repository_get_all(tmp_path: Path) -> None:
    """Test getting all objects."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    obj1 = SimpleModel(name="test1", value=1)
    obj2 = SimpleModel(name="test2", value=2)
    obj3 = SimpleModel(name="test3", value=3)

    repo.add(obj1)
    repo.add(obj2)
    repo.add(obj3)

    all_objects = repo.get_all()
    assert len(all_objects) == 3
    assert obj1.id in [o.id for o in all_objects]
    assert obj2.id in [o.id for o in all_objects]
    assert obj3.id in [o.id for o in all_objects]


def test_repository_add_many(tmp_path: Path) -> None:
    """Test adding multiple objects at once."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    objects = [
        SimpleModel(name="test1", value=1),
        SimpleModel(name="test2", value=2),
        SimpleModel(name="test3", value=3),
    ]

    repo.add_many(objects)

    assert repo.count() == 3
    for obj in objects:
        assert repo.exists(obj.id)


def test_repository_update_object(tmp_path: Path) -> None:
    """Test updating an existing object."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    obj = SimpleModel(name="original", value=1)
    repo.add(obj)

    # Modify and update via .with_(); didactic Models are frozen
    obj = obj.with_(name="updated", value=99)
    repo.update(obj)

    # Verify changes persisted
    loaded = repo.get(obj.id)
    assert loaded is not None
    assert loaded.name == "updated"
    assert loaded.value == 99


def test_repository_delete_object(tmp_path: Path) -> None:
    """Test deleting an object."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    obj1 = SimpleModel(name="test1", value=1)
    obj2 = SimpleModel(name="test2", value=2)

    repo.add(obj1)
    repo.add(obj2)

    # Delete first object
    repo.delete(obj1.id)

    assert not repo.exists(obj1.id)
    assert repo.exists(obj2.id)
    assert repo.count() == 1


def test_repository_exists(tmp_path: Path) -> None:
    """Test checking if object exists."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    obj = SimpleModel(name="test", value=1)
    repo.add(obj)

    assert repo.exists(obj.id)
    assert not repo.exists(uuid4())


def test_repository_count(tmp_path: Path) -> None:
    """Test counting objects."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    assert repo.count() == 0

    repo.add(SimpleModel(name="test1", value=1))
    assert repo.count() == 1

    repo.add(SimpleModel(name="test2", value=2))
    assert repo.count() == 2


# Cache tests
def test_repository_with_cache(tmp_path: Path) -> None:
    """Test repository with caching enabled."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path, use_cache=True)

    obj = SimpleModel(name="test", value=1)
    repo.add(obj)

    # Object should be in cache
    assert obj.id in repo.cache
    assert repo.cache[obj.id].name == obj.name


def test_repository_without_cache(tmp_path: Path) -> None:
    """Test repository without caching."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path, use_cache=False)

    obj = SimpleModel(name="test", value=1)
    repo.add(obj)

    # Cache should be empty
    assert len(repo.cache) == 0

    # But object should still be retrievable
    loaded = repo.get(obj.id)
    assert loaded is not None
    assert loaded.name == obj.name


def test_repository_cache_populated_on_init(tmp_path: Path) -> None:
    """Test that cache is populated from existing file on init."""
    storage_path = tmp_path / "repo.jsonl"

    # Create some objects and write them to file
    objects = [
        SimpleModel(name="test1", value=1),
        SimpleModel(name="test2", value=2),
    ]
    write_jsonlines(objects, storage_path)

    # Create new repository with cache enabled
    repo = Repository[SimpleModel](SimpleModel, storage_path, use_cache=True)

    # Cache should be populated
    assert len(repo.cache) == 2
    assert objects[0].id in repo.cache
    assert objects[1].id in repo.cache


def test_repository_rebuild_cache(tmp_path: Path) -> None:
    """Test manually rebuilding cache."""
    storage_path = tmp_path / "repo.jsonl"

    # Create repository and add object
    repo = Repository[SimpleModel](SimpleModel, storage_path, use_cache=True)
    obj1 = SimpleModel(name="test1", value=1)
    repo.add(obj1)

    # Manually add object to file (simulating external modification)
    obj2 = SimpleModel(name="test2", value=2)
    write_jsonlines([obj1, obj2], storage_path)

    # Cache should not have obj2 yet
    assert obj2.id not in repo.cache

    # Rebuild cache
    repo.rebuild_cache()

    # Now cache should have both objects
    assert obj1.id in repo.cache
    assert obj2.id in repo.cache
    assert len(repo.cache) == 2


# File operations tests
def test_repository_creates_parent_directories(tmp_path: Path) -> None:
    """Test that repository creates parent directories."""
    storage_path = tmp_path / "nested" / "dirs" / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    obj = SimpleModel(name="test", value=1)
    repo.add(obj)

    # Parent directories should be created
    assert storage_path.parent.exists()
    assert storage_path.exists()


def test_repository_clear(tmp_path: Path) -> None:
    """Test clearing all objects."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    # Add objects
    repo.add(SimpleModel(name="test1", value=1))
    repo.add(SimpleModel(name="test2", value=2))

    assert repo.count() == 2
    assert storage_path.exists()

    # Clear
    repo.clear()

    assert repo.count() == 0
    assert not storage_path.exists()
    assert len(repo.cache) == 0


def test_repository_persistence(tmp_path: Path) -> None:
    """Test that objects persist across repository instances."""
    storage_path = tmp_path / "repo.jsonl"

    # Create first repository and add objects
    repo1 = Repository[SimpleModel](SimpleModel, storage_path)
    obj1 = SimpleModel(name="test1", value=1)
    obj2 = SimpleModel(name="test2", value=2)
    repo1.add(obj1)
    repo1.add(obj2)

    # Create second repository from same file
    repo2 = Repository[SimpleModel](SimpleModel, storage_path, use_cache=True)

    # Objects should be loaded
    assert repo2.count() == 2
    loaded1 = repo2.get(obj1.id)
    loaded2 = repo2.get(obj2.id)
    assert loaded1 is not None
    assert loaded2 is not None
    assert loaded1.name == obj1.name
    assert loaded2.name == obj2.name


# Edge case tests
def test_repository_empty_repository(tmp_path: Path) -> None:
    """Test operations on empty repository."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    assert repo.count() == 0
    assert repo.get_all() == []
    assert repo.get(uuid4()) is None
    assert not repo.exists(uuid4())


def test_repository_add_duplicate_id(tmp_path: Path) -> None:
    """Test adding object with same ID twice."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    obj = SimpleModel(name="original", value=1)
    repo.add(obj)

    # Modify and add again (should append, not replace)
    obj = obj.with_(name="modified")
    repo.add(obj)

    # Should have 2 objects with same ID in file
    objects = read_jsonlines(storage_path, SimpleModel)
    assert len(objects) == 2

    # Cache should have the latest version
    assert repo.cache[obj.id].name == "modified"


def test_repository_delete_nonexistent(tmp_path: Path) -> None:
    """Test deleting a nonexistent object doesn't error."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    # Add one object
    obj = SimpleModel(name="test", value=1)
    repo.add(obj)

    # Delete nonexistent ID (should not raise error)
    repo.delete(uuid4())

    # Original object should still exist
    assert repo.exists(obj.id)
    assert repo.count() == 1


def test_repository_add_many_empty_list(tmp_path: Path) -> None:
    """Test adding empty list of objects."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    repo.add_many([])

    assert repo.count() == 0
    assert not storage_path.exists()


def test_repository_delete_last_object(tmp_path: Path) -> None:
    """Test deleting the last object removes the file."""
    storage_path = tmp_path / "repo.jsonl"
    repo = Repository[SimpleModel](SimpleModel, storage_path)

    obj = SimpleModel(name="test", value=1)
    repo.add(obj)

    assert storage_path.exists()

    repo.delete(obj.id)

    assert not storage_path.exists()
    assert repo.count() == 0
