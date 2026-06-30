"""Tests for experimental list models."""

from __future__ import annotations

import time
from uuid import UUID, uuid4

import didactic.api as dx
import pytest
from didactic.api import ValidationError

from bead.lists import ExperimentList, ListCollection
from bead.lists.experiment_list import validate_presentation_order


class TestExperimentList:
    """Tests for ExperimentList model."""

    def test_create_empty_list(self, empty_experiment_list: ExperimentList) -> None:
        """Test creating empty list verifies defaults."""
        assert empty_experiment_list.name == "empty_list"
        assert empty_experiment_list.list_number == 0
        assert len(empty_experiment_list.item_refs) == 0
        assert len(empty_experiment_list.list_constraints) == 0
        assert len(empty_experiment_list.constraint_satisfaction) == 0
        assert empty_experiment_list.presentation_order is None
        assert len(empty_experiment_list.list_metadata) == 0
        assert len(empty_experiment_list.balance_metrics) == 0

    def test_create_with_name_and_number(self) -> None:
        """Test creating list with required fields."""
        exp_list = ExperimentList(name="test_list", list_number=5)
        assert exp_list.name == "test_list"
        assert exp_list.list_number == 5

    def test_name_validation_empty(self) -> None:
        """Test empty name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentList(name="", list_number=0)
        assert "name must be non-empty" in str(exc_info.value)

    def test_name_validation_whitespace(self) -> None:
        """Test whitespace-only name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentList(name="   ", list_number=0)
        assert "name must be non-empty" in str(exc_info.value)

    def test_name_strips_whitespace(self) -> None:
        """Test name whitespace is stripped."""
        exp_list = ExperimentList(name="  test  ", list_number=0)
        assert exp_list.name == "test"

    def test_list_number_validation_negative(self) -> None:
        """Test negative list_number raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentList(name="test", list_number=-1)
        assert "non-negative" in str(exc_info.value)

    def test_list_number_zero(self) -> None:
        """Test zero is valid list_number."""
        exp_list = ExperimentList(name="test", list_number=0)
        assert exp_list.list_number == 0

    def test_add_item(self, empty_experiment_list: ExperimentList) -> None:
        """Test adding item to list."""
        item_id = uuid4()
        original_modified = empty_experiment_list.modified_at

        empty_experiment_list = empty_experiment_list.with_item(item_id)
        assert item_id in empty_experiment_list.item_refs
        assert len(empty_experiment_list.item_refs) == 1
        assert empty_experiment_list.modified_at > original_modified

    def test_add_multiple_items(self, empty_experiment_list: ExperimentList) -> None:
        """Test adding multiple items."""
        item_ids = [uuid4() for _ in range(5)]

        for item_id in item_ids:
            empty_experiment_list = empty_experiment_list.with_item(item_id)
        assert len(empty_experiment_list.item_refs) == 5
        for item_id in item_ids:
            assert item_id in empty_experiment_list.item_refs

    def test_add_duplicate_item(self, empty_experiment_list: ExperimentList) -> None:
        """Test adding duplicate item is allowed (no validation)."""
        item_id = uuid4()

        empty_experiment_list = empty_experiment_list.with_item(item_id)
        empty_experiment_list = empty_experiment_list.with_item(item_id)
        assert len(empty_experiment_list.item_refs) == 2
        assert empty_experiment_list.item_refs.count(item_id) == 2

    def test_without_item(self, experiment_list_with_items: ExperimentList) -> None:
        """Test removing an item returns a new list with the item gone."""
        item_id = experiment_list_with_items.item_refs[0]
        original_length = len(experiment_list_with_items.item_refs)
        original_modified = experiment_list_with_items.modified_at

        smaller = experiment_list_with_items.without_item(item_id)

        assert item_id not in smaller.item_refs
        assert len(smaller.item_refs) == original_length - 1
        assert smaller.modified_at >= original_modified

    def test_without_nonexistent_item(
        self, empty_experiment_list: ExperimentList
    ) -> None:
        """Test removing a non-existent item raises ValueError."""
        item_id = uuid4()

        with pytest.raises((ValueError, dx.ValidationError)) as exc_info:
            empty_experiment_list.without_item(item_id)
        assert "not found in list" in str(exc_info.value)

    def test_shuffle_order_no_seed(
        self, experiment_list_with_items: ExperimentList
    ) -> None:
        """Test shuffling creates presentation_order."""
        experiment_list_with_items = experiment_list_with_items.with_shuffled_order()
        assert experiment_list_with_items.presentation_order is not None
        assert len(experiment_list_with_items.presentation_order) == len(
            experiment_list_with_items.item_refs
        )
        assert set(experiment_list_with_items.presentation_order) == set(
            experiment_list_with_items.item_refs
        )

    def test_shuffle_order_with_seed(
        self, experiment_list_with_items: ExperimentList
    ) -> None:
        """Test shuffling with seed is deterministic."""
        first = experiment_list_with_items.with_shuffled_order(seed=42)
        second = experiment_list_with_items.with_shuffled_order(seed=42)
        assert first.presentation_order == second.presentation_order

    def test_shuffle_order_multiple_times_same_seed(
        self, experiment_list_with_items: ExperimentList
    ) -> None:
        """Test multiple shuffles with same seed produce same result."""
        orders = [
            experiment_list_with_items.with_shuffled_order(seed=42).presentation_order
            for _ in range(3)
        ]
        assert orders[0] == orders[1] == orders[2]

    def test_get_presentation_order_default(
        self, experiment_list_with_items: ExperimentList
    ) -> None:
        """Test get_presentation_order returns item_refs when None."""
        assert experiment_list_with_items.presentation_order is None

        order = experiment_list_with_items.get_presentation_order()

        assert order == experiment_list_with_items.item_refs

    def test_get_presentation_order_custom(
        self, experiment_list_with_items: ExperimentList
    ) -> None:
        """Test get_presentation_order returns custom order when set."""
        experiment_list_with_items = experiment_list_with_items.with_shuffled_order(
            seed=42
        )
        custom_order = experiment_list_with_items.presentation_order

        order = experiment_list_with_items.get_presentation_order()

        assert order == custom_order
        assert order is not experiment_list_with_items.item_refs

    def test_presentation_order_validation_extra_uuids(
        self, sample_item_uuids: list[UUID]
    ) -> None:
        """Test presentation_order with extra UUIDs is flagged."""
        exp_list = ExperimentList(
            name="test",
            list_number=0,
            item_refs=tuple(sample_item_uuids[:3]),
            presentation_order=tuple(sample_item_uuids[:5]),
        )
        with pytest.raises((ValueError, dx.ValidationError), match="extra UUIDs"):
            validate_presentation_order(exp_list)

    def test_presentation_order_validation_missing_uuids(
        self, sample_item_uuids: list[UUID]
    ) -> None:
        """Test presentation_order with missing UUIDs is flagged."""
        exp_list = ExperimentList(
            name="test",
            list_number=0,
            item_refs=tuple(sample_item_uuids[:5]),
            presentation_order=tuple(sample_item_uuids[:3]),
        )
        with pytest.raises((ValueError, dx.ValidationError), match="missing UUIDs"):
            validate_presentation_order(exp_list)

    def test_presentation_order_validation_duplicates(
        self, sample_item_uuids: list[UUID]
    ) -> None:
        """Test presentation_order with duplicates is flagged."""
        item_refs = tuple(sample_item_uuids[:3])
        presentation_order = item_refs[:2] + (item_refs[0],)

        exp_list = ExperimentList(
            name="test",
            list_number=0,
            item_refs=item_refs,
            presentation_order=presentation_order,
        )
        with pytest.raises((ValueError, dx.ValidationError), match="duplicate UUIDs"):
            validate_presentation_order(exp_list)

    def test_serialization_roundtrip(
        self, experiment_list_with_items: ExperimentList
    ) -> None:
        """Test serialization roundtrip works."""
        data = experiment_list_with_items.model_dump()
        restored = ExperimentList(**data)

        assert restored.name == experiment_list_with_items.name
        assert restored.list_number == experiment_list_with_items.list_number
        assert restored.item_refs == experiment_list_with_items.item_refs

    def test_serialization_with_constraints(
        self, experiment_list_with_constraints: ExperimentList
    ) -> None:
        """Test serialization with constraints works."""
        data = experiment_list_with_constraints.model_dump()
        restored = ExperimentList(**data)

        assert len(restored.list_constraints) == len(
            experiment_list_with_constraints.list_constraints
        )

    def test_inherits_beadbasemodel(
        self, empty_experiment_list: ExperimentList
    ) -> None:
        """Test has BeadBaseModel fields."""
        assert hasattr(empty_experiment_list, "id")
        assert hasattr(empty_experiment_list, "created_at")
        assert hasattr(empty_experiment_list, "modified_at")
        assert hasattr(empty_experiment_list, "version")
        assert hasattr(empty_experiment_list, "metadata")

    def test_update_modified_time(self, empty_experiment_list: ExperimentList) -> None:
        """Test manual update_modified_time works."""
        original_modified = empty_experiment_list.modified_at

        time.sleep(0.01)
        empty_experiment_list = empty_experiment_list.touched()
        assert empty_experiment_list.modified_at > original_modified


class TestListCollection:
    """Tests for ListCollection model."""

    def test_create_empty_collection(self, sample_uuid: UUID) -> None:
        """Test creating empty collection."""
        collection = ListCollection(
            name="test_collection",
            source_items_id=sample_uuid,
            partitioning_strategy="balanced",
        )

        assert collection.name == "test_collection"
        assert collection.source_items_id == sample_uuid
        assert collection.partitioning_strategy == "balanced"
        assert len(collection.lists) == 0
        assert len(collection.partitioning_config) == 0
        assert len(collection.partitioning_stats) == 0

    def test_create_with_required_fields(self, sample_uuid: UUID) -> None:
        """Test creating with required fields."""
        collection = ListCollection(
            name="test",
            source_items_id=sample_uuid,
            partitioning_strategy="random",
        )

        assert collection.name == "test"
        assert collection.partitioning_strategy == "random"

    def test_name_validation_empty(self, sample_uuid: UUID) -> None:
        """Test empty name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ListCollection(
                name="", source_items_id=sample_uuid, partitioning_strategy="balanced"
            )
        assert "non-empty" in str(exc_info.value)

    def test_strategy_validation_empty(self, sample_uuid: UUID) -> None:
        """Test empty strategy raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ListCollection(
                name="test", source_items_id=sample_uuid, partitioning_strategy=""
            )
        assert "non-empty" in str(exc_info.value)

    def test_add_list(
        self, sample_uuid: UUID, experiment_list_with_items: ExperimentList
    ) -> None:
        """Test adding list to collection."""
        collection = ListCollection(
            name="test",
            source_items_id=sample_uuid,
            partitioning_strategy="balanced",
        )
        original_modified = collection.modified_at

        collection = collection.with_list(experiment_list_with_items)
        assert len(collection.lists) == 1
        assert collection.lists[0] == experiment_list_with_items
        assert collection.modified_at > original_modified

    def test_add_multiple_lists(self, sample_uuid: UUID) -> None:
        """Test adding multiple lists."""
        collection = ListCollection(
            name="test",
            source_items_id=sample_uuid,
            partitioning_strategy="balanced",
        )

        for i in range(3):
            exp_list = ExperimentList(name=f"list_{i}", list_number=i)
            collection = collection.with_list(exp_list)
        assert len(collection.lists) == 3

    def test_get_list_by_number_found(
        self, sample_list_collection: ListCollection
    ) -> None:
        """Test getting list by number when found."""
        found = sample_list_collection.get_list_by_number(1)

        assert found is not None
        assert found.list_number == 1

    def test_get_list_by_number_not_found(
        self, sample_list_collection: ListCollection
    ) -> None:
        """Test getting list by number when not found."""
        found = sample_list_collection.get_list_by_number(999)

        assert found is None

    def test_get_list_by_number_multiple(self, sample_uuid: UUID) -> None:
        """Test finding correct list among multiple."""
        collection = ListCollection(
            name="test",
            source_items_id=sample_uuid,
            partitioning_strategy="balanced",
        )

        for i in range(5):
            collection = collection.with_list(
                ExperimentList(name=f"list_{i}", list_number=i)
            )
        found = collection.get_list_by_number(3)

        assert found is not None
        assert found.name == "list_3"

    def test_lists_validation_unique_numbers(self, sample_uuid: UUID) -> None:
        """Test duplicate list_numbers raise ValidationError."""
        list1 = ExperimentList(name="list_0", list_number=0)
        list2 = ExperimentList(name="list_0_duplicate", list_number=0)

        with pytest.raises(ValidationError) as exc_info:
            ListCollection(
                name="test",
                source_items_id=sample_uuid,
                partitioning_strategy="balanced",
                lists=[list1, list2],
            )
        assert "Duplicate list_numbers" in str(exc_info.value)

    def test_get_all_item_refs(self, sample_list_collection: ListCollection) -> None:
        """Test getting all item refs across lists."""
        all_refs = sample_list_collection.get_all_item_refs()

        assert len(all_refs) == 20  # List has 20 items
        assert len(set(all_refs)) == 20  # All unique

    def test_get_all_item_refs_no_duplicates(self, sample_uuid: UUID) -> None:
        """Test get_all_item_refs deduplicates across lists."""
        collection = ListCollection(
            name="test",
            source_items_id=sample_uuid,
            partitioning_strategy="balanced",
        )

        shared_item = uuid4()
        list1 = ExperimentList(name="list_0", list_number=0)
        list1 = list1.with_item(shared_item)
        list1 = list1.with_item(uuid4())
        list2 = ExperimentList(name="list_1", list_number=1)
        list2 = list2.with_item(shared_item)  # Same item in both lists
        list2 = list2.with_item(uuid4())
        collection = collection.with_list(list1)
        collection = collection.with_list(list2)
        all_refs = collection.get_all_item_refs()

        assert len(all_refs) == 3  # 3 unique items total

    def test_validate_coverage_complete(
        self, sample_list_collection: ListCollection
    ) -> None:
        """Test validate_coverage with complete assignment."""
        all_items = set(sample_list_collection.lists[0].item_refs)
        result = sample_list_collection.validate_coverage(all_items)

        assert result["valid"] is True
        assert len(result["missing_items"]) == 0
        assert len(result["duplicate_items"]) == 0

    def test_validate_coverage_missing_items(self, sample_uuid: UUID) -> None:
        """Test validate_coverage reports missing items."""
        collection = ListCollection(
            name="test",
            source_items_id=sample_uuid,
            partitioning_strategy="balanced",
        )

        exp_list = ExperimentList(name="list_0", list_number=0)
        exp_list = exp_list.with_item(uuid4())
        collection = collection.with_list(exp_list)
        # Include extra items that aren't assigned
        all_items = set(exp_list.item_refs) | {uuid4(), uuid4()}
        result = collection.validate_coverage(all_items)

        assert result["valid"] is False
        assert len(result["missing_items"]) == 2

    def test_validate_coverage_extra_items(self, sample_uuid: UUID) -> None:
        """Test validate_coverage reports items assigned multiple times."""
        collection = ListCollection(
            name="test",
            source_items_id=sample_uuid,
            partitioning_strategy="balanced",
        )

        shared_item = uuid4()
        list1 = ExperimentList(name="list_0", list_number=0)
        list1 = list1.with_item(shared_item)
        list2 = ExperimentList(name="list_1", list_number=1)
        list2 = list2.with_item(shared_item)  # Duplicate assignment

        collection = collection.with_list(list1)
        collection = collection.with_list(list2)
        result = collection.validate_coverage({shared_item})

        assert result["valid"] is False
        assert len(result["duplicate_items"]) == 1
        assert shared_item in result["duplicate_items"]

    def test_serialization_roundtrip(
        self, sample_list_collection: ListCollection
    ) -> None:
        """Test serialization roundtrip with nested lists."""
        data = sample_list_collection.model_dump()
        restored = ListCollection(**data)

        assert restored.name == sample_list_collection.name
        assert restored.source_items_id == sample_list_collection.source_items_id
        assert len(restored.lists) == len(sample_list_collection.lists)

    def test_inherits_beadbasemodel(
        self, sample_list_collection: ListCollection
    ) -> None:
        """Test has BeadBaseModel fields."""
        assert hasattr(sample_list_collection, "id")
        assert hasattr(sample_list_collection, "created_at")
        assert hasattr(sample_list_collection, "modified_at")
        assert hasattr(sample_list_collection, "version")
        assert hasattr(sample_list_collection, "metadata")
