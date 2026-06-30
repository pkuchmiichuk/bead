"""Tests for TemplateCollection class."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import didactic.api as dx
import pandas as pd
import polars as pl
import pytest

from bead.data.base import BeadBaseModel
from bead.resources import Slot, Template, TemplateCollection

# ============================================================================
# Creation & Basic Operations (6 tests)
# ============================================================================


def test_creation_with_name() -> None:
    """Test collection creation with just a name."""
    collection = TemplateCollection(name="test")
    assert collection.name == "test"
    assert collection.description is None
    assert collection.language_code is None
    assert len(collection.templates) == 0
    assert len(collection.tags) == 0


def test_creation_with_all_fields() -> None:
    """Test collection creation with all fields."""
    collection = TemplateCollection(
        name="transitive",
        description="Transitive verb templates",
        language_code="en",
        tags=["transitive", "test"],
    )
    assert collection.name == "transitive"
    assert collection.description == "Transitive verb templates"
    assert collection.language_code == "en"
    assert collection.tags == ("transitive", "test")


def test_len_returns_correct_count(
    sample_template_collection: TemplateCollection,
) -> None:
    """Test that __len__ returns correct count."""
    assert len(sample_template_collection) == 2


def test_iter_iterates_over_templates(
    sample_template_collection: TemplateCollection,
) -> None:
    """Test that __iter__ iterates over templates."""
    templates = list(sample_template_collection)
    assert len(templates) == 2
    assert all(isinstance(t, Template) for t in templates)


def test_contains_checks_template_presence(
    sample_template_collection: TemplateCollection,
) -> None:
    """Test that __contains__ checks template presence."""
    template = list(sample_template_collection.templates)[0]
    assert template.id in sample_template_collection
    assert uuid4() not in sample_template_collection


def test_collection_inherits_from_bead_base_model() -> None:
    """Test that TemplateCollection inherits from BeadBaseModel."""
    collection = TemplateCollection(name="test")
    assert isinstance(collection, BeadBaseModel)
    assert hasattr(collection, "id")
    assert hasattr(collection, "created_at")
    assert hasattr(collection, "modified_at")


# ============================================================================
# CRUD Operations (6 tests)
# ============================================================================


def test_add_adds_template_successfully() -> None:
    """Test that add() adds a template successfully."""
    collection = TemplateCollection(name="test")
    slot = Slot(name="x")
    template = Template(name="test", template_string="{x}.", slots={"x": slot})

    collection = collection.with_template(template)
    assert len(collection) == 1
    assert template.id in collection


def test_add_raises_error_on_duplicate_id() -> None:
    """Test that add() raises error on duplicate ID."""
    collection = TemplateCollection(name="test")
    slot = Slot(name="x")
    template = Template(name="test", template_string="{x}.", slots={"x": slot})

    collection = collection.with_template(template)
    with pytest.raises((ValueError, dx.ValidationError), match="already exists"):
        collection = collection.with_template(template)


def test_add_many_adds_multiple_templates() -> None:
    """Test that add_many() adds multiple templates."""
    collection = TemplateCollection(name="test")

    templates = [
        Template(name="t1", template_string="{x}.", slots={"x": Slot(name="x")}),
        Template(name="t2", template_string="{y}.", slots={"y": Slot(name="y")}),
        Template(name="t3", template_string="{z}.", slots={"z": Slot(name="z")}),
    ]

    collection = collection.with_templates(templates)
    assert len(collection) == 3


def test_remove_removes_and_returns_template() -> None:
    """Test that remove() removes and returns template."""
    collection = TemplateCollection(name="test")
    template = Template(
        name="test", template_string="{x}.", slots={"x": Slot(name="x")}
    )
    collection = collection.with_template(template)
    collection, removed = collection.without_template(template.id)
    assert removed.name == "test"
    assert len(collection) == 0


def test_remove_raises_key_error_if_not_found() -> None:
    """Test that remove() raises KeyError if not found."""
    collection = TemplateCollection(name="test")
    with pytest.raises(KeyError, match="not found"):
        collection.without_template(uuid4())


def test_get_returns_template_if_exists() -> None:
    """Test that get() returns template if it exists."""
    collection = TemplateCollection(name="test")
    template = Template(
        name="test", template_string="{x}.", slots={"x": Slot(name="x")}
    )
    collection = collection.with_template(template)
    retrieved = collection.by_id(template.id)
    assert retrieved is not None
    assert retrieved.name == "test"


def test_get_returns_none_if_not_exists() -> None:
    """Test that get() returns None if template doesn't exist."""
    collection = TemplateCollection(name="test")
    assert collection.by_id(uuid4()) is None


# ============================================================================
# Filtering Operations (4 tests)
# ============================================================================


def test_filter_with_custom_predicate(
    sample_template_collection: TemplateCollection,
) -> None:
    """Test filter() with custom predicate."""
    # Filter for templates with more than 2 slots
    multi_slot = sample_template_collection.filter(lambda t: len(t.slots) > 2)
    assert len(multi_slot.templates) == 1  # simple_transitive has 3 slots


def test_filter_by_tag_filters_correctly() -> None:
    """Test that filter_by_tag() filters correctly."""
    collection = TemplateCollection(name="test")

    t1 = Template(
        name="t1",
        template_string="{x}.",
        slots={"x": Slot(name="x")},
        tags=["simple"],
    )
    t2 = Template(
        name="t2",
        template_string="{y}.",
        slots={"y": Slot(name="y")},
        tags=["complex"],
    )

    collection = collection.with_template(t1)
    collection = collection.with_template(t2)
    simple = collection.filter_by_tag("simple")
    assert len(simple.templates) == 1


def test_filter_by_slot_count() -> None:
    """Test that filter_by_slot_count() filters correctly."""
    collection = TemplateCollection(name="test")

    t1 = Template(name="t1", template_string="{x}.", slots={"x": Slot(name="x")})
    t2 = Template(
        name="t2",
        template_string="{y} {z}.",
        slots={"y": Slot(name="y"), "z": Slot(name="z")},
    )

    collection = collection.with_template(t1)
    collection = collection.with_template(t2)
    single_slot = collection.filter_by_slot_count(1)
    assert len(single_slot.templates) == 1


def test_filter_returns_new_collection_instance(
    sample_template_collection: TemplateCollection,
) -> None:
    """Test that filter returns new TemplateCollection instance."""
    filtered = sample_template_collection.filter(lambda t: True)
    assert isinstance(filtered, TemplateCollection)
    assert filtered is not sample_template_collection


def test_filter_preserves_collection_metadata(
    sample_template_collection: TemplateCollection,
) -> None:
    """Test that filter preserves collection metadata."""
    filtered = sample_template_collection.filter(lambda t: True)
    assert sample_template_collection.name in filtered.name
    assert filtered.description == sample_template_collection.description
    assert filtered.language_code == sample_template_collection.language_code


# ============================================================================
# Search Operations (2 tests)
# ============================================================================


def test_search_by_name() -> None:
    """Test that search() works by name."""
    collection = TemplateCollection(name="test")
    collection = collection.with_template(
        Template(name="transitive", template_string="{x}.", slots={"x": Slot(name="x")})
    )
    collection = collection.with_template(
        Template(
            name="intransitive", template_string="{y}.", slots={"y": Slot(name="y")}
        )
    )

    results = collection.search("trans", field="name")
    assert len(results.templates) == 2  # Both contain "trans"


def test_search_by_template_string() -> None:
    """Test that search() works by template_string."""
    collection = TemplateCollection(name="test")
    collection = collection.with_template(
        Template(
            name="question",
            template_string="Did {x} happen?",
            slots={"x": Slot(name="x")},
        )
    )
    collection = collection.with_template(
        Template(
            name="statement",
            template_string="{y} happened.",
            slots={"y": Slot(name="y")},
        )
    )

    results = collection.search("Did", field="template_string")
    assert len(results.templates) == 1


def test_search_invalid_field_raises_error() -> None:
    """Test that search() with invalid field raises error."""
    collection = TemplateCollection(name="test")
    with pytest.raises((ValueError, dx.ValidationError), match="Invalid field"):
        collection.search("test", field="invalid")


# ============================================================================
# Merging Operations (3 tests)
# ============================================================================


def test_merge_with_no_overlapping_ids() -> None:
    """Test merge() with no overlapping IDs."""
    c1 = TemplateCollection(name="c1")
    c1 = c1.with_template(
        Template(name="t1", template_string="{x}.", slots={"x": Slot(name="x")})
    )
    c2 = TemplateCollection(name="c2")
    c2 = c2.with_template(
        Template(name="t2", template_string="{y}.", slots={"y": Slot(name="y")})
    )
    merged = c1.merge(c2)
    assert len(merged.templates) == 2


def test_merge_with_error_strategy_raises_on_duplicates() -> None:
    """Test merge() with 'error' strategy raises on duplicates."""
    c1 = TemplateCollection(name="c1")
    template = Template(name="t1", template_string="{x}.", slots={"x": Slot(name="x")})
    c1 = c1.with_template(template)
    c2 = TemplateCollection(name="c2")
    # Add same template to c2
    c2 = c2.with_(templates=(template,))

    with pytest.raises(
        (ValueError, dx.ValidationError), match="Duplicate template IDs found"
    ):
        c1.merge(c2, strategy="error")


def test_merge_preserves_language_code() -> None:
    """Test that merge preserves language code."""
    c1 = TemplateCollection(name="c1", language_code="en")
    c2 = TemplateCollection(name="c2", language_code="es")

    merged = c1.merge(c2)
    assert merged.language_code == "en"  # From c1

    # Test when c1 has no language code
    c3 = TemplateCollection(name="c3")
    merged2 = c3.merge(c2)
    assert merged2.language_code == "es"  # From c2


# ============================================================================
# DataFrame Conversion (2 tests)
# ============================================================================


def test_to_dataframe_pandas_creates_correct_structure() -> None:
    """Test that to_dataframe() creates correct structure for pandas."""
    collection = TemplateCollection(name="test")
    collection = collection.with_template(
        Template(
            name="test",
            template_string="{x} {y}.",
            slots={"x": Slot(name="x"), "y": Slot(name="y")},
            tags=["test"],
        )
    )

    df = collection.to_dataframe(backend="pandas")
    assert isinstance(df, pd.DataFrame)
    assert "name" in df.columns
    assert "template_string" in df.columns
    assert "slot_count" in df.columns
    assert "slot_names" in df.columns
    assert "tags" in df.columns
    assert len(df) == 1


def test_to_dataframe_polars_creates_correct_structure() -> None:
    """Test that to_dataframe() creates correct structure for polars."""
    collection = TemplateCollection(name="test")
    collection = collection.with_template(
        Template(
            name="test",
            template_string="{x}.",
            slots={"x": Slot(name="x")},
            tags=["test"],
        )
    )

    df = collection.to_dataframe(backend="polars")
    assert isinstance(df, pl.DataFrame)
    assert "name" in df.columns
    assert "template_string" in df.columns
    assert len(df) == 1


def test_to_dataframe_empty_collection() -> None:
    """Test to_dataframe() with empty collection."""
    collection = TemplateCollection(name="empty")
    df = collection.to_dataframe()
    assert len(df) == 0
    assert "name" in df.columns


# ============================================================================
# Serialization (2 tests)
# ============================================================================


def test_to_jsonl_writes_file_correctly(tmp_path: Path) -> None:
    """Test that to_jsonl() writes file correctly."""
    collection = TemplateCollection(name="test")
    collection = collection.with_template(
        Template(name="t1", template_string="{x}.", slots={"x": Slot(name="x")})
    )
    collection = collection.with_template(
        Template(name="t2", template_string="{y}.", slots={"y": Slot(name="y")})
    )

    file_path = tmp_path / "test.jsonl"
    collection.to_jsonl(str(file_path))

    assert file_path.exists()
    lines = file_path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_from_jsonl_reads_file_correctly(tmp_path: Path) -> None:
    """Test that from_jsonl() reads file correctly."""
    # First write a file
    collection = TemplateCollection(name="test")
    collection = collection.with_template(
        Template(
            name="t1",
            template_string="{x}.",
            slots={"x": Slot(name="x")},
            tags=["simple"],
        )
    )

    file_path = tmp_path / "test.jsonl"
    collection.to_jsonl(str(file_path))

    # Then read it back
    loaded = TemplateCollection.from_jsonl(str(file_path), "loaded")
    assert len(loaded.templates) == 1


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """Test roundtrip (save and load)."""
    original = TemplateCollection(name="test")
    original = original.with_template(
        Template(
            name="transitive",
            template_string="{subject} {verb} {object}.",
            slots={
                "subject": Slot(name="subject"),
                "verb": Slot(name="verb"),
                "object": Slot(name="object"),
            },
            tags=["transitive"],
        )
    )

    file_path = tmp_path / "test.jsonl"
    original.to_jsonl(str(file_path))

    loaded = TemplateCollection.from_jsonl(str(file_path), "loaded")
    assert len(loaded.templates) == len(original.templates)

    # Check that data was preserved
    orig_template = list(original.templates)[0]
    loaded_template = list(loaded.templates)[0]
    assert loaded_template.name == orig_template.name
    assert loaded_template.template_string == orig_template.template_string
    assert len(loaded_template.slots) == len(orig_template.slots)
