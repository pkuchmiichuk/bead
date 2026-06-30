"""Tests for the item-template and item models."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from didactic.api import ValidationError

from bead.items.item import (
    ConstraintSatisfaction,
    Item,
    ItemCollection,
    ModelOutput,
    UnfilledSlot,
)
from bead.items.item_template import (
    ChunkingSpec,
    ItemElement,
    ItemTemplate,
    ItemTemplateCollection,
    PresentationSpec,
    ScaleBounds,
    ScalePointLabel,
    TaskSpec,
    TimingParams,
)

# ItemElement tests


def test_item_element_text_creation(element_text: ItemElement) -> None:
    assert element_text.element_type == "text"
    assert element_text.element_name == "context"
    assert element_text.content == "Mary loves books."
    assert element_text.filled_template_ref_id is None
    assert element_text.order == 1


def test_item_element_template_ref_creation(element_template_ref: ItemElement) -> None:
    assert element_template_ref.element_type == "filled_template_ref"
    assert element_template_ref.element_name == "sentence"
    assert element_template_ref.content is None
    assert element_template_ref.filled_template_ref_id is not None


def test_item_element_is_text_property(element_text: ItemElement) -> None:
    assert element_text.is_text is True
    assert element_text.is_template_ref is False


def test_item_element_is_template_ref_property(
    element_template_ref: ItemElement,
) -> None:
    assert element_template_ref.is_template_ref is True
    assert element_template_ref.is_text is False


def test_item_element_name_validation_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ItemElement(element_type="text", element_name="", content="text")
    assert "Element name cannot be empty" in str(exc_info.value)


def test_item_element_name_validation_whitespace() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ItemElement(element_type="text", element_name="   ", content="text")
    assert "Element name cannot be empty" in str(exc_info.value)


def test_item_element_with_metadata() -> None:
    elem = ItemElement(
        element_type="text",
        element_name="test",
        content="text",
        element_metadata={"source": "corpus", "frequency": 0.05},
    )
    assert elem.element_metadata["source"] == "corpus"
    assert elem.element_metadata["frequency"] == 0.05


def test_item_element_serialization(element_text: ItemElement) -> None:
    data = element_text.model_dump()
    restored = ItemElement(**data)
    assert restored.element_name == element_text.element_name
    assert restored.content == element_text.content
    assert restored.element_type == element_text.element_type


# ChunkingSpec tests


def test_chunking_spec_word() -> None:
    spec = ChunkingSpec(unit="word")
    assert spec.unit == "word"
    assert spec.parse_type is None
    assert spec.constituent_labels is None


def test_chunking_spec_character() -> None:
    assert ChunkingSpec(unit="character").unit == "character"


def test_chunking_spec_sentence() -> None:
    assert ChunkingSpec(unit="sentence").unit == "sentence"


def test_chunking_spec_constituency() -> None:
    spec = ChunkingSpec(
        unit="constituent",
        parse_type="constituency",
        constituent_labels=("NP", "VP", "S"),
        parser="stanza",
        parse_language="en",
    )
    assert spec.unit == "constituent"
    assert spec.parse_type == "constituency"
    assert spec.constituent_labels == ("NP", "VP", "S")
    assert spec.parser == "stanza"
    assert spec.parse_language == "en"


def test_chunking_spec_dependency() -> None:
    spec = ChunkingSpec(
        unit="constituent",
        parse_type="dependency",
        constituent_labels=("nsubj", "dobj", "root"),
        parser="spacy",
        parse_language="en",
    )
    assert spec.parse_type == "dependency"
    assert spec.constituent_labels == ("nsubj", "dobj", "root")
    assert spec.parser == "spacy"


def test_chunking_spec_custom() -> None:
    spec = ChunkingSpec(unit="custom", custom_boundaries=(0, 3, 7, 10))
    assert spec.unit == "custom"
    assert spec.custom_boundaries == (0, 3, 7, 10)


def test_chunking_spec_serialization() -> None:
    spec = ChunkingSpec(
        unit="constituent",
        parse_type="constituency",
        constituent_labels=("NP",),
        parser="stanza",
        parse_language="en",
    )
    data = spec.model_dump()
    restored = ChunkingSpec(**data)
    assert restored.unit == spec.unit
    assert restored.parse_type == spec.parse_type
    assert restored.constituent_labels == spec.constituent_labels


# TimingParams tests


def test_timing_params_rsvp() -> None:
    params = TimingParams(duration_ms=250, isi_ms=50, cumulative=False, mask_char="_")
    assert params.duration_ms == 250
    assert params.isi_ms == 50
    assert params.cumulative is False
    assert params.mask_char == "_"


def test_timing_params_timeout() -> None:
    params = TimingParams(timeout_ms=5000, cumulative=True)
    assert params.timeout_ms == 5000
    assert params.cumulative is True


def test_timing_params_defaults() -> None:
    params = TimingParams()
    assert params.duration_ms is None
    assert params.isi_ms is None
    assert params.timeout_ms is None
    assert params.cumulative is True


def test_timing_params_serialization() -> None:
    params = TimingParams(duration_ms=300, isi_ms=100)
    restored = TimingParams(**params.model_dump())
    assert restored.duration_ms == params.duration_ms
    assert restored.isi_ms == params.isi_ms


# TaskSpec tests


def test_task_spec_binary() -> None:
    spec = TaskSpec(prompt="Is this sentence acceptable?")
    assert spec.prompt == "Is this sentence acceptable?"


def test_task_spec_ordinal_scale() -> None:
    spec = TaskSpec(
        prompt="How natural does this sentence sound?",
        scale_bounds=ScaleBounds(min=1, max=7),
        scale_labels=(
            ScalePointLabel(point=1, label="Very unnatural"),
            ScalePointLabel(point=7, label="Very natural"),
        ),
    )
    assert spec.scale_bounds is not None
    assert (spec.scale_bounds.min, spec.scale_bounds.max) == (1, 7)
    assert {(lbl.point, lbl.label) for lbl in spec.scale_labels} == {
        (1, "Very unnatural"),
        (7, "Very natural"),
    }


def test_task_spec_categorical() -> None:
    spec = TaskSpec(
        prompt="What is the relationship?",
        options=("Entailment", "Neutral", "Contradiction"),
    )
    assert spec.options == ("Entailment", "Neutral", "Contradiction")


def test_task_spec_forced_choice() -> None:
    spec = TaskSpec(
        prompt="Which sentence sounds more natural?",
        options=("sentence_a", "sentence_b"),
    )
    assert spec.options == ("sentence_a", "sentence_b")


def test_task_spec_free_text() -> None:
    spec = TaskSpec(
        prompt="Who performed the action?",
        max_length=50,
        text_validation_pattern=r"^[A-Za-z\s]+$",
    )
    assert spec.max_length == 50
    assert spec.text_validation_pattern == r"^[A-Za-z\s]+$"


def test_task_spec_prompt_validation_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TaskSpec(prompt="")
    assert "Prompt cannot be empty" in str(exc_info.value)


def test_task_spec_prompt_validation_whitespace() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TaskSpec(prompt="   ")
    assert "Prompt cannot be empty" in str(exc_info.value)


def test_task_spec_serialization() -> None:
    spec = TaskSpec(prompt="Rate this", scale_bounds=ScaleBounds(min=1, max=7))
    restored = TaskSpec.model_validate_json(spec.model_dump_json())
    assert restored.prompt == spec.prompt
    assert restored.scale_bounds is not None
    assert restored.scale_bounds.min == 1
    assert restored.scale_bounds.max == 7


# PresentationSpec tests


def test_presentation_spec_static() -> None:
    spec = PresentationSpec(mode="static")
    assert spec.mode == "static"
    assert spec.chunking.unit == "word"
    assert spec.timing.cumulative is True


def test_presentation_spec_self_paced_word() -> None:
    spec = PresentationSpec(mode="self_paced", chunking=ChunkingSpec(unit="word"))
    assert spec.mode == "self_paced"
    assert spec.chunking.unit == "word"


def test_presentation_spec_self_paced_constituent() -> None:
    spec = PresentationSpec(
        mode="self_paced",
        chunking=ChunkingSpec(
            unit="constituent",
            parse_type="constituency",
            constituent_labels=("NP",),
            parser="stanza",
            parse_language="en",
        ),
    )
    assert spec.chunking.unit == "constituent"
    assert spec.chunking.parse_type == "constituency"


def test_presentation_spec_timed_sequence() -> None:
    spec = PresentationSpec(
        mode="timed_sequence",
        chunking=ChunkingSpec(unit="word"),
        timing=TimingParams(duration_ms=250, isi_ms=50, cumulative=False),
    )
    assert spec.mode == "timed_sequence"
    assert spec.timing.duration_ms == 250


def test_presentation_spec_with_display_format() -> None:
    spec = PresentationSpec(
        mode="static",
        display_format={"font_size": 16, "color": "black"},
    )
    assert spec.display_format["font_size"] == 16
    assert spec.display_format["color"] == "black"


def test_presentation_spec_serialization() -> None:
    spec = PresentationSpec(mode="self_paced", chunking=ChunkingSpec(unit="word"))
    restored = PresentationSpec.model_validate_json(spec.model_dump_json())
    assert restored.mode == spec.mode
    assert restored.chunking.unit == "word"


# UnfilledSlot tests


def test_unfilled_slot_basic() -> None:
    slot = UnfilledSlot(slot_name="determiner", position=0, constraint_ids=())
    assert slot.slot_name == "determiner"
    assert slot.position == 0
    assert len(slot.constraint_ids) == 0


def test_unfilled_slot_with_constraints() -> None:
    constraint_id = uuid4()
    slot = UnfilledSlot(slot_name="verb", position=2, constraint_ids=(constraint_id,))
    assert slot.slot_name == "verb"
    assert len(slot.constraint_ids) == 1
    assert slot.constraint_ids[0] == constraint_id


def test_unfilled_slot_name_validation_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        UnfilledSlot(slot_name="", position=0)
    assert "Slot name cannot be empty" in str(exc_info.value)


def test_unfilled_slot_name_validation_whitespace() -> None:
    with pytest.raises(ValidationError) as exc_info:
        UnfilledSlot(slot_name="   ", position=0)
    assert "Slot name cannot be empty" in str(exc_info.value)


def test_unfilled_slot_serialization() -> None:
    constraint_id = uuid4()
    slot = UnfilledSlot(slot_name="verb", position=2, constraint_ids=(constraint_id,))
    restored = UnfilledSlot(**slot.model_dump())
    assert restored.slot_name == slot.slot_name
    assert restored.position == slot.position
    assert restored.constraint_ids == slot.constraint_ids


# ItemTemplate tests


def test_item_template_simple(item_template_simple: ItemTemplate) -> None:
    assert item_template_simple.name == "simple_rating"
    assert len(item_template_simple.elements) == 1
    assert item_template_simple.task_spec is not None
    assert item_template_simple.presentation_spec is not None


def test_item_template_complex(item_template_complex: ItemTemplate) -> None:
    assert item_template_complex.name == "context_target_rating"
    assert len(item_template_complex.elements) == 2
    assert item_template_complex.presentation_order == ("context", "sentence")


def test_item_template_name_validation_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ItemTemplate(
            name="",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Test prompt"),
            presentation_spec=PresentationSpec(mode="static"),
        )
    assert "Template name cannot be empty" in str(exc_info.value)


def test_item_template_unique_element_names() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            elements=(
                ItemElement(element_type="text", element_name="duplicate", content="1"),
                ItemElement(element_type="text", element_name="duplicate", content="2"),
            ),
            task_spec=TaskSpec(prompt="Test"),
            presentation_spec=PresentationSpec(mode="static"),
        )
    assert "Duplicate element names" in str(exc_info.value)
    assert "duplicate" in str(exc_info.value)


def test_item_template_with_constraints() -> None:
    constraint_uuid = uuid4()
    template = ItemTemplate(
        name="constrained",
        judgment_type="acceptability",
        task_type="binary",
        constraints=(constraint_uuid,),
        task_spec=TaskSpec(prompt="Test"),
        presentation_spec=PresentationSpec(mode="static"),
    )
    assert len(template.constraints) == 1
    assert template.constraints[0] == constraint_uuid


def test_item_template_serialization(item_template_simple: ItemTemplate) -> None:
    payload = item_template_simple.model_dump_json()
    restored = ItemTemplate.model_validate_json(payload)
    assert restored.name == item_template_simple.name
    assert len(restored.elements) == len(item_template_simple.elements)


def test_item_template_get_element_by_name(
    item_template_complex: ItemTemplate,
) -> None:
    elem = item_template_complex.get_element_by_name("context")
    assert elem is not None
    assert elem.element_name == "context"
    assert elem.element_type == "text"


def test_item_template_get_element_by_name_not_found(
    item_template_simple: ItemTemplate,
) -> None:
    assert item_template_simple.get_element_by_name("nonexistent") is None


def test_item_template_get_template_ref_elements(
    item_template_complex: ItemTemplate,
) -> None:
    refs = item_template_complex.get_template_ref_elements()
    assert len(refs) == 1
    assert refs[0].element_name == "sentence"
    assert refs[0].is_template_ref


def test_item_template_get_template_ref_elements_none(
    item_template_simple: ItemTemplate,
) -> None:
    assert len(item_template_simple.get_template_ref_elements()) == 0


def test_item_template_empty_elements_list() -> None:
    template = ItemTemplate(
        name="no_elements",
        judgment_type="acceptability",
        task_type="binary",
        elements=(),
        task_spec=TaskSpec(prompt="Test"),
        presentation_spec=PresentationSpec(mode="static"),
    )
    assert len(template.elements) == 0
    assert len(template.get_template_ref_elements()) == 0


# ItemTemplateCollection tests


def test_item_template_collection_creation(
    item_template_collection: ItemTemplateCollection,
) -> None:
    assert item_template_collection.name == "acceptability_templates"
    assert len(item_template_collection.templates) == 1


def test_item_template_collection_with_template(
    item_template_collection: ItemTemplateCollection,
    item_template_complex: ItemTemplate,
) -> None:
    initial_count = len(item_template_collection.templates)
    new_collection = item_template_collection.with_template(item_template_complex)
    assert len(new_collection.templates) == initial_count + 1


def test_item_template_collection_name_validation_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ItemTemplateCollection(name="")
    assert "Collection name cannot be empty" in str(exc_info.value)


def test_item_template_collection_serialization(
    item_template_collection: ItemTemplateCollection,
) -> None:
    data = item_template_collection.model_dump()
    restored = ItemTemplateCollection(**data)
    assert restored.name == item_template_collection.name
    assert len(restored.templates) == len(item_template_collection.templates)


# ModelOutput tests


def test_model_output_creation(model_output_sample: ModelOutput) -> None:
    assert model_output_sample.model_name == "gpt2"
    assert model_output_sample.model_version == "latest"
    assert model_output_sample.operation == "log_probability"
    assert model_output_sample.output == -12.456
    assert model_output_sample.cache_key == "abc123def456"


def test_model_output_string_field_validation_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ModelOutput(
            model_name="",
            model_version="latest",
            operation="test",
            inputs={},
            output=0,
            cache_key="key",
        )
    assert "Field cannot be empty" in str(exc_info.value)


def test_model_output_with_metadata(model_output_sample: ModelOutput) -> None:
    assert "device" in model_output_sample.computation_metadata
    assert "timestamp" in model_output_sample.computation_metadata


def test_model_output_serialization(model_output_sample: ModelOutput) -> None:
    restored = ModelOutput(**model_output_sample.model_dump())
    assert restored.model_name == model_output_sample.model_name
    assert restored.operation == model_output_sample.operation
    assert restored.output == model_output_sample.output


# Item tests


def test_item_simple_creation(item_simple: Item) -> None:
    assert item_simple.item_template_id is not None
    assert len(item_simple.filled_template_refs) == 0
    assert "sentence" in item_simple.rendered_elements
    assert item_simple.rendered_elements["sentence"] == "The cat broke the vase"


def test_item_with_unfilled_slots() -> None:
    template_id = uuid4()
    constraint_id = uuid4()
    item = Item(
        item_template_id=template_id,
        rendered_elements={"sentence": "The ___ cat ___ the ___"},
        unfilled_slots=(
            UnfilledSlot(
                slot_name="determiner",
                position=0,
                constraint_ids=(constraint_id,),
            ),
            UnfilledSlot(slot_name="verb", position=2, constraint_ids=()),
        ),
    )
    assert len(item.unfilled_slots) == 2
    assert item.unfilled_slots[0].slot_name == "determiner"
    assert item.unfilled_slots[1].slot_name == "verb"


def test_item_with_model_outputs(item_with_model_outputs: Item) -> None:
    assert len(item_with_model_outputs.model_outputs) == 1
    assert len(item_with_model_outputs.constraint_satisfaction) == 1


def test_item_get_model_output(item_with_model_outputs: Item) -> None:
    output = item_with_model_outputs.get_model_output("gpt2", "log_probability")
    assert output is not None
    assert output.model_name == "gpt2"
    assert output.operation == "log_probability"


def test_item_get_model_output_not_found(item_simple: Item) -> None:
    assert item_simple.get_model_output("nonexistent", "operation") is None


def test_item_get_model_output_with_input_filter(
    item_with_model_outputs: Item,
) -> None:
    output = item_with_model_outputs.get_model_output(
        "gpt2", "log_probability", inputs={"text": "The cat broke the vase"}
    )
    assert output is not None


def test_item_get_model_output_input_filter_no_match(
    item_with_model_outputs: Item,
) -> None:
    output = item_with_model_outputs.get_model_output(
        "gpt2", "log_probability", inputs={"text": "Different text"}
    )
    assert output is None


def test_item_with_model_output(
    item_simple: Item, model_output_sample: ModelOutput
) -> None:
    new_item = item_simple.with_model_output(model_output_sample)
    assert len(item_simple.model_outputs) == 0
    assert len(new_item.model_outputs) == 1


def test_item_serialization(item_simple: Item) -> None:
    data = item_simple.model_dump()
    restored = Item(**data)
    assert restored.item_template_id == item_simple.item_template_id
    assert restored.rendered_elements == item_simple.rendered_elements


def test_item_inherits_beadbasemodel(item_simple: Item) -> None:
    assert hasattr(item_simple, "id")
    assert hasattr(item_simple, "created_at")
    assert hasattr(item_simple, "modified_at")
    assert item_simple.id is not None


def test_item_empty_unfilled_slots(sample_uuid: UUID) -> None:
    item = Item(
        item_template_id=sample_uuid,
        rendered_elements={"sentence": "The cat broke the vase"},
        unfilled_slots=(),
    )
    assert len(item.unfilled_slots) == 0


def test_item_multiple_unfilled_slots(sample_uuid: UUID) -> None:
    item = Item(
        item_template_id=sample_uuid,
        rendered_elements={"sentence": "The ___ ___ ___ the ___"},
        unfilled_slots=(
            UnfilledSlot(slot_name="det", position=0),
            UnfilledSlot(slot_name="adj", position=1),
            UnfilledSlot(slot_name="verb", position=2),
            UnfilledSlot(slot_name="obj", position=4),
        ),
    )
    assert len(item.unfilled_slots) == 4


# ItemCollection tests


def test_item_collection_creation(item_collection: ItemCollection) -> None:
    assert item_collection.name == "test_items"
    assert len(item_collection.items) == 1
    assert item_collection.construction_stats["total_constructed"] == 1


def test_item_collection_with_item(
    item_collection: ItemCollection, item_simple: Item
) -> None:
    initial_count = len(item_collection.items)
    new_item = Item(
        item_template_id=uuid4(),
        rendered_elements={"test": "text"},
    )
    new_collection = item_collection.with_item(new_item)
    assert len(new_collection.items) == initial_count + 1


def test_item_collection_name_validation_empty() -> None:
    template_uuid = uuid4()
    filled_uuid = uuid4()
    with pytest.raises(ValidationError) as exc_info:
        ItemCollection(
            name="",
            source_template_collection_id=template_uuid,
            source_filled_collection_id=filled_uuid,
        )
    assert "Collection name cannot be empty" in str(exc_info.value)


def test_item_collection_serialization(item_collection: ItemCollection) -> None:
    data = item_collection.model_dump()
    restored = ItemCollection(**data)
    assert restored.name == item_collection.name
    assert len(restored.items) == len(item_collection.items)
    assert (
        restored.source_template_collection_id
        == item_collection.source_template_collection_id
    )


# Edge cases


def test_item_template_all_task_types() -> None:
    for task_type in (
        "forced_choice",
        "multi_select",
        "ordinal_scale",
        "magnitude",
        "binary",
        "categorical",
        "free_text",
        "cloze",
    ):
        template = ItemTemplate.model_validate(
            {
                "name": f"test_{task_type}",
                "judgment_type": "acceptability",
                "task_type": task_type,
                "task_spec": {"prompt": "Test prompt"},
                "presentation_spec": {"mode": "static"},
            }
        )
        assert template.task_type == task_type


def test_judgment_spec_all_types() -> None:
    for judgment_type in (
        "acceptability",
        "inference",
        "similarity",
        "plausibility",
        "comprehension",
        "preference",
    ):
        template = ItemTemplate.model_validate(
            {
                "name": f"test_{judgment_type}",
                "judgment_type": judgment_type,
                "task_type": "binary",
                "task_spec": {"prompt": "Test prompt"},
                "presentation_spec": {"mode": "static"},
            }
        )
        assert template.judgment_type == judgment_type


def test_nli_item_structure() -> None:
    template = ItemTemplate(
        name="nli_task",
        judgment_type="inference",
        task_type="categorical",
        elements=(
            ItemElement(
                element_type="text", element_name="premise", content="Mary loves books."
            ),
            ItemElement(
                element_type="text",
                element_name="hypothesis",
                content="Mary reads frequently.",
            ),
        ),
        task_spec=TaskSpec(
            prompt="What is the relationship between premise and hypothesis?",
            options=("Entailment", "Neutral", "Contradiction"),
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )
    assert len(template.elements) == 2
    assert template.get_element_by_name("premise") is not None
    assert template.get_element_by_name("hypothesis") is not None


def test_odd_man_out_item_structure() -> None:
    template = ItemTemplate(
        name="odd_man_out_task",
        judgment_type="similarity",
        task_type="forced_choice",
        elements=(
            ItemElement(element_type="text", element_name="option_a", content="Sent A"),
            ItemElement(element_type="text", element_name="option_b", content="Sent B"),
            ItemElement(element_type="text", element_name="option_c", content="Sent C"),
            ItemElement(element_type="text", element_name="option_d", content="Sent D"),
        ),
        task_spec=TaskSpec(
            prompt="Which sentence is different from the others?",
            options=("option_a", "option_b", "option_c", "option_d"),
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )
    assert len(template.elements) == 4


def test_cloze_item_ui_inference() -> None:
    template_id = uuid4()
    constraint_id_dropdown = uuid4()
    template = ItemTemplate(
        name="cloze_inference",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
        elements=(
            ItemElement(
                element_type="text",
                element_name="sentence",
                content="The ___ cat ___ the ___",
            ),
        ),
    )
    item = Item(
        item_template_id=template_id,
        rendered_elements={"sentence": "The ___ cat ___ the ___"},
        unfilled_slots=(
            UnfilledSlot(
                slot_name="determiner",
                position=0,
                constraint_ids=(constraint_id_dropdown,),
            ),
            UnfilledSlot(slot_name="verb", position=2, constraint_ids=()),
            UnfilledSlot(slot_name="object", position=4, constraint_ids=()),
        ),
    )
    assert template.task_type == "cloze"
    assert len(item.unfilled_slots) == 3


def test_item_empty_filled_template_refs(sample_uuid: UUID) -> None:
    item = Item(
        item_template_id=sample_uuid,
        filled_template_refs=(),
        rendered_elements={"text": "Static text"},
    )
    assert len(item.filled_template_refs) == 0


def test_item_multiple_model_outputs(sample_uuid: UUID) -> None:
    item = Item(
        item_template_id=sample_uuid,
        rendered_elements={"sentence": "Test"},
        model_outputs=(
            ModelOutput(
                model_name="gpt2",
                model_version="latest",
                operation="log_probability",
                inputs={"text": "Test"},
                output=-5.0,
                cache_key="key1",
            ),
            ModelOutput(
                model_name="bert",
                model_version="base",
                operation="embedding",
                inputs={"text": "Test"},
                output=[0.1, 0.2, 0.3],
                cache_key="key2",
            ),
        ),
    )
    assert len(item.model_outputs) == 2
    assert item.get_model_output("gpt2", "log_probability") is not None
    assert item.get_model_output("bert", "embedding") is not None


def test_item_constraint_satisfaction_records(sample_uuid: UUID) -> None:
    constraint_id = uuid4()
    item = Item(
        item_template_id=sample_uuid,
        rendered_elements={"sentence": "Test"},
        constraint_satisfaction=(
            ConstraintSatisfaction(constraint_id=constraint_id, satisfied=True),
        ),
    )
    assert len(item.constraint_satisfaction) == 1
    assert item.constraint_satisfaction[0].constraint_id == constraint_id
    assert item.constraint_satisfaction[0].satisfied is True
