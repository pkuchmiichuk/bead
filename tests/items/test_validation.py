"""Tests for item validation utilities."""
# ruff: noqa: PLC0415

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest
from didactic.api import ValidationError

from bead.items.item import ConstraintSatisfaction, Item, ModelOutput
from bead.items.item_template import (
    ItemElement,
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.items.validation import (
    _check_option_keys,
    _check_options,
    get_task_type_requirements,
    infer_task_type_from_item,
    item_passes_all_constraints,
    validate_constraint_satisfaction,
    validate_item,
    validate_item_for_task_type,
    validate_metadata_completeness,
    validate_model_output,
)


@pytest.fixture
def simple_template():
    """Create a simple item template for testing."""
    return ItemTemplate(
        name="test_template",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this natural?"),
        presentation_spec=PresentationSpec(mode="static"),
        elements=[
            ItemElement(
                element_type="text", element_name="sentence", content="Test sentence"
            )
        ],
        constraints=[uuid4(), uuid4()],
    )


@pytest.fixture
def simple_item(simple_template):
    """Create a simple item for testing."""
    return Item(
        item_template_id=simple_template.id,
        rendered_elements={"sentence": "Test sentence"},
        constraint_satisfaction=(
            ConstraintSatisfaction(
                constraint_id=simple_template.constraints[0], satisfied=True
            ),
            ConstraintSatisfaction(
                constraint_id=simple_template.constraints[1], satisfied=True
            ),
        ),
        model_outputs=(),
    )


class TestValidateItem:
    """Tests for validate_item function."""

    def test_valid_item(self, simple_item, simple_template) -> None:
        """Test validation of a valid item."""
        errors = validate_item(simple_item, simple_template)
        assert errors == []

    def test_template_id_mismatch(self, simple_item, simple_template) -> None:
        """Test detection of template ID mismatch."""
        simple_item = simple_item.with_(item_template_id=uuid4())
        errors = validate_item(simple_item, simple_template)
        assert len(errors) == 1
        assert "template ID mismatch" in errors[0]

    def test_missing_rendered_elements(self, simple_item, simple_template) -> None:
        """Test detection of missing rendered elements."""
        simple_item = simple_item.with_(rendered_elements={})
        errors = validate_item(simple_item, simple_template)
        assert any("Missing rendered elements" in e for e in errors)

    def test_extra_rendered_elements(self, simple_item, simple_template) -> None:
        """Test detection of extra rendered elements."""
        rendered = dict(simple_item.rendered_elements)
        rendered["extra"] = "Extra element"
        simple_item = simple_item.with_(rendered_elements=rendered)
        errors = validate_item(simple_item, simple_template)
        assert any("Extra rendered elements" in e for e in errors)

    def test_missing_constraint_evaluation(self, simple_item, simple_template) -> None:
        """Test detection of missing constraint evaluations."""
        simple_item = simple_item.with_(constraint_satisfaction=())
        errors = validate_item(simple_item, simple_template)
        assert any("Missing constraint evaluations" in e for e in errors)

    def test_invalid_model_output(self, simple_item, simple_template) -> None:
        """Test that invalid model outputs are detected."""
        simple_item = simple_item.with_(
            model_outputs=(
                ModelOutput(
                    model_name="test",
                    model_version="1.0",
                    operation="log_probability",
                    inputs={"text": "test"},
                    output="not a number",
                    cache_key="abc123",
                ),
            )
        )
        errors = validate_item(simple_item, simple_template)
        assert any("should be numeric" in e for e in errors)


class TestValidateModelOutput:
    """Tests for validate_model_output function."""

    def test_valid_log_probability_output(self) -> None:
        """Test validation of valid log probability output."""
        output = ModelOutput(
            model_name="gpt2",
            model_version="1.0",
            operation="log_probability",
            inputs={"text": "test"},
            output=-42.5,
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert errors == []

    def test_valid_nli_output(self) -> None:
        """Test validation of valid NLI output."""
        output = ModelOutput(
            model_name="roberta-nli",
            model_version="1.0",
            operation="nli",
            inputs={"premise": "p", "hypothesis": "h"},
            output={"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
            cache_key="xyz789",
        )
        errors = validate_model_output(output)
        assert errors == []

    def test_empty_model_name(self) -> None:
        """Test Pydantic validation prevents empty model name."""
        with pytest.raises(ValidationError):
            ModelOutput(
                model_name="",  # Pydantic will reject this
                model_version="1.0",
                operation="log_probability",
                inputs={"text": "test"},
                output=-42.0,
                cache_key="abc123",
            )

    def test_empty_operation(self) -> None:
        """Test Pydantic validation prevents empty operation."""
        with pytest.raises(ValidationError):
            ModelOutput(
                model_name="gpt2",
                model_version="1.0",
                operation="",  # Pydantic will reject this
                inputs={"text": "test"},
                output=-42.0,
                cache_key="abc123",
            )

    def test_empty_cache_key(self) -> None:
        """Test Pydantic validation prevents empty cache key."""
        with pytest.raises(ValidationError):
            ModelOutput(
                model_name="gpt2",
                model_version="1.0",
                operation="log_probability",
                inputs={"text": "test"},
                output=-42.0,
                cache_key="",  # Pydantic will reject this
            )

    def test_nli_output_not_dict(self) -> None:
        """Test detection of NLI output that's not a dict."""
        output = ModelOutput(
            model_name="roberta-nli",
            model_version="1.0",
            operation="nli",
            inputs={"premise": "p", "hypothesis": "h"},
            output=0.8,  # Should be dict
            cache_key="xyz789",
        )
        errors = validate_model_output(output)
        assert any("should be dict" in e for e in errors)

    def test_nli_output_missing_keys(self) -> None:
        """Test detection of NLI output with missing keys."""
        output = ModelOutput(
            model_name="roberta-nli",
            model_version="1.0",
            operation="nli",
            inputs={"premise": "p", "hypothesis": "h"},
            output={"entailment": 0.8},  # Missing neutral and contradiction
            cache_key="xyz789",
        )
        errors = validate_model_output(output)
        assert any("keys mismatch" in e for e in errors)

    def test_log_probability_non_numeric(self) -> None:
        """Test detection of non-numeric log probability."""
        output = ModelOutput(
            model_name="gpt2",
            model_version="1.0",
            operation="log_probability",
            inputs={"text": "test"},
            output="not a number",
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert any("should be numeric" in e for e in errors)

    def test_perplexity_non_numeric(self) -> None:
        """Test detection of non-numeric perplexity."""
        output = ModelOutput(
            model_name="gpt2",
            model_version="1.0",
            operation="perplexity",
            inputs={"text": "test"},
            output=[1, 2, 3],
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert any("should be numeric" in e for e in errors)

    def test_similarity_non_numeric(self) -> None:
        """Test detection of non-numeric similarity."""
        output = ModelOutput(
            model_name="sentence-transformer",
            model_version="1.0",
            operation="similarity",
            inputs={"text1": "a", "text2": "b"},
            output=None,
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert any("should be numeric" in e for e in errors)

    def test_embedding_non_list(self) -> None:
        """Test detection of embedding that's not a list."""
        output = ModelOutput(
            model_name="sentence-transformer",
            model_version="1.0",
            operation="embedding",
            inputs={"text": "test"},
            output=42,  # Should be list or dict (serialized array)
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert any("should be list/array" in e for e in errors)


class TestValidateConstraintSatisfaction:
    """Tests for validate_constraint_satisfaction function."""

    def test_valid_constraint_satisfaction(self, simple_item, simple_template) -> None:
        """Test validation of valid constraint satisfaction."""
        errors = validate_constraint_satisfaction(simple_item, simple_template)
        assert errors == []

    def test_missing_constraint(self, simple_item, simple_template) -> None:
        """Test detection of missing constraint evaluation."""
        simple_item = simple_item.with_(
            constraint_satisfaction=(
                ConstraintSatisfaction(
                    constraint_id=simple_template.constraints[0], satisfied=True
                ),
            )
        )
        errors = validate_constraint_satisfaction(simple_item, simple_template)
        assert len(errors) == 1
        assert "not evaluated" in errors[0]

    def test_non_boolean_value_rejected_at_construction(self, simple_template) -> None:
        """Non-bool ``satisfied`` is rejected by the model constructor."""
        with pytest.raises((dx.ValidationError, AssertionError)):
            ConstraintSatisfaction(
                constraint_id=simple_template.constraints[0], satisfied="true"
            )

    def test_all_constraints_missing(self, simple_item, simple_template) -> None:
        """Test when all constraints are missing."""
        simple_item = simple_item.with_(constraint_satisfaction=())
        errors = validate_constraint_satisfaction(simple_item, simple_template)
        assert len(errors) == len(simple_template.constraints)


class TestValidateMetadataCompleteness:
    """Tests for validate_metadata_completeness function."""

    def test_valid_metadata(self, simple_item) -> None:
        """Test validation of item with complete metadata."""
        errors = validate_metadata_completeness(simple_item)
        # Should have id, created_at, modified_at from BeadBaseModel
        assert errors == []

    def test_item_has_id(self, simple_item) -> None:
        """Test that item has id field."""
        assert hasattr(simple_item, "id")
        assert simple_item.id is not None

    def test_item_has_timestamps(self, simple_item) -> None:
        """Test that item has timestamp fields."""
        assert hasattr(simple_item, "created_at")
        assert hasattr(simple_item, "modified_at")
        assert simple_item.created_at is not None
        assert simple_item.modified_at is not None


class TestItemPassesAllConstraints:
    """Tests for item_passes_all_constraints function."""

    def test_all_constraints_pass(self, simple_item) -> None:
        """Test when all constraints are satisfied."""
        assert item_passes_all_constraints(simple_item) is True

    def test_one_constraint_fails(self, simple_item, simple_template) -> None:
        """Test when one constraint fails."""
        simple_item = simple_item.with_(
            constraint_satisfaction=(
                ConstraintSatisfaction(
                    constraint_id=simple_template.constraints[0], satisfied=False
                ),
                ConstraintSatisfaction(
                    constraint_id=simple_template.constraints[1], satisfied=True
                ),
            )
        )
        assert item_passes_all_constraints(simple_item) is False

    def test_all_constraints_fail(self, simple_item, simple_template) -> None:
        """Test when all constraints fail."""
        simple_item = simple_item.with_(
            constraint_satisfaction=tuple(
                ConstraintSatisfaction(constraint_id=cid, satisfied=False)
                for cid in simple_template.constraints
            )
        )
        assert item_passes_all_constraints(simple_item) is False

    def test_no_constraints(self) -> None:
        """Test item with no constraints."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"test": "text"},
            constraint_satisfaction=(),
        )
        assert item_passes_all_constraints(item) is True

    def test_mixed_constraints(self, simple_item, simple_template) -> None:
        """Test with mixed constraint satisfaction."""
        simple_item = simple_item.with_(
            constraint_satisfaction=(
                ConstraintSatisfaction(
                    constraint_id=simple_template.constraints[0], satisfied=True
                ),
                ConstraintSatisfaction(
                    constraint_id=simple_template.constraints[1], satisfied=False
                ),
            )
        )
        assert item_passes_all_constraints(simple_item) is False


class TestCheckOptions:
    """Tests for _check_options helper function."""

    def test_valid_two_options(self) -> None:
        """Test with valid two options."""
        item = Item(
            item_template_id=uuid4(),
            options=["A", "B"],
        )
        has_options, n_options = _check_options(item)
        assert has_options is True
        assert n_options == 2

    def test_valid_three_options(self) -> None:
        """Test with valid three options."""
        item = Item(
            item_template_id=uuid4(),
            options=["A", "B", "C"],
        )
        has_options, n_options = _check_options(item)
        assert has_options is True
        assert n_options == 3

    def test_no_options(self) -> None:
        """Test with empty options list."""
        item = Item(
            item_template_id=uuid4(),
            options=[],
        )
        has_options, n_options = _check_options(item)
        assert has_options is False
        assert n_options == 0

    def test_only_one_option(self) -> None:
        """Test with only one option (not enough)."""
        item = Item(
            item_template_id=uuid4(),
            options=["A"],
        )
        has_options, n_options = _check_options(item)
        assert has_options is False
        assert n_options == 0

    def test_default_empty_options(self) -> None:
        """Test item without explicitly setting options field."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Hello"},
        )
        has_options, n_options = _check_options(item)
        assert has_options is False
        assert n_options == 0


class TestCheckOptionKeys:
    """Tests for _check_option_keys helper function (legacy, deprecated)."""

    def test_valid_two_options(self) -> None:
        """Test with valid two options."""
        rendered = {"option_a": "A", "option_b": "B"}
        has_options, n_options = _check_option_keys(rendered)
        assert has_options is True
        assert n_options == 2

    def test_valid_three_options(self) -> None:
        """Test with valid three options."""
        rendered = {"option_a": "A", "option_b": "B", "option_c": "C"}
        has_options, n_options = _check_option_keys(rendered)
        assert has_options is True
        assert n_options == 3

    def test_no_options(self) -> None:
        """Test with no option keys."""
        rendered = {"text": "Hello"}
        has_options, n_options = _check_option_keys(rendered)
        assert has_options is False
        assert n_options == 0

    def test_only_option_a(self) -> None:
        """Test with only option_a (not enough)."""
        rendered = {"option_a": "A"}
        has_options, n_options = _check_option_keys(rendered)
        assert has_options is False
        assert n_options == 1

    def test_non_consecutive_options(self) -> None:
        """Test with non-consecutive option keys."""
        rendered = {"option_a": "A", "option_c": "C"}
        has_options, n_options = _check_option_keys(rendered)
        assert has_options is False
        assert n_options == 1  # Only option_a is consecutive


class TestGetTaskTypeRequirements:
    """Tests for get_task_type_requirements function."""

    def test_forced_choice_requirements(self) -> None:
        """Test requirements for forced_choice."""
        reqs = get_task_type_requirements("forced_choice")
        assert reqs["required_rendered_keys"] == []  # Options stored in options field
        assert reqs["required_metadata_keys"] == []  # n_options not auto-set
        assert "n_options" in reqs["optional_metadata_keys"]
        assert "options" in reqs["special_fields"]

    def test_multi_select_requirements(self) -> None:
        """Test requirements for multi_select."""
        reqs = get_task_type_requirements("multi_select")
        assert reqs["required_rendered_keys"] == []  # Options stored in options field
        assert "min_selections" in reqs["required_metadata_keys"]
        assert "max_selections" in reqs["required_metadata_keys"]
        assert "options" in reqs["special_fields"]

    def test_ordinal_scale_requirements(self) -> None:
        """Test requirements for ordinal_scale."""
        reqs = get_task_type_requirements("ordinal_scale")
        assert "text" in reqs["required_rendered_keys"]
        assert "prompt" in reqs["required_rendered_keys"]
        assert "scale_min" in reqs["required_metadata_keys"]
        assert "scale_max" in reqs["required_metadata_keys"]

    def test_magnitude_requirements(self) -> None:
        """Test requirements for magnitude."""
        reqs = get_task_type_requirements("magnitude")
        assert "text" in reqs["required_rendered_keys"]
        assert "prompt" in reqs["required_rendered_keys"]
        assert "min_value" in reqs["required_metadata_keys"]
        assert "max_value" in reqs["required_metadata_keys"]
        assert "unit" in reqs["optional_metadata_keys"]

    def test_binary_requirements(self) -> None:
        """Test requirements for binary."""
        reqs = get_task_type_requirements("binary")
        assert "text" in reqs["required_rendered_keys"]
        assert "prompt" in reqs["required_rendered_keys"]

    def test_categorical_requirements(self) -> None:
        """Test requirements for categorical."""
        reqs = get_task_type_requirements("categorical")
        assert "text" in reqs["required_rendered_keys"]
        assert "prompt" in reqs["required_rendered_keys"]
        assert "categories" in reqs["required_metadata_keys"]

    def test_free_text_requirements(self) -> None:
        """Test requirements for free_text."""
        reqs = get_task_type_requirements("free_text")
        assert "text" in reqs["required_rendered_keys"]
        assert "prompt" in reqs["required_rendered_keys"]
        assert "max_length" in reqs["optional_metadata_keys"]

    def test_cloze_requirements(self) -> None:
        """Test requirements for cloze."""
        reqs = get_task_type_requirements("cloze")
        assert reqs["required_rendered_keys"] == ["text"]
        assert "n_unfilled_slots" in reqs["required_metadata_keys"]
        assert "unfilled_slots" in reqs["special_fields"]

    def test_unknown_task_type_raises_error(self) -> None:
        """Test that unknown task type raises ValueError."""
        with pytest.raises((ValueError, dx.ValidationError), match="Unknown task type"):
            get_task_type_requirements("unknown_task")


class TestValidateItemForTaskType:
    """Tests for validate_item_for_task_type function."""

    def test_forced_choice_valid(self) -> None:
        """Test valid forced_choice item passes validation."""
        from bead.items.forced_choice import create_forced_choice_item

        item = create_forced_choice_item("A", "B")
        assert validate_item_for_task_type(item, "forced_choice") is True

    def test_forced_choice_invalid_raises_error(self) -> None:
        """Test invalid structure for forced_choice raises ValueError."""
        from bead.items.ordinal_scale import create_ordinal_scale_item

        item = create_ordinal_scale_item("Text", scale_bounds=ScaleBounds(min=1, max=5))
        with pytest.raises(
            (ValueError, dx.ValidationError), match="forced_choice items must have"
        ):
            validate_item_for_task_type(item, "forced_choice")

    def test_multi_select_valid(self) -> None:
        """Test valid multi_select item passes validation."""
        from bead.items.multi_select import create_multi_select_item

        item = create_multi_select_item(
            "A", "B", "C", min_selections=1, max_selections=3
        )
        assert validate_item_for_task_type(item, "multi_select") is True

    def test_multi_select_invalid_min_max(self) -> None:
        """Test multi_select with min > max raises error."""
        item = Item(
            item_template_id=uuid4(),
            options=["A", "B"],
            item_metadata={"min_selections": 3, "max_selections": 1},
        )
        with pytest.raises(
            (ValueError, dx.ValidationError), match="min_selections <= max_selections"
        ):
            validate_item_for_task_type(item, "multi_select")

    def test_ordinal_scale_valid(self) -> None:
        """Test valid ordinal_scale item passes validation."""
        from bead.items.ordinal_scale import create_ordinal_scale_item

        item = create_ordinal_scale_item(
            "How natural?", scale_bounds=ScaleBounds(min=1, max=7)
        )
        assert validate_item_for_task_type(item, "ordinal_scale") is True

    def test_ordinal_scale_invalid_bounds(self) -> None:
        """Test ordinal_scale with min >= max raises error."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Test", "prompt": "Rate this:"},
            item_metadata={"scale_min": 7, "scale_max": 1},
        )
        with pytest.raises(
            (ValueError, dx.ValidationError), match="scale_min < scale_max"
        ):
            validate_item_for_task_type(item, "ordinal_scale")

    def test_magnitude_valid(self) -> None:
        """Test valid magnitude item passes validation."""
        from bead.items.magnitude import create_magnitude_item

        item = create_magnitude_item("Enter reading time", unit="ms")
        assert validate_item_for_task_type(item, "magnitude") is True

    def test_magnitude_with_bounds_valid(self) -> None:
        """Test magnitude with valid bounds passes validation."""
        from bead.items.magnitude import create_magnitude_item

        item = create_magnitude_item("Enter value", bounds=(0.0, 100.0))
        assert validate_item_for_task_type(item, "magnitude") is True

    def test_magnitude_invalid_bounds(self) -> None:
        """Test magnitude with min >= max raises error."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Test", "prompt": "Enter value:"},
            item_metadata={"min_value": 100, "max_value": 0},
        )
        with pytest.raises(
            (ValueError, dx.ValidationError), match="min_value < max_value"
        ):
            validate_item_for_task_type(item, "magnitude")

    def test_binary_valid(self) -> None:
        """Test valid binary item passes validation."""
        from bead.items.binary import create_binary_item

        item = create_binary_item("The cat sat.", prompt="Is this grammatical?")
        assert validate_item_for_task_type(item, "binary") is True

    def test_binary_empty_prompt_raises_error(self) -> None:
        """Test binary with empty prompt raises error."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Test", "prompt": ""},
            item_metadata={},
        )
        with pytest.raises(
            (ValueError, dx.ValidationError), match="non-empty 'prompt'"
        ):
            validate_item_for_task_type(item, "binary")

    def test_categorical_valid(self) -> None:
        """Test valid categorical item passes validation."""
        from bead.items.categorical import create_nli_item

        item = create_nli_item("All dogs bark", "Some dogs bark")
        assert validate_item_for_task_type(item, "categorical") is True

    def test_categorical_missing_categories_raises_error(self) -> None:
        """Test categorical without categories raises error."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Test", "prompt": "Choose"},
            item_metadata={},
        )
        with pytest.raises(
            ValueError,
            match="categorical items must have.*categories.*in item_metadata",
        ):
            validate_item_for_task_type(item, "categorical")

    def test_free_text_valid(self) -> None:
        """Test valid free_text item passes validation."""
        from bead.items.free_text import create_free_text_item

        item = create_free_text_item("The cat sat.", prompt="What is the subject?")
        assert validate_item_for_task_type(item, "free_text") is True

    def test_cloze_valid(self) -> None:
        """Test valid cloze item passes validation."""
        from bead.items.cloze import create_simple_cloze_item

        item = create_simple_cloze_item(
            text="The quick brown fox",
            blank_positions=[1],
            blank_labels=["adjective"],
        )
        assert validate_item_for_task_type(item, "cloze") is True

    def test_cloze_without_unfilled_slots_raises_error(self) -> None:
        """Test cloze validation checks unfilled_slots field."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Test"},
            item_metadata={"n_unfilled_slots": 1},
            unfilled_slots=[],  # Empty!
        )
        with pytest.raises(
            (ValueError, dx.ValidationError), match="unfilled_slots field populated"
        ):
            validate_item_for_task_type(item, "cloze")


class TestInferTaskTypeFromItem:
    """Tests for infer_task_type_from_item function."""

    def test_infer_forced_choice(self) -> None:
        """Test inference of forced_choice from structure."""
        from bead.items.forced_choice import create_forced_choice_item

        item = create_forced_choice_item("A", "B")
        assert infer_task_type_from_item(item) == "forced_choice"

    def test_infer_multi_select(self) -> None:
        """Test inference of multi_select from structure."""
        from bead.items.multi_select import create_multi_select_item

        item = create_multi_select_item(
            "A", "B", "C", min_selections=1, max_selections=2
        )
        assert infer_task_type_from_item(item) == "multi_select"

    def test_infer_ordinal_scale(self) -> None:
        """Test inference of ordinal_scale from structure."""
        from bead.items.ordinal_scale import create_likert_7_item

        item = create_likert_7_item("How natural is this sentence?")
        assert infer_task_type_from_item(item) == "ordinal_scale"

    def test_infer_magnitude(self) -> None:
        """Test inference of magnitude from structure."""
        from bead.items.magnitude import create_magnitude_item

        item = create_magnitude_item("Enter reading time", unit="ms")
        assert infer_task_type_from_item(item) == "magnitude"

    def test_infer_binary(self) -> None:
        """Test inference of binary from structure."""
        from bead.items.binary import create_binary_item

        item = create_binary_item("The cat sat.", prompt="Is this grammatical?")
        assert infer_task_type_from_item(item) == "binary"

    def test_infer_categorical(self) -> None:
        """Test inference of categorical from structure."""
        from bead.items.categorical import create_nli_item

        item = create_nli_item("All dogs bark", "Some dogs bark")
        assert infer_task_type_from_item(item) == "categorical"

    def test_infer_free_text(self) -> None:
        """Test inference of free_text from structure."""
        from bead.items.free_text import create_free_text_item

        item = create_free_text_item("The cat sat.", prompt="What is the subject?")
        assert infer_task_type_from_item(item) == "free_text"

    def test_infer_cloze(self) -> None:
        """Test inference of cloze from structure."""
        from bead.items.cloze import create_simple_cloze_item

        item = create_simple_cloze_item(
            text="The quick brown fox",
            blank_positions=[1],
            blank_labels=["adjective"],
        )
        assert infer_task_type_from_item(item) == "cloze"

    def test_ambiguous_text_only_raises_error(self) -> None:
        """Test that ambiguous structure raises error."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Test"},
            item_metadata={},  # No distinguishing metadata
        )
        with pytest.raises(
            ValueError, match="single 'text' key without unfilled_slots"
        ):
            infer_task_type_from_item(item)

    def test_ambiguous_text_prompt_raises_error(self) -> None:
        """Test that text+prompt without metadata raises error."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Test", "prompt": "Answer"},
            item_metadata={},  # No distinguishing metadata
        )
        with pytest.raises(
            (ValueError, dx.ValidationError), match="Could be binary or free_text"
        ):
            infer_task_type_from_item(item)

    def test_no_match_raises_error(self) -> None:
        """Test that unrecognized structure raises error."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"unknown_key": "value"},
            item_metadata={},
        )
        with pytest.raises(
            (ValueError, dx.ValidationError), match="Could not infer task type"
        ):
            infer_task_type_from_item(item)
