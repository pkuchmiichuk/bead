"""Validation utilities for constructed items.

This module provides validation functions to ensure constructed items
meet all requirements and contain complete, valid data.
"""

from __future__ import annotations

from bead.items.item import Item, ModelOutput
from bead.items.item_template import ItemTemplate, TaskType


def validate_item(item: Item, item_template: ItemTemplate) -> list[str]:
    """Validate a constructed item against its template.

    Check that the item has all required fields, references valid templates,
    has consistent constraint satisfaction, and contains valid model outputs.

    Parameters
    ----------
    item : Item
        Item to validate.
    item_template : ItemTemplate
        Template the item was constructed from.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list if valid.

    Examples
    --------
    >>> errors = validate_item(item, template)
    >>> if errors:
    ...     print(f"Item is invalid: {errors}")
    >>> else:
    ...     print("Item is valid")
    """
    errors: list[str] = []

    # Check item_template_id matches
    if item.item_template_id != item_template.id:
        errors.append(
            f"Item template ID mismatch: {item.item_template_id} != {item_template.id}"
        )

    # Check all elements are rendered
    expected_elements = {elem.element_name for elem in item_template.elements}
    actual_elements = set(item.rendered_elements.keys())

    missing = expected_elements - actual_elements
    if missing:
        errors.append(f"Missing rendered elements: {missing}")

    extra = actual_elements - expected_elements
    if extra:
        errors.append(f"Extra rendered elements: {extra}")

    # Check all constraints are evaluated
    expected_constraints = set(item_template.constraints)
    actual_constraints = {cs.constraint_id for cs in item.constraint_satisfaction}

    missing_constraints = expected_constraints - actual_constraints
    if missing_constraints:
        errors.append(f"Missing constraint evaluations: {missing_constraints}")

    # Check model outputs are valid
    for output in item.model_outputs:
        output_errors = validate_model_output(output)
        errors.extend(output_errors)

    return errors


def validate_model_output(output: ModelOutput) -> list[str]:
    """Validate a model output.

    Check that the model output has all required fields and valid values.

    Parameters
    ----------
    output : ModelOutput
        Model output to validate.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list if valid.

    Examples
    --------
    >>> errors = validate_model_output(output)
    >>> if not errors:
    ...     print("Model output is valid")
    """
    errors: list[str] = []

    # Check required fields are not empty
    if not output.model_name or not output.model_name.strip():
        errors.append("Model output has empty model_name")

    if not output.operation or not output.operation.strip():
        errors.append("Model output has empty operation")

    if not output.cache_key or not output.cache_key.strip():
        errors.append("Model output has empty cache_key")

    # Check operation-specific output structure
    if output.operation == "nli":
        # NLI should return dict with entailment/neutral/contradiction
        if not isinstance(output.output, dict):
            errors.append(f"NLI output should be dict, got {type(output.output)}")
        else:
            expected_keys = {"entailment", "neutral", "contradiction"}
            actual_keys = set(output.output.keys())  # type: ignore[union-attr]
            if actual_keys != expected_keys:
                errors.append(
                    f"NLI output keys mismatch: expected {expected_keys}, "
                    f"got {actual_keys}"
                )

    elif output.operation in ("log_probability", "perplexity", "similarity"):
        # These should return numeric values
        if not isinstance(output.output, int | float):
            errors.append(
                f"{output.operation} output should be numeric, "
                f"got {type(output.output)}"
            )

    elif output.operation == "embedding":
        # Should return list or array
        if not isinstance(output.output, list | dict):
            # dict could be serialized ndarray
            errors.append(
                f"Embedding output should be list/array, got {type(output.output)}"
            )

    return errors


def validate_constraint_satisfaction(
    item: Item, item_template: ItemTemplate
) -> list[str]:
    """Validate constraint satisfaction consistency.

    Check that all constraints in the template have been evaluated and
    that the results are boolean values.

    Parameters
    ----------
    item : Item
        Item to validate.
    item_template : ItemTemplate
        Template with constraints.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list if valid.

    Examples
    --------
    >>> errors = validate_constraint_satisfaction(item, template)
    >>> if not errors:
    ...     print("Constraint satisfaction is valid")
    """
    errors: list[str] = []

    by_id = {cs.constraint_id: cs for cs in item.constraint_satisfaction}
    for constraint_id in item_template.constraints:
        if constraint_id not in by_id:
            errors.append(f"Constraint {constraint_id} not evaluated")
        else:
            value = by_id[constraint_id].satisfied
            if type(value) is not bool:
                errors.append(
                    f"Constraint {constraint_id} satisfaction should be bool, "
                    f"got {type(value)}"
                )

    return errors


def validate_metadata_completeness(item: Item) -> list[str]:
    """Validate that item metadata is complete.

    Check that the item has all expected metadata fields populated.
    Since Item inherits from BeadBaseModel, id, created_at, and modified_at
    are always present. This function is kept for consistency and future
    extensibility.

    Parameters
    ----------
    item : Item
        Item to validate.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list if valid.

    Examples
    --------
    >>> errors = validate_metadata_completeness(item)
    >>> if not errors:
    ...     print("Metadata is complete")
    """
    errors: list[str] = []

    # Check base model fields (from BeadBaseModel)
    # These are always present due to Pydantic model initialization,
    # but we check for completeness
    if not hasattr(item, "id"):
        errors.append("Item missing id field")  # pragma: no cover

    if not hasattr(item, "created_at"):
        errors.append("Item missing created_at timestamp")  # pragma: no cover

    if not hasattr(item, "modified_at"):
        errors.append("Item missing modified_at timestamp")  # pragma: no cover

    return errors


def item_passes_all_constraints(item: Item) -> bool:
    """Check if item satisfies all constraints.

    Convenience function to check if all constraints are satisfied.

    Parameters
    ----------
    item : Item
        Item to check.

    Returns
    -------
    bool
        True if all constraints satisfied, False otherwise.

    Examples
    --------
    >>> if item_passes_all_constraints(item):
    ...     print("Item is valid")
    """
    return all(cs.satisfied for cs in item.constraint_satisfaction)


def _check_options(item: Item) -> tuple[bool, int]:
    """Check if item has valid options list.

    Helper function for detecting forced_choice and multi_select task types.
    Checks the item.options field for a valid list of options.

    Parameters
    ----------
    item : Item
        Item to check for options.

    Returns
    -------
    tuple[bool, int]
        Tuple of (has_options, n_options) where has_options is True if
        the item has at least 2 options, and n_options is the count.

    Examples
    --------
    >>> item = Item(item_template_id=uuid4(), options=["A", "B"])
    >>> _check_options(item)
    (True, 2)
    >>> item = Item(item_template_id=uuid4(), options=[])
    >>> _check_options(item)
    (False, 0)
    >>> item = Item(item_template_id=uuid4(), options=["A"])
    >>> _check_options(item)
    (False, 0)  # Need at least 2 options
    """
    if not item.options:
        return (False, 0)

    n_options = len(item.options)

    # Must have at least 2 options to be valid
    if n_options < 2:
        return (False, 0)

    return (True, n_options)


def _check_option_keys(  # pyright: ignore[reportUnusedFunction]
    rendered_elements: dict[str, str],
) -> tuple[bool, int]:
    """Check if rendered_elements has consecutive option_a, option_b, ... keys.

    .. deprecated::
        This function is deprecated. Use _check_options() instead, which
        checks the item.options list field.

    Helper function for detecting forced_choice and multi_select task types
    in legacy items that store options in rendered_elements.

    Parameters
    ----------
    rendered_elements : dict
        Dictionary of rendered elements to check.

    Returns
    -------
    tuple[bool, int]
        Tuple of (has_options, n_options) where has_options is True if
        consecutive option keys found, and n_options is the count.

    Examples
    --------
    >>> _check_option_keys({"option_a": "A", "option_b": "B"})
    (True, 2)
    >>> _check_option_keys({"text": "Hello"})
    (False, 0)
    >>> _check_option_keys({"option_a": "A", "option_c": "C"})
    (False, 0)  # Not consecutive
    """
    # Check for option_a, option_b, option_c, ...
    if "option_a" not in rendered_elements:
        return (False, 0)

    # Count consecutive options starting from option_a
    n_options = 0
    expected_letters = "abcdefghijklmnopqrstuvwxyz"

    for letter in expected_letters:
        key = f"option_{letter}"
        if key in rendered_elements:
            n_options += 1
        else:
            break

    # Must have at least 2 options to be valid
    return (n_options >= 2, n_options)


def get_task_type_requirements(task_type: TaskType) -> dict[str, list[str] | str]:
    """Get validation requirements for a task type.

    Returns a dictionary describing the structural requirements
    for items of the specified task type. Useful for introspection,
    error messages, and documentation generation.

    Parameters
    ----------
    task_type : TaskType
        Task type to get requirements for.

    Returns
    -------
    dict
        Requirements specification with keys:
        - required_rendered_keys: List of required rendered_elements keys
        - required_metadata_keys: List of required item_metadata keys
        - optional_metadata_keys: List of optional item_metadata keys
        - special_fields: List of special fields (e.g., ["unfilled_slots"])
        - description: Human-readable description

    Examples
    --------
    >>> reqs = get_task_type_requirements("ordinal_scale")
    >>> print(reqs["required_rendered_keys"])
    ['text']
    >>> print(reqs["required_metadata_keys"])
    ['scale_min', 'scale_max']
    """
    requirements = {
        "forced_choice": {
            "required_rendered_keys": [],
            "required_metadata_keys": [],
            "optional_metadata_keys": [
                "source_items",
                "group_key",
                "pair_type",
                "n_options",
            ],
            "special_fields": ["options"],
            "description": (
                "Pick exactly one option from N alternatives (2AFC, 3AFC, ...)"
            ),
        },
        "multi_select": {
            "required_rendered_keys": [],
            "required_metadata_keys": ["min_selections", "max_selections"],
            "optional_metadata_keys": ["source_items", "group_key"],
            "special_fields": ["options"],
            "description": "Pick one or more options (checkboxes)",
        },
        "ordinal_scale": {
            "required_rendered_keys": ["text", "prompt"],
            "required_metadata_keys": ["scale_min", "scale_max"],
            "optional_metadata_keys": ["source_items", "group_key", "scale_labels"],
            "special_fields": [],
            "description": "Value on ordered discrete scale (Likert, slider)",
        },
        "magnitude": {
            "required_rendered_keys": ["text", "prompt"],
            "required_metadata_keys": ["min_value", "max_value"],
            "optional_metadata_keys": [
                "unit",
                "step",
                "source_items",
                "group_key",
            ],
            "special_fields": [],
            "description": "Unbounded numeric value (reading time, confidence)",
        },
        "binary": {
            "required_rendered_keys": ["text", "prompt"],
            "required_metadata_keys": [],
            "optional_metadata_keys": ["binary_options", "source_items", "group_key"],
            "special_fields": [],
            "description": "Yes/No, True/False (absolute judgment)",
        },
        "categorical": {
            "required_rendered_keys": ["text", "prompt"],
            "required_metadata_keys": ["categories"],
            "optional_metadata_keys": ["source_items", "group_key"],
            "special_fields": [],
            "description": "Pick from unordered categories (NLI, semantic relations)",
        },
        "free_text": {
            "required_rendered_keys": ["text", "prompt"],
            "required_metadata_keys": [],
            "optional_metadata_keys": [
                "max_length",
                "validation_pattern",
                "multiline",
                "source_items",
                "group_key",
            ],
            "special_fields": [],
            "description": "Open-ended text (paraphrase, comprehension)",
        },
        "cloze": {
            "required_rendered_keys": ["text"],
            "required_metadata_keys": ["n_unfilled_slots"],
            "optional_metadata_keys": ["source_items", "group_key", "template_id"],
            "special_fields": ["unfilled_slots"],
            "description": "Fill-in-the-blank (constraint-based UI)",
        },
    }

    if task_type not in requirements:
        raise ValueError(
            f"Unknown task type: {task_type}. "
            f"Expected one of: {list(requirements.keys())}"
        )

    return requirements[task_type]


def validate_item_for_task_type(item: Item, task_type: TaskType) -> bool:
    """Validate that an Item's structure matches requirements for a task type.

    Checks that the item has the required rendered_elements keys,
    item_metadata keys, and special fields for the specified task type.
    Raises descriptive ValueError if validation fails.

    Parameters
    ----------
    item : Item
        Item to validate.
    task_type : TaskType
        Expected task type (from bead.items.item_template.TaskType).

    Returns
    -------
    bool
        True if valid.

    Raises
    ------
    ValueError
        If item structure doesn't match task type requirements,
        with detailed explanation of what's wrong.

    Examples
    --------
    >>> from bead.items.ordinal_scale import create_ordinal_scale_item
    >>> item = create_ordinal_scale_item(
    ...     "How natural?", scale_bounds=ScaleBounds(min=1, max=7)
    ... )
    >>> validate_item_for_task_type(item, "ordinal_scale")
    True

    >>> from bead.items.forced_choice import create_forced_choice_item
    >>> fc_item = create_forced_choice_item("A", "B")
    >>> validate_item_for_task_type(fc_item, "ordinal_scale")
    ValueError: ordinal_scale items must have 'text' in rendered_elements...
    """
    reqs = get_task_type_requirements(task_type)

    # Check rendered_elements keys
    actual_rendered = set(item.rendered_elements.keys())
    required_rendered = set(reqs["required_rendered_keys"])

    # Special handling for forced_choice and multi_select (options field)
    if task_type in ("forced_choice", "multi_select"):
        has_options, n_options = _check_options(item)
        if not has_options:
            raise ValueError(
                f"{task_type} items must have at least 2 options in the options field, "
                f"but found {n_options} option(s): {item.options}"
            )
        # For these types, we don't check for exact required rendered_elements keys
    else:
        # Check for exact required keys
        missing_rendered = required_rendered - actual_rendered
        if missing_rendered:
            raise ValueError(
                f"{task_type} items must have {list(required_rendered)} "
                f"in rendered_elements, but missing: {list(missing_rendered)}. "
                f"Found keys: {list(actual_rendered)}"
            )

    # Check item_metadata keys
    actual_metadata = set(item.item_metadata.keys())
    required_metadata = set(reqs["required_metadata_keys"])

    missing_metadata = required_metadata - actual_metadata
    if missing_metadata:
        raise ValueError(
            f"{task_type} items must have {list(required_metadata)} "
            f"in item_metadata, but missing: {list(missing_metadata)}. "
            f"Found keys: {list(actual_metadata)}"
        )

    # Check special fields
    if "unfilled_slots" in reqs["special_fields"]:
        if not item.unfilled_slots:
            raise ValueError(
                f"{task_type} items must have unfilled_slots field populated, "
                f"but found empty list"
            )

    if "options" in reqs["special_fields"]:
        if not item.options or len(item.options) < 2:
            raise ValueError(
                f"{task_type} items must have at least 2 options in the options field, "
                f"but found {len(item.options) if item.options else 0} option(s)"
            )

    # Task-specific validation
    if task_type == "ordinal_scale":
        scale_min = item.item_metadata.get("scale_min")
        scale_max = item.item_metadata.get("scale_max")
        if not isinstance(scale_min, int) or not isinstance(scale_max, int):
            raise ValueError(
                f"ordinal_scale items must have integer scale_min and scale_max, "
                f"but got scale_min={type(scale_min).__name__}, "
                f"scale_max={type(scale_max).__name__}"
            )
        if scale_min >= scale_max:
            raise ValueError(
                f"ordinal_scale items must have scale_min < scale_max, "
                f"but got scale_min={scale_min}, scale_max={scale_max}"
            )

    if task_type == "multi_select":
        min_sel = item.item_metadata.get("min_selections")
        max_sel = item.item_metadata.get("max_selections")
        if not isinstance(min_sel, int) or not isinstance(max_sel, int):
            raise ValueError(
                "multi_select items must have integer min_selections "
                f"and max_selections, but got min_selections="
                f"{type(min_sel).__name__}, max_selections="
                f"{type(max_sel).__name__}"
            )
        if min_sel <= 0 or max_sel <= 0:
            raise ValueError(
                "multi_select items must have positive min_selections "
                f"and max_selections, but got min_selections={min_sel}, "
                f"max_selections={max_sel}"
            )
        if min_sel > max_sel:
            raise ValueError(
                f"multi_select items must have min_selections <= max_selections, "
                f"but got min_selections={min_sel}, max_selections={max_sel}"
            )

    if task_type == "magnitude":
        min_val = item.item_metadata.get("min_value")
        max_val = item.item_metadata.get("max_value")
        if min_val is not None and max_val is not None:
            if not isinstance(min_val, int | float) or not isinstance(
                max_val, int | float
            ):
                raise ValueError(
                    "magnitude items with bounds must have numeric "
                    f"min_value and max_value, but got min_value="
                    f"{type(min_val).__name__}, max_value="
                    f"{type(max_val).__name__}"
                )
            if min_val >= max_val:
                raise ValueError(
                    f"magnitude items must have min_value < max_value, "
                    f"but got min_value={min_val}, max_value={max_val}"
                )

    if task_type in ("binary", "categorical", "free_text"):
        prompt = item.rendered_elements.get("prompt")
        if not prompt or not str(prompt).strip():
            raise ValueError(
                f"{task_type} items must have non-empty 'prompt' in rendered_elements"
            )

    if task_type == "categorical":
        categories = item.item_metadata.get("categories")
        if not isinstance(categories, list | tuple) or len(categories) == 0:
            raise ValueError(
                "categorical items must have non-empty list/tuple in "
                f"item_metadata['categories'], but got "
                f"{type(categories).__name__}"
            )

    # No additional validation needed for forced_choice
    # (n_options is optional metadata, not required)

    return True


def infer_task_type_from_item(item: Item) -> TaskType:
    """Infer most likely task type from Item structure.

    Examines the item's rendered_elements, item_metadata, and special fields
    to determine which task type it matches. Uses priority order to handle
    ambiguous cases.

    Parameters
    ----------
    item : Item
        Item to infer from.

    Returns
    -------
    TaskType
        Inferred task type.

    Raises
    ------
    ValueError
        If item structure doesn't match any task type or is ambiguous.

    Examples
    --------
    >>> from bead.items.ordinal_scale import create_likert_7_item
    >>> item = create_likert_7_item("How natural is this sentence?")
    >>> infer_task_type_from_item(item)
    'ordinal_scale'

    >>> from bead.items.categorical import create_nli_item
    >>> item2 = create_nli_item("All dogs bark", "Some dogs bark")
    >>> infer_task_type_from_item(item2)
    'categorical'
    """
    rendered = item.rendered_elements
    metadata = item.item_metadata

    # Priority 1: Check for cloze (unique unfilled_slots field)
    if item.unfilled_slots:
        if "n_unfilled_slots" in metadata:
            return "cloze"

    # Priority 2: Check for forced_choice/multi_select (options list field)
    has_options, _ = _check_options(item)
    if has_options:
        # Distinguish between forced_choice and multi_select
        if "min_selections" in metadata and "max_selections" in metadata:
            return "multi_select"
        if "n_options" in metadata:
            return "forced_choice"
        # Default to forced_choice if has options but no specific metadata
        return "forced_choice"

    # Priority 3: Check for single "text" key (cloze without unfilled_slots)
    if "text" in rendered and len(rendered) == 1:
        # Must be cloze if only "text" key exists
        # (but we already checked unfilled_slots above)
        # Ambiguous: could be improperly constructed item
        raise ValueError(
            "Item has single 'text' key without unfilled_slots. "
            "If this is a cloze item, ensure unfilled_slots is populated. "
            "Other task types require additional keys."
        )

    # Priority 4: Check for text + prompt
    # (ordinal_scale, magnitude, binary, categorical, free_text)
    if "text" in rendered and "prompt" in rendered:
        # Ordinal scale has scale_min/scale_max
        if "scale_min" in metadata and "scale_max" in metadata:
            return "ordinal_scale"
        # Magnitude has min_value/max_value (always set, may be None)
        if "min_value" in metadata and "max_value" in metadata:
            return "magnitude"
        # Categorical has categories
        if "categories" in metadata:
            return "categorical"
        # Binary may have binary_options
        if "binary_options" in metadata:
            return "binary"
        # Free text may have max_length, validation_pattern, or multiline
        if (
            "max_length" in metadata
            or "validation_pattern" in metadata
            or "multiline" in metadata
        ):
            return "free_text"
        # Could be binary or free_text (most ambiguous case)
        raise ValueError(
            "Could be binary or free_text based on structure. "
            "Item has 'text' and 'prompt' but no distinguishing metadata. "
            "Use explicit task type validation."
        )

    # No match
    raise ValueError(
        f"Could not infer task type from item structure. "
        f"rendered_elements keys: {list(rendered.keys())}, "
        f"item_metadata keys: {list(metadata.keys())}"
    )
