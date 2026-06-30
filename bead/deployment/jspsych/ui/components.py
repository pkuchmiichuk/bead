"""Python helpers for generating UI components for jsPsych experiments.

This module provides functions to generate UI component configurations
from bead models, inferring widget types from slot constraints.
"""

from __future__ import annotations

from uuid import UUID

from bead.data.base import JsonValue
from bead.dsl import ast
from bead.dsl.parser import parse
from bead.items.item import UnfilledSlot
from bead.resources.constraints import Constraint


def create_rating_scale(
    scale_min: int,
    scale_max: int,
    labels: dict[int, str] | None = None,
) -> dict[str, int | dict[int, str]]:
    """Generate jsPsych rating scale configuration.

    Parameters
    ----------
    scale_min : int
        Minimum value of the scale.
    scale_max : int
        Maximum value of the scale.
    labels : dict[int, str] | None
        Optional labels for specific scale points.

    Returns
    -------
    dict[str, int | dict[int, str]]
        Rating scale configuration dictionary.

    Examples
    --------
    >>> config = create_rating_scale(
    ...     1, 7, {1: "Strongly Disagree", 7: "Strongly Agree"}
    ... )
    >>> config["scale_min"]
    1
    >>> config["scale_max"]
    7
    """
    return {
        "scale_min": scale_min,
        "scale_max": scale_max,
        "scale_labels": labels or {},
    }


def create_cloze_fields(
    unfilled_slots: list[UnfilledSlot],
    constraints: dict[UUID, Constraint],
    lexicon: dict[UUID, str] | None = None,
) -> list[dict[str, JsonValue]]:
    """Generate cloze field configurations from slots and constraints.

    Infers widget type from slot constraints:
    - Constraint with "self.id in [...]" pattern: dropdown with specific items
    - Other DSL constraints: text input with validation expression
    - No constraints: free text input

    Parameters
    ----------
    unfilled_slots : list[UnfilledSlot]
        List of unfilled slots in the cloze task.
    constraints : dict[UUID, Constraint]
        Dictionary of constraints keyed by UUID (from slot.constraint_ids).
    lexicon : dict[UUID, str] | None
        Optional mapping from item UUIDs to surface forms for dropdown options.

    Returns
    -------
    list[dict[str, JsonValue]]
        List of field configuration dictionaries with keys:
        - slot_name: Name of the slot
        - position: Token position
        - type: Widget type ("dropdown" or "text")
        - options: List of allowed values (for dropdown)
        - placeholder: Placeholder text
        - dsl_expression: Constraint expression for validation (optional)
        - dsl_context: Constraint context variables (optional)

    Examples
    --------
    >>> from bead.items.item import UnfilledSlot
    >>> from bead.resources.constraints import Constraint
    >>> from uuid import uuid4
    >>> constraint_id = uuid4()
    >>> slot = UnfilledSlot(
    ...     slot_name="determiner",
    ...     position=0,
    ...     constraint_ids=[constraint_id]
    ... )
    >>> id1, id2 = uuid4(), uuid4()
    >>> constraint = Constraint(
    ...     expression=f"self.id in [UUID('{id1}'), UUID('{id2}')]"
    ... )
    >>> lexicon = {id1: "the", id2: "a"}
    >>> fields = create_cloze_fields([slot], {constraint_id: constraint}, lexicon)
    >>> len(fields)
    1
    >>> fields[0]["type"]
    'dropdown'
    >>> len(fields[0]["options"])
    2
    """
    fields: list[dict[str, JsonValue]] = []

    for slot in unfilled_slots:
        # infer widget type
        widget_type = infer_widget_type(slot.constraint_ids, constraints)

        field_config: dict[str, JsonValue] = {
            "slot_name": slot.slot_name,
            "position": slot.position,
            "type": widget_type,
            "options": [],
            "placeholder": slot.slot_name,
        }

        # analyze constraints to extract options and validation info
        for constraint_id in slot.constraint_ids:
            if constraint_id not in constraints:
                continue

            constraint = constraints[constraint_id]

            # include constraint expression for client-side validation
            field_config["dsl_expression"] = constraint.expression
            if constraint.context:
                # convert context to JsonValue compatible format
                field_config["dsl_context"] = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in constraint.context.items()
                }

            # if dropdown, extract allowed item UUIDs and map to surface forms
            if widget_type == "dropdown":
                allowed_items = _extract_allowed_items_from_expression(
                    constraint.expression, constraint.context
                )
                if allowed_items:
                    # map UUIDs to surface forms if lexicon provided
                    if lexicon:
                        options = [
                            lexicon.get(item_id, str(item_id))
                            for item_id in allowed_items
                        ]
                    else:
                        # no lexicon; use UUID strings
                        options = [str(item_id) for item_id in allowed_items]

                    field_config["options"] = sorted(options)
                    field_config["item_ids"] = [
                        str(item_id) for item_id in allowed_items
                    ]

        fields.append(field_config)

    return fields


def create_forced_choice_config(
    alternatives: list[str],
    randomize_position: bool = True,
    enable_keyboard: bool = True,
) -> dict[str, list[str] | bool]:
    """Generate forced choice configuration.

    Parameters
    ----------
    alternatives : list[str]
        List of alternative options to choose from.
    randomize_position : bool
        Whether to randomize left/right position.
    enable_keyboard : bool
        Whether to enable keyboard responses.

    Returns
    -------
    dict[str, list[str] | bool]
        Forced choice configuration dictionary.

    Examples
    --------
    >>> config = create_forced_choice_config(["Option A", "Option B"])
    >>> config["randomize_position"]
    True
    >>> len(config["alternatives"])
    2
    """
    return {
        "alternatives": alternatives,
        "randomize_position": randomize_position,
        "enable_keyboard": enable_keyboard,
    }


def _extract_allowed_items_from_expression(
    expression: str,
    context: dict[str, str | int | float | bool | list[str] | set[str] | set[UUID]]
    | None = None,
) -> set[UUID] | None:
    """Extract allowed item UUIDs from a DSL constraint expression.

    Detects patterns like:
    - self.id in [uuid1, uuid2, ...] (inline list)
    - self.id in allowed_items (context variable)
    - str(self.id) in ['uuid-str1', 'uuid-str2', ...]

    Parameters
    ----------
    expression : str
        DSL constraint expression.
    context : dict[str, str | int | float | bool | list[str] | set[str] \
            | set[UUID]] | None
        Constraint context variables.

    Returns
    -------
    set[UUID] | None
        Set of allowed UUIDs if pattern detected, None otherwise.

    Examples
    --------
    >>> from uuid import UUID
    >>> expr = "self.id in [UUID('...'), UUID('...')]"
    >>> # Would return set of UUIDs if parseable
    """
    try:
        node = parse(expression)
    except Exception:
        return None

    # look for pattern: self.id in [...]
    if isinstance(node, ast.BinaryOp) and node.operator == "in":
        # check if left side is self.id or str(self.id)
        left = node.left
        is_self_id = False

        if isinstance(left, ast.AttributeAccess):
            # self.id
            if (
                isinstance(left.object, ast.Variable)
                and left.object.name == "self"
                and left.attribute == "id"
            ):
                is_self_id = True
        elif isinstance(left, ast.FunctionCall):
            # str(self.id)
            if (
                isinstance(left.function, ast.Variable)
                and left.function.name == "str"
                and len(left.arguments) == 1
            ):
                arg = left.arguments[0]
                if isinstance(arg, ast.AttributeAccess):
                    if (
                        isinstance(arg.object, ast.Variable)
                        and arg.object.name == "self"
                        and arg.attribute == "id"
                    ):
                        is_self_id = True

        if not is_self_id:
            return None

        # check right side; could be inline list or context variable
        if isinstance(node.right, ast.ListLiteral):
            # inline list: self.id in [UUID(...), ...]
            uuids: set[UUID] = set()
            for elem in node.right.elements:
                # handle UUID(...) function calls
                if isinstance(elem, ast.FunctionCall):
                    if (
                        isinstance(elem.function, ast.Variable)
                        and elem.function.name == "UUID"
                        and len(elem.arguments) == 1
                    ):
                        arg = elem.arguments[0]
                        if isinstance(arg, ast.Literal) and isinstance(arg.value, str):
                            try:
                                uuids.add(UUID(arg.value))
                            except ValueError, AttributeError:
                                pass
                # handle string literals (for str(self.id) pattern)
                elif isinstance(elem, ast.Literal) and isinstance(elem.value, str):
                    try:
                        uuids.add(UUID(elem.value))
                    except ValueError, AttributeError:
                        pass

            if uuids:
                return uuids

        elif isinstance(node.right, ast.Variable) and context:
            # context variable: self.id in allowed_items
            var_name = node.right.name
            if var_name in context:
                value = context[var_name]
                # check if it's a set or list of UUIDs
                if isinstance(value, set | list):
                    uuids = set()
                    for item in value:
                        if isinstance(item, UUID):
                            uuids.add(item)
                        elif isinstance(item, str):
                            try:
                                uuids.add(UUID(item))
                            except ValueError, AttributeError:
                                pass
                    if uuids:
                        return uuids

    return None


def infer_widget_type(
    constraint_ids: list[UUID],
    constraints: dict[UUID, Constraint],
) -> str:
    """Infer UI widget type from slot constraints.

    Analyzes the constraint DSL expressions to determine the most appropriate
    UI widget for collecting user input.

    Widget type inference logic:
    - Constraint with pattern "self.id in [uuid1, uuid2, ...]": "dropdown"
    - Other DSL expressions: "text"
    - No constraints: "text"

    Parameters
    ----------
    constraint_ids : list[UUID]
        List of constraint IDs for the slot.
    constraints : dict[UUID, Constraint]
        Dictionary of constraint objects keyed by UUID.

    Returns
    -------
    str
        Widget type: "dropdown" or "text".

    Examples
    --------
    >>> from bead.resources.constraints import Constraint
    >>> from uuid import uuid4
    >>> constraint_id = uuid4()
    >>> id1, id2 = uuid4(), uuid4()
    >>> constraint = Constraint(expression=f"self.id in [UUID('{id1}'), UUID('{id2}')]")
    >>> widget = infer_widget_type([constraint_id], {constraint_id: constraint})
    >>> widget
    'dropdown'
    >>> widget2 = infer_widget_type([], {})
    >>> widget2
    'text'
    """
    if not constraint_ids:
        return "text"

    # check each constraint for extensional pattern
    for constraint_id in constraint_ids:
        if constraint_id not in constraints:
            continue

        constraint = constraints[constraint_id]

        # try to extract allowed items from expression (with context)
        allowed_items = _extract_allowed_items_from_expression(
            constraint.expression, constraint.context
        )
        if allowed_items:
            # found a fixed set of allowed items; use dropdown
            return "dropdown"

    # default to text input
    return "text"
