"""Constraint models for lexical item selection.

Universal constraint model based on DSL expressions. Each constraint is a
DSL expression with optional context variables; scope is determined by
storage location:

- ``Slot.constraints`` — single-slot constraints (``self`` = slot filler).
- ``Template.constraints`` — multi-slot constraints (slot names as
  variables).
- ``TemplateSequence.constraints`` — cross-template constraints.
"""

from __future__ import annotations

import didactic.api as dx

from bead.data.base import BeadBaseModel, JsonValue

type ContextValue = JsonValue


class Constraint(BeadBaseModel):
    """Universal constraint expressed via a DSL expression.

    Attributes
    ----------
    expression : str
        DSL expression (must return a boolean) evaluated against the
        context.
    context : dict[str, ContextValue]
        Context variables available during evaluation. Values are
        JSON-shaped (scalars, lists, dicts); the DSL evaluator coerces
        list values into sets when the surrounding expression uses them
        as a membership test.
    description : str | None
        Optional human-readable description.
    """

    expression: str
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    description: str | None = None

    @classmethod
    def combine(
        cls,
        *constraints: Constraint,
        logic: str = "and",
    ) -> Constraint:
        """Combine multiple constraints into one with AND or OR logic.

        Parameters
        ----------
        *constraints
            Constraints to combine.
        logic
            ``"and"`` or ``"or"``.

        Returns
        -------
        Constraint
            New constraint with combined expressions and merged contexts.

        Raises
        ------
        ValueError
            If no constraints are provided or *logic* is invalid.
        """
        if not constraints:
            raise ValueError("Must provide at least one constraint")
        if logic not in ("and", "or"):
            raise ValueError(f"Invalid logic operator '{logic}'. Must be 'and' or 'or'")
        if len(constraints) == 1:
            return constraints[0]

        expressions = [f"({c.expression})" for c in constraints]
        combined_expression = f" {logic} ".join(expressions)

        combined_context: dict[str, ContextValue] = {}
        for constraint in constraints:
            if constraint.context:
                combined_context.update(constraint.context)

        descriptions = [c.description for c in constraints if c.description]
        combined_description = "; ".join(descriptions) if descriptions else None

        return cls(
            expression=combined_expression,
            context=combined_context,
            description=combined_description,
        )
