"""Constraint resolution for template slot filling.

This module provides the ConstraintResolver class for evaluating constraints
against lexical items to determine which items satisfy template slot requirements.
The resolver is now a thin wrapper around DSLEvaluator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bead.dsl.evaluator import DSLEvaluator
from bead.resources.constraints import Constraint

if TYPE_CHECKING:
    from bead.resources.constraints import ContextValue
    from bead.resources.lexical_item import LexicalItem


class ConstraintResolver:
    """Simplified resolver that wraps DSLEvaluator.

    The ConstraintResolver evaluates DSL constraint expressions against
    lexical items and filled slots. It provides two main methods:
    - evaluate_slot_constraints: for single-slot constraints
    - evaluate_template_constraints: for multi-slot constraints

    All constraint logic is delegated to the DSL evaluator.

    Examples
    --------
    >>> from bead.resources.models import LexicalItem
    >>> from bead.resources.constraints import Constraint
    >>> resolver = ConstraintResolver()
    >>> item = LexicalItem(lemma="walk", pos="VERB")
    >>> constraints = [
    ...     Constraint(expression="self.pos == 'VERB'")
    ... ]
    >>> resolver.evaluate_slot_constraints(item, constraints)
    True
    """

    def __init__(self) -> None:
        """Initialize resolver with DSL evaluator."""
        self.dsl_evaluator = DSLEvaluator()

    def evaluate_slot_constraints(
        self,
        item: LexicalItem,
        constraints: list[Constraint] | tuple[Constraint, ...],
        context: dict[str, ContextValue] | None = None,
    ) -> bool:
        """Evaluate single-slot constraints.

        Single-slot constraints are evaluated with 'self' referring to
        the lexical item being checked. The item is available as 'self'
        in the DSL expression context.

        Parameters
        ----------
        item : LexicalItem
            Lexical item to evaluate constraints against.
        constraints : list[Constraint]
            Single-slot constraints from Slot.constraints.
        context : dict[str, ContextValue] | None
            Additional context variables (optional).

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.

        Examples
        --------
        >>> from bead.resources.models import LexicalItem
        >>> from bead.resources.constraints import Constraint
        >>> resolver = ConstraintResolver()
        >>> item = LexicalItem(lemma="walk", pos="VERB")
        >>> constraints = [
        ...     Constraint(
        ...         expression="self.lemma in motion_verbs",
        ...         context={"motion_verbs": {"walk", "run", "jump"}}
        ...     )
        ... ]
        >>> resolver.evaluate_slot_constraints(item, constraints)
        True
        """
        for constraint in constraints:
            # build evaluation context
            eval_context: dict[str, Any] = {
                "self": item,
                **constraint.context,
            }
            if context:
                eval_context.update(context)

            # evaluate constraint
            result = self.dsl_evaluator.evaluate(constraint.expression, eval_context)
            if not result:
                return False

        return True

    def evaluate_template_constraints(
        self,
        filled_slots: dict[str, LexicalItem],
        constraints: list[Constraint] | tuple[Constraint, ...],
        context: dict[str, ContextValue] | None = None,
    ) -> bool:
        """Evaluate multi-slot constraints.

        Multi-slot constraints are evaluated with slot names as variables.
        For example, a constraint like "subject.features.number == verb.features.number"
        would have access to both the 'subject' and 'verb' slot fillers.

        Parameters
        ----------
        filled_slots : dict[str, LexicalItem]
            Dictionary mapping slot names to their filled items.
        constraints : list[Constraint]
            Multi-slot constraints from Template.constraints.
        context : dict[str, ContextValue] | None
            Additional context variables (optional).

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.

        Examples
        --------
        >>> from bead.resources.models import LexicalItem
        >>> from bead.resources.constraints import Constraint
        >>> resolver = ConstraintResolver()
        >>> subject = LexicalItem(
        ...     lemma="dog",
        ...     pos="NOUN",
        ...     features={"number": "singular"}
        ... )
        >>> verb = LexicalItem(
        ...     lemma="runs",
        ...     pos="VERB",
        ...     features={"number": "singular"}
        ... )
        >>> filled = {"subject": subject, "verb": verb}
        >>> constraints = [
        ...     Constraint(
        ...         expression="subject.features.number == verb.features.number"
        ...     )
        ... ]
        >>> resolver.evaluate_template_constraints(filled, constraints)
        True
        """
        for constraint in constraints:
            # build evaluation context with slot fillers
            eval_context: dict[str, Any] = {
                **filled_slots,
                **constraint.context,
            }
            if context:
                eval_context.update(context)

            # evaluate constraint
            result = self.dsl_evaluator.evaluate(constraint.expression, eval_context)
            if not result:
                return False

        return True

    def clear_cache(self) -> None:
        """Clear DSL evaluator's compiled expression cache.

        Examples
        --------
        >>> resolver = ConstraintResolver()
        >>> resolver.clear_cache()
        """
        self.dsl_evaluator.clear_cache()
