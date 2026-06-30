"""Template filling with backtracking search and constraint propagation.

This module implements a CSP (Constraint Satisfaction Problem) solver for
template filling. It uses backtracking search with forward checking to
efficiently find valid slot fillings that satisfy all constraints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.data.language_codes import LanguageCode, validate_iso639_code
from bead.dsl import ast
from bead.dsl.parser import parse
from bead.resources.lexical_item import LexicalItem
from bead.resources.template import Template
from bead.templates.renderers import DefaultRenderer, TemplateRenderer
from bead.templates.resolver import ConstraintResolver

if TYPE_CHECKING:
    from bead.resources.lexicon import Lexicon


class TemplateFiller(ABC):
    """Abstract base class for template filling.

    Subclasses implement different approaches to filling template slots
    with lexical items from a lexicon. Strategies include constraint
    satisfaction solving (CSP) and enumeration-based strategies.

    Examples
    --------
    >>> from bead.templates.filler import CSPFiller
    >>> filler = CSPFiller(lexicon)
    >>> filled = list(filler.fill(template))
    """

    @abstractmethod
    def fill(
        self,
        template: Template,
        language_code: LanguageCode | None = None,
    ) -> Iterable[FilledTemplate]:
        """Fill template slots with lexical items.

        Parameters
        ----------
        template : Template
            Template to fill.
        language_code : LanguageCode | None
            Optional language code to filter items.

        Returns
        -------
        Iterable[FilledTemplate]
            Filled template instances (iterator or list).

        Raises
        ------
        ValueError
            If template cannot be filled.
        """
        pass


class ConstraintUnsatisfiableError(Exception):
    """Raised when template constraints cannot be satisfied.

    This error indicates that the backtracking search exhausted all
    possibilities without finding a valid assignment.

    Attributes
    ----------
    template_name : str
        Name of the template that could not be filled.
    slot_name : str | None
        Name of the slot where filling failed (if known).
    attempted_combinations : int
        Number of partial assignments tried before failure.
    message : str
        Diagnostic message explaining the failure.

    Examples
    --------
    >>> raise ConstraintUnsatisfiableError(
    ...     template_name="transitive",
    ...     slot_name="verb",
    ...     attempted_combinations=1523,
    ...     message="No VERB items satisfy agreement constraints"
    ... )
    """

    def __init__(
        self,
        template_name: str,
        slot_name: str | None = None,
        attempted_combinations: int = 0,
        message: str = "Could not satisfy all constraints",
    ) -> None:
        self.template_name = template_name
        self.slot_name = slot_name
        self.attempted_combinations = attempted_combinations
        self.message = message
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format diagnostic error message."""
        parts = [f"Template '{self.template_name}': {self.message}"]
        if self.slot_name:
            parts.append(f"Failed at slot: {self.slot_name}")
        if self.attempted_combinations > 0:
            parts.append(f"Tried {self.attempted_combinations} combinations")
        return ". ".join(parts)


class FilledTemplate(BeadBaseModel):
    """A template populated with lexical items.

    Represents a single instance of a template with specific
    items filling each slot.

    Attributes
    ----------
    template_id : str
        ID of the source template.
    template_name : str
        Name of the source template.
    slot_fillers : dict[str, LexicalItem]
        Mapping of slot names to items that fill them.
    rendered_text : str
        Template string with slots replaced by item lemmas.
    strategy_name : str
        Name of strategy used to generate this filled template.
    template_slots : dict[str, bool]
        Mapping of all template slot names to whether they are required.
        Used to determine unfilled slots.

    Examples
    --------
    >>> filled = FilledTemplate(
    ...     template_id="t1",
    ...     template_name="transitive",
    ...     slot_fillers={"subject": noun_item, "verb": verb_item},
    ...     rendered_text="cat broke the object",
    ...     strategy_name="exhaustive",
    ...     template_slots={"subject": True, "verb": True, "object": True}
    ... )
    """

    template_id: str
    template_name: str
    slot_fillers: dict[str, dx.Embed[LexicalItem]]
    rendered_text: str
    strategy_name: str = "exhaustive"
    template_slots: dict[str, bool] = dx.field(default_factory=dict)

    @property
    def unfilled_slots(self) -> frozenset[str]:
        """Names of slots present in the template but not in ``slot_fillers``."""
        return frozenset(self.template_slots.keys()) - frozenset(
            self.slot_fillers.keys()
        )

    @property
    def unfilled_required_slots(self) -> frozenset[str]:
        """Names of required slots that were not filled."""
        return frozenset(
            slot_name
            for slot_name, is_required in self.template_slots.items()
            if is_required and slot_name not in self.slot_fillers
        )

    @property
    def is_complete(self) -> bool:
        """Check if all required slots are filled.

        Returns
        -------
        bool
            True if all required slots have fillers.
        """
        return len(self.unfilled_required_slots) == 0


class CSPFiller(TemplateFiller):
    """Fill templates using backtracking search with forward checking.

    Implements a CSP (Constraint Satisfaction Problem) solver with these guarantees:
    1. Completeness: Will find a solution if one exists
    2. Correctness: All returned assignments satisfy all constraints
    3. Termination: Will halt (either with solution or error)

    The algorithm uses:
    - Backtracking search to explore assignment space
    - Forward checking to prune search space early
    - Most-constrained-first slot ordering heuristic
    - Constraint propagation for multi-slot constraints

    Use this filler when templates have multi-slot constraints (Template.constraints)
    that require agreement or relational checking. For simple templates with only
    single-slot constraints, StrategyFiller is 10-100x faster.

    Parameters
    ----------
    lexicon : Lexicon
        Lexicon containing candidate items.
    max_attempts : int
        Maximum number of partial assignments to try (default: 10000).
    renderer : TemplateRenderer | None
        Template renderer to use for generating rendered_text. If None,
        uses DefaultRenderer() which does simple slot substitution.

    Examples
    --------
    >>> from bead.resources.lexicon import Lexicon
    >>> from bead.templates.filler import CSPFiller
    >>> lexicon = Lexicon(items=[...])
    >>> filler = CSPFiller(lexicon)
    >>> try:
    ...     filled = next(filler.fill(template))
    ... except ConstraintUnsatisfiableError as e:
    ...     print(f"Could not fill: {e}")
    """

    def __init__(
        self,
        lexicon: Lexicon,
        max_attempts: int = 10000,
        renderer: TemplateRenderer | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.max_attempts = max_attempts
        self.resolver = ConstraintResolver()
        self.renderer = renderer if renderer is not None else DefaultRenderer()

    def fill(
        self,
        template: Template,
        language_code: LanguageCode | None = None,
        count: int = 1,
    ) -> Iterator[FilledTemplate]:
        """Fill template with lexical items using backtracking search.

        Yields filled templates one at a time as they are found.
        Stops after yielding `count` templates or exhausting possibilities.

        Parameters
        ----------
        template : Template
            Template to fill.
        language_code : LanguageCode | None
            Optional language code to filter lexicon items.
        count : int
            Maximum number of filled templates to generate (default: 1).

        Yields
        ------
        FilledTemplate
            Filled template instance satisfying all constraints.

        Raises
        ------
        ConstraintUnsatisfiableError
            If no valid assignment exists after exhaustive search.
        ValueError
            If template has no slots or invalid structure.

        Examples
        --------
        >>> filler = CSPFiller(lexicon)
        >>> # Get first valid filling
        >>> filled = next(filler.fill(template))
        >>> # Get up to 10 different fillings
        >>> fillings = list(filler.fill(template, count=10))
        """
        if not template.slots:
            raise ValueError(f"Template '{template.name}' has no slots")

        # 1. Build candidate pools for each slot
        candidate_pools = self._build_candidate_pools(template, language_code)

        # 2. Check for empty pools
        empty_slots = [name for name, pool in candidate_pools.items() if not pool]
        if empty_slots:
            raise ConstraintUnsatisfiableError(
                template_name=template.name,
                slot_name=empty_slots[0],
                message=f"No valid candidates for slot(s): {', '.join(empty_slots)}",
            )

        # 3. Determine slot ordering (most constrained first)
        slot_order = self._order_slots(template, candidate_pools)

        # 4. Run backtracking search
        generated = 0
        attempt_count = [0]  # Use list to make it mutable in nested function

        for filled in self._backtrack_search(
            template, candidate_pools, slot_order, {}, attempt_count
        ):
            yield filled
            generated += 1
            if generated >= count:
                return

        # If we got here, we didn't find enough solutions
        if generated == 0:
            raise ConstraintUnsatisfiableError(
                template_name=template.name,
                attempted_combinations=attempt_count[0],
                message="Exhausted all possibilities without finding valid assignment",
            )

    def _build_candidate_pools(
        self, template: Template, language_code: LanguageCode | None = None
    ) -> dict[str, list[LexicalItem]]:
        """Build candidate pools for each slot.

        For each slot, get all lexicon items that satisfy the slot's
        single-slot constraints.

        Parameters
        ----------
        template : Template
            Template with slots and constraints.
        language_code : LanguageCode | None
            Optional language code to filter items.

        Returns
        -------
        dict[str, list[LexicalItem]]
            Mapping of slot names to candidate items.
        """
        # Normalize language code if provided
        normalized_lang = validate_iso639_code(language_code) if language_code else None

        candidate_pools: dict[str, list[LexicalItem]] = {}

        for slot_name, slot in template.slots.items():
            candidates: list[LexicalItem] = []
            for item in self.lexicon.items:
                # Filter by language code if specified
                if normalized_lang:
                    # Normalize item language code for comparison
                    item_lang = (
                        validate_iso639_code(item.language_code)
                        if item.language_code
                        else None
                    )
                    if item_lang != normalized_lang:
                        continue

                # Check if item satisfies slot constraints
                if self.resolver.evaluate_slot_constraints(item, slot.constraints):
                    candidates.append(item)
            candidate_pools[slot_name] = candidates

        return candidate_pools

    def _order_slots(
        self, template: Template, candidate_pools: dict[str, list[LexicalItem]]
    ) -> list[str]:
        """Order slots using most-constrained-first heuristic.

        Slots with fewer candidates are filled first to fail fast
        and prune the search space earlier.

        Parameters
        ----------
        template : Template
            Template with slots.
        candidate_pools : dict[str, list[LexicalItem]]
            Candidate items for each slot.

        Returns
        -------
        list[str]
            Slot names in optimal filling order.
        """

        # Sort slots by:
        # 1. Number of candidates (fewer first, most constrained)
        # 2. Number of constraints (more first, more likely to fail)
        # 3. Alphabetical (for determinism)
        def slot_key(slot_name: str) -> tuple[int, int, str]:
            num_candidates = len(candidate_pools[slot_name])
            num_constraints = len(template.slots[slot_name].constraints)
            return (num_candidates, -num_constraints, slot_name)

        return sorted(template.slots.keys(), key=slot_key)

    def _backtrack_search(
        self,
        template: Template,
        candidate_pools: dict[str, list[LexicalItem]],
        slot_order: list[str],
        assignment: dict[str, LexicalItem],
        attempt_count: list[int],
    ) -> Iterator[FilledTemplate]:
        """Backtracking search with forward checking.

        Recursively fill slots one at a time, checking constraints
        at each step to prune invalid branches early.

        Parameters
        ----------
        template : Template
            Template being filled.
        candidate_pools : dict[str, list[LexicalItem]]
            Candidate items for each slot.
        slot_order : list[str]
            Order in which to fill slots.
        assignment : dict[str, LexicalItem]
            Current partial assignment.
        attempt_count : list[int]
            Mutable counter for number of attempts.

        Yields
        ------
        FilledTemplate
            Valid complete assignments.
        """
        # Check attempt limit
        if attempt_count[0] >= self.max_attempts:
            return

        # Base case: all slots filled
        if len(assignment) == len(slot_order):
            # Check template level multi slot constraints
            if self.resolver.evaluate_template_constraints(
                assignment, template.constraints
            ):
                yield self._create_filled_template(template, assignment)
            return

        # Recursive case: fill next slot
        slot_name = slot_order[len(assignment)]
        slot = template.slots[slot_name]

        for candidate in candidate_pools[slot_name]:
            attempt_count[0] += 1

            # Check single slot constraints
            if not self.resolver.evaluate_slot_constraints(candidate, slot.constraints):
                continue

            # Create extended assignment
            extended_assignment = {**assignment, slot_name: candidate}

            # Forward checking: check partial multi slot constraints
            if not self._check_partial_constraints(
                template, extended_assignment, slot_order[: len(extended_assignment)]
            ):
                continue

            # Recurse with extended assignment
            yield from self._backtrack_search(
                template,
                candidate_pools,
                slot_order,
                extended_assignment,
                attempt_count,
            )

    def _check_partial_constraints(
        self,
        template: Template,
        partial_assignment: dict[str, LexicalItem],
        filled_slots: list[str],
    ) -> bool:
        """Check if partial assignment satisfies applicable constraints.

        Only check constraints that involve only slots that have been
        filled so far (forward checking optimization). This method parses
        the constraint AST to determine which variables are referenced.

        Parameters
        ----------
        template : Template
            Template with constraints.
        partial_assignment : dict[str, LexicalItem]
            Current partial assignment.
        filled_slots : list[str]
            Names of slots that have been filled.

        Returns
        -------
        bool
            True if all applicable constraints are satisfied.
        """
        filled_set = set(filled_slots)

        for constraint in template.constraints:
            ast_node = parse(constraint.expression)

            # Extract all variable names referenced in the expression
            referenced_vars = self._extract_variables(ast_node)

            # Filter to only slot related variables (exclude context variables)
            referenced_slots = referenced_vars - set(constraint.context.keys())

            # Check if all referenced slots have been filled
            if not referenced_slots.issubset(filled_set):
                # Some referenced slots haven't been filled yet; skip this constraint
                continue

            # All referenced slots are filled; evaluate the constraint
            if not self.resolver.evaluate_template_constraints(
                partial_assignment, [constraint]
            ):
                return False

        return True

    def _extract_variables(self, node: ast.ASTNode) -> set[str]:
        """Extract all variable names from an AST node.

        Recursively traverses the AST to find all Variable nodes.

        Parameters
        ----------
        node : ast.ASTNode
            AST node to traverse.

        Returns
        -------
        set[str]
            Set of all variable names referenced in the expression.
        """
        variables: set[str] = set()

        if isinstance(node, ast.Variable):
            variables.add(node.name)
        elif isinstance(node, ast.BinaryOp):
            variables.update(self._extract_variables(node.left))
            variables.update(self._extract_variables(node.right))
        elif isinstance(node, ast.UnaryOp):
            variables.update(self._extract_variables(node.operand))
        elif isinstance(node, ast.FunctionCall):
            # Extract from function (Variable or AttributeAccess for methods)
            variables.update(self._extract_variables(node.function))
            # Extract from arguments
            for arg in node.arguments:
                variables.update(self._extract_variables(arg))
        elif isinstance(node, ast.AttributeAccess):
            variables.update(self._extract_variables(node.object))
        elif isinstance(node, ast.Subscript):
            variables.update(self._extract_variables(node.object))
            variables.update(self._extract_variables(node.index))
        elif isinstance(node, ast.ListLiteral):
            for element in node.elements:
                variables.update(self._extract_variables(element))
        # Literal nodes don't contain variables

        return variables

    def _create_filled_template(
        self, template: Template, assignment: dict[str, LexicalItem]
    ) -> FilledTemplate:
        """Create FilledTemplate from assignment.

        Parameters
        ----------
        template : Template
            Source template.
        assignment : dict[str, LexicalItem]
            Complete slot assignment.

        Returns
        -------
        FilledTemplate
            Filled template instance.
        """
        # Render template string using renderer
        rendered = self.renderer.render(
            template.template_string, assignment, template.slots
        )

        return FilledTemplate(
            template_id=str(template.id),
            template_name=template.name,
            slot_fillers=assignment.copy(),
            rendered_text=rendered,
            strategy_name="backtracking",
            template_slots={
                name: slot.required for name, slot in template.slots.items()
            },
        )
