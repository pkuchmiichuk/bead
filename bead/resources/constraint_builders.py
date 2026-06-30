"""Abstract base classes for programmatic constraint generation.

This module provides language-agnostic base classes for building constraints
programmatically. Language-specific implementations should extend these bases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bead.resources.constraints import Constraint, ContextValue


class ConstraintBuilder(ABC):
    """Abstract base class for programmatic constraint generation.

    Constraint builders encapsulate logic for generating DSL constraints
    based on configuration and rules. Subclasses implement specific
    constraint generation strategies.

    Examples
    --------
    >>> class NumberAgreementBuilder(ConstraintBuilder):
    ...     def build(self, *slot_names: str) -> Constraint:
    ...         # Generate number agreement constraint
    ...         pairs = []
    ...         for i, slot1 in enumerate(slot_names):
    ...             for slot2 in slot_names[i+1:]:
    ...                 pairs.append(f"{slot1}.number == {slot2}.number")
    ...         return Constraint(
    ...             expression=" and ".join(pairs),
    ...             description=f"Number agreement: {', '.join(slot_names)}"
    ...         )
    """

    @abstractmethod
    def build(self, *args: Any, **kwargs: Any) -> Constraint:
        """Build a Constraint object.

        Parameters
        ----------
        *args : Any
            Positional arguments (slot names, properties, etc.).
        **kwargs : Any
            Keyword arguments (configuration options).

        Returns
        -------
        Constraint
            Generated constraint.
        """
        ...


class AgreementConstraintBuilder(ConstraintBuilder):
    """Builder for feature agreement constraints.

    Generates constraints that enforce feature agreement across slots
    (e.g., number, gender, case). Supports exact matching or equivalence
    classes via agreement rules.

    Parameters
    ----------
    feature_name : str
        Name of the feature to enforce agreement on (e.g., "number", "gender").
    agreement_rules : dict[str, list[str]] | None
        Optional equivalence classes. Maps canonical value to list of
        equivalent values. For example:
        {"singular": ["singular", "sing", "sg"], "plural": ["plural", "pl"]}

    Examples
    --------
    Exact number agreement:
    >>> builder = AgreementConstraintBuilder("number")
    >>> constraint = builder.build("subject", "verb")
    >>> expr = "subject.features.get('number') == verb.features.get('number')"
    >>> expr in constraint.expression
    True

    Agreement with equivalence rules:
    >>> rules = {"singular": ["sing", "sg"], "plural": ["pl"]}
    >>> builder = AgreementConstraintBuilder("number", agreement_rules=rules)
    >>> constraint = builder.build("det", "noun")
    >>> "equiv_" in constraint.expression  # Uses equivalence class checks
    True
    """

    def __init__(
        self,
        feature_name: str,
        *,
        agreement_rules: dict[str, list[str]] | None = None,
    ) -> None:
        self.feature_name = feature_name
        self.agreement_rules = agreement_rules

    def build(self, *slot_names: str) -> Constraint:
        """Build agreement constraint for given slots.

        Parameters
        ----------
        *slot_names : str
            Names of slots to enforce agreement between (≥2 required).

        Returns
        -------
        Constraint
            Agreement constraint.

        Raises
        ------
        ValueError
            If fewer than 2 slot names provided.
        """
        if len(slot_names) < 2:
            raise ValueError("Agreement requires at least 2 slot names")

        if self.agreement_rules:
            return self._build_with_rules(slot_names)
        else:
            return self._build_exact_match(slot_names)

    def _build_exact_match(self, slot_names: tuple[str, ...]) -> Constraint:
        """Build exact match agreement constraint."""
        # create pairwise equality checks
        pairs: list[str] = []
        for i, slot1 in enumerate(slot_names):
            for slot2 in slot_names[i + 1 :]:
                left = f"{slot1}.features.get('{self.feature_name}')"
                right = f"{slot2}.features.get('{self.feature_name}')"
                expr = f"{left} == {right}"
                pairs.append(expr)

        expression: str = " and ".join(pairs)
        slot_list = ", ".join(slot_names)
        description = f"{self.feature_name.capitalize()} agreement: {slot_list}"

        return Constraint(expression=expression, description=description)

    def _build_with_rules(self, slot_names: tuple[str, ...]) -> Constraint:
        """Build agreement constraint with equivalence classes."""
        # build context with equivalence class sets
        context: dict[str, Any] = {}
        for canonical, variants in self.agreement_rules.items():  # type: ignore
            context[f"equiv_{canonical}"] = tuple(sorted(set(variants)))

        # build expression: check if all slots' values are in same equivalence class
        equiv_checks: list[str] = []
        for canonical in self.agreement_rules.keys():  # type: ignore
            # all slots must have values in this equivalence class
            slot_checks: list[str] = [
                f"{slot}.features.get('{self.feature_name}') in equiv_{canonical}"
                for slot in slot_names
            ]
            equiv_checks.append(f"({' and '.join(slot_checks)})")

        expression: str = " or ".join(equiv_checks)
        slot_list = ", ".join(slot_names)
        feat_name = self.feature_name.capitalize()
        description = f"{feat_name} agreement with rules: {slot_list}"

        return Constraint(
            expression=expression, context=context, description=description
        )


class ConditionalConstraintBuilder(ConstraintBuilder):
    """Builder for IF-THEN (conditional) constraints.

    Generates constraints that enforce requirements when conditions are met.
    Implements logical implication: IF condition THEN requirement.

    Examples
    --------
    >>> builder = ConditionalConstraintBuilder()
    >>> constraint = builder.build(
    ...     condition="det.lemma == 'a'",
    ...     requirement="noun.features.get('number') == 'singular'",
    ...     description="'a' requires singular noun"
    ... )
    >>> "not (" in constraint.expression  # IF-THEN encoded as: not cond or req
    True
    """

    def build(
        self,
        *,
        condition: str,
        requirement: str,
        description: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Constraint:
        """Build conditional constraint.

        Parameters
        ----------
        condition : str
            Condition expression (IF part).
        requirement : str
            Requirement expression (THEN part).
        description : str | None
            Human-readable description.
        context : dict[str, Any] | None
            Context variables for evaluation.

        Returns
        -------
        Constraint
            Conditional constraint.

        Notes
        -----
        Logical implication (IF A THEN B) is encoded as: (NOT A) OR B
        """
        # encode IF-THEN as: (NOT condition) OR requirement
        expression = f"not ({condition}) or ({requirement})"

        return Constraint(
            expression=expression,
            context=context or {},
            description=description,
        )


class SetMembershipConstraintBuilder(ConstraintBuilder):
    """Builder for whitelist/blacklist constraints.

    Generates constraints that restrict slot properties to allowed values
    (whitelist) or exclude forbidden values (blacklist).

    Parameters
    ----------
    slot_name : str
        Name of slot to constrain.
    property_path : str
        Dot-separated path to property (e.g., "lemma", "features.number").
    allowed_values : set | None
        Whitelist of allowed values (mutually exclusive with forbidden_values).
    forbidden_values : set | None
        Blacklist of forbidden values.
    description : str | None
        Custom description.

    Examples
    --------
    Whitelist constraint:
    >>> builder = SetMembershipConstraintBuilder()
    >>> constraint = builder.build(
    ...     slot_name="verb",
    ...     property_path="lemma",
    ...     allowed_values={"walk", "run", "jump"},
    ...     description="Motion verbs only"
    ... )
    >>> "verb.lemma in allowed_values" in constraint.expression
    True

    Blacklist constraint:
    >>> constraint = builder.build(
    ...     slot_name="verb",
    ...     property_path="lemma",
    ...     forbidden_values={"be", "have"},
    ...     description="Exclude copula and auxiliary"
    ... )
    >>> "verb.lemma not in forbidden_values" in constraint.expression
    True
    """

    def build(
        self,
        *,
        slot_name: str,
        property_path: str,
        allowed_values: set[str] | None = None,
        forbidden_values: set[str] | None = None,
        description: str | None = None,
    ) -> Constraint:
        """Build set membership constraint.

        Parameters
        ----------
        slot_name : str
            Slot to constrain.
        property_path : str
            Property path within slot.
        allowed_values : set | None
            Whitelist of allowed values.
        forbidden_values : set | None
            Blacklist of forbidden values.
        description : str | None
            Constraint description.

        Returns
        -------
        Constraint
            Set membership constraint.

        Raises
        ------
        ValueError
            If neither or both of allowed_values/forbidden_values provided.
        """
        # exactly one of allowed_values or forbidden_values must be provided
        if (allowed_values is None) == (forbidden_values is None):
            raise ValueError(
                "Exactly one of 'allowed_values' or 'forbidden_values' must be provided"
            )

        expression: str
        context: dict[str, ContextValue]

        if allowed_values is not None:
            expression = f"{slot_name}.{property_path} in allowed_values"
            context = {"allowed_values": tuple(sorted(allowed_values))}
            if description is None:
                prop_path = f"{slot_name}.{property_path}"
                description = f"Restrict {prop_path} to allowed values"
        else:
            assert forbidden_values is not None
            expression = f"{slot_name}.{property_path} not in forbidden_values"
            context = {"forbidden_values": tuple(sorted(forbidden_values))}
            if description is None:
                prop_path = f"{slot_name}.{property_path}"
                description = f"Exclude {prop_path} from forbidden values"

        return Constraint(
            expression=expression, context=context, description=description
        )
