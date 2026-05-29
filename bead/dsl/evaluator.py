"""Constraint evaluator for DSL.

This module provides the Evaluator class that executes AST nodes
against an evaluation context to produce boolean results, and the
DSLEvaluator class that provides a high-level interface for evaluating
constraint expressions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bead.dsl import ast
from bead.dsl.context import EvaluationContext
from bead.dsl.errors import EvaluationError
from bead.dsl.parser import parse
from bead.dsl.stdlib import register_stdlib

if TYPE_CHECKING:
    from bead.data.base import BeadBaseModel, JsonValue
    from bead.resources.constraints import ContextValue

# Every value an expression can produce or operate on: DSL scalars, collections,
# and bead model objects (reached via attribute access on a bound ``self`` /
# ``item``). Attribute and subscript access ultimately bottom out in model
# fields, which are themselves of this shape.
type DslValue = (
    str
    | int
    | float
    | bool
    | None
    | list["DslValue"]
    | tuple["DslValue", ...]
    | dict[str, "DslValue"]
    | set["DslValue"]
    | frozenset["DslValue"]
    | BeadBaseModel
    | JsonValue
)


def _compare(operator: str, left: DslValue, right: DslValue) -> bool:
    """Apply an ordering operator to two numeric or two string operands."""
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        lf, rf = float(left), float(right)
        if operator == "<":
            return lf < rf
        if operator == ">":
            return lf > rf
        if operator == "<=":
            return lf <= rf
        return lf >= rf
    if isinstance(left, str) and isinstance(right, str):
        if operator == "<":
            return left < right
        if operator == ">":
            return left > right
        if operator == "<=":
            return left <= right
        return left >= right
    raise EvaluationError(
        f"Cannot compare {type(left).__name__} and {type(right).__name__}"
    )


def _arithmetic(operator: str, left: DslValue, right: DslValue) -> int | float | str:
    """Apply an arithmetic operator, preserving int/float/str result types."""
    if isinstance(left, int) and isinstance(right, int):
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/":
            if right == 0:
                raise EvaluationError("Division by zero")
            return left / right
        if right == 0:
            raise EvaluationError("Modulo by zero")
        return left % right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        lf, rf = float(left), float(right)
        if operator == "+":
            return lf + rf
        if operator == "-":
            return lf - rf
        if operator == "*":
            return lf * rf
        if operator == "/":
            if rf == 0:
                raise EvaluationError("Division by zero")
            return lf / rf
        if rf == 0:
            raise EvaluationError("Modulo by zero")
        return lf % rf
    if operator == "+" and isinstance(left, str) and isinstance(right, str):
        return left + right
    raise EvaluationError(
        f"Cannot apply '{operator}' to "
        f"{type(left).__name__} and {type(right).__name__}"
    )


def _contains(left: DslValue, right: DslValue) -> bool:
    """Test membership of ``left`` in a container ``right``."""
    if isinstance(right, str):
        if isinstance(left, str):
            return left in right
        raise EvaluationError("Substring test requires a string on the left")
    if isinstance(right, (list, tuple, set, frozenset, dict)):
        return left in right
    raise EvaluationError(
        f"Membership test requires a container, got {type(right).__name__}"
    )


class Evaluator:
    """Evaluator for constraint AST nodes.

    The evaluator walks the AST and computes values based on the
    evaluation context. It supports:
    - All AST node types
    - Operator evaluation
    - Function calls
    - Attribute access
    - Caching for performance

    Parameters
    ----------
    use_cache : bool
        Whether to cache evaluation results.

    Examples
    --------
    >>> from bead.dsl.context import EvaluationContext
    >>> from bead.dsl.parser import parse
    >>> ctx = EvaluationContext()
    >>> ctx.set_variable("x", 10)
    >>> evaluator = Evaluator()
    >>> node = parse("x > 5")
    >>> evaluator.evaluate(node, ctx)
    True
    """

    def __init__(self, use_cache: bool = True) -> None:
        self._use_cache = use_cache
        self._cache: dict[tuple[str, ...], DslValue] = {}

    def evaluate(self, node: ast.ASTNode, context: EvaluationContext) -> DslValue:
        """Evaluate an AST node in the given context.

        Parameters
        ----------
        node : ast.ASTNode
            AST node to evaluate.
        context : EvaluationContext
            Evaluation context with variables and functions.

        Returns
        -------
        DslValue
            Result of evaluation.

        Raises
        ------
        EvaluationError
            If evaluation fails (undefined variable, type error, etc.).
        """
        # dispatch to specific evaluation methods
        if isinstance(node, ast.Literal):
            return self._evaluate_literal(node, context)
        elif isinstance(node, ast.Variable):
            return self._evaluate_variable(node, context)
        elif isinstance(node, ast.BinaryOp):
            return self._evaluate_binary_op(node, context)
        elif isinstance(node, ast.UnaryOp):
            return self._evaluate_unary_op(node, context)
        elif isinstance(node, ast.FunctionCall):
            return self._evaluate_function_call(node, context)
        elif isinstance(node, ast.AttributeAccess):
            return self._evaluate_attribute_access(node, context)
        elif isinstance(node, ast.Subscript):
            return self._evaluate_subscript(node, context)
        elif isinstance(node, ast.ListLiteral):
            return self._evaluate_list_literal(node, context)
        else:
            raise EvaluationError(f"Unknown node type: {type(node).__name__}")

    def _evaluate_literal(
        self, node: ast.Literal, context: EvaluationContext
    ) -> DslValue:
        """Evaluate literal node.

        Parameters
        ----------
        node : ast.Literal
            Literal node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        DslValue
            Literal value.
        """
        return node.value

    def _evaluate_variable(
        self, node: ast.Variable, context: EvaluationContext
    ) -> DslValue:
        """Evaluate variable node.

        Parameters
        ----------
        node : ast.Variable
            Variable node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        DslValue
            Variable value from context.

        Raises
        ------
        EvaluationError
            If variable is not defined.
        """
        if not context.has_variable(node.name):
            raise EvaluationError(f"Undefined variable: {node.name}")
        return context.get_variable(node.name)

    def _evaluate_binary_op(
        self, node: ast.BinaryOp, context: EvaluationContext
    ) -> DslValue:
        """Evaluate binary operation node.

        Parameters
        ----------
        node : ast.BinaryOp
            Binary operation node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        DslValue
            Result of binary operation.

        Raises
        ------
        EvaluationError
            If operator is unknown or operation fails.
        """
        # short-circuit evaluation for logical operators
        if node.operator == "and":
            left = self.evaluate(node.left, context)
            if not left:
                return False
            return bool(self.evaluate(node.right, context))
        elif node.operator == "or":
            left = self.evaluate(node.left, context)
            if left:
                return True
            return bool(self.evaluate(node.right, context))

        # evaluate both operands for other operators
        left = self.evaluate(node.left, context)
        right = self.evaluate(node.right, context)

        # equality works on any pair of values
        if node.operator == "==":
            return left == right
        if node.operator == "!=":
            return left != right
        # ordering operators (numeric or string operands)
        if node.operator in ("<", ">", "<=", ">="):
            return _compare(node.operator, left, right)
        # membership operators
        if node.operator == "in":
            return _contains(left, right)
        if node.operator == "not in":
            return not _contains(left, right)
        # arithmetic operators
        if node.operator in ("+", "-", "*", "/", "%"):
            return _arithmetic(node.operator, left, right)
        raise EvaluationError(f"Unknown operator: {node.operator}")

    def _evaluate_unary_op(
        self, node: ast.UnaryOp, context: EvaluationContext
    ) -> DslValue:
        """Evaluate unary operation node.

        Parameters
        ----------
        node : ast.UnaryOp
            Unary operation node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        DslValue
            Result of unary operation.

        Raises
        ------
        EvaluationError
            If operator is unknown or operation fails.
        """
        operand = self.evaluate(node.operand, context)

        if node.operator == "not":
            return not operand
        if node.operator in ("-", "+"):
            if not isinstance(operand, (int, float)):
                raise EvaluationError(
                    f"Unary '{node.operator}' requires a number, got "
                    f"{type(operand).__name__}"
                )
            return -operand if node.operator == "-" else +operand
        raise EvaluationError(f"Unknown unary operator: {node.operator}")

    def _evaluate_function_call(
        self, node: ast.FunctionCall, context: EvaluationContext
    ) -> DslValue:
        """Evaluate function call node.

        Parameters
        ----------
        node : ast.FunctionCall
            Function call node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        DslValue
            Function return value.

        Raises
        ------
        EvaluationError
            If function is not defined or call fails.
        """
        # evaluate arguments
        args = [self.evaluate(arg, context) for arg in node.arguments]

        # handle method calls (e.g., subject.features.get(...))
        if isinstance(node.function, ast.AttributeAccess):
            # evaluate the object
            obj = self.evaluate(node.function.object, context)
            # get the method
            method_name = node.function.attribute
            try:
                method = getattr(obj, method_name)
                return method(*args)
            except AttributeError as e:
                raise EvaluationError(
                    f"Object of type {type(obj).__name__} has no method: {method_name}"
                ) from e
            except TypeError as e:
                raise EvaluationError(f"Error calling method {method_name}: {e}") from e

        # handle regular function calls (e.g., len(...))
        if isinstance(node.function, ast.Variable):
            func_name = node.function.name
            return context.call_function(func_name, args)

        func_type = type(node.function).__name__
        raise EvaluationError(
            f"Function must be a variable or attribute access, got {func_type}"
        )

    def _evaluate_attribute_access(
        self, node: ast.AttributeAccess, context: EvaluationContext
    ) -> DslValue:
        """Evaluate attribute access node.

        Parameters
        ----------
        node : ast.AttributeAccess
            Attribute access node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        DslValue
            Attribute value.

        Raises
        ------
        EvaluationError
            If attribute access fails.
        """
        obj = self.evaluate(node.object, context)

        # try dictionary-style access first
        if isinstance(obj, dict):
            if node.attribute not in obj:
                raise EvaluationError(f"Dictionary does not have key: {node.attribute}")
            return obj[node.attribute]

        # try attribute access
        try:
            return getattr(obj, node.attribute)
        except AttributeError as e:
            raise EvaluationError(
                f"Object of type {type(obj).__name__} has no attribute: "
                f"{node.attribute}"
            ) from e

    def _evaluate_subscript(
        self, node: ast.Subscript, context: EvaluationContext
    ) -> DslValue:
        """Evaluate subscript access node.

        Parameters
        ----------
        node : ast.Subscript
            Subscript access node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        DslValue
            Subscripted value.

        Raises
        ------
        EvaluationError
            If subscript access fails.
        """
        obj = self.evaluate(node.object, context)
        index = self.evaluate(node.index, context)

        try:
            if isinstance(obj, dict):
                if not isinstance(index, str):
                    raise EvaluationError(
                        f"Dictionary index must be a string, got "
                        f"{type(index).__name__}"
                    )
                return obj[index]
            if isinstance(obj, (list, tuple, str)):
                if not isinstance(index, int):
                    raise EvaluationError(
                        f"Sequence index must be an integer, got "
                        f"{type(index).__name__}"
                    )
                return obj[index]
            raise EvaluationError(
                f"Subscript access not supported on {type(obj).__name__}"
            )
        except (KeyError, IndexError) as e:
            obj_type = type(obj).__name__
            raise EvaluationError(
                f"Subscript access failed on {obj_type} with index {index!r}: {e}"
            ) from e

    def _evaluate_list_literal(
        self, node: ast.ListLiteral, context: EvaluationContext
    ) -> list[DslValue]:
        """Evaluate list literal node.

        Parameters
        ----------
        node : ast.ListLiteral
            List literal node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        list[DslValue]
            Evaluated list elements.
        """
        return [self.evaluate(element, context) for element in node.elements]

    def clear_cache(self) -> None:
        """Clear evaluation cache.

        Examples
        --------
        >>> evaluator = Evaluator()
        >>> evaluator.clear_cache()
        """
        self._cache.clear()


class DSLEvaluator:
    """High-level evaluator for DSL constraint expressions.

    This class provides a simplified interface for evaluating constraint
    expressions. It handles:

    - Parsing expression strings to AST
    - Building evaluation contexts from dictionaries
    - Caching compiled ASTs
    - Registering standard library functions
    - Property extraction for list partitioning

    The DSLEvaluator is the primary interface for constraint evaluation
    in the bead package. It wraps the lower-level Evaluator class.

    Attributes
    ----------
    evaluator : Evaluator
        The underlying AST evaluator instance.
    compiled_cache : dict[str, ast.ASTNode]
        Cache mapping expression strings to their compiled AST nodes.

    Examples
    --------
    >>> from bead.resources.items import LexicalItem
    >>> evaluator = DSLEvaluator()
    >>> item = LexicalItem(lemma="walk", pos="VERB")
    >>> evaluator.evaluate(
    ...     "self.pos == 'VERB'",
    ...     {"self": item}
    ... )
    True
    >>> evaluator.evaluate(
    ...     "self.lemma in motion_verbs",
    ...     {"self": item, "motion_verbs": {"walk", "run", "jump"}}
    ... )
    True
    """

    def __init__(self) -> None:
        self.evaluator = Evaluator(use_cache=True)
        self.compiled_cache: dict[str, ast.ASTNode] = {}

    def evaluate(
        self,
        expression: str,
        context: Mapping[str, DslValue],
    ) -> DslValue:
        """Evaluate DSL expression with given context.

        Parameters
        ----------
        expression : str
            DSL expression to evaluate.
        context : Mapping[str, DslValue]
            Variables available during evaluation. Values may be DSL scalars,
            collections, or bead models (e.g. a ``LexicalItem`` bound to
            ``self`` for single-slot constraints, a ``FilledTemplate`` for
            multi-slot constraints, or an ``Item`` for list partitioning).

        Returns
        -------
        DslValue
            Result of evaluation.

        Raises
        ------
        EvaluationError
            If evaluation fails (parse error, undefined variable, etc.).

        Examples
        --------
        >>> evaluator = DSLEvaluator()
        >>> evaluator.evaluate("x > 5", {"x": 10})
        True
        >>> evaluator.evaluate(
        ...     "subject.lemma == verb.lemma",
        ...     {"subject": item1, "verb": item2}
        ... )
        False
        """
        # get or compile AST
        if expression in self.compiled_cache:
            ast_node = self.compiled_cache[expression]
        else:
            ast_node = parse(expression)
            self.compiled_cache[expression] = ast_node

        # build evaluation context
        eval_context = EvaluationContext()
        register_stdlib(eval_context)

        # add context variables
        for name, value in context.items():
            eval_context.set_variable(name, value)

        # evaluate
        return self.evaluator.evaluate(ast_node, eval_context)

    def extract_property_value(
        self,
        obj: DslValue,
        property_expression: str,
        context: dict[str, ContextValue] | None = None,
    ) -> DslValue:
        """Extract property value using DSL expression.

        This method is used by ListPartitioner to extract property values
        from items using DSL expressions. The property_expression is evaluated
        with the object available as 'item' in the context.

        Parameters
        ----------
        obj : DslValue
            Object to extract property from (typically a LexicalItem or Item).
        property_expression : str
            DSL expression that accesses object properties (e.g., "item.lemma",
            "item.features.number", "len(item.lemma)").
        context : dict[str, ContextValue] | None
            Additional context variables (e.g., constants, helper data).

        Returns
        -------
        DslValue
            Extracted property value.

        Raises
        ------
        EvaluationError
            If property extraction fails.

        Examples
        --------
        >>> evaluator = DSLEvaluator()
        >>> item = LexicalItem(lemma="walk", pos="VERB")
        >>> evaluator.extract_property_value(item, "item.lemma")
        'walk'
        >>> evaluator.extract_property_value(item, "len(item.lemma)")
        4
        """
        eval_context_dict: dict[str, DslValue] = {"item": obj}
        if context:
            for key, value in context.items():
                eval_context_dict[key] = value

        return self.evaluate(property_expression, eval_context_dict)

    def clear_cache(self) -> None:
        """Clear compiled AST cache.

        This should be called if you want to free memory or if expression
        strings might have changed meaning.

        Examples
        --------
        >>> evaluator = DSLEvaluator()
        >>> evaluator.clear_cache()
        """
        self.compiled_cache.clear()
        self.evaluator.clear_cache()
