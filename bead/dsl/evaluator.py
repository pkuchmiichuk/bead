"""Constraint evaluator for DSL.

This module provides the Evaluator class that executes AST nodes
against an evaluation context to produce boolean results, and the
DSLEvaluator class that provides a high-level interface for evaluating
constraint expressions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from bead.dsl import ast
from bead.dsl.context import EvaluationContext
from bead.dsl.errors import EvaluationError
from bead.dsl.parser import parse
from bead.dsl.stdlib import register_stdlib

if TYPE_CHECKING:
    from bead.items.item import Item
    from bead.resources.constraints import ContextValue
    from bead.resources.lexical_item import LexicalItem
    from bead.templates.filler import FilledTemplate


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
        self._cache: dict[tuple[str, ...], Any] = {}

    def evaluate(self, node: ast.ASTNode, context: EvaluationContext) -> Any:
        """Evaluate an AST node in the given context.

        Parameters
        ----------
        node : ast.ASTNode
            AST node to evaluate.
        context : EvaluationContext
            Evaluation context with variables and functions.

        Returns
        -------
        Any
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

    def _evaluate_literal(self, node: ast.Literal, context: EvaluationContext) -> Any:
        """Evaluate literal node.

        Parameters
        ----------
        node : ast.Literal
            Literal node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
            Literal value.
        """
        return node.value

    def _evaluate_variable(self, node: ast.Variable, context: EvaluationContext) -> Any:
        """Evaluate variable node.

        Parameters
        ----------
        node : ast.Variable
            Variable node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
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
    ) -> Any:
        """Evaluate binary operation node.

        Parameters
        ----------
        node : ast.BinaryOp
            Binary operation node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
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

        try:
            # comparison operators
            if node.operator == "==":
                return left == right
            elif node.operator == "!=":
                return left != right
            elif node.operator == "<":
                return left < right
            elif node.operator == ">":
                return left > right
            elif node.operator == "<=":
                return left <= right
            elif node.operator == ">=":
                return left >= right
            # membership operators
            elif node.operator == "in":
                return left in right
            elif node.operator == "not in":
                return left not in right
            # arithmetic operators
            elif node.operator == "+":
                return left + right
            elif node.operator == "-":
                return left - right
            elif node.operator == "*":
                return left * right
            elif node.operator == "/":
                if right == 0:
                    raise EvaluationError("Division by zero")
                return left / right
            elif node.operator == "%":
                if right == 0:
                    raise EvaluationError("Modulo by zero")
                return left % right
            else:
                raise EvaluationError(f"Unknown operator: {node.operator}")
        except TypeError as e:
            raise EvaluationError(
                f"Type error in operation '{node.operator}': "
                f"cannot operate on {type(left).__name__} and {type(right).__name__}"
            ) from e
        except ZeroDivisionError as e:
            raise EvaluationError("Division by zero") from e

    def _evaluate_unary_op(self, node: ast.UnaryOp, context: EvaluationContext) -> Any:
        """Evaluate unary operation node.

        Parameters
        ----------
        node : ast.UnaryOp
            Unary operation node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
            Result of unary operation.

        Raises
        ------
        EvaluationError
            If operator is unknown or operation fails.
        """
        operand = self.evaluate(node.operand, context)

        try:
            if node.operator == "not":
                return not operand
            elif node.operator == "-":
                return -operand
            elif node.operator == "+":
                return +operand
            else:
                raise EvaluationError(f"Unknown unary operator: {node.operator}")
        except TypeError as e:
            raise EvaluationError(
                f"Type error in unary operation '{node.operator}': "
                f"cannot operate on {type(operand).__name__}"
            ) from e

    def _evaluate_function_call(
        self, node: ast.FunctionCall, context: EvaluationContext
    ) -> Any:
        """Evaluate function call node.

        Parameters
        ----------
        node : ast.FunctionCall
            Function call node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
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
    ) -> Any:
        """Evaluate attribute access node.

        Parameters
        ----------
        node : ast.AttributeAccess
            Attribute access node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
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
            return obj[node.attribute]  # type: ignore[reportUnknownVariableType]

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
    ) -> Any:
        """Evaluate subscript access node.

        Parameters
        ----------
        node : ast.Subscript
            Subscript access node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
            Subscripted value.

        Raises
        ------
        EvaluationError
            If subscript access fails.
        """
        obj = self.evaluate(node.object, context)
        index = self.evaluate(node.index, context)

        try:
            return obj[index]  # type: ignore[reportUnknownVariableType]
        except (KeyError, IndexError, TypeError) as e:
            obj_type = type(obj).__name__
            raise EvaluationError(
                f"Subscript access failed on {obj_type} with index {index}: {e}"
            ) from e

    def _evaluate_list_literal(
        self, node: ast.ListLiteral, context: EvaluationContext
    ) -> list[Any]:
        """Evaluate list literal node.

        Parameters
        ----------
        node : ast.ListLiteral
            List literal node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        list[Any]
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
        context: Mapping[str, ContextValue | LexicalItem | FilledTemplate | Item],
    ) -> bool | str | int | float | list[Any]:
        """Evaluate DSL expression with given context.

        Parameters
        ----------
        expression : str
            DSL expression to evaluate.
        context : dict[str, ContextValue | LexicalItem | FilledTemplate | Item]
            Variables available during evaluation. Can include:
            - ContextValue: primitive values, lists, sets
            - LexicalItem: lexical items for single-slot constraints
            - FilledTemplate: filled templates for multi-slot constraints
            - Item: items for list partitioning

        Returns
        -------
        bool | str | int | float | list[Any]
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
        obj: Any,
        property_expression: str,
        context: dict[str, ContextValue] | None = None,
    ) -> Any:
        """Extract property value using DSL expression.

        This method is used by ListPartitioner to extract property values
        from items using DSL expressions. The property_expression is evaluated
        with the object available as 'item' in the context.

        Parameters
        ----------
        obj : Any
            Object to extract property from (typically a LexicalItem or Item).
        property_expression : str
            DSL expression that accesses object properties (e.g., "item.lemma",
            "item.features.number", "len(item.lemma)").
        context : dict[str, ContextValue] | None
            Additional context variables (e.g., constants, helper data).

        Returns
        -------
        Any
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
        eval_context_dict: dict[str, Any] = {"item": obj}
        if context:
            eval_context_dict.update(context)

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
