"""Tests for constraint evaluator."""

from __future__ import annotations

import pytest

from bead.dsl import (
    EvaluationContext,
    Evaluator,
    ast,
    evaluate,
    parse,
    register_stdlib,
)
from bead.dsl.errors import EvaluationError


# Test literals
def test_evaluate_string_literal() -> None:
    """Test evaluating string literal."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.Literal(kind="literal", value="hello")
    result = evaluator.evaluate(node, ctx)
    assert result == "hello"


def test_evaluate_int_literal() -> None:
    """Test evaluating integer literal."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.Literal(kind="literal", value=42)
    result = evaluator.evaluate(node, ctx)
    assert result == 42


def test_evaluate_float_literal() -> None:
    """Test evaluating float literal."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.Literal(kind="literal", value=3.14)
    result = evaluator.evaluate(node, ctx)
    assert result == 3.14


def test_evaluate_boolean_literal_true() -> None:
    """Test evaluating boolean literal True."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.Literal(kind="literal", value=True)
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_boolean_literal_false() -> None:
    """Test evaluating boolean literal False."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.Literal(kind="literal", value=False)
    result = evaluator.evaluate(node, ctx)
    assert result is False


# Test variables
def test_evaluate_variable_defined() -> None:
    """Test evaluating defined variable."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    node = ast.Variable(kind="variable", name="x")
    result = evaluator.evaluate(node, ctx)
    assert result == 42


def test_evaluate_variable_undefined_raises_error() -> None:
    """Test evaluating undefined variable raises error."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.Variable(kind="variable", name="x")
    with pytest.raises(EvaluationError, match="Undefined variable: x"):
        evaluator.evaluate(node, ctx)


# Test comparison operators
def test_evaluate_equality_true() -> None:
    """Test evaluating == operator (true case)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    node = ast.BinaryOp(
        kind="binary_op",
        operator="==",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Literal(kind="literal", value=42),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_equality_false() -> None:
    """Test evaluating == operator (false case)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    node = ast.BinaryOp(
        kind="binary_op",
        operator="==",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Literal(kind="literal", value=50),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is False


def test_evaluate_not_equal() -> None:
    """Test evaluating != operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    node = ast.BinaryOp(
        kind="binary_op",
        operator="!=",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Literal(kind="literal", value=50),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_less_than() -> None:
    """Test evaluating < operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    node = ast.BinaryOp(
        kind="binary_op",
        operator="<",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Literal(kind="literal", value=50),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_greater_than() -> None:
    """Test evaluating > operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    node = ast.BinaryOp(
        kind="binary_op",
        operator=">",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Literal(kind="literal", value=30),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_less_than_or_equal() -> None:
    """Test evaluating <= operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    node = ast.BinaryOp(
        kind="binary_op",
        operator="<=",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Literal(kind="literal", value=42),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_greater_than_or_equal() -> None:
    """Test evaluating >= operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    node = ast.BinaryOp(
        kind="binary_op",
        operator=">=",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Literal(kind="literal", value=42),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


# Test logical operators
def test_evaluate_and_both_true() -> None:
    """Test evaluating and operator (both true)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="and",
        left=ast.Literal(kind="literal", value=True),
        right=ast.Literal(kind="literal", value=True),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_and_one_false() -> None:
    """Test evaluating and operator (one false)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="and",
        left=ast.Literal(kind="literal", value=True),
        right=ast.Literal(kind="literal", value=False),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is False


def test_evaluate_and_short_circuit() -> None:
    """Test and operator short-circuits on false."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", False)
    # y is not defined, but this should not raise an error due to short-circuit
    node = ast.BinaryOp(
        kind="binary_op",
        operator="and",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Variable(kind="variable", name="y"),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is False


def test_evaluate_or_both_false() -> None:
    """Test evaluating or operator (both false)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="or",
        left=ast.Literal(kind="literal", value=False),
        right=ast.Literal(kind="literal", value=False),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is False


def test_evaluate_or_one_true() -> None:
    """Test evaluating or operator (one true)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="or",
        left=ast.Literal(kind="literal", value=True),
        right=ast.Literal(kind="literal", value=False),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_or_short_circuit() -> None:
    """Test or operator short-circuits on true."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", True)
    # y is not defined, but this should not raise an error due to short-circuit
    node = ast.BinaryOp(
        kind="binary_op",
        operator="or",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Variable(kind="variable", name="y"),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_not_true() -> None:
    """Test evaluating not operator (true)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.UnaryOp(
        kind="unary_op", operator="not", operand=ast.Literal(kind="literal", value=True)
    )
    result = evaluator.evaluate(node, ctx)
    assert result is False


def test_evaluate_not_false() -> None:
    """Test evaluating not operator (false)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.UnaryOp(
        kind="unary_op",
        operator="not",
        operand=ast.Literal(kind="literal", value=False),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


# Test membership operators
def test_evaluate_in_present() -> None:
    """Test evaluating in operator (element present)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="in",
        left=ast.Literal(kind="literal", value=2),
        right=ast.ListLiteral(
            kind="list_literal",
            elements=[
                ast.Literal(kind="literal", value=1),
                ast.Literal(kind="literal", value=2),
                ast.Literal(kind="literal", value=3),
            ],
        ),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_in_absent() -> None:
    """Test evaluating in operator (element absent)."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="in",
        left=ast.Literal(kind="literal", value=5),
        right=ast.ListLiteral(
            kind="list_literal",
            elements=[
                ast.Literal(kind="literal", value=1),
                ast.Literal(kind="literal", value=2),
                ast.Literal(kind="literal", value=3),
            ],
        ),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is False


def test_evaluate_not_in() -> None:
    """Test evaluating not in operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="not in",
        left=ast.Literal(kind="literal", value=5),
        right=ast.ListLiteral(
            kind="list_literal",
            elements=[
                ast.Literal(kind="literal", value=1),
                ast.Literal(kind="literal", value=2),
                ast.Literal(kind="literal", value=3),
            ],
        ),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


# Test arithmetic operators
def test_evaluate_addition() -> None:
    """Test evaluating addition operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="+",
        left=ast.Literal(kind="literal", value=5),
        right=ast.Literal(kind="literal", value=3),
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 8


def test_evaluate_subtraction() -> None:
    """Test evaluating subtraction operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="-",
        left=ast.Literal(kind="literal", value=5),
        right=ast.Literal(kind="literal", value=3),
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 2


def test_evaluate_multiplication() -> None:
    """Test evaluating multiplication operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="*",
        left=ast.Literal(kind="literal", value=5),
        right=ast.Literal(kind="literal", value=3),
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 15


def test_evaluate_division() -> None:
    """Test evaluating division operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="/",
        left=ast.Literal(kind="literal", value=6),
        right=ast.Literal(kind="literal", value=3),
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 2.0


def test_evaluate_modulo() -> None:
    """Test evaluating modulo operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="%",
        left=ast.Literal(kind="literal", value=7),
        right=ast.Literal(kind="literal", value=3),
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 1


def test_evaluate_unary_minus() -> None:
    """Test evaluating unary minus operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.UnaryOp(
        kind="unary_op", operator="-", operand=ast.Literal(kind="literal", value=5)
    )
    result = evaluator.evaluate(node, ctx)
    assert result == -5


def test_evaluate_unary_plus() -> None:
    """Test evaluating unary plus operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.UnaryOp(
        kind="unary_op", operator="+", operand=ast.Literal(kind="literal", value=5)
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 5


# Test function calls
def test_evaluate_function_call_no_args() -> None:
    """Test evaluating function call with no arguments."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_function("get_value", lambda: 42)
    node = ast.FunctionCall(
        kind="function_call",
        function=ast.Variable(kind="variable", name="get_value"),
        arguments=[],
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 42


def test_evaluate_function_call_one_arg() -> None:
    """Test evaluating function call with one argument."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_function("double", lambda x: x * 2)
    node = ast.FunctionCall(
        kind="function_call",
        function=ast.Variable(kind="variable", name="double"),
        arguments=[ast.Literal(kind="literal", value=5)],
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 10


def test_evaluate_function_call_multiple_args() -> None:
    """Test evaluating function call with multiple arguments."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_function("add", lambda x, y: x + y)
    node = ast.FunctionCall(
        kind="function_call",
        function=ast.Variable(kind="variable", name="add"),
        arguments=[
            ast.Literal(kind="literal", value=3),
            ast.Literal(kind="literal", value=4),
        ],
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 7


def test_evaluate_undefined_function_raises_error() -> None:
    """Test evaluating undefined function raises error."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.FunctionCall(
        kind="function_call",
        function=ast.Variable(kind="variable", name="foo"),
        arguments=[],
    )
    with pytest.raises(EvaluationError, match="Undefined function: foo"):
        evaluator.evaluate(node, ctx)


def test_evaluate_nested_function_calls() -> None:
    """Test evaluating nested function calls."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_function("double", lambda x: x * 2)
    ctx.set_function("add", lambda x, y: x + y)
    # add(double(3), 4)
    node = ast.FunctionCall(
        kind="function_call",
        function=ast.Variable(kind="variable", name="add"),
        arguments=[
            ast.FunctionCall(
                kind="function_call",
                function=ast.Variable(kind="variable", name="double"),
                arguments=[ast.Literal(kind="literal", value=3)],
            ),
            ast.Literal(kind="literal", value=4),
        ],
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 10


# Test attribute access
def test_evaluate_attribute_on_object() -> None:
    """Test evaluating attribute access on object."""
    evaluator = Evaluator()
    ctx = EvaluationContext()

    class Item:
        def __init__(self) -> None:
            self.lemma = "walk"

    ctx.set_variable("item", Item())
    node = ast.AttributeAccess(
        kind="attribute_access",
        object=ast.Variable(kind="variable", name="item"),
        attribute="lemma",
    )
    result = evaluator.evaluate(node, ctx)
    assert result == "walk"


def test_evaluate_attribute_on_dict() -> None:
    """Test evaluating attribute access on dict."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("item", {"lemma": "walk", "pos": "VERB"})
    node = ast.AttributeAccess(
        kind="attribute_access",
        object=ast.Variable(kind="variable", name="item"),
        attribute="lemma",
    )
    result = evaluator.evaluate(node, ctx)
    assert result == "walk"


def test_evaluate_undefined_attribute_raises_error() -> None:
    """Test evaluating undefined attribute raises error."""
    evaluator = Evaluator()
    ctx = EvaluationContext()

    class Item:
        def __init__(self) -> None:
            self.lemma = "walk"

    ctx.set_variable("item", Item())
    node = ast.AttributeAccess(
        kind="attribute_access",
        object=ast.Variable(kind="variable", name="item"),
        attribute="foo",
    )
    with pytest.raises(EvaluationError, match="has no attribute: foo"):
        evaluator.evaluate(node, ctx)


def test_evaluate_nested_attribute_access() -> None:
    """Test evaluating nested attribute access."""
    evaluator = Evaluator()
    ctx = EvaluationContext()

    class Inner:
        def __init__(self) -> None:
            self.value = 42

    class Outer:
        def __init__(self) -> None:
            self.inner = Inner()

    ctx.set_variable("obj", Outer())
    node = ast.AttributeAccess(
        kind="attribute_access",
        object=ast.AttributeAccess(
            kind="attribute_access",
            object=ast.Variable(kind="variable", name="obj"),
            attribute="inner",
        ),
        attribute="value",
    )
    result = evaluator.evaluate(node, ctx)
    assert result == 42


# Test list literals
def test_evaluate_empty_list() -> None:
    """Test evaluating empty list."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.ListLiteral(kind="list_literal", elements=[])
    result = evaluator.evaluate(node, ctx)
    assert result == []


def test_evaluate_list_with_elements() -> None:
    """Test evaluating list with elements."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.ListLiteral(
        kind="list_literal",
        elements=[
            ast.Literal(kind="literal", value=1),
            ast.Literal(kind="literal", value=2),
            ast.Literal(kind="literal", value=3),
        ],
    )
    result = evaluator.evaluate(node, ctx)
    assert result == [1, 2, 3]


# Test complex expressions
def test_evaluate_nested_boolean_expression() -> None:
    """Test evaluating nested boolean expression."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 10)
    ctx.set_variable("y", 20)
    # (x > 5) and (y < 30)
    node = ast.BinaryOp(
        kind="binary_op",
        operator="and",
        left=ast.BinaryOp(
            kind="binary_op",
            operator=">",
            left=ast.Variable(kind="variable", name="x"),
            right=ast.Literal(kind="literal", value=5),
        ),
        right=ast.BinaryOp(
            kind="binary_op",
            operator="<",
            left=ast.Variable(kind="variable", name="y"),
            right=ast.Literal(kind="literal", value=30),
        ),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


def test_evaluate_complex_expression_with_function() -> None:
    """Test evaluating expression with function and operators."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    register_stdlib(ctx)
    ctx.set_variable("lemma", "walking")
    # len(lemma) > 5
    node = ast.BinaryOp(
        kind="binary_op",
        operator=">",
        left=ast.FunctionCall(
            kind="function_call",
            function=ast.Variable(kind="variable", name="len"),
            arguments=[ast.Variable(kind="variable", name="lemma")],
        ),
        right=ast.Literal(kind="literal", value=5),
    )
    result = evaluator.evaluate(node, ctx)
    assert result is True


# Test error handling
def test_evaluate_type_error_in_operator() -> None:
    """Test type error in operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="<",
        left=ast.Literal(kind="literal", value="hello"),
        right=ast.Literal(kind="literal", value=5),
    )
    with pytest.raises(EvaluationError, match="Cannot compare"):
        evaluator.evaluate(node, ctx)


def test_evaluate_division_by_zero() -> None:
    """Test division by zero."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="/",
        left=ast.Literal(kind="literal", value=5),
        right=ast.Literal(kind="literal", value=0),
    )
    with pytest.raises(EvaluationError, match="Division by zero"):
        evaluator.evaluate(node, ctx)


def test_evaluate_modulo_by_zero() -> None:
    """Test modulo by zero."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="%",
        left=ast.Literal(kind="literal", value=5),
        right=ast.Literal(kind="literal", value=0),
    )
    with pytest.raises(EvaluationError, match="Modulo by zero"):
        evaluator.evaluate(node, ctx)


def test_evaluate_unknown_operator() -> None:
    """Test unknown operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="***",
        left=ast.Literal(kind="literal", value=5),
        right=ast.Literal(kind="literal", value=3),
    )
    with pytest.raises(EvaluationError, match="Unknown operator"):
        evaluator.evaluate(node, ctx)


def test_evaluate_unknown_unary_operator() -> None:
    """Test unknown unary operator."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.UnaryOp(
        kind="unary_op", operator="~", operand=ast.Literal(kind="literal", value=5)
    )
    with pytest.raises(EvaluationError, match="Unknown unary operator"):
        evaluator.evaluate(node, ctx)


# Test caching
def test_evaluator_cache_enabled() -> None:
    """Test evaluator with cache enabled."""
    evaluator = Evaluator(use_cache=True)
    assert evaluator._use_cache is True


def test_evaluator_cache_disabled() -> None:
    """Test evaluator with cache disabled."""
    evaluator = Evaluator(use_cache=False)
    assert evaluator._use_cache is False


def test_clear_cache() -> None:
    """Test clearing cache."""
    evaluator = Evaluator()
    evaluator.clear_cache()
    assert len(evaluator._cache) == 0


# Test integration with standard library
def test_evaluate_with_stdlib_functions() -> None:
    """Test evaluation with standard library functions."""
    ctx = EvaluationContext()
    register_stdlib(ctx)
    ctx.set_variable("text", "hello")
    node = parse("len(text) == 5")
    result = evaluate(node, ctx)
    assert result is True


def test_evaluate_complete_constraint_expression() -> None:
    """Test evaluating complete constraint expression."""
    ctx = EvaluationContext()
    register_stdlib(ctx)
    ctx.set_variable("pos", "VERB")
    ctx.set_variable("lemma", "walk")
    node = parse("pos == 'VERB' and len(lemma) > 3")
    result = evaluate(node, ctx)
    assert result is True


# Test convenience function
def test_evaluate_convenience_function() -> None:
    """Test evaluate convenience function."""
    ctx = EvaluationContext()
    ctx.set_variable("x", 10)
    node = parse("x > 5")
    result = evaluate(node, ctx)
    assert result is True


def test_evaluate_convenience_function_with_cache() -> None:
    """Test evaluate convenience function with cache parameter."""
    ctx = EvaluationContext()
    ctx.set_variable("x", 10)
    node = parse("x > 5")
    result = evaluate(node, ctx, use_cache=False)
    assert result is True


# Test string concatenation with +
def test_evaluate_string_concatenation() -> None:
    """Test evaluating string concatenation."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    node = ast.BinaryOp(
        kind="binary_op",
        operator="+",
        left=ast.Literal(kind="literal", value="hello"),
        right=ast.Literal(kind="literal", value=" world"),
    )
    result = evaluator.evaluate(node, ctx)
    assert result == "hello world"


# Test unary operators on variables
def test_evaluate_unary_minus_on_variable() -> None:
    """Test evaluating unary minus on variable."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("x", 5)
    node = ast.UnaryOp(
        kind="unary_op", operator="-", operand=ast.Variable(kind="variable", name="x")
    )
    result = evaluator.evaluate(node, ctx)
    assert result == -5


def test_evaluate_dict_key_not_found() -> None:
    """Test evaluating attribute access on dict with missing key."""
    evaluator = Evaluator()
    ctx = EvaluationContext()
    ctx.set_variable("item", {"lemma": "walk"})
    node = ast.AttributeAccess(
        kind="attribute_access",
        object=ast.Variable(kind="variable", name="item"),
        attribute="pos",
    )
    with pytest.raises(EvaluationError, match="Dictionary does not have key: pos"):
        evaluator.evaluate(node, ctx)
