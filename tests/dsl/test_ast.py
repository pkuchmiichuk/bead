"""Tests for AST node classes."""

from __future__ import annotations

import pytest
from didactic.api import ValidationError

from bead.dsl import ast


def test_literal_string() -> None:
    """Test Literal with string value."""
    node = ast.Literal(kind="literal", value="hello")
    assert node.value == "hello"
    assert isinstance(node, ast.ASTNode)


def test_literal_int() -> None:
    """Test Literal with integer value."""
    node = ast.Literal(kind="literal", value=42)
    assert node.value == 42
    assert isinstance(node.value, int)


def test_literal_float() -> None:
    """Test Literal with float value."""
    node = ast.Literal(kind="literal", value=3.14)
    assert node.value == 3.14
    assert isinstance(node.value, float)


def test_literal_bool_true() -> None:
    """Test Literal with boolean true value."""
    node = ast.Literal(kind="literal", value=True)
    assert node.value is True
    assert isinstance(node.value, bool)


def test_literal_bool_false() -> None:
    """Test Literal with boolean false value."""
    node = ast.Literal(kind="literal", value=False)
    assert node.value is False
    assert isinstance(node.value, bool)


def test_variable_creation() -> None:
    """Test Variable creation."""
    node = ast.Variable(kind="variable", name="lemma")
    assert node.name == "lemma"
    assert isinstance(node, ast.ASTNode)


def test_variable_with_underscore() -> None:
    """Test Variable with underscore in name."""
    node = ast.Variable(kind="variable", name="is_transitive")
    assert node.name == "is_transitive"


def test_binary_op_creation() -> None:
    """Test BinaryOp creation."""
    left = ast.Variable(kind="variable", name="pos")
    right = ast.Literal(kind="literal", value="VERB")
    node = ast.BinaryOp(kind="binary_op", operator="==", left=left, right=right)
    assert node.operator == "=="
    assert node.left == left
    assert node.right == right


def test_binary_op_nested() -> None:
    """Test BinaryOp with nested operands."""
    # (a == b) and (c == d)
    left_op = ast.BinaryOp(
        kind="binary_op",
        operator="==",
        left=ast.Variable(kind="variable", name="a"),
        right=ast.Variable(kind="variable", name="b"),
    )
    right_op = ast.BinaryOp(
        kind="binary_op",
        operator="==",
        left=ast.Variable(kind="variable", name="c"),
        right=ast.Variable(kind="variable", name="d"),
    )
    node = ast.BinaryOp(kind="binary_op", operator="and", left=left_op, right=right_op)
    assert node.operator == "and"
    assert isinstance(node.left, ast.BinaryOp)
    assert isinstance(node.right, ast.BinaryOp)


def test_unary_op_creation() -> None:
    """Test UnaryOp creation."""
    operand = ast.Variable(kind="variable", name="x")
    node = ast.UnaryOp(kind="unary_op", operator="not", operand=operand)
    assert node.operator == "not"
    assert node.operand == operand


def test_unary_op_minus() -> None:
    """Test UnaryOp with minus operator."""
    operand = ast.Literal(kind="literal", value=42)
    node = ast.UnaryOp(kind="unary_op", operator="-", operand=operand)
    assert node.operator == "-"
    assert node.operand.value == 42


def test_function_call_no_args() -> None:
    """Test FunctionCall with no arguments."""
    func = ast.Variable(kind="variable", name="now")
    node = ast.FunctionCall(kind="function_call", function=func, arguments=[])
    assert node.function.name == "now"
    assert len(node.arguments) == 0


def test_function_call_one_arg() -> None:
    """Test FunctionCall with one argument."""
    func = ast.Variable(kind="variable", name="len")
    arg = ast.Variable(kind="variable", name="lemma")
    node = ast.FunctionCall(kind="function_call", function=func, arguments=[arg])
    assert node.function.name == "len"
    assert len(node.arguments) == 1
    assert node.arguments[0] == arg


def test_function_call_multiple_args() -> None:
    """Test FunctionCall with multiple arguments."""
    func = ast.Variable(kind="variable", name="substring")
    args = [
        ast.Variable(kind="variable", name="text"),
        ast.Literal(kind="literal", value=0),
        ast.Literal(kind="literal", value=5),
    ]
    node = ast.FunctionCall(kind="function_call", function=func, arguments=args)
    assert node.function.name == "substring"
    assert len(node.arguments) == 3


def test_list_literal_empty() -> None:
    """Test ListLiteral with no elements."""
    node = ast.ListLiteral(kind="list_literal", elements=[])
    assert len(node.elements) == 0


def test_list_literal_with_elements() -> None:
    """Test ListLiteral with elements."""
    elements = [
        ast.Literal(kind="literal", value="a"),
        ast.Literal(kind="literal", value="b"),
        ast.Literal(kind="literal", value="c"),
    ]
    node = ast.ListLiteral(kind="list_literal", elements=elements)
    assert len(node.elements) == 3
    assert node.elements[0].value == "a"
    assert node.elements[1].value == "b"
    assert node.elements[2].value == "c"


def test_attribute_access_creation() -> None:
    """Test AttributeAccess creation."""
    obj = ast.Variable(kind="variable", name="item")
    node = ast.AttributeAccess(kind="attribute_access", object=obj, attribute="lemma")
    assert node.object == obj
    assert node.attribute == "lemma"


def test_attribute_access_nested() -> None:
    """Test AttributeAccess with nested object."""
    # obj.attr1.attr2
    inner = ast.AttributeAccess(
        kind="attribute_access",
        object=ast.Variable(kind="variable", name="obj"),
        attribute="attr1",
    )
    outer = ast.AttributeAccess(
        kind="attribute_access", object=inner, attribute="attr2"
    )
    assert outer.attribute == "attr2"
    assert isinstance(outer.object, ast.AttributeAccess)


def test_ast_node_serialization() -> None:
    """Test AST node serialization to dict."""
    node = ast.Literal(kind="literal", value=42)
    data = node.model_dump()
    assert isinstance(data, dict)
    assert data["value"] == 42


def test_ast_node_deserialization() -> None:
    """Test AST node deserialization from dict."""
    data = {"kind": "literal", "value": "hello"}
    node = ast.Literal(**data)
    assert node.value == "hello"


def test_ast_node_validation_error() -> None:
    """Test AST node validation with invalid types."""
    with pytest.raises(ValidationError):
        ast.Variable.model_validate({"kind": "variable"})


def test_nested_ast_structure() -> None:
    """Test complex nested AST structure."""
    # (pos == "VERB" and len(lemma) > 3) or transitive == true
    left_left = ast.BinaryOp(
        kind="binary_op",
        operator="==",
        left=ast.Variable(kind="variable", name="pos"),
        right=ast.Literal(kind="literal", value="VERB"),
    )
    left_right = ast.BinaryOp(
        kind="binary_op",
        operator=">",
        left=ast.FunctionCall(
            kind="function_call",
            function=ast.Variable(kind="variable", name="len"),
            arguments=[ast.Variable(kind="variable", name="lemma")],
        ),
        right=ast.Literal(kind="literal", value=3),
    )
    left = ast.BinaryOp(
        kind="binary_op", operator="and", left=left_left, right=left_right
    )

    right = ast.BinaryOp(
        kind="binary_op",
        operator="==",
        left=ast.Variable(kind="variable", name="transitive"),
        right=ast.Literal(kind="literal", value=True),
    )

    root = ast.BinaryOp(kind="binary_op", operator="or", left=left, right=right)

    assert root.operator == "or"
    assert isinstance(root.left, ast.BinaryOp)
    assert isinstance(root.right, ast.BinaryOp)


def test_ast_node_equality() -> None:
    """Test AST node equality."""
    node1 = ast.Literal(kind="literal", value=42)
    node2 = ast.Literal(kind="literal", value=42)
    # Pydantic models compare by value
    assert node1.value == node2.value


def test_ast_node_model_dump() -> None:
    """Test AST node model_dump method."""
    node = ast.BinaryOp(
        kind="binary_op",
        operator="==",
        left=ast.Variable(kind="variable", name="x"),
        right=ast.Literal(kind="literal", value=10),
    )
    data = node.model_dump()
    assert data["operator"] == "=="
    assert isinstance(data["left"], dict)
    assert isinstance(data["right"], dict)
    assert data["left"]["kind"] == "variable"
    assert data["right"]["kind"] == "literal"


def test_ast_node_with_returns_new_instance() -> None:
    """Test AST node ``with_`` returns a new instance with the requested change."""
    original = ast.Variable(kind="variable", name="test")
    updated = original.with_(name="renamed")
    assert original.name == "test"
    assert updated.name == "renamed"
    assert updated is not original


def test_binary_op_all_operators() -> None:
    """Test BinaryOp with various operators."""
    operators = [
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "and",
        "or",
        "in",
        "not in",
        "+",
        "-",
        "*",
        "/",
        "%",
    ]
    left = ast.Variable(kind="variable", name="x")
    right = ast.Literal(kind="literal", value=5)

    for op in operators:
        node = ast.BinaryOp(kind="binary_op", operator=op, left=left, right=right)
        assert node.operator == op


def test_unary_op_all_operators() -> None:
    """Test UnaryOp with various operators."""
    operators = ["not", "-", "+"]
    operand = ast.Variable(kind="variable", name="x")

    for op in operators:
        node = ast.UnaryOp(kind="unary_op", operator=op, operand=operand)
        assert node.operator == op
