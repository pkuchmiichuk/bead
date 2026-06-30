"""AST node definitions for the constraint DSL.

The hierarchy is a discriminated union rooted at ``ASTNode``; the
``kind`` field distinguishes variants and supports JSON round-trips.
"""

from __future__ import annotations

from typing import Literal as _Lit

import didactic.api as dx


class ASTNode(dx.TaggedUnion, discriminator="kind"):
    """Discriminated union of AST node variants."""


class Literal(ASTNode):
    """Literal value node (string, number, boolean).

    Examples
    --------
    >>> node = Literal(kind="literal", value="hello")
    >>> node.value
    'hello'
    """

    kind: _Lit["literal"]
    value: str | int | float | bool


class Variable(ASTNode):
    """Variable reference node.

    Examples
    --------
    >>> node = Variable(kind="variable", name="lemma")
    >>> node.name
    'lemma'
    """

    kind: _Lit["variable"]
    name: str


class BinaryOp(ASTNode):
    """Binary operation node (e.g. ``a == b``, ``x and y``).

    Attributes
    ----------
    operator : str
        Operator symbol or keyword.
    left : ASTNode
        Left operand.
    right : ASTNode
        Right operand.
    """

    kind: _Lit["binary_op"]
    operator: str
    left: ASTNode
    right: ASTNode


class UnaryOp(ASTNode):
    """Unary operation node (e.g. ``not x``, ``-y``)."""

    kind: _Lit["unary_op"]
    operator: str
    operand: ASTNode


class FunctionCall(ASTNode):
    """Function or method call node.

    Attributes
    ----------
    function : ASTNode
        The function being called (Variable for ``len(x)``,
        AttributeAccess for ``obj.method(...)``).
    arguments : tuple[ASTNode, ...]
        Argument expressions.
    """

    kind: _Lit["function_call"]
    function: ASTNode
    arguments: tuple[ASTNode, ...] = ()


class ListLiteral(ASTNode):
    """List literal node (e.g. ``["a", "b"]``)."""

    kind: _Lit["list_literal"]
    elements: tuple[ASTNode, ...] = ()


class AttributeAccess(ASTNode):
    """Attribute access node (e.g. ``item.lemma``)."""

    kind: _Lit["attribute_access"]
    object: ASTNode
    attribute: str


class Subscript(ASTNode):
    """Subscript access node (e.g. ``item['key']``, ``obj[0]``)."""

    kind: _Lit["subscript"]
    object: ASTNode
    index: ASTNode
