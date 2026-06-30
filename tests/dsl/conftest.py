"""Pytest fixtures for DSL module tests."""

from __future__ import annotations

import pytest

from bead.dsl import EvaluationContext, ast, register_stdlib


@pytest.fixture
def sample_literal_string() -> ast.Literal:
    """Provide sample string literal node."""
    return ast.Literal(kind="literal", value="hello")


@pytest.fixture
def sample_literal_int() -> ast.Literal:
    """Provide sample integer literal node."""
    return ast.Literal(kind="literal", value=42)


@pytest.fixture
def sample_variable() -> ast.Variable:
    """Provide sample variable node."""
    return ast.Variable(kind="variable", name="lemma")


@pytest.fixture
def sample_binary_op(
    sample_variable: ast.Variable, sample_literal_string: ast.Literal
) -> ast.BinaryOp:
    """Provide sample binary operation node."""
    return ast.BinaryOp(
        kind="binary_op",
        operator="==",
        left=sample_variable,
        right=sample_literal_string,
    )


@pytest.fixture
def empty_context() -> EvaluationContext:
    """Provide empty evaluation context."""
    return EvaluationContext()


@pytest.fixture
def context_with_stdlib() -> EvaluationContext:
    """Provide evaluation context with standard library."""
    ctx = EvaluationContext()
    register_stdlib(ctx)
    return ctx


@pytest.fixture
def context_with_variables() -> EvaluationContext:
    """Provide evaluation context with sample variables."""
    ctx = EvaluationContext()
    ctx.set_variable("pos", "VERB")
    ctx.set_variable("lemma", "walk")
    ctx.set_variable("transitive", True)
    ctx.set_variable("count", 5)
    return ctx
