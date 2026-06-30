# Contributing

This guide explains how to contribute to bead, including fork and branch workflow, commit conventions, pull request process, documentation requirements, code style, and common contribution patterns.

## Getting Started

Welcome to bead! We appreciate your interest in contributing. Before you start, review these resources:

1. **[Architecture](architecture.md)**: Understand system design and architectural decisions
2. **[Setup](setup.md)**: Configure your development environment
3. **[Testing](testing.md)**: Learn testing practices and coverage requirements

For questions or discussions, open an issue or start a discussion on the GitHub repository.

### Ways to Contribute

1. **Report Bugs**: Open an issue with steps to reproduce
2. **Request Features**: Propose new features with use cases
3. **Submit Pull Requests**: Fix bugs or implement features
4. **Improve Documentation**: Fix typos, add examples, clarify explanations
5. **Add Gallery Examples**: Contribute research examples for new languages

All contributions must follow the guidelines in this document.

## Fork and Branch Workflow

bead uses a fork and branch workflow for contributions.

### 1. Fork the Repository

Click "Fork" on the GitHub repository page to create your copy:

```
https://github.com/FACTSlab/bead → fork → https://github.com/your-username/bead
```

### 2. Clone Your Fork

Clone your fork locally:

```bash
git clone https://github.com/your-username/bead.git
cd bead
```

### 3. Add Upstream Remote

Add the original repository as upstream:

```bash
git remote add upstream https://github.com/FACTSlab/bead.git
```

Verify remotes:

```bash
git remote -v
# origin    https://github.com/your-username/bead.git (fetch)
# origin    https://github.com/your-username/bead.git (push)
# upstream  https://github.com/FACTSlab/bead.git (fetch)
# upstream  https://github.com/FACTSlab/bead.git (push)
```

### 4. Create Feature Branch

Create a branch for your work:

```bash
# Sync with upstream first
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/add-ranking-task-type
```

**Branch naming conventions**:
- `feature/description`: New features
- `fix/description`: Bug fixes
- `docs/description`: Documentation changes
- `refactor/description`: Code refactoring
- `test/description`: Test additions or fixes

Use descriptive names: `feature/add-ranking-task-type` not `feature/new-stuff`.

### 5. Make Changes

Make your changes in the feature branch. Commit frequently with descriptive messages (see Commit Message Conventions below).

### 6. Keep Branch Updated

Periodically sync with upstream main:

```bash
git checkout main
git pull upstream main
git checkout feature/add-ranking-task-type
git rebase main
```

Resolve any conflicts during rebase.

### 7. Push to Fork

Push your branch to your fork:

```bash
git push origin feature/add-ranking-task-type
```

### 8. Open Pull Request

Go to the original repository on GitHub. Click "New Pull Request" and select your branch.

## Commit Message Conventions

Write clear, descriptive commit messages that explain what changed and why.

### Format

```
<verb> <object> <context>

<optional detailed explanation>
```

**Examples**:

```
Add ranking task type with utilities

Implements create_ranking_item() and batch creation functions following
the established task-type utilities pattern. Includes validation for
ranking-specific metadata (items_to_rank, allow_ties).
```

```
Fix constraint satisfaction bug in partitioner

ListPartitioner.partition_with_batch_constraints() was not correctly
evaluating BatchDiversityConstraint when max_lists_per_value was 1.
Fixed by adjusting constraint evaluation logic.
```

```
Update GLMM documentation with variance component examples

Added examples showing how to interpret variance components for
participant and item random effects. Clarifies when to use each
mixed-effects mode.
```

### Guidelines

**Use imperative mood**: "Add feature" not "Added feature" or "Adds feature"

**Be specific**: "Fix constraint evaluation" not "Fix bug"

**Reference issues**: Include "Closes #123" or "Fixes #456" to auto-close issues

**Keep first line under 72 characters**: Summaries should fit on one line

**Add detailed explanation when helpful**: Explain why, not just what

**Make atomic commits**: Each commit should represent one logical change

### Examples of Good vs Bad Commits

**GOOD**:
```
Add create_ranking_item() function

Implements ranking task type following forced_choice.py pattern.
Includes n_items_to_rank metadata and validation. Tests included
in tests/items/test_ranking.py with 95% coverage.
```

**BAD**:
```
stuff

changed some things
```

**GOOD**:
```
Fix batch coverage constraint for empty target lists

BatchCoverageConstraint.evaluate() raised IndexError when
target_values was empty. Fixed by checking length before iteration.
Added test case in test_constraints.py.

Closes #234
```

**BAD**:
```
fixed bug
```

## Documentation Requirements

All public API requires NumPy-format docstrings with executable examples.

### Docstring Requirements

**All public functions and classes** must have docstrings with:

1. **Short Summary**: One-line description
2. **Parameters**: Type and description for each parameter
3. **Returns**: Type and description of return value
4. **Examples**: Executable examples in doctest format
5. **Raises** (if applicable): Exceptions that may be raised

### NumPy Format

Use NumPy docstring format (not Google or Sphinx):

```python
def create_ranking_item(
    *items_to_rank: str,
    prompt: str = "Rank these items from best to worst:",
    allow_ties: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Item:
    """Create a ranking task item.

    Parameters
    ----------
    *items_to_rank : str
        Items to be ranked by the participant.
    prompt : str, default="Rank these items from best to worst:"
        Instruction text shown to participant.
    allow_ties : bool, default=False
        Whether to allow tied rankings.
    metadata : dict[str, Any] | None, optional
        Additional metadata to attach to the item.

    Returns
    -------
    Item
        Ranking task item with task_type="ranking".

    Raises
    ------
    ValueError
        If fewer than 2 items provided for ranking.

    Examples
    --------
    >>> item = create_ranking_item("Option A", "Option B", "Option C")
    >>> item.item_metadata["n_items_to_rank"]
    3

    >>> item = create_ranking_item("X", "Y", allow_ties=True)
    >>> item.item_metadata["allow_ties"]
    True
    """
    if len(items_to_rank) < 2:
        raise ValueError("Must provide at least 2 items for ranking")

    # Implementation...
```

### Style Guidelines

Write clear, technical documentation:

**Language**:
- Use simple, direct language
- Use active voice
- Use specific technical terms (avoid vague adjectives like "comprehensive", "robust", "powerful", "seamless")
- Avoid marketing language ("showcase", "leverage", "foster", "delve")

**Formatting**:
- Use colons for sections: `Parameters: name, description`
- Avoid dash separators: `Parameters - name - Description`
- Use natural prose paragraphs instead of excessive bullet lists
- Keep docstrings concise and focused on essential information

### Validation Tools

**Run pydocstyle** (NumPy convention):
```bash
uv run pydocstyle bead/module.py
```

**Run darglint** (signature consistency):
```bash
uv run darglint bead/module.py
```

Pre-commit hooks run these automatically, but you should run them manually during development.

### Examples Section

All public functions must include an Examples section with executable doctest examples:

```python
Examples
--------
>>> create_ranking_item("A", "B", "C")
Item(...)

>>> create_ranking_item("X")  # Should raise
Traceback (most recent call last):
ValueError: Must provide at least 2 items for ranking
```

Run doctests:
```bash
uv run pytest --doctest-modules bead/items/ranking.py
```

## Code Style

bead enforces strict code style using ruff and pyright.

### Ruff (Linting and Formatting)

**Check for issues**:
```bash
uv run ruff check bead/
```

**Auto-fix issues**:
```bash
uv run ruff check bead/ --fix
```

**Format code**:
```bash
uv run ruff format bead/
```

**Configuration** (from pyproject.toml):
- Line length: 88 characters
- Target: Python 3.14
- Conventions: PEP 8, NumPy docstrings
- Rules: E (errors), F (PyFlakes), I (imports), N (naming), D (docstrings), UP (upgrades), ANN (annotations), B (bugbear), A (builtins), C4 (comprehensions), PLC (Pylint)

### Pyright (Type Checking)

**Check types**:
```bash
uv run pyright bead/
```

**Type hint requirements**:
- All function parameters must have type hints
- All return types must be annotated
- No `Any` or `object` in core code (only in adapters)
- Use `| None` for optional types (Python 3.10+ syntax)
- Use `list[T]`, `dict[K, V]` (Python 3.9+ syntax)

**Example**:
```python
def partition_items(
    items: list[UUID],
    n_lists: int,
    metadata: dict[UUID, dict[str, Any]],
    random_seed: int | None = None,
) -> list[ExperimentList]:
    """Partition items into lists."""
    ...
```

**Configuration** (from pyproject.toml):
- Mode: strict
- Python version: 3.14
- Excluded: tests/, adapters/ (external APIs have dynamic types)

### Running All Checks

Run all quality checks before committing:

```bash
uv run ruff check bead/ && uv run ruff format bead/ && uv run pyright bead/
```

Or create a shell alias:

```bash
alias bead-lint="uv run ruff check bead/ && uv run ruff format bead/ && uv run pyright bead/"
```

All checks must pass with zero errors and zero warnings.

## Testing Requirements

All new code requires tests with >90% coverage.

### Test Organization

Tests mirror source code structure:

```
bead/items/ranking.py  →  tests/items/test_ranking.py
```

### Test File Template

```python
"""Tests for ranking task type utilities."""

from __future__ import annotations

import pytest

from bead.items.ranking import create_ranking_item


class TestCreateRankingItem:
    """Test create_ranking_item function."""

    def test_creates_item_with_all_fields(self) -> None:
        """Test creating ranking item with all fields."""
        item = create_ranking_item(
            "A",
            "B",
            "C",
            prompt="Rank these:",
            allow_ties=True,
            metadata={"source": "test"},
        )

        assert item.task_type == "ranking"
        assert item.item_metadata["n_items_to_rank"] == 3
        assert item.item_metadata["allow_ties"] is True
        assert item.metadata["source"] == "test"

    def test_raises_for_single_item(self) -> None:
        """Test ValueError raised for single item."""
        with pytest.raises(ValueError) as exc_info:
            create_ranking_item("A")

        assert "at least 2 items" in str(exc_info.value)

    def test_default_prompt(self) -> None:
        """Test default prompt is set."""
        item = create_ranking_item("A", "B")

        assert "Rank these items" in item.rendered_elements["prompt"]
```

### Coverage Requirements

Run tests with coverage:

```bash
uv run pytest tests/items/test_ranking.py --cov=bead.items.ranking --cov-report=term-missing
```

Target >90% coverage. If coverage is below 90%:

1. Add tests for uncovered lines
2. Remove dead code (if any)
3. Add edge case tests

### Fixtures

If multiple tests need the same data, add fixtures to conftest.py:

```python
# tests/items/conftest.py
@pytest.fixture
def sample_ranking_items() -> list[str]:
    """Provide sample items for ranking tests."""
    return ["Option A", "Option B", "Option C", "Option D"]
```

### Running Tests

**Run your new tests**:
```bash
uv run pytest tests/items/test_ranking.py -v
```

**Run all tests**:
```bash
uv run pytest tests/
```

**Run with coverage**:
```bash
uv run pytest tests/ --cov=bead --cov-report=term-missing
```

All tests must pass before opening a pull request.

## Pull Request Process

### 1. Push Branch

Push your feature branch to your fork:

```bash
git push origin feature/add-ranking-task-type
```

### 2. Open Pull Request

On GitHub, click "New Pull Request":

1. Base repository: `FACTSlab/bead`
2. Base branch: `main`
3. Head repository: `your-username/bead`
4. Compare branch: `feature/add-ranking-task-type`

### 3. PR Title Format

Use the same format as commit messages:

```
Add ranking task type with utilities
```

### 4. PR Description

Provide context and details:

```markdown
## Summary

Implements ranking task type following the established task-type utilities pattern
in bead/items/forced_choice.py.

## Changes

- Added bead/items/ranking.py with:
  - create_ranking_item()
  - create_ranking_items_from_texts()
  - create_ranking_items_from_groups()
  - create_filtered_ranking_items()
- Added tests in tests/items/test_ranking.py (95% coverage)
- Updated bead/items/validation.py with ranking-specific validation
- Added ranking to TASK_TYPE_UTILITIES_PLAN.md

## Testing

```bash
pytest tests/items/test_ranking.py -v
# All tests pass, coverage 95%
```

## Checklist

- [x] Tests pass locally
- [x] Coverage >90%
- [x] Linters pass (ruff, pyright, pydocstyle, darglint)
- [x] Docstrings for all public API
- [x] Examples tested with doctest
- [x] Documentation updated
```

### 5. Review Checklist

Before submitting, verify:

**Code Quality**:
- [ ] Tests pass: `pytest tests/`
- [ ] Coverage >90%: `pytest tests/ --cov=bead --cov-report=term-missing`
- [ ] Ruff passes: `ruff check bead/ && ruff format bead/`
- [ ] Pyright passes: `pyright bead/`
- [ ] pydocstyle passes: `pre-commit run pydocstyle --all-files`
- [ ] darglint passes: `pre-commit run darglint --all-files`

**Documentation**:
- [ ] Docstrings for all public functions/classes
- [ ] NumPy format used (not Google/Sphinx)
- [ ] Examples section included and tested
- [ ] No AI-pattern words (comprehensive, robust, etc.)
- [ ] User guide updated (if applicable)
- [ ] API reference updated (if applicable)

**Testing**:
- [ ] Test file created matching source file
- [ ] Happy path tested
- [ ] Edge cases tested (empty inputs, boundary values)
- [ ] Error cases tested (invalid inputs, exceptions)
- [ ] Fixtures added to conftest.py if needed

**Code**:
- [ ] Follows established patterns (see Common Contribution Patterns below)
- [ ] No breaking changes (unless discussed in issue)
- [ ] Type hints for all parameters and returns
- [ ] No `Any` or `object` in core code
- [ ] Commits are atomic and well-described

### 6. Address Review Comments

Reviewers may request changes. Make changes in your branch:

```bash
git checkout feature/add-ranking-task-type
# Make changes
git add .
git commit -m "Address review comments: improve error messages"
git push origin feature/add-ranking-task-type
```

The PR updates automatically.

### 7. Squash Commits (if requested)

Maintainers may ask you to squash commits before merging:

```bash
# Squash last 3 commits
git rebase -i HEAD~3

# In editor, change "pick" to "squash" for commits to merge
# Save and exit

# Force push (rewrite history)
git push origin feature/add-ranking-task-type --force
```

## Common Contribution Patterns

This section shows how to implement common contributions.

### Adding a New Task Type

Task types provide utilities for creating experimental items. Follow the forced_choice.py pattern.

**Example: Adding ranking task type**

**1. Create module**: `bead/items/ranking.py`

```python
"""Ranking task type utilities.

Provides functions for creating ranking task items where participants order
multiple alternatives.
"""

from __future__ import annotations

from typing import Any, Callable
from uuid import UUID

from bead.items.item import Item


def create_ranking_item(
    *items_to_rank: str,
    prompt: str = "Rank these items from best to worst:",
    allow_ties: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Item:
    """Create a ranking task item.

    Parameters
    ----------
    *items_to_rank : str
        Items to be ranked by the participant.
    prompt : str, default="Rank these items from best to worst:"
        Instruction text shown to participant.
    allow_ties : bool, default=False
        Whether to allow tied rankings.
    metadata : dict[str, Any] | None, optional
        Additional metadata to attach to the item.

    Returns
    -------
    Item
        Ranking task item with task_type="ranking".

    Raises
    ------
    ValueError
        If fewer than 2 items provided for ranking.

    Examples
    --------
    >>> item = create_ranking_item("A", "B", "C")
    >>> item.item_metadata["n_items_to_rank"]
    3
    """
    if len(items_to_rank) < 2:
        raise ValueError("Must provide at least 2 items for ranking")

    item_metadata = {"n_items_to_rank": len(items_to_rank), "allow_ties": allow_ties}

    rendered_elements = {
        "prompt": prompt,
        **{f"item_{i}": item for i, item in enumerate(items_to_rank)},
    }

    return Item(
        task_type="ranking",
        rendered_elements=rendered_elements,
        item_metadata=item_metadata,
        metadata=metadata or {},
    )


def create_ranking_items_from_texts(
    texts: list[str],
    n_items_per_ranking: int,
    prompt: str = "Rank these items from best to worst:",
    allow_ties: bool = False,
    metadata_fn: Callable[[list[str]], dict[str, Any]] | None = None,
) -> list[Item]:
    """Create ranking items from a list of texts.

    Parameters
    ----------
    texts : list[str]
        Source texts to create rankings from.
    n_items_per_ranking : int
        Number of items in each ranking task.
    prompt : str, default="Rank these items from best to worst:"
        Instruction text.
    allow_ties : bool, default=False
        Whether to allow tied rankings.
    metadata_fn : Callable[[list[str]], dict[str, Any]] | None, optional
        Function to generate metadata from items.

    Returns
    -------
    list[Item]
        List of ranking task items.

    Examples
    --------
    >>> texts = ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4"]
    >>> items = create_ranking_items_from_texts(texts, n_items_per_ranking=3)
    >>> len(items)
    4
    """
    # Implementation...
```

**2. Add validation**: Update `bead/items/validation.py`

```python
def validate_ranking_item(item: Item) -> None:
    """Validate ranking task item structure."""
    if "n_items_to_rank" not in item.item_metadata:
        raise ValueError("Ranking item missing n_items_to_rank metadata")

    if item.item_metadata["n_items_to_rank"] < 2:
        raise ValueError("Ranking item must have at least 2 items to rank")
```

**3. Add tests**: Create `tests/items/test_ranking.py`

**4. Update exports**: Add to `bead/items/__init__.py`

**5. Update documentation**: Add to user guide (docs/user-guide/items.md)

**6. Update plans**: Add to TASK_TYPE_UTILITIES_PLAN.md

### Adding a New Constraint Type

Constraints control list partitioning. Add to `bead/lists/constraints.py`.

**Example: Adding SequentialConstraint (ensure items appear in order)**

```python
class SequentialConstraint(ListConstraint):
    """Ensure items with specified property appear in sequential order.

    Parameters
    ----------
    property_expression : str
        DSL expression to extract property value (must be orderable).
    ascending : bool, default=True
        Whether to enforce ascending order.

    Examples
    --------
    >>> constraint = SequentialConstraint(
    ...     property_expression="item['quantile']",
    ...     ascending=True
    ... )
    >>> items = [
    ...     {"quantile": 1},
    ...     {"quantile": 2},
    ...     {"quantile": 3}
    ... ]
    >>> constraint.evaluate(items)
    True

    >>> items_unsorted = [
    ...     {"quantile": 3},
    ...     {"quantile": 1},
    ...     {"quantile": 2}
    ... ]
    >>> constraint.evaluate(items_unsorted)
    False
    """

    property_expression: str
    ascending: bool = True

    def evaluate(self, items: list[dict[str, Any]]) -> bool:
        """Evaluate if items are in sequential order.

        Parameters
        ----------
        items : list[dict[str, Any]]
            Items to evaluate.

        Returns
        -------
        bool
            True if items are in order, False otherwise.
        """
        values = [
            self._evaluate_expression(self.property_expression, item) for item in items
        ]

        if self.ascending:
            return values == sorted(values)
        else:
            return values == sorted(values, reverse=True)
```

**Testing**:

```python
# tests/lists/test_constraints.py
class TestSequentialConstraint:
    """Test SequentialConstraint."""

    def test_ascending_order(self):
        """Test constraint accepts ascending order."""
        constraint = SequentialConstraint(
            property_expression="item['value']", ascending=True
        )
        items = [{"value": 1}, {"value": 2}, {"value": 3}]
        assert constraint.evaluate(items) is True

    def test_rejects_unsorted(self):
        """Test constraint rejects unsorted items."""
        constraint = SequentialConstraint(
            property_expression="item['value']", ascending=True
        )
        items = [{"value": 3}, {"value": 1}, {"value": 2}]
        assert constraint.evaluate(items) is False
```

### Extending Configuration System

Add new configuration models to `bead/config/`.

**Example: Adding RankingConfig**

**1. Create module**: `bead/config/ranking.py`

```python
"""Configuration for ranking task type."""

from pydantic import Field

from bead.data.base import BeadBaseModel


class RankingConfig(BeadBaseModel):
    """Configuration for ranking task items.

    Attributes
    ----------
    n_items_per_ranking : int
        Number of items in each ranking task.
    allow_ties : bool
        Whether to allow tied rankings.
    prompt_template : str
        Template for ranking prompt.
    """

    n_items_per_ranking: int = Field(ge=2, le=10)
    allow_ties: bool = Field(default=False)
    prompt_template: str = Field(default="Rank these items from best to worst:")
```

**2. Update root config**: Add to `bead/config/config.py`

```python
from bead.config.ranking import RankingConfig


class ItemsConfig(BeadBaseModel):
    """Items configuration."""

    ...
    ranking: RankingConfig | None = None
```

**3. Add tests**: Create `tests/config/test_ranking.py`

**4. Update example**: Add to `gallery/eng/argument_structure/config.yaml`

### Adding CLI Commands

Add CLI commands to appropriate module in `bead/cli/`.

**Example: Adding `bead items create-ranking` command**

**1. Add to CLI module**: `bead/cli/items.py`

```python
@items.command("create-ranking")
@click.argument("items-file", type=click.Path(exists=True))
@click.option("--n-items", type=int, required=True, help="Items per ranking")
@click.option("--allow-ties", is_flag=True, help="Allow tied rankings")
@click.option("--output", type=click.Path(), required=True, help="Output path")
def create_ranking_command(
    items_file: str, n_items: int, allow_ties: bool, output: str
) -> None:
    """Create ranking task items from input file.

    ITEMS_FILE should be a JSONL file containing source items.
    """
    from pathlib import Path
    from bead.items.ranking import create_ranking_items_from_texts
    from bead.data.serialization import read_jsonl, write_jsonl

    # Load source items
    source_items = read_jsonl(Path(items_file))

    # Create rankings
    ranking_items = create_ranking_items_from_texts(
        texts=[item["text"] for item in source_items],
        n_items_per_ranking=n_items,
        allow_ties=allow_ties,
    )

    # Write output
    write_jsonl(Path(output), [item.model_dump() for item in ranking_items])

    click.echo(f"Created {len(ranking_items)} ranking items")
    click.echo(f"Saved to {output}")
```

**2. Test command**:

```bash
uv run bead items create-ranking --help
```

**3. Update documentation**: Add to `docs/cli/reference.md`

## Summary

Follow these contribution guidelines:

**Before starting**:
1. Fork repository and create feature branch
2. Read architecture, setup, and testing guides
3. Check existing issues for similar work

**While coding**:
1. Follow established patterns (see Common Contribution Patterns)
2. Write NumPy docstrings for all public API
3. Add tests with >90% coverage
4. Run linters (ruff, pyright, pydocstyle, darglint)
5. Commit frequently with descriptive messages

**Before submitting PR**:
1. All tests pass
2. Coverage >90%
3. All linters pass (zero errors, zero warnings)
4. Documentation updated
5. Review checklist complete

**After submitting PR**:
1. Address review comments promptly
2. Update PR with requested changes
3. Squash commits if requested

For technical details, see [architecture.md](architecture.md). For development setup, see [setup.md](setup.md). For testing practices, see [testing.md](testing.md).

Thank you for contributing to bead!
