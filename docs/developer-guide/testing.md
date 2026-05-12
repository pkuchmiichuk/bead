# Testing

This guide explains testing practices in bead, including test organization, fixture patterns, coverage requirements, and testing strategies for different components. Following these patterns ensures reliable, maintainable tests.

## Testing Philosophy

bead follows these testing principles:

### 1. Test Organization Mirrors Source

Tests are organized to match the source code structure:

```
bead/resources/lexical_item.py  →  tests/resources/test_lexical_item.py
bead/lists/partitioner.py       →  tests/lists/test_partitioner.py
bead/items/forced_choice.py     →  tests/items/test_forced_choice.py
```

This one-to-one mapping makes finding tests easy and ensures coverage is tracked accurately.

### 2. Unit Tests with Mocks

Unit tests should test individual components in isolation. Use mocks for external dependencies:

- Mock external APIs (OpenAI, Anthropic, HuggingFace)
- Mock file system operations when testing I/O logic
- Mock expensive operations (model inference, large computations)

### 3. Integration Tests for Workflows

Integration tests verify that modules work together correctly. Place these in tests/integration/:

- End-to-end pipeline tests (resources → templates → items → lists)
- CLI command integration
- Configuration loading and validation

### 4. Target Coverage >90%

Aim for >90% line coverage across the codebase. Current coverage is ~94%. Coverage below 90% indicates missing tests or unreachable code.

### 5. Test Names Describe Behavior

Test names should describe what they test, not implementation details:

```python
# GOOD: Describes behavior
def test_create_with_all_fields(self) -> None:
    """Test creating a lexical item with all fields."""


# BAD: Describes implementation
def test_init_sets_attributes(self) -> None:
    """Test __init__ method sets attributes."""
```

## pytest Organization

bead uses pytest as the test framework. Tests are organized in a hierarchical structure matching the source code.

### Test Directory Structure

```
tests/
├── conftest.py                    # Root fixtures (tests_dir, sample_data_dir)
├── data/                          # Tests for bead/data/
│   ├── conftest.py
│   ├── test_base.py
│   ├── test_identifiers.py
│   └── test_serialization.py
├── resources/                     # Tests for bead/resources/
│   ├── conftest.py                # Resource fixtures
│   ├── test_lexical_item.py
│   ├── test_lexicon.py
│   ├── test_template.py
│   └── test_constraints.py
├── items/                         # Tests for bead/items/
│   ├── conftest.py                # Item fixtures
│   ├── test_item.py
│   ├── test_forced_choice.py
│   └── test_validation.py
├── lists/                         # Tests for bead/lists/
│   ├── conftest.py
│   ├── test_partitioner.py
│   ├── test_constraints.py
│   └── test_experiment_list.py
└── integration/                   # Integration tests
    └── test_task_type_pipeline.py
```

### Test File Naming

- Test files: `test_*.py` (pytest discovers these automatically)
- Test classes: `TestClassName` (groups related tests)
- Test functions: `test_descriptive_name()`

### Running Tests

**All tests**:
```bash
uv run pytest tests/
```

**Specific module**:
```bash
uv run pytest tests/resources/
uv run pytest tests/lists/
```

**Specific file**:
```bash
uv run pytest tests/resources/test_lexical_item.py
```

**Specific test**:
```bash
uv run pytest tests/resources/test_lexical_item.py::TestLexicalItemCreation::test_create_with_all_fields
```

**With verbose output**:
```bash
uv run pytest tests/ -v
```

**Stop on first failure**:
```bash
uv run pytest tests/ -x
```

**Show print statements**:
```bash
uv run pytest tests/ -s
```

## Fixtures

Fixtures provide reusable test data and setup code. bead uses pytest fixtures extensively.

### Fixture Hierarchy

Fixtures are organized hierarchically using conftest.py files:

**tests/conftest.py** (root-level fixtures):
```python
@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Get tests directory path."""
    return Path(__file__).parent


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
```

**tests/resources/conftest.py** (resource fixtures):
```python
@pytest.fixture
def sample_lexical_item() -> LexicalItem:
    """Provide a sample lexical item."""
    return LexicalItem(
        lemma="walk",
        language_code="eng",
        features={
            "pos": "VERB",
            "tense": "present",
            "transitive": True,
            "frequency": 1000,
        },
        source="manual",
    )


@pytest.fixture
def sample_template(sample_slot: Slot) -> Template:
    """Provide a sample template."""
    verb_slot = Slot(name="verb", required=True)
    object_slot = Slot(name="object", required=True)
    return Template(
        name="simple_transitive",
        template_string="{subject} {verb} {object}.",
        slots={
            "subject": sample_slot,
            "verb": verb_slot,
            "object": object_slot,
        },
        tags=["transitive", "simple"],
    )


@pytest.fixture
def sample_lexicon() -> Lexicon:
    """Provide a sample lexicon with multiple items."""
    lexicon = Lexicon(name="test_lexicon", language_code="en")
    lexicon.add(
        LexicalItem(
            lemma="walk",
            language_code="eng",
            features={"pos": "VERB", "frequency": 1000},
        )
    )
    lexicon.add(
        LexicalItem(
            lemma="run", language_code="eng", features={"pos": "VERB", "frequency": 800}
        )
    )
    return lexicon
```

**tests/items/conftest.py** (item fixtures):
```python
@pytest.fixture
def sample_uuid() -> UUID:
    """Create a sample UUID for testing."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def task_spec_forced_choice() -> TaskSpec:
    """Create a forced choice task specification."""
    return TaskSpec(
        prompt="Which sentence sounds more natural?",
        options=["Sentence A", "Sentence B", "Both equally natural"],
    )
```

### Fixture Scope

Control fixture lifespan with scope parameter:

- `scope="function"` (default): Create new fixture for each test
- `scope="class"`: Share fixture across tests in a class
- `scope="module"`: Share fixture across all tests in a file
- `scope="session"`: Create fixture once for entire test run

```python
@pytest.fixture(scope="session")
def expensive_resource():
    """Load expensive resource once for all tests."""
    resource = load_large_dataset()
    return resource
```

### Fixture Dependencies

Fixtures can depend on other fixtures:

```python
@pytest.fixture
def sample_slot(sample_intensional_constraint: Constraint) -> Slot:
    """Provide a sample slot."""
    return Slot(
        name="subject",
        description="The subject of the sentence",
        constraints=[sample_intensional_constraint],
        required=True,
    )


@pytest.fixture
def sample_template(sample_slot: Slot) -> Template:
    """Provide a sample template."""
    # Uses sample_slot fixture
    return Template(
        name="simple",
        template_string="{subject} verbs.",
        slots={"subject": sample_slot},
    )
```

### When to Create Fixtures

Create a fixture when:

1. Multiple tests need the same data
2. Setup is expensive or complex
3. Teardown is required (cleanup resources)
4. You want to isolate test data creation

Use inline data when:

1. Data is used in only one test
2. Setup is trivial (single line)
3. Test-specific values improve clarity

## Test Coverage

bead uses pytest-cov to measure code coverage. Target >90% coverage for all modules.

### Measuring Coverage

**Run with coverage**:
```bash
uv run pytest tests/ --cov=bead --cov-report=term-missing
```

Output shows coverage per file with uncovered line numbers:

```
---------- coverage: platform darwin, python 3.14.0 -----------
Name                                   Stmts   Miss  Cover   Missing
--------------------------------------------------------------------
bead/__init__.py                           3      0   100%
bead/data/base.py                         42      2    95%   67-68
bead/data/identifiers.py                  10      0   100%
bead/data/timestamps.py                    8      0   100%
bead/resources/lexical_item.py            85      5    94%   142-146
bead/resources/lexicon.py                 102     8    92%   187-194
...
--------------------------------------------------------------------
TOTAL                                   8547    512    94%
```

The "Missing" column shows line numbers not covered by tests.

### HTML Coverage Report

Generate visual coverage report:

```bash
uv run pytest tests/ --cov=bead --cov-report=html
```

Open htmlcov/index.html in a browser. This shows:
- Coverage percentage per file
- Line-by-line highlighting (green: covered, red: uncovered)
- Branch coverage (if/else paths)

### Coverage Configuration

Configuration in pyproject.toml:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra", "--strict-markers", "--cov=bead", "--cov-report=term-missing"]
```

### Improving Coverage

To improve coverage:

1. **Identify uncovered lines**: Run `pytest --cov=bead --cov-report=term-missing`
2. **Write tests for uncovered code**: Focus on "Missing" line numbers
3. **Remove dead code**: Delete unreachable code (if any)
4. **Add edge case tests**: Test error paths, boundary conditions

### Coverage Targets by Module

- **Core modules** (data, resources, items, lists): Target 95%+
- **CLI commands**: Target 85%+ (UI code harder to test)
- **Adapters** (external APIs): Target 80%+ (some paths require live APIs)
- **Overall project**: Target 90%+

## Mocking

Use pytest-mock (included in dev dependencies) for mocking external dependencies.

### Mocking with monkeypatch

The `monkeypatch` fixture replaces objects temporarily:

```python
def test_api_call_with_mock(monkeypatch):
    """Test API call with mocked response."""

    # Mock external API call
    def mock_api_call(prompt: str) -> str:
        return "mocked response"

    monkeypatch.setattr("bead.items.adapters.openai.call_openai_api", mock_api_call)

    # Test code that calls API
    result = generate_completion("test prompt")
    assert result == "mocked response"
```

### Mocking File I/O

Mock file operations to avoid creating actual files:

```python
def test_save_to_jsonl(monkeypatch, tmp_path):
    """Test saving to JSONL with mocked file writing."""

    written_data = []

    def mock_write(path: Path, data: list):
        written_data.extend(data)

    monkeypatch.setattr("bead.data.serialization.write_jsonl", mock_write)

    lexicon = Lexicon(name="test", language_code="eng")
    lexicon.save(tmp_path / "output.jsonl")

    assert len(written_data) > 0
```

### Mocking Model Outputs

Mock expensive model inference:

```python
def test_item_creation_with_model_scores(monkeypatch):
    """Test item creation with mocked model scores."""

    def mock_compute_score(text: str) -> float:
        return 0.85  # Mock score

    monkeypatch.setattr(
        "bead.items.adapters.huggingface.compute_perplexity", mock_compute_score
    )

    item = create_forced_choice_item_with_scores("Option A", "Option B")
    assert item.metadata["scores"] == [0.85, 0.85]
```

### Mocking External APIs

Mock OpenAI, Anthropic, HuggingFace APIs:

```python
def test_openai_adapter(monkeypatch):
    """Test OpenAI adapter with mocked API."""

    class MockResponse:
        def __init__(self):
            self.choices = [
                type(
                    "obj",
                    (object,),
                    {"message": type("obj", (object,), {"content": "mocked"})()},
                )()
            ]

    def mock_create(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("openai.ChatCompletion.create", mock_create)

    result = call_openai_gpt("prompt")
    assert result == "mocked"
```

### Mocking Environment Variables

```python
def test_api_key_from_env(monkeypatch):
    """Test loading API key from environment."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

    api_key = load_api_key()
    assert api_key == "test-key-123"
```

## Testing Patterns

### Testing Pydantic Models

Test model validation and field defaults:

```python
class TestLexicalItemValidation:
    """Test lexical item validation."""

    def test_empty_lemma_fails(self) -> None:
        """Test that empty lemma validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="", language_code="eng")
        assert "lemma must be non-empty" in str(exc_info.value)

    def test_whitespace_only_lemma_fails(self) -> None:
        """Test that whitespace-only lemma validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="   ", language_code="eng")
        assert "lemma must be non-empty" in str(exc_info.value)
```

### Testing UUID Generation

Test that UUIDs are generated correctly:

```python
def test_auto_id_generation():
    """Test that ID is automatically generated."""
    item1 = LexicalItem(lemma="test1", language_code="eng")
    item2 = LexicalItem(lemma="test2", language_code="eng")

    # Each item gets unique UUID
    assert item1.id != item2.id

    # UUIDs are UUIDv7 (time-ordered)
    assert item1.created_at < item2.created_at
    # Earlier creation time should have earlier UUID
```

### Testing Stand-off Annotation

Test that objects store UUID references, not full objects:

```python
def test_item_stores_uuid_references():
    """Test that Item stores filled template UUIDs, not full objects."""
    template_uuid1 = UUID("12345678-1234-5678-1234-567812345678")
    template_uuid2 = UUID("87654321-4321-8765-4321-876543218765")

    item = Item(
        filled_template_refs=[template_uuid1, template_uuid2],
        judgment_type="forced_choice",
    )

    # Stores UUIDs
    assert len(item.filled_template_refs) == 2
    assert item.filled_template_refs[0] == template_uuid1
    assert isinstance(item.filled_template_refs[0], UUID)
```

### Testing Constraint Satisfaction

Test constraint evaluation logic:

```python
def test_uniqueness_constraint():
    """Test UniquenessConstraint evaluates correctly."""
    constraint = UniquenessConstraint(property_expression="item['verb_lemma']")

    # Items with unique verb lemmas
    items_unique = [
        {"verb_lemma": "walk"},
        {"verb_lemma": "run"},
        {"verb_lemma": "jump"},
    ]
    assert constraint.evaluate(items_unique) is True

    # Items with duplicate verb lemmas
    items_duplicate = [
        {"verb_lemma": "walk"},
        {"verb_lemma": "run"},
        {"verb_lemma": "walk"},  # Duplicate
    ]
    assert constraint.evaluate(items_duplicate) is False
```

### Testing CLI Commands

Test CLI commands in isolation:

```python
from click.testing import CliRunner


def test_config_create_command():
    """Test 'bead config create' command."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["config", "create", "--name", "test_project", "--language", "eng"]
        )

        assert result.exit_code == 0
        assert "Created configuration" in result.output
        assert Path("config.yaml").exists()
```

### Testing File I/O

Use tmp_path fixture for temporary files:

```python
def test_save_and_load_lexicon(tmp_path):
    """Test saving lexicon to JSONL and loading it back."""
    lexicon = Lexicon(name="test", language_code="eng")
    lexicon.add(LexicalItem(lemma="walk", language_code="eng"))

    # Save to temporary file
    output_path = tmp_path / "lexicon.jsonl"
    lexicon.save(output_path)

    # Load from file
    loaded = Lexicon.load(output_path)

    assert loaded.name == "test"
    assert len(loaded) == 1
    assert loaded[0].lemma == "walk"
```

### Testing Error Handling

Test that functions raise appropriate exceptions:

```python
def test_partition_with_invalid_n_lists():
    """Test partitioner raises error for invalid n_lists."""
    partitioner = ListPartitioner()

    with pytest.raises(ValueError) as exc_info:
        partitioner.partition(
            items=[uuid4() for _ in range(10)],
            n_lists=0,  # Invalid: must be > 0
            metadata={},
        )

    assert "n_lists must be positive" in str(exc_info.value)
```

### Testing Async Code

If testing async functions (future feature):

```python
import pytest


@pytest.mark.asyncio
async def test_async_model_call():
    """Test async model API call."""
    result = await call_model_async("prompt")
    assert isinstance(result, str)
```

### Testing with Parametrize

Test multiple inputs with parametrize:

```python
@pytest.mark.parametrize(
    "lemma,expected_pos",
    [
        ("walk", "VERB"),
        ("dog", "NOUN"),
        ("quickly", "ADV"),
    ],
)
def test_infer_pos(lemma, expected_pos):
    """Test POS inference for different lemmas."""
    pos = infer_pos(lemma)
    assert pos == expected_pos
```

## doctest for Examples

bead uses doctest to verify examples in docstrings work correctly.

### Writing Doctest Examples

Include executable examples in docstrings:

```python
def add(a: int, b: int) -> int:
    """Add two integers.

    Parameters
    ----------
    a : int
        First integer.
    b : int
        Second integer.

    Returns
    -------
    int
        Sum of a and b.

    Examples
    --------
    >>> add(2, 3)
    5
    >>> add(10, -5)
    5
    >>> add(0, 0)
    0
    """
    return a + b
```

### Running Doctests

Run doctests with pytest:

```bash
uv run pytest --doctest-modules bead/
```

This executes all `>>>` examples in docstrings and verifies output matches.

### Doctest Best Practices

1. **Keep examples simple**: Test basic usage, not edge cases
2. **Show expected output**: Output should be deterministic
3. **Avoid randomness**: Use fixed seeds or avoid random operations
4. **Skip non-deterministic examples**: Use `# doctest: +SKIP` for non-reproducible output

```python
>>> generate_uuid()  # doctest: +SKIP
UUID('...')  # Random UUID
```

## Test Configuration

Configuration in pyproject.toml:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "-ra",                      # Show summary of all test results
    "--strict-markers",          # Fail on unknown markers
    "--cov=bead",               # Measure coverage for bead package
    "--cov-report=term-missing" # Show uncovered line numbers
]
```

### Custom Markers

Define custom markers for test categorization:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "requires_api: marks tests that require external API keys",
]
```

Use markers in tests:

```python
@pytest.mark.slow
def test_expensive_operation():
    """Test expensive operation (marked as slow)."""
    result = compute_large_matrix()
    assert result is not None


@pytest.mark.requires_api
def test_openai_integration():
    """Test OpenAI integration (requires API key)."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    result = call_openai("prompt")
    assert isinstance(result, str)
```

Run tests excluding slow tests:

```bash
uv run pytest tests/ -m "not slow"
```

## Continuous Integration

Tests run automatically in CI on every push and pull request.

### GitHub Actions Workflow

The CI workflow (if configured) runs:

1. Install Python 3.14
2. Install dependencies: `uv sync --all-extras`
3. Run linters: `uv run ruff check bead/`
4. Run type checker: `uv run pyright bead/`
5. Run tests: `uv run pytest tests/ --cov=bead`
6. Upload coverage to Codecov (optional)

### Required Checks for Pull Requests

Pull requests must pass:

- All tests pass
- Coverage >90%
- Ruff linting passes (zero errors)
- Pyright type checking passes (zero errors)
- pydocstyle passes (NumPy convention)
- darglint passes (signature consistency)

## Writing New Tests

When adding new features, follow this process:

### 1. Identify What to Test

For a new function:
- Normal inputs (happy path)
- Edge cases (empty lists, None values, boundaries)
- Error cases (invalid inputs, exceptions)
- Integration with existing code

### 2. Create Test File

If testing `bead/lists/new_feature.py`, create `tests/lists/test_new_feature.py`:

```python
"""Tests for new_feature module."""

from __future__ import annotations

import pytest

from bead.lists.new_feature import new_function


class TestNewFunction:
    """Test new_function behavior."""

    def test_normal_case(self) -> None:
        """Test function with normal inputs."""
        result = new_function(input_data)
        assert result == expected_output

    def test_edge_case_empty(self) -> None:
        """Test function with empty input."""
        result = new_function([])
        assert result == []

    def test_invalid_input_raises(self) -> None:
        """Test function raises ValueError for invalid input."""
        with pytest.raises(ValueError):
            new_function(invalid_data)
```

### 3. Add Fixtures if Needed

If multiple tests need same data, add fixture to conftest.py:

```python
@pytest.fixture
def sample_new_feature_data():
    """Provide sample data for new_feature tests."""
    return create_sample_data()
```

### 4. Run Tests Locally

```bash
uv run pytest tests/lists/test_new_feature.py -v
```

Verify all tests pass and coverage is >90%.

### 5. Update Documentation

Add docstrings to test functions explaining what they test and why.

## Summary

Follow these testing practices:

1. **Organize tests** to mirror source code structure
2. **Use fixtures** for reusable test data
3. **Mock external dependencies** (APIs, file I/O, expensive operations)
4. **Target >90% coverage** for all modules
5. **Write descriptive test names** that explain behavior
6. **Test edge cases and errors**, not just happy paths
7. **Use parametrize** for testing multiple inputs
8. **Include doctest examples** in docstrings
9. **Run tests before committing**: `uv run pytest tests/`
10. **Check coverage**: `uv run pytest tests/ --cov=bead --cov-report=term-missing`

For architecture details, see [architecture.md](architecture.md). For contribution guidelines, see [contributing.md](contributing.md). For development setup, see [setup.md](setup.md).
