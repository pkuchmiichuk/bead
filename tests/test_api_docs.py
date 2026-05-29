"""Test Python code blocks in API documentation.

Uses pytest-examples to extract and test code blocks from markdown files.
"""

import os
import sys
from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

# Check if glazing data is available
GLAZING_DATA_DIR = Path.home() / ".local" / "share" / "glazing" / "converted"
GLAZING_DATA_AVAILABLE = (GLAZING_DATA_DIR / "verbnet.jsonl").exists()

# Path to API documentation
DOCS_DIR = Path(__file__).parent.parent / "docs" / "user-guide" / "api"

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "api_docs"

# Path to gallery (for importing gallery utils)
GALLERY_DIR = Path(__file__).parent.parent / "gallery" / "eng" / "argument_structure"


@pytest.fixture(scope="module")
def setup_test_environment():
    """Set up environment for executing code examples.

    This fixture:
    1. Creates a temporary working directory for API tests
    2. Copies all fixtures from api_docs to api_work
    3. Adds gallery to sys.path for imports
    4. Cleans up after all tests complete
    """
    import shutil  # noqa: PLC0415

    # Add gallery to sys.path so we can import utils
    if str(GALLERY_DIR) not in sys.path:
        sys.path.insert(0, str(GALLERY_DIR))

    # Create temporary working directory
    work_dir = Path(__file__).parent / "fixtures" / "api_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy fixtures to working directory
    for item in FIXTURES_DIR.iterdir():
        dest = work_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()

        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Change to working directory so relative paths work
    original_dir = os.getcwd()
    os.chdir(work_dir)

    yield

    # Restore original directory and clean up
    os.chdir(original_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.parametrize("example", find_examples(DOCS_DIR), ids=str)
def test_api_docs_code_blocks(
    example: CodeExample, eval_example: EvalExample, setup_test_environment: None
) -> None:
    """Test that code blocks in API docs are syntactically valid and executable.

    This uses pytest-examples to:
    1. Extract Python code blocks from markdown
    2. Check syntax validity via linting (black + ruff)
    3. Execute code blocks to verify they actually work
    4. When --update-examples is used, format code blocks in place

    Parameters
    ----------
    example : CodeExample
        The code example extracted from markdown
    eval_example : EvalExample
        The evaluator fixture provided by pytest-examples
    """
    # Skip glazing-related examples if glazing data is not available
    # Check for glazing imports or VerbNet/PropBank/FrameNet extractors that use glazing
    glazing_indicators = ["glazing", "verbnet", "propbank", "framenet"]
    if not GLAZING_DATA_AVAILABLE and any(
        ind in example.source.lower() for ind in glazing_indicators
    ):
        pytest.skip("Glazing data not available (run 'glazing download' first)")

    # Skip examples that require optional NLP parser models (spaCy/Stanza) or
    # external model APIs (OpenAI/Anthropic) - these resources are not available
    # in CI, like glazing data above.
    optional_backend_indicators = [
        "StanzaParser",
        "SpacyParser",
        "create_parser",
        "sample_corpus",
        "parse_records",
        "filter_by_structure",
        "CompletionCorpusSource",
        "OpenAIAdapter",
        "AnthropicAdapter",
    ]
    if any(ind in example.source for ind in optional_backend_indicators):
        pytest.skip("Requires an optional NLP parser model or model API")

    # Ignore D100 (module docstrings), D102 (method docstrings), F821 (undefined),
    # F401 (unused imports), E402 (imports not at top), I001 (import sorting) -
    # isolated documentation snippets showing specific concepts, not complete scripts
    eval_example.set_config(
        ruff_ignore=["D100", "D102", "F821", "F401", "E402", "I001"]
    )

    # When --update-examples is passed, format and update print statements
    # Otherwise, lint and execute to verify the code actually runs
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        # Execute the code to verify it works
        eval_example.run(example)
