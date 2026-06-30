# Development Setup

This guide walks you through setting up a development environment for contributing to bead. Follow these steps to install dependencies, configure tooling, and verify your setup works correctly.

## Prerequisites

### Required Software

bead requires Python 3.14+ for modern type hint syntax (PEP 695 generic type parameters). Check your version:

```bash
python3 --version  # Should show 3.14.0 or higher
```

If you need to install Python 3.14:

- **macOS**: `brew install python@3.14`
- **Linux**: Install from source or use pyenv
- **Windows**: Download from python.org

**Git**

Clone the repository and manage version control:

```bash
git --version  # Any recent version works
```

**Package Manager**

uv is required for package management:

```bash
uv --version
```

### Platform Compatibility

bead development works on macOS, Linux, and Windows. This guide uses Unix-style paths and commands. Windows users should adapt:

- Use backslashes `\` for paths (or forward slashes `/` in PowerShell)
- Some bash commands may require Git Bash or WSL

## Clone Repository

Clone the bead repository from GitHub:

```bash
git clone https://github.com/FACTSlab/bead.git
cd bead
```

### Repository Structure

After cloning, you'll see this structure:

```
bead/
├── bead/                   # Main package code (14 modules)
├── tests/                  # Test suite (143 test files)
├── docs/                   # Documentation source
├── gallery/                # Example projects (eng/argument_structure)
├── pyproject.toml          # Project configuration
├── README.md               # Package overview
├── .pre-commit-config.yaml # Pre-commit hooks
└── .gitignore              # Git ignore rules
```

## Development Dependencies

Install bead with all development dependencies using uv:

```bash
uv sync --all-extras
```

This command:
- Installs bead in editable mode: Changes to source code take effect immediately
- Installs base dependencies (Pydantic, PyYAML, Click, etc.)
- Installs `dev` group: pytest, ruff, pyright, pytest-cov, pytest-mock
- Installs `api` group: openai, anthropic, google-generativeai
- Installs `training` group: pytorch-lightning, tensorboard

### Dependency Groups

**Base dependencies** (required, from pyproject.toml):
- pydantic>=2.0.0: Data validation
- pyyaml>=6.0.0: Configuration loading
- jinja2>=3.0.0: Template rendering
- uuid-utils>=0.7.0: UUIDv7 generation
- pandas>=2.0.0, polars>=0.19.0: Data manipulation
- langcodes>=3.3.0: Language code handling
- glazing>=0.2.0: VerbNet, PropBank, FrameNet access
- unimorph>=0.0.4: Morphological features
- torch>=2.0.0: PyTorch for models
- transformers>=4.30.0: HuggingFace models
- sentence-transformers>=2.0.0: Sentence embeddings
- scikit-learn>=1.3.0: ML utilities
- click>=8.0.0: CLI framework
- rich>=13.0.0: Rich terminal output
- lark>=1.0.0: DSL parser
- psutil>=5.9.0: System utilities

**dev dependencies** (development tools):
- pytest>=7.4.0: Test framework
- pytest-cov>=4.1.0: Coverage reporting
- pytest-mock>=3.11.0: Mocking utilities
- ruff>=0.1.0: Linter and formatter
- pyright>=1.1.0: Type checker
- pandas-stubs>=2.0.0: Type stubs for pandas

**api dependencies** (external model APIs):
- openai>=1.0.0: OpenAI API
- anthropic>=0.8.0: Anthropic Claude API
- google-generativeai>=0.3.0: Google Gemini API

**training dependencies** (active learning):
- pytorch-lightning>=2.0.0: Training framework
- tensorboard>=2.13.0: Visualization

### Verify Installation

```bash
# Check bead CLI installed
uv run bead --version
# Output: bead, version 0.2.0

# Check development tools
uv run pytest --version
# Output: pytest 7.4.x

uv run ruff --version
# Output: ruff 0.x.x

uv run pyright --version
# Output: pyright 1.1.x
```

If any command fails, the dependency didn't install correctly. Try:

```bash
uv sync --all-extras --reinstall
```

## Pre-commit Hooks

Install pre-commit hooks to automatically check code quality before commits.

### Install Hooks

```bash
uv run pre-commit install
```

This installs git hooks in .git/hooks/ that run automatically before each commit.

### Configured Hooks

The .pre-commit-config.yaml file configures two hooks:

**1. pydocstyle** (NumPy convention):
```yaml
- repo: https://github.com/pycqa/pydocstyle
  rev: 6.3.0
  hooks:
    - id: pydocstyle
      args:
        - --convention=numpy
        - --add-ignore=D105,D107  # Ignore magic methods and __init__
      exclude: ^(tests/|bead/dsl/parser\.py)
```

Checks docstrings follow NumPy format (see [Documentation Requirements](contributing.md#documentation-requirements) in contributing.md).

**2. darglint** (signature consistency):
```yaml
- repo: https://github.com/terrencepreilly/darglint
  rev: v1.8.1
  hooks:
    - id: darglint
      args:
        - -v
        - "2"  # Strictness level 2
      exclude: ^(tests/|bead/dsl/parser\.py|bead/active_learning/|bead/simulation/)
```

Checks docstring parameters match function signatures.

### Run Hooks Manually

Test hooks without making a commit:

```bash
uv run pre-commit run --all-files
```

This runs pydocstyle and darglint on all Python files. Expected output:

```
pydocstyle...........................................................Passed
darglint.............................................................Passed
```

If hooks fail, fix the reported issues before committing.

### Skip Hooks (not recommended)

To bypass hooks for a specific commit:

```bash
git commit --no-verify -m "message"
```

Use this sparingly, as it skips quality checks.

## Linting and Type Checking

bead uses ruff for linting/formatting and pyright for type checking.

### Ruff (Linter and Formatter)

**Check for issues**:
```bash
uv run ruff check bead/
```

This reports:
- E errors: PEP 8 violations
- F errors: PyFlakes issues (unused imports, undefined names)
- I errors: Import sorting
- N errors: Naming conventions
- D errors: Docstring issues
- UP errors: Python version upgrade suggestions
- ANN errors: Missing type annotations
- B errors: Bugbear (likely bugs)
- A errors: Built-in shadowing
- C4 errors: Comprehensions
- PLC errors: Pylint conventions

**Auto-fix issues**:
```bash
uv run ruff check bead/ --fix
```

Many issues can be fixed automatically.

**Format code**:
```bash
uv run ruff format bead/
```

This reformats code to 88-character line length (black-compatible).

**Configuration** (pyproject.toml):
```toml
[tool.ruff]
line-length = 88
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "D", "UP", "ANN", "B", "A", "C4", "PLC"]
ignore = ["ANN401"]  # Allow Any type in specific cases

[tool.ruff.lint.pydocstyle]
convention = "numpy"
```

### Pyright (Type Checker)

**Check types**:
```bash
uv run pyright bead/
```

This performs static type analysis, catching:
- Type mismatches
- Missing return types
- Incorrect function calls
- Undefined variables

**Configuration** (pyproject.toml):
```toml
[tool.pyright]
typeCheckingMode = "strict"
pythonVersion = "3.14"
exclude = [
    "tests/**",  # Tests don't require full type checking
    "bead/active_learning/**",
    "bead/resources/adapters/unimorph.py",
    "bead/resources/adapters/glazing.py",
    "bead/templates/adapters/huggingface.py",
    "bead/templates/strategies.py",
    "bead/items/adapters/**",  # External APIs have dynamic types
    "bead/dsl/parser.py",
    "bead/deployment/jspsych/ui/components.py",
    "bead/simulation/**"
]
```

Adapters are excluded because external APIs (OpenAI, HuggingFace) have dynamic types.

**Fix type errors**: Add type hints or adjust type annotations based on error messages.

### Run All Checks

Run all quality checks before committing:

```bash
uv run ruff check bead/ && uv run ruff format bead/ && uv run pyright bead/
```

Or create a shell alias:

```bash
alias bead-lint="uv run ruff check bead/ && uv run ruff format bead/ && uv run pyright bead/"
```

## Running Tests

Run tests using uv:

```bash
uv run pytest tests/
```

### Run All Tests

```bash
uv run pytest tests/
```

This runs all 143 test files. Expected output:

```
============================= test session starts ==============================
platform darwin -- Python 3.14.0, pytest-7.4.3, pluggy-1.3.0
rootdir: /path/to/bead
configfile: pyproject.toml
plugins: cov-4.1.0, mock-3.11.0
collected 1247 items

tests/data/test_base.py ........                                          [  0%]
tests/data/test_identifiers.py .....                                      [  0%]
...
tests/resources/test_lexical_item.py .........................            [  8%]
...
============================== 1247 passed in 45.23s ===============================
```

### Run Specific Module

Test a specific module:

```bash
uv run pytest tests/resources/
uv run pytest tests/lists/
uv run pytest tests/items/
```

Test a specific file:

```bash
uv run pytest tests/resources/test_lexical_item.py
```

Test a specific test class or function:

```bash
uv run pytest tests/resources/test_lexical_item.py::TestLexicalItemCreation
uv run pytest tests/resources/test_lexical_item.py::TestLexicalItemCreation::test_create_with_all_fields
```

### Run with Verbose Output

```bash
uv run pytest tests/ -v
```

Shows each test name as it runs:

```
tests/data/test_base.py::TestBeadBaseModel::test_default_values PASSED   [  0%]
tests/data/test_base.py::TestBeadBaseModel::test_auto_id PASSED          [  0%]
...
```

### Run with Coverage

```bash
uv run pytest tests/ --cov=bead --cov-report=term-missing
```

This shows code coverage with line numbers of uncovered code:

```
---------- coverage: platform darwin, python 3.14.0 -----------
Name                                   Stmts   Miss  Cover   Missing
--------------------------------------------------------------------
bead/__init__.py                           3      0   100%
bead/data/base.py                         42      2    95%   67-68
bead/data/identifiers.py                  10      0   100%
...
--------------------------------------------------------------------
TOTAL                                   8547    512    94%
```

Generate HTML coverage report:

```bash
uv run pytest tests/ --cov=bead --cov-report=html
```

Open htmlcov/index.html in a browser to see visual coverage report.

### Stop on First Failure

```bash
uv run pytest tests/ -x
```

Stops after the first failing test (useful for debugging).

### Run Tests in Parallel

```bash
uv run pytest tests/ -n auto
```

Runs tests in parallel using all CPU cores (requires pytest-xdist, not included by default).

### Configuration

Test configuration in pyproject.toml:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra", "--strict-markers", "--cov=bead", "--cov-report=term-missing"]
```

- `-ra`: Show summary of all test results
- `--strict-markers`: Fail on unknown markers
- `--cov=bead`: Measure coverage for bead/ package
- `--cov-report=term-missing`: Show uncovered lines

## IDE Configuration

Configure your IDE for optimal development experience.

### VS Code

**Recommended Extensions**:
- Python (ms-python.python): Language support
- Pylance (ms-python.vscode-pylance): Type checking and IntelliSense
- Ruff (charliermarsh.ruff): Linting and formatting
- autoDocstring (njpwerner.autodocstring): NumPy docstring generation

**Settings** (.vscode/settings.json):
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.analysis.typeCheckingMode": "strict",
  "editor.formatOnSave": true,
  "editor.rulers": [88],
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "ruff.format.args": ["--line-length=88"],
  "ruff.lint.args": ["--select=E,F,I,N,D,UP,ANN,B,A,C4,PLC"],
  "autoDocstring.docstringFormat": "numpy",
  "autoDocstring.startOnNewLine": false
}
```

### PyCharm

**Configure Interpreter**:
1. File → Settings → Project → Python Interpreter
2. Click gear icon → Add
3. Select "Existing environment"
4. Choose .venv/bin/python

**Enable Type Checking**:
1. File → Settings → Editor → Inspections
2. Enable "Python → Type checker"
3. Set severity to "Error"

**Configure Docstring Format**:
1. File → Settings → Tools → Python Integrated Tools
2. Set "Docstring format" to "NumPy"

**External Tools** (ruff, pyright):
1. File → Settings → Tools → External Tools
2. Add tool for "ruff check"
3. Add tool for "ruff format"
4. Add tool for "pyright"

## Verify Setup

Run these commands to verify your development environment is fully functional:

### 1. Check CLI

```bash
uv run bead --version
# Expected: bead, version 0.2.0
```

### 2. Run Quick Test

```bash
uv run pytest tests/data/test_base.py -v
# Expected: All tests pass
```

### 3. Check Linting

```bash
uv run ruff check bead/data/base.py
# Expected: No issues (or minor warnings)
```

### 4. Check Types

```bash
uv run pyright bead/data/base.py
# Expected: 0 errors, 0 warnings
```

### 5. Test Pre-commit

```bash
uv run pre-commit run --all-files
# Expected: pydocstyle and darglint pass
```

### 6. Import bead

```bash
uv run python -c "from bead.resources import LexicalItem; print(LexicalItem.__name__)"
# Expected: LexicalItem
```

If all checks pass, your development environment is ready.

## Troubleshooting

### Python Version Issues

**Problem**: `uv sync` fails with "Requires Python >=3.14"

**Solution**: Install Python 3.14:
```bash
# macOS
brew install python@3.14

# Linux (pyenv)
pyenv install 3.14.0
pyenv local 3.14.0

# Windows
# Download from python.org
```

### Module Import Errors

**Problem**: `ModuleNotFoundError: No module named 'bead'`

**Solution**: Ensure you are using `uv run` to execute Python commands:
```bash
uv run python -c "import bead"
```

### Pre-commit Hook Failures

**Problem**: Pre-commit hooks fail with "command not found"

**Solution**: Install pre-commit hooks:
```bash
uv run pre-commit install
```

### Pyright Not Found

**Problem**: `pyright: command not found`

**Solution**: Run via uv (included in dev dependencies):
```bash
uv run pyright bead/
```

### Test Failures on Clean Install

**Problem**: Tests fail immediately after cloning

**Solution**:
1. Ensure Python 3.14+ is available
2. Reinstall dependencies: `uv sync --all-extras --reinstall`
3. Clear pytest cache: `rm -rf .pytest_cache`
4. Run tests again: `uv run pytest tests/`

## Next Steps

Now that your development environment is configured:

1. **Read Architecture**: Understand system design in [architecture.md](architecture.md)
2. **Learn Testing**: Review testing patterns in [testing.md](testing.md)
3. **Start Contributing**: Follow guidelines in [contributing.md](contributing.md)
4. **Explore Codebase**: Run gallery examples in gallery/eng/argument_structure/
5. **Join Discussions**: Open issues or discussions on GitHub

For questions, open an issue or start a discussion on the GitHub repository.
