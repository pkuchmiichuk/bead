# bead

[![CI](https://github.com/FACTSlab/bead/actions/workflows/ci.yml/badge.svg)](https://github.com/FACTSlab/bead/actions/workflows/ci.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://factslab.io/bead/)

A Python framework for constructing, deploying, and analyzing large-scale linguistic judgment experiments with active learning.

## Overview

`bead` implements a complete pipeline for linguistic research: from lexical resource construction through experimental deployment to model training with active learning. It handles the combinatorial explosion of linguistic stimuli while maintaining full provenance tracking.

The name refers to the way sealant is applied while glazing a window, a play on the [glazing](https://github.com/FACTSlab/glazing) package for accessing VerbNet, PropBank, and FrameNet.

## Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install bead
uv pip install bead

# With optional dependencies
uv pip install bead[api]       # OpenAI, Anthropic, Google APIs
uv pip install bead[training]  # PyTorch Lightning, TensorBoard
```

### Development

```bash
git clone https://github.com/FACTSlab/bead.git
cd bead
uv sync --all-extras
uv run pytest tests/
```

Always use `uv run` to execute commands.

Integration tests that exercise the layers publish path against a real ATProto
PDS are deselected by default. With docker running, opt in with
`uv run pytest --run-integration`; the test stands up a throwaway
[bluesky PDS](https://github.com/bluesky-social/pds) container and skips cleanly
when docker is unavailable.

## Quick Start

```python
from bead.items.forced_choice import create_forced_choice_item
from bead.lists.partitioner import ListPartitioner
from bead.protocol import (
    AnnotationProtocol,
    QuestionFamily,
    ResponseSpace,
    ScaleType,
    SemanticAnchor,
)
from bead.protocol.items import family_to_item_template

# 1. Declare the question being asked
anchor = SemanticAnchor(
    name="acceptability",
    target_property="acceptability",
    canonical_prompt="Which sentence sounds more natural?",
    response_space=ResponseSpace(
        options=("first", "second"),
        is_ordered=False,
        scale_type=ScaleType.FORCED_CHOICE,
    ),
    required_keywords=frozenset({"natural"}),
)
protocol = AnnotationProtocol(families=[QuestionFamily(anchor=anchor)])

# 2. Build the deployable item template from the protocol
template = family_to_item_template(
    protocol.family_by_name("acceptability"),
    judgment_type="acceptability",
)

# 3. Build forced-choice items (one per minimal pair)
items = [
    create_forced_choice_item(
        "The cat sat on the mat.",
        "The cats sat on the mat.",
        item_template_id=template.id,
        metadata={"anchor": "acceptability", "contrast": "number"},
    ),
    # ... more pairs
]

# 4. Partition into experiment lists
partitioner = ListPartitioner(random_seed=42)
lists = partitioner.partition(
    [item.id for item in items],
    n_lists=4,
    metadata={item.id: dict(item.item_metadata) for item in items},
)
```

Or, drive the same pipeline from a single declarative config:

```python
from bead.config import load_config

# Composes profile defaults → defaults: [...] entries → primary YAML
# → extras → CLI-style overrides → resolves ${...} interpolation
config = load_config(
    "config.yaml",
    overrides=["paths.data_dir=/tmp/data"],
)
protocol = config.protocol.build()
```

## Pipeline Stages

| Stage | Purpose | Output |
|-------|---------|--------|
| **Resources** | Define lexical items and templates | `lexicons/*.jsonl`, `templates/*.jsonl` |
| **Templates** | Fill templates with lexical items | `filled_templates/*.jsonl` |
| **Items** | Construct experimental items | `items/*.jsonl` |
| **Lists** | Partition into balanced lists | `lists/*.jsonl` |
| **Deployment** | Generate jsPsych experiments | `deployment/*.jzip` |
| **Training** | Active learning until convergence | Model checkpoints |

## Key Features

- **Stand-off annotation**: UUID-based references for full provenance tracking
- **8 task types**: forced-choice, ordinal scale, binary, categorical, multi-select, magnitude, free text, cloze
- **Constraint satisfaction**: batch and list-level constraints for balanced designs
- **Model integration**: HuggingFace, OpenAI, Anthropic with caching
- **Active learning**: uncertainty sampling with convergence detection
- **Annotation protocols**: type-theoretic stack of `SemanticAnchor` (the question type), `ProtocolContext` (the dependent index), `RealizationStrategy` (template / contextual / LM phrasings), and `DriftGuard` (the type-checker over realized prompts), composed into conditional `AnnotationProtocol`s
- **Config composer** (`bead.config.compose`): the full OmegaConf interpolation grammar — `${section.field}`, `${.x}` / `${..y}` relative references, `${a.b[0]}` / `${a.b.0}` list indexing, `${a.${b}}` nesting, `\${literal}` escape, built-in resolvers (`oc.env`, `oc.select`, `oc.decode`, `oc.deprecated`, `oc.create`, `oc.dict.keys`, `oc.dict.values`); `defaults: [...]` composition; strict-merge against didactic schemas; YAML and TOML
- **jsPsych 8.x**: Material Design UI with JATOS deployment

## CLI

```bash
bead init my-experiment            # Create project structure
bead templates fill                # Fill templates
bead items construct               # Construct items
bead lists partition               # Create experiment lists
bead deploy                        # Generate jsPsych experiment
bead training run                  # Train with active learning
bead protocol validate             # Validate the protocol section of a config
bead protocol realize              # Materialize realizations for contexts
bead protocol items                # Bridge a protocol to item templates
```

Every command accepts repeatable `--set KEY=VALUE` overrides applied
through the config composer, so any field of `BeadConfig` (including
nested `paths.data_dir`, `protocol.drift.min_length`, etc.) can be
overridden from the shell without editing the YAML.

## Documentation

Full documentation: [bead.readthedocs.io](https://factslab.io/bead/)

- [Installation Guide](https://factslab.io/bead/installation/)
- [User Guide](https://factslab.io/bead/user-guide/)
- [API Reference](https://factslab.io/bead/api/)
- [Gallery Examples](https://factslab.io/bead/examples/)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{white2026bead,
  author = {White, Aaron Steven},
  title = {bead: A framework for large-scale linguistic judgment experiments},
  year = {2026},
  url = {https://github.com/FACTSlab/bead},
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This project was developed by [Aaron Steven White](https://aaronstevenwhite.io/) at the University of Rochester with support from the National Science Foundation (NSF-BCS-2237175 *CAREER: Logical Form Induction*, NSF-BCS-2040831 *Computational Modeling of the Internal Structure of Events*). It was architected and implemented with the assistance of Claude Code.
