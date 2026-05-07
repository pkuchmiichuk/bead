# bead

[![CI](https://github.com/FACTSlab/bead/actions/workflows/ci.yml/badge.svg)](https://github.com/FACTSlab/bead/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://bead.readthedocs.io)

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

## Quick Start

```python
from bead.resources import LexicalItem, Template, Lexicon
from bead.templates import TemplateFiller
from bead.items import ItemConstructor
from bead.lists import ListPartitioner

# 1. Define resources
verbs = Lexicon(items=[
    LexicalItem(lemma="walk", pos="VERB", features={"transitive": False}),
    LexicalItem(lemma="eat", pos="VERB", features={"transitive": True}),
])

template = Template(
    text="The person {verb} the thing",
    slots=["verb"],
    language_code="en"
)

# 2. Fill templates
filler = TemplateFiller(strategy="exhaustive")
filled = filler.fill(templates=[template], lexicons={"verbs": verbs})

# 3. Construct items
constructor = ItemConstructor(models=["gpt2"])
items = constructor.construct_forced_choice_items(filled, n_alternatives=2)

# 4. Partition into lists
partitioner = ListPartitioner()
lists = partitioner.partition(items.get_uuids(), n_lists=4)

# 5. Deploy
lists.save("lists/experiment.jsonl")
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
- **jsPsych 8.x**: Material Design UI with JATOS deployment

## CLI

```bash
bead init my-experiment     # Create project structure
bead templates fill         # Fill templates
bead items construct        # Construct items
bead lists partition        # Create experiment lists
bead deploy                 # Generate jsPsych experiment
bead training run           # Train with active learning
```

## Documentation

Full documentation: [bead.readthedocs.io](https://bead.readthedocs.io)

- [Installation Guide](https://bead.readthedocs.io/installation/)
- [User Guide](https://bead.readthedocs.io/user-guide/)
- [API Reference](https://bead.readthedocs.io/api/)
- [Gallery Examples](https://bead.readthedocs.io/examples/)

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
