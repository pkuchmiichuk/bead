# User Guide Overview

This guide shows how to use bead for linguistic judgment experiments. Bead provides two complementary approaches for building experimental pipelines:

## CLI Approach

**Command-line interface for configuration-driven workflows**

[Get started with the CLI →](cli/index.md)

**Best for:**
- Shell scripts and automation
- Simple, linear workflows
- Avoiding Python programming
- Quick prototyping and testing
- Single-operation tasks

**Example:**
```bash
# Complete pipeline in 6 commands
uv run bead resources import-verbnet --output lexicons/verbs.jsonl
uv run bead templates fill templates.jsonl lexicons/*.jsonl filled.jsonl
uv run bead items construct --filled-templates filled.jsonl items.jsonl
uv run bead lists partition items.jsonl lists/ --n-lists 5
uv run bead deployment generate lists/ items.jsonl experiment/
uv run bead training collect-data results.jsonl
```

## API Approach

**Python API for programmatic control**

[Get started with the API →](api/index.md)

**Best for:**
- Batch operations (creating 1000s of items)
- Complex logic and control flow
- Custom processing pipelines
- Integration with existing Python code
- Dynamic configuration

**Example:**
```python
from bead.resources.adapters.glazing import GlazingAdapter
from bead.templates.filler import TemplateFiller
from bead.items.forced_choice import create_forced_choice_items_from_groups

# Programmatic pipeline with custom logic
adapter = GlazingAdapter(resource="verbnet")
verbs = adapter.fetch_items(query="break", language_code="en")
filled = filler.fill(strategy="exhaustive")
items = create_forced_choice_items_from_groups(filled, group_by=lambda t: t.template_id)
```

## Which Approach Should I Use?

Use the **CLI** when:
- You prefer configuration files over code
- Your workflow is straightforward and linear
- You're comfortable with shell commands
- You want minimal setup

Use the **API** when:
- You need batch operations or loops
- You have complex conditional logic
- You want to integrate with other Python libraries
- You need fine-grained control over the pipeline

**You can mix both approaches!** Use CLI for simple stages and Python for complex ones.

## Core Concepts

Before using either approach, familiarize yourself with these concepts:

- [Pipeline Architecture](concepts.md): 6-stage experimental pipeline
- [Configuration System](configuration.md): YAML-based project configuration
- [Stand-off Annotation](concepts.md#stand-off-annotation): UUID-based data provenance
- [Annotation Protocols](protocols.md): anchors as types, contexts as
  dependent indices, realization strategies as computational content,
  and drift guards as type-checkers

## Quick Start

**New to bead?** Start here:

1. Read [Core Concepts](concepts.md) to understand the architecture
2. Choose your approach: [CLI](cli/index.md) or [API](api/index.md)
3. Follow a complete workflow example: [CLI Workflows](cli/workflows.md) or API Workflows (coming in Phase 3)
4. Explore stage-specific guides as needed

## Getting Help

- **CLI Reference**: Run `uv run bead --help` or `uv run bead <command> --help`
- **API Reference**: See module docstrings and the gallery examples in `gallery/eng/argument_structure/`
- **Examples**: Browse `gallery/` directory for complete working examples

## Next Steps

- [CLI Guide](cli/index.md): Command-line interface
- [API Guide](api/index.md): Python API (Phase 3)
- [Configuration](configuration.md): Project setup
- [Concepts](concepts.md): Core principles
