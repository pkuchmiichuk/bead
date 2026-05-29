# Python API

The bead Python API provides programmatic control over the entire experimental pipeline. Use the API when you need batch operations, complex logic, or integration with existing Python code.

## When to Use the API

**Use the Python API when:**
- You need batch operations (creating 1000s of items)
- You need complex logic or control flow
- You want to integrate with existing Python pipelines
- You need dynamic configuration

**Use the CLI when:**
- You have simple, linear workflows
- You prefer configuration files over code
- You want quick prototyping

See [CLI guide](../cli/index.md) for command-line workflows.

## Installation

```bash
# Install with API dependencies
uv sync --extra api

# Or with all dependencies
uv sync --all-extras
```

## Quick Start

This example shows all 6 stages of the pipeline:

```python
from pathlib import Path

# Stage 1: Resources - Load lexicons
from bead.resources.lexicon import Lexicon

nouns = Lexicon.from_jsonl(Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns")

print(f"Loaded {len(nouns.items)} nouns")

# Stage 2: Templates - Load pre-generated filled templates
from bead.data.serialization import read_jsonlines
from bead.templates.filler import FilledTemplate

filled = read_jsonlines(
    Path("filled_templates/generic_frames_filled.jsonl"),
    FilledTemplate,
)[:20]

print(f"Loaded {len(filled)} filled templates")

# Stage 3: Items - Create experimental items
from bead.items.forced_choice import create_forced_choice_item

# Convert filled templates to texts
texts = [ft.rendered_text for ft in filled]

# Create forced choice items
items = []
for i in range(0, len(texts) - 1, 2):
    item = create_forced_choice_item(texts[i], texts[i + 1])
    items.append(item)

print(f"Created {len(items)} items")

# Stage 4: Lists - Partition into balanced lists
from bead.lists.partitioner import ListPartitioner

partitioner = ListPartitioner(random_seed=42)
item_uuids = [item.id for item in items]
metadata = {item.id: {"metadata": dict(item.item_metadata)} for item in items}

lists = partitioner.partition(
    items=item_uuids,
    n_lists=2,
    strategy="balanced",
    metadata=metadata,
)

print(f"Created {len(lists)} lists")

# Stage 5: Deployment - Generate jsPsych experiment
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jspsych.config import ExperimentConfig
from bead.deployment.jspsych.config import InstructionsConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec

# Create minimal item template
template = ItemTemplate(
    name="forced_choice",
    description="2AFC",
    judgment_type="acceptability",
    task_type="forced_choice",
    task_spec=TaskSpec(
        prompt="Which sentence sounds more natural?",
        options=["Option A", "Option B"],
    ),
    presentation_spec=PresentationSpec(mode="static"),
)

# Link items to template
items = [item.with_(item_template_id=template.id) for item in items]
items_dict = {item.id: item for item in items}

config = ExperimentConfig(
    experiment_type="forced_choice",
    title="Acceptability Study",
    description="Rate sentence pairs",
    instructions=InstructionsConfig.from_text("Select the more natural sentence"),
    distribution_strategy=ListDistributionStrategy(
        strategy_type=DistributionStrategyType.BALANCED
    ),
)

generator = JsPsychExperimentGenerator(
    config=config,
    output_dir=Path("/tmp/deployment"),
)

output_dir = generator.generate(
    lists=lists,
    items=items_dict,
    templates={template.id: template},
)

print(f"Experiment generated in {output_dir}")
```

## Stage-by-Stage Documentation

Each stage has detailed documentation:

- [Stage 1: Resources](resources.md) - Lexicons, templates, external databases
- [Stage 2: Templates](templates.md) - Filling strategies, renderers
- [Stage 3: Items](items.md) - Task-type utilities, scoring
- [Stage 4: Lists](lists.md) - Partitioning, constraints
- [Stage 5: Deployment](deployment.md) - jsPsych generation, JATOS export
- [Stage 6: Training](training.md) - Active learning, convergence detection

Upstream of Stage 1, you can build naturalistic stimuli directly from text:

- [Corpus Ingestion](corpus.md) - Stream a corpus, dependency-parse it, and keep
  only sentences whose syntactic structure matches a constraint

## Complete Workflow

See [workflows.md](workflows.md) for complete end-to-end examples with all configuration options.

## Design Principles

The bead API follows these principles:

1. **Stand-off Annotation**: Objects reference each other by UUID, never copy data
2. **Type Safety**: Full Python 3.13 type hints with Pydantic v2 validation
3. **Metadata Preservation**: Every object has UUID, timestamps, and metadata
4. **Configuration-First**: Single YAML file can orchestrate entire pipeline

## Gallery Examples

Working examples in `gallery/eng/argument_structure/` show production usage:

- `generate_lexicons.py` - Stage 1 with VerbNet, UniMorph, CSV
- `fill_templates.py` - Stage 2 with MLM strategies
- `create_2afc_pairs.py` - Stage 3 with LM scoring
- `generate_lists.py` - Stage 4 with complex constraints
- `generate_deployment.py` - Stage 5 with JATOS export
- `run_pipeline.py` - Stage 6 with active learning

All gallery scripts use the same API documented here.
