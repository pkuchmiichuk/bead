# Complete API Workflows

This guide shows complete end-to-end workflows using the Python API, from lexicon creation to deployment.

## Overview

The complete pipeline has 6 stages:

1. **Resources**: Create lexicons and templates
2. **Templates**: Fill templates with lexical items
3. **Items**: Create experimental items from filled templates
4. **Lists**: Partition items into balanced experimental lists
5. **Deployment**: Generate jsPsych/JATOS experiments
6. **Training**: Active learning with human judgments (optional)

## Stage 1: Resources

Create lexicons from CSV files:

```python
from bead.items.item_template import ScaleBounds, ScalePointLabel  # noqa
from pathlib import Path

from bead.resources.lexicon import Lexicon

# Load existing lexicon from fixtures
nouns = Lexicon.from_jsonl(Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns")

print(f"Loaded {len(nouns.items)} nouns")
```

## Stage 2: Templates

Fill templates using strategies:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.templates.filler import FilledTemplate

# Load pre-generated filled templates from fixtures
filled = read_jsonlines(
    Path("filled_templates/generic_frames_filled.jsonl"),
    FilledTemplate,
)

print(f"Loaded {len(filled)} filled templates")

# Example of configuring a MixedFillingStrategy (for reference):
# from bead.templates.strategies import MixedFillingStrategy
# slot_strategies = {
#     "noun": ("exhaustive", {}),
#     "verb": ("exhaustive", {}),
#     "adjective": ("mlm", {"beam_size": 5, "top_k": 10}),
# }
# strategy = MixedFillingStrategy(slot_strategies=slot_strategies)
```

## Stage 3: Items

Create forced choice items from filled templates:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.forced_choice import create_forced_choice_items_from_groups
from bead.items.item import Item
from bead.items.scoring import LanguageModelScorer

# Load existing items from fixtures
source_items = read_jsonlines(
    Path("items/cross_product_items.jsonl"),
    Item,
)

print(f"Loaded {len(source_items)} source items")

# Score with language model (score first 10 for speed)
scorer = LanguageModelScorer(
    model_name="gpt2",
    cache_dir=Path(".cache/scoring"),
    device="cpu",
    text_key="template_string",
)
items_to_score = source_items[:10]
scores = scorer.score_batch(items_to_score)

# Add scores to metadata
for item, score in zip(items_to_score, scores, strict=True):
    item.item_metadata["lm_score"] = score

print(f"Scored {len(items_to_score)} items")

# Create 2AFC items grouped by verb
items = create_forced_choice_items_from_groups(
    items=items_to_score,
    group_by=lambda item: item.item_metadata["verb_lemma"],
    n_alternatives=2,
    extract_text=lambda item: item.rendered_elements.get("template_string", ""),
)

print(f"Created {len(items)} 2AFC items")
```

## Stage 4: Lists

Partition items into experimental lists:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.lists import DiversityConstraint, UniquenessConstraint
from bead.lists.constraints import BatchCoverageConstraint
from bead.lists.partitioner import ListPartitioner

# Load items from fixtures
items = read_jsonlines(Path("items/2afc_pairs.jsonl"), Item)

print(f"Loaded {len(items)} items")

# Build metadata dict
metadata = {item.id: {"metadata": dict(item.item_metadata)} for item in items}

# Define constraints
list_constraints = [
    UniquenessConstraint(
        constraint_type="uniqueness", property_expression="item.metadata.group_key"
    ),
    DiversityConstraint(
        constraint_type="diversity",
        property_expression="item.metadata.template_id",
        min_unique_values=5,
    ),
]

batch_constraints = [
    BatchCoverageConstraint(
        constraint_type="coverage",
        property_expression="item.metadata.template_id",
        target_values=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        min_coverage=1.0,
    ),
]

# Partition
partitioner = ListPartitioner(random_seed=42)
lists = partitioner.partition_with_batch_constraints(
    items=[item.id for item in items],
    n_lists=10,
    list_constraints=list_constraints,
    batch_constraints=batch_constraints,
    metadata=metadata,
)

print(f"Created {len(lists)} lists")
```

## Stage 5: Deployment

Generate jsPsych experiment:

```python
from bead.items.item_template import ScaleBounds
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import ExperimentConfig
from bead.deployment.jspsych.config import InstructionsConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec
from bead.lists import ExperimentList

# Load lists and items from fixtures
lists = read_jsonlines(Path("lists/experiment_lists.jsonl"), ExperimentList)
items = read_jsonlines(Path("items/2afc_pairs.jsonl"), Item)

print(f"Loaded {len(lists)} lists and {len(items)} items")

# Create item template
template = ItemTemplate(
    name="likert_rating",
    description="7-point acceptability",
    judgment_type="acceptability",
    task_type="ordinal_scale",
    task_spec=TaskSpec(
        prompt="How natural does this sentence sound?",
        scale_bounds=ScaleBounds(min=1, max=7),
    ),
    presentation_spec=PresentationSpec(mode="static"),
)

# Link items to template
items_dict = {item.id: item for item in items}
items_dict = {
    item.id: item.with_(item_template_id=template.id) for item in items_dict.values()
}

# Create experiment config
config = ExperimentConfig(
    experiment_type="likert_rating",
    title="Sentence Acceptability Study",
    description="Rate sentence acceptability",
    instructions=InstructionsConfig.from_text("Rate how natural each sentence sounds"),
    randomize_trial_order=True,
    show_progress_bar=True,
    distribution_strategy=ListDistributionStrategy(
        strategy_type=DistributionStrategyType.BALANCED
    ),
    use_jatos=True,
)

# Generate experiment
generator = JsPsychExperimentGenerator(
    config=config,
    output_dir=Path("/tmp/deployment/experiment"),
)

output_dir = generator.generate(
    lists=lists,
    items=items_dict,
    templates={template.id: template},
)

print(f"Experiment generated in {output_dir}")

# Export to JATOS
exporter = JATOSExporter(
    study_title="Acceptability Study",
    study_description="Likert-scale acceptability judgments",
)

exporter.export(
    experiment_dir=output_dir,
    output_path=Path("/tmp/deployment/study.jzip"),
)

print("JATOS package ready")
```

## Stage 6: Training (Optional)

Run active learning loop:

```python
from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.selection import UncertaintySampler
from bead.config.active_learning import (
    ActiveLearningLoopConfig,
    UncertaintySamplerConfig,
)
from bead.evaluation.convergence import ConvergenceDetector

# Create convergence detector
detector = ConvergenceDetector(
    human_agreement_metric="krippendorff_alpha",
    convergence_threshold=0.05,
    min_iterations=3,
)

# Create selector
selector_config = UncertaintySamplerConfig(method="entropy")
selector = UncertaintySampler(config=selector_config)

# Create loop config
loop_config = ActiveLearningLoopConfig(
    max_iterations=10,
    budget_per_iteration=25,
)

# Create loop
loop = ActiveLearningLoop(
    item_selector=selector,
    config=loop_config,
)

print("Active learning loop initialized")
```

## Minimal Complete Script

Here's a minimal workflow in a single script:

```python
#!/usr/bin/env python3
from pathlib import Path

# Stage 1: Resources
from bead.resources.lexicon import Lexicon

nouns = Lexicon.from_jsonl(Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns")

print(f"Loaded {len(nouns.items)} nouns")

# Stage 2: Templates (load pre-generated)
from bead.data.serialization import read_jsonlines
from bead.templates.filler import FilledTemplate

filled = read_jsonlines(
    Path("filled_templates/generic_frames_filled.jsonl"),
    FilledTemplate,
)[:20]

print(f"Loaded {len(filled)} filled templates")

# Stage 3: Items
from bead.items.forced_choice import create_forced_choice_item

items = []
texts = [ft.rendered_text for ft in filled]
for i in range(0, len(texts) - 1, 2):
    item = create_forced_choice_item(texts[i], texts[i + 1])
    items.append(item)

print(f"Created {len(items)} items")

# Stage 4: Lists
from bead.lists.partitioner import ListPartitioner

partitioner = ListPartitioner(random_seed=42)
metadata = {item.id: {"metadata": dict(item.item_metadata)} for item in items}
lists = partitioner.partition(
    items=[item.id for item in items],
    n_lists=2,
    strategy="balanced",
    metadata=metadata,
)

print(f"Created {len(lists)} lists")

# Stage 5: Deployment
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jspsych.config import ExperimentConfig, InstructionsConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec

template = ItemTemplate(
    name="fc",
    description="2AFC",
    judgment_type="acceptability",
    task_type="forced_choice",
    task_spec=TaskSpec(prompt="Which is more natural?", options=["A", "B"]),
    presentation_spec=PresentationSpec(mode="static"),
)

items_dict = {item.id: item for item in items}
items_dict = {
    item.id: item.with_(item_template_id=template.id) for item in items_dict.values()
}

config = ExperimentConfig(
    experiment_type="forced_choice",
    title="Acceptability Study",
    description="Rate sentences",
    instructions=InstructionsConfig.from_text("Select the more natural sentence"),
    distribution_strategy=ListDistributionStrategy(
        strategy_type=DistributionStrategyType.BALANCED
    ),
)

generator = JsPsychExperimentGenerator(
    config=config, output_dir=Path("/tmp/deployment")
)
output_dir = generator.generate(
    lists=lists, items=items_dict, templates={template.id: template}
)

print(f"Complete! Experiment generated in {output_dir}")
```

## CLI vs API Comparison

The same workflow using the CLI:

```bash
# Stage 1: Resources
uv run bead resources from-csv resources/nouns.csv --output lexicons/nouns.jsonl

# Stage 2: Templates
uv run bead templates fill templates.jsonl lexicons/*.jsonl output/filled.jsonl \\
  --strategy exhaustive

# Stage 3: Items
uv run bead items create-forced-choice output/filled.jsonl \\
  --output output/items.jsonl --n-alternatives 2

# Stage 4: Lists
uv run bead lists create output/items.jsonl --output output/lists.jsonl \\
  --n-lists 10 --strategy balanced

# Stage 5: Deployment
uv run bead deployment generate output/lists.jsonl output/items.jsonl \\
  --output deployment/ --platform jatos
```

**When to use which approach**:

- **CLI**: Simple workflows, single-file operations, shell scripting
- **API**: Batch operations, complex logic, custom plugins, integration

## Next Steps

- See individual module documentation for detailed API reference
- Explore [gallery/eng/argument_structure](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/) for production example
- Read [CLI workflows](../cli/workflows.md) for configuration-driven approach
