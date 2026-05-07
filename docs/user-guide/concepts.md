# Core Concepts

This section introduces the fundamental design principles and architecture underlying bead.

## Stand-off Annotation

Bead uses stand-off annotation to minimize data duplication and maintain provenance tracking throughout the experimental pipeline.

Objects reference each other by UUID rather than embedding full copies. An experimental list stores item UUIDs, not the items themselves. Items store filled template UUIDs, not the template content. This pattern creates a single source of truth for each piece of data.

Example: when you modify a template, all items referencing that template automatically reflect the change (if you regenerate them). You don't need to update copies scattered across different files.

The stand-off approach enables:

- **Provenance tracking**: trace any experimental stimulus back through its complete history
- **Space efficiency**: store each piece of data once, reference it many times
- **Consistency**: modifications update all references automatically
- **Modularity**: swap components (e.g., different templates) without rebuilding everything

When resolving references, pass metadata dictionaries mapping UUIDs to their data:

```python
# Items stored as UUIDs
item_uuids = [uuid1, uuid2, uuid3]

# Metadata for constraint evaluation
item_metadata = {
    uuid1: {"verb": "put", "frame": "transitive"},
    uuid2: {"verb": "place", "frame": "transitive"},
    uuid3: {"verb": "drop", "frame": "transitive"},
}

# Partitioner receives both
partitioner.partition_with_batch_constraints(items=item_uuids, metadata=item_metadata)
```

## BeadBaseModel

All bead data structures inherit from `BeadBaseModel`, which provides three standard fields:

- **id**: UUIDv7 identifier (time-ordered for sortability)
- **created_at**: ISO 8601 timestamp of creation
- **modified_at**: ISO 8601 timestamp of last modification

Additionally, every model includes a `metadata` dictionary for arbitrary key-value pairs. Metadata flows through the entire pipeline: add a field to a lexical item, and it appears in filled templates, items, and experimental lists.

Example metadata usage:

```jsonl
{"id": "uuid", "lemma": "run", "pos": "V", "metadata": {"frequency": 1000, "source": "verbnet"}}
```

The `frequency` field propagates through filling, item construction, and partitioning, enabling constraint-based experimental design.

## 6-Stage Pipeline

Bead implements a linear pipeline with six stages. Data flows forward through these stages, with each stage consuming output from the previous one.

### Stage 1: Resources

Create lexical items and templates with optional constraints.

- **Lexical items**: words or phrases with linguistic features
- **Templates**: sentence patterns with slots for lexical items
- **Constraints**: rules restricting which items can fill which slots

Output: `lexicons/*.jsonl`, `templates/*.jsonl`

### Stage 2: Templates

Fill template slots with lexical items using various strategies.

- **Strategies**: exhaustive, random, stratified, MLM-based, mixed
- **Constraint satisfaction**: respect slot requirements and relational constraints
- **Streaming**: handle large combinatorial spaces efficiently

Output: `filled_templates/*.jsonl`

### Stage 3: Items

Convert filled templates into experimental items with task-specific structure.

- **9 task types**: forced-choice, ordinal scale, binary, categorical, multi-select, magnitude, free text, cloze, span labeling
- **Factory functions**: task-specific constructors with validation
- **Batch creation**: generate items from groups or cross-products

Output: `items/*.jsonl`

### Stage 4: Lists

Partition items into experimental lists satisfying constraints.

- **List constraints**: per-list requirements (uniqueness, balance, quantile, diversity)
- **Batch constraints**: cross-list requirements (coverage, balance, min occurrence)
- **Strategies**: balanced, random, stratified, latin square

Output: `lists/*.jsonl`

### Stage 5: Deployment

Generate jsPsych 8.x experiments for JATOS deployment.

- **Batch mode**: all lists packaged in single experiment
- **Distribution strategies**: 8 strategies for participant-to-list assignment
- **Trial configuration**: timing, UI, response collection
- **Material Design UI**: consistent styling across experiments

Output: `experiment/` directory with HTML/JavaScript/CSS

### Stage 6: Training

Train models on collected data, optionally using active learning.

- **GLMM models**: generalized linear mixed models with random effects
- **Active learning**: uncertainty sampling for efficient data collection
- **Convergence detection**: stop when model matches human agreement
- **Evaluation**: metrics, cross-validation, agreement measures

Output: `models/` directory with trained weights and config

## Task Types vs Judgment Types

Bead distinguishes between **task types** (UI presentation) and **judgment types** (underlying measurement).

**Task types** define the interface participants see:

- `forced_choice`: select one option from N alternatives
- `ordinal_scale`: rate on Likert or slider scale
- `binary`: yes/no, true/false choice
- `categorical`: select from unordered categories
- `multi_select`: select multiple options
- `magnitude`: enter numeric value (reading time, confidence)
- `free_text`: open-ended text response
- `cloze`: fill in blanks
- `span_labeling`: annotate text spans with labels

**Judgment types** describe the measurement goal:

- Acceptability judgments (ordinal scale task)
- Forced-choice comparisons (forced choice task)
- Truth value judgments (binary task)
- Semantic relation classification (categorical task)

The same judgment type may use different task types depending on experimental goals. Acceptability can use ordinal scales (rate sentence naturalness) or forced choice (which sentence is more natural).

## Annotation Protocols

Above the task / judgment distinction sits a separate type-theoretic
layer for *what* a question measures and *how* it is phrased. The
[`bead.protocol`](protocols.md) package factors annotation design
into four roles:

- A `SemanticAnchor` is the *type* of a question: a declarative
  specification of the property being measured, the response space,
  and the structural constraints any phrasing must preserve.
- A `ProtocolContext` is the dependent *index*: everything known
  about the current annotation target, including responses already
  recorded for earlier questions.
- A `RealizationStrategy` is the *computational content* of the
  dependent function `Pi(ctx). Question(ctx)`. Three strategies are
  shipped: a fixed template, a context-conditional template selector,
  and an LM paraphraser.
- A `DriftGuard` is the *type-checker* over realized prompts; it
  composes structural, embedding, and perplexity validators.

`QuestionFamily` packages an anchor with a realization strategy and a
drift guard; `AnnotationProtocol` sequences families into the
iterated dependent product
`Sigma(a_1 : Q_1(ctx)). Sigma(a_2 : Q_2(ctx, a_1)). ...`, threading
each response into the context so later questions can condition on
earlier answers. See the [protocols user guide](protocols.md) for
the full walkthrough.

## Configuration-First Design

Bead orchestrates the entire pipeline from a single YAML configuration file. The config specifies paths, strategies, constraints, and parameters for all six stages.

Benefits:

- **Reproducibility**: config file fully documents experimental design
- **Version control**: track experimental modifications via git
- **Reusability**: apply same design to different linguistic phenomena
- **Clarity**: all decisions visible in one location

The CLI commands read config files to set default parameters, reducing typing and ensuring consistency across pipeline stages.

## Language-Agnostic Principles

Bead works with any language supported by linguistic resources like UniMorph.

Language-specific information lives in:

- **Lexical items**: morphological features (e.g., `{\"tense\": \"past\", \"number\": \"plural\"}`)
- **UniMorph integration**: automatic inflection generation
- **VerbNet/PropBank**: frame information for syntactic structures
- **Configuration**: language code fields throughout

The constraint system uses language-neutral DSL expressions, allowing the same constraint logic across languages. Template slot patterns remain language-independent.

When adapting experiments to new languages:

1. Create lexicons with appropriate morphological features
2. Generate templates matching the language's syntax
3. Use UniMorph for inflection (if available)
4. Adjust trial timing for language reading speed

The core pipeline stages remain identical across languages.

## Next Steps

Now that you understand bead's architecture, explore specific pipeline stages:

- [Resources CLI](cli/resources.md): create lexicons, templates, and constraints
- [Templates CLI](cli/templates.md): fill templates with strategies
- [Items CLI](cli/items.md): construct task-specific items
- [Lists CLI](cli/lists.md): partition items with constraints
- [Deployment CLI](cli/deployment.md): generate jsPsych experiments
- [Training CLI](cli/training.md): train GLMM models

For complete command reference, see the [API documentation](../api/resources.md).
