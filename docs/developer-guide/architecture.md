# Architecture

This document explains the system architecture of bead, including the 6-stage pipeline design, module organization, design principles, and key architectural decisions.

## System Overview

bead implements a 6-stage pipeline for constructing, deploying, and analyzing large-scale linguistic judgment experiments. Each stage transforms data from the previous stage while maintaining complete provenance through UUID-based references.

### The 6-Stage Pipeline

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | `resources/` | Lexical items and templates with constraints |
| 2 | `templates/` | Template filling strategies (exhaustive, stratified, random) |
| 3 | `items/` | Experimental item construction (9 task types) |
| 4 | `lists/` | List partitioning with constraint satisfaction |
| 5 | `deployment/` | jsPsych experiment generation for JATOS |
| 6 | `active_learning/` | Training with human-in-the-loop convergence |

Each stage reads data from the previous stage using UUID references, processes it, adds metadata, and writes new data with its own UUIDs. This creates an unbroken chain of provenance from lexical resources to trained models.

The `bead.protocol` package sits *across* the 6-stage pipeline, not
inside it. Anchors, contexts, realization strategies, and drift
guards together define *what question is being asked* of an
annotator and *how it is phrased*; the resulting prompt strings flow
into Stage 3 item construction, and annotator responses flow back to
Stage 6 training and evaluation. The protocol layer is intentionally
domain-neutral and pipeline-orthogonal so that any annotation domain
can reuse the same anchor / drift / realization machinery.

### Data Flow Example

A typical experiment follows this data flow:

1. **Resources**: Create verbs (UUIDs: v1, v2) and templates (UUIDs: t1, t2)
2. **Filled Templates**: Fill templates with verbs (UUIDs: f1, f2, f3)
   - f1 references [v1, t1], stores slot_fillers in metadata
3. **Items**: Create forced-choice items from filled templates (UUIDs: i1, i2)
   - i1 references [f1, f2], stores model_scores in metadata
4. **Lists**: Partition items into participant lists (UUIDs: l1, l2)
   - l1 references [i1, i3, i5], stores balance_metrics in metadata
5. **Deployment**: Generate jsPsych experiment
   - Resolves all UUID chains to create trial data
   - Packages as .jzip for JATOS deployment
6. **Training**: Collect responses and train models
   - Links responses back to i1 → f1 → v1, t1 for analysis

At every step, objects store only UUID references to their sources, never copying data. This ensures a single source of truth and complete provenance tracking.

## Module Organization

bead consists of 17 top-level modules organized by function:

### Core Pipeline Stages (6 modules)

**bead/resources/** - Stage 1: Lexical items and templates

- `lexical_item.py`: LexicalItem, MultiWordExpression
- `lexicon.py`: Lexicon collection
- `template.py`: Template, Slot, TemplateSequence, TemplateTree
- `template_collection.py`: TemplateCollection
- `constraints.py`: Constraint (DSL expressions)
- `loaders.py`: Resource loading utilities
- `constraint_builders.py`: Constraint creation helpers
- `classification.py`: Resource classification
- `template_generation.py`: Template generation utilities

**bead/templates/** - Stage 2: Template filling

- `filler.py`: TemplateFiller (main engine)
- `strategies.py`: Exhaustive, Random, Stratified strategies
- `resolver.py`: ConstraintResolver
- `streaming.py`: Streaming output for large combinatorics
- `combinatorics.py`: Combinatorial generation utilities

**bead/items/** - Stage 3: Item construction

- `item.py`: Item, UnfilledSlot, ModelOutput, ItemCollection
- `item_template.py`: ItemTemplate, TaskType, JudgmentType, ChunkingSpec
- `forced_choice.py`: create_forced_choice_item() and batch utilities
- `ordinal_scale.py`, `binary.py`, `categorical.py`, `multi_select.py`,
  `magnitude.py`, `free_text.py`, `cloze.py`: Task-type utilities
- `spans.py`: Span annotation data models
- `span_labeling.py`: Span labeling utilities (9th task type)
- `validation.py`: validate_item_for_task_type()
- `constructor.py`: ItemConstructor for item creation
- `generation.py`: Item generation utilities
- `scoring.py`: Item scoring functions
- `adapters/`: Model integrations (HuggingFace, OpenAI, Anthropic, Google, TogetherAI)
- `cache.py`: Content-addressable cache for model outputs

**bead/lists/** - Stage 4: List partitioning

- `experiment_list.py`: ExperimentList (stores item UUIDs)
- `list_collection.py`: ListCollection
- `constraints.py`: ListConstraint (8 types), BatchConstraint (4 types)
- `partitioner.py`: ListPartitioner with constraint satisfaction
- `balancer.py`: QuantileBalancer for stratified sampling
- `stratification.py`: Stratification strategies

**bead/deployment/** - Stage 5: Experiment generation

- `jspsych/generator.py`: JsPsychExperimentGenerator (batch mode only)
- `jspsych/config.py`: ExperimentConfig, ListDistributionStrategy
- `jspsych/trials.py`: Trial generation for jsPsych 8.x
- `jspsych/randomizer.py`: Randomization logic
- `jspsych/ui/`: Material Design UI components
- `jatos/exporter.py`: JATOS .jzip export
- `jatos/api.py`: JATOS API client
- `distribution.py`: 8 list distribution strategies

**bead/active_learning/** - Stage 6: Training and convergence

- `loop.py`: ActiveLearningLoop orchestration
- `selection.py`: UncertaintySampler for item selection
- `strategies.py`: Active learning strategies
- `models/`: Task-specific models (9 types matching items/)
  - `forced_choice.py`: ForcedChoiceModel with GLMM support
  - `base.py`: ActiveLearningModel interface
  - `random_effects.py`: RandomEffectsManager for mixed effects
  - `binary.py`, `categorical.py`, `cloze.py`, `free_text.py`, `magnitude.py`, `multi_select.py`, `ordinal_scale.py`: Task-specific models
  - `lora.py`: LoRA adapter support
- `trainers/`: Training backends
  - `base.py`: Trainer interface
  - `huggingface.py`: HuggingFace Trainer integration
  - `lightning.py`: PyTorch Lightning integration
  - `registry.py`: Trainer registry
- `config.py`: MixedEffectsConfig, RandomEffectsSpec

### Supporting Modules (11 modules)

**bead/data/** - Foundation layer

- `base.py`: BeadBaseModel (UUID, timestamps, metadata)
- `identifiers.py`: generate_uuid() (UUIDv7)
- `timestamps.py`: now_iso8601() (ISO 8601 timestamps)
- `serialization.py`: JSONL read/write utilities
- `validation.py`: Data validation functions
- `metadata.py`: Metadata tracking
- `language_codes.py`: ISO 639 language code handling
- `repository.py`: Data repository pattern

**bead/dsl/** - Constraint DSL (7 files)

- `parser.py`: Lark-based parser for constraint expressions
- `evaluator.py`: DSL evaluation engine
- `stdlib.py`: Built-in functions (membership, comparison, arithmetic)
- `ast.py`: Abstract syntax tree nodes
- `context.py`: Evaluation context
- `errors.py`: DSL-specific exceptions
- `__init__.py`: Module exports

**bead/config/** - Configuration system (18 files)

- `config.py`: ProjectConfig (root configuration)
- `paths.py`: PathsConfig
- `resources.py`: ResourcesConfig
- `template.py`: TemplatesConfig
- `item.py`: ItemsConfig
- `list.py`: ListsConfig
- `deployment.py`: DeploymentConfig
- `active_learning.py`: ActiveLearningConfig
- `simulation.py`: SimulationConfig
- Plus 9 other modules: `defaults.py`, `env.py`, `loader.py`, `logging.py`, `model.py`, `profiles.py`, `serialization.py`, `validation.py`, `__init__.py`

**bead/evaluation/** - Metrics and reporting

- `convergence.py`: ConvergenceDetector (Krippendorff's alpha)
- `interannotator.py`: InterAnnotatorMetrics (Cohen, Fleiss, Krippendorff)
- `reliability.py`: AnnotationRecord, AnnotatorReliability,
  per-annotator Shannon-entropy diagnostics, and
  `low_entropy_annotators` flagger

**bead/protocol/** - Annotation-protocol primitives (cross-cutting layer)

- `anchor.py`: SemanticAnchor (the *type* of a question), ResponseSpace,
  SemanticPoles
- `context.py`: ProtocolContext (the dependent *index*), ContextItem,
  context-predicate registry (`register_context_predicate`,
  `get_context_predicate`, `list_context_predicates`)
- `realization.py`: RealizationStrategy Protocol with three
  implementations (TemplateRealization,
  ContextualTemplateRealization, LMRealization), TemplateVariant,
  LMClient Protocol
- `drift.py`: DriftScore, DriftValidator Protocol, three concrete
  validators (StructuralDriftValidator, EmbeddingDriftValidator,
  PerplexityDriftValidator), DriftGuard composite, plus
  EmbeddingAdapter and PerplexityAdapter Protocols for backends
- `family.py`: QuestionFamily (Pi(ctx). Question(ctx)),
  AnnotationProtocol (the iterated dependent product, with
  `depends_on` graph validation), QuestionRealization
- `encoding.py`: ScaleType (binary / ordinal / nominal),
  ResponseEncoding (likelihood-agnostic, with invariant validators
  for `n_levels == len(labels)`, label uniqueness, and
  BINARY-must-have-2-levels), `encode_response_space` bridge
- `diagnostics.py`: DiagnosticLevel, DiagnosticRecord, DatasetReport
  (immutable; `with_*` mutators), ConditionalObservationValidator
  (drives off `QuestionFamily.depends_on`), RecordLike Protocol

**bead/simulation/** - Simulation framework

- `annotators/`: Simulated annotators
  - `base.py`: Annotator interface
  - `lm_based.py`: Language model annotators
  - `oracle.py`: Oracle (ground truth) annotator
  - `random.py`: Random annotator
  - `distance_based.py`: Distance-based annotators
- `noise_models/`: Noise injection
  - `base.py`: Noise model interface
  - `temperature.py`: Temperature-based noise
  - `random_noise.py`: Random noise injection
  - `systematic.py`: Systematic noise patterns
- `strategies/`: Task-specific simulation strategies (9 types)
  - `base.py`: Strategy interface
  - One strategy per task type: `binary.py`, `categorical.py`, `cloze.py`, `forced_choice.py`, `free_text.py`, `magnitude.py`, `multi_select.py`, `ordinal_scale.py`
- `dsl_extension/`: DSL extensions for simulation
- `runner.py`: Simulation orchestration

**bead/data_collection/** - Data retrieval

- `jatos.py`: JATOS API client for downloading results
- `prolific.py`: Prolific metadata integration
- `merger.py`: Merge JATOS results with Prolific metadata

**bead/adapters/** - External resources

- `huggingface.py`: HuggingFace model integration

**bead/cli/** - Command-line interface

- `main.py`: Click root command (entry point)
- `resources.py`: Resource commands
- `templates.py`: Template commands
- `items.py`: Item commands (task-specific create-* commands)
- `lists.py`: List commands
- `deployment.py`: Deployment commands
- `deployment_ui.py`: UI customization commands
- `deployment_trials.py`: Trial generation commands
- `training.py`: Training commands (collect-data, evaluate, etc.)
- `models.py`: Model commands (train-model, predict, predict-proba)
- `active_learning.py`: Active learning commands
- `simulate.py`: Simulation commands
- `workflow.py`: Workflow commands (run, init, status, resume, rollback)
- `config.py`: Configuration commands
- `display.py`: Display utilities for rich output
- `items_factories.py`: Item factory utilities
- `list_constraints.py`: List constraint utilities
- `constraint_builders.py`: Constraint builder utilities
- `resource_loaders.py`: Resource loading utilities
- `completion.py`: Shell completion
- `utils.py`: CLI utilities

**bead/behavioral/** - Behavioral analytics

- `analytics.py`: JudgmentAnalytics and aggregation
- `extraction.py`: Extract behavioral measures from experiment responses
- `merging.py`: Merge behavioral data across participants and sessions

**bead/participants/** - Participant metadata

- `models.py`: Participant, ParticipantIDMapping models
- `collection.py`: ParticipantCollection management
- `merging.py`: Merge participant data from multiple sources
- `metadata_spec.py`: ParticipantMetadataSpec and FieldSpec validation

**bead/tokenization/** - Multilingual tokenization

- `tokenizers.py`: WhitespaceTokenizer, SpacyTokenizer, StanzaTokenizer
- `config.py`: TokenizerConfig, TokenizerBackend
- `alignment.py`: Token-to-character alignment utilities

## Design Principles

bead follows five core design principles that guide all architectural decisions:

### 1. Stand-off Annotation

Objects reference each other by UUID rather than embedding full copies. This maintains a single source of truth and enables complete provenance tracking.

**Example**: An Item stores filled_template_refs (list of UUIDs), not filled_templates (list of FilledTemplate objects). To resolve references, use separate metadata dictionaries:

```python
# CORRECT: UUID references
item = Item(
    filled_template_refs=[uuid1, uuid2], judgment_type="forced_choice"  # Just UUIDs
)

# Resolve references using metadata dict
template1 = templates_dict[uuid1]  # Look up by UUID

# INCORRECT: Embedding objects
item = Item(filled_templates=[template_obj1, template_obj2])  # Wrong!
```

This pattern applies throughout:
- FilledTemplate references LexicalItem UUIDs (via slot_fillers metadata)
- Item references FilledTemplate UUIDs (filled_template_refs)
- ExperimentList references Item UUIDs (item_refs)
- JATOS results reference Item UUIDs (maintains provenance)

**Rationale**: Stand-off annotation prevents data duplication, reduces memory usage, simplifies updates, and maintains complete provenance chains. Changing a template definition updates all items that reference it automatically.

### 2. Metadata Preservation

Every BeadBaseModel tracks complete metadata for provenance and processing history. All models inherit from BeadBaseModel:

```python
class BeadBaseModel(BaseModel):
    id: UUID = Field(default_factory=generate_uuid)  # UUIDv7 (time-ordered)
    created_at: datetime = Field(default_factory=now_iso8601)  # ISO 8601
    modified_at: datetime = Field(default_factory=now_iso8601)
    version: str = Field(default="1.0.0")  # Schema version
    metadata: dict[str, Any] = Field(default_factory=dict)  # Arbitrary key-value
```

Metadata accumulates through the pipeline:

**Stage 1 (resources)**: Lexical features, source information
```python
lexical_item.metadata = {"source": "verbnet", "frame": "run-51.3.2"}
```

**Stage 2 (templates)**: Slot fillers, constraint satisfaction
```python
filled_template.metadata = {
    "slot_fillers": {"verb": uuid1, "noun": uuid2},
    "constraints_satisfied": [constraint_uuid1],
}
```

**Stage 3 (items)**: Model scores, embeddings
```python
item.metadata = {
    "lm_score_diff": 2.3,
    "model_outputs": [...],
    "pair_type": "minimal_pair",
}
```

**Stage 4 (lists)**: Balance metrics, constraint violations
```python
experiment_list.metadata = {
    "balance_metrics": {"verb_diversity": 0.85},
    "quantile_distribution": [10, 10, 10, ...],
}
```

**Rationale**: Metadata tracking enables reproducibility, debugging, and post-hoc analysis. Every decision made by the pipeline is recorded.

### 3. Type Safety

bead uses full Python 3.13 type hints with Pydantic v2 validation. No `Any` or `object` types appear in core code (only in adapters for external APIs with dynamic types).

**Type annotations**:
```python
def partition_with_batch_constraints(
    self,
    items: list[UUID],  # Explicit: list of UUIDs
    n_lists: int,
    metadata: dict[UUID, dict[str, Any]],  # Explicit: UUID → metadata
    batch_constraints: list[BatchConstraint],
    max_iterations: int = 100,
) -> list[ExperimentList]: ...
```

**Pydantic validation**:
```python
class ExperimentList(BeadBaseModel):
    name: str
    list_number: int  # Validated >= 0
    item_refs: list[UUID] = Field(default_factory=list)  # Type-safe UUID list

    @field_validator("list_number")
    @classmethod
    def validate_list_number(cls, v: int) -> int:
        if v < 0:
            raise ValueError("list_number must be non-negative")
        return v
```

**Pyright configuration** (strict mode):
```toml
[tool.pyright]
typeCheckingMode = "strict"
pythonVersion = "3.13"
exclude = [
    "tests/**",  # Tests don't require full type checking
    "bead/items/adapters/**",  # External APIs have dynamic types
    "bead/templates/adapters/**",
    "bead/resources/adapters/**"
]
```

**Rationale**: Type safety catches errors at development time, enables better IDE support, and serves as documentation. Pydantic validation ensures data integrity at runtime.

### 4. Configuration-First

A single YAML file orchestrates the entire pipeline. All pipeline parameters are specified in config.yaml rather than hardcoded.

**Example configuration** (gallery/eng/argument_structure/config.yaml):
```yaml
project:
  name: "argument_structure"
  language_code: "eng"

resources:
  lexicons:
    - path: "lexicons/verbnet_verbs.jsonl"
  templates:
    - path: "templates/generic_frames.jsonl"

template:
  filling_strategy: "exhaustive"

items:
  judgment_type: "forced_choice"
  n_alternatives: 2

lists:
  n_lists: 8
  strategy: "quantile_balanced_minimal_pairs"
  batch_constraints:
    - type: "coverage"
      property_expression: "item['template_id']"
      target_values: [0, 1, 2, 3, 4, 5]

deployment:
  platform: "jatos"
  distribution_strategy:
    strategy_type: "balanced"
```

Configuration models in bead/config/ validate all settings using Pydantic. The CLI reads config.yaml and passes validated configuration objects to each stage.

**Rationale**: Configuration-first design enables reproducibility, parameter sweeps, and easy sharing of experimental setups. Researchers can share config.yaml files as complete pipeline specifications.

### 5. Language-Agnostic

bead uses ISO 639 language codes and avoids English-specific assumptions. All linguistic resources specify language_code explicitly.

**Example**:
```python
# English lexical item
verb_en = LexicalItem(
    lemma="walk", language_code="eng", features={"pos": "VERB"}  # ISO 639-3
)

# Korean lexical item
verb_ko = LexicalItem(
    lemma="걷다", language_code="kor", features={"pos": "VERB"}  # ISO 639-3
)

# Template with language code
template_en = Template(
    template_string="{subject} {verb} {object}.", language_code="eng"
)

template_ko = Template(
    template_string="{subject} {object} {verb}.", language_code="kor"  # SOV word order
)
```

Language codes use langcodes package for validation and normalization. Constraint expressions work with any language. Template filling strategies (exhaustive, MLM) are language-agnostic.

**Rationale**: Language-agnostic design enables cross-linguistic research and reduces maintenance burden. The same pipeline works for any language with appropriate resources.

## Key Architectural Decisions

This section documents major architectural decisions and their rationale.

### No models.py Files

**Decision**: Eliminate monolithic models.py files in favor of focused, co-located modules.

**Before refactoring**:
```
bead/lists/
├── models.py           # ExperimentList, ListCollection, ListPartitioner
├── constraints.py
└── strategies.py
```

**After refactoring**:
```
bead/lists/
├── experiment_list.py       # ExperimentList model + operations
├── list_collection.py       # ListCollection model + operations
├── partitioner.py          # ListPartitioner + partitioning logic
├── constraints.py          # Constraint models + evaluation
├── balancer.py            # Balancing logic
└── stratification.py      # Stratification strategies
```

**Rationale**: Monolithic models.py files violate single responsibility principle. Co-locating models with their operations improves discoverability and reduces coupling. When adding a new constraint type, you edit bead/lists/constraints.py, not a generic models.py file.

**Rule**: When creating new functionality, create semantically meaningful modules rather than adding to models.py. Examples: experiment_list.py (not lists_models.py), item_template.py (not item_metadata.py).

### Stand-off Annotation with UUID References

**Decision**: Objects store UUID references to other objects, never embed full copies.

**Implementation**:
```python
class Item(BeadBaseModel):
    filled_template_refs: list[UUID] = Field(default_factory=list)  # UUIDs only
    # NOT: filled_templates: list[FilledTemplate]


class ExperimentList(BeadBaseModel):
    item_refs: list[UUID] = Field(default_factory=list)  # UUIDs only
    # NOT: items: list[Item]
```

To resolve references, use separate metadata dictionaries:
```python
partitioner.partition_with_batch_constraints(
    items=item_uuids, metadata=item_metadata  # list[UUID]  # dict[UUID, dict[str, Any]]
)
```

**Rationale**:
1. **Single source of truth**: Updating a template definition affects all items that reference it
2. **Reduced memory**: Storing UUIDs (16 bytes) vs full objects (kilobytes)
3. **Simplified serialization**: JSONL files store UUIDs as strings
4. **Provenance tracking**: UUID chains provide complete lineage
5. **Lazy loading**: Load only needed objects, not entire dependency graphs

**Trade-off**: Requires metadata dictionaries for constraint evaluation. Accepting this complexity in exchange for correctness and efficiency.

### Task-Type Utilities Pattern

**Decision**: Provide 9 task-type-specific modules with consistent API for item creation.

**Task types**: forced_choice, ordinal_scale, binary, categorical, multi_select, magnitude, free_text, cloze, span_labeling

**API pattern** (consistent across all 9 types):
```python
# Core creation function
def create_forced_choice_item(
    *alternatives: str, metadata: dict[str, Any] | None = None
) -> Item: ...


# Batch creation from texts
def create_forced_choice_items_from_texts(
    texts: list[str],
    n_alternatives: int,
    metadata_fn: Callable[[str], dict[str, Any]] | None = None,
) -> list[Item]: ...


# Batch creation from groups
def create_forced_choice_items_from_groups(
    source_items: list[Item],
    group_by: Callable[[Item], Any],
    n_alternatives: int,
    item_filter: Callable[[Item], bool] | None = None,
) -> list[Item]: ...


# Filtered creation
def create_filtered_forced_choice_items(
    source_items: list[Item],
    group_by: Callable[[Item], Any],
    n_alternatives: int,
    item_filter: Callable[[Item], bool] | None = None,
    group_filter: Callable[[Any, list[Item]], bool] | None = None,
) -> list[Item]: ...
```

**Validation**:
```python
from bead.items.validation import validate_item_for_task_type

validate_item_for_task_type(item, "forced_choice")  # Raises ValueError if invalid
```

**Rationale**:
1. **Correctness**: Type-specific utilities enforce correct structure (e.g., forced_choice requires n_alternatives metadata)
2. **Discoverability**: IDE autocomplete shows create_forced_choice_item() in bead.items.forced_choice
3. **Consistency**: All 9 task types follow same API pattern
4. **Future expansion**: Adding task type 9 follows established pattern

**Comparison**:
```python
# Direct Item() constructor (manual metadata):
item = Item(
    rendered_elements={"option_a": "A", "option_b": "B"}, item_metadata={"n_options": 2}
)

# Task-type utility (automatic metadata):
from bead.items.forced_choice import create_forced_choice_item

item = create_forced_choice_item("A", "B")  # n_options added automatically
```

### GLMM Integration with 3 Mixed-Effects Modes

**Decision**: Support Generalized Linear Mixed Models with 3 modes for participant and item random effects.

**Modes**:

**1. Fixed Effects Only** (default):
```python
config = ForcedChoiceModelConfig(
    model_name="bert-base-uncased", mixed_effects_mode="fixed_only"
)
model.train(items, labels, participant_ids=["p1", "p1", "p2", "p2"])
```

**2. Random Effects Only**:
```python
config = ForcedChoiceModelConfig(
    mixed_effects_mode="random_only",
    mixed_effects_config=MixedEffectsConfig(
        random_effects_spec=RandomEffectsSpec(
            participant_intercept=True,  # Participant-level random intercepts
            item_intercept=True,  # Item-level random intercepts
            interaction=False,
        )
    ),
)
```

**3. Mixed Effects** (fixed + random):
```python
config = ForcedChoiceModelConfig(
    mixed_effects_mode="mixed",
    mixed_effects_config=MixedEffectsConfig(
        random_effects_spec=RandomEffectsSpec(
            participant_intercept=True,
            item_intercept=True,
            interaction=True,  # Participant × item interactions
        )
    ),
)
```

**RandomEffectsManager** handles:
- Participant intercepts: Baseline differences between participants
- Item intercepts: Difficulty differences between items
- Interaction terms: Participant × item interactions
- Variance component estimation

**Critical requirement**: All model methods (train, predict, predict_proba) require participant_ids parameter:
```python
model.train(items, labels, participant_ids=participant_ids)
predictions = model.predict(items, participant_ids=participant_ids)
```

**Rationale**:
1. **Statistical validity**: Account for non-independence in repeated measures
2. **Generalizability**: Mixed effects models generalize to new participants and items
3. **Flexibility**: Three modes support different research designs
4. **Active learning**: Uncertainty estimates account for random effects
5. **Research alignment**: Standard approach in psycholinguistics

### Batch-Only Deployment Architecture

**Decision**: All deployment generates unified batch experiments with server-side list distribution. No single-list mode.

**Architecture**:
```python
# Batch mode (required)
config = ExperimentConfig(
    distribution_strategy=ListDistributionStrategy(
        strategy_type=DistributionStrategyType.BALANCED
    )
)

generator = JsPsychExperimentGenerator(config=config, output_dir=Path("output"))

# Requires lists (plural), no single-list mode
output_dir = generator.generate(
    lists=[list1, list2, list3, ...],  # Required, must be non-empty
    items=items_dict,
    templates=templates_dict,
)
```

**8 distribution strategies**:
1. RANDOM: Random selection
2. SEQUENTIAL: Round-robin (0, 1, 2, ..., N, 0, 1, ...)
3. BALANCED: Assign to least-used list
4. LATIN_SQUARE: Counterbalancing with Bradley's algorithm
5. WEIGHTED_RANDOM: Non-uniform probabilities
6. STRATIFIED: Balance across metadata factors
7. QUOTA_BASED: Fixed quota per list
8. METADATA_BASED: Filter and rank by metadata

**Generated file structure**:
```
output_dir/
├── index.html
├── js/
│   ├── experiment.js
│   └── list_distributor.js  # Client-side assignment via JATOS batch sessions
├── css/experiment.css
└── data/
    ├── config.json
    ├── lists.jsonl           # All lists in JSONL format
    ├── items.jsonl           # All items in JSONL format
    └── distribution.json     # Strategy configuration
```

**JATOS integration**:
- Uses jatos.batchSession for server-side state
- JavaScript ListDistributor handles assignment on load
- Lock mechanism prevents race conditions
- Each participant assigned exactly one list

**Rationale**:
1. **No fallbacks**: Explicit distribution_strategy required (no default)
2. **Simplified deployment**: Single experiment package for all lists
3. **Server-side control**: JATOS batch sessions manage assignment
4. **Participant isolation**: Participants never see other lists
5. **Research validity**: Proper counterbalancing and quota management

**Design requirement**: All experiments must specify lists (plural) and distribution_strategy. No single-list mode exists.

### 12-Type Constraint System with DSL

**Decision**: Provide 12 constraint types (8 list + 4 batch) with Domain-Specific Language (DSL) for expressions.

**List constraints** (apply to individual lists):

1. **UniquenessConstraint**: No duplicate property values in list
```python
UniquenessConstraint(property_expression="item['verb_lemma']")
```

2. **CountConstraint**: Exact count of items matching condition
```python
CountConstraint(
    filter_expression="item['pair_type'] == 'minimal_pair'", target_count=50
)
```

3. **ProportionConstraint**: Target distribution of property values
```python
ProportionConstraint(
    property_expression="item['pair_type']",
    target_distribution={"minimal_pair": 0.5, "control": 0.5},
    tolerance=0.05,
)
```

4. **DiversityConstraint**: Minimum unique values required
5. **RangeConstraint**: Property values within range
6. **ExclusionConstraint**: Exclude items matching condition
7. **DependencyConstraint**: Conditional requirements
8. **GroupedQuantileConstraint**: Stratified sampling by group

**Batch constraints** (apply across all lists):

1. **BatchCoverageConstraint**: Ensure all target values appear
```python
BatchCoverageConstraint(
    property_expression="item['template_id']",
    target_values=list(range(26)),  # All 26 templates
    min_coverage=1.0,  # 100% coverage required
)
```

2. **BatchBalanceConstraint**: Maintain global distribution
```python
BatchBalanceConstraint(
    property_expression="item['pair_type']",
    target_distribution={"minimal_pair": 0.5, "control": 0.5},
    tolerance=0.05,
)
```

3. **BatchDiversityConstraint**: Spread values across lists
```python
BatchDiversityConstraint(
    property_expression="item['verb_lemma']",
    max_lists_per_value=4,  # Each verb in ≤4 lists
)
```

4. **BatchMinOccurrenceConstraint**: Minimum occurrence guarantees

**DSL syntax**:
```python
# Membership test
"item['verb_lemma'] in ['walk', 'run', 'jump']"

# Comparison
"item['lm_score_diff'] > 2.0"

# Boolean operators
"item['pair_type'] == 'minimal_pair' and item['quantile'] >= 5"

# Attribute access
"item.metadata.get('valid', True)"

# Function calls
"len(item['verb_lemma']) > 4"
```

**Constraint evaluation**:
- ListPartitioner evaluates constraints during partitioning
- Uses metadata dictionaries for property access
- Iterative refinement to satisfy constraints
- Priority-weighted satisfaction when conflicts occur

**Rationale**:
1. **Expressiveness**: DSL allows complex constraints without code
2. **Separation**: List vs batch constraints address different requirements
3. **Flexibility**: 12 types cover most experimental designs
4. **Safety**: DSL evaluation sandboxed (no arbitrary code execution)
5. **Composability**: Multiple constraints combine via priority weights

### Content-Addressable Caching

**Decision**: Cache model outputs using content-addressable keys (hash of inputs).

**Implementation**:
```python
# Cache key: hash(model_name, input_text, generation_params)
cache_key = hashlib.sha256(
    f"{model_name}:{input_text}:{json.dumps(params, sort_keys=True)}".encode()
).hexdigest()

# Cache directory: .cache/bead/model_outputs/{model_name}/
cache_path = cache_dir / model_name / cache_key[:2] / f"{cache_key}.json"
```

**Benefits**:
- **Deterministic**: Same inputs always produce same cache key
- **Efficient**: O(1) lookup by hash
- **Shareable**: Researchers can share cache directories
- **Versioned**: Model name in cache key ensures isolation
- **Distributed**: Two-level directory structure (cache_key[:2]/cache_key.json) handles millions of files

**Rationale**: Model inference is expensive (minutes to hours for large experiments). Content-addressable caching enables:
1. Incremental development (add items without recomputing existing scores)
2. Parameter sweeps (reuse scores across configurations)
3. Reproducibility (share cache with config.yaml for exact replication)
4. Cost savings (avoid redundant API calls to OpenAI/Anthropic)

## Module Dependencies

Understanding module dependencies helps navigate the codebase and avoid circular imports.

### Dependency Layers

```
Layer 1 (Foundation):
    bead/data/         → No internal dependencies

Layer 2 (Core Resources):
    bead/dsl/          → bead/data/
    bead/resources/    → bead/data/, bead/dsl/
    bead/config/       → bead/data/

Layer 3 (Pipeline Stage 2-3):
    bead/templates/    → bead/data/, bead/resources/, bead/config/
    bead/items/        → bead/data/, bead/templates/, bead/config/

Layer 4 (Pipeline Stage 4-5):
    bead/lists/        → bead/data/, bead/items/, bead/config/
    bead/deployment/   → bead/data/, bead/items/, bead/lists/, bead/config/

Layer 5 (Pipeline Stage 6):
    bead/active_learning/ → bead/items/, bead/lists/, bead/evaluation/, bead/config/
    bead/evaluation/      → bead/data/, bead/items/

Layer 6 (External Integrations):
    bead/adapters/        → bead/resources/
    bead/data_collection/ → bead/items/, bead/lists/
    bead/simulation/      → bead/items/, bead/active_learning/

Layer 7 (Interface):
    bead/cli/          → All modules
```

**Rule**: Higher layers can import from lower layers, but not vice versa. This prevents circular dependencies.

### Critical Import Paths

**Creating items from templates**:
```python
from bead.templates.filler import TemplateFiller
from bead.items.forced_choice import create_forced_choice_items_from_groups

# templates/ → resources/ → data/
# items/ → templates/ → resources/ → data/
```

**Partitioning items into lists**:
```python
from bead.lists.partitioner import ListPartitioner
from bead.lists.constraints import BatchCoverageConstraint

# lists/ → items/ → templates/ → resources/ → data/
```

**Deploying experiments**:
```python
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.deployment.distribution import ListDistributionStrategy

# deployment/ → lists/ → items/ → ... → data/
```

**Training with active learning**:
```python
from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.models.forced_choice import ForcedChoiceModel

# active_learning/ → lists/ → items/ → ... → data/
```

### Avoiding Circular Dependencies

**Problem**: items/ needs templates/ for FilledTemplate, templates/ needs items/ for item construction.

**Solution**: Stand-off annotation breaks circular dependency:
```python
# items/item.py
class Item(BeadBaseModel):
    filled_template_refs: list[UUID]  # UUID references, not FilledTemplate objects


# templates/filler.py
from bead.resources import Template  # No import from items/
```

Items reference filled templates by UUID, not by importing FilledTemplate. Constraint evaluation receives metadata dictionaries, not full objects.

## Summary

bead's architecture prioritizes:

1. **Provenance**: UUID-based stand-off annotation creates unbroken provenance chains
2. **Modularity**: 17 modules organized by function, 6 pipeline stages
3. **Type Safety**: Full Python 3.13 type hints with Pydantic v2 validation
4. **Flexibility**: Configuration-first design, 9 task types, 12 constraint types
5. **Research Validity**: GLMM support, batch deployment, convergence detection

Key architectural decisions (no models.py, stand-off annotation, task-type utilities, GLMM modes, batch-only deployment, 12-type constraints, content-addressable caching) reflect lessons learned from production linguistic research workflows.

Understanding this architecture enables effective contribution to the codebase. For specific contribution patterns, see [Contributing Guide](contributing.md). For development environment setup, see [Setup Guide](setup.md). For testing guidelines, see [Testing Guide](testing.md).
