# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### `bead.corpus` — streaming corpus ingestion and structural sampling

- New subpackage `bead.corpus` for turning raw text corpora into experimental
  `Item`s. `CorpusRecord` carries text plus flat provenance; `CorpusSource` is
  a streaming-source protocol.
- Sources: `JsonlCorpusSource` (JSON Lines, transparently decompressing
  Zstandard `.zst` files), `CsvCorpusSource` (CSV/TSV), and
  `CompletionCorpusSource` (a language model as a corpus source, via the new
  `TextGenerator` protocol on the OpenAI and Anthropic adapters).
- Lazy pipeline: `parse_records`, `filter_by_structure`, `sample_corpus`, and
  `record_to_item` stream records through a dependency parser and keep only
  those whose parse satisfies a structural DSL constraint, producing `Item`s
  with standoff parse annotations and source provenance. The pipeline never
  loads the full corpus into memory.
- New `corpus` optional-dependency extra (`zstandard`).

#### Dependency parsing in `bead.tokenization`

- New `bead.tokenization.parsers`: `SpacyParser`, `StanzaParser`, and
  `create_parser` produce a per-sentence `ParsedSentence` of `ParsedToken`
  records (token, lemma, upos, xpos, head, deprel, morphology, offsets).
- `parse_to_spans` projects a dependency parse onto the standoff `Span` +
  `SpanRelation` models: one single-token span per token (with its governor as
  `head_index` and its features in `span_metadata`) and one directed
  head-to-dependent relation per syntactic arc.

#### Structural-query builtins in the constraint DSL

- New `bead.dsl` standard-library functions query a dependency parse stored on
  an `Item`: `upos`, `xpos`, `lemma_of`, `form_of`, `deprel`, `morph`, `head`,
  `dependents`, `has_relation`, `root`, `subtree`, `path_to_root`,
  `tokens_with_upos`, `tokens_with_deprel`, `any_deprel`, and `filter_upos`.
  Constraints can now match syntactic structure, e.g.
  `upos(self, root(self)) == "VERB" and len(dependents(self, root(self), "obj")) > 0`.

#### Text transforms for corpus cleanup

- New transforms in `bead.transforms.text`: `MarkdownStripTransform`,
  `RedditCleanupTransform`, and the `split_sentences` helper (parser-backed or
  regex fallback). The first two are registered in the default transform
  registry.

### Changed

- Minimum `didactic` raised to `>=0.7.2` and `panproto` to `>=0.51.0`.

## [0.5.0] - 2026-05-12

### Added

#### `bead.config.compose` — didactic-grounded config composer

- New subpackage `bead.config.compose` replaces the hand-rolled config
  loader. Generic over any `dx.Model` schema; supports the full
  OmegaConf interpolation grammar (`${section.field}`, `${.x}` /
  `${..x}` relative, `${a.b[0]}` and `${a.b.0}` list indexing,
  `${a.${b}}` nested, `\${literal}` escape, cycle detection).
- Built-in resolvers: `oc.env`, `oc.env:VAR,default`, `oc.select`,
  `oc.decode` (base64), `oc.deprecated`, `oc.create`,
  `oc.dict.keys`, `oc.dict.values`. Application-specific resolvers
  register via `bead.config.compose.register_resolver`.
- Bead-specific resolvers in `bead.config.resolvers`:
  `${bead.path:rel}` joins against the active root's
  `paths.data_dir`; `${bead.anchor:name[,attr]}` post-validation
  expansion.
- `defaults: [...]` composition at the top of any YAML/TOML config
  composes referenced files left-to-right before the primary body.
- Strict-merge rejects unknown keys with the dotted path to the
  offending site, walking nested `dx.Embed[T]` models from
  `__field_specs__`.
- TOML configs (`.toml`) supported alongside YAML out of the box.
- `bead.config.load_config` is now a thin wrapper around
  `compose(schema=BeadConfig, ...)`. The previous
  `load_yaml_file` / `merge_configs` helpers are removed.
- CLI: every `bead ...` invocation accepts repeatable
  `--set KEY=VALUE` overrides threaded into the compose pipeline.

#### `ScaleType.FORCED_CHOICE`

- New `ScaleType.FORCED_CHOICE` variant covers N-alternative
  forced-choice tasks where per-item options vary across items
  (response space is a fixed positional label set, e.g.
  `("first", "second")`, but each `Item` carries its own
  alternatives). `family_to_item_template` and the
  active-learning model registry route forced-choice anchors to
  `ForcedChoiceModel`.
- `AnchorSpec.scale_type` is an optional explicit override so config
  files declare the task type alongside the response space.

#### Gallery: `gallery/eng/argument_structure/` v0.4.0 wiring

- New `protocol.py` module exposes `build_protocol()` /
  `acceptability_family()` / `acceptability_anchor()`. The 2AFC
  acceptability question is declared once in `config.yaml` under
  `protocol:` and consumed by every script.
- `generate_deployment.py` and `simulate_pipeline.py` build their
  `ItemTemplate` via `family_to_item_template` instead of literal
  prompt strings.
- `create_2afc_pairs.py` threads the protocol anchor name
  (`"acceptability"`) into every pair's `item_metadata` so the
  JATOS-result → `AnnotationRecord` bridge can match responses
  back to the canonical anchor.
- `make validate-protocol` builds the live `AnnotationProtocol`
  from `config.yaml` and prints the family, prompt, and scale
  type. Wired in as a prerequisite to `make data`.
- `tests/test_protocol.py` covers the config-to-protocol round
  trip, the forced-choice scale type, the `family_to_item_template`
  prompt agreement, and the active-learning model selection for
  the resulting encoding.

## [0.4.0] - 2026-05-07

### Added

#### Pipeline-wide integration of the protocol layer

- `bead.labels` is the single canonical home for the
  `[[label]]` / `[[label:text]]` / `[[label|transform]]` syntax.
  `parse_label_refs`, `find_label_names`, and `replace_label_refs`
  replace the three independent regex implementations that previously
  lived in `bead.protocol.drift`, `bead.deployment.jspsych.trials`,
  and `bead.items.span_labeling`.
- `bead.config.protocol.ProtocolConfig` plugs into `BeadConfig.protocol`
  with declarative TOML/YAML configuration: anchor specs, drift
  settings, realization strategies (template / contextual / lm), and
  family composition. `ProtocolConfig.build(lm_client=..., cache=...)`
  materializes a live `AnnotationProtocol`.
- `bead.protocol.items` provides the canonical
  `QuestionRealization → Item` and protocol-wide
  `family_to_item_template` / `protocol_to_item_templates` /
  `realize_protocol_to_items` bridges, plus `scale_type_to_task_type`
  as the single canonical mapping from `ScaleType` to `TaskType`.
- `bead.active_learning.models.registry` exposes
  `MODEL_CLASSES` / `CONFIG_CLASSES` and
  `model_class_for_task_type` / `config_class_for_task_type` /
  `model_class_for_encoding` / `config_class_for_encoding` as the
  single canonical task-type → model-class / config-class registry.
  `bead.cli.models` and `bead.cli.training` consume the registry
  directly, replacing two parallel string-keyed dicts and a dynamic
  `_import_class` helper.
- `bead.deployment.protocol_trials.protocol_to_jspsych_trials` is the
  canonical end-to-end bridge from an `AnnotationProtocol` and a
  sequence of `ProtocolContext` records to a flat list of jsPsych
  trial dicts.
- `bead.data_collection.jatos_results_to_annotation_records` converts
  raw JATOS results into `AnnotationRecord` instances, the input
  shape consumed by `annotator_reliability` and
  `InterAnnotatorMetrics`.
- `bead protocol` CLI subcommand: `bead protocol validate`,
  `bead protocol realize`, `bead protocol items` drive the
  configured protocol from the shell.

### Changed

- `LMRealization` accepts a `ModelOutputCache` (the bead-wide
  content-addressable cache) via its required `cache` keyword and a
  required `model_name` keyword for cache-key isolation. The internal
  FIFO dict and the `cache` / `max_cache_size` / `clear_cache` /
  `cache_size` parameters and methods are removed; the
  `ModelOutputCache` is the single canonical caching surface.
- `bead.cli.models` no longer maintains `TASK_TYPE_MODELS` /
  `TASK_TYPE_CONFIGS` string-path dicts or the `_import_class`
  helper; they are replaced by direct calls into
  `bead.active_learning.models.registry`. `bead.cli.training` follows
  the same pattern.
- `bead.deployment.jspsych.trials._parse_prompt_references`,
  `_SpanReference`, `_SPAN_REF_PATTERN`, and the duplicated
  `_SPAN_REF_PATTERN` in `bead.items.span_labeling` are removed in
  favor of `bead.labels.parse_label_refs` / `LabelRef`.

#### `bead.protocol`: annotation protocol primitives

A new top-level package providing a type-theoretic stack for defining
annotation protocols: anchors as types, contexts as dependent
indices, realization strategies as computational content, and drift
guards as type-checkers.

- `bead.protocol.anchor` defines `SemanticAnchor` (the type-level
  spec of a question, with required span labels, required keywords,
  optional embedding center and `max_drift`) and `ResponseSpace` /
  `SemanticPoles`.
- `bead.protocol.context` defines a generic `ProtocolContext` and
  `ContextItem` plus a module-level **predicate registry**
  (`register_context_predicate`, `get_context_predicate`,
  `list_context_predicates`) for callers to register named context
  predicates at import time.
- `bead.protocol.realization` provides `RealizationStrategy`
  (`typing.Protocol`), `TemplateRealization`,
  `ContextualTemplateRealization` (rule-based selection from ranked
  variants), and `LMRealization` (with caching and FIFO eviction)
  plus an `LMClient` `Protocol` with explicit
  `temperature` / `max_tokens` keyword parameters.
- `bead.protocol.drift` defines `DriftScore`, the `DriftValidator`
  `Protocol`, and three concrete validators
  (`StructuralDriftValidator`, `EmbeddingDriftValidator`,
  `PerplexityDriftValidator`) plus a composite `DriftGuard`. The
  embedding and perplexity validators consume narrow
  `EmbeddingAdapter` / `PerplexityAdapter` `Protocol`s, so any object
  exposing the right method (including bead's
  `bead.items.adapters.ModelAdapter`) conforms.
- `bead.protocol.family` defines `QuestionFamily` (with explicit
  `depends_on` for conditional dependencies) and `AnnotationProtocol`
  (the iterated dependent product), with `realize_all` threading
  responses through the context. `AnnotationProtocol` rejects
  duplicate anchor names, self-dependencies, and forward / unknown
  `depends_on` references at construction and on `append`.
- `bead.protocol.encoding` defines `ScaleType`
  (`StrEnum: binary / ordinal / nominal`) and `ResponseEncoding` (with
  invariant validators for `n_levels == len(labels)`, label
  uniqueness, and `BINARY` having exactly 2 levels), plus
  `encode_response_space` as the bridge from `ResponseSpace`.
- `bead.protocol.diagnostics` defines `DiagnosticLevel`,
  `DiagnosticRecord`, `DatasetReport` (immutable, with `with_*`
  mutators), `ConditionalObservationValidator` (which operates on
  `AnnotationProtocol.depends_on`), and the `RecordLike` `Protocol`
  for the structural record shape consumed by the validator.
- `LMRealization` raises `RuntimeError` on backend failures and on
  empty / whitespace-only responses (instead of caching an empty
  string).

#### `bead.evaluation.reliability`: per-annotator reliability

- `AnnotationRecord` is a `BeadBaseModel` with the canonical
  `(annotator_id, item_id, question_name, response_label)` shape.
- `annotator_reliability(records, encodings=...)` returns
  per-annotator response distributions and Shannon entropy in bits,
  optionally filtering unrecognized labels.
- `low_entropy_annotators(profiles, threshold=...)` flags annotators
  who collapse the response space.

### Documentation

- `docs/api/protocol.md` and `docs/api/evaluation.md` updates expose
  the new modules through `mkdocstrings`.
- `docs/user-guide/protocols.md` walks through anchors, contexts
  (including the predicate registry and per-dependent attributes),
  the three realization strategies, drift validation (with the named
  `EmbeddingAdapter` and `PerplexityAdapter` Protocols), protocol
  composition, the structural construction-time invariants, the
  `encode_response_space` bridge to modeling, conditional-observation
  diagnostics (including the `RecordLike` Protocol), and reliability.
- The protocol layer is cross-linked from
  `docs/user-guide/concepts.md`, `docs/user-guide/index.md`,
  `docs/index.md`, the project `README.md`, and a new "Protocol layer"
  paragraph in `docs/developer-guide/architecture.md` that places it
  as a cross-cutting layer feeding into the existing 6-stage pipeline.

## [0.3.0] - 2026-05-06

### Changed

#### Model layer migrated from Pydantic to didactic

- Every `pydantic.BaseModel` and `@dataclass` model in `bead.data`, `bead.items`,
  `bead.tokenization`, `bead.resources`, `bead.lists`, `bead.participants`,
  `bead.config`, `bead.active_learning.config`, `bead.dsl.ast`,
  `bead.deployment.distribution`, `bead.deployment.jspsych.config`,
  `bead.behavioral.analytics`, `bead.templates.filler.FilledTemplate`, and
  `bead.transforms.base.TransformContext` now extends `dx.Model` (or
  `dx.TaggedUnion` for the discriminated unions in `bead.lists.constraints` and
  `bead.dsl.ast`).
- All models are frozen by default; mutating `add_*` / `update_modified_time`
  methods become pure `with_*` / `touched` methods that return new instances.
  Cross-field `@model_validator(mode="after")` checks extract to free
  `validate_*` functions that callers invoke explicitly.
- `list[T]` field declarations become `tuple[T, ...]`; nested-Model fields wrap
  with `dx.Embed[T]`. Heterogeneous tuples (e.g. `tuple[int, int]` for scale
  bounds, `tuple[UUID, UUID]` for ordering precedence pairs) become small
  named records (`ScaleBounds`, `OrderingPair`).
- `dict[UUID, X]` and `dict[int, X]` mappings become tuples of records
  (`ConstraintSatisfaction`, `ScalePointLabel`, etc.) — didactic dict keys must
  be `str`. `dict[str, JsonValue]` replaces `dict[str, Any]`.
- `Path`-typed configuration fields (`PathsConfig.data_dir`, `LoggingConfig.file`,
  `TrainerConfig.logging_dir`, `TemplateConfig.mlm_cache_dir`,
  `ResourceConfig.{lexicon,templates,constraints}_path`,
  `ModelMetadata.{training_data_path,eval_data_path,best_checkpoint}`) are
  stored as `str`; callers wrap with `pathlib.Path` on access. Will revert to
  `Path` once panproto/didactic#21 lands.
- `ExperimentConfig.instructions` no longer accepts a bare `str`; pass
  `InstructionsConfig.from_text("...")` for a single-page string.
- `BeadBaseModel.update_modified_time()` (mutating) renamed to `touched()`
  (pure, returns a new instance).

#### Toolchain

- `requires-python` raised to `>=3.14`; pyright `pythonVersion` and ruff
  `target-version` follow.
- `pydantic` removed from project dependencies; `didactic>=0.4.3` and
  `panproto>=0.43` added.

### Added

#### Transforms (`bead.transforms`)

- **TransformContext** carrying span metadata (lemma, POS, head index, tokens) for value-level transformations
- **TransformRegistry** for registering and resolving named transform pipelines
- **morphology** transforms: inflection adjustments driven by `InflectionSpec`
- **text** transforms: case and whitespace normalizers
- Pipeline syntax `[[label|transform1|transform2]]` in prompt span references; transforms resolve display text against the registry at trial generation time
- 69 tests covering registry, morphology, text, and prompt integration

### Changed

- `bead.deployment.jspsych.trials` accepts an optional `TransformRegistry` and passes it through prompt resolution
- Prompt span reference regex now recognizes the `|transform` suffix on `[[label]]` and `[[label:text]]` forms

## [0.2.0] - 2026-02-09

### Added

#### Span Labeling Data Model (`bead.items`)

- **Span**, **SpanLabel**, **SpanSegment** models for stand-off token-level annotation
- **SpanSpec** for defining label vocabularies and relation types
- **SpanRelation** for directed labeled relations between spans
- `add_spans_to_item()` composability function for attaching spans to any item type
- Prompt span references: `[[label]]` and `[[label:text]]` template syntax
  - Auto-fills span token text or uses explicit display text
  - Colors match between stimulus highlighting and prompt highlighting
  - Resolved Python-side at trial generation; plugins receive pre-rendered HTML
  - Early validation warning in `add_spans_to_item()`, hard validation at trial generation

#### Tokenization (`bead.tokenization`)

- **Token** model with `text`, `whitespace`, `index`, `token_space_after` fields
- **TokenizedText** container with token-level access and reconstruction
- Tokenizer backends: whitespace (default), spaCy, Stanza
- Lazy imports for optional NLP dependencies

#### jsPsych Plugins (`bead.deployment.jspsych`)

- 8 new TypeScript plugins following the `JsPsychPlugin` pattern:
  - **bead-binary-choice**: two-alternative forced choice with keyboard support
  - **bead-categorical**: labeled category selection (radio buttons)
  - **bead-free-text**: open-ended text input with optional word count
  - **bead-magnitude**: numeric magnitude estimation with reference stimulus
  - **bead-multi-select**: checkbox-based multi-selection with min/max constraints
  - **bead-slider-rating**: continuous slider with labeled endpoints
  - **bead-rating**: Likert-scale ordinal rating with keyboard shortcuts
  - **bead-span-label**: interactive span highlighting with label assignment, relations, and search
- **span-renderer** library for token-level span highlighting with overlap support
- **gallery-bundle** IIFE build aggregating all plugins for standalone HTML demos
- Keyboard navigation support in forced-choice, rating, and binary-choice plugins
- Material Design styling with responsive layout

#### Deployment Pipeline

- `SpanDisplayConfig` with `color_palette` and `dark_color_palette` for consistent span coloring
- `SpanColorMap` dataclass for deterministic color assignment (same label = same color pair)
- `_assign_span_colors()` shared between stimulus and prompt renderers
- `_generate_span_stimulus_html()` for token-level highlighting in deployed experiments
- Prompt span reference resolution integrated into all 5 composite trial creators (likert, slider, binary, forced-choice, span-labeling)
- Deployment CSS for `.bead-q-highlight`, `.bead-q-chip`, `.bead-span-subscript` in experiment template

#### Interactive Gallery

- 17 demo pages using stimuli from MegaAcceptability, MegaVeridicality, and Semantic Proto-Roles
- Demos cover all plugin types and composite span+task combinations
- Gallery documentation with tabbed Demo / Python / Trial JSON views
- Standalone HTML demos with gallery-bundle.js (no build step required)

#### Tests

- 79 Python span-related tests (items, tokenization, deployment)
- 42 TypeScript tests (20 plugin + 22 span-renderer)
- Prompt span reference tests: parser, color assignment, resolver, integration

### Changed

- Trial generation now supports span-aware stimulus rendering for all task types
- Forced-choice and rating plugins updated with keyboard shortcut support
- Span-label plugin enhanced with searchable fixed labels, interactive relation creation, and relation cleanup on span deletion

## [0.1.0] - 2026-02-04

### Added

#### Core Pipeline (6 Stages)

- **Resources** (`bead.resources`): Lexical items and templates with linguistic features
  - LexicalItem, MultiWordExpression, Lexicon models
  - Template, TemplateSequence, TemplateTree, Slot models
  - Constraint DSL for slot, template, and cross-template constraints
  - Adapters: UniMorph (morphology), Glazing (VerbNet, PropBank, FrameNet)
  - AdapterCache for caching resource adapter results

- **Templates** (`bead.templates`): Template filling and stimulus generation
  - CSPFiller with backtracking and forward checking
  - Strategies: exhaustive, random, stratified
  - ConstraintResolver for DSL-based constraint evaluation
  - HuggingFace MLM adapter for model-based slot ranking
  - Streaming iterator-based generation for large datasets

- **Items** (`bead.items`): Experimental item construction
  - 8 task types: binary, forced-choice, categorical, cloze, free-text, ordinal-scale, magnitude, multi-select
  - ItemTemplate with chunking, timing, and parsing modes
  - Model adapters: HuggingFace (LM, MLM, NLI, SentenceTransformer), OpenAI, Anthropic, Google, TogetherAI
  - ModelOutputCache for efficient caching
  - Rate limiting and retry-with-backoff for API calls

- **Lists** (`bead.lists`): List partitioning with constraint satisfaction
  - ExperimentList and ListCollection models
  - Constraints: uniqueness, balance, quantile, grouped-quantile, diversity, size, ordering, conditional-uniqueness
  - Partitioner and Balancer for balanced assignment
  - JSONL serialization via `to_jsonl()` and `from_jsonl()`

- **Deployment** (`bead.deployment`): Web experiment generation
  - jsPsych 8.x experiment generator with Material Design UI
  - JATOS batch exporter with server-side list distribution
  - 8 distribution strategies: random, sequential, balanced, latin-square, stratified, weighted-random, quota-based, metadata-based
  - Demographics, instructions, and rating scale configuration

- **Training** (`bead.active_learning`): Active learning with convergence detection
  - ActiveLearningLoop orchestrator
  - UncertaintySampler (entropy-based) and RandomSelector
  - 8 task-specific models matching item types
  - Random effects support with participant-level intercepts and slopes
  - HuggingFace and PyTorch Lightning trainers
  - Mixed effects training support

#### Supporting Modules

- **Simulation** (`bead.simulation`): Synthetic judgment generation
  - Annotators: LM-based, random, oracle, distance-based
  - Noise models: temperature, random, systematic
  - Task-specific strategies for all 8 item types
  - SimulationRunner for multi-annotator simulation

- **Evaluation** (`bead.evaluation`): Performance assessment
  - ConvergenceDetector with statistical significance testing
  - InterAnnotatorMetrics: percentage agreement, Cohen's kappa, Fleiss' kappa, Krippendorff's alpha

- **Behavioral** (`bead.behavioral`): Behavioral analytics via slopit
  - JudgmentAnalytics and ParticipantBehavioralSummary models
  - Keystroke, focus, timing, and paste detection analysis
  - Quality control filtering and exclusion list generation

- **Participants** (`bead.participants`): Participant metadata
  - UUID-based participant identification
  - Privacy-preserving external ID mapping
  - Configurable metadata schema validation
  - Merge utilities for pandas and polars

- **Data Collection** (`bead.data_collection`): Platform integration
  - JATOSDataCollector with authentication
  - ProlificDataCollector with webhook support
  - DataMerger for multi-source data

- **DSL** (`bead.dsl`): Constraint domain-specific language
  - Lark-based parser with AST construction
  - Cached evaluation with variable scoping
  - Standard library: string, math, collection, logic functions

- **Config** (`bead.config`): Configuration system
  - YAML-based configuration with environment variable support
  - Profiles: default, dev, prod, test
  - Validation and merging utilities

#### CLI

- `bead init`: Project scaffolding
- `bead config`: Configuration management (show, validate, export, profiles)
- `bead resources`: Resource loading and inspection
- `bead templates`: Template filling
- `bead items`: Item construction
- `bead lists`: List partitioning
- `bead deploy`: jsPsych/JATOS export
- `bead simulate`: Annotation simulation
- `bead training`: Active learning loop
- `bead workflow`: Pipeline orchestration
- `bead shell`: Interactive REPL

#### Infrastructure

- Python 3.13+ with full type annotations
- Pydantic v2 validation
- TypeScript plugins for jsPsych with Biome linting
- MkDocs documentation with mkdocstrings
- CI/CD: GitHub Actions for testing, docs, PyPI publishing
- Read the Docs integration

[Unreleased]: https://github.com/FACTSlab/bead/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/FACTSlab/bead/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/FACTSlab/bead/releases/tag/v0.1.0
