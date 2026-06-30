# Annotation Protocols

The `bead.protocol` package gives you a type-theoretic stack for
defining annotation protocols. Four roles work together:

- A **semantic anchor** is the *type* of a question: a declarative
  specification of what is being measured.
- A **protocol context** is the dependent *index*: everything known
  about the current target.
- A **realization strategy** is the computational *content* of the
  dependent function `Pi(ctx). Question(ctx)`: it produces the
  prompt string a participant will see.
- A **drift guard** is the *type-checker*: it verifies that a realized
  prompt still inhabits the type defined by its anchor.

`QuestionFamily` packages these together; `AnnotationProtocol`
sequences families into the iterated dependent product
`Sigma(a_1 : Q_1(ctx)). Sigma(a_2 : Q_2(ctx, a_1)). ...`, threading
responses through the context so later questions can condition on
earlier answers.

## Why a protocol layer?

Without a separation between the question's type and its phrasing,
two questions that elicit different responses can look identical, and
two phrasings of the *same* question can look different. The protocol
layer makes the invariants explicit:

- The anchor declares which property is measured, the response space,
  required keywords, and required span references.
- The realization can vary by context (template variants, LM
  paraphrase) but must preserve the anchor's invariants.
- The drift guard catches realizations that fail to preserve them.

## Defining an anchor

```python
from bead.protocol import ResponseSpace, SemanticAnchor
from bead.protocol.anchor import SemanticPoles

response_space = ResponseSpace(
    options=(
        "definitely no",
        "probably no",
        "unsure",
        "probably yes",
        "definitely yes",
    ),
    is_ordered=True,
    semantic_poles=SemanticPoles(
        low="definitely no", high="definitely yes",
    ),
)

completion = SemanticAnchor(
    name="completion",
    target_property="telicity",
    canonical_prompt="Does [[situation]] reach a definite endpoint?",
    response_space=response_space,
    required_span_labels=frozenset({"situation"}),
    required_keywords=frozenset({"endpoint"}),
    description="Whether the event reaches a culmination.",
)
```

Use `SemanticAnchor.from_response_options` for the common case of an
anchor whose response space is built inline:

```python
completion = SemanticAnchor.from_response_options(
    name="completion",
    target_property="telicity",
    canonical_prompt="Does [[situation]] reach an endpoint?",
    options=("no", "yes"),
    is_ordered=False,
    required_span_labels=frozenset({"situation"}),
)
```

## Building a context

`ProtocolContext` carries sentence-level, target-level, and
dependent-level information common to most annotation protocols.
Domain-specific data lives in the inherited `metadata` map (a JSON
dict from `BeadBaseModel`):

```python
from bead.protocol import ContextItem, ProtocolContext

ctx = ProtocolContext(
    sentence="Mary built a sandcastle.",
    target_lemma="build",
    target_form="built",
    target_upos="VERB",
    target_position=2,
    target_span_text="built a sandcastle",
    target_span_positions=(2, 3, 4),
    dependents=(
        ContextItem(
            head_lemma="Mary", head_upos="PROPN",
            head_position=1, span_text="Mary",
        ),
        ContextItem(
            head_lemma="sandcastle", head_upos="NOUN",
            head_position=4, span_text="a sandcastle",
            attributes={"definiteness": 0.0},
        ),
    ),
)
```

### Domain-specific dependent attributes

Each `ContextItem` carries an `attributes: dict[str, float]` map for
domain-specific scalar properties (semantic-role probabilities,
definiteness scores, frequency, ...). `ContextItem.attribute(name)`
returns `None` when the attribute is absent so callers do not have
to handle a separate `KeyError`:

```python
mary, sandcastle = ctx.dependents
sandcastle.attribute("definiteness")   # 0.0
sandcastle.attribute("missing") is None  # True
```

### The context-predicate registry

`ContextualTemplateRealization` and other strategies look predicates
up by name from a module-level registry rather than passing functions
directly. Register at import time, look up at realization time:

```python
from bead.protocol import (
    register_context_predicate, get_context_predicate,
    list_context_predicates, ProtocolContext,
)

def has_plural_dependent(ctx: ProtocolContext) -> bool:
    return any(d.is_plural for d in ctx.dependents)

register_context_predicate("has_plural_dependent", has_plural_dependent)

assert get_context_predicate("has_plural_dependent") is has_plural_dependent
assert "has_plural_dependent" in list_context_predicates()
```

The protocol layer ships one predicate, `always`, which is also the
default condition for `TemplateVariant`. The registry is global
mutable state, populated at import time and read at realization
time; it is not designed for per-request mutation.

## Threading dependent responses

`ProtocolContext.with_response` returns a new context with one
additional response recorded; the original is unchanged.

```python
ctx2 = ctx.with_response("change", "yes")
ctx3 = ctx2.with_response("completion", "probably yes")
ctx3.previous_responses
# {'change': 'yes', 'completion': 'probably yes'}
```

## Realization strategies

`TemplateRealization` returns a fixed template (or the anchor's
canonical prompt when no template is configured):

```python
from bead.protocol import TemplateRealization

tr = TemplateRealization()  # echoes anchor.canonical_prompt
```

`ContextualTemplateRealization` selects from ranked variants:

```python
from bead.protocol import ContextualTemplateRealization, TemplateVariant

contextual = ContextualTemplateRealization(
    variants=(
        TemplateVariant(
            template="Does [[situation]] end at a specific point?",
            condition=lambda ctx: ctx.target_upos == "VERB",
            priority=10,
        ),
        TemplateVariant(
            template="Does [[situation]] have a specific end?",
            priority=0,
        ),
    ),
)
```

`LMRealization` paraphrases the canonical prompt via a language-model
client. Always pair it with a drift guard, and pass a
`bead.items.cache.ModelOutputCache` so realizations participate in
bead's single canonical caching surface:

```python
from bead.items.cache import ModelOutputCache
from bead.protocol import LMClient, LMRealization

class StubClient:
    def complete(
        self, prompt: str, *, temperature: float, max_tokens: int,
    ) -> str:
        return "Did the event reach an endpoint?"

assert isinstance(StubClient(), LMClient)
cache = ModelOutputCache(backend="memory")
lm = LMRealization(
    StubClient(),
    model_name="stub-paraphraser",
    cache=cache,
    temperature=0.3,
    max_tokens=200,
)
```

`LMClient` is a `typing.Protocol`: any object with a `complete(prompt,
*, temperature, max_tokens) -> str` method conforms. Cache entries
key off `(model_name, "lm_completion", prompt=full_prompt)`; passing
the same `ModelOutputCache` to multiple `LMRealization`s with
different `model_name` values keeps their entries isolated. Without a
cache (`cache=None`), every `realize()` call hits the backend.
`LMRealization.realize` raises `RuntimeError` on backend failures or
empty / whitespace-only responses, so a misbehaving LM cannot silently
pollute the cache.

## Drift validation

The drift guard composes structural, embedding, and perplexity
validators.

```python
from bead.protocol import (
    DriftGuard,
    EmbeddingDriftValidator,
    PerplexityDriftValidator,
    StructuralDriftValidator,
)

guard = DriftGuard(
    validators=[
        StructuralDriftValidator(min_length=15),
        EmbeddingDriftValidator(adapter, max_distance=0.4),
        PerplexityDriftValidator(adapter, max_perplexity=80.0),
    ],
)
```

`StructuralDriftValidator` checks `[[label]]` references, required
keywords, length, and trailing `?`. `EmbeddingDriftValidator` runs on
the embedding adapter; if the anchor sets `embedding_center` and
`max_drift`, those are used as the cosine ceiling.
`PerplexityDriftValidator` flags realizations whose perplexity
exceeds a configured ceiling.

The embedding and perplexity validators consume narrow
`typing.Protocol`s, so any object with the right shape can serve as
the backend:

```python
from bead.protocol import EmbeddingAdapter, PerplexityAdapter

# Conforms structurally:
class MyBackend:
    def get_embedding(self, text: str) -> Sequence[float]: ...
    def compute_perplexity(self, text: str) -> float: ...

assert isinstance(MyBackend(), EmbeddingAdapter)
assert isinstance(MyBackend(), PerplexityAdapter)
```

Bead's `bead.items.adapters.ModelAdapter` family conforms out of the
box.

## Composing a protocol

```python
from bead.protocol import AnnotationProtocol, QuestionFamily

change = QuestionFamily(
    anchor=change_anchor,
    realization=contextual,
    drift_guard=guard,
)

uniformity = QuestionFamily(
    anchor=uniformity_anchor,
    realization=TemplateRealization(),
    drift_guard=guard,
    condition=(
        lambda ctx: ctx.previous_responses.get("change") == "yes"
    ),
    depends_on=("change",),
)

protocol = AnnotationProtocol(
    families=[change, uniformity], name="aspect-protocol",
)

realizations = protocol.realize_all(
    ctx, responses={"change": "yes"},
)
```

`realize_all` threads each response into the context before evaluating
the next family; non-applicable families are skipped. When a response
is not pre-supplied, the first option of the family's response space
is used as a placeholder so downstream conditional families can still
be exercised in dry-run mode.

## Constructor invariants

The protocol layer enforces a small set of structural invariants at
construction time so configuration errors fail loudly instead of
manifesting as confusing behavior at realization time:

- `ResponseEncoding` requires `n_levels == len(labels)`, rejects
  duplicate labels, and rejects `BINARY` scales with anything other
  than 2 levels. (`encode_response_space` derives all three from the
  source `ResponseSpace` so it never produces an invalid encoding.)
- `AnnotationProtocol` rejects duplicate anchor names, families that
  depend on themselves, and families whose `depends_on` references a
  family that is not present *earlier* in the sequence (forward
  references and unknown references are both refused). The same
  validation runs on `AnnotationProtocol.append`.
- `LMRealization(client, max_cache_size=...)` requires
  `max_cache_size > 0`.
- `PerplexityDriftValidator(..., max_perplexity=...)` requires
  `max_perplexity > 0`.

Together with the drift validators that fire at realization time,
these invariants make the construction of an
`AnnotationProtocol` a complete static check: if construction
succeeds, every realization is well-formed up to the LM's behavior,
and any LM misbehavior is caught by the drift guard.

## Bridging to the modeling layer

`encode_response_space` converts a `ResponseSpace` into a
likelihood-agnostic `ResponseEncoding`:

```python
from bead.protocol import encode_response_space

encoding = encode_response_space("change", change_anchor.response_space)
encoding.is_binary       # True for two-option, unordered spaces
encoding.label_to_index("yes")
encoding.index_to_label(0)
```

`bead.active_learning.models` registers the canonical
`ScaleType` → model-class mapping. To pick the active-learning model
class for an encoding:

```python
from bead.active_learning.models import (
    config_class_for_encoding,
    model_class_for_encoding,
)

ModelClass = model_class_for_encoding(encoding)   # e.g. BinaryModel
ConfigClass = config_class_for_encoding(encoding)  # e.g. BinaryModelConfig
model = ModelClass(ConfigClass(model_name="bert-base-uncased"))
```

The same registry (`MODEL_CLASSES` / `CONFIG_CLASSES`) drives the
`bead models train-model` and `bead training` CLI commands, so there
is exactly one mapping from task type to model class across the
codebase.

## Bridging to item construction

`bead.protocol.items` is the single canonical bridge from a realized
question to a fully-populated `bead.items.Item`:

```python
from bead.protocol import (
    family_to_item_template,
    realization_to_item,
    realize_protocol_to_items,
)

# Per-family templates (one per anchor):
template = family_to_item_template(
    family_change, judgment_type="acceptability",
)

# Per-context realization → Item:
realization = family_change.realize(ctx)
item = realization_to_item(realization, item_template=template)

# Whole-protocol convenience:
pairs = realize_protocol_to_items(
    protocol, ctx, judgment_type="acceptability",
)
for realization, item in pairs:
    ...  # downstream item processing
```

`scale_type_to_task_type` is the canonical translation used here and
in the active-learning registry. There is no other mapping: every
protocol family produces exactly one `ItemTemplate`.

### Forced-choice anchors

`ScaleType.FORCED_CHOICE` covers N-alternative forced-choice
questions where the *response space* is a fixed positional label
set (e.g. `("first", "second")`) but the per-item alternatives
vary across items. Declare the scale type explicitly on the
anchor — it cannot be inferred from the response space alone:

```python
anchor = SemanticAnchor(
    name="acceptability",
    canonical_prompt="Which sentence sounds more natural?",
    response_space=ResponseSpace(
        options=("first", "second"),
        is_ordered=False,
        scale_type=ScaleType.FORCED_CHOICE,
    ),
    ...,
)
```

`family_to_item_template` maps `FORCED_CHOICE` to
`task_type="forced_choice"` with `task_spec.options=None`
(per-item alternatives live on each `Item`); the active-learning
registry routes the resulting encoding to `ForcedChoiceModel`.

In YAML, set `scale_type: "forced_choice"` on the `AnchorSpec`.

## Bridging to deployment

`bead.deployment.protocol_trials.protocol_to_jspsych_trials` is the
single canonical bridge from a configured protocol and a sequence of
contexts to a flat list of jsPsych trial dicts ready for batch
deployment:

```python
from bead.deployment.protocol_trials import protocol_to_jspsych_trials

trials = protocol_to_jspsych_trials(
    protocol,
    contexts,
    experiment_config=experiment_config,
    judgment_type="acceptability",
    rating_config=rating_config,    # for ordinal scales
    choice_config=choice_config,    # for binary / categorical
)
```

Each context is realized through every applicable family; each
resulting realization is packaged as an `Item`, bound to its
family's `ItemTemplate`, and fed through
`bead.deployment.jspsych.trials.create_trial`. Trials are returned
in `(context_order, family_order)` with consecutive `trial_number`
fields.

## Bridging back from JATOS

After deployment, `bead.data_collection.jatos_results_to_annotation_records`
is the single canonical conversion from raw JATOS results to
`bead.evaluation.AnnotationRecord` instances:

```python
from bead.data_collection import (
    JATOSDataCollector,
    jatos_results_to_annotation_records,
)
from bead.evaluation import annotator_reliability

results = JATOSDataCollector(...).download_results(Path("results.jsonl"))
records = jatos_results_to_annotation_records(results)
profiles = annotator_reliability(records)
```

The bridge looks up the annotator id in `urlQueryParameters`
(`"PROLIFIC_PID"` by default; configurable), then walks each result's
trial array picking the trials with `item_id` and `template_name`
fields populated by the jsPsych deployment layer. Trials missing
those fields (instructions, consent, demographics) are skipped.
Numeric responses are stringified so the resulting
`response_label` matches the encoding's labels for the corresponding
family.

## Configuration-driven workflow

`bead.config.protocol.ProtocolConfig` is the single canonical
declarative form of a protocol. It plugs into `BeadConfig` as the
`protocol` section, and a complete protocol is materialized via
`ProtocolConfig.build()`:

```yaml
# bead.yaml
protocol:
  name: aspect-protocol
  drift:
    min_length: 15
    require_question_mark: true
  lm_model_name: gpt-4o-mini
  lm_temperature: 0.3
  families:
    - anchor:
        name: change
        target_property: dynamicity
        canonical_prompt: "Is anything changing in [[situation]] over time?"
        options: ["no", "yes"]
        is_ordered: false
        required_span_labels: [situation]
      realization_kind: contextual
      variants:
        - template: "Is [[situation]] something that is changing?"
          condition_name: always
          priority: 0
    - anchor:
        name: completion
        target_property: telicity
        canonical_prompt: "Does [[situation]] reach a definite endpoint?"
        options: ["definitely no", "probably no", "unsure",
                  "probably yes", "definitely yes"]
        is_ordered: true
        semantic_pole_low: "definitely no"
        semantic_pole_high: "definitely yes"
        required_span_labels: [situation]
      realization_kind: lm
      condition_name: always
      depends_on: [change]
```

```python
from bead.config import load_config

config = load_config("bead.yaml")
protocol = config.protocol.build(
    lm_client=my_lm_client,
    cache=ModelOutputCache(backend="filesystem"),
)
```

Predicates (`condition_name`) are looked up by name from the registry
documented above. Every realization strategy (`template`,
`contextual`, `lm`) and drift validator
(`StructuralDriftValidator` always on; `EmbeddingDriftValidator` and
`PerplexityDriftValidator` opt-in via `drift.enable_embedding` and
`drift.enable_perplexity`) is reachable from configuration without
writing Python.

## CLI

The `bead protocol` subcommand drives the configuration-loaded
protocol from the shell:

```bash
# Validate the protocol config and report each family's scale + deps
bead protocol validate

# Realize prompts for every context in contexts.jsonl
bead protocol realize contexts.jsonl realizations.jsonl

# Realize and emit fully-populated Items (skip the realization step)
bead protocol realize contexts.jsonl items.jsonl --emit-items

# Emit per-family ItemTemplates
bead protocol items templates.jsonl --judgment-type acceptability
```

Every CLI command reads the same `BeadConfig` as the Python API, so
configuration is the single source of truth.

## Diagnostics

`DatasetReport` accumulates immutable diagnostic findings. Every
mutating method returns a new instance.

```python
from bead.protocol import DatasetReport, DiagnosticLevel

report = (
    DatasetReport(n_records_input=42, n_items=20)
    .with_coverage("change", 0.95)
    .add(DiagnosticLevel.WARNING, "missing_response", "item i12 has no response")
)
print(report.summary())
```

`ConditionalObservationValidator` inspects records against the
protocol's `depends_on` graph:

```python
from bead.protocol import ConditionalObservationValidator
from bead.evaluation import AnnotationRecord

records = {
    "change": [
        AnnotationRecord(
            annotator_id="a1", item_id="i1",
            question_name="change", response_label="yes",
        ),
    ],
    "uniformity": [
        AnnotationRecord(
            annotator_id="a1", item_id="i1",
            question_name="uniformity", response_label="yes",
        ),
    ],
}

validator = ConditionalObservationValidator(
    conditioning_values={"uniformity": {"yes"}},
)
findings = validator.validate(records, protocol)
```

`ConditionalObservationValidator` accepts any record type conforming
to the `RecordLike` Protocol (anything with `item_id`,
`response_label`, and `question_name` string attributes), so callers
are not bound to `bead.evaluation.AnnotationRecord` specifically.

## Reliability

`bead.evaluation.reliability` complements
`bead.evaluation.InterAnnotatorMetrics` with per-annotator entropy:

```python
from bead.evaluation import (
    annotator_reliability, low_entropy_annotators,
)

profiles = annotator_reliability(records_flat)
flagged = low_entropy_annotators(profiles, threshold=0.5)
```

Low entropy means the annotator is collapsing the response space
(always picking the same label, always picking the midpoint, ...).

`annotator_reliability(records, encodings=...)` accepts an optional
`Mapping[str, ResponseEncoding]` keyed by anchor name. When supplied,
response labels not present in the encoding for a question are
silently skipped, which is useful after schema evolution invalidates
some legacy labels.

`low_entropy_annotators` accepts two refinements:

- `question_name="..."` restricts the threshold check to a single
  question's entropy (otherwise the *minimum* per-question entropy
  is checked).
- `require_min_responses=N` skips annotators with fewer than `N`
  recorded responses, so an annotator who answered only one or two
  items is not flagged purely on small-sample entropy.

## Bridging to bead's item layer

`QuestionRealization.prompt` is a string. It can be passed straight
into bead's existing item-construction pipeline (`ItemTemplate`,
`ItemConstructor`, ...) where the `[[label]]` markers in the
realization are resolved against the item's spans. The protocol layer
deliberately does *not* perform that resolution itself: the anchor
and the realization stay agnostic to bead's `{slot}` template syntax,
so the same realization can be reused across runtimes.
