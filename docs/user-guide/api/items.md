# Items Module

The `bead.items` module provides task-type-specific utilities for creating experimental items.

## Task-Type Utilities

The items module provides 9 task-type-specific utilities for programmatic item creation. All utilities follow a consistent API pattern.

### Forced Choice

Create N-alternative forced choice items (2AFC, 3AFC, etc.):

```python
from bead.items.item_template import ScaleBounds, ScalePointLabel  # noqa
from bead.items.forced_choice import create_forced_choice_item

# Create 2AFC item
item = create_forced_choice_item(
    "The cat sleeps",
    "The cat sleep",
)

# Create 3AFC item
item = create_forced_choice_item(
    "Option A",
    "Option B",
    "Option C",
)

# With metadata
item = create_forced_choice_item(
    "The cat sleeps",
    "The cat sleep",
    metadata={"condition": "agreement"},
)
```

**Batch creation from groups**:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.forced_choice import create_forced_choice_items_from_groups
from bead.items.item import Item

# Load existing source items from cross-product items
# Note: tests cd to fixtures dir, so paths are relative to tests/fixtures/api_docs/
source_items = read_jsonlines(
    Path("items/cross_product_items.jsonl"),
    Item,
)

# Create 2AFC items within groups (group by verb_lemma metadata)
# This will create pairs of items that share the same verb
items = create_forced_choice_items_from_groups(
    items=source_items,
    group_by=lambda item: item.item_metadata["verb_lemma"],
    n_alternatives=2,
    extract_text=lambda item: item.rendered_elements.get("template_string", ""),
)

print(f"Created {len(items)} 2AFC items from {len(source_items)} source items")
```

### Ordinal Scale

Create Likert-scale or slider items:

```python
from bead.items.item_template import ScaleBounds, ScalePointLabel
from bead.items.ordinal_scale import create_ordinal_scale_item

# Create 7-point Likert item
item = create_ordinal_scale_item(
    text="How natural is this sentence?",
    scale_bounds=ScaleBounds(min=1, max=7),
    prompt="Rate the sentence:",
    scale_labels=(
        ScalePointLabel(point=1, label="Very unnatural"),
        ScalePointLabel(point=7, label="Very natural"),
    ),
)

# Default 7-point scale
item = create_ordinal_scale_item(
    text="The cat sleeps",
)
```

**Batch creation**:

```python
from bead.items.item_template import ScaleBounds
from bead.items.ordinal_scale import create_ordinal_scale_items_from_texts

sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]

items = create_ordinal_scale_items_from_texts(
    sentences,
    scale_bounds=ScaleBounds(min=1, max=7),
    metadata_fn=lambda text: {"length": len(text)},
)
```

### Binary

Create yes/no or true/false items:

```python
from bead.items.binary import create_binary_item

item = create_binary_item(
    text="Is this sentence grammatical?",
    prompt="Judge grammaticality:",
    binary_options=("Yes", "No"),
)

print(f"Created binary item with options: {item.rendered_elements.get('options')}")
```

### Categorical

Create items with unordered categories (NLI, semantic relations):

```python
from bead.items.categorical import create_categorical_item

item = create_categorical_item(
    text="All dogs bark",
    categories=["entailment", "contradiction", "neutral"],
    prompt="What is the relationship?",
)

# Specialized NLI helper
from bead.items.categorical import create_nli_item

item = create_nli_item(
    premise="All dogs bark",
    hypothesis="Some dogs bark",
)
```

### Free Text

Create open-ended text response items:

```python
from bead.items.free_text import create_free_text_item

item = create_free_text_item(
    text="Translate this sentence to Spanish:",
    prompt="Enter translation:",
    max_length=500,
)
```

### Cloze

Create fill-in-the-blank items:

```python
from bead.items.cloze import create_simple_cloze_item

item = create_simple_cloze_item(
    text="The quick brown fox",
    blank_positions=[1],  # "quick" is blank
    blank_labels=["adjective"],
)
```

### Multi-Select

Create checkbox-style items:

```python
from bead.items.multi_select import create_multi_select_item

item = create_multi_select_item(
    "grammatical",
    "natural",
    "formal",
    "colloquial",
    min_selections=1,
    max_selections=3,
)

n_options = len([k for k in item.rendered_elements if k.startswith("option_")])
print(f"Created multi-select item with {n_options} options")
```

### Magnitude

Create unbounded numeric value items:

```python
from bead.items.magnitude import create_magnitude_item

item = create_magnitude_item(
    text="Reading time in milliseconds:",
    unit="ms",
    bounds=(0, 10000),
    prompt="Enter reading time:",
)

print(f"Created magnitude item with unit: {item.item_metadata.get('unit')}")
```

### Span Labeling

Create items with span annotations for entity labeling, relation extraction, and similar tasks. Spans can be added as standalone items or composed onto any existing task type.

**Standalone span item with pre-defined spans**:

```python
from bead.items.span_labeling import create_span_item
from bead.items.spans import Span, SpanSegment, SpanLabel
from bead.tokenization.config import TokenizerConfig

# create a span item with pre-tokenized text and labeled spans
item = create_span_item(
    text="The quick brown fox jumps over the lazy dog",
    spans=[
        Span(
            span_id="s1",
            segments=[SpanSegment(element_name="text", indices=[1, 2])],
            label=SpanLabel(label="ADJ"),
        ),
        Span(
            span_id="s2",
            segments=[SpanSegment(element_name="text", indices=[3])],
            label=SpanLabel(label="NOUN"),
        ),
    ],
    prompt="Review the highlighted spans:",
    tokenizer_config=TokenizerConfig(backend="whitespace"),
)

print(f"Created span item with {len(item.spans)} spans")
print(f"Tokens: {item.tokenized_elements['text']}")
```

**Interactive span item for participant annotation**:

```python
from bead.items.span_labeling import create_interactive_span_item
from bead.tokenization.config import TokenizerConfig

# create an interactive item where participants select and label spans
item = create_interactive_span_item(
    text="Marie Curie discovered radium in Paris.",
    prompt="Select all named entities and assign a label:",
    tokenizer_config=TokenizerConfig(backend="whitespace"),
    label_set=["PERSON", "LOCATION", "SUBSTANCE"],
    label_source="fixed",
)

print("Created interactive span item")
print(f"Tokens: {item.tokenized_elements['text']}")
```

**Composing spans onto an existing item** (any task type):

```python
from bead.items.item_template import ScaleBounds
from bead.items.ordinal_scale import create_ordinal_scale_item
from bead.items.span_labeling import add_spans_to_item
from bead.items.spans import Span, SpanSegment, SpanLabel
from bead.tokenization.config import TokenizerConfig

# start with a rating item
rating_item = create_ordinal_scale_item(
    text="The scientist discovered a new element.",
    scale_bounds=ScaleBounds(min=1, max=7),
    prompt="Rate the naturalness of this sentence:",
)

# add span annotations as an overlay
item_with_spans = add_spans_to_item(
    item=rating_item,
    spans=[
        Span(
            span_id="agent",
            segments=[SpanSegment(element_name="text", indices=[0, 1])],
            label=SpanLabel(label="AGENT"),
        ),
    ],
    tokenizer_config=TokenizerConfig(backend="whitespace"),
)

print(f"Original spans: {len(rating_item.spans)}")
print(f"After adding: {len(item_with_spans.spans)}")
```

### Prompt Span References

When composing spans with other task types, prompts can reference span labels using `[[label]]` syntax. At deployment time, these references are replaced with color-highlighted HTML that matches the span colors in the stimulus text.

**Syntax**:

| Pattern | Behavior |
|---------|----------|
| `[[label]]` | Auto-fills with the span's token text (e.g., "The boy") |
| `[[label:custom text]]` | Uses the provided text instead (e.g., "the breaking") |

**Example**: a rating item with highlighted prompt references:

```python
from bead.items.item_template import ScaleBounds, ScalePointLabel
from bead.items.ordinal_scale import create_ordinal_scale_item
from bead.items.span_labeling import add_spans_to_item
from bead.items.spans import Span, SpanLabel, SpanSegment

item = create_ordinal_scale_item(
    text="The boy broke the vase.",
    prompt="How likely is it that [[breaker]] existed after [[event:the breaking]]?",
    scale_bounds=ScaleBounds(min=1, max=5),
    scale_labels=(
        ScalePointLabel(point=1, label="Very unlikely"),
        ScalePointLabel(point=5, label="Very likely"),
    ),
)

from bead.tokenization.config import TokenizerConfig

item = add_spans_to_item(
    item,
    spans=[
        Span(
            span_id="span_0",
            segments=[SpanSegment(element_name="text", indices=[0, 1])],
            label=SpanLabel(label="breaker"),
        ),
        Span(
            span_id="span_1",
            segments=[SpanSegment(element_name="text", indices=[2])],
            label=SpanLabel(label="event"),
        ),
    ],
    tokenizer_config=TokenizerConfig(backend="whitespace"),
)
```

When this item is deployed, the prompt renders as:

> How likely is it that <span style="background:#BBDEFB;padding:1px 4px;border-radius:3px">The boy</span> existed after <span style="background:#C8E6C9;padding:1px 4px;border-radius:3px">the breaking</span>?

Colors are assigned deterministically: the same label always gets the same color pair in both the stimulus and the prompt. Auto-fill (`[[breaker]]`) reconstructs the span's token text by joining tokens from `tokenized_elements` and respecting `token_space_after` flags. Custom text (`[[event:the breaking]]`) lets you use a different surface form when the prompt needs a morphological variant of the span text (e.g., "ran" in the target vs. "the running" in the prompt).

If a prompt references a label that doesn't exist among the item's spans, `add_spans_to_item()` issues a warning at item construction time, and trial generation raises a `ValueError`.

**Adding tokenization to an existing item**:

```python
from bead.items.binary import create_binary_item
from bead.items.span_labeling import tokenize_item
from bead.tokenization.config import TokenizerConfig

# create a binary item without tokenization
binary_item = create_binary_item(
    text="The cat sat on the mat.",
    prompt="Is this sentence grammatical?",
)

# add tokenization data
tokenized = tokenize_item(
    binary_item,
    tokenizer_config=TokenizerConfig(backend="whitespace"),
)

print(f"Tokenized elements: {list(tokenized.tokenized_elements.keys())}")
print(f"Tokens for 'text': {tokenized.tokenized_elements.get('text')}")
```

**Batch creation with a span extractor**:

```python
from bead.items.span_labeling import create_span_items_from_texts
from bead.items.spans import Span, SpanSegment, SpanLabel
from bead.tokenization.config import TokenizerConfig


# define a span extractor function
def find_capitalized_spans(text: str, tokens: list[str]) -> list[Span]:
    """Extract spans for capitalized words (simple NER heuristic)."""
    spans: list[Span] = []
    for i, token in enumerate(tokens):
        if token[0].isupper() and i > 0:
            spans.append(
                Span(
                    span_id=f"cap_{i}",
                    segments=[SpanSegment(element_name="text", indices=[i])],
                    label=SpanLabel(label="ENTITY"),
                )
            )
    return spans


sentences = [
    "Marie Curie was born in Warsaw.",
    "Albert Einstein developed relativity in Berlin.",
    "Ada Lovelace wrote the first algorithm.",
]

items = create_span_items_from_texts(
    texts=sentences,
    span_extractor=find_capitalized_spans,
    prompt="Review the detected entities:",
    tokenizer_config=TokenizerConfig(backend="whitespace"),
    labels=["ENTITY"],
)

print(f"Created {len(items)} span items")
for item in items:
    print(f"  {item.rendered_elements['text']}: {len(item.spans)} spans")
```

## Language Model Scoring

Score items with language models:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.items.scoring import LanguageModelScorer

# Load items from fixtures
source_items = read_jsonlines(
    Path("items/cross_product_items.jsonl"),
    Item,
)

# Create scorer
scorer = LanguageModelScorer(
    model_name="gpt2",
    cache_dir=Path(".cache/scoring"),
    device="cpu",
    text_key="template_string",
)

# Score first few items
items_to_score = source_items[:3]
scores = scorer.score_batch(items_to_score)

# Add scores to metadata
for item, score in zip(items_to_score, scores, strict=True):
    item.item_metadata["lm_score"] = score

print(f"Scored {len(items_to_score)} items")
```

## Item Validation

Validate items conform to task-type requirements:

```python
from bead.items.item_template import ScaleBounds
from bead.items.ordinal_scale import create_ordinal_scale_item
from bead.items.validation import (
    get_task_type_requirements,
    infer_task_type_from_item,
    validate_item_for_task_type,
)

# Create an item to validate
item = create_ordinal_scale_item(
    text="The cat sleeps", scale_bounds=ScaleBounds(min=1, max=7)
)

# Validate structure
validate_item_for_task_type(item, "ordinal_scale")  # Raises ValueError if invalid
print("Item is valid for ordinal_scale")

# Infer task type
task_type = infer_task_type_from_item(item)
print(f"Inferred task type: {task_type}")

# Get requirements
reqs = get_task_type_requirements("ordinal_scale")
print(f"Requirements: {list(reqs.keys())}")
```

## Complete Example

From [gallery/eng/argument_structure/create_2afc_pairs.py](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/create_2afc_pairs.py):

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.forced_choice import create_forced_choice_items_from_groups
from bead.items.item import Item
from bead.items.scoring import LanguageModelScorer

# Load source items (already in Item format)
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
afc_items = create_forced_choice_items_from_groups(
    items=items_to_score,
    group_by=lambda item: item.item_metadata["verb_lemma"],
    n_alternatives=2,
    extract_text=lambda item: item.rendered_elements.get("template_string", ""),
)

print(f"Created {len(afc_items)} 2AFC items")

# Save example (commented out for testing)
# from bead.data.serialization import write_jsonlines
# write_jsonlines(afc_items, Path("output/2afc_items.jsonl"))
```

## Design Principles

1. **NO Silent Fallbacks**: All errors raise `ValueError` with descriptive messages
2. **Strict Validation**: Use `zip(..., strict=True)`, explicit parameter checks
3. **Consistent API**: Same pattern across all 9 task types
4. **Automatic Metadata**: Utilities populate task-specific metadata (n_options, scale_min/max, etc.)

## Task Type Summary

| Task Type | Use For | Key Function |
|-----------|---------|--------------|
| `forced_choice` | N-AFC items | `create_forced_choice_item()` |
| `ordinal_scale` | Likert, slider | `create_ordinal_scale_item()` |
| `binary` | Yes/No | `create_binary_item()` |
| `categorical` | NLI, relations | `create_categorical_item()` |
| `free_text` | Open-ended | `create_free_text_item()` |
| `cloze` | Fill-in-blank | `create_cloze_item()` |
| `multi_select` | Checkboxes | `create_multi_select_item()` |
| `magnitude` | Numeric | `create_magnitude_item()` |
| `span_labeling` | Entity/span annotation | `create_span_item()` |

## Next Steps

- [Lists module](lists.md): Partition items into balanced lists
- [CLI reference](../cli/items.md): Command-line equivalents
- [Gallery example](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/create_2afc_pairs.py): Full working script
