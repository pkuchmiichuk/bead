# Templates Module

The `bead.templates` module provides template filling strategies and renderers for generating experimental stimuli.

## Loading Filled Templates

After template filling, load previously generated filled templates:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.templates.filler import FilledTemplate

# Load filled templates from JSONL
# Note: tests cd to fixtures dir, so paths are relative to tests/fixtures/api_docs/
filled_templates = read_jsonlines(
    Path("filled_templates/generic_frames_filled.jsonl"),
    FilledTemplate,
)

print(f"Loaded {len(filled_templates)} filled templates")

# Access first filled template
first = filled_templates[0]
print(f"Template: {first.template_name}")
print(f"Rendered: {first.rendered_text}")
print(f"Slot fillers: {list(first.slot_fillers.keys())}")
```

**CSPFiller parameters:**

- `lexicon` (required): Single lexicon to draw items from
- `max_attempts` (default: 10000): Maximum backtracking attempts
- `renderer` (optional): Custom renderer for template strings

## Filling Strategies

Strategies determine how to combine lexical items to fill template slots. There are two categories:

### Basic Strategies (generate_combinations)

These strategies work with pre-filtered slot items and generate combinations:

**ExhaustiveStrategy**: all possible combinations

```python
from pathlib import Path

from bead.resources.lexicon import Lexicon
from bead.templates.strategies import ExhaustiveStrategy

# Load lexicons
nouns = Lexicon.from_jsonl(Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns")
verbs = Lexicon.from_jsonl(Path("lexicons/verbnet_verbs.jsonl"), "verbnet_verbs")

strategy = ExhaustiveStrategy()

# Requires pre-filtered slot items
noun_items = list(nouns.items)[:2]
verb_items = list(verbs.items)[:2]

slot_items = {
    "subject": noun_items,
    "verb": verb_items,
}
combinations = strategy.generate_combinations(slot_items)
print(f"Generated {len(list(combinations))} combinations")
```

**RandomStrategy**: random sampling

```python
from pathlib import Path

from bead.resources.lexicon import Lexicon
from bead.templates.strategies import RandomStrategy

# Load lexicons and prepare slot items
nouns = Lexicon.from_jsonl(Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns")
verbs = Lexicon.from_jsonl(Path("lexicons/verbnet_verbs.jsonl"), "verbnet_verbs")

slot_items = {
    "subject": list(nouns.items),
    "verb": list(verbs.items),
}

strategy = RandomStrategy(
    n_samples=100,
    seed=42,
)
combinations = strategy.generate_combinations(slot_items)
print(f"Generated {len(list(combinations))} combinations")
```

**StratifiedStrategy**: balanced sampling across feature values

```python
from pathlib import Path

from bead.resources.lexicon import Lexicon
from bead.templates.strategies import StratifiedStrategy

# Load lexicons and prepare slot items
nouns = Lexicon.from_jsonl(Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns")
verbs = Lexicon.from_jsonl(Path("lexicons/verbnet_verbs.jsonl"), "verbnet_verbs")

slot_items = {
    "subject": list(nouns.items),
    "verb": list(verbs.items),
}

strategy = StratifiedStrategy(
    grouping_property="pos",
    n_samples=100,
    seed=42,
)
combinations = strategy.generate_combinations(slot_items)
print(f"Generated {len(list(combinations))} combinations")
```

### MLM-Based Strategies (generate_from_template)

These strategies use masked language models and work directly with templates:

**MLMFillingStrategy**: beam search with language models

```python
from pathlib import Path

from bead.resources.lexicon import Lexicon
from bead.resources.template_collection import TemplateCollection
from bead.templates.adapters.cache import ModelOutputCache
from bead.templates.adapters.huggingface import HuggingFaceMLMAdapter
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import MLMFillingStrategy

# Load template and lexicons
templates = TemplateCollection.from_jsonl(
    Path("templates/generic_frames.jsonl"), "generic_frames"
)
template = list(templates.templates)[0]

nouns = Lexicon.from_jsonl(Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns")
verbs = Lexicon.from_jsonl(Path("lexicons/verbnet_verbs.jsonl"), "verbnet_verbs")

# Initialize model
adapter = HuggingFaceMLMAdapter("bert-base-uncased")
adapter.load_model()

# Initialize cache and resolver
cache = ModelOutputCache(cache_dir=Path(".cache"))
resolver = ConstraintResolver()

# Create strategy
strategy = MLMFillingStrategy(
    resolver=resolver,
    model_adapter=adapter,
    beam_size=5,
    fill_direction="left_to_right",
    top_k=10,
    cache=cache,
)

# Fill template directly
combinations = list(
    strategy.generate_from_template(
        template=template,
        lexicons=[nouns, verbs],
        language_code="en",
    )
)
print(f"Generated {len(combinations)} combinations")
```

**MixedFillingStrategy**: different strategies per slot

```python
from pathlib import Path

from bead.templates.strategies import MixedFillingStrategy

# MixedFillingStrategy allows different filling strategies per slot
# For example, use exhaustive for some slots and MLM for others

# Define per-slot strategies
# Each slot maps to (strategy_name, strategy_kwargs)
slot_strategies = {
    "det": ("exhaustive", {}),  # All determiners
    "noun": ("exhaustive", {}),  # All nouns
    "verb": ("exhaustive", {}),  # All verbs
    "adjective": (
        "mlm",  # Use masked language model for adjectives
        {
            "beam_size": 5,
            "top_k": 10,
        },
    ),
}

strategy = MixedFillingStrategy(slot_strategies=slot_strategies)

# When calling generate_from_template(), the strategy will:
# - Use exhaustive enumeration for det, noun, verb slots
# - Use MLM beam search for adjective slot
# This is useful when some slots have small vocabularies (exhaustive)
# and others need context-aware selection (MLM)

print(f"Strategy configured for {len(slot_strategies)} slots")
```

## Creating FilledTemplate Objects

After generating combinations, create `FilledTemplate` objects:

```python
from pathlib import Path

from bead.resources.lexicon import Lexicon
from bead.resources.template_collection import TemplateCollection
from bead.templates.filler import FilledTemplate
from bead.templates.renderers import DefaultRenderer
from bead.templates.strategies import ExhaustiveStrategy

# Load template and lexicons
templates = TemplateCollection.from_jsonl(
    Path("templates/generic_frames.jsonl"), "generic_frames"
)
template = list(templates.templates)[0]

nouns = Lexicon.from_jsonl(Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns")
verbs = Lexicon.from_jsonl(Path("lexicons/verbnet_verbs.jsonl"), "verbnet_verbs")

# Generate combinations
strategy = ExhaustiveStrategy()
slot_items = {
    "subject": list(nouns.items)[:2],
    "verb": list(verbs.items)[:2],
}
combinations = list(strategy.generate_combinations(slot_items))

renderer = DefaultRenderer()

filled_templates = []
for combo in combinations:
    # Render text
    rendered = renderer.render(
        template.template_string,
        combo,
        template.slots,
    )

    # Create FilledTemplate
    filled = FilledTemplate(
        template_id=str(template.id),
        template_name=template.name,
        slot_fillers=combo,
        rendered_text=rendered,
        strategy_name="exhaustive",
        template_slots={name: slot.required for name, slot in template.slots.items()},
    )
    filled_templates.append(filled)

print(f"Created {len(filled_templates)} FilledTemplate objects")
```

## Custom Renderers

Renderers control how templates are converted to text:

**DefaultRenderer**: basic string formatting

```python
from pathlib import Path

from bead.resources.lexicon import Lexicon
from bead.resources.template_collection import TemplateCollection
from bead.templates.renderers import DefaultRenderer

# Load lexicons and template
nouns = Lexicon.from_jsonl(Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns")
verbs = Lexicon.from_jsonl(Path("lexicons/verbnet_verbs.jsonl"), "verbnet_verbs")
templates = TemplateCollection.from_jsonl(
    Path("templates/generic_frames.jsonl"), "generic_frames"
)
template = list(templates.templates)[0]

# Get sample items
noun_item = list(nouns.items)[0]
verb_item = list(verbs.items)[0]

renderer = DefaultRenderer()
text = renderer.render(
    template_string="{subj} {verb} {obj}",
    slot_fillers={"subj": noun_item, "verb": verb_item, "obj": noun_item},
    template_slots=template.slots,
)
print(f"Rendered: {text}")
```

**Custom renderer** (from gallery example):

```python
# gallery/eng/argument_structure/utils/renderers.py
from bead.templates.renderers import TemplateRenderer
from bead.resources.template import Slot
from bead.resources.lexical_item import LexicalItem


class OtherNounRenderer(TemplateRenderer):
    """Renderer with special handling for repeated nouns."""

    def render(
        self,
        template_string: str,
        slot_fillers: dict[str, LexicalItem],
        template_slots: dict[str, Slot],
    ) -> str:
        # Custom rendering logic for "another", "the other"
        # See gallery for full implementation
        pass
```

## Complete Example

From [gallery/eng/argument_structure/fill_templates.py](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/fill_templates.py):

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines, write_jsonlines
from bead.resources.lexicon import Lexicon
from bead.resources.template_collection import TemplateCollection
from bead.templates.filler import FilledTemplate
from bead.templates.renderers import DefaultRenderer
from bead.templates.strategies import ExhaustiveStrategy

# Load templates and lexicons
templates = TemplateCollection.from_jsonl(
    Path("templates/generic_frames.jsonl"), "generic_frames"
)
lexicons = {
    "nouns": Lexicon.from_jsonl(
        Path("lexicons/bleached_nouns.jsonl"), "bleached_nouns"
    ),
    "verbs": Lexicon.from_jsonl(Path("lexicons/verbnet_verbs.jsonl"), "verbnet_verbs"),
    "dets": Lexicon.from_jsonl(Path("lexicons/determiners.jsonl"), "determiners"),
}

# Get first template
template = list(templates.templates)[0]
print(f"Template: {template.name}")
print(f"Slots: {list(template.slots.keys())}")

# Create simple exhaustive strategy
strategy = ExhaustiveStrategy()

# Prepare slot items (limit to 2 items per slot for speed)
slot_items = {}
for slot_name, _slot in template.slots.items():
    # Map slot to appropriate lexicon based on constraints
    if "det" in slot_name.lower():
        items = list(lexicons["dets"].items)[:2]
    elif "noun" in slot_name.lower():
        items = list(lexicons["nouns"].items)[:2]
    elif "verb" in slot_name.lower():
        items = list(lexicons["verbs"].items)[:2]
    else:
        continue
    slot_items[slot_name] = items

# Generate combinations
combinations = list(strategy.generate_combinations(slot_items))[:5]

# Create FilledTemplate objects
renderer = DefaultRenderer()
filled_templates = []
for combo in combinations:
    rendered = renderer.render(
        template.template_string,
        combo,
        template.slots,
    )

    filled = FilledTemplate(
        template_id=str(template.id),
        template_name=template.name,
        slot_fillers=combo,
        rendered_text=rendered,
        strategy_name="exhaustive",
        template_slots={name: slot.required for name, slot in template.slots.items()},
    )
    filled_templates.append(filled)

print(f"Generated {len(filled_templates)} filled templates")

# Save example (commented out)
# write_jsonlines(filled_templates, Path("output/filled_templates.jsonl"))
```

## Strategy Selection Guide

| Strategy | Use When | Requires MLM |
|----------|----------|--------------|
| `ExhaustiveStrategy` | Small combinatorial spaces | No |
| `RandomStrategy` | Large spaces, need sampling | No |
| `StratifiedStrategy` | Need balanced feature distribution | No |
| `MLMFillingStrategy` | Want linguistically plausible fillers | Yes |
| `MixedFillingStrategy` | Different slots need different approaches | Mixed |

## Next Steps

- [Items module](items.md): Create experimental items from filled templates
- [CLI reference](../cli/templates.md): Command-line equivalents
- [Gallery example](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/fill_templates.py): Full working script
