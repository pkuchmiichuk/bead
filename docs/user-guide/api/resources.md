# Resources Module

The `bead.resources` module provides lexicons, templates, and adapters for external linguistic databases.

This guide walks through the EXACT workflow from [gallery/eng/argument_structure/generate_lexicons.py](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/generate_lexicons.py), showing how to extract VerbNet verbs, get morphological forms, and load controlled lexicons from CSV.

## Complete Lexicon Generation Workflow

The gallery example generates 7 lexicons. Here's the complete workflow with 3 verb classes:

```python
from pathlib import Path

from bead.resources.adapters.cache import AdapterCache
from bead.resources.lexicon import Lexicon
from bead.resources.loaders import from_csv

# Set up paths (adapted from generate_lexicons.py lines 27-33)
# Note: tests cd to fixtures dir, so paths are relative to tests/fixtures/api_docs/
base_dir = Path(".")
lexicons_dir = base_dir / "lexicons"
resources_dir = base_dir / "resources"

# Ensure directories exist
lexicons_dir.mkdir(exist_ok=True)

# Initialize adapter cache for VerbNet and UniMorph
cache = AdapterCache()
```

## Extracting VerbNet Verbs

Use the gallery's `VerbNetExtractor` to fetch verbs with frame information:

```python
from pathlib import Path

from utils.morphology import MorphologyExtractor
from utils.verbnet_parser import VerbNetExtractor

from bead.resources.adapters.cache import AdapterCache
from bead.resources.lexical_item import LexicalItem

# Set up cache and paths
cache = AdapterCache()
base_dir = Path(".")
lexicons_dir = base_dir / "lexicons"

# Initialize extractors with caching (lines 36-38)
verbnet = VerbNetExtractor(cache=cache)
morph = MorphologyExtractor(cache=cache)

# Extract all VerbNet verbs (line 47)
base_verbs = verbnet.extract_all_verbs()
print(f"Found {len(base_verbs)} verb-class pairs from VerbNet")

# Limit to 3 verbs for testing (lines 52-54)
base_verbs = base_verbs[:3]
print(f"Using first {len(base_verbs)} verbs")

# Get inflected forms for each verb (lines 58-78)
verb_items_dict: dict[str, LexicalItem] = {}

for base_verb in base_verbs:
    lemma = base_verb.lemma
    print(f"Processing {lemma}...")

    # Get all inflected forms (base, 3sg, past, progressive, past participle)
    forms = morph.get_all_required_forms(lemma)

    # Add VerbNet metadata to each form
    for form_item in forms:
        form_item.features.update(
            {
                "verbnet_class": base_verb.features.get("verbnet_class", ""),
                "themroles": base_verb.features.get("themroles", []),
                "frame_count": base_verb.features.get("frame_count", 0),
            }
        )
        verb_items_dict[str(form_item.id)] = form_item

print(f"Created {len(verb_items_dict)} verb form entries")
```

## Creating and Saving Lexicons

Create `Lexicon` objects and save to JSONL:

```python
from pathlib import Path

from utils.morphology import MorphologyExtractor
from utils.verbnet_parser import VerbNetExtractor

from bead.resources.adapters.cache import AdapterCache
from bead.resources.lexicon import Lexicon

# Set up cache and paths
cache = AdapterCache()
base_dir = Path(".")
lexicons_dir = base_dir / "lexicons"

# Extract and process verbs (abbreviated version)
verbnet = VerbNetExtractor(cache=cache)
morph = MorphologyExtractor(cache=cache)
base_verbs = verbnet.extract_all_verbs()[:3]

verb_items_dict = {}
for base_verb in base_verbs:
    forms = morph.get_all_required_forms(base_verb.lemma)
    for form_item in forms:
        form_item.features.update(
            {
                "verbnet_class": base_verb.features.get("verbnet_class", ""),
                "themroles": base_verb.features.get("themroles", []),
                "frame_count": base_verb.features.get("frame_count", 0),
            }
        )
        verb_items_dict[str(form_item.id)] = form_item

# Create VerbNet verbs lexicon (lines 82-91)
verb_lexicon = Lexicon(
    name="verbnet_verbs",
    description="All VerbNet verbs with inflected forms",
    language_code="eng",
    items=tuple(verb_items_dict.values()),
)

output_path = lexicons_dir / "verbnet_verbs.jsonl"
verb_lexicon.to_jsonl(str(output_path))
print(f"Saved to {output_path}")
```

## Loading Lexicons from CSV

Load controlled lexicons from CSV files (lines 100-114):

```python
from pathlib import Path

from bead.resources.loaders import from_csv

# Set up paths
base_dir = Path(".")
lexicons_dir = base_dir / "lexicons"
resources_dir = base_dir / "resources"

# Load bleached nouns from CSV
csv_path = resources_dir / "bleached_nouns.csv"

noun_lexicon = from_csv(
    path=csv_path,
    name="bleached_nouns",
    feature_columns=["number", "countability", "semantic_class"],
    language_code="eng",
    description="Controlled noun inventory for templates",
    pos="NOUN",
)

print(f"Loaded {len(noun_lexicon.items)} bleached nouns")

# Save to JSONL
output_path = lexicons_dir / "bleached_nouns.jsonl"
noun_lexicon.to_jsonl(str(output_path))
```

**CSV format** (`resources/bleached_nouns.csv`):

```csv
word,number,countability,semantic_class
person,singular,count,animate
people,plural,count,animate
thing,singular,count,inanimate
things,plural,count,inanimate
place,singular,count,location
```

## Using Gallery Morphology Extractor

The gallery's `MorphologyExtractor` wraps `UniMorphAdapter` to get all required verb forms:

```python
from utils.morphology import MorphologyExtractor

from bead.resources.adapters.cache import AdapterCache

# Initialize morphology extractor with cache
cache = AdapterCache()
morph = MorphologyExtractor(cache=cache)

# All required forms: base, 3sg present, past, progressive, past participle
forms = morph.get_all_required_forms("annihilate")

for form in forms:
    print(f"{form.form} - {form.features}")

# Output:
# annihilate - {'pos': 'V', 'tense': 'base'}
# annihilates - {'pos': 'V', 'tense': 'present', 'person': '3', 'number': 'sg'}
# annihilated - {'pos': 'V', 'tense': 'past'}
# annihilating - {'pos': 'V', 'aspect': 'progressive'}
# annihilated - {'pos': 'V', 'aspect': 'perfect'}
```

See [gallery/eng/argument_structure/utils/morphology.py](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/utils/morphology.py:78-138) for implementation.

## Loading Existing Lexicons

Load previously generated lexicons from JSONL:

```python
from pathlib import Path

from bead.resources.lexicon import Lexicon

# Set up paths
base_dir = Path(".")
lexicons_dir = base_dir / "lexicons"

# Load VerbNet verbs lexicon
verb_lexicon_path = lexicons_dir / "verbnet_verbs.jsonl"
verb_lexicon = Lexicon.from_jsonl(verb_lexicon_path, "verbnet_verbs")

print(f"Loaded {len(verb_lexicon.items)} verb forms")

# Access specific items
for item in list(verb_lexicon.items)[:3]:
    print(f"{item.lemma} → {item.form}")
    print(f"  VerbNet class: {item.features.get('verbnet_class')}")
    print(f"  Thematic roles: {item.features.get('themroles', [])}")
```

## Next Steps

For template generation and filling, see:
- [Templates module](templates.md): Generate VerbNet templates and fill with lexicons
- [Gallery example](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/generate_lexicons.py): Full working script

