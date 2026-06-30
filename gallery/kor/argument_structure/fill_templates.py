#!/usr/bin/env python3
"""Fill templates using configuration-driven MLM strategy.

This script loads templates and lexicons, creates a TemplateFiller with
MixedFillingStrategy using slot_strategies from config.yaml, and outputs
filled templates.

All parameters are configurable via config.yaml, with optional CLI overrides.

Korean-specific rendering rules are applied:
- Nouns appearing twice use "다른"
- Nouns appearing three or more times use ordinals like "두번째", "세번째", etc.
"""

import argparse
import logging
from itertools import islice
from pathlib import Path

import yaml

from bead.data.serialization import write_jsonlines
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot
from bead.resources.template_collection import TemplateCollection
from bead.templates.adapters.cache import ModelOutputCache
from bead.templates.adapters.huggingface import HuggingFaceMLMAdapter
from bead.templates.filler import FilledTemplate
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import MixedFillingStrategy

logger = logging.getLogger(__name__)


def count_noun_occurrences(
    slot_fillers: dict[str, LexicalItem],
    template_slots: dict[str, Slot]
) -> dict[str, int]:
    """Count how many times each noun lemma appears."""
    noun_counts: dict[str, int] = {}

    for slot_name in template_slots:
        if slot_name.startswith('noun_') and slot_name in slot_fillers:
            noun_lemma = slot_fillers[slot_name].lemma
            noun_counts[noun_lemma] = noun_counts.get(noun_lemma, 0) + 1

    return noun_counts


def ordinal_word(n: int) -> str:
    """Convert number to ordinal word."""
    ordinals = {
        1: "첫번째", 2: "두번째", 3: "세번째", 4: "네번째", 5: "다섯번째",
        6: "여섯번째", 7: "일곱번째", 8: "여덟번째", 9: "아홉번째", 10: "열번째"
    }
    return ordinals.get(n, f"{n}번째")

# render template for korean specific so there is no need to handle a/an/the separately but just the noun occurrences
def render_template_kor(
    template_string: str,
    slot_fillers: dict[str, LexicalItem],
    template_slots: dict[str, Slot]
) -> str:
    """Render template with proper form selection and 'other' handling for Korean.

    Rules:
    - Uses item.form if available, otherwise item.lemma
    - If noun appears 2x: use "다른"
    - If noun appears 3+x: use ordinals "두번째", "세번째", etc.
    """
    # Count total occurrences of each noun
    noun_total_counts = count_noun_occurrences(slot_fillers, template_slots)

    # Identify noun slots in template order
    noun_slots_in_order = [
        name for name in template_slots.keys()
        if name.startswith('noun_')
    ]

    # Track current occurrence of each noun
    noun_usage: dict[str, int] = {}

    # Build result
    result = template_string

    for noun_slot in noun_slots_in_order:
        noun_item = slot_fillers[noun_slot]

        noun_surface = noun_item.form if noun_item.form else noun_item.lemma
        noun_lemma = noun_item.lemma

        occurrence = noun_usage.get(noun_lemma, 0)
        total_occurrences = noun_total_counts[noun_lemma]

        # Determine rendering
        if occurrence == 0:
            # First occurrence: use original noun
            rendered = f"{noun_surface}"
        elif total_occurrences == 2:
            # Exactly 2 occurrences: use "다른"
            rendered = f"다른 {noun_surface}"
        else:
            # 3+ occurrences: use ordinals
            ordinal = ordinal_word(occurrence + 1)
            rendered = f"{ordinal} {noun_surface}"

        # Replace noun slot
        pattern = f"{{{noun_slot}}}"
        result = result.replace(pattern, rendered, 1)

        noun_usage[noun_lemma] = occurrence + 1

    # Handle remaining slots (verbs, preps, etc.)
    for slot_name, item in slot_fillers.items():
        placeholder = f"{{{slot_name}}}"
        if placeholder in result:
            surface = item.form if item.form else item.lemma
            result = result.replace(placeholder, surface)

    return result

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Fill templates using config-driven MLM strategy."""
    parser = argparse.ArgumentParser(description="Fill templates with MLM strategy")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: use 10 verbs with one simple and one progressive template (both with 3+ nouns)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit verb lexicon to N unique lemmas (for quick test runs)",
    )
    parser.add_argument(
        "--max-per-template",
        type=int,
        default=None,
        help="Cap output to N filled templates per template (for test runs)",
    )
    parser.add_argument(
        "--skip-templates",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Template names to skip (e.g. subj-verb-that.)",
    )
    parser.add_argument(
        "--only-templates",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Only fill these templates (e.g. subj_nom-verb. subj_nom-obj_acc-verb.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path from config",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
    )

    logger.info("Loading templates and lexicons...")

    # Resolve paths
    templates_path = Path(config["resources"]["templates"][0]["path"])
    output_path = args.output or Path(config["template"]["output_path"])

    # Load templates
    template_collection = TemplateCollection.from_jsonl(
        templates_path, "generic_frames"
    )
    logger.info(
        f"Loaded {len(template_collection.templates)} templates from {templates_path}"
    )

    # Apply dry-run mode: select specific templates
    templates = list(template_collection.templates)
    if args.dry_run:
        logger.info("DRY RUN MODE: Selecting 1 simple + 1 progressive template with 3 noun slots")

        # Find templates with 3 noun slots
        def count_noun_slots(template):
            return sum(1 for slot_name in template.slots if slot_name.startswith('noun_'))

        templates_with_3_nouns = [t for t in templates if count_noun_slots(t) == 3]

        # Separate simple and progressive
        simple_templates = [t for t in templates_with_3_nouns if 'progressive' not in t.name]
        progressive_templates = [t for t in templates_with_3_nouns if 'progressive' in t.name]

        # Select one of each
        selected = []
        if simple_templates:
            selected.append(simple_templates[0])
            logger.info(f"  Simple template: {simple_templates[0].name}")
        if progressive_templates:
            selected.append(progressive_templates[0])
            logger.info(f"  Progressive template: {progressive_templates[0].name}")

        templates = selected if selected else templates[:2]
        logger.info(f"Selected {len(templates)} templates for dry run")

    # Load lexicons
    lexicons: list[Lexicon] = []

    for lex_config in config["resources"]["lexicons"]:
        lex_path = Path(lex_config["path"])
        lexicon = Lexicon.from_jsonl(lex_path, lex_config["name"])

        # Determine verb limit: dry-run uses 10, --limit uses the given value
        verb_limit = None
        if args.dry_run and lex_config["name"] == "verbs":
            verb_limit = 10
        elif args.limit is not None and lex_config["name"] == "verbs":
            verb_limit = args.limit

        if verb_limit is not None:
            unique_lemmas: list[str] = []
            for item in lexicon.items:
                if item.lemma not in unique_lemmas:
                    unique_lemmas.append(item.lemma)
                if len(unique_lemmas) == verb_limit:
                    break
            allowed = set(unique_lemmas)
            lexicon = lexicon.filter(lambda item, _a=allowed: item.lemma in _a)
            label = "DRY RUN" if args.dry_run else "LIMIT"
            logger.info(
                f"{label}: Limited verbs to {verb_limit} lemmas ({len(lexicon.items)} forms)"
            )

        lexicons.append(lexicon)
        logger.info(f"Loaded {len(lexicon.items)} items from {lex_config['name']}")

    # Initialize constraint resolver
    resolver = ConstraintResolver()

    # Initialize model adapter for MLM
    mlm_config = config["template"]["mlm"]
    logger.info(f"Loading MLM model: {mlm_config['model_name']}...")
    model_adapter = HuggingFaceMLMAdapter(
        model_name=mlm_config["model_name"],
        device=mlm_config.get("device", "cpu"),
    )
    model_adapter.load_model()
    logger.info("MLM model loaded successfully")

    # Initialize cache
    cache_dir = Path(config["paths"]["cache_dir"])
    cache = ModelOutputCache(cache_dir=cache_dir)

    # Build slot_strategies dict for MixedFillingStrategy
    # Format: {slot_name: (strategy_name, config_dict)}
    slot_strategies: dict[str, tuple[str, dict]] = {}

    for slot_name, slot_config in config["template"]["slot_strategies"].items():
        strategy_name = slot_config["strategy"]

        if strategy_name == "mlm":
            # MLM strategy needs special config with resolver, model_adapter, etc.
            mlm_slot_config = {
                "resolver": resolver,
                "model_adapter": model_adapter,
                "cache": cache,
                "beam_size": mlm_config.get("beam_size", 5),
                "top_k": mlm_config.get("top_k", 10),
            }
            # Add per-slot max_fills and enforce_unique if specified
            if "max_fills" in slot_config:
                mlm_slot_config["max_fills"] = slot_config["max_fills"]
            if "enforce_unique" in slot_config:
                mlm_slot_config["enforce_unique"] = slot_config["enforce_unique"]

            slot_strategies[slot_name] = ("mlm", mlm_slot_config)
        else:
            # For other strategies (exhaustive, random, etc.)
            slot_strategies[slot_name] = (strategy_name, {})

    # Create filler with MixedFillingStrategy
    logger.info("Creating template filler with mixed strategy...")
    strategy = MixedFillingStrategy(
        slot_strategies=slot_strategies,
    )

    # Fill templates
    logger.info("Filling templates...")
    filled_templates = []
    for i, template in enumerate(templates, 1):
        if args.only_templates and template.name not in args.only_templates:
            continue
        if template.name in args.skip_templates:
            logger.info(f"Skipping template {i}/{len(templates)}: {template.name}")
            continue
        logger.info(f"Filling template {i}/{len(templates)}: {template.name}")
        try:
            gen = strategy.generate_from_template(
                template=template, lexicons=lexicons, language_code="kor"
            )
            if args.max_per_template is not None:
                combos = list(islice(gen, args.max_per_template))
            else:
                combos = list(gen)

            # Convert combinations to FilledTemplate objects
            for combo in combos:
                # Render text with smart ordinal handling
                rendered = render_template_kor(
                    template.template_string,
                    combo,
                    template.slots
                )

                filled = FilledTemplate(
                    template_id=str(template.id),
                    template_name=template.name,
                    slot_fillers=combo,
                    rendered_text=rendered,
                    strategy_name="mixed",
                    template_slots={
                        name: slot.required for name, slot in template.slots.items()
                    },
                )
                filled_templates.append(filled)

            logger.info(f"  Generated {len(combos)} filled templates")
        except Exception as e:
            logger.error(f"  Failed to fill template {template.name}: {e}")
            continue

    logger.info(f"Total filled templates: {len(filled_templates)}")

    # Save filled templates
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonlines(filled_templates, output_path)
    logger.info(f"Saved filled templates to {output_path}")


if __name__ == "__main__":
    main()
