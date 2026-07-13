#!/usr/bin/env python3
"""Generate JSONL lexicon files for the argument structure alternations dataset.

This script creates all required lexicons using the bead adapter infrastructure:
1. verbnet_verbs.jsonl - All VerbNet verbs with inflected forms
2. bleached_nouns.jsonl - Controlled noun inventory from CSV
3. bleached_verbs.jsonl - Controlled verb inventory from CSV
4. bleached_adjectives.jsonl - Controlled adjective inventory from CSV
5. prepositions.jsonl - Comprehensive English preposition list
6. determiners.jsonl - Basic determiner inventory [a, the, some]
"""

import argparse
import sys
from pathlib import Path
from uuid import UUID

import layers_io
import pandas as pd
from utils.morphology import MorphologyExtractor
from utils.verbnet_parser import VerbNetExtractor

from bead.cli.display import (
    console,
    create_progress,
    create_summary_table,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.resources.adapters.cache import AdapterCache
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.loaders import from_csv  # Use bead loader utilities


def main(verb_limit: int | None = None) -> None:
    """Generate lexicons for argument structure experiment."""
    try:
        # set up paths
        base_dir = Path(__file__).parent
        lexicons_dir = base_dir / "lexicons"
        resources_dir = base_dir / "resources"

        # ensure directories exist
        lexicons_dir.mkdir(exist_ok=True)

        # initialize adapters with caching
        cache = AdapterCache()
        verbnet = VerbNetExtractor(cache=cache)
        morph = MorphologyExtractor(cache=cache)

        # 1. generate VerbNet verbs lexicon
        print_header("Generating VerbNet Verbs Lexicon")
        print_info("Extracting VerbNet verbs...")

        verb_items_dict: dict[UUID, LexicalItem] = {}
        base_verbs = verbnet.extract_all_verbs()

        print_success(f"Found {len(base_verbs):,} verb-class pairs from VerbNet")

        # apply limit if specified
        if verb_limit is not None:
            print_warning(f"[TEST MODE] Limiting to first {verb_limit} verbs")
            base_verbs = base_verbs[:verb_limit]

        print_info(f"Getting inflected forms for {len(base_verbs):,} verbs...")

        with create_progress() as progress:
            task = progress.add_task("Processing verbs...", total=len(base_verbs))
            for base_verb in base_verbs:
                lemma = base_verb.lemma

                # get all inflected forms
                forms = morph.get_all_required_forms(lemma)

                # add VerbNet metadata to each form
                for form_item in forms:
                    form_item.features.update(
                        {
                            "verbnet_class": base_verb.features.get(
                                "verbnet_class", ""
                            ),
                            "themroles": base_verb.features.get("themroles", []),
                            "frame_count": base_verb.features.get("frame_count", 0),
                        }
                    )

                    # use LexicalItem's UUID as key
                    verb_items_dict[form_item.id] = form_item

                progress.advance(task)

        print_success(f"Created {len(verb_items_dict):,} verb form entries")

        verb_lexicon = Lexicon(
            name="verbnet_verbs",
            description="All VerbNet verbs with inflected forms",
            language_code="eng",
            items=verb_items_dict,
        )

        output_path = lexicons_dir / "verbnet_verbs.jsonl"
        layers_io.save_lexicon(verb_lexicon, output_path)
        print_success(f"Saved to {output_path}")

        # 2. generate bleached nouns lexicon
        print_header("Generating Bleached Nouns Lexicon")

        csv_path = resources_dir / "bleached_nouns.csv"

        noun_lexicon = from_csv(
            path=csv_path,
            name="bleached_nouns",
            column_mapping={"word": "lemma"},
            feature_columns=["number", "countability", "semantic_class"],
            language_code="eng",
            description="Controlled noun inventory for templates",
            pos="NOUN",
        )

        print_success(
            f"Loaded {len(noun_lexicon.items):,} bleached nouns from {csv_path}"
        )

        output_path = lexicons_dir / "bleached_nouns.jsonl"
        layers_io.save_lexicon(noun_lexicon, output_path)
        print_success(f"Saved to {output_path}")

        # 3. generate bleached verbs lexicon
        print_header("Generating Bleached Verbs Lexicon")

        csv_path = resources_dir / "bleached_verbs.csv"

        bleached_verb_lexicon = from_csv(
            path=csv_path,
            name="bleached_verbs",
            column_mapping={"word": "lemma"},
            feature_columns=["tense", "semantic_class"],
            language_code="eng",
            description="Controlled verb inventory for templates",
            pos="V",
        )

        verb_count = len(bleached_verb_lexicon.items)
        print_success(f"Loaded {verb_count:,} bleached verbs from {csv_path}")

        output_path = lexicons_dir / "bleached_verbs.jsonl"
        layers_io.save_lexicon(bleached_verb_lexicon, output_path)
        print_success(f"Saved to {output_path}")

        # 4. generate bleached adjectives lexicon
        print_header("Generating Bleached Adjectives Lexicon")

        csv_path = resources_dir / "bleached_adjectives.csv"

        adj_lexicon = from_csv(
            path=csv_path,
            name="bleached_adjectives",
            column_mapping={"word": "lemma"},
            feature_columns=["semantic_class"],
            language_code="eng",
            description="Controlled adjective inventory for templates",
            pos="ADJ",
        )

        print_success(
            f"Loaded {len(adj_lexicon.items):,} bleached adjectives from {csv_path}"
        )

        output_path = lexicons_dir / "bleached_adjectives.jsonl"
        layers_io.save_lexicon(adj_lexicon, output_path)
        print_success(f"Saved to {output_path}")

        # 5. generate prepositions lexicon
        print_header("Generating Prepositions Lexicon")

        csv_path = resources_dir / "prepositions.csv"

        prep_lexicon = from_csv(
            path=csv_path,
            name="prepositions",
            column_mapping={"lemma": "lemma"},
            feature_columns=["pos"],
            language_code="eng",
            description="Comprehensive English preposition inventory",
            pos="ADP",
        )

        print_success(
            f"Loaded {len(prep_lexicon.items):,} prepositions from {csv_path}"
        )

        output_path = lexicons_dir / "prepositions.jsonl"
        layers_io.save_lexicon(prep_lexicon, output_path)
        print_success(f"Saved to {output_path}")

        # 6. generate determiners lexicon
        print_header("Generating Determiners Lexicon")

        csv_path = resources_dir / "determiners.csv"

        det_lexicon = from_csv(
            path=csv_path,
            name="determiners",
            column_mapping={"lemma": "lemma"},
            feature_columns=["pos"],
            language_code="eng",
            description="Basic determiner inventory",
            pos="DET",
        )

        print_success(f"Loaded {len(det_lexicon.items):,} determiners from {csv_path}")

        output_path = lexicons_dir / "determiners.jsonl"
        layers_io.save_lexicon(det_lexicon, output_path)
        print_success(f"Saved to {output_path}")

        # 7. generate "be" verb forms lexicon
        print_header("Generating Be Verb Lexicon")

        # load from CSV (with custom handling for form column)
        csv_path = resources_dir / "be_forms.csv"
        df = pd.read_csv(csv_path)

        be_items: dict[UUID, LexicalItem] = {}
        for _, row in df.iterrows():
            # build features dict from all columns except lemma and form
            features = {"pos": str(row["pos"])}

            # add optional feature columns if not empty
            for col in ["tense", "person", "number", "verb_form"]:
                if col in df.columns and pd.notna(row[col]) and str(row[col]).strip():
                    features[col] = str(row[col])

            item = LexicalItem(
                lemma=str(row["lemma"]),
                form=str(row["form"]),
                language_code="eng",
                features=features,
                source="csv",
            )
            be_items[item.id] = item

        print_success(f"Loaded {len(be_items):,} forms of 'be' from {csv_path}")

        be_lexicon = Lexicon(
            name="be_forms",
            description="Inflected forms of auxiliary 'be'",
            language_code="eng",
            items=be_items,
        )

        output_path = lexicons_dir / "be_forms.jsonl"
        layers_io.save_lexicon(be_lexicon, output_path)
        print_success(f"Saved to {output_path}")

        # summary
        print_header("Lexicon Generation Complete")

        table = create_summary_table(
            {
                "verbnet_verbs.jsonl": f"{len(verb_items_dict):,} entries",
                "bleached_nouns.jsonl": f"{len(noun_lexicon.items):,} entries",
                "bleached_verbs.jsonl": f"{len(bleached_verb_lexicon.items):,} entries",
                "bleached_adjectives.jsonl": f"{len(adj_lexicon.items):,} entries",
                "prepositions.jsonl": f"{len(prep_lexicon.items):,} entries",
                "determiners.jsonl": f"{len(det_lexicon.items):,} entries",
                "be_forms.jsonl": f"{len(be_items):,} entries",
            },
            title="Generated Lexicons",
        )
        console.print(table)
        print_info(f"All files saved to: {lexicons_dir}/")

    except Exception as e:
        print_error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JSONL lexicon files for argument structure dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of VerbNet verbs to process (for testing)",
    )
    args = parser.parse_args()

    main(verb_limit=args.limit)
