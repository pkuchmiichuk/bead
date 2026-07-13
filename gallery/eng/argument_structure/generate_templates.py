#!/usr/bin/env python3
"""Generate Template objects from VerbNet frames.

Output: templates/verbnet_frames.jsonl
"""

import argparse
import sys
from pathlib import Path

import layers_io
from utils.template_generator import generate_templates_for_all_verbs
from utils.verbnet_parser import VerbNetExtractor

from bead.cli.display import (
    confirm,
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


def main(verb_limit: int | None = None, *, yes: bool = False) -> None:
    """Generate and save templates from VerbNet frames.

    Parameters
    ----------
    verb_limit : int | None
        Limit number of verb-class pairs to process (for testing).
    yes : bool
        Skip confirmation prompts (for non-interactive use).
    """
    try:
        # set up paths
        base_dir = Path(__file__).parent
        templates_dir = base_dir / "templates"
        templates_dir.mkdir(exist_ok=True)

        # initialize with caching
        cache = AdapterCache()
        extractor = VerbNetExtractor(cache=cache)

        # get all verbs with detailed frame information
        print_header("Extracting VerbNet Verbs with Frames")
        verbs_with_frames = extractor.extract_all_verbs_with_frames()
        print_success(f"Found {len(verbs_with_frames):,} verb-class pairs")

        # apply limit if specified
        if verb_limit is not None:
            print_warning(
                f"[TEST MODE] Limiting to first {verb_limit} verb-class pairs"
            )
            verbs_with_frames = verbs_with_frames[:verb_limit]

        # generate templates for all verbs
        print_header("Generating Templates")
        print_info(f"Processing {len(verbs_with_frames):,} verb-class pairs...")

        templates = generate_templates_for_all_verbs(verbs_with_frames)
        print_success(f"Generated {len(templates):,} templates")

        # save templates to JSONL
        print_header("Saving Templates")
        output_path = templates_dir / "verbnet_frames.jsonl"

        if output_path.exists() and not yes:
            if not confirm(f"Overwrite {output_path}?", default=False):
                print_info("Operation cancelled.")
                return

        with create_progress() as progress:
            task = progress.add_task("Writing templates...", total=len(templates))
            with open(output_path, "w") as f:
                for template in templates:
                    # Convert Template to JSON string (handles UUID serialization)
                    template_json = template.model_dump_json()
                    f.write(template_json + "\n")
                    progress.advance(task)

        print_success(f"Saved {len(templates):,} templates to {output_path}")

        # also persist as layers resource templates
        layers_path = output_path.with_suffix(".layers.json")
        layers_io.write_templates_layers(list(templates), layers_path)
        print_success(f"Wrote layers templates to {layers_path}")

        # summary statistics
        print_header("Template Generation Complete")

        summary_data = {
            "Verb-class pairs processed": f"{len(verbs_with_frames):,}",
            "Templates generated": f"{len(templates):,}",
        }
        if verbs_with_frames:
            avg_templates = len(templates) / len(verbs_with_frames)
            summary_data["Avg templates per verb-class"] = f"{avg_templates:.2f}"

        table = create_summary_table(summary_data)
        console.print(table)

    except Exception as e:
        print_error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Template objects from VerbNet frames"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of verb-class pairs to process (for testing)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (for non-interactive use)",
    )
    args = parser.parse_args()

    main(verb_limit=args.limit, yes=args.yes)
