#!/usr/bin/env python3
"""Extract generic frame templates from verb-specific VerbNet templates.

This script analyzes the 21,453 verb-specific templates and extracts the
26 unique structural patterns (template_strings). Each generic template
includes metadata about which VerbNet frames map to it.

Output: templates/generic_frames.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import layers_io
from utils.constraint_builder import (
    build_be_participle_constraint,
    build_be_subject_agreement_constraint,
)

from bead.cli.display import (
    confirm,
    console,
    create_progress,
    create_summary_table,
    print_error,
    print_header,
    print_info,
    print_success,
)
from bead.resources.constraints import Constraint
from bead.resources.template import Slot, Template


def create_progressive_variant(base_template: Template, tense: str) -> Template:
    """Create progressive variant of a template.

    Parameters
    ----------
    base_template : Template
        Base template with finite verb.
    tense : str
        "present" or "past" for the auxiliary.

    Returns
    -------
    Template
        Progressive variant with {be} and {verb} slots.
    """
    # Only create progressive for templates with {verb} slot
    if "verb" not in base_template.slots:
        return None

    # Create new template string by replacing {verb} with {be} {verb}
    new_template_string = base_template.template_string.replace("{verb}", "{be} {verb}")

    # Create new slots dict
    new_slots = dict(base_template.slots)

    # Add be slot with tense constraint
    if tense == "present":
        be_constraint_expr = (
            "self.lemma == 'be' and self.features.get('tense') == 'PRS'"
        )
        variant_name = "present_progressive"
        description_tense = "present progressive"
    else:  # past
        be_constraint_expr = (
            "self.lemma == 'be' and self.features.get('tense') == 'PST'"
        )
        variant_name = "past_progressive"
        description_tense = "past progressive"

    new_slots["be"] = Slot(
        name="be",
        description=f"Auxiliary 'be' ({tense})",
        constraints=[Constraint(expression=be_constraint_expr)],
        required=True,
    )

    # Update verb slot to require present participle
    verb_slot = new_slots["verb"]
    new_slots["verb"] = Slot(
        name=verb_slot.name,
        description=f"{verb_slot.description} (present participle)",
        constraints=[
            Constraint(expression="self.features.get('verb_form') == 'V.PTCP'"),
            Constraint(expression="self.features.get('tense') == 'PRS'"),
        ],
        required=verb_slot.required,
        default_value=verb_slot.default_value,
    )

    # Add be-participle constraint to ensure proper progressive structure
    progressive_constraints = []

    # Copy base constraints, but replace subject-verb agreement constraint
    # For progressive: check {be} agreement with {noun_subj}, not {verb} agreement
    has_subject_verb_agreement = False
    for constraint in base_template.constraints:
        # Identify subject-verb agreement constraint by checking description
        if (
            constraint.description
            and "Subject-verb" in constraint.description
            and "noun_subj" in constraint.description
            and "verb" in constraint.description
        ):
            has_subject_verb_agreement = True
            # Skip this constraint - we'll replace it with be agreement
            continue
        else:
            # Keep other constraints
            progressive_constraints.append(constraint)

    # Add be-participle constraint
    be_participle_constraint = build_be_participle_constraint("be", "verb")
    progressive_constraints.append(be_participle_constraint)

    # Add subject-be agreement for {be} with {noun_subj} if needed
    if has_subject_verb_agreement and "noun_subj" in new_slots:
        det_subj = "det_subj" if "det_subj" in new_slots else None
        if det_subj:
            be_agreement_constraint = build_be_subject_agreement_constraint(
                det_subj, "noun_subj", "be"
            )
            progressive_constraints.append(be_agreement_constraint)

    # Create new template
    progressive_template = Template(
        name=f"{base_template.name}_{variant_name}",
        template_string=new_template_string,
        slots=new_slots,
        constraints=progressive_constraints,
        description=f"{base_template.description} ({description_tense})",
        language_code=base_template.language_code,
        tags=base_template.tags + [variant_name, "progressive"],
        metadata={
            **base_template.metadata,
            "base_template_id": str(base_template.id),
            "tense_aspect_variant": variant_name,
            "is_progressive": True,
        },
    )

    return progressive_template


def main(
    input_file: str = "templates/verbnet_frames.jsonl",
    *,
    yes: bool = False,
) -> None:
    """Extract generic templates from verb-specific templates.

    Parameters
    ----------
    input_file : str
        Path to verb-specific templates file.
    yes : bool
        Skip confirmation prompts (for non-interactive use).
    """
    try:
        base_dir = Path(__file__).parent
        input_path = base_dir / input_file
        output_path = base_dir / "templates" / "generic_frames.jsonl"

        # Group templates by template_string
        template_groups: dict[str, list[dict]] = defaultdict(list)

        print_header("[1/4] Reading Verb-Specific Templates")

        with open(input_path) as f:
            for line in f:
                template = json.loads(line)
                template_string = template["template_string"]
                template_groups[template_string].append(template)

        total = sum(len(g) for g in template_groups.values())
        print_success(f"Loaded {total:,} verb-specific templates")
        print_success(f"Found {len(template_groups)} unique template structures")

        # Create generic templates
        print_header("[2/4] Creating Generic Templates")

        generic_templates = []

        for template_string, specific_templates in sorted(
            template_groups.items(), key=lambda x: -len(x[1])
        ):
            # Use first template as prototype
            prototype = specific_templates[0]

            # Collect all VerbNet frames that use this structure
            frame_primaries = set()
            verbnet_classes = set()
            example_verbs = []

            for spec_template in specific_templates:
                frame_primaries.add(spec_template["metadata"]["frame_primary"])
                verbnet_classes.add(spec_template["metadata"]["verbnet_class"])
                verb = spec_template["metadata"]["verb_lemma"]
                if len(example_verbs) < 10:  # Keep 10 example verbs
                    example_verbs.append(verb)

            # Create generic template name
            # Use first few words of template_string as name
            name_parts = template_string.replace("{", "").replace("}", "").split()[:5]
            template_name = "_".join(name_parts[:3])  # Use first 3 slots

            # Create generic template (reusing constraints from prototype)
            # Constraints are generic (reference slot names, not verb-specific items)
            generic_template = Template(
                name=template_name,
                template_string=template_string,
                slots=prototype["slots"],  # Reuse slots structure
                constraints=prototype.get(
                    "constraints", []
                ),  # Reuse multi-slot constraints
                description=f"Generic frame structure: {template_string}",
                language_code="eng",
                tags=["verbnet", "generic_frame"],
                metadata={
                    "template_structure": template_string,
                    "verb_count": len(specific_templates),
                    "frame_primaries": sorted(frame_primaries),
                    "verbnet_class_count": len(verbnet_classes),
                    "example_verbs": example_verbs,
                    "is_generic": True,
                },
            )

            generic_templates.append(generic_template)

            verb_count = len(specific_templates)
            frame_count = len(frame_primaries)
            console.print(
                f"  {len(generic_templates):2d}. {template_name:30s} "
                f"[{verb_count:5d} verbs, {frame_count:3d} frames]"
            )

        print_success(f"Created {len(generic_templates)} base templates")

        # Generate progressive variants
        print_header("[3/4] Generating Progressive Variants")

        progressive_templates = []
        with create_progress() as progress:
            task = progress.add_task(
                "Creating progressive variants...", total=len(generic_templates)
            )
            for base_template in generic_templates:
                # Create present progressive variant
                present_prog = create_progressive_variant(base_template, "present")
                if present_prog:
                    progressive_templates.append(present_prog)

                # Create past progressive variant
                past_prog = create_progressive_variant(base_template, "past")
                if past_prog:
                    progressive_templates.append(past_prog)

                progress.advance(task)

        print_success(f"Generated {len(progressive_templates)} progressive variants")

        # Combine base and progressive templates
        all_templates = generic_templates + progressive_templates

        # Save generic templates
        print_header("[4/4] Saving Templates")

        if output_path.exists() and not yes:
            if not confirm(f"Overwrite {output_path}?", default=False):
                print_info("Operation cancelled.")
                return

        with open(output_path, "w") as f:
            for template in all_templates:
                template_json = template.model_dump_json()
                f.write(template_json + "\n")

        print_success(f"Saved {len(all_templates)} templates to {output_path}")

        # also persist as layers resource templates
        layers_path = output_path.with_suffix(".layers.json")
        layers_io.write_templates_layers(list(all_templates), layers_path)
        print_success(f"Wrote layers templates to {layers_path}")

        # Summary statistics
        print_header("Summary")

        table = create_summary_table(
            {
                "Base generic frames": str(len(generic_templates)),
                "Progressive variants": str(len(progressive_templates)),
                "Total templates": str(len(all_templates)),
                "Cross-product size": f"~2,880 verbs x {len(all_templates)} templates",
                "Total combinations": f"~{2880 * len(all_templates):,}",
            }
        )
        console.print(table)

    except Exception as e:
        print_error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract generic templates from verb-specific templates"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="templates/verbnet_frames.jsonl",
        help="Input file with verb-specific templates",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (for non-interactive use)",
    )
    args = parser.parse_args()

    main(input_file=args.input, yes=args.yes)
