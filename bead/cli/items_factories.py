"""Task-type-specific item creation commands for bead CLI.

This module provides CLI commands for creating experimental items for all 8 supported
task types. Each task type has specialized creation functions that wrap the core
utilities in bead.items.

Supported task types:
- forced_choice: N-AFC (2AFC, 3AFC, etc.)
- ordinal_scale: Likert scales, sliders
- categorical: Unordered categories (NLI, semantic relations)
- binary: Yes/No, True/False
- multi_select: Multiple selection (checkboxes)
- magnitude: Unbounded numeric (reading time, confidence)
- free_text: Open-ended text responses
- cloze: Fill-in-the-blank
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import click
from rich.console import Console

from bead.cli.display import (
    create_progress,
    display_file_stats,
    print_error,
    print_info,
    print_success,
)
from bead.cli.utils import parse_key_value_pairs
from bead.items.binary import create_binary_items_from_texts
from bead.items.categorical import (
    create_categorical_item,
    create_nli_item,
)
from bead.items.cloze import create_simple_cloze_item
from bead.items.forced_choice import create_forced_choice_item
from bead.items.free_text import (
    create_free_text_items_from_texts,
)
from bead.items.item import Item, MetadataValue
from bead.items.magnitude import (
    create_magnitude_items_from_texts,
)
from bead.items.multi_select import create_multi_select_item
from bead.items.ordinal_scale import (
    create_likert_7_item,
    create_ordinal_scale_items_from_texts,
)

console = Console()


# ==================== Forced Choice Commands ====================


@click.command()
@click.argument("options", nargs=-1, required=True)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--metadata",
    type=str,
    help="Metadata as key=value pairs (comma-separated)",
)
def create_forced_choice(
    options: tuple[str, ...],
    output: Path,
    metadata: str | None,
) -> None:
    r"""Create a single forced-choice item from options.

    Examples
    --------
    $ bead items create-forced-choice "Option A" "Option B" -o item.jsonl

    $ bead items create-forced-choice "The cat" "The dog" "The bird" \\
        --metadata "contrast=subject" -o item.jsonl
    """
    try:
        if len(options) < 2:
            print_error("At least 2 options required")
            return

        # Parse metadata
        meta_dict_str: dict[str, str] = (
            parse_key_value_pairs(metadata) if metadata else {}
        )
        # Cast to MetadataValue dict (str is a valid MetadataValue)
        meta_dict: dict[str, MetadataValue] = dict(meta_dict_str)

        # Create item
        item: Item = create_forced_choice_item(*options, metadata=meta_dict)

        # Save
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(item.model_dump_json() + "\n")

        print_success(f"Created forced-choice item: {output}")

    except Exception as e:
        print_error(f"Failed to create forced-choice item: {e}")


@click.command()
@click.option(
    "--texts-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="File with text options (one per line)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--n-alternatives",
    type=int,
    default=2,
    help="Number of alternatives per item (default: 2)",
)
@click.option(
    "--sample",
    type=int,
    help="Sample N items randomly",
)
def create_forced_choice_from_texts(
    texts_file: Path,
    output: Path,
    n_alternatives: int,
    sample: int | None,
) -> None:
    r"""Create forced-choice items from text file.

    Examples
    --------
    $ bead items create-forced-choice-from-texts \\
        --texts-file sentences.txt --output items.jsonl

    $ bead items create-forced-choice-from-texts \\
        --texts-file sentences.txt --n-alternatives 3 \\
        --sample 100 --output items.jsonl
    """
    try:
        # Load texts
        texts: list[str] = [
            line.strip() for line in texts_file.read_text().splitlines() if line.strip()
        ]
        print_info(f"Loaded {len(texts)} texts")

        # Create items by generating all combinations of n_alternatives from texts
        items: list[Item] = []
        for combination in itertools.combinations(texts, n_alternatives):
            item: Item = create_forced_choice_item(*combination)
            items.append(item)

        print_info(f"Created {len(items)} forced-choice items")

        # Sample if requested
        if sample and sample < len(items):
            items = random.sample(items, sample)
            print_info(f"Sampled {sample} items")

        # Save
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

        display_file_stats(output, len(items), "forced-choice items")

    except Exception as e:
        print_error(f"Failed to create forced-choice items: {e}")


# ==================== Ordinal Scale Commands ====================


@click.command()
@click.option(
    "--text",
    type=str,
    required=True,
    help="Text to rate",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--prompt",
    type=str,
    default="Rate this item:",
    help="Rating prompt",
)
def create_likert_7(
    text: str,
    output: Path,
    prompt: str,
) -> None:
    r"""Create a 7-point Likert scale item.

    Examples
    --------
    $ bead items create-likert-7 --text "The cat sat on the mat" -o item.jsonl

    $ bead items create-likert-7 --text "Sentence text" \\
        --prompt "How natural is this?" -o item.jsonl
    """
    try:
        item = create_likert_7_item(text, prompt=prompt)

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(item.model_dump_json() + "\n")

        print_success(f"Created Likert-7 item: {output}")

    except Exception as e:
        print_error(f"Failed to create Likert-7 item: {e}")


@click.command()
@click.option(
    "--texts-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="File with texts to rate (one per line)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--scale-min",
    type=int,
    default=1,
    help="Minimum scale value (default: 1)",
)
@click.option(
    "--scale-max",
    type=int,
    default=7,
    help="Maximum scale value (default: 7)",
)
@click.option(
    "--prompt",
    type=str,
    default="Rate this item:",
    help="Rating prompt",
)
def create_ordinal_scale_from_texts(
    texts_file: Path,
    output: Path,
    scale_min: int,
    scale_max: int,
    prompt: str,
) -> None:
    r"""Create ordinal scale items from text file.

    Examples
    --------
    $ bead items create-ordinal-scale-from-texts \\
        --texts-file sentences.txt --output items.jsonl

    $ bead items create-ordinal-scale-from-texts \\
        --texts-file sentences.txt --scale-min 1 --scale-max 5 \\
        --prompt "How acceptable?" --output items.jsonl
    """
    try:
        # Load texts
        texts = [
            line.strip() for line in texts_file.read_text().splitlines() if line.strip()
        ]
        print_info(f"Loaded {len(texts)} texts")

        # Create items
        with create_progress() as progress:
            task = progress.add_task(
                "Creating ordinal scale items...", total=len(texts)
            )
            from bead.items.item_template import ScaleBounds  # noqa: PLC0415

            items = create_ordinal_scale_items_from_texts(
                texts,
                scale_bounds=ScaleBounds(min=scale_min, max=scale_max),
                prompt=prompt,
            )
            progress.update(task, completed=len(texts))

        # Save
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

        display_file_stats(output, len(items), "ordinal scale items")

    except Exception as e:
        print_error(f"Failed to create ordinal scale items: {e}")


# ==================== Categorical Commands ====================


@click.command()
@click.option(
    "--premise",
    type=str,
    required=True,
    help="Premise text",
)
@click.option(
    "--hypothesis",
    type=str,
    required=True,
    help="Hypothesis text",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
def create_nli(
    premise: str,
    hypothesis: str,
    output: Path,
) -> None:
    r"""Create an NLI (natural language inference) item.

    Examples
    --------
    $ bead items create-nli \\
        --premise "All dogs bark" \\
        --hypothesis "Some dogs bark" \\
        -o item.jsonl
    """
    try:
        item = create_nli_item(premise, hypothesis)

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(item.model_dump_json() + "\n")

        print_success(f"Created NLI item: {output}")

    except Exception as e:
        print_error(f"Failed to create NLI item: {e}")


@click.command()
@click.option(
    "--text",
    type=str,
    required=True,
    help="Text for categorization",
)
@click.option(
    "--categories",
    type=str,
    required=True,
    help="Categories (comma-separated)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--prompt",
    type=str,
    default="Categorize this item:",
    help="Categorization prompt",
)
def create_categorical(
    text: str,
    categories: str,
    output: Path,
    prompt: str,
) -> None:
    r"""Create a categorical item.

    Examples
    --------
    $ bead items create-categorical --text "Example text" \\
        --categories "entailment,contradiction,neutral" -o item.jsonl
    """
    try:
        cat_list = [c.strip() for c in categories.split(",")]
        item = create_categorical_item(text, categories=cat_list, prompt=prompt)

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(item.model_dump_json() + "\n")

        print_success(f"Created categorical item: {output}")

    except Exception as e:
        print_error(f"Failed to create categorical item: {e}")


# ==================== Binary Commands ====================


@click.command()
@click.option(
    "--texts-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="File with texts (one per line)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--prompt",
    type=str,
    default="Is this acceptable?",
    help="Binary judgment prompt",
)
def create_binary_from_texts(
    texts_file: Path,
    output: Path,
    prompt: str,
) -> None:
    r"""Create binary judgment items from text file.

    Examples
    --------
    $ bead items create-binary-from-texts \\
        --texts-file sentences.txt --output items.jsonl

    $ bead items create-binary-from-texts \\
        --texts-file sentences.txt \\
        --prompt "Is this grammatical?" --output items.jsonl
    """
    try:
        # Load texts
        texts = [
            line.strip() for line in texts_file.read_text().splitlines() if line.strip()
        ]
        print_info(f"Loaded {len(texts)} texts")

        # Create items
        items = create_binary_items_from_texts(texts, prompt=prompt)

        # Save
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

        display_file_stats(output, len(items), "binary items")

    except Exception as e:
        print_error(f"Failed to create binary items: {e}")


# ==================== Multi-Select Commands ====================


@click.command()
@click.option(
    "--texts-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="File with texts (one per line)",
)
@click.option(
    "--options",
    type=str,
    required=True,
    help="Options (comma-separated)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--min-selections",
    type=int,
    default=1,
    help="Minimum selections (default: 1)",
)
@click.option(
    "--max-selections",
    type=int,
    help="Maximum selections (default: all)",
)
def create_multi_select_from_texts(
    texts_file: Path,
    options: str,
    output: Path,
    min_selections: int,
    max_selections: int | None,
) -> None:
    r"""Create multi-select items from text file.

    Examples
    --------
    $ bead items create-multi-select-from-texts \\
        --texts-file sentences.txt \\
        --options "Agent,Patient,Theme,Goal" \\
        --output items.jsonl

    $ bead items create-multi-select-from-texts \\
        --texts-file sentences.txt \\
        --options "Semantic,Syntactic,Pragmatic" \\
        --min-selections 1 --max-selections 2 \\
        --output items.jsonl
    """
    try:
        # Load texts
        texts: list[str] = [
            line.strip() for line in texts_file.read_text().splitlines() if line.strip()
        ]
        print_info(f"Loaded {len(texts)} texts")

        # Parse options
        option_list: list[str] = [o.strip() for o in options.split(",")]

        # Create items - one multi-select item per text, using options as selections
        items: list[Item] = []
        for text in texts:
            # For multi-select, we need options. Use the text as metadata
            # and the option_list as the actual options
            meta: dict[str, MetadataValue] = {"stimulus": text}
            item: Item = create_multi_select_item(
                *option_list,
                min_selections=min_selections,
                max_selections=max_selections,
                metadata=meta,
            )
            items.append(item)

        print_info(f"Created {len(items)} multi-select items")

        # Save
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

        display_file_stats(output, len(items), "multi-select items")

    except Exception as e:
        print_error(f"Failed to create multi-select items: {e}")


# ==================== Magnitude Commands ====================


@click.command()
@click.option(
    "--texts-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="File with texts (one per line)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--measure",
    type=str,
    default="value",
    help="Measure name (default: 'value')",
)
@click.option(
    "--prompt",
    type=str,
    default="Enter value:",
    help="Input prompt",
)
def create_magnitude_from_texts(
    texts_file: Path,
    output: Path,
    measure: str,
    prompt: str,
) -> None:
    r"""Create magnitude estimation items from text file.

    Examples
    --------
    $ bead items create-magnitude-from-texts \\
        --texts-file sentences.txt --output items.jsonl

    $ bead items create-magnitude-from-texts \\
        --texts-file sentences.txt \\
        --measure "reading_time_ms" \\
        --prompt "Reading time (ms):" \\
        --output items.jsonl
    """
    try:
        # Load texts
        texts = [
            line.strip() for line in texts_file.read_text().splitlines() if line.strip()
        ]
        print_info(f"Loaded {len(texts)} texts")

        # Create items
        items = create_magnitude_items_from_texts(texts, unit=measure, prompt=prompt)

        # Save
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

        display_file_stats(output, len(items), "magnitude items")

    except Exception as e:
        print_error(f"Failed to create magnitude items: {e}")


# ==================== Free Text Commands ====================


@click.command()
@click.option(
    "--texts-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="File with texts (one per line)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--prompt",
    type=str,
    default="Provide your response:",
    help="Response prompt",
)
def create_free_text_from_texts(
    texts_file: Path,
    output: Path,
    prompt: str,
) -> None:
    r"""Create free text response items from text file.

    Examples
    --------
    $ bead items create-free-text-from-texts \\
        --texts-file sentences.txt --output items.jsonl

    $ bead items create-free-text-from-texts \\
        --texts-file sentences.txt \\
        --prompt "Paraphrase this sentence:" \\
        --output items.jsonl
    """
    try:
        # Load texts
        texts = [
            line.strip() for line in texts_file.read_text().splitlines() if line.strip()
        ]
        print_info(f"Loaded {len(texts)} texts")

        # Create items
        items = create_free_text_items_from_texts(texts, prompt=prompt)

        # Save
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

        display_file_stats(output, len(items), "free-text items")

    except Exception as e:
        print_error(f"Failed to create free-text items: {e}")


# ==================== Cloze Commands ====================


@click.command()
@click.option(
    "--text",
    type=str,
    required=True,
    help="Text with blank",
)
@click.option(
    "--blank-position",
    type=int,
    required=True,
    help="Position of blank (0-indexed word)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--blank-label",
    type=str,
    default="blank",
    help="Label for blank (default: 'blank')",
)
def create_simple_cloze(
    text: str,
    blank_position: int,
    output: Path,
    blank_label: str,
) -> None:
    r"""Create a simple cloze item.

    Examples
    --------
    $ bead items create-simple-cloze \\
        --text "The quick brown fox" \\
        --blank-position 1 \\
        -o item.jsonl

    $ bead items create-simple-cloze \\
        --text "The cat sat on the mat" \\
        --blank-position 3 \\
        --blank-label "preposition" \\
        -o item.jsonl
    """
    try:
        item = create_simple_cloze_item(
            text=text,
            blank_positions=[blank_position],
            blank_labels=[blank_label],
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(item.model_dump_json() + "\n")

        print_success(f"Created cloze item: {output}")

    except Exception as e:
        print_error(f"Failed to create cloze item: {e}")


# Export all commands
__all__ = [
    "create_forced_choice",
    "create_forced_choice_from_texts",
    "create_likert_7",
    "create_ordinal_scale_from_texts",
    "create_nli",
    "create_categorical",
    "create_binary_from_texts",
    "create_multi_select_from_texts",
    "create_magnitude_from_texts",
    "create_free_text_from_texts",
    "create_simple_cloze",
]
