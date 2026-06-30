"""External resource loader commands for bead CLI.

This module provides commands for importing lexical resources from external
sources like VerbNet, UniMorph, PropBank, and FrameNet.
"""

from __future__ import annotations

from pathlib import Path

import click
from didactic.api import ValidationError

from bead.cli.display import create_progress, print_error, print_info, print_success
from bead.resources.adapters.cache import AdapterCache
from bead.resources.adapters.glazing import GlazingAdapter
from bead.resources.adapters.unimorph import UniMorphAdapter
from bead.resources.lexicon import Lexicon


@click.group()
def resource_loaders() -> None:
    r"""External resource loader commands.

    Commands for importing lexical items from external linguistic resources.

    \b
    Examples:
        $ bead resources import-verbnet --output lexicons/verbs.jsonl
        $ bead resources import-unimorph --language-code eng --query walk \
            --output lexicons/inflections.jsonl
        $ bead resources import-propbank --query eat.01 \
            --output lexicons/propbank.jsonl
    """


@click.command()
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output lexicon file path",
)
@click.option(
    "--query",
    "-q",
    help="Lemma or verb class to query (e.g., 'break', 'put-9.1'). Omit for all verbs.",
)
@click.option(
    "--verb-class",
    help="VerbNet class to filter (e.g., 'put-9.1', 'break-45.1')",
)
@click.option(
    "--language-code",
    default="eng",
    help="ISO 639 language code (default: 'eng')",
)
@click.option(
    "--include-frames",
    is_flag=True,
    help="Include detailed frame information in metadata",
)
@click.option(
    "--limit",
    type=int,
    help="Maximum number of verbs to import",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".cache/bead"),
    help="Cache directory for adapter results",
)
@click.pass_context
def import_verbnet(
    ctx: click.Context,
    output_file: Path,
    query: str | None,
    verb_class: str | None,
    language_code: str,
    include_frames: bool,
    limit: int | None,
    cache_dir: Path,
) -> None:
    r"""Import verbs from VerbNet.

    Fetches verb frame information from VerbNet and converts it to
    LexicalItem format. Frame information is stored in the features field.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output lexicon file.
    query : str | None
        Lemma or verb class to query.
    verb_class : str | None
        VerbNet class filter.
    language_code : str
        ISO 639 language code.
    include_frames : bool
        Include detailed frame information.
    limit : int | None
        Maximum number of items.
    cache_dir : Path
        Cache directory path.

    Examples
    --------
    # Import all verbs
    $ bead resources import-verbnet --output lexicons/verbnet_verbs.jsonl

    # Import specific verb
    $ bead resources import-verbnet --query break \
        --output lexicons/break_verbs.jsonl

    # Import verb class with frames
    $ bead resources import-verbnet --verb-class put-9.1 --include-frames \
        --output lexicons/put_verbs.jsonl

    # Limit results
    $ bead resources import-verbnet --limit 100 \
        --output lexicons/verbs_sample.jsonl
    """
    try:
        print_info("Initializing VerbNet adapter...")

        # Create cache (cache_dir not used by AdapterCache, it's in-memory only)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = AdapterCache()

        # Create adapter
        adapter = GlazingAdapter(resource="verbnet", cache=cache)

        # Fetch items with progress
        verb_class_info = f" (class: {verb_class})" if verb_class else ""
        print_info(f"Fetching verbs from VerbNet{verb_class_info}...")

        with create_progress() as progress:
            task = progress.add_task("[cyan]Fetching from VerbNet...", total=None)

            items = adapter.fetch_items(
                query=query,
                language_code=language_code,
                include_frames=include_frames,
                verb_class=verb_class,
            )

            progress.update(task, completed=True, total=1)

        # Apply limit if specified
        if limit is not None and len(items) > limit:
            print_info(f"Limiting results to {limit} items (from {len(items)})")
            items = items[:limit]

        # Create lexicon
        verb_class_desc = f" for class {verb_class}" if verb_class else ""
        lexicon = Lexicon(
            name=f"verbnet_{verb_class or query or 'all'}",
            language_code=language_code,
            description=f"VerbNet verbs{verb_class_desc}",
        )

        for item in items:
            lexicon = lexicon.with_item(item)
        # Save lexicon
        output_file.parent.mkdir(parents=True, exist_ok=True)
        lexicon.to_jsonl(str(output_file))

        print_success(f"Imported {len(items)} verbs from VerbNet: {output_file}")

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to import from VerbNet: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output lexicon file path",
)
@click.option(
    "--query",
    "-q",
    help="Lemma to query (e.g., 'walk', 'run'). Omit for all forms in language.",
)
@click.option(
    "--language-code",
    "-l",
    required=True,
    help="ISO 639 language code (e.g., 'eng', 'spa', 'fra')",
)
@click.option(
    "--pos",
    help="Part of speech filter (e.g., 'VERB', 'NOUN', 'ADJ')",
)
@click.option(
    "--features",
    help="UniMorph features to filter (e.g., 'V;PST', 'N;PL')",
)
@click.option(
    "--limit",
    type=int,
    help="Maximum number of forms to import",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".cache/bead"),
    help="Cache directory for adapter results",
)
@click.pass_context
def import_unimorph(
    ctx: click.Context,
    output_file: Path,
    query: str | None,
    language_code: str,
    pos: str | None,
    features: str | None,
    limit: int | None,
    cache_dir: Path,
) -> None:
    r"""Import inflected forms from UniMorph.

    Fetches morphological paradigms from UniMorph and converts them to
    LexicalItem format. Morphological features are stored in the features field.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output lexicon file.
    query : str | None
        Lemma to query.
    language_code : str
        ISO 639 language code (required).
    pos : str | None
        Part of speech filter.
    features : str | None
        UniMorph features filter.
    limit : int | None
        Maximum number of items.
    cache_dir : Path
        Cache directory path.

    Examples
    --------
    # Import all English verb forms for "walk"
    $ bead resources import-unimorph --language-code eng --query walk \
        --pos VERB --output lexicons/walk_forms.jsonl

    # Import past tense forms
    $ bead resources import-unimorph --language-code eng --query run \
        --features "V;PST" --output lexicons/run_past.jsonl

    # Import all Spanish verb forms (limited)
    $ bead resources import-unimorph --language-code spa --pos VERB \
        --limit 1000 --output lexicons/spanish_verbs.jsonl
    """
    try:
        print_info("Initializing UniMorph adapter...")

        # Create cache (cache_dir not used by AdapterCache, it's in-memory only)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = AdapterCache()

        # Create adapter
        adapter = UniMorphAdapter(cache=cache)

        # Fetch items with progress
        lemma_info = f" (lemma: {query})" if query else ""
        print_info(f"Fetching forms from UniMorph for {language_code}{lemma_info}...")

        with create_progress() as progress:
            task = progress.add_task("[cyan]Fetching from UniMorph...", total=None)

            items = adapter.fetch_items(
                query=query,
                language_code=language_code,
                pos=pos,
                features=features,
            )

            progress.update(task, completed=True, total=1)

        # Apply limit if specified
        if limit is not None and len(items) > limit:
            print_info(f"Limiting results to {limit} items (from {len(items)})")
            items = items[:limit]

        # Create lexicon
        lemma_desc = f" (lemma: {query})" if query else ""
        lexicon = Lexicon(
            name=f"unimorph_{language_code}_{query or 'all'}",
            language_code=language_code,
            description=f"UniMorph inflections for {language_code}{lemma_desc}",
        )

        for item in items:
            lexicon = lexicon.with_item(item)
        # Save lexicon
        output_file.parent.mkdir(parents=True, exist_ok=True)
        lexicon.to_jsonl(str(output_file))

        print_success(
            f"Imported {len(items)} inflected forms from UniMorph: {output_file}"
        )

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to import from UniMorph: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output lexicon file path",
)
@click.option(
    "--query",
    "-q",
    help="Predicate to query (e.g., 'eat.01', 'break.01'). Omit for all predicates.",
)
@click.option(
    "--frameset",
    help="PropBank frameset to filter (e.g., 'eat.01')",
)
@click.option(
    "--language-code",
    default="eng",
    help="ISO 639 language code (default: 'eng')",
)
@click.option(
    "--include-frames",
    is_flag=True,
    help="Include detailed frame information in metadata",
)
@click.option(
    "--limit",
    type=int,
    help="Maximum number of predicates to import",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".cache/bead"),
    help="Cache directory for adapter results",
)
@click.pass_context
def import_propbank(
    ctx: click.Context,
    output_file: Path,
    query: str | None,
    frameset: str | None,
    language_code: str,
    include_frames: bool,
    limit: int | None,
    cache_dir: Path,
) -> None:
    r"""Import predicates from PropBank.

    Fetches predicate-argument structure information from PropBank and
    converts it to LexicalItem format.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output lexicon file.
    query : str | None
        Predicate to query.
    frameset : str | None
        PropBank frameset filter.
    language_code : str
        ISO 639 language code.
    include_frames : bool
        Include detailed frame information.
    limit : int | None
        Maximum number of items.
    cache_dir : Path
        Cache directory path.

    Examples
    --------
    # Import specific predicate
    $ bead resources import-propbank --query eat.01 \
        --output lexicons/eat_propbank.jsonl

    # Import all predicates (limited)
    $ bead resources import-propbank --limit 500 \
        --output lexicons/propbank_sample.jsonl

    # Import with frame information
    $ bead resources import-propbank --frameset break.01 --include-frames \
        --output lexicons/break_frames.jsonl
    """
    try:
        print_info("Initializing PropBank adapter...")

        # Create cache (cache_dir not used by AdapterCache, it's in-memory only)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = AdapterCache()

        # Create adapter
        adapter = GlazingAdapter(resource="propbank", cache=cache)

        # Fetch items with progress
        frameset_info = f" (frameset: {frameset})" if frameset else ""
        print_info(f"Fetching predicates from PropBank{frameset_info}...")

        with create_progress() as progress:
            task = progress.add_task("[cyan]Fetching from PropBank...", total=None)

            items = adapter.fetch_items(
                query=query,
                language_code=language_code,
                include_frames=include_frames,
                frameset=frameset,
            )

            progress.update(task, completed=True, total=1)

        # Apply limit if specified
        if limit is not None and len(items) > limit:
            print_info(f"Limiting results to {limit} items (from {len(items)})")
            items = items[:limit]

        # Create lexicon
        frameset_desc = f" for frameset {frameset}" if frameset else ""
        lexicon = Lexicon(
            name=f"propbank_{frameset or query or 'all'}",
            language_code=language_code,
            description=f"PropBank predicates{frameset_desc}",
        )

        for item in items:
            lexicon = lexicon.with_item(item)
        # Save lexicon
        output_file.parent.mkdir(parents=True, exist_ok=True)
        lexicon.to_jsonl(str(output_file))

        print_success(f"Imported {len(items)} predicates from PropBank: {output_file}")

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to import from PropBank: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output lexicon file path",
)
@click.option(
    "--query",
    "-q",
    help="Frame to query (e.g., 'Ingestion', 'Motion'). Omit for all frames.",
)
@click.option(
    "--frame",
    help="FrameNet frame to filter (e.g., 'Ingestion')",
)
@click.option(
    "--language-code",
    default="eng",
    help="ISO 639 language code (default: 'eng')",
)
@click.option(
    "--include-frames",
    is_flag=True,
    help="Include detailed frame information in metadata",
)
@click.option(
    "--limit",
    type=int,
    help="Maximum number of frames to import",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".cache/bead"),
    help="Cache directory for adapter results",
)
@click.pass_context
def import_framenet(
    ctx: click.Context,
    output_file: Path,
    query: str | None,
    frame: str | None,
    language_code: str,
    include_frames: bool,
    limit: int | None,
    cache_dir: Path,
) -> None:
    r"""Import frames from FrameNet.

    Fetches frame semantic information from FrameNet and converts it to
    LexicalItem format.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output lexicon file.
    query : str | None
        Frame to query.
    frame : str | None
        FrameNet frame filter.
    language_code : str
        ISO 639 language code.
    include_frames : bool
        Include detailed frame information.
    limit : int | None
        Maximum number of items.
    cache_dir : Path
        Cache directory path.

    Examples
    --------
    # Import specific frame
    $ bead resources import-framenet --query Ingestion \
        --output lexicons/ingestion_frame.jsonl

    # Import all frames (limited)
    $ bead resources import-framenet --limit 100 \
        --output lexicons/framenet_sample.jsonl

    # Import with frame information
    $ bead resources import-framenet --frame Motion --include-frames \
        --output lexicons/motion_frames.jsonl
    """
    try:
        print_info("Initializing FrameNet adapter...")

        # Create cache (cache_dir not used by AdapterCache, it's in-memory only)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = AdapterCache()

        # Create adapter
        adapter = GlazingAdapter(resource="framenet", cache=cache)

        # Fetch items with progress
        frame_info = f" (frame: {frame})" if frame else ""
        print_info(f"Fetching frames from FrameNet{frame_info}...")

        with create_progress() as progress:
            task = progress.add_task("[cyan]Fetching from FrameNet...", total=None)

            items = adapter.fetch_items(
                query=query,
                language_code=language_code,
                include_frames=include_frames,
                frame=frame,
            )

            progress.update(task, completed=True, total=1)

        # Apply limit if specified
        if limit is not None and len(items) > limit:
            print_info(f"Limiting results to {limit} items (from {len(items)})")
            items = items[:limit]

        # Create lexicon
        lexicon = Lexicon(
            name=f"framenet_{frame or query or 'all'}",
            language_code=language_code,
            description=f"FrameNet frames{f' for {frame}' if frame else ''}",
        )

        for item in items:
            lexicon = lexicon.with_item(item)
        # Save lexicon
        output_file.parent.mkdir(parents=True, exist_ok=True)
        lexicon.to_jsonl(str(output_file))

        print_success(f"Imported {len(items)} frames from FrameNet: {output_file}")

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to import from FrameNet: {e}")
        ctx.exit(1)


# Register commands
resource_loaders.add_command(import_verbnet, name="import-verbnet")
resource_loaders.add_command(import_unimorph, name="import-unimorph")
resource_loaders.add_command(import_propbank, name="import-propbank")
resource_loaders.add_command(import_framenet, name="import-framenet")
