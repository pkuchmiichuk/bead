"""CLI commands for the annotation-protocol layer.

Exposes the :class:`~bead.config.protocol.ProtocolConfig`-driven
workflow as ``bead protocol`` subcommands. Every command operates on
a :class:`~bead.config.config.BeadConfig` loaded via
:func:`bead.config.loader.load_config`, so the same TOML / YAML
configuration drives both Python and CLI invocations.

Subcommands:

- ``validate`` reports a per-family summary of the configured protocol
  and verifies that it materializes without errors.
- ``realize`` reads protocol contexts from a JSONL file and writes
  realized questions (or full :class:`~bead.items.item.Item` objects
  when ``--emit-items`` is given) to a JSONL output file.
- ``items`` renders the per-family
  :class:`~bead.items.item_template.ItemTemplate` collection to JSONL
  for downstream stages.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import click

from bead.cli.display import print_error, print_info, print_success
from bead.config.loader import load_config
from bead.data.serialization import read_jsonlines, write_jsonlines
from bead.items.cache import ModelOutputCache
from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.protocol import (
    AnnotationProtocol,
    ProtocolContext,
    family_to_item_template,
    realize_protocol_to_items,
)


def _load_protocol(
    config_path: Path | None,
    profile: str,
    overrides: Sequence[str] = (),
) -> AnnotationProtocol:
    """Load a :class:`BeadConfig` and materialize its protocol."""
    config = load_config(
        config_path=config_path,
        profile=profile,
        overrides=overrides,
    )
    cache = ModelOutputCache(
        cache_dir=config.paths.cache_dir / "models",
        backend="filesystem",
    )
    return config.protocol.build(cache=cache)


@click.group()
def protocol() -> None:
    r"""Annotation-protocol commands.

    Drive the :class:`~bead.protocol.AnnotationProtocol` declared in
    the project's BeadConfig (``protocol`` section): validate the
    configuration, realize prompts for a batch of contexts, and emit
    the per-family ItemTemplate collection.

    \b
    Examples:
        # Verify a protocol declaration in bead.toml
        $ bead protocol validate

        # Realize prompts for all contexts in contexts.jsonl
        $ bead protocol realize contexts.jsonl realizations.jsonl

        # Emit ItemTemplates for downstream item construction
        $ bead protocol items --judgment-type acceptability templates.jsonl
    """


@protocol.command()
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a project config file (defaults to bead.toml).",
)
@click.option("--profile", default="default", help="Configuration profile.")
@click.pass_context
def validate(
    ctx: click.Context,
    config_file: Path | None,
    profile: str,
) -> None:
    """Validate the protocol configuration and report its families.

    Loads the configured :class:`AnnotationProtocol`, prints a one-line
    summary per family (anchor name, scale type, number of options,
    declared dependencies), and exits non-zero on any construction
    error.
    """
    set_overrides: tuple[str, ...] = ctx.obj.get("set_overrides", ()) if ctx.obj else ()
    try:
        proto = _load_protocol(config_file, profile, set_overrides)
    except Exception as exc:  # noqa: BLE001
        print_error(f"Protocol failed to materialize: {exc}")
        raise SystemExit(1) from exc

    print_success(f"Protocol {proto.name!r}: {len(proto)} families")
    for family in proto.families:
        anchor = family.anchor
        rs = anchor.response_space
        scale = "ordinal" if rs.is_ordered else "binary" if len(rs) == 2 else "nominal"
        deps = ", ".join(family.depends_on) if family.depends_on else "(none)"
        print_info(
            f"  {family.name:20s}  scale={scale:8s}  "
            f"n_options={len(rs):2d}  depends_on={deps}"
        )


@protocol.command()
@click.argument("contexts_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a project config file (defaults to bead.toml).",
)
@click.option("--profile", default="default", help="Configuration profile.")
@click.option(
    "--emit-items",
    is_flag=True,
    help=(
        "Emit full Item records bound to per-family ItemTemplates "
        "instead of bare QuestionRealizations."
    ),
)
@click.option(
    "--judgment-type",
    default="acceptability",
    help=("Judgment type for emitted ItemTemplates (used when --emit-items is set)."),
)
@click.pass_context
def realize(
    ctx: click.Context,
    contexts_file: Path,
    output_file: Path,
    config_file: Path | None,
    profile: str,
    emit_items: bool,
    judgment_type: str,
) -> None:
    """Realize protocol questions for every context in CONTEXTS_FILE.

    CONTEXTS_FILE is a JSONL file of :class:`ProtocolContext` records
    (one per line). OUTPUT_FILE will be written as JSONL with one
    record per realized question (skipping non-applicable families).
    """
    set_overrides: tuple[str, ...] = ctx.obj.get("set_overrides", ()) if ctx.obj else ()
    proto = _load_protocol(config_file, profile, set_overrides)
    if len(proto) == 0:
        print_error(
            "Configured protocol is empty; nothing to realize. Add "
            "families to the [protocol.families] section."
        )
        raise SystemExit(1)

    contexts = read_jsonlines(contexts_file, ProtocolContext)

    if emit_items:
        items: list[Item] = []
        for ctx in contexts:
            for _realization, item in realize_protocol_to_items(
                proto,
                ctx,
                judgment_type=judgment_type,  # type: ignore[arg-type]
            ):
                items.append(item)
        write_jsonlines(items, output_file)
        print_success(
            f"Wrote {len(items)} Items from {len(contexts)} contexts to {output_file}"
        )
        return

    realizations = []
    for ctx in contexts:
        realizations.extend(proto.realize_all(ctx))
    write_jsonlines(realizations, output_file)
    print_success(
        f"Wrote {len(realizations)} realizations from {len(contexts)} "
        f"contexts to {output_file}"
    )


@protocol.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a project config file (defaults to bead.toml).",
)
@click.option("--profile", default="default", help="Configuration profile.")
@click.option(
    "--judgment-type",
    default="acceptability",
    help="Judgment type assigned to every ItemTemplate.",
)
@click.pass_context
def items(
    ctx: click.Context,
    output_file: Path,
    config_file: Path | None,
    profile: str,
    judgment_type: str,
) -> None:
    """Emit per-family ItemTemplates as JSONL.

    Builds one :class:`~bead.items.item_template.ItemTemplate` per
    family in the configured protocol and writes them to OUTPUT_FILE
    as JSONL. The resulting file feeds Stage 3 (item construction).
    """
    set_overrides: tuple[str, ...] = ctx.obj.get("set_overrides", ()) if ctx.obj else ()
    proto = _load_protocol(config_file, profile, set_overrides)
    if len(proto) == 0:
        print_error("Configured protocol is empty; no templates to emit.")
        raise SystemExit(1)

    templates: list[ItemTemplate] = [
        family_to_item_template(family, judgment_type=judgment_type)  # type: ignore[arg-type]
        for family in proto.families
    ]
    write_jsonlines(templates, output_file)
    print_success(f"Wrote {len(templates)} ItemTemplates to {output_file}")
