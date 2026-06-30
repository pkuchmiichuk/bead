"""Layers interop commands for the bead CLI.

Convert a bead ``ItemCollection`` to and from the ``layers`` schema through the
``bead`` lairs codec, materialize a collection as an Arrow/Parquet corpus, and
publish a committed corpus revision to an ATProto PDS.
"""

from __future__ import annotations

from pathlib import Path

import click
from lairs.integrations.codecs import CorpusFragment
from lairs.store import Repository
from rich.console import Console

from bead.cli.utils import print_info, print_success
from bead.interop.layers.codec import BeadCodec
from bead.interop.layers.corpus_io import (
    items_to_corpus,
    materialize_corpus,
    publish_corpus,
)
from bead.items.item import ItemCollection

console = Console()


@click.group()
def layers() -> None:
    r"""Layers interop commands.

    Convert a bead item collection to and from the ``layers`` schema and
    materialize it as an Arrow/Parquet corpus.

    \b
    Examples:
        $ bead layers encode items.json --out fragment.json
        $ bead layers decode fragment.json --out items.json
        $ bead layers materialize items.json --out corpus/
    """


@layers.command()
@click.argument("items_file", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_file", type=click.Path(path_type=Path), default=None)
def encode(items_file: Path, out_file: Path | None) -> None:
    """Encode an ItemCollection JSON document into a layers fragment."""
    fragment = BeadCodec().decode(items_file.read_text())
    payload = fragment.model_dump_json()
    if out_file is None:
        console.print_json(payload)
    else:
        out_file.write_text(payload)
        print_success(f"Wrote layers fragment to {out_file}")


@layers.command()
@click.argument("fragment_file", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_file", type=click.Path(path_type=Path), default=None)
def decode(fragment_file: Path, out_file: Path | None) -> None:
    """Decode a layers fragment back into an ItemCollection JSON document."""
    fragment = CorpusFragment.model_validate_json(fragment_file.read_text())
    payload = BeadCodec().encode(fragment.records)
    if out_file is None:
        console.print_json(payload)
    else:
        out_file.write_text(payload)
        print_success(f"Wrote item collection to {out_file}")


@layers.command()
@click.argument("items_file", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True)
@click.option("--name", default="corpus", help="Corpus name")
def materialize(items_file: Path, out_dir: Path, name: str) -> None:
    """Materialize an ItemCollection as an Arrow/Parquet layers corpus."""
    collection = ItemCollection.model_validate_json(items_file.read_text())
    corpus = items_to_corpus(collection, corpus_name=name)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = materialize_corpus(corpus, out_dir)
    for path in paths:
        print_info(f"  {path}")
    print_success(f"Materialized {len(paths)} view(s) to {out_dir}")


@layers.command()
@click.argument("repo_path", type=click.Path(exists=True, path_type=Path))
@click.argument("revision")
@click.option("--to", "to_did", required=True, help="Target PDS DID")
@click.option("--dry-run/--no-dry-run", default=True, help="Plan only (default)")
def publish(repo_path: Path, revision: str, to_did: str, dry_run: bool) -> None:
    """Publish a committed corpus revision to a PDS (dry run by default)."""
    repo = Repository.open(repo_path)
    plan = publish_corpus(repo, revision, to=to_did, dry_run=dry_run)
    if dry_run:
        print_info(str(plan))
        print_success("Computed publish plan (dry run; nothing written)")
    else:
        print_success(f"Published revision {revision} to {to_did}")
