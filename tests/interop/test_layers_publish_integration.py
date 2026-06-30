"""Integration test: publish a bead corpus to a real PDS and read it back.

Gated behind ``--run-integration``; the ``pds_server`` fixture stands up a real
bluesky PDS in docker and skips cleanly when docker is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
import pytest
from lairs.atproto.pds import PdsClient
from lairs.author.publish import decode
from lairs.records import expression
from lairs.store import Repository

from bead.interop.layers import corpus_io
from bead.items.item import Item, ItemCollection

if TYPE_CHECKING:
    from tests.conftest import PdsServer

_EXPRESSION_COLLECTION = "pub.layers.expression.expression"


def _collection() -> ItemCollection:
    return ItemCollection(
        name="published",
        source_template_collection_id=uuid4(),
        source_filled_collection_id=uuid4(),
        items=(
            Item(item_template_id=uuid4(), rendered_elements={"text": "dogs bark"}),
        ),
    )


@pytest.mark.integration
def test_publish_corpus_to_live_pds(pds_server: PdsServer, tmp_path: Path) -> None:
    # Mint the corpus under the target account's DID so the records publish into
    # that repository, then commit it to a local repo for publishing.
    corpus = corpus_io.items_to_corpus(
        _collection(), corpus_name="study-1", authority=pds_server.did
    )
    revision = corpus_io.save_corpus_repo(corpus, tmp_path / "repo")
    repo = Repository.open(tmp_path / "repo")

    headers = {"Authorization": f"Bearer {pds_server.access_jwt}"}
    with httpx.Client(headers=headers) as client:
        plan = corpus_io.publish_corpus(
            repo,
            revision,
            to=pds_server.did,
            endpoint=pds_server.endpoint,
            client=client,
            dry_run=False,
        )
        # The expression record was created on the PDS.
        created = [op.uri for op in plan.creates]
        expr_uri = next(uri for uri in created if _EXPRESSION_COLLECTION in uri)
        rkey = expr_uri.rsplit("/", 1)[-1]

        reader = PdsClient(pds_server.endpoint, client)
        envelope = reader.get_record(pds_server.did, _EXPRESSION_COLLECTION, rkey)
        assert decode(envelope, expression.Expression).text == "dogs bark"

        # Re-publishing the unchanged revision is a no-op (idempotent).
        again = corpus_io.publish_corpus(
            repo,
            revision,
            to=pds_server.did,
            endpoint=pds_server.endpoint,
            client=client,
            dry_run=True,
        )
        assert again.is_empty()
