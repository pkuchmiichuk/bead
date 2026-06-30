"""A ``lairs`` codec that round-trips a bead ``ItemCollection``.

:class:`BeadCodec` binds the :class:`lairs.integrations.ports.Codec` port so a
downstream user can reach it through ``lairs.codec("bead")`` once both packages
are installed (the codec is registered via the ``lairs.codecs`` entry point in
bead's project metadata).

The codec's external format is a bead ``ItemCollection`` serialized as JSON.
``decode`` runs each item through :data:`~bead.interop.layers.item_bridge.ITEM_LAYERS`
and concatenates the resulting canonical ``layers`` records, threading each item's
lens complement and the collection's own fields as private records under the
``bead.interop.complement`` NSID (a layers consumer ignores unknown NSIDs).
Because the complement rides along, ``encode(decode(x)) == x`` holds losslessly
for every collection, and a single canonical mapping (the lens) drives both
directions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from uuid import UUID

from lairs.integrations.codecs import CorpusFragment, FragmentRecord

from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    apply_identity,
    identity_of,
    j_int,
    j_obj,
    j_str,
)
from bead.interop.layers.item_bridge import ITEM_LAYERS
from bead.items.item import Item, ItemCollection

_COMPLEMENT_NSID = "bead.interop.complement"
_COLLECTION_LOCAL_ID = "collection"


class BeadCodec:
    """Bidirectional codec ``ItemCollection JSON <-> layers corpus fragment``."""

    name = "bead"

    def decode(
        self, src: str | bytes, *, into: CorpusFragment | None = None
    ) -> CorpusFragment:
        """Decode a bead ``ItemCollection`` JSON document into a layers fragment."""
        collection = ItemCollection.model_validate_json(_as_text(src))
        records: list[FragmentRecord] = list(into.records) if into is not None else []
        records.append(
            FragmentRecord(
                local_id=_COLLECTION_LOCAL_ID,
                nsid=_COMPLEMENT_NSID,
                value_json=json.dumps(_collection_complement(collection)),
            )
        )
        for index, item in enumerate(collection.items):
            fragment, complement = ITEM_LAYERS.forward(item)
            prefix = f"item{index}:"
            for record in fragment.records:
                records.append(
                    FragmentRecord(
                        local_id=f"{prefix}{record.local_id}",
                        nsid=record.nsid,
                        value_json=record.value_json,
                    )
                )
            records.append(
                FragmentRecord(
                    local_id=f"{prefix}complement",
                    nsid=_COMPLEMENT_NSID,
                    value_json=json.dumps(complement),
                )
            )
        return CorpusFragment(records=tuple(records), source="bead")

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Encode layers fragment records back into ``ItemCollection`` JSON."""
        collection_complement: JsonValue = {}
        item_records: dict[int, list[FragmentRecord]] = {}
        item_complements: dict[int, JsonValue] = {}
        for record in records:
            if record.local_id == _COLLECTION_LOCAL_ID:
                collection_complement = json.loads(record.value_json)
                continue
            index, original = _split_index(record.local_id)
            if index is None:
                continue
            if record.nsid == _COMPLEMENT_NSID and original == "complement":
                item_complements[index] = json.loads(record.value_json)
            else:
                item_records.setdefault(index, []).append(
                    FragmentRecord(
                        local_id=original,
                        nsid=record.nsid,
                        value_json=record.value_json,
                    )
                )
        items: list[Item] = []
        for index in sorted(item_records):
            fragment = CorpusFragment(records=tuple(item_records[index]), source="bead")
            items.append(ITEM_LAYERS.backward(fragment, item_complements[index]))
        collection = _rebuild_collection(collection_complement, tuple(items))
        return collection.model_dump_json()


def _collection_complement(collection: ItemCollection) -> JsonValue:
    return {
        "identity": identity_of(collection),
        "name": collection.name,
        "source_template_collection_id": str(collection.source_template_collection_id),
        "source_filled_collection_id": str(collection.source_filled_collection_id),
        "construction_stats": dict(collection.construction_stats),
    }


def _rebuild_collection(
    complement: JsonValue, items: tuple[Item, ...]
) -> ItemCollection:
    comp = j_obj(complement)
    stats = comp["construction_stats"]
    collection = ItemCollection(
        name=j_str(comp["name"]),
        source_template_collection_id=UUID(
            j_str(comp["source_template_collection_id"])
        ),
        source_filled_collection_id=UUID(j_str(comp["source_filled_collection_id"])),
        items=items,
        construction_stats={
            key: j_int(value)
            for key, value in (stats.items() if isinstance(stats, dict) else ())
        },
    )
    return apply_identity(collection, comp["identity"])


def _split_index(local_id: str) -> tuple[int | None, str]:
    head, _, rest = local_id.partition(":")
    if not head.startswith("item") or not rest:
        return None, local_id
    try:
        return int(head.removeprefix("item")), rest
    except ValueError:
        return None, local_id


def _as_text(src: str | bytes) -> str:
    return src.decode("utf-8") if isinstance(src, bytes) else src
