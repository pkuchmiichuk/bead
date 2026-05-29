"""Bridge lenses between bead-native models and layers constructs.

These map the things bead's pipeline actually produces onto layers records:

- :class:`~bead.corpus.records.CorpusRecord` <-> a layers ``expression``.

The layers view is a faithful, standalone projection; the lens complement holds
the bead-only remainder (framework identity and fields layers has no slot for),
so the round-trip is exact and the GetPut/PutGet laws hold.
"""

from __future__ import annotations

import didactic.api as dx

from bead.corpus.records import CorpusRecord
from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    apply_identity,
    from_feature_map_scalar,
    identity_of,
    j_int,
    j_obj,
    j_str,
    to_feature_map,
)

_EXPRESSION_KIND = "expression"


class RecordExpressionLens(dx.Lens[CorpusRecord, JsonValue, JsonValue]):
    """Lossless lens ``CorpusRecord <-> (layers expression view, complement)``."""

    def forward(self, record: CorpusRecord) -> tuple[JsonValue, JsonValue]:
        """Project a record to a layers expression view and bead complement."""
        view: JsonValue = {
            "id": str(record.id),
            "kind": _EXPRESSION_KIND,
            "text": record.text,
            "features": to_feature_map(record.provenance),
            "createdAt": record.created_at.isoformat(),
        }
        complement: JsonValue = {
            "identity": identity_of(record),
            "source_name": record.source_name,
            "record_index": record.record_index,
        }
        return view, complement

    def backward(self, view: JsonValue, complement: JsonValue) -> CorpusRecord:
        """Reconstruct a record from its layers expression view and complement."""
        view_obj = j_obj(view)
        comp = j_obj(complement)
        record = CorpusRecord(
            text=j_str(view_obj["text"]),
            source_name=j_str(comp["source_name"]),
            record_index=j_int(comp["record_index"]),
            provenance=from_feature_map_scalar(view_obj["features"]),
        )
        return apply_identity(record, comp["identity"])


RECORD_EXPRESSION = RecordExpressionLens()


def record_to_expression(record: CorpusRecord) -> JsonValue:
    """Return the standalone layers ``expression`` view of a corpus record."""
    view, _complement = RECORD_EXPRESSION.forward(record)
    return view
