"""Bridge lens between a bead ``CorpusRecord`` and a layers ``expression``.

A :class:`~bead.corpus.records.CorpusRecord` projects to a canonical
:class:`lairs.records.expression.Expression`; the lens complement holds the
bead-only remainder (framework identity, the source name, and the record index,
which ``layers`` has no slot for), so the round-trip is exact and the
GetPut/PutGet laws hold.
"""

from __future__ import annotations

import didactic.api as dx
from lairs.records import expression

from bead.corpus.records import CorpusRecord
from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    apply_identity,
    feature_map,
    identity_of,
    j_int,
    j_obj,
    j_str,
    read_feature_map_scalar,
)

# CorpusRecord has no kind field; corpus records are atomic, sentence-level text
# units, so the synthesized expression kind is a fixed, valid slug.
_EXPRESSION_KIND = "sentence"


class RecordExpressionLens(dx.Lens[CorpusRecord, expression.Expression, JsonValue]):
    """Lossless lens ``CorpusRecord <-> (layers expression, complement)``."""

    def forward(self, record: CorpusRecord) -> tuple[expression.Expression, JsonValue]:
        """Project a record to a layers expression and bead complement."""
        view = expression.Expression(
            id=str(record.id),
            kind=_EXPRESSION_KIND,
            createdAt=record.created_at,
            text=record.text,
            features=feature_map(record.provenance),
        )
        complement: JsonValue = {
            "identity": identity_of(record),
            "source_name": record.source_name,
            "record_index": record.record_index,
        }
        return view, complement

    def backward(
        self, view: expression.Expression, complement: JsonValue
    ) -> CorpusRecord:
        """Reconstruct a record from its layers expression and complement."""
        comp = j_obj(complement)
        record = CorpusRecord(
            text=view.text if view.text is not None else "",
            source_name=j_str(comp["source_name"]),
            record_index=j_int(comp["record_index"]),
            provenance=read_feature_map_scalar(view.features),
        )
        return apply_identity(record, comp["identity"])


RECORD_EXPRESSION = RecordExpressionLens()


def record_to_expression(record: CorpusRecord) -> expression.Expression:
    """Return the standalone layers ``expression`` view of a corpus record."""
    view, _complement = RECORD_EXPRESSION.forward(record)
    return view
