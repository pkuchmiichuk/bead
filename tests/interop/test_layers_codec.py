"""Tests for the BeadCodec lairs codec."""

from __future__ import annotations

import subprocess
import sys
from uuid import uuid4

from lairs.integrations.ports import Codec
from lairs.records import annotation

from bead.interop.layers.codec import BeadCodec
from bead.items.item import Item, ItemCollection
from bead.items.spans import Span, SpanLabel, SpanSegment


def _collection() -> ItemCollection:
    return ItemCollection(
        name="demo",
        source_template_collection_id=uuid4(),
        source_filled_collection_id=uuid4(),
        items=(
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "The cat sat"},
                tokenized_elements={"text": ("The", "cat", "sat")},
                token_space_after={"text": (True, True, False)},
                spans=(
                    Span(
                        span_id="s1",
                        segments=(SpanSegment(element_name="text", indices=(1,)),),
                        label=SpanLabel(label="ANIMAL", label_id="Q5"),
                    ),
                ),
            ),
            Item(item_template_id=uuid4(), rendered_elements={"text": "Hello"}),
        ),
        construction_stats={"built": 2},
    )


def test_codec_name_and_protocol() -> None:
    assert BeadCodec.name == "bead"
    assert isinstance(BeadCodec(), Codec)


def test_encode_decode_is_lossless() -> None:
    codec = BeadCodec()
    source = _collection().model_dump_json()
    fragment = codec.decode(source)
    assert codec.encode(fragment.records) == source


def test_decoded_records_validate_as_lairs_models() -> None:
    fragment = BeadCodec().decode(_collection().model_dump_json())
    layers = [
        annotation.AnnotationLayer.model_validate_json(record.value_json)
        for record in fragment.records
        if record.nsid == "pub.layers.annotation.annotationLayer"
    ]
    assert any(layer.kind == "span" for layer in layers)


def test_importing_bead_does_not_import_lairs() -> None:
    # bead.interop.layers is opt-in; importing the top-level package must not
    # pull lairs onto the base import path.
    code = "import sys; import bead; print('lairs' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
