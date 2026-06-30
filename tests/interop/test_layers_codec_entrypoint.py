"""The BeadCodec is discoverable through lairs' codec registry."""

from __future__ import annotations

import lairs
from lairs.integrations import registry

from bead.interop.layers.codec import BeadCodec


def test_codec_resolves_by_name() -> None:
    assert lairs.codec("bead") is BeadCodec


def test_codec_is_listed_as_available() -> None:
    assert "bead" in registry.available("codecs")
