"""Round-trip tests for the gallery's protocol declaration."""

from __future__ import annotations

import sys
from pathlib import Path

GALLERY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GALLERY_DIR))

from protocol import (  # noqa: E402
    ACCEPTABILITY_ANCHOR_NAME,
    acceptability_anchor,
    acceptability_family,
    build_protocol,
)

from bead.active_learning.models.registry import (  # noqa: E402
    model_class_for_encoding,
)
from bead.protocol.encoding import (  # noqa: E402
    ScaleType,
    encode_response_space,
)
from bead.protocol.items import family_to_item_template  # noqa: E402

CONFIG_PATH = GALLERY_DIR / "config.yaml"


def test_protocol_has_acceptability_family() -> None:
    protocol = build_protocol(CONFIG_PATH)
    assert len(protocol.families) == 1
    assert protocol.families[0].anchor.name == ACCEPTABILITY_ANCHOR_NAME


def test_anchor_carries_forced_choice_scale_type() -> None:
    anchor = acceptability_anchor(build_protocol(CONFIG_PATH))
    assert anchor.response_space.scale_type is ScaleType.FORCED_CHOICE
    assert anchor.response_space.options == ("first", "second")


def test_family_to_item_template_round_trips_prompt() -> None:
    family = acceptability_family(build_protocol(CONFIG_PATH))
    template = family_to_item_template(family, judgment_type="acceptability")
    assert template.task_type == "forced_choice"
    assert template.task_spec.prompt == family.anchor.canonical_prompt


def test_forced_choice_encoding_picks_forced_choice_model() -> None:
    anchor = acceptability_anchor(build_protocol(CONFIG_PATH))
    encoding = encode_response_space(anchor.name, anchor.response_space)
    model_cls = model_class_for_encoding(encoding)
    assert model_cls.__name__ == "ForcedChoiceModel"
