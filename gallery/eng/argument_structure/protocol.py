"""Protocol declarations for the argument-structure gallery.

The 2AFC acceptability question is declared once in ``config.yaml``
under the ``protocol:`` section and materialized here as a live
:class:`~bead.protocol.AnnotationProtocol`. Every downstream script
(``create_2afc_pairs.py``, ``generate_deployment.py``,
``run_pipeline.py``, ``simulate_pipeline.py``) imports
:func:`build_protocol` so the prompt string, response options, and
drift thresholds have a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

from bead.config.protocol import ProtocolConfig
from bead.protocol import AnnotationProtocol, QuestionFamily, SemanticAnchor

ACCEPTABILITY_ANCHOR_NAME: Final = "acceptability"


def load_protocol_config(config_path: Path | str = "config.yaml") -> ProtocolConfig:
    """Parse the ``protocol:`` section of ``config_path`` into a ProtocolConfig."""
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return ProtocolConfig.model_validate(data["protocol"])


def build_protocol(config_path: Path | str = "config.yaml") -> AnnotationProtocol:
    """Build the live :class:`AnnotationProtocol` from ``config_path``."""
    return load_protocol_config(config_path).build()


def acceptability_family(
    protocol: AnnotationProtocol,
) -> QuestionFamily:
    """Return the acceptability family from ``protocol``."""
    return protocol.family_by_name(ACCEPTABILITY_ANCHOR_NAME)


def acceptability_anchor(protocol: AnnotationProtocol) -> SemanticAnchor:
    """Return the acceptability anchor from ``protocol``."""
    return acceptability_family(protocol).anchor
