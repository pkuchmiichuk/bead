"""Lexical item models for words and multi-word expressions.

Lexical items are the atomic units that fill template slots during sentence
generation. The module covers single words, multi-word expressions (MWEs),
and the components that make up an MWE.
"""

from __future__ import annotations

import didactic.api as dx

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.language_codes import LanguageCode, validate_iso639_code
from bead.resources.constraints import Constraint


class LexicalItem(BeadBaseModel):
    """A lexical item with linguistic features.

    Follows the UniMorph structure of lemma, surface form, and feature
    bundle.

    Attributes
    ----------
    lemma : str
        Base / citation form (e.g. ``"walk"``, ``"the"``).
    form : str | None
        Inflected surface form. ``None`` means the form equals the lemma.
    language_code : LanguageCode
        ISO 639-3 language code.
    features : dict[str, JsonValue]
        Feature bundle (POS, morphological features, lexical-resource
        information).
    source : str | None
        Provenance (e.g. ``"VerbNet"``, ``"UniMorph"``, ``"manual"``).
    """

    lemma: str
    language_code: LanguageCode
    form: str | None = None
    features: dict[str, JsonValue] = dx.field(default_factory=dict)
    source: str | None = None

    @dx.validates("lemma")
    def _check_lemma(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("lemma must be non-empty")
        return value

    @dx.validates("language_code")
    def _check_language_code(self, value: LanguageCode) -> LanguageCode:
        return validate_iso639_code(value)


class MWEComponent(LexicalItem):
    """A component of a multi-word expression.

    Attributes
    ----------
    role : str
        Role within the MWE (e.g. ``"verb"``, ``"particle"``).
    required : bool
        Whether the component must be present.
    constraints : tuple[Constraint, ...]
        Component-specific constraints (in addition to base
        ``LexicalItem`` constraints).
    """

    role: str
    required: bool = True
    constraints: tuple[dx.Embed[Constraint], ...] = ()


class MultiWordExpression(LexicalItem):
    """Multi-word expression as a lexical item.

    Attributes
    ----------
    components : tuple[MWEComponent, ...]
        Component lexical items that make up the MWE.
    separable : bool
        Whether components can be separated by intervening words.
    adjacency_pattern : str | None
        DSL expression defining valid adjacency patterns. Variables are
        component roles plus ``distance`` between components.
    """

    components: tuple[dx.Embed[MWEComponent], ...] = ()
    separable: bool = False
    adjacency_pattern: str | None = None
