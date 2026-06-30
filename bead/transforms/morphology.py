"""Morphological transforms backed by UniMorph paradigms.

Each :class:`MorphologicalTransform` targets a specific inflectional
feature bundle (e.g. present participle) and applies the inflection to
the *head* token of the span text.  Non-head tokens are preserved as-is,
producing natural multi-word results like ``"running to the store"``
from a span ``"run to the store"`` with a ``gerund`` transform.

The system is language-agnostic at the protocol level: the same
:class:`MorphologicalTransform` class works for any language supported
by UniMorph — the language is selected via ``language_code`` at
construction time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from bead.resources.adapters.unimorph import UniMorphAdapter
from bead.transforms.base import TransformContext, TransformRegistry

logger = logging.getLogger(__name__)


# feature matching predicates
FeaturePredicate = Callable[[dict[str, str]], bool]


@dataclass(frozen=True)
class InflectionSpec:
    """Specification for a target inflectional form.

    Attributes
    ----------
    name : str
        Human-readable name (e.g. ``"gerund"``).
    predicate : FeaturePredicate
        A callable that returns ``True`` for a UniMorph feature dict
        matching the desired form.
    description : str
        Short description of the inflection.
    """

    name: str
    predicate: FeaturePredicate
    description: str = ""


class MorphologicalTransform:
    """Apply a morphological inflection to the head token of span text.

    Given a span like ``"run to the store"`` and an inflection spec
    for the present participle, this transform produces ``"running to
    the store"`` by inflecting only the head token (defaulting to
    the first token when ``context.head_index`` is not set).

    Parameters
    ----------
    inflection_spec : InflectionSpec
        Specifies which inflected form to target.
    language_code : str
        ISO 639 language code for UniMorph lookup.
    lemmatize : bool
        If ``True`` and ``context.lemma`` is ``None``, attempt to
        find the paradigm by trying the head token directly as a
        lemma. Defaults to ``True``.

    Examples
    --------
    >>> spec = InflectionSpec(
    ...     name="gerund",
    ...     predicate=lambda f: (
    ...         f.get("verb_form") == "V.PTCP" and f.get("tense") == "PRS"
    ...     ),
    ... )
    >>> t = MorphologicalTransform(spec, language_code="eng")
    >>> ctx = TransformContext(
    ...     lemma="run", head_index=0, tokens=["run", "to", "the", "store"]
    ... )
    >>> t("run to the store", ctx)
    'running to the store'
    """

    def __init__(
        self,
        inflection_spec: InflectionSpec,
        language_code: str,
        *,
        lemmatize: bool = True,
    ) -> None:
        self._spec = inflection_spec
        self._language_code = language_code
        self._lemmatize = lemmatize

        # lazily initialised adapter (avoid import cost until first use)
        self._adapter: UniMorphAdapter | None = None
        # cache: lemma → inflected form string
        self._cache: dict[str, str | None] = {}

    @property
    def inflection_spec(self) -> InflectionSpec:
        """The inflection specification for this transform."""
        return self._spec

    def _get_adapter(self) -> UniMorphAdapter:
        """Lazily import and create the UniMorph adapter."""
        if self._adapter is None:
            self._adapter = UniMorphAdapter()
        return self._adapter

    def _lookup_inflection(self, lemma: str) -> str | None:
        """Look up the target inflected form for a lemma.

        Parameters
        ----------
        lemma : str
            Base form to inflect.

        Returns
        -------
        str | None
            The inflected surface form, or ``None`` if not found.
        """
        if lemma in self._cache:
            return self._cache[lemma]

        try:
            adapter = self._get_adapter()
            items = adapter.fetch_items(query=lemma, language_code=self._language_code)

        except Exception:
            logger.debug(
                "UniMorph lookup failed for lemma=%r, lang=%s",
                lemma,
                self._language_code,
                exc_info=True,
            )
            self._cache[lemma] = None
            return None

        for item in items:
            features = {str(k): str(v) for k, v in (item.features or {}).items()}
            if self._spec.predicate(features):
                self._cache[lemma] = item.form
                return item.form

        self._cache[lemma] = None
        return None

    def _identify_head(
        self, text: str, context: TransformContext
    ) -> tuple[int, list[str]]:
        """Determine the head token index and token list.

        Parameters
        ----------
        text : str
            Full span text.
        context : TransformContext
            Context with optional ``head_index`` and ``tokens``.

        Returns
        -------
        tuple[int, list[str]]
            Head index and token list.
        """
        tokens: list[str] = list(context.tokens) if context.tokens else text.split()
        head_index = context.head_index if context.head_index is not None else 0
        head_index = max(0, min(head_index, len(tokens) - 1))
        return head_index, tokens

    def __call__(self, text: str, context: TransformContext) -> str:
        """Apply the inflection to the span text.

        Parameters
        ----------
        text : str
            The span text to transform.
        context : TransformContext
            Metadata about the span (lemma, head_index, etc.).

        Returns
        -------
        str
            Text with the head token inflected. Falls back to the
            original text if the inflection cannot be resolved.
        """
        head_index, tokens = self._identify_head(text, context)

        if not tokens:
            return text

        # determine the lemma to look up
        lemma = context.lemma or tokens[head_index]
        inflected = self._lookup_inflection(lemma)

        if inflected is None:
            return text

        # replace the head token with the inflected form
        result_tokens = list(tokens)
        result_tokens[head_index] = inflected

        return " ".join(result_tokens)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return (
            f"MorphologicalTransform("
            f"spec={self._spec.name!r}, "
            f"lang={self._language_code!r})"
        )


# common feature predicates, organised by category


def _is_present_participle(features: dict[str, str]) -> bool:
    """Match present participle / gerund (V;V.PTCP;PRS)."""
    return features.get("verb_form") == "V.PTCP" and features.get("tense") == "PRS"


def _is_past_tense(features: dict[str, str]) -> bool:
    """Match simple past (V;PST) — exclude participles."""
    return features.get("tense") == "PST" and features.get("verb_form") not in (
        "V.PTCP",
    )


def _is_past_participle(features: dict[str, str]) -> bool:
    """Match past participle (V;V.PTCP;PST)."""
    return features.get("verb_form") == "V.PTCP" and features.get("tense") == "PST"


def _is_present_3sg(features: dict[str, str]) -> bool:
    """Match 3rd person singular present (V;PRS;3;SG)."""
    return (
        features.get("tense") == "PRS"
        and features.get("person") == "3"
        and features.get("number") == "SG"
        and features.get("verb_form") != "V.PTCP"
    )


def _is_infinitive(features: dict[str, str]) -> bool:
    """Match infinitive / base form (V with no inflection markers)."""
    return (
        features.get("pos") == "V"
        and not features.get("tense")
        and not features.get("person")
        and not features.get("number")
        and not features.get("verb_form")
    )


# standard inflection specs

GERUND = InflectionSpec(
    name="gerund",
    predicate=_is_present_participle,
    description="present participle / gerund form (e.g. walking)",
)

PAST_TENSE = InflectionSpec(
    name="past_tense",
    predicate=_is_past_tense,
    description="simple past form (e.g. walked)",
)

PAST_PARTICIPLE = InflectionSpec(
    name="past_participle",
    predicate=_is_past_participle,
    description="past participle form (e.g. walked, broken)",
)

PRESENT_3SG = InflectionSpec(
    name="present_3sg",
    predicate=_is_present_3sg,
    description="3rd person singular present (e.g. walks)",
)

INFINITIVE = InflectionSpec(
    name="infinitive",
    predicate=_is_infinitive,
    description="base / infinitive form (e.g. walk)",
)


def register_morphological_transforms(
    registry: TransformRegistry,
    language_code: str,
) -> None:
    """Register standard morphological transforms for a language.

    Adds ``gerund``, ``past_tense``, ``past_participle``,
    ``present_3sg``, and ``infinitive`` transforms backed by UniMorph.

    Parameters
    ----------
    registry : TransformRegistry
        Registry to populate.
    language_code : str
        ISO 639 language code.
    """
    specs = [GERUND, PAST_TENSE, PAST_PARTICIPLE, PRESENT_3SG, INFINITIVE]

    for spec in specs:
        registry.register(
            spec.name,
            MorphologicalTransform(spec, language_code=language_code),
        )
