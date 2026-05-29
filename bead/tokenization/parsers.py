"""Dependency parsing into standoff spans.

Provides dependency parsers (spaCy, Stanza) that produce a per-sentence
``ParsedSentence`` of ``ParsedToken`` records (token, lemma, upos, xpos,
morphological features, head, deprel), and ``parse_to_spans`` which projects a
parse onto bead's standoff ``Span`` + ``SpanRelation`` models.

The projection is deliberately aligned with the ``layers`` linguistic
annotation model so a parse stored on an ``Item`` carries every field a layers
dependency ``AnnotationLayer``/``Annotation`` needs: each token becomes a
single-token ``Span`` whose ``head_index`` is its governor and whose
``span_metadata`` carries ``upos``/``xpos``/``lemma``/``deprel``/``formalism``/
``tool`` plus morphological features and character offsets; each syntactic arc
becomes a directed ``SpanRelation`` from head to dependent labeled with the
dependency relation. The conventions below (Universal Dependencies labels,
``head -> dependent`` arc direction, retained character offsets) keep that
mapping lossless without coupling bead to layers' wire format.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import didactic.api as dx

from bead.items.spans import (
    MetadataValue,
    Span,
    SpanLabel,
    SpanRelation,
    SpanSegment,
)
from bead.tokenization.config import TokenizerConfig
from bead.tokenization.tokenizers import spacy_space_after

if TYPE_CHECKING:
    from spacy.language import Language

# layers-aligned conventions, recorded once so both projects stay matched.
UNIVERSAL_DEPENDENCIES = "universal-dependencies"
ROOT_DEPREL = "root"


@runtime_checkable
class DependencyParser(Protocol):
    """A callable that dependency-parses text into sentences.

    Carries a ``tool`` identifier recorded in the layers-aligned provenance of
    any spans projected from its output.
    """

    tool: str

    def __call__(self, text: str) -> tuple[ParsedSentence, ...]:
        """Dependency-parse text into sentences."""
        ...


class ParsedToken(dx.Model):
    """A dependency-parsed token.

    A superset of ``DisplayToken``: it adds the syntactic and morphological
    fields produced by a dependency parser. Indices are sentence-local and
    0-based; ``head`` is the 0-based index of the governor token, or ``None``
    for the sentence root.

    Attributes
    ----------
    index : int
        Sentence-local 0-based token index.
    text : str
        Surface form of the token.
    lemma : str | None
        Lemma of the token.
    upos : str | None
        Universal part-of-speech tag.
    xpos : str | None
        Language-specific (treebank) part-of-speech tag.
    deprel : str | None
        Dependency relation of the token to its head.
    head : int | None
        Sentence-local 0-based index of the governor token; ``None`` for the
        root.
    morph : dict[str, str]
        Morphological features (e.g. ``{"Number": "Sing"}``).
    space_after : bool
        Whether whitespace follows this token in the source text.
    start_char : int
        Character offset of the token start in the sentence text.
    end_char : int
        Character offset of the token end in the sentence text.
    """

    index: int
    text: str
    lemma: str | None = None
    upos: str | None = None
    xpos: str | None = None
    deprel: str | None = None
    head: int | None = None
    morph: dict[str, str] = dx.field(default_factory=dict)
    space_after: bool = True
    start_char: int = 0
    end_char: int = 0


class ParsedSentence(dx.Model):
    """A single dependency-parsed sentence.

    Attributes
    ----------
    original_text : str
        The sentence text.
    tokens : tuple[ParsedToken, ...]
        The parsed tokens, in order.
    """

    original_text: str
    tokens: tuple[dx.Embed[ParsedToken], ...] = ()


def _parse_feats(feats: str | None) -> dict[str, str]:
    """Parse a CoNLL-U ``feats`` string into a feature dict.

    Parameters
    ----------
    feats : str | None
        Pipe-separated ``Key=Value`` morphological features, or ``None``.

    Returns
    -------
    dict[str, str]
        Parsed features (empty when ``feats`` is ``None`` or ``"_"``).
    """
    if not feats or feats == "_":
        return {}
    result: dict[str, str] = {}
    for pair in feats.split("|"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key] = value
    return result


class SpacyParser:
    """spaCy-based dependency parser.

    Loads a spaCy pipeline with tagger, parser, lemmatizer, and morphologizer
    components and yields one ``ParsedSentence`` per sentence.

    Parameters
    ----------
    language : str
        ISO 639 language code.
    model_name : str | None
        Explicit spaCy model name. When ``None``, uses
        ``{language}_core_web_sm``.
    """

    tool = "spacy"

    def __init__(self, language: str = "en", model_name: str | None = None) -> None:
        self._language = language
        self._model_name = model_name
        self._nlp: Language | None = None

    def _load(self) -> Language:
        if self._nlp is not None:
            return self._nlp

        try:
            spacy = importlib.import_module("spacy")
        except ImportError as e:
            raise ImportError(
                "spaCy is required for SpacyParser. "
                "Install it with: pip install 'bead[tokenization]'"
            ) from e

        model = self._model_name or f"{self._language}_core_web_sm"
        try:
            nlp: Language = spacy.load(model)
        except OSError as e:
            raise ImportError(
                f"spaCy model {model!r} is required for dependency parsing. "
                f"Install it with: python -m spacy download {model}"
            ) from e

        self._nlp = nlp
        return nlp

    def __call__(self, text: str) -> tuple[ParsedSentence, ...]:
        """Parse text into dependency-parsed sentences.

        Parameters
        ----------
        text : str
            Input text (may contain multiple sentences).

        Returns
        -------
        tuple[ParsedSentence, ...]
            One ``ParsedSentence`` per detected sentence.
        """
        nlp = self._load()
        doc = nlp(text)
        sentences: list[ParsedSentence] = []
        for sent in doc.sents:
            offset = sent.start
            base_char = sent.start_char
            tokens: list[ParsedToken] = []
            for token in sent:
                local_index = token.i - offset
                head_local = token.head.i - offset
                head = None if token.head.i == token.i else head_local
                tokens.append(
                    ParsedToken(
                        index=local_index,
                        text=token.text,
                        lemma=token.lemma_ or None,
                        upos=token.pos_ or None,
                        xpos=token.tag_ or None,
                        deprel=token.dep_.lower() or None,
                        head=head,
                        morph=_parse_feats(str(token.morph) or None),
                        space_after=spacy_space_after(token),
                        start_char=token.idx - base_char,
                        end_char=token.idx + len(token.text) - base_char,
                    )
                )
            sentences.append(
                ParsedSentence(original_text=sent.text, tokens=tuple(tokens))
            )
        return tuple(sentences)


class StanzaParser:
    """Stanza-based dependency parser.

    Loads a Stanza pipeline with ``tokenize,pos,lemma,depparse`` processors and
    yields one ``ParsedSentence`` per sentence.

    Parameters
    ----------
    language : str
        ISO 639 language code.
    model_name : str | None
        Explicit Stanza package name. When ``None``, uses the default package.
    """

    tool = "stanza"

    def __init__(self, language: str = "en", model_name: str | None = None) -> None:
        self._language = language
        self._model_name = model_name
        self._nlp: _StanzaPipelineProtocol | None = None

    def _load(self) -> _StanzaPipelineProtocol:
        if self._nlp is not None:
            return self._nlp

        try:
            stanza = importlib.import_module("stanza")
        except ImportError as e:
            raise ImportError(
                "Stanza is required for StanzaParser. "
                "Install it with: pip install 'bead[tokenization]'"
            ) from e

        pkg = self._model_name
        pkg_kwarg = {"package": pkg} if pkg is not None else {}
        processors = "tokenize,pos,lemma,depparse"

        try:
            nlp: _StanzaPipelineProtocol = stanza.Pipeline(
                lang=self._language,
                processors=processors,
                verbose=False,
                **pkg_kwarg,
            )
        except Exception:
            stanza.download(self._language, verbose=False)
            nlp = stanza.Pipeline(
                lang=self._language,
                processors=processors,
                verbose=False,
                **pkg_kwarg,
            )

        self._nlp = nlp
        return nlp

    def __call__(self, text: str) -> tuple[ParsedSentence, ...]:
        """Parse text into dependency-parsed sentences.

        Parameters
        ----------
        text : str
            Input text (may contain multiple sentences).

        Returns
        -------
        tuple[ParsedSentence, ...]
            One ``ParsedSentence`` per detected sentence.
        """
        nlp = self._load()
        doc = nlp(text)
        sentences: list[ParsedSentence] = []
        for sentence in doc.sentences:
            base_char = sentence.words[0].start_char if sentence.words else 0
            tokens: list[ParsedToken] = []
            for word in sentence.words:
                # Stanza ids are 1-based within the sentence; head 0 is root.
                head = None if word.head == 0 else word.head - 1
                deprel = word.deprel.lower() if word.deprel else None
                tokens.append(
                    ParsedToken(
                        index=word.id - 1,
                        text=word.text,
                        lemma=word.lemma or None,
                        upos=word.upos or None,
                        xpos=word.xpos or None,
                        deprel=deprel,
                        head=head,
                        morph=_parse_feats(word.feats),
                        space_after=_stanza_word_space_after(word, text),
                        start_char=word.start_char - base_char,
                        end_char=word.end_char - base_char,
                    )
                )
            sentences.append(
                ParsedSentence(original_text=sentence.text, tokens=tuple(tokens))
            )
        return tuple(sentences)


def _stanza_word_space_after(word: _StanzaWordProtocol, text: str) -> bool:
    """Whether whitespace follows a Stanza word in the source text."""
    if word.misc:
        return "SpaceAfter=No" not in word.misc
    if word.end_char < len(text):
        return text[word.end_char] == " "
    return True


def create_parser(config: TokenizerConfig) -> DependencyParser:
    """Return a dependency-parsing function for the given config.

    Parameters
    ----------
    config : TokenizerConfig
        Tokenizer configuration. The ``backend`` selects the parser; the
        ``whitespace`` backend cannot parse and raises.

    Returns
    -------
    DependencyParser
        A callable that dependency-parses text into sentences.

    Raises
    ------
    ValueError
        If the backend cannot produce a dependency parse.
    """
    if config.backend == "spacy":
        return SpacyParser(language=config.language, model_name=config.model_name)
    if config.backend == "stanza":
        return StanzaParser(language=config.language, model_name=config.model_name)
    raise ValueError(
        f"Backend {config.backend!r} cannot produce a dependency parse; "
        "use 'spacy' or 'stanza'."
    )


def parse_to_spans(
    sentence: ParsedSentence,
    *,
    element_name: str = "text",
    tokenization_id: str,
    formalism: str = UNIVERSAL_DEPENDENCIES,
    tool: str,
) -> tuple[tuple[Span, ...], tuple[SpanRelation, ...]]:
    """Project a parsed sentence onto standoff spans and relations.

    Each token becomes a single-token ``Span`` (``span_type == "token"``) whose
    ``head_index`` is the governor index and whose ``span_metadata`` carries the
    layers-aligned fields. Each non-root token contributes one directed
    ``SpanRelation`` from its head (``source``) to itself (``target``), labeled
    with the dependency relation. This function is the single canonical owner of
    the ``span_id`` scheme and the ``head -> dependent`` arc direction.

    Parameters
    ----------
    sentence : ParsedSentence
        The parsed sentence to project.
    element_name : str
        Rendered-element name the token indices refer to.
    tokenization_id : str
        Stable identifier of the tokenization these tokens belong to (mirrors
        layers' ``TokenRef.tokenization_id``). Recorded in each span's metadata.
    formalism : str
        Dependency formalism slug (default ``"universal-dependencies"``).
    tool : str
        Identifier of the parser that produced the analysis.

    Returns
    -------
    tuple[tuple[Span, ...], tuple[SpanRelation, ...]]
        The token spans and the dependency-arc relations.
    """
    spans: list[Span] = []
    relations: list[SpanRelation] = []

    for token in sentence.tokens:
        span_metadata: dict[str, MetadataValue] = {
            "tokenization_id": tokenization_id,
            "formalism": formalism,
            "tool": tool,
            "start_char": token.start_char,
            "end_char": token.end_char,
        }
        if token.upos is not None:
            span_metadata["upos"] = token.upos
        if token.xpos is not None:
            span_metadata["xpos"] = token.xpos
        if token.lemma is not None:
            span_metadata["lemma"] = token.lemma
        if token.deprel is not None:
            span_metadata["deprel"] = token.deprel
        if token.morph:
            morph_value: dict[str, MetadataValue] = {}
            for feature, value in token.morph.items():
                morph_value[feature] = value
            span_metadata["morph"] = morph_value

        label = (
            SpanLabel(label=token.upos) if token.upos is not None else None
        )
        spans.append(
            Span(
                span_id=f"{element_name}:tok:{token.index}",
                segments=(
                    SpanSegment(element_name=element_name, indices=(token.index,)),
                ),
                head_index=token.head,
                label=label,
                span_type="token",
                span_metadata=span_metadata,
            )
        )

        if token.head is not None:
            relation_label = (
                SpanLabel(label=token.deprel) if token.deprel is not None else None
            )
            relations.append(
                SpanRelation(
                    relation_id=f"{element_name}:dep:{token.index}",
                    source_span_id=f"{element_name}:tok:{token.head}",
                    target_span_id=f"{element_name}:tok:{token.index}",
                    label=relation_label,
                    directed=True,
                )
            )

    return tuple(spans), tuple(relations)


# structural typing protocols for the untyped Stanza pipeline
class _StanzaWordProtocol(Protocol):
    """Structural type for a parsed Stanza ``Word``."""

    id: int
    text: str
    lemma: str | None
    upos: str | None
    xpos: str | None
    feats: str | None
    head: int
    deprel: str | None
    start_char: int
    end_char: int
    misc: str | None


class _StanzaSentenceProtocol(Protocol):
    """Structural type for a parsed Stanza sentence."""

    text: str
    words: list[_StanzaWordProtocol]


class _StanzaDocProtocol(Protocol):
    """Structural type for a parsed Stanza document."""

    sentences: list[_StanzaSentenceProtocol]


class _StanzaPipelineProtocol(Protocol):
    """Structural type for a Stanza ``Pipeline``."""

    def __call__(self, text: str) -> _StanzaDocProtocol:
        """Parse text into a Stanza document."""
        ...
