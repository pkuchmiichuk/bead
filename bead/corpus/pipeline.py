"""Streaming corpus pipeline: parse, structurally filter, build items.

Composable lazy generators that turn a ``CorpusSource`` into structurally
filtered ``Item``s:

``parse_records`` -> ``filter_by_structure`` -> ``Item``s.

The whole chain is lazy, so a structural query (a DSL constraint over the
dependency parse, e.g. a transitive-verb pattern) can be run over a
multi-gigabyte corpus without loading it into memory.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from uuid import UUID

from bead.corpus.records import CorpusRecord
from bead.dsl.evaluator import DSLEvaluator
from bead.items.item import Item, MetadataValue
from bead.tokenization.parsers import (
    UNIVERSAL_DEPENDENCIES,
    DependencyParser,
    ParsedSentence,
    parse_to_spans,
)


def record_to_item(
    record: CorpusRecord,
    parsed: ParsedSentence,
    *,
    item_template_id: UUID,
    tool: str,
    element_name: str = "text",
    formalism: str = UNIVERSAL_DEPENDENCIES,
) -> Item:
    """Build an ``Item`` from a corpus record and its parse.

    The parse is projected onto spans and relations via ``parse_to_spans``; the
    record's provenance plus the layers-aligned layer discriminators are stored
    on ``item_metadata``.

    Parameters
    ----------
    record : CorpusRecord
        The source record (supplies text and provenance).
    parsed : ParsedSentence
        The dependency parse of ``record.text`` (or one of its sentences).
    item_template_id : UUID
        Template the resulting item is associated with.
    tool : str
        Parser identifier, recorded as provenance.
    element_name : str
        Rendered-element name for the parsed text.
    formalism : str
        Dependency formalism slug.

    Returns
    -------
    Item
        The constructed item with spans, relations, and provenance.
    """
    tokenization_id = str(record.id)
    spans, relations = parse_to_spans(
        parsed,
        element_name=element_name,
        tokenization_id=tokenization_id,
        formalism=formalism,
        tool=tool,
    )

    item_metadata: dict[str, MetadataValue] = {}
    for key, value in record.provenance.items():
        item_metadata[key] = value
    item_metadata["source_name"] = record.source_name
    item_metadata["corpus_record_id"] = str(record.id)
    item_metadata["record_index"] = record.record_index
    item_metadata["parser_tool"] = tool
    item_metadata["formalism"] = formalism
    item_metadata["subkind"] = "dependency"
    item_metadata["tokenization_id"] = tokenization_id

    return Item(
        item_template_id=item_template_id,
        rendered_elements={element_name: parsed.original_text},
        spans=spans,
        span_relations=relations,
        tokenized_elements={element_name: tuple(t.text for t in parsed.tokens)},
        token_space_after={element_name: tuple(t.space_after for t in parsed.tokens)},
        item_metadata=item_metadata,
    )


def parse_records(
    source: Iterable[CorpusRecord],
    parser: DependencyParser,
    *,
    split_sentences: bool = True,
) -> Iterator[tuple[CorpusRecord, ParsedSentence]]:
    """Parse each record, yielding ``(record, sentence)`` pairs.

    Parameters
    ----------
    source : Iterable[CorpusRecord]
        The records to parse.
    parser : DependencyParser
        The dependency parser to apply.
    split_sentences : bool
        When ``True`` (default), multi-sentence records fan out to one pair per
        sentence. When ``False``, only records that parse to exactly one
        sentence are emitted (multi-sentence records are skipped).

    Yields
    ------
    tuple[CorpusRecord, ParsedSentence]
        A record paired with one of its parsed sentences.
    """
    for record in source:
        sentences = parser(record.text)
        if not split_sentences and len(sentences) != 1:
            continue
        for sentence in sentences:
            yield record, sentence


def filter_by_structure(
    parsed: Iterable[tuple[CorpusRecord, ParsedSentence]],
    constraint: str,
    *,
    item_template_id: UUID,
    tool: str,
    element_name: str = "text",
    formalism: str = UNIVERSAL_DEPENDENCIES,
    evaluator: DSLEvaluator | None = None,
) -> Iterator[Item]:
    """Yield items whose parse satisfies a structural DSL constraint.

    Parameters
    ----------
    parsed : Iterable[tuple[CorpusRecord, ParsedSentence]]
        ``(record, sentence)`` pairs (e.g. from ``parse_records``).
    constraint : str
        A DSL expression evaluated with the item bound as ``self`` and ``item``
        (e.g. ``'upos(self, root(self)) == "VERB"'``).
    item_template_id : UUID
        Template the resulting items are associated with.
    tool : str
        Parser identifier, recorded as provenance.
    element_name : str
        Rendered-element name for the parsed text.
    formalism : str
        Dependency formalism slug.
    evaluator : DSLEvaluator | None
        Reused evaluator (one is created if ``None``).

    Yields
    ------
    Item
        Items whose parse satisfies ``constraint``.
    """
    engine = evaluator if evaluator is not None else DSLEvaluator()
    for record, sentence in parsed:
        item = record_to_item(
            record,
            sentence,
            item_template_id=item_template_id,
            tool=tool,
            element_name=element_name,
            formalism=formalism,
        )
        if engine.evaluate(constraint, {"self": item, "item": item}):
            yield item


def sample_corpus(
    source: Iterable[CorpusRecord],
    parser: DependencyParser,
    constraint: str,
    *,
    item_template_id: UUID,
    element_name: str = "text",
    formalism: str = UNIVERSAL_DEPENDENCIES,
    split_sentences: bool = True,
    limit: int | None = None,
    evaluator: DSLEvaluator | None = None,
) -> Iterator[Item]:
    """Stream, parse, and structurally filter a corpus into items.

    Convenience composition of ``parse_records`` and ``filter_by_structure``,
    optionally capped at ``limit`` items.

    Parameters
    ----------
    source : Iterable[CorpusRecord]
        The corpus source.
    parser : DependencyParser
        The dependency parser to apply (its ``tool`` is recorded as provenance).
    constraint : str
        Structural DSL constraint each item must satisfy.
    item_template_id : UUID
        Template the resulting items are associated with.
    element_name : str
        Rendered-element name for the parsed text.
    formalism : str
        Dependency formalism slug.
    split_sentences : bool
        Whether multi-sentence records fan out (see ``parse_records``).
    limit : int | None
        Maximum number of items to yield.
    evaluator : DSLEvaluator | None
        Reused evaluator (one is created if ``None``).

    Yields
    ------
    Item
        Matching items, at most ``limit`` of them.
    """
    pairs = parse_records(source, parser, split_sentences=split_sentences)
    items = filter_by_structure(
        pairs,
        constraint,
        item_template_id=item_template_id,
        tool=parser.tool,
        element_name=element_name,
        formalism=formalism,
        evaluator=evaluator,
    )
    if limit is not None:
        items = itertools.islice(items, limit)
    yield from items
