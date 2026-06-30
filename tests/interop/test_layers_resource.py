"""Round-trip law tests for the resource overlap lenses."""

from __future__ import annotations

from bead.interop.layers.resource_lens import (
    LEXICAL_ITEM_ENTRY,
    LEXICON_COLLECTION,
    TEMPLATE_LAYERS,
)
from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template


class TestLexicalItemEntry:
    """LexicalItem <-> layers entry."""

    def test_full(self) -> None:
        item = LexicalItem(
            lemma="run",
            language_code="eng",
            form="ran",
            features={"pos": "VERB", "tense": "past"},
            source="UniMorph",
        )
        view, complement = LEXICAL_ITEM_ENTRY.forward(item)
        assert view.form == "ran"
        assert view.lemma == "run"
        assert LEXICAL_ITEM_ENTRY.backward(view, complement) == item

    def test_form_defaults_to_lemma_in_view(self) -> None:
        item = LexicalItem(lemma="dog", language_code="eng")
        view, complement = LEXICAL_ITEM_ENTRY.forward(item)
        assert view.form == "dog"  # faithful entry.form
        # but the original None form is recovered exactly
        restored = LEXICAL_ITEM_ENTRY.backward(view, complement)
        assert restored.form is None
        assert restored == item


class TestLexiconCollection:
    """Lexicon <-> layers collection + entries."""

    def test_roundtrip(self) -> None:
        lexicon = Lexicon(
            name="verbs",
            description="motion verbs",
            language_code="eng",
            items=(
                LexicalItem(lemma="run", language_code="eng", features={"pos": "VERB"}),
                LexicalItem(lemma="walk", language_code="eng", form="walked"),
            ),
            tags=("motion", "manner"),
        )
        view, complement = LEXICON_COLLECTION.forward(lexicon)
        assert view.collection.kind == "lexicon"
        assert len(view.entries) == 2
        assert LEXICON_COLLECTION.backward(view, complement) == lexicon

    def test_empty(self) -> None:
        lexicon = Lexicon(name="empty")
        view, complement = LEXICON_COLLECTION.forward(lexicon)
        assert LEXICON_COLLECTION.backward(view, complement) == lexicon


class TestTemplateLayers:
    """Template <-> layers template (with slots and constraints)."""

    def test_roundtrip(self) -> None:
        template = Template(
            name="transitive",
            template_string="The {subj} {verb} the {obj}.",
            slots={
                "subj": Slot(name="subj", required=True),
                "verb": Slot(
                    name="verb",
                    description="a transitive verb",
                    constraints=(
                        Constraint(expression="self.pos == 'VERB'", description="verb"),
                    ),
                ),
                "obj": Slot(name="obj", default_value="ball"),
            },
            constraints=(
                Constraint(
                    expression="subj.number == obj.number",
                    context={"strict": True},
                ),
            ),
            description="a 2-argument frame",
            language_code="eng",
            tags=("syntax",),
            metadata={"source": "manual"},
        )
        view, complement = TEMPLATE_LAYERS.forward(template)
        assert view.text == "The {subj} {verb} the {obj}."
        assert {slot.name for slot in view.slots} == {"subj", "verb", "obj"}
        verb_slot = next(slot for slot in view.slots if slot.name == "verb")
        assert verb_slot.constraints is not None
        assert verb_slot.constraints[0].expression == "self.pos == 'VERB'"
        assert TEMPLATE_LAYERS.backward(view, complement) == template

    def test_minimal(self) -> None:
        template = Template(name="t", template_string="{x}")
        view, complement = TEMPLATE_LAYERS.forward(template)
        assert TEMPLATE_LAYERS.backward(view, complement) == template
