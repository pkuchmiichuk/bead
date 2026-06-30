"""Test strategy-based template filler."""

from __future__ import annotations

import pytest

from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template
from bead.templates.filler import FilledTemplate
from bead.templates.strategies import (
    ExhaustiveStrategy,
    RandomStrategy,
    StrategyFiller,
    StratifiedStrategy,
)


@pytest.fixture
def sample_lexicon() -> Lexicon:
    """Create sample lexicon for testing."""
    lexicon = Lexicon(name="test_lexicon")

    # Add nouns
    lexicon = lexicon.with_item(
        LexicalItem(lemma="cat", language_code="en", features={"pos": "NOUN"})
    )
    lexicon = lexicon.with_item(
        LexicalItem(lemma="dog", language_code="en", features={"pos": "NOUN"})
    )

    # Add verbs
    lexicon = lexicon.with_item(
        LexicalItem(lemma="broke", language_code="en", features={"pos": "VERB"})
    )
    lexicon = lexicon.with_item(
        LexicalItem(lemma="ate", language_code="en", features={"pos": "VERB"})
    )

    # Add adjectives
    lexicon = lexicon.with_item(
        LexicalItem(lemma="quick", language_code="en", features={"pos": "ADJ"})
    )
    lexicon = lexicon.with_item(
        LexicalItem(lemma="lazy", language_code="en", features={"pos": "ADJ"})
    )

    return lexicon


@pytest.fixture
def simple_template() -> Template:
    """Create simple template with two slots."""
    return Template(
        name="simple_transitive",
        template_string="{subject} {verb} it",
        slots={
            "subject": Slot(
                name="subject",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'NOUN'")
                ],
            ),
            "verb": Slot(
                name="verb",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'VERB'")
                ],
            ),
        },
    )


def test_filled_template_model() -> None:
    """Test FilledTemplate model creation."""
    item1 = LexicalItem(lemma="cat", language_code="en", features={"pos": "NOUN"})
    item2 = LexicalItem(lemma="broke", language_code="en", features={"pos": "VERB"})

    filled = FilledTemplate(
        template_id="t1",
        template_name="test",
        slot_fillers={"subject": item1, "verb": item2},
        rendered_text="cat broke it",
        strategy_name="exhaustive",
    )

    assert filled.template_id == "t1"
    assert filled.template_name == "test"
    assert filled.slot_fillers["subject"].lemma == "cat"
    assert filled.slot_fillers["verb"].lemma == "broke"
    assert filled.rendered_text == "cat broke it"
    assert filled.strategy_name == "exhaustive"


def test_filler_exhaustive_strategy(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test filling with exhaustive strategy."""
    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())
    filled = filler.fill(simple_template)

    # 2 nouns * 2 verbs = 4 combinations
    assert len(filled) == 4

    # Check all have correct structure
    assert all(isinstance(f, FilledTemplate) for f in filled)
    assert all(f.template_name == "simple_transitive" for f in filled)
    assert all(f.strategy_name == "exhaustive" for f in filled)

    # Check rendered texts
    rendered_texts = {f.rendered_text for f in filled}
    assert rendered_texts == {
        "cat broke it",
        "cat ate it",
        "dog broke it",
        "dog ate it",
    }


def test_filler_random_strategy(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test filling with random strategy."""
    strategy = RandomStrategy(n_samples=2, seed=42)
    filler = StrategyFiller(sample_lexicon, strategy=strategy)
    filled = filler.fill(simple_template)

    # Should generate 2 samples
    assert len(filled) == 2
    assert all(f.strategy_name == "random" for f in filled)


def test_filler_random_strategy_deterministic(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test random strategy is deterministic with seed."""
    strategy1 = RandomStrategy(n_samples=5, seed=42)
    filler1 = StrategyFiller(sample_lexicon, strategy=strategy1)
    filled1 = filler1.fill(simple_template)

    strategy2 = RandomStrategy(n_samples=5, seed=42)
    filler2 = StrategyFiller(sample_lexicon, strategy=strategy2)
    filled2 = filler2.fill(simple_template)

    # Same results with same seed
    assert len(filled1) == len(filled2)
    for f1, f2 in zip(filled1, filled2, strict=True):
        assert f1.rendered_text == f2.rendered_text


def test_filler_stratified_strategy(sample_lexicon: Lexicon) -> None:
    """Test filling with stratified strategy."""
    # Create template with single slot
    template = Template(
        name="single_slot",
        template_string="The {word} thing",
        slots={
            "word": Slot(
                name="word",
                constraints=[
                    Constraint(expression="self.features.get('pos') in ['NOUN', 'ADJ']")
                ],
            ),
        },
    )

    strategy = StratifiedStrategy(n_samples=10, grouping_property="pos", seed=42)
    filler = StrategyFiller(sample_lexicon, strategy=strategy)
    filled = filler.fill(template)

    assert len(filled) == 10
    assert all(f.strategy_name == "stratified" for f in filled)


def test_filler_language_filtering(sample_lexicon: Lexicon) -> None:
    """Test language code filtering."""
    # Add Spanish items
    sample_lexicon = sample_lexicon.with_item(
        LexicalItem(lemma="gato", language_code="es", features={"pos": "NOUN"})
    )
    sample_lexicon = sample_lexicon.with_item(
        LexicalItem(lemma="perro", language_code="es", features={"pos": "NOUN"})
    )

    template = Template(
        name="simple",
        template_string="{noun}",
        slots={
            "noun": Slot(
                name="noun",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'NOUN'")
                ],
            ),
        },
    )

    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())

    # Fill with English only
    filled_en = filler.fill(template, language_code="en")
    rendered_en = {f.rendered_text for f in filled_en}
    assert rendered_en == {"cat", "dog"}

    # Fill with Spanish only
    filled_es = filler.fill(template, language_code="es")
    rendered_es = {f.rendered_text for f in filled_es}
    assert rendered_es == {"gato", "perro"}


def test_filler_empty_slot_error(sample_lexicon: Lexicon) -> None:
    """Test error when slot has no valid items."""
    template = Template(
        name="impossible",
        template_string="{adverb}",
        slots={
            "adverb": Slot(
                name="adverb",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'ADV'")
                ],
            ),
        },
    )

    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())

    with pytest.raises(ValueError, match="No valid items for slots"):
        filler.fill(template)


def test_filler_no_constraint_slot(sample_lexicon: Lexicon) -> None:
    """Test slot with no constraint accepts all items."""
    template = Template(
        name="any_word",
        template_string="The {word}",
        slots={"word": Slot(name="word", constraints=[])},
    )

    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())
    filled = filler.fill(template)

    # Should include all items in lexicon (6 items)
    assert len(filled) == 6


def test_filler_extensional_constraint(sample_lexicon: Lexicon) -> None:
    """Test filling with extensional constraint."""
    # Get IDs of cat and dog
    cat_id = None
    dog_id = None
    for item in sample_lexicon.items:
        item_id = item.id
        if item.lemma == "cat":
            cat_id = str(item_id)
        elif item.lemma == "dog":
            dog_id = str(item_id)

    assert cat_id is not None
    assert dog_id is not None

    template = Template(
        name="specific_animals",
        template_string="The {animal}",
        slots={
            "animal": Slot(
                name="animal",
                constraints=[
                    Constraint(expression=f"str(self.id) in ['{cat_id}', '{dog_id}']")
                ],
            ),
        },
    )

    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())
    filled = filler.fill(template)

    assert len(filled) == 2
    rendered = {f.rendered_text for f in filled}
    assert rendered == {"The cat", "The dog"}


def test_filler_count_combinations(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test counting combinations."""
    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())
    count = filler.count_combinations(simple_template)

    # 2 nouns * 2 verbs = 4
    assert count == 4


def test_filler_count_combinations_three_slots(sample_lexicon: Lexicon) -> None:
    """Test counting with three slots."""
    template = Template(
        name="three_slots",
        template_string="{adj} {noun} {verb}",
        slots={
            "adj": Slot(
                name="adj",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'ADJ'")
                ],
            ),
            "noun": Slot(
                name="noun",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'NOUN'")
                ],
            ),
            "verb": Slot(
                name="verb",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'VERB'")
                ],
            ),
        },
    )

    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())
    count = filler.count_combinations(template)

    # 2 adjectives * 2 nouns * 2 verbs = 8
    assert count == 8


def test_filler_render_template(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test template rendering."""
    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())
    filled = filler.fill(simple_template)

    # All rendered texts should have slots replaced
    for f in filled:
        assert "{" not in f.rendered_text
        assert "}" not in f.rendered_text
        assert " it" in f.rendered_text


def test_filler_metadata_preservation(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test that metadata is preserved in filled templates."""
    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())
    filled = filler.fill(simple_template)

    for f in filled:
        # Check template metadata
        assert f.template_id == str(simple_template.id)
        assert f.template_name == simple_template.name

        # Check slot fillers
        assert "subject" in f.slot_fillers
        assert "verb" in f.slot_fillers
        assert isinstance(f.slot_fillers["subject"], LexicalItem)
        assert isinstance(f.slot_fillers["verb"], LexicalItem)


def test_filler_single_slot_template(sample_lexicon: Lexicon) -> None:
    """Test filling template with single slot."""
    template = Template(
        name="single",
        template_string="The {noun}",
        slots={
            "noun": Slot(
                name="noun",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'NOUN'")
                ],
            ),
        },
    )

    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())
    filled = filler.fill(template)

    assert len(filled) == 2
    rendered = {f.rendered_text for f in filled}
    assert rendered == {"The cat", "The dog"}


def test_filler_complex_template(sample_lexicon: Lexicon) -> None:
    """Test filling complex template with multiple constraints."""
    template = Template(
        name="complex",
        template_string="{adj1} {noun1} and {adj2} {noun2}",
        slots={
            "adj1": Slot(
                name="adj1",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'ADJ'")
                ],
            ),
            "noun1": Slot(
                name="noun1",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'NOUN'")
                ],
            ),
            "adj2": Slot(
                name="adj2",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'ADJ'")
                ],
            ),
            "noun2": Slot(
                name="noun2",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'NOUN'")
                ],
            ),
        },
    )

    filler = StrategyFiller(sample_lexicon, strategy=ExhaustiveStrategy())
    count = filler.count_combinations(template)

    # 2 adj * 2 noun * 2 adj * 2 noun = 16
    assert count == 16

    filled = filler.fill(template)
    assert len(filled) == 16


def test_filler_empty_lexicon() -> None:
    """Test filling with empty lexicon."""
    lexicon = Lexicon(name="empty")
    template = Template(
        name="simple",
        template_string="{word}",
        slots={
            "word": Slot(
                name="word",
                constraints=[
                    Constraint(expression="self.features.get('pos') == 'NOUN'")
                ],
            ),
        },
    )

    filler = StrategyFiller(lexicon, strategy=ExhaustiveStrategy())

    with pytest.raises(ValueError, match="No valid items for slots"):
        filler.fill(template)
