"""Test streaming template filler."""

from __future__ import annotations

import pytest

from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template
from bead.templates.filler import FilledTemplate
from bead.templates.streaming import StreamingFiller


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
    lexicon = lexicon.with_item(
        LexicalItem(lemma="bird", language_code="en", features={"pos": "NOUN"})
    )

    # Add verbs
    lexicon = lexicon.with_item(
        LexicalItem(lemma="broke", language_code="en", features={"pos": "VERB"})
    )
    lexicon = lexicon.with_item(
        LexicalItem(lemma="ate", language_code="en", features={"pos": "VERB"})
    )
    lexicon = lexicon.with_item(
        LexicalItem(lemma="found", language_code="en", features={"pos": "VERB"})
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


def test_streaming_basic(sample_lexicon: Lexicon, simple_template: Template) -> None:
    """Test basic streaming functionality."""
    filler = StreamingFiller(sample_lexicon)
    filled_list = list(filler.stream(simple_template))

    # 3 nouns * 3 verbs = 9 combinations
    assert len(filled_list) == 9
    assert all(isinstance(f, FilledTemplate) for f in filled_list)
    assert all(f.strategy_name == "streaming" for f in filled_list)


def test_streaming_lazy_evaluation(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test that streaming uses lazy evaluation."""
    filler = StreamingFiller(sample_lexicon)
    stream = filler.stream(simple_template)

    # Take first 3 items
    first_three = []
    for i, filled in enumerate(stream):
        if i >= 3:
            break
        first_three.append(filled)

    assert len(first_three) == 3
    assert all(isinstance(f, FilledTemplate) for f in first_three)


def test_streaming_max_combinations(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test max_combinations parameter."""
    filler = StreamingFiller(sample_lexicon, max_combinations=5)
    filled_list = list(filler.stream(simple_template))

    # Should stop at 5 combinations
    assert len(filled_list) == 5


def test_streaming_early_termination(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test early termination of stream."""
    filler = StreamingFiller(sample_lexicon)
    stream = filler.stream(simple_template)

    # Only consume first item
    first = next(stream)

    assert isinstance(first, FilledTemplate)
    assert first.strategy_name == "streaming"


def test_streaming_language_filtering(sample_lexicon: Lexicon) -> None:
    """Test language code filtering in streaming."""
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

    filler = StreamingFiller(sample_lexicon)

    # Stream with English only
    filled_en = list(filler.stream(template, language_code="en"))
    rendered_en = {f.rendered_text for f in filled_en}
    assert rendered_en == {"cat", "dog", "bird"}

    # Stream with Spanish only
    filled_es = list(filler.stream(template, language_code="es"))
    rendered_es = {f.rendered_text for f in filled_es}
    assert rendered_es == {"gato", "perro"}


def test_streaming_empty_slot_error(sample_lexicon: Lexicon) -> None:
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

    filler = StreamingFiller(sample_lexicon)

    with pytest.raises(ValueError, match="No valid items for slots"):
        list(filler.stream(template))


def test_streaming_single_slot(sample_lexicon: Lexicon) -> None:
    """Test streaming with single slot template."""
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

    filler = StreamingFiller(sample_lexicon)
    filled = list(filler.stream(template))

    assert len(filled) == 3
    rendered = {f.rendered_text for f in filled}
    assert rendered == {"The cat", "The dog", "The bird"}


def test_streaming_metadata(sample_lexicon: Lexicon, simple_template: Template) -> None:
    """Test that metadata is preserved in streamed templates."""
    filler = StreamingFiller(sample_lexicon, max_combinations=3)
    filled_list = list(filler.stream(simple_template))

    for f in filled_list:
        assert f.template_id == str(simple_template.id)
        assert f.template_name == simple_template.name
        assert "subject" in f.slot_fillers
        assert "verb" in f.slot_fillers
        assert isinstance(f.slot_fillers["subject"], LexicalItem)
        assert isinstance(f.slot_fillers["verb"], LexicalItem)


def test_streaming_rendered_text(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test template rendering in streaming."""
    filler = StreamingFiller(sample_lexicon, max_combinations=5)
    filled_list = list(filler.stream(simple_template))

    # All rendered texts should have slots replaced
    for f in filled_list:
        assert "{" not in f.rendered_text
        assert "}" not in f.rendered_text
        assert " it" in f.rendered_text


def test_streaming_complex_template(sample_lexicon: Lexicon) -> None:
    """Test streaming with complex multi-slot template."""
    # Add adjectives for more complexity
    sample_lexicon = sample_lexicon.with_item(
        LexicalItem(lemma="quick", language_code="en", features={"pos": "ADJ"})
    )
    sample_lexicon = sample_lexicon.with_item(
        LexicalItem(lemma="lazy", language_code="en", features={"pos": "ADJ"})
    )

    template = Template(
        name="complex",
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

    filler = StreamingFiller(sample_lexicon)
    filled = list(filler.stream(template))

    # 2 adj * 3 noun * 3 verb = 18 combinations
    assert len(filled) == 18


def test_streaming_max_combinations_less_than_total(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test max_combinations when less than total combinations."""
    filler = StreamingFiller(sample_lexicon, max_combinations=2)
    filled = list(filler.stream(simple_template))

    # Should only return 2 combinations (not all 9)
    assert len(filled) == 2


def test_streaming_max_combinations_more_than_total(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test max_combinations when more than total combinations."""
    filler = StreamingFiller(sample_lexicon, max_combinations=100)
    filled = list(filler.stream(simple_template))

    # Should return all 9 combinations (not 100)
    assert len(filled) == 9


def test_streaming_no_constraint_slot(sample_lexicon: Lexicon) -> None:
    """Test streaming with no constraint slot."""
    template = Template(
        name="any_word",
        template_string="The {word}",
        slots={"word": Slot(name="word", constraints=[])},
    )

    filler = StreamingFiller(sample_lexicon)
    filled = list(filler.stream(template))

    # Should include all items in lexicon (6 items: 3 nouns + 3 verbs)
    assert len(filled) == 6


def test_streaming_empty_lexicon() -> None:
    """Test streaming with empty lexicon."""
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

    filler = StreamingFiller(lexicon)

    with pytest.raises(ValueError, match="No valid items for slots"):
        list(filler.stream(template))


def test_streaming_iterator_behavior(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test that stream returns a proper iterator."""
    filler = StreamingFiller(sample_lexicon)
    stream = filler.stream(simple_template)

    # Check it's an iterator
    assert hasattr(stream, "__iter__")
    assert hasattr(stream, "__next__")

    # Consume it item by item
    count = 0
    for _ in stream:
        count += 1

    assert count == 9


def test_streaming_multiple_iterations(
    sample_lexicon: Lexicon, simple_template: Template
) -> None:
    """Test that we can create multiple streams."""
    filler = StreamingFiller(sample_lexicon, max_combinations=5)

    # First stream
    filled1 = list(filler.stream(simple_template))
    assert len(filled1) == 5

    # Second stream (should work independently)
    filled2 = list(filler.stream(simple_template))
    assert len(filled2) == 5
