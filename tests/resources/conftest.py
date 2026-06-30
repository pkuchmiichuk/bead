"""Pytest fixtures for resource tests."""

from __future__ import annotations

from uuid import UUID

import pytest

from bead.resources import (
    Constraint,
    LexicalItem,
    Lexicon,
    Slot,
    Template,
    TemplateCollection,
)
from bead.resources.classification import LexicalItemClass, TemplateClass


@pytest.fixture
def sample_lexical_item() -> LexicalItem:
    """Provide a sample lexical item."""
    return LexicalItem(
        lemma="walk",
        language_code="eng",
        features={
            "pos": "VERB",
            "tense": "present",
            "transitive": True,
            "frequency": 1000,
        },
        source="manual",
    )


@pytest.fixture
def sample_noun() -> LexicalItem:
    """Provide a sample noun."""
    return LexicalItem(
        lemma="dog",
        language_code="eng",
        features={
            "pos": "NOUN",
            "number": "singular",
            "animacy": "animate",
            "frequency": 500,
        },
    )


@pytest.fixture
def sample_extensional_constraint(
    sample_lexical_item: LexicalItem,
) -> Constraint:
    """Provide a sample extensional constraint."""
    return Constraint(
        expression=f"self.id in [{sample_lexical_item.id}]",
    )


@pytest.fixture
def sample_intensional_constraint() -> Constraint:
    """Provide a sample intensional constraint."""
    return Constraint(
        expression="self.features.get('pos') == 'VERB'",
    )


@pytest.fixture
def sample_relational_constraint() -> Constraint:
    """Provide a sample relational constraint."""
    return Constraint(
        expression="slots.subject.lemma != slots.object.lemma",
    )


@pytest.fixture
def sample_dsl_constraint() -> Constraint:
    """Provide a sample DSL constraint."""
    return Constraint(
        expression="self.features.get('pos') == 'VERB' and len(self.lemma) > 3",
    )


@pytest.fixture
def sample_slot(sample_intensional_constraint: Constraint) -> Slot:
    """Provide a sample slot."""
    return Slot(
        name="subject",
        description="The subject of the sentence",
        constraints=[sample_intensional_constraint],
        required=True,
    )


@pytest.fixture
def sample_template(sample_slot: Slot) -> Template:
    """Provide a sample template."""
    verb_slot = Slot(name="verb", required=True)
    object_slot = Slot(name="object", required=True)
    return Template(
        name="simple_transitive",
        template_string="{subject} {verb} {object}.",
        slots={
            "subject": sample_slot,
            "verb": verb_slot,
            "object": object_slot,
        },
        tags=["transitive", "simple"],
    )


@pytest.fixture
def sample_lexicon() -> Lexicon:
    """Provide a sample lexicon with multiple items."""
    lexicon = Lexicon(name="test_lexicon", language_code="en")
    lexicon = lexicon.with_item(
        LexicalItem(
            lemma="walk",
            language_code="eng",
            features={"pos": "VERB", "frequency": 1000},
        )
    )
    lexicon = lexicon.with_item(
        LexicalItem(
            lemma="run", language_code="eng", features={"pos": "VERB", "frequency": 800}
        )
    )
    lexicon = lexicon.with_item(
        LexicalItem(
            lemma="dog", language_code="eng", features={"pos": "NOUN", "frequency": 500}
        )
    )
    return lexicon


@pytest.fixture
def sample_template_collection(sample_template: Template) -> TemplateCollection:
    """Provide a sample template collection."""
    collection = TemplateCollection(name="test_collection")
    collection = collection.with_template(sample_template)

    slot_x = Slot(name="x", required=True)
    template2 = Template(
        name="intransitive",
        template_string="{x} happened.",
        slots={"x": slot_x},
        tags=["intransitive", "simple"],
    )
    collection = collection.with_template(template2)

    return collection


@pytest.fixture
def english_item() -> LexicalItem:
    """Provide an English lexical item."""
    return LexicalItem(lemma="walk", language_code="en", features={"pos": "VERB"})


@pytest.fixture
def korean_item() -> LexicalItem:
    """Provide a Korean lexical item."""
    return LexicalItem(lemma="먹다", language_code="ko", features={"pos": "VERB"})


@pytest.fixture
def english_template() -> Template:
    """Provide an English template."""
    slot = Slot(name="x", required=True)
    return Template(
        name="english_test",
        template_string="{x}.",
        slots={"x": slot},
        language_code="en",
    )


@pytest.fixture
def korean_template() -> Template:
    """Provide a Korean template."""
    slot = Slot(name="x", required=True)
    return Template(
        name="korean_test",
        template_string="{x}.",
        slots={"x": slot},
        language_code="ko",
    )


# Classification fixtures


@pytest.fixture
def english_causative_verbs() -> dict[UUID, LexicalItem]:
    """Sample English causative verbs for testing."""
    item1 = LexicalItem(
        lemma="break",
        language_code="en",
        features={
            "pos": "VERB",
            "causative": True,
            "transitive": True,
            "frequency": 500,
        },
    )
    item2 = LexicalItem(
        lemma="open",
        language_code="en",
        features={
            "pos": "VERB",
            "causative": True,
            "transitive": True,
            "frequency": 800,
        },
    )
    item3 = LexicalItem(
        lemma="close",
        language_code="en",
        features={
            "pos": "VERB",
            "causative": True,
            "transitive": True,
            "frequency": 700,
        },
    )
    return {item1.id: item1, item2.id: item2, item3.id: item3}


@pytest.fixture
def korean_causative_verbs() -> dict[UUID, LexicalItem]:
    """Sample Korean causative verbs for testing."""
    item1 = LexicalItem(
        lemma="kkakta",
        language_code="ko",
        features={
            "pos": "VERB",
            "causative": True,
            "transitive": True,
            "frequency": 400,
        },
    )
    item2 = LexicalItem(
        lemma="yeolda",
        language_code="ko",
        features={
            "pos": "VERB",
            "causative": True,
            "transitive": True,
            "frequency": 600,
        },
    )
    return {item1.id: item1, item2.id: item2}


@pytest.fixture
def zulu_causative_verbs() -> dict[UUID, LexicalItem]:
    """Sample Zulu causative verbs for testing."""
    item1 = LexicalItem(
        lemma="phula",
        language_code="zu",
        features={
            "pos": "VERB",
            "causative": True,
            "transitive": True,
            "frequency": 300,
        },
    )
    return {item1.id: item1}


@pytest.fixture
def multilingual_causative_class(
    english_causative_verbs: dict[UUID, LexicalItem],
    korean_causative_verbs: dict[UUID, LexicalItem],
) -> LexicalItemClass:
    """Sample multilingual causative verb class."""
    cls = LexicalItemClass(
        name="causative_verbs_crossling",
        description="Causative verbs across English and Korean",
        property_name="causative",
        property_value=True,
        tags=["causative", "cross-linguistic", "verbs"],
    )
    for item in english_causative_verbs.values():
        cls = cls.with_item(item)
    for item in korean_causative_verbs.values():
        cls = cls.with_item(item)
    return cls


@pytest.fixture
def monolingual_causative_class(
    english_causative_verbs: dict[UUID, LexicalItem],
) -> LexicalItemClass:
    """Sample monolingual causative verb class (English only)."""
    cls = LexicalItemClass(
        name="causative_verbs_en",
        description="Causative verbs in English",
        property_name="causative",
        property_value=True,
        tags=["causative", "english", "verbs"],
    )
    for item in english_causative_verbs.values():
        cls = cls.with_item(item)
    return cls


@pytest.fixture
def english_transitive_templates() -> dict[UUID, Template]:
    """Sample English transitive templates for testing."""
    t1 = Template(
        name="svo_simple",
        template_string="{subject} {verb} {object}.",
        slots={
            "subject": Slot(name="subject"),
            "verb": Slot(name="verb"),
            "object": Slot(name="object"),
        },
        language_code="en",
        tags=["transitive", "simple"],
    )
    t2 = Template(
        name="svo_with_adverb",
        template_string="{subject} {adverb} {verb} {object}.",
        slots={
            "subject": Slot(name="subject"),
            "adverb": Slot(name="adverb"),
            "verb": Slot(name="verb"),
            "object": Slot(name="object"),
        },
        language_code="en",
        tags=["transitive", "adverb"],
    )
    return {t1.id: t1, t2.id: t2}


@pytest.fixture
def korean_transitive_templates() -> dict[UUID, Template]:
    """Sample Korean transitive templates for testing."""
    t1 = Template(
        name="sov_simple",
        template_string="{subject} {object} {verb}.",
        slots={
            "subject": Slot(name="subject"),
            "object": Slot(name="object"),
            "verb": Slot(name="verb"),
        },
        language_code="ko",
        tags=["transitive", "simple"],
    )
    t2 = Template(
        name="sov_with_adverb",
        template_string="{subject} {adverb} {object} {verb}.",
        slots={
            "subject": Slot(name="subject"),
            "adverb": Slot(name="adverb"),
            "object": Slot(name="object"),
            "verb": Slot(name="verb"),
        },
        language_code="ko",
        tags=["transitive", "adverb"],
    )
    return {t1.id: t1, t2.id: t2}


@pytest.fixture
def multilingual_transitive_template_class(
    english_transitive_templates: dict[UUID, Template],
    korean_transitive_templates: dict[UUID, Template],
) -> TemplateClass:
    """Sample multilingual transitive template class."""
    cls = TemplateClass(
        name="transitive_templates_crossling",
        description="Transitive templates across English and Korean",
        property_name="transitive",
        property_value=True,
        tags=["transitive", "cross-linguistic"],
    )
    for template in english_transitive_templates.values():
        cls = cls.with_template(template)
    for template in korean_transitive_templates.values():
        cls = cls.with_template(template)
    return cls


@pytest.fixture
def monolingual_transitive_template_class(
    english_transitive_templates: dict[UUID, Template],
) -> TemplateClass:
    """Sample monolingual transitive template class (English only)."""
    cls = TemplateClass(
        name="transitive_templates_en",
        description="Transitive templates in English",
        property_name="transitive",
        property_value=True,
        tags=["transitive", "english"],
    )
    for template in english_transitive_templates.values():
        cls = cls.with_template(template)
    return cls
