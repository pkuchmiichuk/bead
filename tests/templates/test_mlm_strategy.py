"""Test MLM-based filling strategy with constraint system."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import MLMFillingStrategy


@pytest.fixture
def resolver() -> ConstraintResolver:
    """Create constraint resolver."""
    return ConstraintResolver()


@pytest.fixture
def mock_model_adapter() -> MagicMock:
    """Create mock model adapter."""
    adapter = MagicMock()
    adapter.model_name = "test-model"
    adapter.is_loaded.return_value = True
    adapter.get_mask_token.return_value = "[MASK]"
    # Mock predict_masked_token to return predictions
    adapter.predict_masked_token.return_value = [
        ("run", -1.0),
        ("walk", -2.0),
        ("jump", -3.0),
    ]

    # Mock predict_masked_token_batch to return the right number of predictions
    def batch_predict(texts, **kwargs):
        return [[("run", -1.0), ("walk", -2.0), ("jump", -3.0)] for _ in texts]

    adapter.predict_masked_token_batch.side_effect = batch_predict
    return adapter


@pytest.fixture
def sample_lexicon() -> Lexicon:
    """Create sample lexicon with motion verbs."""
    lexicon = Lexicon(name="test_verbs")
    items = [
        LexicalItem(lemma="run", language_code="en", features={"pos": "VERB"}),
        LexicalItem(lemma="walk", language_code="en", features={"pos": "VERB"}),
        LexicalItem(lemma="jump", language_code="en", features={"pos": "VERB"}),
        LexicalItem(lemma="sit", language_code="en", features={"pos": "VERB"}),
    ]
    for item in items:
        lexicon = lexicon.with_item(item)
    return lexicon


@pytest.fixture
def constrained_template() -> Template:
    """Create template with DSL constraint."""
    # Constraint: verb must be VERB pos
    constraint = Constraint(expression="self.features.get('pos') == 'VERB'")

    slot = Slot(name="verb", constraints=[constraint])

    return Template(
        name="test",
        template_string="{verb}",
        slots={"verb": slot},
    )


def test_mlm_strategy_constraint_checking(
    resolver: ConstraintResolver,
    mock_model_adapter: MagicMock,
    sample_lexicon: Lexicon,
    constrained_template: Template,
) -> None:
    """Test MLMFillingStrategy correctly evaluates constraints."""
    strategy = MLMFillingStrategy(
        resolver=resolver,
        model_adapter=mock_model_adapter,
        top_k=10,
    )

    # Get candidates for the verb slot
    slot = constrained_template.slots["verb"]
    candidates = strategy._get_mlm_candidates(
        template=constrained_template,
        slot_names=["verb"],
        slot_idx=0,
        filled_slots={},
        slot=slot,
        lexicons=[sample_lexicon],
        language_code="en",
    )

    # Should return candidates that match both lemma and constraints
    assert len(candidates) > 0
    # All candidates should be verbs
    for item, _log_prob in candidates:
        assert item.features.get("pos") == "VERB"
    # Should have found matching items
    lemmas = {item.lemma for item, _ in candidates}
    assert "run" in lemmas or "walk" in lemmas or "jump" in lemmas


def test_mlm_strategy_no_constraints(
    resolver: ConstraintResolver,
    mock_model_adapter: MagicMock,
    sample_lexicon: Lexicon,
) -> None:
    """Test MLMFillingStrategy works without constraints."""
    strategy = MLMFillingStrategy(
        resolver=resolver,
        model_adapter=mock_model_adapter,
        top_k=10,
    )

    # Template with no constraints
    slot = Slot(name="verb")
    template = Template(
        name="test",
        template_string="{verb}",
        slots={"verb": slot},
    )

    candidates = strategy._get_mlm_candidates(
        template=template,
        slot_names=["verb"],
        slot_idx=0,
        filled_slots={},
        slot=slot,
        lexicons=[sample_lexicon],
        language_code="en",
    )

    # Should return candidates that match lemma (no filtering by constraints)
    assert len(candidates) > 0
    lemmas = {item.lemma for item, _ in candidates}
    assert "run" in lemmas or "walk" in lemmas or "jump" in lemmas


def test_mlm_strategy_extensional_constraint(
    resolver: ConstraintResolver,
    mock_model_adapter: MagicMock,
    sample_lexicon: Lexicon,
) -> None:
    """Test MLMFillingStrategy with extensional (whitelist) constraint."""
    # Get IDs of specific items
    run_item = next(i for i in sample_lexicon.items if i.lemma == "run")
    walk_item = next(i for i in sample_lexicon.items if i.lemma == "walk")

    # Constraint: only allow "run" and "walk"
    constraint = Constraint(
        expression="self.id in allowed_verbs",
        context={"allowed_verbs": (str(run_item.id), str(walk_item.id))},
    )

    slot = Slot(name="verb", constraints=[constraint])
    template = Template(
        name="test",
        template_string="{verb}",
        slots={"verb": slot},
    )

    strategy = MLMFillingStrategy(
        resolver=resolver,
        model_adapter=mock_model_adapter,
        top_k=10,
    )

    candidates = strategy._get_mlm_candidates(
        template=template,
        slot_names=["verb"],
        slot_idx=0,
        filled_slots={},
        slot=slot,
        lexicons=[sample_lexicon],
        language_code="en",
    )

    # Should only return run and walk
    lemmas = {item.lemma for item, _ in candidates}
    assert lemmas <= {"run", "walk"}  # Subset of allowed items


def test_mlm_strategy_max_fills(
    resolver: ConstraintResolver,
    mock_model_adapter: MagicMock,
    sample_lexicon: Lexicon,
) -> None:
    """Test MLMFillingStrategy with max_fills limiting candidates."""
    constraint = Constraint(expression="self.features.get('pos') == 'VERB'")
    slot = Slot(name="verb", constraints=[constraint])
    template = Template(
        name="test",
        template_string="{verb}",
        slots={"verb": slot},
    )

    strategy = MLMFillingStrategy(
        resolver=resolver,
        model_adapter=mock_model_adapter,
        top_k=10,
    )

    # Get candidates with max_fills=2
    candidates = strategy._get_mlm_candidates(
        template=template,
        slot_names=["verb"],
        slot_idx=0,
        filled_slots={},
        slot=slot,
        lexicons=[sample_lexicon],
        language_code="en",
        max_fills=2,
    )

    # Should return at most 2 candidates
    assert len(candidates) <= 2


def test_mlm_strategy_enforce_unique(
    resolver: ConstraintResolver,
    mock_model_adapter: MagicMock,
    sample_lexicon: Lexicon,
) -> None:
    """Test MLMFillingStrategy with uniqueness enforcement."""
    constraint = Constraint(expression="self.features.get('pos') == 'VERB'")
    slot = Slot(name="verb", constraints=[constraint])
    template = Template(
        name="test",
        template_string="{verb}",
        slots={"verb": slot},
    )

    # Get IDs of items to mark as seen
    run_item = next(i for i in sample_lexicon.items if i.lemma == "run")
    seen_items = {run_item.id}

    strategy = MLMFillingStrategy(
        resolver=resolver,
        model_adapter=mock_model_adapter,
        top_k=10,
    )

    # Get candidates with seen_items
    candidates = strategy._get_mlm_candidates(
        template=template,
        slot_names=["verb"],
        slot_idx=0,
        filled_slots={},
        slot=slot,
        lexicons=[sample_lexicon],
        language_code="en",
        seen_items=seen_items,
    )

    # Should not return "run" since it's in seen_items
    lemmas = {item.lemma for item, _ in candidates}
    assert "run" not in lemmas


def test_mlm_strategy_max_fills_with_enforce_unique(
    resolver: ConstraintResolver,
    mock_model_adapter: MagicMock,
    sample_lexicon: Lexicon,
) -> None:
    """Test MLMFillingStrategy with both max_fills and enforce_unique."""
    constraint = Constraint(expression="self.features.get('pos') == 'VERB'")
    slot = Slot(name="verb", constraints=[constraint])
    template = Template(
        name="test",
        template_string="{verb}",
        slots={"verb": slot},
    )

    # Mark one item as seen
    run_item = next(i for i in sample_lexicon.items if i.lemma == "run")
    seen_items = {run_item.id}

    strategy = MLMFillingStrategy(
        resolver=resolver,
        model_adapter=mock_model_adapter,
        top_k=10,
    )

    # Get candidates with both max_fills and seen_items
    candidates = strategy._get_mlm_candidates(
        template=template,
        slot_names=["verb"],
        slot_idx=0,
        filled_slots={},
        slot=slot,
        lexicons=[sample_lexicon],
        language_code="en",
        seen_items=seen_items,
        max_fills=2,
    )

    # Should return at most 2 candidates, excluding "run"
    assert len(candidates) <= 2
    lemmas = {item.lemma for item, _ in candidates}
    assert "run" not in lemmas


def test_mlm_strategy_per_slot_config(
    resolver: ConstraintResolver,
    mock_model_adapter: MagicMock,
    sample_lexicon: Lexicon,
) -> None:
    """Test MLMFillingStrategy with per-slot max_fills and enforce_unique."""
    constraint = Constraint(expression="self.features.get('pos') == 'VERB'")
    slot = Slot(name="verb", constraints=[constraint])
    template = Template(
        name="test",
        template_string="{verb}",
        slots={"verb": slot},
    )

    strategy = MLMFillingStrategy(
        resolver=resolver,
        model_adapter=mock_model_adapter,
        top_k=10,
        per_slot_max_fills={"verb": 2},
        per_slot_enforce_unique={"verb": True},
    )

    # Generate from template
    results = list(
        strategy.generate_from_template(
            template=template,
            lexicons=[sample_lexicon],
            language_code="en",
        )
    )

    # Should generate results
    assert len(results) > 0

    # All results should have different verb items (due to enforce_unique)
    verb_ids = [r["verb"].id for r in results]
    assert len(verb_ids) == len(set(verb_ids))  # All unique

    # Should have at most 2 different verbs (due to max_fills=2)
    assert len(set(verb_ids)) <= 2
