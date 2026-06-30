"""Tests for item scoring utilities."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.item import Item
from bead.items.scoring import ForcedChoiceScorer, ItemScorer, LanguageModelScorer


class TestItemScorer:
    """Test ItemScorer abstract base class."""

    def test_is_abstract(self) -> None:
        """Test that ItemScorer is abstract."""
        with pytest.raises(TypeError):
            ItemScorer()  # type: ignore[abstract]

    def test_subclass_must_implement_score(self) -> None:
        """Test that subclasses must implement score()."""

        class IncompleteScorer(ItemScorer):
            pass

        with pytest.raises(TypeError):
            IncompleteScorer()  # type: ignore[abstract]

    def test_default_score_batch(self) -> None:
        """Test default score_batch() implementation."""

        class SimpleScorer(ItemScorer):
            def score(self, item: Item) -> float:
                return 1.0

        scorer = SimpleScorer()
        items = [
            Item(item_template_id=uuid4(), rendered_elements={"text": "test1"}),
            Item(item_template_id=uuid4(), rendered_elements={"text": "test2"}),
            Item(item_template_id=uuid4(), rendered_elements={"text": "test3"}),
        ]

        scores = scorer.score_batch(items)

        assert len(scores) == 3
        assert all(score == 1.0 for score in scores)

    def test_score_with_metadata(self) -> None:
        """Test score_with_metadata() method."""

        class SimpleScorer(ItemScorer):
            def score(self, item: Item) -> float:
                return 42.0

        scorer = SimpleScorer()
        items = [
            Item(item_template_id=uuid4(), rendered_elements={"text": "test1"}),
            Item(item_template_id=uuid4(), rendered_elements={"text": "test2"}),
        ]

        results = scorer.score_with_metadata(items)

        assert len(results) == 2
        assert all(item.id in results for item in items)
        assert all(results[item.id]["score"] == 42.0 for item in items)


class TestLanguageModelScorer:
    """Test LanguageModelScorer."""

    def test_initialization(self) -> None:
        """Test LanguageModelScorer initialization."""
        scorer = LanguageModelScorer(
            model_name="gpt2",
            cache_dir=Path(".cache/test"),
            device="cpu",
            text_key="sentence",
        )

        assert scorer.model_name == "gpt2"
        assert scorer.cache_dir == Path(".cache/test")
        assert scorer.device == "cpu"
        assert scorer.text_key == "sentence"

    def test_initialization_with_string_cache_dir(self) -> None:
        """Test initialization with string cache_dir."""
        scorer = LanguageModelScorer(
            model_name="gpt2", cache_dir=".cache/test", device="cpu"
        )

        assert scorer.cache_dir == Path(".cache/test")

    def test_initialization_with_none_cache_dir(self) -> None:
        """Test initialization with None cache_dir."""
        scorer = LanguageModelScorer(model_name="gpt2", cache_dir=None, device="cpu")

        assert scorer.cache_dir is None

    def test_default_text_key(self) -> None:
        """Test default text_key is 'text'."""
        scorer = LanguageModelScorer(model_name="gpt2", device="cpu")

        assert scorer.text_key == "text"

    def test_score_missing_text_key_raises_error(self) -> None:
        """Test that missing text_key raises KeyError."""
        scorer = LanguageModelScorer(
            model_name="gpt2", cache_dir=None, device="cpu", text_key="sentence"
        )

        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "test"},  # Wrong key
        )

        with pytest.raises(KeyError, match="sentence"):
            scorer.score(item)

    def test_score_batch_missing_text_key_raises_error(self) -> None:
        """Test that score_batch raises KeyError for missing text_key."""
        scorer = LanguageModelScorer(
            model_name="gpt2", cache_dir=None, device="cpu", text_key="sentence"
        )

        items = [
            Item(item_template_id=uuid4(), rendered_elements={"text": "test1"}),
            Item(item_template_id=uuid4(), rendered_elements={"text": "test2"}),
        ]

        with pytest.raises(KeyError, match="sentence"):
            scorer.score_batch(items)

    def test_score_with_metadata_includes_model_name(self) -> None:
        """Test that score_with_metadata includes model name."""

        # Create a mock scorer that doesn't actually load the model
        class MockLMScorer(LanguageModelScorer):
            @property
            def model(self):
                # Return a mock model
                class MockModel:
                    def compute_log_probability(self, text):
                        return -1.5

                return MockModel()

        scorer = MockLMScorer(model_name="gpt2", device="cpu")
        items = [Item(item_template_id=uuid4(), rendered_elements={"text": "test"})]

        results = scorer.score_with_metadata(items)

        assert results[items[0].id]["model"] == "gpt2"
        assert results[items[0].id]["log_probability"] == -1.5
        assert results[items[0].id]["score"] == -1.5


class TestForcedChoiceScorer:
    """Test ForcedChoiceScorer."""

    def test_initialization(self) -> None:
        """Test ForcedChoiceScorer initialization."""

        class DummyScorer(ItemScorer):
            def score(self, item: Item) -> float:
                return 1.0

        base_scorer = DummyScorer()
        fc_scorer = ForcedChoiceScorer(base_scorer=base_scorer, option_prefix="opt")

        assert fc_scorer.base_scorer is base_scorer
        assert fc_scorer.option_prefix == "opt"

    def test_default_option_prefix(self) -> None:
        """Test default option_prefix is 'option'."""

        class DummyScorer(ItemScorer):
            def score(self, item: Item) -> float:
                return 1.0

        fc_scorer = ForcedChoiceScorer(base_scorer=DummyScorer())

        assert fc_scorer.option_prefix == "option"

    def test_score_with_two_options(self) -> None:
        """Test scoring 2AFC item."""

        class DummyScorer(ItemScorer):
            def score(self, item: Item) -> float:
                text = item.rendered_elements.get("text", "")
                return len(text)  # Score by length

        fc_scorer = ForcedChoiceScorer(
            base_scorer=DummyScorer(),
            comparison_fn=lambda scores: max(scores) - min(scores),  # Range
        )

        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "option_a": "short",  # length 5
                "option_b": "much longer text",  # length 16
            },
        )

        score = fc_scorer.score(item)

        # Range: 16 - 5 = 11
        assert score == 11.0

    def test_score_with_three_options(self) -> None:
        """Test scoring 3AFC item."""

        class DummyScorer(ItemScorer):
            def score(self, item: Item) -> float:
                text = item.rendered_elements.get("text", "")
                return float(len(text))

        fc_scorer = ForcedChoiceScorer(
            base_scorer=DummyScorer(),
            comparison_fn=lambda scores: max(scores) - min(scores),
        )

        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "option_a": "x",  # length 1
                "option_b": "xy",  # length 2
                "option_c": "xyz",  # length 3
            },
        )

        score = fc_scorer.score(item)

        # Range: 3 - 1 = 2
        assert score == 2.0

    def test_score_with_precomputed_scores_numeric(self) -> None:
        """Test scoring with precomputed numeric scores (lm_score_0, lm_score_1)."""

        class DummyScorer(ItemScorer):
            def score(self, item: Item) -> float:
                return 999.0  # Should not be called

        fc_scorer = ForcedChoiceScorer(
            base_scorer=DummyScorer(),
            comparison_fn=lambda scores: max(scores) - min(scores),
        )

        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"option_a": "test1", "option_b": "test2"},
            item_metadata={
                "lm_score_0": -2.5,
                "lm_score_1": -1.0,
            },
        )

        score = fc_scorer.score(item)

        # Uses precomputed scores: -1.0 - (-2.5) = 1.5
        assert score == 1.5

    def test_score_with_precomputed_scores_letters(self) -> None:
        """Test scoring with precomputed letter scores (lm_score_a, lm_score_b)."""

        class DummyScorer(ItemScorer):
            def score(self, item: Item) -> float:
                return 999.0  # Should not be called

        fc_scorer = ForcedChoiceScorer(
            base_scorer=DummyScorer(),
            comparison_fn=lambda scores: max(scores) - min(scores),
        )

        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"option_a": "test1", "option_b": "test2"},
            item_metadata={
                "lm_score_a": -3.0,
                "lm_score_b": -1.5,
            },
        )

        score = fc_scorer.score(item)

        # Uses precomputed scores: -1.5 - (-3.0) = 1.5
        assert score == 1.5

    def test_score_without_options_raises_error(self) -> None:
        """Test that item without options raises ValueError."""

        class DummyScorer(ItemScorer):
            def score(self, item: Item) -> float:
                return 1.0

        fc_scorer = ForcedChoiceScorer(base_scorer=DummyScorer())

        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "no options here"},
        )

        with pytest.raises((ValueError, dx.ValidationError), match="has no options"):
            fc_scorer.score(item)

    def test_default_comparison_function(self) -> None:
        """Test default comparison function (standard deviation)."""

        class DummyScorer(ItemScorer):
            def score(self, item: Item) -> float:
                return 1.0

        fc_scorer = ForcedChoiceScorer(base_scorer=DummyScorer())

        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"option_a": "test", "option_b": "test"},
            item_metadata={
                "lm_score_0": 1.0,
                "lm_score_1": 3.0,
            },
        )

        score = fc_scorer.score(item)

        # Standard deviation of [1.0, 3.0] = 1.0
        assert score == pytest.approx(1.0)

    def test_custom_option_prefix(self) -> None:
        """Test using custom option_prefix."""

        class DummyScorer(ItemScorer):
            def score(self, item: Item) -> float:
                text = item.rendered_elements.get("text", "")
                return float(len(text))

        fc_scorer = ForcedChoiceScorer(
            base_scorer=DummyScorer(),
            option_prefix="choice",
            comparison_fn=lambda scores: sum(scores),
        )

        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "choice_a": "ab",  # length 2
                "choice_b": "abc",  # length 3
            },
        )

        score = fc_scorer.score(item)

        # Sum: 2 + 3 = 5
        assert score == 5.0


class TestConcreteImplementation:
    """Test complete concrete implementation."""

    def test_custom_scorer_implementation(self) -> None:
        """Test a custom scorer implementation."""

        class LengthScorer(ItemScorer):
            """Scores items by text length."""

            def score(self, item: Item) -> float:
                text = item.rendered_elements.get("text", "")
                return float(len(text))

        scorer = LengthScorer()
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "hello world"},
        )

        score = scorer.score(item)

        assert score == 11.0

    def test_batch_scoring_with_custom_implementation(self) -> None:
        """Test batch scoring with custom optimization."""

        class OptimizedScorer(ItemScorer):
            """Scorer with optimized batch processing."""

            def score(self, item: Item) -> float:
                return 1.0

            def score_batch(self, items: list[Item]) -> list[float]:
                # Override with batch optimization
                return [2.0 * len(items) for _ in items]

        scorer = OptimizedScorer()
        items = [
            Item(item_template_id=uuid4(), rendered_elements={"text": f"test{i}"})
            for i in range(3)
        ]

        scores = scorer.score_batch(items)

        # Batch implementation returns 2.0 * 3 = 6.0 for each
        assert scores == [6.0, 6.0, 6.0]
