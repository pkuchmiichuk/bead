"""Tests for ForcedChoiceModel with mixed effects support."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from bead.active_learning.config import MixedEffectsConfig
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.config.active_learning import ForcedChoiceModelConfig
from bead.items.item import Item

# mark all tests in this module as slow model training tests
pytestmark = pytest.mark.slow_model_training


@pytest.fixture
def sample_items() -> list[Item]:
    """Create sample forced choice items."""
    items = []
    for i in range(20):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "option_a": f"Sentence A variant {i}",
                "option_b": f"Sentence B variant {i}",
            },
        )
        items.append(item)
    return items


@pytest.fixture
def sample_labels() -> list[str]:
    """Create sample labels (alternating)."""
    return ["option_a" if i % 2 == 0 else "option_b" for i in range(20)]


class TestFixedEffectsMode:
    """Test ForcedChoiceModel with fixed effects mode."""

    def test_train_with_fixed_mode(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test training with fixed effects mode."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = ForcedChoiceModel(config)

        # Fixed effects: use placeholder participant_ids
        metrics = model.train(sample_items, sample_labels)

        assert "train_accuracy" in metrics
        assert "train_loss" in metrics
        # Fixed mode should not have participant variance
        assert "participant_variance" not in metrics

    def test_predict_with_fixed_mode(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test prediction with fixed effects mode."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = ForcedChoiceModel(config)
        model.train(sample_items, sample_labels)

        # Predict with same participant_ids
        predictions = model.predict(sample_items[:5])

        assert len(predictions) == 5
        for pred in predictions:
            assert pred.predicted_class in ["option_a", "option_b"]
            assert 0.0 <= pred.confidence <= 1.0

    def test_train_accepts_default_fixed_mode(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Default mixed_effects.mode is 'fixed'; train runs without participant_ids."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased", num_epochs=1, device="cpu"
        )
        model = ForcedChoiceModel(config)
        model.train(sample_items, sample_labels)

    def test_train_validates_participant_ids_length(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that train validates participant_ids length."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased", num_epochs=1, device="cpu"
        )
        model = ForcedChoiceModel(config)

        # Wrong length
        participant_ids = ["default"] * (len(sample_items) - 1)

        with pytest.raises(ValueError, match="Length mismatch"):
            model.train(sample_items, sample_labels, participant_ids)

    def test_train_validates_empty_participant_ids(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that train rejects empty participant_ids."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased", num_epochs=1, device="cpu"
        )
        model = ForcedChoiceModel(config)

        # Empty string in participant_ids
        participant_ids = ["default"] * len(sample_items)
        participant_ids[5] = ""

        with pytest.raises(ValueError, match="cannot contain empty strings"):
            model.train(sample_items, sample_labels, participant_ids)


class TestRandomInterceptsMode:
    """Test ForcedChoiceModel with random intercepts mode."""

    def test_train_with_random_intercepts(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test training with random intercepts mode."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                prior_variance=0.5,
                regularization_strength=0.01,
                estimate_variance_components=True,
            ),
        )
        model = ForcedChoiceModel(config)

        # Multiple participants
        participant_ids = [f"participant_{i % 3}" for i in range(len(sample_items))]
        metrics = model.train(sample_items, sample_labels, participant_ids)

        assert "train_accuracy" in metrics
        assert "train_loss" in metrics
        # Should have variance components
        assert "participant_variance" in metrics
        assert "n_participants" in metrics
        assert metrics["n_participants"] == 3

    def test_random_intercepts_creates_intercepts(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that random intercepts are created for participants."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = ForcedChoiceModel(config)

        participant_ids = ["alice", "bob", "alice", "bob"] * 5
        model.train(sample_items, sample_labels, participant_ids)

        # Check intercepts were created (nested: intercepts[param][participant])
        assert "mu" in model.random_effects.intercepts
        assert "alice" in model.random_effects.intercepts["mu"]
        assert "bob" in model.random_effects.intercepts["mu"]
        assert len(model.random_effects.intercepts["mu"]) == 2

    def test_predict_with_known_participant(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test prediction for known participant uses their intercepts."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = ForcedChoiceModel(config)

        participant_ids = ["alice"] * 10 + ["bob"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        # Predict for alice
        predictions = model.predict(sample_items[:5], ["alice"] * 5)
        assert len(predictions) == 5

    def test_predict_with_unknown_participant_uses_prior(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test prediction for unknown participant uses prior mean."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = ForcedChoiceModel(config)

        participant_ids = ["alice", "bob"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        # Predict for unknown participant
        predictions = model.predict(sample_items[:5], ["charlie"] * 5)
        assert len(predictions) == 5
        # Should work without error (uses prior mean)

    def test_variance_components_tracked(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that variance components are tracked during training."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts", estimate_variance_components=True
            ),
        )
        model = ForcedChoiceModel(config)

        participant_ids = [f"p{i % 5}" for i in range(len(sample_items))]
        model.train(sample_items, sample_labels, participant_ids)

        # Variance history should be populated
        assert len(model.variance_history) > 0
        var_comp = model.variance_history[0]
        assert var_comp.grouping_factor == "participant"
        assert var_comp.effect_type == "intercept"
        assert var_comp.n_groups == 5


class TestRandomSlopesMode:
    """Test ForcedChoiceModel with random slopes mode."""

    def test_train_with_random_slopes(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test training with random slopes mode."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_slopes",
                prior_variance=0.1,
                estimate_variance_components=True,
            ),
        )
        model = ForcedChoiceModel(config)

        participant_ids = [f"participant_{i % 3}" for i in range(len(sample_items))]
        metrics = model.train(sample_items, sample_labels, participant_ids)

        assert "train_accuracy" in metrics
        assert "participant_variance" in metrics
        assert metrics["n_participants"] == 3

    def test_random_slopes_creates_participant_heads(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that random slopes creates participant-specific heads."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = ForcedChoiceModel(config)

        participant_ids = ["alice", "bob"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        # Check slopes were created
        assert "alice" in model.random_effects.slopes
        assert "bob" in model.random_effects.slopes
        assert len(model.random_effects.slopes) == 2

    def test_predict_with_unknown_participant_uses_fixed_head(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test prediction for unknown participant uses fixed head."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = ForcedChoiceModel(config)

        participant_ids = ["alice", "bob"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        # Predict for unknown participant (should use fixed head)
        predictions = model.predict(sample_items[:5], ["charlie"] * 5)
        assert len(predictions) == 5


class TestPredictProba:
    """Test predict_proba with mixed effects."""

    def test_predict_proba_with_fixed_mode(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test predict_proba with fixed effects mode."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = ForcedChoiceModel(config)
        model.train(sample_items, sample_labels)

        proba = model.predict_proba(sample_items[:5])

        assert proba.shape == (5, 2)
        # Each row should sum to 1
        import numpy as np  # noqa: PLC0415

        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_with_random_intercepts(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test predict_proba with random intercepts."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = ForcedChoiceModel(config)

        participant_ids = ["alice", "bob"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        proba = model.predict_proba(sample_items[:5], ["alice"] * 5)

        assert proba.shape == (5, 2)
        import numpy as np  # noqa: PLC0415

        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_validates_participant_ids_length(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that predict_proba validates participant_ids length."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = ForcedChoiceModel(config)

        participant_ids = ["alice", "bob"] * (len(sample_items) // 2)
        model.train(sample_items, sample_labels, participant_ids)

        # Wrong length for predict_proba
        with pytest.raises(ValueError, match="Length mismatch"):
            model.predict_proba(sample_items[:5], ["default"] * 3)


class TestSaveLoad:
    """Test saving and loading models with mixed effects."""

    def test_save_load_with_random_intercepts(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test saving and loading model with random intercepts."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts", estimate_variance_components=True
            ),
        )
        model = ForcedChoiceModel(config)

        participant_ids = (["alice", "bob", "charlie"] * 6) + ["alice", "bob"]
        model.train(sample_items, sample_labels, participant_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model.save(str(model_path))

            # Load into new model
            loaded_model = ForcedChoiceModel()
            loaded_model.load(str(model_path))

            # Check intercepts preserved (nested: intercepts[param][participant])
            assert "mu" in loaded_model.random_effects.intercepts
            assert "alice" in loaded_model.random_effects.intercepts["mu"]
            assert "bob" in loaded_model.random_effects.intercepts["mu"]
            assert "charlie" in loaded_model.random_effects.intercepts["mu"]

            # Check variance history preserved
            assert len(loaded_model.variance_history) == len(model.variance_history)

            # Predictions should work
            predictions = loaded_model.predict(sample_items[:5], ["alice"] * 5)
            assert len(predictions) == 5

    def test_save_load_with_random_slopes(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test saving and loading model with random slopes."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = ForcedChoiceModel(config)

        participant_ids = ["alice", "bob"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model.save(str(model_path))

            # Load into new model
            loaded_model = ForcedChoiceModel()
            loaded_model.load(str(model_path))

            # Check slopes preserved
            assert "alice" in loaded_model.random_effects.slopes
            assert "bob" in loaded_model.random_effects.slopes

            # Predictions should work
            predictions = loaded_model.predict(sample_items[:5], ["alice"] * 5)
            assert len(predictions) == 5


class TestValidation:
    """Test validation with mixed effects."""

    def test_validation_with_placeholder_participant_ids(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that validation uses placeholder participant_ids."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = ForcedChoiceModel(config)

        # Train with real participant IDs
        participant_ids = ["alice", "bob"] * 10
        val_items = sample_items[:5]
        val_labels = sample_labels[:5]

        metrics = model.train(
            sample_items[5:],
            sample_labels[5:],
            participant_ids[5:],
            validation_items=val_items,
            validation_labels=val_labels,
        )

        # Should have validation accuracy
        assert "val_accuracy" in metrics
        assert 0.0 <= metrics["val_accuracy"] <= 1.0


class TestAdaptiveRegularization:
    """Test adaptive regularization for mixed effects."""

    def test_adaptive_regularization_enabled(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that adaptive regularization works."""
        config = ForcedChoiceModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                adaptive_regularization=True,
                min_samples_for_random_effects=5,
            ),
        )
        model = ForcedChoiceModel(config)

        # Alice has many samples, bob has few
        participant_ids = ["alice"] * 15 + ["bob"] * 5
        model.train(sample_items, sample_labels, participant_ids)

        # Both should be registered with correct counts
        assert model.random_effects.participant_sample_counts["alice"] == 15
        assert model.random_effects.participant_sample_counts["bob"] == 5
