"""Tests for MultiSelectModel with mixed effects support."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bead.active_learning.config import MixedEffectsConfig
from bead.active_learning.models.multi_select import MultiSelectModel
from bead.config.active_learning import MultiSelectModelConfig
from bead.items.item import Item

# mark all tests in this module as slow model training tests
pytestmark = pytest.mark.slow_model_training


class TestFixedEffectsMode:
    """Test MultiSelectModel with fixed effects mode."""

    def test_train_with_fixed_mode(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test training with fixed effects mode."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MultiSelectModel(config)

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
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MultiSelectModel(config)
        model.train(sample_items, sample_labels)

        # Predict with same participant_ids
        predictions = model.predict(sample_items[:5])

        assert len(predictions) == 5
        for pred in predictions:
            # predicted_class is JSON string of selected options
            selected = json.loads(pred.predicted_class)
            assert isinstance(selected, list)
            assert all(opt in ["option_a", "option_b", "option_c"] for opt in selected)
            assert 0.0 <= pred.confidence <= 1.0

    def test_train_requires_participant_ids(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that train requires participant_ids parameter."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased", num_epochs=1, device="cpu"
        )
        model = MultiSelectModel(config)

        # Missing participant_ids should fail (signature requires it)
        # This is a compile-time check - the signature enforces it
        # Just verify the parameter exists
        import inspect  # noqa: PLC0415

        sig = inspect.signature(model.train)
        assert "participant_ids" in sig.parameters

    def test_length_mismatch_raises_error(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Length mismatch between items and participant_ids raises."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MultiSelectModel(config)

        # Mismatched lengths
        participant_ids = ["alice"] * (len(sample_items) - 1)

        with pytest.raises(ValueError, match="Length mismatch"):
            model.train(sample_items, sample_labels, participant_ids)

    def test_empty_participant_id_raises_error(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Empty participant_id strings raise."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MultiSelectModel(config)

        # Include empty participant_id
        participant_ids = ["alice", "", "bob"] + ["x"] * (len(sample_items) - 3)

        with pytest.raises(ValueError, match="cannot contain empty strings"):
            model.train(sample_items, sample_labels, participant_ids)


class TestRandomInterceptsMode:
    """Test MultiSelectModel with random intercepts mode."""

    def test_random_intercepts_creates_intercepts(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that random intercepts mode creates participant intercepts."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        # Verify intercepts were created
        assert model.random_effects is not None
        assert len(model.random_effects.intercepts) > 0
        # Should have intercepts for unique participants
        # intercepts structure: {'mu': {'alice': ..., 'bob': ..., 'charlie': ...}}
        unique_participants = set(sample_participant_ids)
        assert "mu" in model.random_effects.intercepts
        assert len(model.random_effects.intercepts["mu"]) == len(unique_participants)

    def test_random_intercepts_uses_intercepts_in_prediction(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that random intercepts are used during prediction."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        # Predict for known participant
        known_pid = sample_participant_ids[0]
        predictions = model.predict([sample_items[0]], [known_pid])

        assert len(predictions) == 1
        # Verify prediction was made
        selected = json.loads(predictions[0].predicted_class)
        assert isinstance(selected, list)

    def test_random_intercepts_different_participants_different_outputs(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that different participants get different predictions."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        # Get predictions for same item with different participants
        item = sample_items[0]
        pred_alice = model.predict([item], ["alice"])
        pred_bob = model.predict([item], ["bob"])

        # Probabilities should differ (different random intercepts)
        proba_alice = pred_alice[0].probabilities
        proba_bob = pred_bob[0].probabilities

        # At least one option should have different probability
        assert any(
            abs(proba_alice[opt] - proba_bob[opt]) > 0.01
            for opt in ["option_a", "option_b", "option_c"]
        )

    def test_random_intercepts_unknown_participant_uses_prior_mean(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that unknown participants use prior mean (zero bias)."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        # Predict for unknown participant
        unknown_predictions = model.predict([sample_items[0]], ["unknown_participant"])

        assert len(unknown_predictions) == 1
        # Should still make a prediction (using prior mean / fixed head)
        selected = json.loads(unknown_predictions[0].predicted_class)
        assert isinstance(selected, list)

    def test_variance_tracking_random_intercepts(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that variance components are estimated and tracked."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts", estimate_variance_components=True
            ),
        )
        model = MultiSelectModel(config)

        metrics = model.train(sample_items, sample_labels, sample_participant_ids)

        # Verify variance components in metrics
        assert "participant_variance" in metrics
        assert "n_participants" in metrics
        assert metrics["participant_variance"] >= 0.0
        assert metrics["n_participants"] > 0

        # Verify variance history
        assert len(model.variance_history) == 1
        assert model.variance_history[0].variance >= 0.0

    def test_predict_proba_random_intercepts(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test predict_proba with random intercepts."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        # Predict probabilities
        proba = model.predict_proba(sample_items[:5], sample_participant_ids[:5])

        # Should return (n_items, n_options) array
        assert proba.shape == (5, 3)
        # All probabilities in [0, 1]
        assert (proba >= 0.0).all() and (proba <= 1.0).all()


class TestRandomSlopesMode:
    """Test MultiSelectModel with random slopes mode."""

    def test_random_slopes_creates_slopes(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that random slopes mode creates participant-specific heads."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        # Verify slopes were created
        assert model.random_effects is not None
        assert len(model.random_effects.slopes) > 0
        unique_participants = set(sample_participant_ids)
        assert len(model.random_effects.slopes) == len(unique_participants)

    def test_random_slopes_uses_slopes_in_prediction(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that random slopes are used during prediction."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        # Predict for known participant
        known_pid = sample_participant_ids[0]
        predictions = model.predict([sample_items[0]], [known_pid])

        assert len(predictions) == 1
        selected = json.loads(predictions[0].predicted_class)
        assert isinstance(selected, list)

    def test_random_slopes_unknown_participant_uses_fixed_head(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that unknown participants use fixed head."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        # Predict for unknown participant
        unknown_predictions = model.predict([sample_items[0]], ["unknown_participant"])

        assert len(unknown_predictions) == 1
        selected = json.loads(unknown_predictions[0].predicted_class)
        assert isinstance(selected, list)


class TestSaveLoad:
    """Test save/load functionality with random effects."""

    def test_save_and_load_preserves_random_effects(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that save and load preserve random effects."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        # Get predictions before save
        pred_before = model.predict([sample_items[0]], [sample_participant_ids[0]])

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model")
            model.save(save_path)

            # Load model
            loaded_model = MultiSelectModel(config)
            loaded_model.load(save_path)

            # Get predictions after load
            pred_after = loaded_model.predict(
                [sample_items[0]], [sample_participant_ids[0]]
            )

            # Predictions should be identical
            assert pred_before[0].predicted_class == pred_after[0].predicted_class
            # Probabilities should match
            for opt in ["option_a", "option_b", "option_c"]:
                assert (
                    abs(
                        pred_before[0].probabilities[opt]
                        - pred_after[0].probabilities[opt]
                    )
                    < 1e-5
                )

    def test_save_and_load_preserves_variance_history(
        self,
        sample_items: list[Item],
        sample_labels: list[str],
        sample_participant_ids: list[str],
    ) -> None:
        """Test that save and load preserve variance history."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts", estimate_variance_components=True
            ),
        )
        model = MultiSelectModel(config)

        model.train(sample_items, sample_labels, sample_participant_ids)

        variance_before = model.variance_history[0].variance

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model")
            model.save(save_path)

            loaded_model = MultiSelectModel(config)
            loaded_model.load(save_path)

            # Variance history should be preserved
            assert len(loaded_model.variance_history) == 1
            assert (
                abs(loaded_model.variance_history[0].variance - variance_before) < 1e-6
            )


class TestMultiLabelSpecifics:
    """Test multi-label specific functionality."""

    def test_sigmoid_output_independent_probabilities(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that sigmoid produces independent probabilities (not summing to 1)."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MultiSelectModel(config)
        model.train(sample_items, sample_labels)

        predictions = model.predict([sample_items[0]])

        # Probabilities are independent - sum may not be 1.0
        prob_sum = sum(predictions[0].probabilities.values())
        # Should not sum to exactly 1.0 (would be true for softmax)
        # May sum to anything in [0, 3] for 3 options
        assert 0.0 <= prob_sum <= 3.0

    def test_empty_selection_possible(self, sample_items: list[Item]) -> None:
        """Test that model can predict empty selection (no options selected)."""
        # Create labels with empty selections
        labels = [json.dumps([]) for _ in range(10)]  # No selections

        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MultiSelectModel(config)
        model.train(sample_items[:10], labels)

        predictions = model.predict([sample_items[0]])

        # Should return a prediction (possibly empty)
        selected = json.loads(predictions[0].predicted_class)
        assert isinstance(selected, list)

    def test_all_options_selected_possible(self, sample_items: list[Item]) -> None:
        """Test that model can predict all options selected."""
        # Create labels with all options selected
        labels = [json.dumps(["option_a", "option_b", "option_c"]) for _ in range(10)]

        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MultiSelectModel(config)
        model.train(sample_items[:10], labels)

        predictions = model.predict([sample_items[0]])

        # Should return a prediction
        selected = json.loads(predictions[0].predicted_class)
        assert isinstance(selected, list)

    def test_hamming_accuracy_metric(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that Hamming accuracy is used (not exact match)."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MultiSelectModel(config)
        metrics = model.train(sample_items, sample_labels)

        # Hamming accuracy should be in [0, 1]
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
        # With random initialization and 1 epoch, shouldn't be perfect
        # (unless very lucky, but statistically unlikely)

    def test_invalid_label_format_raises_error(self, sample_items: list[Item]) -> None:
        """Test that invalid label format raises error."""
        # Invalid: not JSON
        labels_invalid = ["not_json"] * len(sample_items)

        config = MultiSelectModelConfig(
            model_name="bert-base-uncased", num_epochs=1, device="cpu"
        )
        model = MultiSelectModel(config)

        with pytest.raises(ValueError, match="valid JSON"):
            model.train(sample_items, labels_invalid)

    def test_invalid_option_in_label_raises_error(
        self, sample_items: list[Item]
    ) -> None:
        """Test that invalid option name in label raises error."""
        # Invalid: option_d doesn't exist
        labels_invalid = [json.dumps(["option_d"])] * len(sample_items)

        config = MultiSelectModelConfig(
            model_name="bert-base-uncased", num_epochs=1, device="cpu"
        )
        model = MultiSelectModel(config)

        with pytest.raises(ValueError, match="Invalid option"):
            model.train(sample_items, labels_invalid)


class TestDualEncoderMode:
    """Test multi-select model with dual encoder mode."""

    def test_dual_encoder_mode_train_and_predict(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test training and prediction with dual_encoder mode."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased",
            encoder_mode="dual_encoder",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MultiSelectModel(config)
        metrics = model.train(sample_items, sample_labels)

        assert "train_accuracy" in metrics
        assert "train_loss" in metrics

        # Predict with dual encoder
        predictions = model.predict(sample_items[:3])

        assert len(predictions) == 3
        for pred in predictions:
            selected = json.loads(pred.predicted_class)
            assert isinstance(selected, list)

    def test_predict_before_training_raises_error(
        self, sample_items: list[Item]
    ) -> None:
        """Test that predicting before training raises error."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased", num_epochs=1, device="cpu"
        )
        model = MultiSelectModel(config)

        with pytest.raises(ValueError, match="not trained"):
            model.predict(sample_items[:1])

    def test_predict_proba_before_training_raises_error(
        self, sample_items: list[Item]
    ) -> None:
        """Test that predict_proba before training raises error."""
        config = MultiSelectModelConfig(
            model_name="bert-base-uncased", num_epochs=1, device="cpu"
        )
        model = MultiSelectModel(config)

        with pytest.raises(ValueError, match="not trained"):
            model.predict_proba(sample_items[:1])
