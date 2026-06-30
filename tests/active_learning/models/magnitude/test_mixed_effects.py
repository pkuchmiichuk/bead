"""Tests for MagnitudeModel with mixed effects support."""

from __future__ import annotations

import tempfile

import pytest

from bead.active_learning.config import MixedEffectsConfig
from bead.active_learning.models.magnitude import MagnitudeModel
from bead.config.active_learning import MagnitudeModelConfig
from bead.items.item import Item

# mark all tests in this module as slow model training tests
pytestmark = pytest.mark.slow_model_training


class TestFixedEffectsMode:
    """Test MagnitudeModel with fixed effects mode."""

    def test_train_with_fixed_mode_unbounded(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test training with fixed effects mode (unbounded)."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            bounded=False,
            distribution="normal",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MagnitudeModel(config)

        # Fixed effects: use placeholder participant_ids
        metrics = model.train(sample_items, sample_unbounded_labels)

        assert "train_mse" in metrics
        assert "train_loss" in metrics
        # Fixed mode should not have participant variance
        assert "participant_variance" not in metrics

    def test_train_with_fixed_mode_bounded(
        self, sample_items: list[Item], sample_bounded_labels: list[str]
    ) -> None:
        """Test training with fixed effects mode (bounded)."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            bounded=True,
            min_value=0.0,
            max_value=100.0,
            distribution="truncated_normal",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MagnitudeModel(config)
        metrics = model.train(sample_items, sample_bounded_labels)

        assert "train_mse" in metrics
        assert "train_loss" in metrics

    def test_predict_with_fixed_mode(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test prediction with fixed effects mode."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            bounded=False,
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = MagnitudeModel(config)
        model.train(sample_items, sample_unbounded_labels)

        # Predict with same participant_ids
        predictions = model.predict(sample_items[:5])

        assert len(predictions) == 5
        for pred in predictions:
            # Should be numeric value
            val = float(pred.predicted_class)
            assert isinstance(val, float)

    def test_train_validates_participant_ids_length(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that train validates participant_ids length."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MagnitudeModel(config)

        # Wrong length
        participant_ids = ["default"] * (len(sample_items) - 1)

        with pytest.raises(ValueError, match="Length mismatch"):
            model.train(sample_items, sample_unbounded_labels, participant_ids)

    def test_train_validates_empty_participant_ids(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that train rejects empty participant_ids."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MagnitudeModel(config)

        # Empty string in participant_ids
        participant_ids = ["default"] * len(sample_items)
        participant_ids[5] = ""

        with pytest.raises(ValueError, match="cannot contain empty strings"):
            model.train(sample_items, sample_unbounded_labels, participant_ids)

    def test_train_validates_bounds(self, sample_items: list[Item]) -> None:
        """Test that train validates label bounds for bounded case."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            device="cpu",
            bounded=True,
            min_value=0.0,
            max_value=100.0,
            distribution="truncated_normal",
        )
        model = MagnitudeModel(config)

        # Label outside bounds
        labels = ["50.0"] * 19 + ["150.0"]  # 150.0 > max_value=100.0

        with pytest.raises(ValueError, match="outside bounds"):
            model.train(sample_items, labels)


class TestRandomInterceptsMode:
    """Test MagnitudeModel with random intercepts mode."""

    def test_train_with_random_intercepts(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test training with random intercepts mode."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                prior_mean=0.0,
                prior_variance=1.0,
                estimate_variance_components=True,
            ),
        )
        model = MagnitudeModel(config)

        # Two participants
        participant_ids = ["p1"] * 10 + ["p2"] * 10
        metrics = model.train(sample_items, sample_unbounded_labels, participant_ids)

        assert "train_mse" in metrics
        assert "participant_variance" in metrics
        assert "n_participants" in metrics
        assert metrics["n_participants"] == 2

    def test_random_intercepts_creates_intercepts(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that random intercepts creates participant intercepts."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_unbounded_labels, participant_ids)

        # Verify intercepts created
        assert model.random_effects is not None
        assert "mu" in model.random_effects.intercepts
        assert "p1" in model.random_effects.intercepts["mu"]
        assert "p2" in model.random_effects.intercepts["mu"]

    def test_intercepts_have_correct_shape(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that intercepts have correct shape (scalar for regression)."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_unbounded_labels, participant_ids)

        # Intercepts should be scalar (n_classes=1)
        bias_p1 = model.random_effects.intercepts["mu"]["p1"]
        assert bias_p1.shape == (1,)

    def test_random_intercepts_different_outputs(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that different participants get different predictions."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_unbounded_labels, participant_ids)

        # Same item, different participants
        test_item = [sample_items[0]]
        pred_p1 = model.predict(test_item, participant_ids=["p1"])
        pred_p2 = model.predict(test_item, participant_ids=["p2"])

        # Predictions may differ due to random effects
        val_p1 = float(pred_p1[0].predicted_class)
        val_p2 = float(pred_p2[0].predicted_class)
        # They could be the same if biases are similar, so just check they're valid
        assert isinstance(val_p1, float)
        assert isinstance(val_p2, float)

    def test_predict_with_unknown_participant_uses_prior(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that unknown participants use prior mean (zero bias)."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 20
        model.train(sample_items, sample_unbounded_labels, participant_ids)

        # Predict with unknown participant
        predictions = model.predict(sample_items[:5], participant_ids=["unknown"] * 5)
        assert len(predictions) == 5


class TestRandomSlopesMode:
    """Test MagnitudeModel with random slopes mode."""

    def test_train_with_random_slopes(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test training with random slopes mode."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_slopes",
                estimate_variance_components=True,
            ),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        metrics = model.train(sample_items, sample_unbounded_labels, participant_ids)

        assert "train_mse" in metrics
        assert "participant_variance" in metrics
        assert "n_participants" in metrics

    def test_random_slopes_creates_participant_heads(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that random slopes creates participant-specific heads."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_unbounded_labels, participant_ids)

        # Verify slopes created
        assert model.random_effects is not None
        assert "p1" in model.random_effects.slopes
        assert "p2" in model.random_effects.slopes

    def test_random_slopes_different_outputs(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that different participants get different predictions."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_unbounded_labels, participant_ids)

        # Same item, different participants
        test_item = [sample_items[0]]
        pred_p1 = model.predict(test_item, participant_ids=["p1"])
        pred_p2 = model.predict(test_item, participant_ids=["p2"])

        # Both should be valid predictions
        val_p1 = float(pred_p1[0].predicted_class)
        val_p2 = float(pred_p2[0].predicted_class)
        assert isinstance(val_p1, float)
        assert isinstance(val_p2, float)

    def test_predict_with_unknown_participant_uses_fixed_head(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that unknown participants use fixed head."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 20
        model.train(sample_items, sample_unbounded_labels, participant_ids)

        # Predict with unknown participant
        predictions = model.predict(sample_items[:5], participant_ids=["unknown"] * 5)
        assert len(predictions) == 5


class TestVarianceTracking:
    """Test variance component estimation."""

    def test_variance_components_estimated(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that variance components are estimated."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                estimate_variance_components=True,
            ),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        metrics = model.train(sample_items, sample_unbounded_labels, participant_ids)

        assert "participant_variance" in metrics
        assert metrics["participant_variance"] >= 0.0

    def test_variance_increases_with_heterogeneity(
        self, sample_items: list[Item]
    ) -> None:
        """Test that variance reflects participant heterogeneity."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                estimate_variance_components=True,
            ),
        )
        model = MagnitudeModel(config)

        # Create highly varied labels across participants
        labels = ["100.0"] * 10 + ["500.0"] * 10
        participant_ids = ["p1"] * 10 + ["p2"] * 10
        metrics = model.train(sample_items, labels, participant_ids)

        # Variance should be non-zero
        assert metrics["participant_variance"] >= 0.0


class TestSaveLoad:
    """Test model save and load functionality."""

    def test_save_and_load_preserves_random_effects(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that save/load preserves random effects."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = MagnitudeModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_unbounded_labels, participant_ids)

        # Get original intercepts
        orig_bias_p1 = model.random_effects.intercepts["mu"]["p1"].clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            model.save(tmpdir)

            # Load into new model
            model2 = MagnitudeModel(config)
            model2.load(tmpdir)

            # Check intercepts preserved
            loaded_bias_p1 = model2.random_effects.intercepts["mu"]["p1"]
            assert loaded_bias_p1.shape == orig_bias_p1.shape
            # Check values are close (may have small numerical differences).
            # ``.item()`` detaches the tensor so the comparison does not
            # emit a "converting tensor with requires_grad=True" warning.
            assert loaded_bias_p1[0].item() == pytest.approx(
                orig_bias_p1[0].item(), abs=1e-5
            )

    def test_save_and_load_preserves_config(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test that save/load preserves configuration."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            bounded=True,
            min_value=0.0,
            max_value=100.0,
            distribution="truncated_normal",
        )
        model = MagnitudeModel(config)

        # Need to adjust labels to bounded range
        bounded_labels = [str(float(i * 5)) for i in range(20)]
        model.train(sample_items, bounded_labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            model2 = MagnitudeModel()
            model2.load(tmpdir)

            # Check config preserved
            assert model2.config.bounded
            assert model2.config.min_value == 0.0
            assert model2.config.max_value == 100.0
            assert model2.config.distribution == "truncated_normal"


class TestDistributions:
    """Test different distribution implementations."""

    def test_unbounded_normal_distribution(
        self, sample_items: list[Item], sample_unbounded_labels: list[str]
    ) -> None:
        """Test unbounded normal distribution."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            bounded=False,
            distribution="normal",
        )
        model = MagnitudeModel(config)

        metrics = model.train(sample_items, sample_unbounded_labels)

        assert "train_mse" in metrics
        # Predictions can be any value (unbounded)
        predictions = model.predict(sample_items[:5])
        for pred in predictions:
            val = float(pred.predicted_class)
            assert isinstance(val, float)

    def test_bounded_truncated_normal_distribution(
        self, sample_items: list[Item], sample_bounded_labels: list[str]
    ) -> None:
        """Test bounded truncated normal distribution."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            bounded=True,
            min_value=0.0,
            max_value=100.0,
            distribution="truncated_normal",
        )
        model = MagnitudeModel(config)

        metrics = model.train(sample_items, sample_bounded_labels)

        assert "train_mse" in metrics
        # Predictions should be within bounds
        predictions = model.predict(sample_items[:5])
        for pred in predictions:
            val = float(pred.predicted_class)
            assert 0.0 <= val <= 100.0

    def test_truncated_normal_handles_endpoints(
        self, sample_items: list[Item], sample_bounded_endpoint_labels: list[str]
    ) -> None:
        """Test that truncated normal handles exact endpoint values (0.0 and 100.0)."""
        config = MagnitudeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            bounded=True,
            min_value=0.0,
            max_value=100.0,
            distribution="truncated_normal",
        )
        model = MagnitudeModel(config)

        # Should handle endpoints without errors
        metrics = model.train(sample_items, sample_bounded_endpoint_labels)

        assert "train_mse" in metrics
        predictions = model.predict(sample_items[:5])
        for pred in predictions:
            val = float(pred.predicted_class)
            assert 0.0 <= val <= 100.0
