"""Tests for OrdinalScaleModel with mixed effects support."""

from __future__ import annotations

import tempfile

import pytest

from bead.active_learning.config import MixedEffectsConfig
from bead.active_learning.models.ordinal_scale import OrdinalScaleModel
from bead.config.active_learning import OrdinalScaleModelConfig
from bead.data.range import Range
from bead.items.item import Item

# mark all tests in this module as slow model training tests
pytestmark = pytest.mark.slow_model_training


class TestFixedEffectsMode:
    """Test OrdinalScaleModel with fixed effects mode."""

    def test_train_with_fixed_mode(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test training with fixed effects mode."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = OrdinalScaleModel(config)

        # Fixed effects: participant_ids are not used.
        metrics = model.train(sample_items, sample_labels)

        assert "train_mse" in metrics
        assert "train_loss" in metrics
        # Fixed mode should not have participant variance
        assert "participant_variance" not in metrics

    def test_predict_with_fixed_mode(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test prediction with fixed effects mode."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = OrdinalScaleModel(config)

        model.train(sample_items, sample_labels)
        predictions = model.predict(sample_items[:5])

        assert len(predictions) == 5
        for pred in predictions:
            val = float(pred.predicted_class)
            assert 0.0 <= val <= 1.0

    def test_train_validates_participant_ids_length(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that train validates participant_ids length."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = OrdinalScaleModel(config)

        # Wrong length
        participant_ids = ["default"] * (len(sample_items) - 1)

        with pytest.raises(ValueError, match="Length mismatch"):
            model.train(sample_items, sample_labels, participant_ids)

    def test_train_validates_empty_participant_ids(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that train rejects empty participant_ids."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = OrdinalScaleModel(config)

        # Empty string in participant_ids
        participant_ids = ["default"] * len(sample_items)
        participant_ids[5] = ""

        with pytest.raises(ValueError, match="cannot contain empty strings"):
            model.train(sample_items, sample_labels, participant_ids)

    def test_train_validates_label_bounds(self, sample_items: list[Item]) -> None:
        """Test that train validates label bounds."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased", num_epochs=1, device="cpu"
        )
        model = OrdinalScaleModel(config)

        # Label outside bounds
        labels = ["0.5"] * 19 + ["1.5"]  # 1.5 > scale_max=1.0

        with pytest.raises(ValueError, match="outside bounds"):
            model.train(sample_items, labels)


class TestRandomInterceptsMode:
    """Test OrdinalScaleModel with random intercepts mode."""

    def test_train_with_random_intercepts(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test training with random intercepts mode."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                prior_mean=0.0,
                prior_variance=1.0,
            ),
        )
        model = OrdinalScaleModel(config)

        # Two participants
        participant_ids = ["p1"] * 10 + ["p2"] * 10
        metrics = model.train(sample_items, sample_labels, participant_ids)

        assert "train_mse" in metrics
        assert "participant_variance" in metrics
        assert "n_participants" in metrics
        assert metrics["n_participants"] == 2

    def test_random_intercepts_creates_intercepts(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that random intercepts creates participant intercepts."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        # Check intercepts created
        assert model.random_effects is not None
        assert "mu" in model.random_effects.intercepts
        assert "p1" in model.random_effects.intercepts["mu"]
        assert "p2" in model.random_effects.intercepts["mu"]

    def test_predict_with_known_participant(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test prediction with known participant uses intercepts."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        # Predict with known participant
        predictions = model.predict(sample_items[:2], participant_ids[:2])
        assert len(predictions) == 2
        for pred in predictions:
            val = float(pred.predicted_class)
            assert 0.0 <= val <= 1.0

    def test_predict_with_unknown_participant_uses_prior(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test prediction with unknown participant uses prior mean."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 20
        model.train(sample_items, sample_labels, participant_ids)

        # Predict with unknown participant
        unknown_pids = ["p_unknown"] * 2
        predictions = model.predict(sample_items[:2], unknown_pids)
        assert len(predictions) == 2
        for pred in predictions:
            val = float(pred.predicted_class)
            assert 0.0 <= val <= 1.0

    def test_variance_components_tracked(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that variance components are tracked."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                estimate_variance_components=True,
            ),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        metrics = model.train(sample_items, sample_labels, participant_ids)

        assert "participant_variance" in metrics
        assert metrics["participant_variance"] >= 0.0
        assert len(model.variance_history) > 0

    def test_intercepts_have_correct_shape(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that intercepts have correct shape (scalar for μ)."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 20
        model.train(sample_items, sample_labels, participant_ids)

        # Intercepts should be scalar (shape [1])
        assert model.random_effects is not None
        p1_intercept = model.random_effects.intercepts["mu"]["p1"]
        assert p1_intercept.shape == (1,)


class TestRandomSlopesMode:
    """Test OrdinalScaleModel with random slopes mode."""

    def test_train_with_random_slopes(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test training with random slopes mode."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        metrics = model.train(sample_items, sample_labels, participant_ids)

        assert "train_mse" in metrics
        # Random slopes mode tracks slope variance
        assert "participant_variance" in metrics

    def test_random_slopes_creates_participant_heads(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that random slopes creates participant-specific heads."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        # Check slopes created
        assert model.random_effects is not None
        assert "p1" in model.random_effects.slopes
        assert "p2" in model.random_effects.slopes

    def test_predict_with_unknown_participant_uses_fixed_head(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test prediction with unknown participant uses fixed head."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 20
        model.train(sample_items, sample_labels, participant_ids)

        # Predict with unknown participant - should use fixed head
        unknown_pids = ["p_unknown"] * 2
        predictions = model.predict(sample_items[:2], unknown_pids)
        assert len(predictions) == 2


class TestPredictProba:
    """Test predict_proba method."""

    def test_predict_proba_with_fixed_mode(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test predict_proba returns continuous values."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = OrdinalScaleModel(config)

        model.train(sample_items, sample_labels)

        # predict_proba should return μ values
        proba = model.predict_proba(sample_items[:5])
        assert proba.shape == (5, 1)
        assert all(0.0 <= val[0] <= 1.0 for val in proba)

    def test_predict_proba_validates_participant_ids_length(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that predict_proba validates participant_ids length."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 20
        model.train(sample_items, sample_labels, participant_ids)

        # Wrong length
        wrong_pids = ["p1"] * 3
        with pytest.raises(ValueError, match="Length mismatch"):
            model.predict_proba(sample_items[:5], wrong_pids)


class TestSaveLoad:
    """Test save/load functionality."""

    def test_save_load_with_random_intercepts(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test save/load preserves random intercepts."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 10 + ["p2"] * 10
        model.train(sample_items, sample_labels, participant_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            # Load model
            model2 = OrdinalScaleModel()
            model2.load(tmpdir)

            # Check intercepts preserved
            assert model2.random_effects is not None
            assert "mu" in model2.random_effects.intercepts
            assert "p1" in model2.random_effects.intercepts["mu"]
            assert "p2" in model2.random_effects.intercepts["mu"]

    def test_save_load_with_random_slopes(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test save/load preserves random slopes."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = OrdinalScaleModel(config)

        participant_ids = ["p1"] * 20
        model.train(sample_items, sample_labels, participant_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            # Load model
            model2 = OrdinalScaleModel()
            model2.load(tmpdir)

            # Check slopes preserved
            assert model2.random_effects is not None
            assert "p1" in model2.random_effects.slopes


class TestEndpointHandling:
    """Test handling of endpoint values (0.0 and 1.0)."""

    def test_train_with_endpoint_values(
        self, sample_items: list[Item], sample_endpoint_labels: list[str]
    ) -> None:
        """Test training with 0.0 and 1.0 values (no nudging)."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = OrdinalScaleModel(config)

        metrics = model.train(sample_items, sample_endpoint_labels)

        # Should train successfully with endpoints
        assert "train_mse" in metrics
        assert "train_loss" in metrics

    def test_predict_clamps_to_bounds(
        self, sample_items: list[Item], sample_labels: list[str]
    ) -> None:
        """Test that predictions are clamped to [scale_min, scale_max]."""
        config = OrdinalScaleModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=4,
            device="cpu",
            scale=Range[float](min=0.0, max=1.0),
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = OrdinalScaleModel(config)

        model.train(sample_items, sample_labels)
        predictions = model.predict(sample_items)

        # All predictions should be in bounds
        for pred in predictions:
            val = float(pred.predicted_class)
            assert 0.0 <= val <= 1.0
