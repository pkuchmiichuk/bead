"""Tests for base active learning model interfaces with mixed effects support."""

from __future__ import annotations

import numpy as np
import pytest
from didactic.api import ValidationError

from bead.active_learning.config import (
    MixedEffectsConfig,
    RandomEffectsSpec,
    VarianceComponents,
)
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction


class TestVarianceComponents:
    """Test VarianceComponents Pydantic model."""

    def test_valid_variance_components(self) -> None:
        """Test creating valid variance components."""
        vc = VarianceComponents(
            grouping_factor="participant",
            effect_type="intercept",
            variance=0.25,
            n_groups=50,
            n_observations_per_group={"p1": 10, "p2": 15, "p3": 8},
        )

        assert vc.grouping_factor == "participant"
        assert vc.effect_type == "intercept"
        assert vc.variance == 0.25
        assert vc.n_groups == 50
        assert len(vc.n_observations_per_group) == 3

    def test_variance_components_slope_type(self) -> None:
        """Test variance components with slope effect type."""
        vc = VarianceComponents(
            grouping_factor="item",
            effect_type="slope",
            variance=0.5,
            n_groups=100,
            n_observations_per_group={"item1": 20},
        )

        assert vc.effect_type == "slope"
        assert vc.grouping_factor == "item"

    def test_variance_must_be_non_negative(self) -> None:
        """Test that variance must be >= 0."""
        with pytest.raises(ValidationError, match="non-negative"):
            VarianceComponents(
                grouping_factor="participant",
                effect_type="intercept",
                variance=-0.1,  # Invalid: negative variance
                n_groups=50,
                n_observations_per_group={"p1": 10},
            )

    def test_n_groups_must_be_positive(self) -> None:
        """Test that n_groups must be >= 1."""
        with pytest.raises(ValidationError, match=">= 1"):
            VarianceComponents(
                grouping_factor="participant",
                effect_type="intercept",
                variance=0.25,
                n_groups=0,  # Invalid: must be >= 1
                n_observations_per_group={"p1": 10},
            )


class TestRandomEffectsSpec:
    """Test RandomEffectsSpec Pydantic model."""

    def test_single_grouping_factor_intercept(self) -> None:
        """Test specification with single grouping factor (intercept)."""
        spec = RandomEffectsSpec(grouping_factors={"participant": "intercept"})

        assert spec.grouping_factors == {"participant": "intercept"}
        assert spec.correlation_structure == "independent"

    def test_single_grouping_factor_slope(self) -> None:
        """Test specification with single grouping factor (slope)."""
        spec = RandomEffectsSpec(grouping_factors={"participant": "slope"})

        assert spec.grouping_factors == {"participant": "slope"}

    def test_both_intercept_and_slope(self) -> None:
        """Test specification with both intercept and slope."""
        spec = RandomEffectsSpec(
            grouping_factors={"participant": "both"},
            correlation_structure="correlated",
        )

        assert spec.grouping_factors == {"participant": "both"}
        assert spec.correlation_structure == "correlated"

    def test_multiple_grouping_factors(self) -> None:
        """Test specification with multiple grouping factors (Phase 6+)."""
        spec = RandomEffectsSpec(
            grouping_factors={"participant": "intercept", "item": "intercept"}
        )

        assert len(spec.grouping_factors) == 2
        assert spec.grouping_factors["participant"] == "intercept"
        assert spec.grouping_factors["item"] == "intercept"


class TestMixedEffectsConfig:
    """Test MixedEffectsConfig Pydantic model."""

    def test_default_config_is_fixed_mode(self) -> None:
        """Test that default mode is 'fixed'."""
        config = MixedEffectsConfig()

        assert config.mode == "fixed"
        assert config.prior_mean == 0.0
        assert config.prior_variance == 1.0
        assert config.estimate_variance_components is True
        assert config.variance_estimation_method == "mle"

    def test_random_intercepts_config(self) -> None:
        """Test configuration for random intercepts mode."""
        config = MixedEffectsConfig(
            mode="random_intercepts",
            prior_mean=0.0,
            prior_variance=0.5,
            regularization_strength=0.01,
        )

        assert config.mode == "random_intercepts"
        assert config.prior_variance == 0.5
        assert config.regularization_strength == 0.01

    def test_random_slopes_config(self) -> None:
        """Test configuration for random slopes mode."""
        config = MixedEffectsConfig(
            mode="random_slopes",
            prior_variance=0.1,
            adaptive_regularization=True,
            min_samples_for_random_effects=10,
        )

        assert config.mode == "random_slopes"
        assert config.prior_variance == 0.1
        assert config.adaptive_regularization is True
        assert config.min_samples_for_random_effects == 10

    def test_variance_estimation_method_mle(self) -> None:
        """Test MLE variance estimation method."""
        config = MixedEffectsConfig(variance_estimation_method="mle")

        assert config.variance_estimation_method == "mle"

    def test_variance_estimation_method_reml(self) -> None:
        """Test REML variance estimation method."""
        config = MixedEffectsConfig(variance_estimation_method="reml")

        assert config.variance_estimation_method == "reml"

    def test_prior_variance_must_be_non_negative(self) -> None:
        """Test that prior_variance must be >= 0."""
        with pytest.raises(ValidationError, match="non-negative"):
            MixedEffectsConfig(prior_variance=-0.1)

    def test_regularization_strength_must_be_non_negative(self) -> None:
        """Test that regularization_strength must be >= 0."""
        with pytest.raises(ValidationError, match="non-negative"):
            MixedEffectsConfig(regularization_strength=-0.01)

    def test_min_samples_must_be_positive(self) -> None:
        """Test that min_samples_for_random_effects must be >= 1."""
        with pytest.raises(ValidationError, match=">= 1"):
            MixedEffectsConfig(min_samples_for_random_effects=0)

    def test_random_effects_spec_optional(self) -> None:
        """Test that random_effects_spec is optional (Phase 6+)."""
        config = MixedEffectsConfig()
        assert config.random_effects_spec is None

        spec = RandomEffectsSpec(grouping_factors={"participant": "intercept"})
        config_with_spec = MixedEffectsConfig(random_effects_spec=spec)
        assert config_with_spec.random_effects_spec == spec


class DummyModel(ActiveLearningModel):
    """Dummy model for testing ActiveLearningModel base class."""

    @property
    def supported_task_types(self):
        """Return supported task types."""
        return ["forced_choice"]

    def validate_item_compatibility(self, item, item_template):
        """Validate item compatibility."""

    def _prepare_training_data(self, items, labels, participant_ids):
        """Prepare training data."""
        return [], [], []

    def _initialize_random_effects(self, n_classes):
        """Initialize random effects."""

    def _do_training(self, items, labels, participant_ids, validation_data):
        """Perform training."""
        return {}

    def _do_predict(self, items, participant_ids):
        """Make predictions."""
        return []

    def _do_predict_proba(self, items, participant_ids):
        """Return prediction probabilities."""
        return np.array([])

    def _get_save_state(self):
        """Get save state."""
        return {}

    def _save_model_components(self, save_path):
        """Save model components."""

    def _load_model_components(self, load_path):
        """Load model components."""

    def _restore_training_state(self, config_dict):
        """Restore training state."""

    def _get_random_effects_fixed_head(self):
        """Get fixed head for random effects."""
        return None

    def _get_n_classes_for_random_effects(self):
        """Get number of classes for random effects."""
        return 2


class DummyConfig:
    """Dummy config with mixed_effects field."""

    def __init__(self, mixed_effects=None):
        self.mixed_effects = (
            mixed_effects if mixed_effects is not None else MixedEffectsConfig()
        )


class TestActiveLearningModel:
    """Test ActiveLearningModel base class initialization and validation."""

    def test_init_with_valid_config(self) -> None:
        """Test initialization with valid config containing mixed_effects."""
        config = DummyConfig(mixed_effects=MixedEffectsConfig(mode="fixed"))
        model = DummyModel(config)

        assert model.config == config
        assert model.config.mixed_effects.mode == "fixed"

    def test_init_without_mixed_effects_field_raises(self) -> None:
        """Test that config without mixed_effects field raises ValueError."""

        class BadConfig:
            pass

        config = BadConfig()

        with pytest.raises(ValueError, match="must have a 'mixed_effects' field"):
            DummyModel(config)

    def test_init_with_wrong_type_mixed_effects_raises(self) -> None:
        """Test that config with wrong type for mixed_effects raises ValueError."""

        class BadConfig:
            def __init__(self):
                self.mixed_effects = "not_a_MixedEffectsConfig"  # Wrong type

        config = BadConfig()

        with pytest.raises(ValueError, match="must be MixedEffectsConfig"):
            DummyModel(config)

    def test_init_with_random_intercepts_config(self) -> None:
        """Test initialization with random_intercepts mode."""
        config = DummyConfig(
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts", prior_variance=0.5
            )
        )
        model = DummyModel(config)

        assert model.config.mixed_effects.mode == "random_intercepts"
        assert model.config.mixed_effects.prior_variance == 0.5

    def test_init_with_random_slopes_config(self) -> None:
        """Test initialization with random_slopes mode."""
        config = DummyConfig(
            mixed_effects=MixedEffectsConfig(
                mode="random_slopes", adaptive_regularization=True
            )
        )
        model = DummyModel(config)

        assert model.config.mixed_effects.mode == "random_slopes"
        assert model.config.mixed_effects.adaptive_regularization is True


class TestModelPrediction:
    """Test ModelPrediction model."""

    def test_create_model_prediction(self) -> None:
        """Test creating a model prediction."""
        pred = ModelPrediction(
            item_id="abc123",
            probabilities={"option_a": 0.7, "option_b": 0.3},
            predicted_class="option_a",
            confidence=0.7,
        )

        assert pred.item_id == "abc123"
        assert pred.predicted_class == "option_a"
        assert pred.confidence == 0.7
        assert pred.probabilities["option_a"] == 0.7
        assert pred.probabilities["option_b"] == 0.3
