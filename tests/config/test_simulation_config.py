"""Tests for simulation configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from didactic.api import ValidationError

from bead.config.simulation import (
    NoiseModelConfig,
    SimulatedAnnotatorConfig,
    SimulationRunnerConfig,
)


class TestNoiseModelConfig:
    """Tests for NoiseModelConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating NoiseModelConfig with default values."""
        config = NoiseModelConfig()
        assert config.noise_type == "temperature"
        assert config.temperature == 1.0
        assert config.bias_strength == 0.0
        assert config.bias_type is None
        assert config.random_noise_stddev == 0.0

    def test_temperature_noise(self) -> None:
        """Test creating temperature noise model."""
        config = NoiseModelConfig(noise_type="temperature", temperature=2.0)
        assert config.noise_type == "temperature"
        assert config.temperature == 2.0

    def test_systematic_noise(self) -> None:
        """Test creating systematic noise model."""
        config = NoiseModelConfig(
            noise_type="systematic", bias_strength=0.3, bias_type="length"
        )
        assert config.noise_type == "systematic"
        assert config.bias_strength == 0.3
        assert config.bias_type == "length"

    def test_random_noise(self) -> None:
        """Test creating random noise model."""
        config = NoiseModelConfig(noise_type="random", random_noise_stddev=0.1)
        assert config.noise_type == "random"
        assert config.random_noise_stddev == 0.1

    def test_no_noise(self) -> None:
        """Test creating no noise model."""
        config = NoiseModelConfig(noise_type="none")
        assert config.noise_type == "none"

    def test_temperature_validation_minimum(self) -> None:
        """Test temperature must be >= 0.01."""
        with pytest.raises(ValidationError):
            NoiseModelConfig(temperature=0.0)

    def test_temperature_validation_maximum(self) -> None:
        """Test temperature must be <= 10.0."""
        with pytest.raises(ValidationError):
            NoiseModelConfig(temperature=11.0)

    def test_bias_strength_validation_minimum(self) -> None:
        """Test bias_strength must be >= 0.0."""
        with pytest.raises(ValidationError):
            NoiseModelConfig(bias_strength=-0.1)

    def test_bias_strength_validation_maximum(self) -> None:
        """Test bias_strength must be <= 1.0."""
        with pytest.raises(ValidationError):
            NoiseModelConfig(bias_strength=1.1)

    def test_random_noise_stddev_validation(self) -> None:
        """Test random_noise_stddev must be >= 0.0."""
        with pytest.raises(ValidationError):
            NoiseModelConfig(random_noise_stddev=-0.1)


class TestSimulatedAnnotatorConfig:
    """Tests for SimulatedAnnotatorConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating SimulatedAnnotatorConfig with default values."""
        config = SimulatedAnnotatorConfig()
        assert config.strategy == "lm_score"
        assert config.noise_model.noise_type == "temperature"
        assert config.dsl_expression is None
        assert config.random_state is None
        assert config.model_output_key == "lm_score"
        assert config.fallback_to_random is True

    def test_lm_score_strategy(self) -> None:
        """Test LM score strategy configuration."""
        config = SimulatedAnnotatorConfig(
            strategy="lm_score",
            noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.5),
            random_state=42,
        )
        assert config.strategy == "lm_score"
        assert config.noise_model.temperature == 1.5
        assert config.random_state == 42

    def test_distance_strategy(self) -> None:
        """Test distance-based strategy configuration."""
        config = SimulatedAnnotatorConfig(
            strategy="distance",
            model_output_key="embedding",
            noise_model=NoiseModelConfig(noise_type="none"),
        )
        assert config.strategy == "distance"
        assert config.model_output_key == "embedding"
        assert config.noise_model.noise_type == "none"

    def test_random_strategy(self) -> None:
        """Test random strategy configuration."""
        config = SimulatedAnnotatorConfig(strategy="random", random_state=123)
        assert config.strategy == "random"
        assert config.random_state == 123

    def test_oracle_strategy(self) -> None:
        """Test oracle strategy configuration."""
        config = SimulatedAnnotatorConfig(strategy="oracle")
        assert config.strategy == "oracle"

    def test_dsl_strategy(self) -> None:
        """Test DSL strategy configuration."""
        config = SimulatedAnnotatorConfig(
            strategy="dsl",
            dsl_expression="sample_categorical(softmax(model_scores))",
        )
        assert config.strategy == "dsl"
        assert config.dsl_expression == "sample_categorical(softmax(model_scores))"

    def test_custom_model_output_key(self) -> None:
        """Test custom model output key."""
        config = SimulatedAnnotatorConfig(model_output_key="custom_score")
        assert config.model_output_key == "custom_score"

    def test_fallback_to_random_disabled(self) -> None:
        """Test disabling fallback to random."""
        config = SimulatedAnnotatorConfig(fallback_to_random=False)
        assert config.fallback_to_random is False


class TestSimulationRunnerConfig:
    """Tests for SimulationRunnerConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating SimulationRunnerConfig with default values."""
        config = SimulationRunnerConfig()
        assert len(config.annotator_configs) == 1
        assert config.n_annotators == 1
        assert config.inter_annotator_correlation is None
        assert config.output_format == "dict"
        assert config.save_path is None

    def test_single_annotator(self) -> None:
        """Test single annotator configuration."""
        config = SimulationRunnerConfig(
            annotator_configs=[SimulatedAnnotatorConfig(strategy="lm_score")],
            n_annotators=1,
        )
        assert len(config.annotator_configs) == 1
        assert config.n_annotators == 1

    def test_multiple_annotators(self) -> None:
        """Test multiple annotators configuration."""
        config = SimulationRunnerConfig(
            annotator_configs=[
                SimulatedAnnotatorConfig(strategy="lm_score", random_state=1),
                SimulatedAnnotatorConfig(strategy="lm_score", random_state=2),
                SimulatedAnnotatorConfig(strategy="lm_score", random_state=3),
            ],
            n_annotators=3,
        )
        assert len(config.annotator_configs) == 3
        assert config.n_annotators == 3

    def test_correlated_annotators(self) -> None:
        """Test correlated annotators configuration."""
        config = SimulationRunnerConfig(
            annotator_configs=[SimulatedAnnotatorConfig(strategy="lm_score")],
            n_annotators=5,
            inter_annotator_correlation=0.7,
        )
        assert config.n_annotators == 5
        assert config.inter_annotator_correlation == 0.7

    def test_output_format_dict(self) -> None:
        """Test dict output format."""
        config = SimulationRunnerConfig(output_format="dict")
        assert config.output_format == "dict"

    def test_output_format_dataframe(self) -> None:
        """Test dataframe output format."""
        config = SimulationRunnerConfig(output_format="dataframe")
        assert config.output_format == "dataframe"

    def test_output_format_jsonl(self) -> None:
        """Test jsonl output format."""
        config = SimulationRunnerConfig(output_format="jsonl")
        assert config.output_format == "jsonl"

    def test_save_path(self) -> None:
        """Test save path configuration."""
        config = SimulationRunnerConfig(save_path=Path("results.json"))
        assert config.save_path == Path("results.json")

    def test_n_annotators_validation_minimum(self) -> None:
        """Test n_annotators must be >= 1."""
        with pytest.raises(ValidationError):
            SimulationRunnerConfig(n_annotators=0)

    def test_n_annotators_validation_maximum(self) -> None:
        """Test n_annotators must be <= 100."""
        with pytest.raises(ValidationError):
            SimulationRunnerConfig(n_annotators=101)

    def test_correlation_validation_minimum(self) -> None:
        """Test correlation must be >= 0.0."""
        with pytest.raises(ValidationError):
            SimulationRunnerConfig(inter_annotator_correlation=-0.1)

    def test_correlation_validation_maximum(self) -> None:
        """Test correlation must be <= 1.0."""
        with pytest.raises(ValidationError):
            SimulationRunnerConfig(inter_annotator_correlation=1.1)
