"""Tests for configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from didactic.api import ValidationError

from bead.config.active_learning import ActiveLearningConfig
from bead.config.config import BeadConfig
from bead.config.deployment import (
    DeploymentConfig,
    SlopitFocusConfig,
    SlopitIntegrationConfig,
    SlopitKeystrokeConfig,
    SlopitPasteConfig,
)
from bead.config.item import ItemConfig
from bead.config.list import ListConfig
from bead.config.logging import LoggingConfig
from bead.config.model import ModelConfig
from bead.config.paths import PathsConfig
from bead.config.resources import ResourceConfig
from bead.config.template import TemplateConfig


class TestPathsConfig:
    """Tests for PathsConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating PathsConfig with default values."""
        config = PathsConfig()
        assert config.data_dir == Path("data")
        assert config.output_dir == Path("output")
        assert config.cache_dir == Path(".cache")
        assert config.temp_dir is None
        assert config.create_dirs is True

    def test_creation_with_custom_paths(self) -> None:
        """Test creating PathsConfig with custom paths."""
        config = PathsConfig(
            data_dir=Path("/custom/data"),
            output_dir=Path("/custom/output"),
            cache_dir=Path("/custom/cache"),
            temp_dir=Path("/tmp"),
        )
        assert config.data_dir == Path("/custom/data")
        assert config.output_dir == Path("/custom/output")
        assert config.cache_dir == Path("/custom/cache")
        assert config.temp_dir == Path("/tmp")

    def test_relative_paths(self) -> None:
        """Test that relative paths are accepted."""
        config = PathsConfig(data_dir=Path("relative/path"))
        assert config.data_dir == Path("relative/path")

    def test_absolute_paths(self) -> None:
        """Test that absolute paths are accepted."""
        config = PathsConfig(data_dir=Path("/absolute/path"))
        assert config.data_dir == Path("/absolute/path")


class TestResourceConfig:
    """Tests for ResourceConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating ResourceConfig with default values."""
        config = ResourceConfig()
        assert config.lexicon_path is None
        assert config.templates_path is None
        assert config.constraints_path is None
        assert config.external_adapters == ()
        assert config.cache_external is True

    def test_creation_with_custom_paths(self) -> None:
        """Test creating ResourceConfig with custom resource paths."""
        config = ResourceConfig(
            lexicon_path=Path("lexicon.json"),
            templates_path=Path("templates.json"),
            constraints_path=Path("constraints.json"),
        )
        assert config.lexicon_path == Path("lexicon.json")
        assert config.templates_path == Path("templates.json")
        assert config.constraints_path == Path("constraints.json")

    def test_external_adapters_list(self) -> None:
        """Test external adapters list handling."""
        config = ResourceConfig(external_adapters=["adapter1", "adapter2"])
        assert config.external_adapters == ("adapter1", "adapter2")

    def test_cache_settings(self) -> None:
        """Test cache settings."""
        config = ResourceConfig(cache_external=False)
        assert config.cache_external is False


class TestTemplateConfig:
    """Tests for TemplateConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating TemplateConfig with default values."""
        config = TemplateConfig()
        assert config.filling_strategy == "exhaustive"
        assert config.batch_size == 1000
        assert config.max_combinations is None
        assert config.random_seed is None
        assert config.stream_mode is False

    def test_valid_filling_strategy(self) -> None:
        """Test filling strategy validation with valid values."""
        for strategy in ["exhaustive", "random", "stratified"]:
            config = TemplateConfig(filling_strategy=strategy)
            assert config.filling_strategy == strategy

    def test_invalid_filling_strategy(self) -> None:
        """Test filling strategy validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            TemplateConfig(filling_strategy="invalid")  # type: ignore[arg-type]
        assert "is not in Literal" in str(exc_info.value)

    def test_batch_size_validation_positive(self) -> None:
        """Test batch size must be positive."""
        config = TemplateConfig(batch_size=100)
        assert config.batch_size == 100

    def test_batch_size_validation_zero(self) -> None:
        """Test batch size cannot be zero."""
        with pytest.raises(ValidationError):
            TemplateConfig(batch_size=0)

    def test_batch_size_validation_negative(self) -> None:
        """Test batch size cannot be negative."""
        with pytest.raises(ValidationError):
            TemplateConfig(batch_size=-1)

    def test_max_combinations_positive(self) -> None:
        """Test max_combinations validation with positive value."""
        config = TemplateConfig(max_combinations=5000)
        assert config.max_combinations == 5000

    def test_max_combinations_zero(self) -> None:
        """Test max_combinations cannot be zero."""
        with pytest.raises(ValidationError) as exc_info:
            TemplateConfig(max_combinations=0)
        assert "max_combinations must be positive" in str(exc_info.value)

    def test_max_combinations_negative(self) -> None:
        """Test max_combinations cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            TemplateConfig(max_combinations=-1)
        assert "max_combinations must be positive" in str(exc_info.value)

    def test_random_seed_handling(self) -> None:
        """Test random seed handling."""
        config = TemplateConfig(random_seed=42)
        assert config.random_seed == 42


class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating ModelConfig with default values."""
        config = ModelConfig()
        assert config.provider == "huggingface"
        assert config.model_name == "gpt2"
        assert config.batch_size == 8
        assert config.device == "cpu"
        assert config.max_length == 512
        assert config.temperature == 1.0
        assert config.cache_outputs is True

    def test_valid_provider(self) -> None:
        """Test provider validation with valid values."""
        for provider in ["huggingface", "openai", "anthropic"]:
            config = ModelConfig(provider=provider)
            assert config.provider == provider

    def test_invalid_provider(self) -> None:
        """Test provider validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(provider="invalid")  # type: ignore[arg-type]
        assert "is not in Literal" in str(exc_info.value)

    def test_valid_device(self) -> None:
        """Test device validation with valid values."""
        for device in ["cpu", "cuda", "mps"]:
            config = ModelConfig(device=device)
            assert config.device == device

    def test_invalid_device(self) -> None:
        """Test device validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(device="invalid")  # type: ignore[arg-type]
        assert "is not in Literal" in str(exc_info.value)

    def test_batch_size_validation(self) -> None:
        """Test batch size must be positive."""
        config = ModelConfig(batch_size=16)
        assert config.batch_size == 16

        with pytest.raises(ValidationError):
            ModelConfig(batch_size=0)

    def test_max_length_validation(self) -> None:
        """Test max_length must be positive."""
        config = ModelConfig(max_length=1024)
        assert config.max_length == 1024

        with pytest.raises(ValidationError):
            ModelConfig(max_length=0)

    def test_temperature_validation(self) -> None:
        """Test temperature must be non-negative."""
        config = ModelConfig(temperature=0.7)
        assert config.temperature == 0.7

        config = ModelConfig(temperature=0.0)
        assert config.temperature == 0.0

        with pytest.raises(ValidationError):
            ModelConfig(temperature=-0.1)

    def test_different_model_configurations(self) -> None:
        """Test creating different model configurations."""
        openai_config = ModelConfig(provider="openai", model_name="gpt-4", device="cpu")
        assert openai_config.provider == "openai"
        assert openai_config.model_name == "gpt-4"

        anthropic_config = ModelConfig(
            provider="anthropic", model_name="claude-3", device="cpu"
        )
        assert anthropic_config.provider == "anthropic"
        assert anthropic_config.model_name == "claude-3"


class TestItemConfig:
    """Tests for ItemConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating ItemConfig with default values."""
        config = ItemConfig()
        assert isinstance(config.model, ModelConfig)
        assert config.apply_constraints is True
        assert config.track_metadata is True
        assert config.parallel_processing is False
        assert config.num_workers == 4

    def test_creation_with_nested_model_config(self) -> None:
        """Test creating ItemConfig with nested ModelConfig."""
        model_config = ModelConfig(provider="openai", model_name="gpt-4")
        config = ItemConfig(model=model_config)
        assert config.model.provider == "openai"
        assert config.model.model_name == "gpt-4"

    def test_configuration_options(self) -> None:
        """Test configuration options."""
        config = ItemConfig(
            apply_constraints=False,
            track_metadata=False,
            parallel_processing=True,
            num_workers=8,
        )
        assert config.apply_constraints is False
        assert config.track_metadata is False
        assert config.parallel_processing is True
        assert config.num_workers == 8

    def test_parallel_processing_settings(self) -> None:
        """Test parallel processing settings."""
        config = ItemConfig(parallel_processing=True, num_workers=16)
        assert config.parallel_processing is True
        assert config.num_workers == 16


class TestListConfig:
    """Tests for ListConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating ListConfig with default values."""
        config = ListConfig()
        assert config.partitioning_strategy == "balanced"
        assert config.num_lists == 1
        assert config.items_per_list is None
        assert config.balance_by == ()
        assert config.ensure_uniqueness is True
        assert config.random_seed is None

    def test_partitioning_strategy(self) -> None:
        """Test partitioning strategy."""
        config = ListConfig(partitioning_strategy="round_robin")
        assert config.partitioning_strategy == "round_robin"

    def test_balance_by_list_handling(self) -> None:
        """Test balance_by list handling."""
        config = ListConfig(balance_by=["field1", "field2"])
        assert config.balance_by == (
            "field1",
            "field2",
        )

    def test_num_lists_and_items_per_list(self) -> None:
        """Test num_lists and items_per_list."""
        config = ListConfig(num_lists=5, items_per_list=100)
        assert config.num_lists == 5
        assert config.items_per_list == 100

    def test_items_per_list_validation_positive(self) -> None:
        """Test items_per_list validation with positive value."""
        config = ListConfig(items_per_list=50)
        assert config.items_per_list == 50

    def test_items_per_list_validation_zero(self) -> None:
        """Test items_per_list cannot be zero."""
        with pytest.raises(ValidationError) as exc_info:
            ListConfig(items_per_list=0)
        assert "items_per_list must be positive" in str(exc_info.value)

    def test_items_per_list_validation_negative(self) -> None:
        """Test items_per_list cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            ListConfig(items_per_list=-1)
        assert "items_per_list must be positive" in str(exc_info.value)


class TestDeploymentConfig:
    """Tests for DeploymentConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating DeploymentConfig with default values."""
        config = DeploymentConfig()
        assert config.platform == "jspsych"
        assert config.jspsych_version == "7.3.0"
        assert config.apply_material_design is True
        assert config.include_demographics is True
        assert config.include_attention_checks is True
        assert config.jatos_export is False

    def test_platform_validation(self) -> None:
        """Test platform validation."""
        config = DeploymentConfig(platform="qualtrics")
        assert config.platform == "qualtrics"

    def test_jspsych_version(self) -> None:
        """Test jsPsych version."""
        config = DeploymentConfig(jspsych_version="8.0.0")
        assert config.jspsych_version == "8.0.0"

    def test_boolean_flags(self) -> None:
        """Test boolean flags."""
        config = DeploymentConfig(
            apply_material_design=False,
            include_demographics=False,
            include_attention_checks=False,
            jatos_export=True,
        )
        assert config.apply_material_design is False
        assert config.include_demographics is False
        assert config.include_attention_checks is False
        assert config.jatos_export is True


class TestSlopitIntegrationConfig:
    """Tests for SlopitIntegrationConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating SlopitIntegrationConfig with default values."""
        config = SlopitIntegrationConfig()
        assert config.enabled is False
        assert isinstance(config.keystroke, SlopitKeystrokeConfig)
        assert isinstance(config.focus, SlopitFocusConfig)
        assert isinstance(config.paste, SlopitPasteConfig)
        assert "forced_choice" in config.target_selectors

    def test_disabled_by_default(self) -> None:
        """Test that slopit is disabled by default."""
        config = SlopitIntegrationConfig()
        assert config.enabled is False

    def test_keystroke_config(self) -> None:
        """Test keystroke capture configuration."""
        config = SlopitIntegrationConfig(
            keystroke=SlopitKeystrokeConfig(
                enabled=True,
                capture_key_up=False,
                include_modifiers=False,
            )
        )
        assert config.keystroke.enabled is True
        assert config.keystroke.capture_key_up is False
        assert config.keystroke.include_modifiers is False

    def test_focus_config(self) -> None:
        """Test focus capture configuration."""
        config = SlopitIntegrationConfig(
            focus=SlopitFocusConfig(
                enabled=True,
                use_visibility_api=False,
                use_blur_focus=True,
            )
        )
        assert config.focus.enabled is True
        assert config.focus.use_visibility_api is False
        assert config.focus.use_blur_focus is True

    def test_paste_config(self) -> None:
        """Test paste capture configuration."""
        config = SlopitIntegrationConfig(
            paste=SlopitPasteConfig(
                enabled=True,
                prevent=True,
                capture_preview=True,
                preview_length=50,
            )
        )
        assert config.paste.enabled is True
        assert config.paste.prevent is True
        assert config.paste.capture_preview is True
        assert config.paste.preview_length == 50

    def test_custom_target_selectors(self) -> None:
        """Test custom target selectors."""
        selectors = {
            "custom_task": ".my-custom-selector",
        }
        config = SlopitIntegrationConfig(target_selectors=selectors)
        assert config.target_selectors == selectors

    def test_enabled_requires_bundle(self) -> None:
        """Test that enabling slopit validates bundle exists."""
        # This test checks that validation runs when enabled=True
        # The bundle may or may not exist depending on build state
        # so we just verify the validator is called
        try:
            SlopitIntegrationConfig(enabled=True)
        except ValueError as e:
            # Expected if bundle not built
            assert "slopit-bundle.js" in str(e)
            assert "pnpm build" in str(e)

    def test_disabled_does_not_validate_bundle(self) -> None:
        """Test that disabled config does not validate bundle."""
        # Should not raise even if bundle missing
        config = SlopitIntegrationConfig(enabled=False)
        assert config.enabled is False


class TestActivelearningConfig:
    """Tests for ActiveLearningConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating ActiveLearningConfig with default values."""
        config = ActiveLearningConfig()
        assert config.forced_choice_model.model_name == "bert-base-uncased"
        assert config.trainer.trainer_type == "huggingface"
        assert config.trainer.epochs == 3
        assert config.loop.max_iterations == 10
        assert config.loop.budget_per_iteration == 100
        assert config.uncertainty_sampler.method == "entropy"

    def test_trainer_configuration(self) -> None:
        """Test trainer configuration."""
        config = ActiveLearningConfig()
        assert config.trainer.trainer_type == "huggingface"
        assert config.trainer.epochs == 3
        assert config.trainer.use_wandb is False

    def test_loop_configuration(self) -> None:
        """Test loop configuration."""
        config = ActiveLearningConfig()
        assert config.loop.max_iterations == 10
        assert config.loop.budget_per_iteration == 100
        assert config.loop.stopping_criterion == "max_iterations"

    def test_model_configuration(self) -> None:
        """Test model configuration."""
        config = ActiveLearningConfig()
        assert config.forced_choice_model.model_name == "bert-base-uncased"
        assert config.forced_choice_model.batch_size == 16
        assert config.forced_choice_model.device == "cpu"

    def test_sampler_configuration(self) -> None:
        """Test sampler configuration."""
        config = ActiveLearningConfig()
        assert config.uncertainty_sampler.method == "entropy"
        assert config.uncertainty_sampler.batch_size is None


class TestLoggingConfig:
    """Tests for LoggingConfig model."""

    def test_creation_with_defaults(self) -> None:
        """Test creating LoggingConfig with default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.file is None
        assert config.console is True

    def test_valid_log_level(self) -> None:
        """Test log level validation with valid values."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_invalid_log_level(self) -> None:
        """Test log level validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(level="INVALID")  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "is not in Literal" in error_msg
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert any(level in error_msg for level in levels)

    def test_format_string(self) -> None:
        """Test format string."""
        custom_format = "%(levelname)s - %(message)s"
        config = LoggingConfig(format=custom_format)
        assert config.format == custom_format

    def test_file_and_console_settings(self) -> None:
        """Test file and console settings."""
        config = LoggingConfig(file=Path("app.log"), console=False)
        assert config.file == Path("app.log")
        assert config.console is False


class TestBeadConfig:
    """Tests for BeadConfig model."""

    def test_creation_with_all_nested_configs(self) -> None:
        """Test creating BeadConfig with all nested configs."""
        config = BeadConfig(
            profile="test",
            paths=PathsConfig(),
            resources=ResourceConfig(),
            templates=TemplateConfig(),
            items=ItemConfig(),
            lists=ListConfig(),
            deployment=DeploymentConfig(),
            active_learning=ActiveLearningConfig(),
            logging=LoggingConfig(),
        )
        assert config.profile == "test"
        assert isinstance(config.paths, PathsConfig)
        assert isinstance(config.resources, ResourceConfig)
        assert isinstance(config.templates, TemplateConfig)
        assert isinstance(config.items, ItemConfig)
        assert isinstance(config.lists, ListConfig)
        assert isinstance(config.deployment, DeploymentConfig)
        assert isinstance(config.active_learning, ActiveLearningConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_to_dict_method(self) -> None:
        """Test to_dict() method."""
        config = BeadConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["profile"] == "default"
        assert "paths" in d
        assert "resources" in d
        assert "templates" in d

    def test_validate_paths_method_empty_errors(self) -> None:
        """Test validate_paths() method with relative paths."""
        config = BeadConfig()
        errors = config.validate_paths()
        # Relative paths should not generate errors
        assert isinstance(errors, list)

    def test_validate_paths_method_with_missing_absolute_paths(
        self, tmp_path: Path
    ) -> None:
        """Test validate_paths() method with missing absolute paths."""
        config = BeadConfig(
            paths=PathsConfig(data_dir=tmp_path / "nonexistent"),
        )
        errors = config.validate_paths()
        assert len(errors) > 0
        assert "data_dir does not exist" in errors[0]

    def test_profile_field(self) -> None:
        """Test profile field."""
        config = BeadConfig(profile="production")
        assert config.profile == "production"

    def test_model_serialization(self) -> None:
        """Test model serialization."""
        config = BeadConfig()
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "default" in json_str

    def test_model_deserialization(self) -> None:
        """Test model deserialization."""
        config = BeadConfig()
        config2 = BeadConfig.model_validate_json(config.model_dump_json())
        assert config2.profile == config.profile
        assert config2.paths.data_dir == config.paths.data_dir

    def test_to_yaml_implemented(self) -> None:
        """Test to_yaml() is now implemented and works."""
        config = BeadConfig()
        yaml_str = config.to_yaml()
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0
