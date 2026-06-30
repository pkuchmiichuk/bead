"""Tests for configuration validation."""

from pathlib import Path

from bead.config.defaults import get_default_config
from bead.config.profiles import get_profile
from bead.config.validation import (
    check_deployment_configuration,
    check_model_configuration,
    check_paths_exist,
    check_resource_compatibility,
    check_training_configuration,
    validate_config,
)


class TestCheckPathsExist:
    """Tests for check_paths_exist function."""

    def test_check_paths_exist_with_defaults(self) -> None:
        """Test path checking with default config (relative paths)."""
        config = get_default_config()
        errors = check_paths_exist(config)
        # Default config uses relative paths, should not error
        assert isinstance(errors, list)

    def test_check_paths_exist_with_nonexistent_absolute_path(
        self, tmp_path: Path
    ) -> None:
        """Test path checking with nonexistent absolute path."""
        config = get_default_config()
        config = config.with_(
            paths=config.paths.with_(data_dir=tmp_path / "nonexistent")
        )
        errors = check_paths_exist(config)
        assert len(errors) > 0
        assert any("data_dir does not exist" in e for e in errors)

    def test_check_paths_exist_with_existing_absolute_path(
        self, tmp_path: Path
    ) -> None:
        """Test path checking with existing absolute path."""
        config = get_default_config()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = config.with_(paths=config.paths.with_(data_dir=data_dir))
        errors = check_paths_exist(config)
        # Should not complain about data_dir
        assert not any("data_dir does not exist" in e for e in errors)

    def test_check_paths_exist_with_missing_resource_path(self, tmp_path: Path) -> None:
        """Test path checking with missing resource path."""
        config = get_default_config()
        config = config.with_(
            resources=config.resources.with_(lexicon_path=tmp_path / "nonexistent.json")
        )
        errors = check_paths_exist(config)
        assert len(errors) > 0
        assert any("lexicon_path does not exist" in e for e in errors)

    def test_check_paths_exist_with_missing_logging_dir(self, tmp_path: Path) -> None:
        """Test path checking with missing logging directory."""
        config = get_default_config()
        config = config.with_(
            active_learning=config.active_learning.with_(
                trainer=config.active_learning.trainer.with_(
                    logging_dir=tmp_path / "nonexistent"
                )
            )
        )
        errors = check_paths_exist(config)
        assert len(errors) > 0
        assert any("logging_dir does not exist" in e for e in errors)


class TestCheckResourceCompatibility:
    """Tests for check_resource_compatibility function."""

    def test_check_resource_compatibility_with_defaults(self) -> None:
        """Test resource compatibility with default config."""
        config = get_default_config()
        errors = check_resource_compatibility(config)
        assert isinstance(errors, list)

    def test_check_resource_compatibility_templates_without_lexicon(self) -> None:
        """Test that templates_path without lexicon_path raises error."""
        config = get_default_config()
        config = config.with_(
            resources=config.resources.with_(templates_path=Path("templates.json"))
        )
        config = config.with_(resources=config.resources.with_(lexicon_path=None))
        errors = check_resource_compatibility(config)
        assert len(errors) > 0
        assert any("lexicon" in e.lower() for e in errors)

    def test_check_resource_compatibility_both_specified(self) -> None:
        """Test that both templates_path and lexicon_path work together."""
        config = get_default_config()
        config = config.with_(
            resources=config.resources.with_(templates_path=Path("templates.json"))
        )
        config = config.with_(
            resources=config.resources.with_(lexicon_path=Path("lexicon.json"))
        )
        errors = check_resource_compatibility(config)
        # Should not error about missing lexicon
        assert not any("lexicon" in e.lower() and "not" in e.lower() for e in errors)


class TestCheckModelConfiguration:
    """Tests for check_model_configuration function."""

    def test_check_model_configuration_cpu(self) -> None:
        """Test model configuration with CPU device."""
        config = get_default_config()
        config = config.with_(
            items=config.items.with_(model=config.items.model.with_(device="cpu"))
        )
        errors = check_model_configuration(config)
        # CPU should always work
        assert len(errors) == 0

    def test_check_model_configuration_cuda_without_torch(self) -> None:
        """Test model configuration with CUDA device."""
        config = get_default_config()
        config = config.with_(
            items=config.items.with_(model=config.items.model.with_(device="cuda"))
        )
        errors = check_model_configuration(config)
        # Will error if torch not installed or CUDA not available
        # We don't know the test environment, so just check it returns a list
        assert isinstance(errors, list)

    def test_check_model_configuration_mps(self) -> None:
        """Test model configuration with MPS device."""
        config = get_default_config()
        config = config.with_(
            items=config.items.with_(model=config.items.model.with_(device="mps"))
        )
        errors = check_model_configuration(config)
        # Will error if torch not installed or MPS not available
        # We don't know the test environment, so just check it returns a list
        assert isinstance(errors, list)


class TestCheckTrainingConfiguration:
    """Tests for check_training_configuration function."""

    def test_check_training_configuration_valid(self) -> None:
        """Test training configuration with valid settings."""
        config = get_default_config()
        errors = check_training_configuration(config)
        assert len(errors) == 0

    def test_check_training_configuration_negative_batch_size(self) -> None:
        """Test training configuration with negative batch size."""
        config = get_default_config()
        config = config.with_(
            active_learning=config.active_learning.with_(
                forced_choice_model=config.active_learning.forced_choice_model.with_(
                    batch_size=-1
                )
            )
        )
        errors = check_training_configuration(config)
        assert len(errors) > 0
        assert any("batch size" in e.lower() for e in errors)

    def test_check_training_configuration_zero_batch_size(self) -> None:
        """Test training configuration with zero batch size."""
        config = get_default_config()
        config = config.with_(
            active_learning=config.active_learning.with_(
                forced_choice_model=config.active_learning.forced_choice_model.with_(
                    batch_size=0
                )
            )
        )
        errors = check_training_configuration(config)
        assert len(errors) > 0
        assert any("batch size" in e.lower() for e in errors)

    def test_check_training_configuration_negative_epochs(self) -> None:
        """Test training configuration with negative epochs."""
        config = get_default_config()
        config = config.with_(
            active_learning=config.active_learning.with_(
                trainer=config.active_learning.trainer.with_(epochs=-5)
            )
        )
        errors = check_training_configuration(config)
        assert len(errors) > 0
        assert any("epochs" in e.lower() for e in errors)

    def test_check_training_configuration_zero_epochs(self) -> None:
        """Test training configuration with zero epochs."""
        config = get_default_config()
        config = config.with_(
            active_learning=config.active_learning.with_(
                trainer=config.active_learning.trainer.with_(epochs=0)
            )
        )
        errors = check_training_configuration(config)
        assert len(errors) > 0
        assert any("epochs" in e.lower() for e in errors)

    def test_check_training_configuration_negative_learning_rate(self) -> None:
        """Test training configuration with negative learning rate."""
        config = get_default_config()
        config = config.with_(
            active_learning=config.active_learning.with_(
                forced_choice_model=config.active_learning.forced_choice_model.with_(
                    learning_rate=-0.001
                )
            )
        )
        errors = check_training_configuration(config)
        assert len(errors) > 0
        assert any("learning rate" in e.lower() for e in errors)

    def test_check_training_configuration_zero_learning_rate(self) -> None:
        """Test training configuration with zero learning rate."""
        config = get_default_config()
        config = config.with_(
            active_learning=config.active_learning.with_(
                forced_choice_model=config.active_learning.forced_choice_model.with_(
                    learning_rate=0.0
                )
            )
        )
        errors = check_training_configuration(config)
        assert len(errors) > 0
        assert any("learning rate" in e.lower() for e in errors)


class TestCheckDeploymentConfiguration:
    """Tests for check_deployment_configuration function."""

    def test_check_deployment_configuration_jspsych_valid(self) -> None:
        """Test deployment configuration for jsPsych with version."""
        config = get_default_config()
        config = config.with_(deployment=config.deployment.with_(platform="jspsych"))
        config = config.with_(
            deployment=config.deployment.with_(jspsych_version="7.3.0")
        )
        errors = check_deployment_configuration(config)
        assert len(errors) == 0

    def test_check_deployment_configuration_jspsych_missing_version(self) -> None:
        """Test deployment configuration for jsPsych without version."""
        config = get_default_config()
        config = config.with_(deployment=config.deployment.with_(platform="jspsych"))
        config = config.with_(deployment=config.deployment.with_(jspsych_version=None))
        errors = check_deployment_configuration(config)
        assert len(errors) > 0
        assert any("jspsych_version" in e.lower() for e in errors)

    def test_check_deployment_configuration_other_platform(self) -> None:
        """Test deployment configuration for non-jsPsych platform."""
        config = get_default_config()
        config = config.with_(deployment=config.deployment.with_(platform="qualtrics"))
        errors = check_deployment_configuration(config)
        # Non-jsPsych platforms don't require jspsych_version
        assert not any("jspsych_version" in e.lower() for e in errors)


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_config_default(self) -> None:
        """Test validation of default configuration."""
        config = get_default_config()
        errors = validate_config(config)
        # Default config should be valid
        assert len(errors) == 0

    def test_validate_config_dev_profile(self) -> None:
        """Test validation of dev profile."""
        config = get_profile("dev")
        errors = validate_config(config)
        assert isinstance(errors, list)

    def test_validate_config_prod_profile(self) -> None:
        """Test validation of prod profile."""
        config = get_profile("prod")
        errors = validate_config(config)
        assert isinstance(errors, list)

    def test_validate_config_test_profile(self) -> None:
        """Test validation of test profile."""
        config = get_profile("test")
        errors = validate_config(config)
        assert isinstance(errors, list)

    def test_validate_config_with_multiple_errors(self, tmp_path: Path) -> None:
        """Test validation with multiple errors."""
        config = get_default_config()
        # Create multiple errors
        config = config.with_(
            paths=config.paths.with_(data_dir=tmp_path / "nonexistent")
        )
        config = config.with_(
            active_learning=config.active_learning.with_(
                forced_choice_model=config.active_learning.forced_choice_model.with_(
                    batch_size=-1
                )
            )
        )
        config = config.with_(
            active_learning=config.active_learning.with_(
                trainer=config.active_learning.trainer.with_(epochs=0)
            )
        )
        errors = validate_config(config)
        # Should have multiple errors
        assert len(errors) >= 3

    def test_validate_config_aggregates_all_checks(self) -> None:
        """Test that validate_config runs all check functions."""
        config = get_default_config()
        errors = validate_config(config)
        # Should return a list (even if empty)
        assert isinstance(errors, list)
        # All check functions should have been called
        # (we can't verify this directly, but the function should at least run)
