"""Tests for configuration profiles."""

from __future__ import annotations

import pytest

from bead.config.config import BeadConfig
from bead.config.profiles import (
    DEV_CONFIG,
    PROD_CONFIG,
    PROFILES,
    TEST_CONFIG,
    get_profile,
    list_profiles,
)


class TestDevConfig:
    """Tests for DEV_CONFIG profile."""

    def test_dev_config_is_valid(self) -> None:
        """Test DEV_CONFIG is a valid BeadConfig."""
        assert isinstance(DEV_CONFIG, BeadConfig)

    def test_dev_config_has_dev_profile(self) -> None:
        """Test DEV_CONFIG has 'dev' profile."""
        assert DEV_CONFIG.profile == "dev"

    def test_dev_has_debug_logging(self) -> None:
        """Test DEV_CONFIG has DEBUG logging level."""
        assert DEV_CONFIG.logging.level == "DEBUG"

    def test_dev_has_small_template_batch(self) -> None:
        """Test DEV_CONFIG has small template batch size."""
        assert DEV_CONFIG.templates.batch_size == 100

    def test_dev_has_small_model_batch(self) -> None:
        """Test DEV_CONFIG has small model batch size."""
        assert DEV_CONFIG.items.model.batch_size == 4

    def test_dev_has_cpu_device(self) -> None:
        """Test DEV_CONFIG uses CPU device."""
        assert DEV_CONFIG.items.model.device == "cpu"

    def test_dev_has_no_parallel_processing(self) -> None:
        """Test DEV_CONFIG disables parallel processing."""
        assert DEV_CONFIG.items.parallel_processing is False

    def test_dev_has_caching_disabled(self) -> None:
        """Test DEV_CONFIG disables external caching."""
        assert DEV_CONFIG.resources.cache_external is False

    def test_dev_has_console_logging(self) -> None:
        """Test DEV_CONFIG has console logging enabled."""
        assert DEV_CONFIG.logging.console is True

    def test_dev_has_minimal_training(self) -> None:
        """Test DEV_CONFIG has minimal training epochs."""
        assert DEV_CONFIG.active_learning.trainer.epochs == 1


class TestProdConfig:
    """Tests for PROD_CONFIG profile."""

    def test_prod_config_is_valid(self) -> None:
        """Test PROD_CONFIG is a valid BeadConfig."""
        assert isinstance(PROD_CONFIG, BeadConfig)

    def test_prod_config_has_prod_profile(self) -> None:
        """Test PROD_CONFIG has 'prod' profile."""
        assert PROD_CONFIG.profile == "prod"

    def test_prod_has_warning_logging(self) -> None:
        """Test PROD_CONFIG has WARNING logging level."""
        assert PROD_CONFIG.logging.level == "WARNING"

    def test_prod_has_large_template_batch(self) -> None:
        """Test PROD_CONFIG has large template batch size."""
        assert PROD_CONFIG.templates.batch_size == 10000

    def test_prod_has_large_model_batch(self) -> None:
        """Test PROD_CONFIG has large model batch size."""
        assert PROD_CONFIG.items.model.batch_size == 32

    def test_prod_has_cuda_device(self) -> None:
        """Test PROD_CONFIG uses CUDA device."""
        assert PROD_CONFIG.items.model.device == "cuda"

    def test_prod_has_parallel_processing(self) -> None:
        """Test PROD_CONFIG enables parallel processing."""
        assert PROD_CONFIG.items.parallel_processing is True

    def test_prod_has_caching_enabled(self) -> None:
        """Test PROD_CONFIG enables external caching."""
        assert PROD_CONFIG.resources.cache_external is True

    def test_prod_has_no_console_logging(self) -> None:
        """Test PROD_CONFIG disables console logging."""
        assert PROD_CONFIG.logging.console is False

    def test_prod_has_full_training(self) -> None:
        """Test PROD_CONFIG has full training epochs."""
        assert PROD_CONFIG.active_learning.trainer.epochs == 10

    def test_prod_has_wandb_enabled(self) -> None:
        """Test PROD_CONFIG enables W&B tracking."""
        assert PROD_CONFIG.active_learning.trainer.use_wandb is True

    def test_prod_has_stream_mode(self) -> None:
        """Test PROD_CONFIG enables stream mode."""
        assert PROD_CONFIG.templates.stream_mode is True

    def test_prod_has_multiple_workers(self) -> None:
        """Test PROD_CONFIG has multiple workers."""
        assert PROD_CONFIG.items.num_workers == 8


class TestTestConfig:
    """Tests for TEST_CONFIG profile."""

    def test_test_config_is_valid(self) -> None:
        """Test TEST_CONFIG is a valid BeadConfig."""
        assert isinstance(TEST_CONFIG, BeadConfig)

    def test_test_config_has_test_profile(self) -> None:
        """Test TEST_CONFIG has 'test' profile."""
        assert TEST_CONFIG.profile == "test"

    def test_test_has_critical_logging(self) -> None:
        """Test TEST_CONFIG has CRITICAL logging level."""
        assert TEST_CONFIG.logging.level == "CRITICAL"

    def test_test_has_tiny_template_batch(self) -> None:
        """Test TEST_CONFIG has tiny template batch size."""
        assert TEST_CONFIG.templates.batch_size == 10

    def test_test_has_tiny_model_batch(self) -> None:
        """Test TEST_CONFIG has tiny model batch size."""
        assert TEST_CONFIG.items.model.batch_size == 1

    def test_test_has_cpu_device(self) -> None:
        """Test TEST_CONFIG uses CPU device."""
        assert TEST_CONFIG.items.model.device == "cpu"

    def test_test_has_no_parallel_processing(self) -> None:
        """Test TEST_CONFIG disables parallel processing."""
        assert TEST_CONFIG.items.parallel_processing is False

    def test_test_has_caching_disabled(self) -> None:
        """Test TEST_CONFIG disables external caching."""
        assert TEST_CONFIG.resources.cache_external is False

    def test_test_has_no_console_logging(self) -> None:
        """Test TEST_CONFIG disables console logging."""
        assert TEST_CONFIG.logging.console is False

    def test_test_has_random_seed(self) -> None:
        """Test TEST_CONFIG has fixed random seed."""
        assert TEST_CONFIG.templates.random_seed == 42
        assert TEST_CONFIG.lists.random_seed == 42

    def test_test_has_max_combinations(self) -> None:
        """Test TEST_CONFIG limits max combinations."""
        assert TEST_CONFIG.templates.max_combinations == 100

    def test_test_has_minimal_training(self) -> None:
        """Test TEST_CONFIG has minimal training."""
        assert TEST_CONFIG.active_learning.trainer.epochs == 1
        assert TEST_CONFIG.active_learning.forced_choice_model.batch_size == 2

    def test_test_has_no_wandb(self) -> None:
        """Test TEST_CONFIG disables W&B tracking."""
        assert TEST_CONFIG.active_learning.trainer.use_wandb is False


class TestProfilesRegistry:
    """Tests for PROFILES registry."""

    def test_profiles_contains_default(self) -> None:
        """Test PROFILES registry contains 'default'."""
        assert "default" in PROFILES

    def test_profiles_contains_dev(self) -> None:
        """Test PROFILES registry contains 'dev'."""
        assert "dev" in PROFILES

    def test_profiles_contains_prod(self) -> None:
        """Test PROFILES registry contains 'prod'."""
        assert "prod" in PROFILES

    def test_profiles_contains_test(self) -> None:
        """Test PROFILES registry contains 'test'."""
        assert "test" in PROFILES

    def test_profiles_default_is_valid(self) -> None:
        """Test default profile in registry is valid."""
        assert isinstance(PROFILES["default"], BeadConfig)
        assert PROFILES["default"].profile == "default"

    def test_profiles_dev_is_dev_config(self) -> None:
        """Test dev profile matches DEV_CONFIG."""
        assert PROFILES["dev"].profile == "dev"
        assert PROFILES["dev"].logging.level == "DEBUG"

    def test_profiles_prod_is_prod_config(self) -> None:
        """Test prod profile matches PROD_CONFIG."""
        assert PROFILES["prod"].profile == "prod"
        assert PROFILES["prod"].logging.level == "WARNING"

    def test_profiles_test_is_test_config(self) -> None:
        """Test test profile matches TEST_CONFIG."""
        assert PROFILES["test"].profile == "test"
        assert PROFILES["test"].logging.level == "CRITICAL"


class TestGetProfile:
    """Tests for get_profile function."""

    def test_get_profile_with_default(self) -> None:
        """Test get_profile with 'default' name."""
        config = get_profile("default")
        assert isinstance(config, BeadConfig)
        assert config.profile == "default"

    def test_get_profile_with_dev(self) -> None:
        """Test get_profile with 'dev' name."""
        config = get_profile("dev")
        assert isinstance(config, BeadConfig)
        assert config.profile == "dev"
        assert config.logging.level == "DEBUG"

    def test_get_profile_with_prod(self) -> None:
        """Test get_profile with 'prod' name."""
        config = get_profile("prod")
        assert isinstance(config, BeadConfig)
        assert config.profile == "prod"
        assert config.logging.level == "WARNING"

    def test_get_profile_with_test(self) -> None:
        """Test get_profile with 'test' name."""
        config = get_profile("test")
        assert isinstance(config, BeadConfig)
        assert config.profile == "test"
        assert config.logging.level == "CRITICAL"

    def test_get_profile_with_invalid_name(self) -> None:
        """Test get_profile raises ValueError for invalid name."""
        with pytest.raises(ValueError) as exc_info:
            get_profile("invalid")
        assert "Profile 'invalid' not found" in str(exc_info.value)
        assert "Available profiles:" in str(exc_info.value)

    def test_get_profile_returns_copy(self) -> None:
        """Test get_profile returns a copy, not the original."""
        config1 = get_profile("dev")
        config2 = get_profile("dev")
        assert config1 is not config2

    def test_get_profile_modifications_dont_affect_original(self) -> None:
        """Test modifications to returned config don't affect PROFILES."""
        config = get_profile("dev")
        config = config.with_(
            profile="modified",
            templates=config.templates.with_(batch_size=9999),
        )

        # Check original is unchanged
        assert PROFILES["dev"].profile == "dev"
        assert PROFILES["dev"].templates.batch_size == 100


class TestListProfiles:
    """Tests for list_profiles function."""

    def test_list_profiles_returns_list(self) -> None:
        """Test list_profiles returns a list."""
        profiles = list_profiles()
        assert isinstance(profiles, list)

    def test_list_profiles_contains_all_profiles(self) -> None:
        """Test list_profiles contains all profile names."""
        profiles = list_profiles()
        assert "default" in profiles
        assert "dev" in profiles
        assert "prod" in profiles
        assert "test" in profiles

    def test_list_profiles_is_sorted(self) -> None:
        """Test list_profiles returns sorted list."""
        profiles = list_profiles()
        assert profiles == sorted(profiles)

    def test_list_profiles_count(self) -> None:
        """Test list_profiles returns correct number of profiles."""
        profiles = list_profiles()
        assert len(profiles) == 4


class TestProfilesIndependence:
    """Tests for profile independence."""

    def test_profiles_are_independent(self) -> None:
        """Test modifying one profile doesn't affect others."""
        default = get_profile("default")
        dev = get_profile("dev")
        prod = get_profile("prod")
        test = get_profile("test")

        dev = dev.with_(
            profile="modified",
            templates=dev.templates.with_(batch_size=9999),
        )

        assert default.profile == "default"
        assert prod.profile == "prod"
        assert test.profile == "test"
        assert default.templates.batch_size == 1000
        assert prod.templates.batch_size == 10000
        assert test.templates.batch_size == 10

    def test_nested_configs_are_independent(self) -> None:
        """Test nested configs in different profiles are independent."""
        dev = get_profile("dev")
        prod = get_profile("prod")

        dev = dev.with_(
            items=dev.items.with_(model=dev.items.model.with_(batch_size=9999))
        )

        assert prod.items.model.batch_size == 32
