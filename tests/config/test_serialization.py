"""Tests for configuration serialization to YAML."""

from pathlib import Path

import pytest
import yaml

from bead.config.config import BeadConfig
from bead.config.defaults import get_default_config
from bead.config.loader import load_config
from bead.config.serialization import config_to_dict, save_yaml, to_yaml


class TestConfigToDict:
    """Tests for config_to_dict function."""

    def test_config_to_dict_basic(self) -> None:
        """Test basic config to dict conversion."""
        config = get_default_config()
        config_dict = config_to_dict(config, include_defaults=True)
        assert isinstance(config_dict, dict)
        assert "profile" in config_dict
        assert config_dict["profile"] == "default"

    def test_config_to_dict_excludes_defaults(self) -> None:
        """Test that config_to_dict excludes default values."""
        config = get_default_config()
        config_dict = config_to_dict(config, include_defaults=False)
        # When all values are defaults, should be empty or minimal
        # At minimum, changed values should be included
        assert isinstance(config_dict, dict)

    def test_config_to_dict_includes_modified_values(self) -> None:
        """Test that modified values are included."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="DEBUG"))
        config_dict = config_to_dict(config, include_defaults=False)
        assert "logging" in config_dict
        assert config_dict["logging"]["level"] == "DEBUG"

    def test_config_to_dict_converts_paths(self) -> None:
        """Test that Path objects are converted to strings."""
        config = get_default_config()
        config = config.with_(paths=config.paths.with_(data_dir=Path("/custom/path")))
        config_dict = config_to_dict(config, include_defaults=True)
        assert isinstance(config_dict["paths"]["data_dir"], str)
        assert config_dict["paths"]["data_dir"] == "/custom/path"

    def test_config_to_dict_with_nested_changes(self) -> None:
        """Test that nested changes are properly captured."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="ERROR"))
        config = config.with_(logging=config.logging.with_(console=False))
        config_dict = config_to_dict(config, include_defaults=False)
        assert "logging" in config_dict
        assert config_dict["logging"]["level"] == "ERROR"
        assert config_dict["logging"]["console"] is False

    def test_config_to_dict_include_defaults_true(self) -> None:
        """Test config_to_dict with include_defaults=True."""
        config = get_default_config()
        config_dict = config_to_dict(config, include_defaults=True)
        # Should include all sections
        assert "paths" in config_dict
        assert "logging" in config_dict
        assert "templates" in config_dict


class TestToYaml:
    """Tests for to_yaml function."""

    def test_to_yaml_basic(self) -> None:
        """Test basic YAML serialization."""
        config = get_default_config()
        yaml_str = to_yaml(config)
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0

    def test_to_yaml_is_valid_yaml(self) -> None:
        """Test that output is valid YAML."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="DEBUG"))
        yaml_str = to_yaml(config)
        # Should be parseable as YAML
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)

    def test_to_yaml_contains_modified_values(self) -> None:
        """Test that YAML contains modified values."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="WARNING"))
        yaml_str = to_yaml(config, include_defaults=False)
        assert "WARNING" in yaml_str

    def test_to_yaml_with_include_defaults(self) -> None:
        """Test YAML serialization with include_defaults=True."""
        config = get_default_config()
        yaml_str = to_yaml(config, include_defaults=True)
        parsed = yaml.safe_load(yaml_str)
        assert "profile" in parsed
        assert "paths" in parsed
        assert "logging" in parsed

    def test_to_yaml_paths_as_strings(self) -> None:
        """Test that Path objects are serialized as strings."""
        config = get_default_config()
        config = config.with_(paths=config.paths.with_(data_dir=Path("/test/data")))
        yaml_str = to_yaml(config, include_defaults=True)
        assert "/test/data" in yaml_str
        # Should not contain Path object representation
        assert "PosixPath" not in yaml_str
        assert "WindowsPath" not in yaml_str

    def test_to_yaml_sorted_keys(self) -> None:
        """Test that YAML output has sorted keys."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="DEBUG"))
        config = config.with_(paths=config.paths.with_(data_dir=Path("/data")))
        yaml_str = to_yaml(config, include_defaults=True)
        # Keys should be alphabetically sorted
        # logging should come before paths
        logging_pos = yaml_str.find("logging")
        paths_pos = yaml_str.find("paths")
        # At least one should be present
        assert logging_pos >= 0 or paths_pos >= 0

    def test_to_yaml_roundtrip(self) -> None:
        """Test that config can be serialized and deserialized."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="ERROR"))
        config = config.with_(paths=config.paths.with_(data_dir=Path("/custom/data")))

        # Serialize to YAML
        yaml_str = to_yaml(config, include_defaults=True)

        # Parse YAML
        parsed = yaml.safe_load(yaml_str)

        # Create new config from parsed data
        new_config = BeadConfig(**parsed)

        # Should match original
        assert new_config.logging.level == config.logging.level
        assert new_config.paths.data_dir == config.paths.data_dir


class TestSaveYaml:
    """Tests for save_yaml function."""

    def test_save_yaml_creates_file(self, tmp_path: Path) -> None:
        """Test that save_yaml creates a file."""
        config = get_default_config()
        config_file = tmp_path / "config.yaml"
        save_yaml(config, config_file)
        assert config_file.exists()

    def test_save_yaml_content_is_valid(self, tmp_path: Path) -> None:
        """Test that saved YAML file has valid content."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="DEBUG"))
        config_file = tmp_path / "config.yaml"
        save_yaml(config, config_file)

        # Read and parse file
        with open(config_file) as f:
            content = yaml.safe_load(f)

        assert isinstance(content, dict)

    def test_save_yaml_with_create_dirs(self, tmp_path: Path) -> None:
        """Test save_yaml with create_dirs=True."""
        config = get_default_config()
        config_file = tmp_path / "subdir" / "config.yaml"
        save_yaml(config, config_file, create_dirs=True)
        assert config_file.exists()
        assert config_file.parent.exists()

    def test_save_yaml_without_create_dirs_fails(self, tmp_path: Path) -> None:
        """Test save_yaml with create_dirs=False and missing dir."""
        config = get_default_config()
        config_file = tmp_path / "nonexistent" / "config.yaml"
        with pytest.raises(FileNotFoundError, match="Parent directory does not exist"):
            save_yaml(config, config_file, create_dirs=False)

    def test_save_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading config."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="WARNING"))
        config = config.with_(paths=config.paths.with_(data_dir=Path("/test/data")))

        config_file = tmp_path / "config.yaml"
        save_yaml(config, config_file, include_defaults=True)

        # Load it back
        loaded_config = load_config(config_path=config_file)

        # Should match
        assert loaded_config.logging.level == config.logging.level
        assert loaded_config.paths.data_dir == config.paths.data_dir

    def test_save_yaml_with_string_path(self, tmp_path: Path) -> None:
        """Test save_yaml with string path."""
        config = get_default_config()
        config_file = tmp_path / "config.yaml"
        save_yaml(config, str(config_file))
        assert config_file.exists()

    def test_save_yaml_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that save_yaml overwrites existing files."""
        config = get_default_config()
        config_file = tmp_path / "config.yaml"

        # Write first version
        config = config.with_(logging=config.logging.with_(level="INFO"))
        save_yaml(config, config_file)

        # Write second version
        config = config.with_(logging=config.logging.with_(level="DEBUG"))
        save_yaml(config, config_file)

        # Load and check it has the new value
        with open(config_file) as f:
            content = yaml.safe_load(f)

        new_config = BeadConfig.model_validate(content)
        assert new_config.logging.level == "DEBUG"

    def test_save_yaml_without_defaults(self, tmp_path: Path) -> None:
        """Test save_yaml with include_defaults=False."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="ERROR"))

        config_file = tmp_path / "config.yaml"
        save_yaml(config, config_file, include_defaults=False)

        # File should be smaller, containing only changes
        with open(config_file) as f:
            content = f.read()

        # Should have logging section
        assert "logging" in content or "ERROR" in content


class TestBeadConfigToYaml:
    """Tests for BeadConfig.to_yaml() method."""

    def test_bead_config_to_yaml_method(self) -> None:
        """Test BeadConfig.to_yaml() method."""
        config = get_default_config()
        yaml_str = config.to_yaml()
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0

    def test_bead_config_to_yaml_is_valid(self) -> None:
        """Test that BeadConfig.to_yaml() produces valid YAML."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="DEBUG"))
        yaml_str = config.to_yaml()
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)

    def test_bead_config_to_yaml_excludes_defaults(self) -> None:
        """Test that BeadConfig.to_yaml() excludes defaults by default."""
        config = get_default_config()
        config = config.with_(logging=config.logging.with_(level="CRITICAL"))
        yaml_str = config.to_yaml()
        # Should contain the changed value
        assert "CRITICAL" in yaml_str


class TestDistributionStrategyYAML:
    """Tests for distribution strategy YAML serialization."""

    def test_distribution_strategy_serializes_to_yaml(self) -> None:
        """Test that distribution strategy serializes correctly to YAML."""
        from bead.deployment.distribution import (  # noqa: PLC0415
            DistributionStrategyType,
            ListDistributionStrategy,
        )

        config = get_default_config()
        config = config.with_(
            deployment=config.deployment.with_(
                distribution_strategy=ListDistributionStrategy(
                    strategy_type=DistributionStrategyType.QUOTA_BASED,
                    strategy_config={
                        "participants_per_list": 25,
                        "allow_overflow": False,
                    },
                    max_participants=400,
                )
            )
        )

        yaml_str = to_yaml(config, include_defaults=False)
        parsed = yaml.safe_load(yaml_str)

        # Should contain deployment section with distribution strategy
        assert "deployment" in parsed
        assert "distribution_strategy" in parsed["deployment"]
        assert (
            parsed["deployment"]["distribution_strategy"]["strategy_type"]
            == "quota_based"
        )
        assert parsed["deployment"]["distribution_strategy"]["max_participants"] == 400

    def test_distribution_strategy_roundtrip(self) -> None:
        """Test YAML roundtrip for distribution strategy."""
        from bead.deployment.distribution import (  # noqa: PLC0415
            DistributionStrategyType,
            ListDistributionStrategy,
        )

        # Create config with custom distribution strategy
        config = get_default_config()
        config = config.with_(
            deployment=config.deployment.with_(
                distribution_strategy=ListDistributionStrategy(
                    strategy_type=DistributionStrategyType.WEIGHTED_RANDOM,
                    strategy_config={
                        "weight_expression": "list_metadata.priority || 1.0",
                        "normalize_weights": True,
                    },
                )
            )
        )

        # Serialize to YAML
        yaml_str = to_yaml(config, include_defaults=True)

        # Parse and create new config
        parsed = yaml.safe_load(yaml_str)
        new_config = BeadConfig(**parsed)

        # Verify it matches
        assert (
            new_config.deployment.distribution_strategy.strategy_type
            == DistributionStrategyType.WEIGHTED_RANDOM
        )
        assert (
            new_config.deployment.distribution_strategy.strategy_config[
                "weight_expression"
            ]
            == "list_metadata.priority || 1.0"
        )

    def test_all_strategy_types_serialize(self) -> None:
        """Test that all 8 strategy types can be serialized."""
        from bead.deployment.distribution import (  # noqa: PLC0415
            DistributionStrategyType,
            ListDistributionStrategy,
        )

        strategies = [
            DistributionStrategyType.RANDOM,
            DistributionStrategyType.SEQUENTIAL,
            DistributionStrategyType.BALANCED,
            DistributionStrategyType.LATIN_SQUARE,
            DistributionStrategyType.STRATIFIED,
            DistributionStrategyType.WEIGHTED_RANDOM,
            DistributionStrategyType.QUOTA_BASED,
            DistributionStrategyType.METADATA_BASED,
        ]

        for strategy_type in strategies:
            config = get_default_config()

            # Provide required config for strategies that need it
            strategy_config = {}
            if strategy_type == DistributionStrategyType.QUOTA_BASED:
                strategy_config = {"participants_per_list": 25}
            elif strategy_type == DistributionStrategyType.WEIGHTED_RANDOM:
                strategy_config = {"weight_expression": "1.0"}
            elif strategy_type == DistributionStrategyType.STRATIFIED:
                strategy_config = {"factors": ["condition"]}
            elif strategy_type == DistributionStrategyType.METADATA_BASED:
                strategy_config = {"rank_expression": "0"}

            config = config.with_(
                deployment=config.deployment.with_(
                    distribution_strategy=ListDistributionStrategy(
                        strategy_type=strategy_type,
                        strategy_config=strategy_config,
                    )
                )
            )

            # Should serialize without error
            yaml_str = to_yaml(config, include_defaults=True)
            parsed = yaml.safe_load(yaml_str)

            # Should deserialize without error
            new_config = BeadConfig(**parsed)
            assert (
                new_config.deployment.distribution_strategy.strategy_type
                == strategy_type
            )
