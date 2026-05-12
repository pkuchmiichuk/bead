"""Configuration system for the bead pipeline.

Provides configuration models, default settings, and named profiles for
development, testing, and production environments.
"""

from __future__ import annotations

from bead.config.active_learning import ActiveLearningConfig
from bead.config.compose import (
    ComposeValue,
    ConfigError,
    InterpolationError,
    compose,
    register_resolver,
)
from bead.config.config import BeadConfig
from bead.config.defaults import DEFAULT_CONFIG, get_default_config
from bead.config.deployment import DeploymentConfig
from bead.config.env import load_from_env
from bead.config.item import ItemConfig
from bead.config.list import ListConfig
from bead.config.loader import load_config
from bead.config.logging import LoggingConfig
from bead.config.model import ModelConfig
from bead.config.paths import PathsConfig
from bead.config.profiles import (
    DEV_CONFIG,
    PROD_CONFIG,
    PROFILES,
    TEST_CONFIG,
    get_profile,
    list_profiles,
)
from bead.config.protocol import (
    AnchorSpec,
    DriftConfig,
    FamilySpec,
    ProtocolConfig,
    RealizationKind,
    TemplateVariantSpec,
)
from bead.config.resources import ResourceConfig
from bead.config.serialization import save_yaml, to_yaml
from bead.config.template import SlotStrategyConfig, TemplateConfig
from bead.config.validation import validate_config

__all__ = [
    # main config
    "BeadConfig",
    # config sections
    "PathsConfig",
    "ResourceConfig",
    "SlotStrategyConfig",
    "TemplateConfig",
    "ModelConfig",
    "ItemConfig",
    "ListConfig",
    "DeploymentConfig",
    "ActiveLearningConfig",
    "LoggingConfig",
    "ProtocolConfig",
    # protocol sub-specs
    "AnchorSpec",
    "TemplateVariantSpec",
    "FamilySpec",
    "DriftConfig",
    "RealizationKind",
    # defaults
    "DEFAULT_CONFIG",
    "get_default_config",
    # profiles
    "DEV_CONFIG",
    "PROD_CONFIG",
    "TEST_CONFIG",
    "PROFILES",
    "get_profile",
    "list_profiles",
    # loading + composition
    "ComposeValue",
    "ConfigError",
    "InterpolationError",
    "compose",
    "load_config",
    "register_resolver",
    # environment
    "load_from_env",
    # validation
    "validate_config",
    # serialization
    "to_yaml",
    "save_yaml",
]
