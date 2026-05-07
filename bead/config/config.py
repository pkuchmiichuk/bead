"""Main configuration model for the bead package."""

from __future__ import annotations

from typing import Any

import didactic.api as dx

from bead.config.active_learning import ActiveLearningConfig
from bead.config.deployment import DeploymentConfig
from bead.config.item import ItemConfig
from bead.config.list import ListConfig
from bead.config.logging import LoggingConfig
from bead.config.paths import PathsConfig
from bead.config.protocol import ProtocolConfig
from bead.config.resources import ResourceConfig
from bead.config.template import TemplateConfig


def _default_paths() -> PathsConfig:
    return PathsConfig()


def _default_resources() -> ResourceConfig:
    return ResourceConfig()


def _default_templates() -> TemplateConfig:
    return TemplateConfig()


def _default_items() -> ItemConfig:
    return ItemConfig()


def _default_lists() -> ListConfig:
    return ListConfig()


def _default_deployment() -> DeploymentConfig:
    return DeploymentConfig()


def _default_active_learning() -> ActiveLearningConfig:
    return ActiveLearningConfig()


def _default_logging() -> LoggingConfig:
    return LoggingConfig()


def _default_protocol() -> ProtocolConfig:
    return ProtocolConfig()


class BeadConfig(dx.Model):
    """Main configuration for the bead package.

    Attributes
    ----------
    profile : str
        Configuration profile name.
    paths : PathsConfig
        Paths configuration.
    resources : ResourceConfig
        Resources configuration.
    templates : TemplateConfig
        Templates configuration.
    items : ItemConfig
        Items configuration.
    lists : ListConfig
        Lists configuration.
    deployment : DeploymentConfig
        Deployment configuration.
    active_learning : ActiveLearningConfig
        Active learning configuration.
    logging : LoggingConfig
        Logging configuration.
    protocol : ProtocolConfig
        Annotation-protocol configuration.
    """

    profile: str = "default"
    paths: dx.Embed[PathsConfig] = dx.field(default_factory=_default_paths)
    resources: dx.Embed[ResourceConfig] = dx.field(default_factory=_default_resources)
    templates: dx.Embed[TemplateConfig] = dx.field(default_factory=_default_templates)
    items: dx.Embed[ItemConfig] = dx.field(default_factory=_default_items)
    lists: dx.Embed[ListConfig] = dx.field(default_factory=_default_lists)
    deployment: dx.Embed[DeploymentConfig] = dx.field(
        default_factory=_default_deployment
    )
    active_learning: dx.Embed[ActiveLearningConfig] = dx.field(
        default_factory=_default_active_learning
    )
    logging: dx.Embed[LoggingConfig] = dx.field(default_factory=_default_logging)
    protocol: dx.Embed[ProtocolConfig] = dx.field(default_factory=_default_protocol)

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration as a plain ``dict``."""
        return self.model_dump()

    def to_yaml(self) -> str:
        """Render the configuration as a YAML string."""
        from bead.config.serialization import to_yaml  # noqa: PLC0415

        return to_yaml(self, include_defaults=False)

    def validate_paths(self) -> list[str]:
        """Return any path-related validation errors.

        Empty list means every required path either exists or is a
        relative path (in which case the caller is responsible for
        creating it).
        """
        errors: list[str] = []

        for label, path in (
            ("data_dir", self.paths.data_dir),
            ("output_dir", self.paths.output_dir),
            ("cache_dir", self.paths.cache_dir),
        ):
            if not path.exists() and path.is_absolute():
                errors.append(f"{label} does not exist: {path}")
        if self.paths.temp_dir is not None and not self.paths.temp_dir.exists():
            errors.append(f"temp_dir does not exist: {self.paths.temp_dir}")

        for label, path in (
            ("lexicon_path", self.resources.lexicon_path),
            ("templates_path", self.resources.templates_path),
            ("constraints_path", self.resources.constraints_path),
        ):
            if path is not None and not path.exists():
                errors.append(f"{label} does not exist: {path}")

        logging_dir = self.active_learning.trainer.logging_dir
        if not logging_dir.exists() and logging_dir.is_absolute():
            errors.append(f"logging_dir does not exist: {logging_dir}")

        if self.logging.file is not None and not self.logging.file.parent.exists():
            errors.append(
                f"logging file parent directory does not exist: "
                f"{self.logging.file.parent}"
            )

        return errors
