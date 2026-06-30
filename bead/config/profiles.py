"""Configuration profiles for the bead package.

Pre-configured profiles for different environments (development,
production, testing) with optimized settings for each use case.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

from bead.config.active_learning import (
    ActiveLearningConfig,
    ForcedChoiceModelConfig,
    TrainerConfig,
)
from bead.config.config import BeadConfig
from bead.config.deployment import DeploymentConfig
from bead.config.item import ItemConfig
from bead.config.list import ListConfig
from bead.config.logging import LoggingConfig
from bead.config.model import ModelConfig
from bead.config.paths import PathsConfig
from bead.config.resources import ResourceConfig
from bead.config.template import TemplateConfig

DEV_CONFIG = BeadConfig(
    profile="dev",
    paths=PathsConfig(
        data_dir=Path("data"),
        output_dir=Path("output"),
        cache_dir=Path(".cache"),
        temp_dir=Path(gettempdir()) / "bead_dev",
        create_dirs=True,
    ),
    resources=ResourceConfig(
        cache_external=False,
    ),
    templates=TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=100,
        stream_mode=False,
    ),
    items=ItemConfig(
        model=ModelConfig(
            provider="huggingface",
            model_name="gpt2",
            batch_size=4,
            device="cpu",
        ),
        parallel_processing=False,
    ),
    lists=ListConfig(
        num_lists=1,
    ),
    deployment=DeploymentConfig(),
    active_learning=ActiveLearningConfig(
        forced_choice_model=ForcedChoiceModelConfig(
            num_epochs=1,
            batch_size=8,
            learning_rate=2e-5,
        ),
        trainer=TrainerConfig(epochs=1),
    ),
    logging=LoggingConfig(
        level="DEBUG",
        console=True,
    ),
)
"""Development configuration profile."""


PROD_CONFIG = BeadConfig(
    profile="prod",
    paths=PathsConfig(
        data_dir=Path("/var/bead/data").absolute(),
        output_dir=Path("/var/bead/output").absolute(),
        cache_dir=Path("/var/bead/cache").absolute(),
        temp_dir=Path("/var/bead/temp").absolute(),
        create_dirs=True,
    ),
    resources=ResourceConfig(
        cache_external=True,
    ),
    templates=TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=10000,
        stream_mode=True,
    ),
    items=ItemConfig(
        model=ModelConfig(
            provider="huggingface",
            model_name="gpt2",
            batch_size=32,
            device="cuda",
        ),
        parallel_processing=True,
        num_workers=8,
    ),
    lists=ListConfig(
        num_lists=1,
    ),
    deployment=DeploymentConfig(
        apply_material_design=True,
        include_demographics=True,
        include_attention_checks=True,
    ),
    active_learning=ActiveLearningConfig(
        forced_choice_model=ForcedChoiceModelConfig(
            num_epochs=10,
            batch_size=32,
            learning_rate=2e-5,
        ),
        trainer=TrainerConfig(epochs=10, use_wandb=True),
    ),
    logging=LoggingConfig(
        level="WARNING",
        console=False,
        file=Path("/var/log/bead/app.log"),
    ),
)
"""Production configuration profile."""


TEST_CONFIG = BeadConfig(
    profile="test",
    paths=PathsConfig(
        data_dir=Path(gettempdir()) / "bead_test" / "data",
        output_dir=Path(gettempdir()) / "bead_test" / "output",
        cache_dir=Path(gettempdir()) / "bead_test" / "cache",
        temp_dir=Path(gettempdir()) / "bead_test" / "temp",
        create_dirs=True,
    ),
    resources=ResourceConfig(
        cache_external=False,
    ),
    templates=TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=10,
        max_combinations=100,
        random_seed=42,
    ),
    items=ItemConfig(
        model=ModelConfig(
            provider="huggingface",
            model_name="gpt2",
            batch_size=1,
            device="cpu",
        ),
        parallel_processing=False,
        num_workers=1,
    ),
    lists=ListConfig(
        num_lists=1,
        random_seed=42,
    ),
    deployment=DeploymentConfig(
        apply_material_design=False,
        include_demographics=False,
        include_attention_checks=False,
    ),
    active_learning=ActiveLearningConfig(
        forced_choice_model=ForcedChoiceModelConfig(
            num_epochs=1,
            batch_size=2,
            learning_rate=2e-5,
        ),
        trainer=TrainerConfig(epochs=1, use_wandb=False),
    ),
    logging=LoggingConfig(
        level="CRITICAL",
        console=False,
    ),
)
"""Test configuration profile."""


PROFILES: dict[str, BeadConfig] = {
    "default": BeadConfig(),
    "dev": DEV_CONFIG,
    "prod": PROD_CONFIG,
    "test": TEST_CONFIG,
}
"""Registry of all available configuration profiles."""


def get_profile(name: str) -> BeadConfig:
    """Return the configuration profile registered under *name*.

    didactic Models are frozen, so the returned instance is shared with
    the registry; callers can pass it to ``with_(...)`` to derive
    overrides without affecting other consumers.

    Raises
    ------
    ValueError
        If *name* is not a registered profile.
    """
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise ValueError(f"Profile {name!r} not found. Available profiles: {available}")
    return PROFILES[name].with_()


def list_profiles() -> list[str]:
    """Return the registered profile names, sorted."""
    return sorted(PROFILES.keys())
