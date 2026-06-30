"""Deployment configuration."""

from __future__ import annotations

from pathlib import Path

import didactic.api as dx

from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)


class SlopitKeystrokeConfig(dx.Model):
    """Slopit keystroke capture options.

    Attributes
    ----------
    enabled : bool
        Capture keystroke events.
    capture_key_up : bool
        Also capture keyup events.
    include_modifiers : bool
        Record modifier-key states.
    """

    enabled: bool = True
    capture_key_up: bool = True
    include_modifiers: bool = True


class SlopitFocusConfig(dx.Model):
    """Slopit focus/blur capture options.

    Attributes
    ----------
    enabled : bool
        Capture focus events.
    use_visibility_api : bool
        Use the Page Visibility API.
    use_blur_focus : bool
        Track blur and focus events.
    """

    enabled: bool = True
    use_visibility_api: bool = True
    use_blur_focus: bool = True


class SlopitPasteConfig(dx.Model):
    """Slopit paste-event capture options.

    Attributes
    ----------
    enabled : bool
        Capture paste events.
    prevent : bool
        Block paste actions.
    capture_preview : bool
        Capture a preview of pasted text.
    preview_length : int
        Number of characters to include in the preview (>= 0).
    """

    enabled: bool = True
    prevent: bool = False
    capture_preview: bool = True
    preview_length: int = 100


def _default_keystroke_config() -> SlopitKeystrokeConfig:
    return SlopitKeystrokeConfig()


def _default_focus_config() -> SlopitFocusConfig:
    return SlopitFocusConfig()


def _default_paste_config() -> SlopitPasteConfig:
    return SlopitPasteConfig()


def _default_target_selectors() -> dict[str, str]:
    return {
        "likert_rating": ".bead-rating-button",
        "slider_rating": ".bead-slider",
        "forced_choice": ".bead-choice-button",
        "cloze": ".bead-cloze-field",
    }


class SlopitIntegrationConfig(dx.Model):
    """Configuration for slopit behavioral capture integration.

    Attributes
    ----------
    enabled : bool
        Enable slopit behavioral capture.
    keystroke : SlopitKeystrokeConfig
        Keystroke capture settings.
    focus : SlopitFocusConfig
        Focus/blur capture settings.
    paste : SlopitPasteConfig
        Paste event capture settings.
    target_selectors : dict[str, str]
        CSS selectors for capture targets by task type.
    """

    enabled: bool = False
    keystroke: dx.Embed[SlopitKeystrokeConfig] = dx.field(
        default_factory=_default_keystroke_config
    )
    focus: dx.Embed[SlopitFocusConfig] = dx.field(default_factory=_default_focus_config)
    paste: dx.Embed[SlopitPasteConfig] = dx.field(default_factory=_default_paste_config)
    target_selectors: dict[str, str] = dx.field(
        default_factory=_default_target_selectors
    )


def validate_slopit_integration(config: SlopitIntegrationConfig) -> None:
    """Raise ``ValueError`` if slopit is enabled but the compiled bundle is missing."""
    if not config.enabled:
        return
    bundle_path = (
        Path(__file__).parent.parent
        / "deployment"
        / "jspsych"
        / "dist"
        / "slopit-bundle.js"
    )
    if not bundle_path.exists():
        raise ValueError(
            f"Slopit bundle not found at {bundle_path}. "
            "Run 'pnpm build' in bead/deployment/jspsych to compile TypeScript."
        )


def _default_distribution_strategy() -> ListDistributionStrategy:
    return ListDistributionStrategy(strategy_type=DistributionStrategyType.BALANCED)


class DeploymentConfig(dx.Model):
    """Configuration for experiment deployment.

    Attributes
    ----------
    platform : str
        Deployment platform.
    jspsych_version : str
        jsPsych version.
    apply_material_design : bool
        Use Material Design styling.
    include_demographics : bool
        Include a demographics survey.
    include_attention_checks : bool
        Include attention checks.
    jatos_export : bool
        Export to JATOS.
    distribution_strategy : ListDistributionStrategy
        List distribution strategy for batch experiments.
    """

    platform: str = "jspsych"
    jspsych_version: str | None = "7.3.0"
    apply_material_design: bool = True
    include_demographics: bool = True
    include_attention_checks: bool = True
    jatos_export: bool = False
    distribution_strategy: dx.Embed[ListDistributionStrategy] = dx.field(
        default_factory=_default_distribution_strategy
    )
