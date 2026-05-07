"""Configuration models for mixed-effects active learning."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

__all__ = [
    "MixedEffectsConfig",
    "RandomEffectsSpec",
    "VarianceComponents",
]


class VarianceComponents(dx.Model):
    """Variance-covariance components for one random effect.

    Attributes
    ----------
    grouping_factor : str
        Name of grouping factor (e.g. ``"participant"``).
    effect_type : Literal["intercept", "slope"]
        Type of random effect.
    variance : float
        Estimated variance for this random effect (>= 0).
    n_groups : int
        Number of groups (>= 1).
    n_observations_per_group : dict[str, int]
        Number of observations per group.
    """

    grouping_factor: str
    effect_type: Literal["intercept", "slope"]
    variance: float
    n_groups: int
    n_observations_per_group: dict[str, int]

    __axioms__ = (
        dx.axiom("variance >= 0", message="variance must be non-negative"),
        dx.axiom("n_groups >= 1", message="n_groups must be >= 1"),
    )


class RandomEffectsSpec(dx.Model):
    """Specification of random effects structure.

    Attributes
    ----------
    grouping_factors : dict[str, Literal["intercept", "slope", "both"]]
        Mapping from grouping factor name to effect type.
    correlation_structure : Literal["independent", "correlated"]
        Whether intercept and slope are correlated when both are
        specified.
    """

    grouping_factors: dict[str, Literal["intercept", "slope", "both"]]
    correlation_structure: Literal["independent", "correlated"] = "independent"


class MixedEffectsConfig(dx.Model):
    """Configuration for mixed-effects modeling.

    Attributes
    ----------
    mode : Literal["fixed", "random_intercepts", "random_slopes"]
        Modeling mode.
    prior_mean : float
        Mean of Gaussian prior for random effects.
    prior_variance : float
        Variance of Gaussian prior (>= 0).
    estimate_variance_components : bool
        Whether to estimate the variance-covariance matrix.
    variance_estimation_method : Literal["mle", "reml"]
        Method for variance-component estimation.
    regularization_strength : float
        Strength of regularization toward the prior (>= 0).
    adaptive_regularization : bool
        Use stronger regularization for groups with fewer samples.
    min_samples_for_random_effects : int
        Minimum samples before estimating group-specific random effects (>= 1).
    random_effects_spec : RandomEffectsSpec | None
        Advanced specification for multiple grouping factors.
    """

    mode: Literal["fixed", "random_intercepts", "random_slopes"] = "fixed"
    prior_mean: float = 0.0
    prior_variance: float = 1.0
    estimate_variance_components: bool = True
    variance_estimation_method: Literal["mle", "reml"] = "mle"
    regularization_strength: float = 0.01
    adaptive_regularization: bool = True
    min_samples_for_random_effects: int = 5
    random_effects_spec: dx.Embed[RandomEffectsSpec] | None = None

    __axioms__ = (
        dx.axiom(
            "prior_variance >= 0",
            message="prior_variance must be non-negative",
        ),
        dx.axiom(
            "regularization_strength >= 0",
            message="regularization_strength must be non-negative",
        ),
        dx.axiom(
            "min_samples_for_random_effects >= 1",
            message="min_samples_for_random_effects must be >= 1",
        ),
    )
