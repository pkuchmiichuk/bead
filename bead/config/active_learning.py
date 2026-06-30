"""Active learning configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import didactic.api as dx

from bead.active_learning.config import MixedEffectsConfig
from bead.data.range import Range


def _default_mixed_effects() -> MixedEffectsConfig:
    return MixedEffectsConfig()


class BaseEncoderModelConfig(dx.Model):
    """Base configuration for encoder-based active learning models.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length (must be > 0).
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for input processing.
    include_instructions : bool
        Whether to include task instructions.
    learning_rate : float
        Learning rate for AdamW (must be > 0).
    batch_size : int
        Batch size for training (must be > 0).
    num_epochs : int
        Number of training epochs (must be > 0).
    device : Literal["cpu", "cuda", "mps"]
        Device to train on.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.
    """

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    encoder_mode: Literal["single_encoder", "dual_encoder"] = "single_encoder"
    include_instructions: bool = False
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    mixed_effects: dx.Embed[MixedEffectsConfig] = dx.field(
        default_factory=_default_mixed_effects
    )


class ForcedChoiceModelConfig(BaseEncoderModelConfig):
    """Forced-choice active-learning model configuration."""


class CategoricalModelConfig(BaseEncoderModelConfig):
    """Categorical active-learning model configuration."""


class BinaryModelConfig(BaseEncoderModelConfig):
    """Binary active-learning model configuration."""


class MultiSelectModelConfig(BaseEncoderModelConfig):
    """Multi-select active-learning model configuration."""


class UncertaintySamplerConfig(dx.Model):
    """Configuration for uncertainty sampling.

    Attributes
    ----------
    method : Literal["entropy", "margin", "least_confidence"]
        Uncertainty method.
    batch_size : int | None
        Items to select per iteration; ``None`` defers to the loop's
        ``budget_per_iteration``.
    """

    method: Literal["entropy", "margin", "least_confidence"] = "entropy"
    batch_size: int | None = None


class JatosDataCollectionConfig(dx.Model):
    """Configuration for JATOS data collection.

    Attributes
    ----------
    base_url : str
        JATOS base URL.
    api_token : str
        JATOS API token.
    study_id : int
        JATOS study identifier.
    """

    base_url: str
    api_token: str
    study_id: int


class ProlificDataCollectionConfig(dx.Model):
    """Configuration for Prolific data collection.

    Attributes
    ----------
    api_key : str
        Prolific API key.
    study_id : str
        Prolific study identifier.
    """

    api_key: str
    study_id: str


class ActiveLearningLoopConfig(dx.Model):
    """Configuration for the active-learning loop.

    Attributes
    ----------
    max_iterations : int
        Maximum number of iterations (> 0).
    budget_per_iteration : int
        Items selected per iteration (> 0).
    stopping_criterion : Literal["max_iterations", "convergence", \
"performance_threshold"]
        Stopping criterion.
    performance_threshold : float | None
        Performance threshold for stopping (0.0-1.0).
    metric_name : str
        Metric name for convergence / threshold checks.
    convergence_patience : int
        Iterations to wait before declaring convergence (> 0).
    convergence_threshold : float
        Minimum improvement to avoid convergence (> 0).
    jatos : JatosDataCollectionConfig | None
        JATOS data-collection configuration.
    prolific : ProlificDataCollectionConfig | None
        Prolific data-collection configuration.
    data_collection_timeout : int
        Timeout in seconds for data collection (> 0).
    """

    max_iterations: int = 10
    budget_per_iteration: int = 100
    stopping_criterion: Literal[
        "max_iterations", "convergence", "performance_threshold"
    ] = "max_iterations"
    performance_threshold: float | None = None
    metric_name: str = "accuracy"
    convergence_patience: int = 3
    convergence_threshold: float = 0.01
    jatos: dx.Embed[JatosDataCollectionConfig] | None = None
    prolific: dx.Embed[ProlificDataCollectionConfig] | None = None
    data_collection_timeout: int = 3600


class TrainerConfig(dx.Model):
    """Configuration for active-learning trainers.

    Attributes
    ----------
    trainer_type : Literal["huggingface", "lightning"]
        Trainer type.
    epochs : int
        Number of training epochs (> 0).
    eval_strategy : str
        Evaluation strategy.
    save_strategy : str
        Save strategy.
    logging_dir : Path
        Logging directory.
    use_wandb : bool
        Use Weights & Biases.
    wandb_project : str | None
        W&B project name.
    """

    trainer_type: Literal["huggingface", "lightning"] = "huggingface"
    epochs: int = 3
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_dir: Path = dx.field(default_factory=lambda: Path("logs"))
    use_wandb: bool = False
    wandb_project: str | None = None


def _default_scale() -> Range[float]:
    return Range[float](min=0.0, max=1.0)


class OrdinalScaleModelConfig(dx.Model):
    """Configuration for ordinal-scale active-learning models.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length (> 0).
    encoder_mode : Literal["single_encoder"]
        Encoding strategy.
    include_instructions : bool
        Whether to include task instructions.
    learning_rate : float
        Learning rate (> 0).
    batch_size : int
        Batch size (> 0).
    num_epochs : int
        Training epochs (> 0).
    device : Literal["cpu", "cuda", "mps"]
        Training device.
    scale : Range[float]
        Numeric range for the ordinal scale.
    distribution : Literal["truncated_normal"]
        Distribution for bounded continuous responses.
    sigma : float
        Standard deviation (> 0).
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration.
    """

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    encoder_mode: Literal["single_encoder"] = "single_encoder"
    include_instructions: bool = False
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    scale: dx.Embed[Range[float]] = dx.field(default_factory=_default_scale)
    distribution: Literal["truncated_normal"] = "truncated_normal"
    sigma: float = 0.1
    mixed_effects: dx.Embed[MixedEffectsConfig] = dx.field(
        default_factory=_default_mixed_effects
    )


class MagnitudeModelConfig(dx.Model):
    """Configuration for magnitude active-learning models.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length (> 0).
    encoder_mode : Literal["single_encoder"]
        Encoding strategy.
    include_instructions : bool
        Whether to include task instructions.
    learning_rate : float
        Learning rate (> 0).
    batch_size : int
        Batch size (> 0).
    num_epochs : int
        Training epochs (> 0).
    device : Literal["cpu", "cuda", "mps"]
        Training device.
    bounded : bool
        Whether magnitude values are bounded.
    min_value : float | None
        Minimum value (required when ``bounded=True``).
    max_value : float | None
        Maximum value (required when ``bounded=True``).
    distribution : Literal["normal", "truncated_normal"]
        Response distribution.
    sigma : float
        Standard deviation (> 0).
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration.
    """

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    encoder_mode: Literal["single_encoder"] = "single_encoder"
    include_instructions: bool = False
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    bounded: bool = False
    min_value: float | None = None
    max_value: float | None = None
    distribution: Literal["normal", "truncated_normal"] = "normal"
    sigma: float = 0.1
    mixed_effects: dx.Embed[MixedEffectsConfig] = dx.field(
        default_factory=_default_mixed_effects
    )


def validate_magnitude_model_config(config: MagnitudeModelConfig) -> None:
    """Raise ``ValueError`` if *config*'s bounded options are inconsistent."""
    if config.bounded:
        if config.min_value is None or config.max_value is None:
            raise ValueError(
                "bounded=True requires both min_value and max_value to be set. "
                f"Got min_value={config.min_value}, max_value={config.max_value}."
            )
        if config.min_value >= config.max_value:
            raise ValueError(
                f"min_value ({config.min_value}) must be less than "
                f"max_value ({config.max_value})."
            )
        if config.distribution != "truncated_normal":
            raise ValueError(
                "bounded=True requires distribution='truncated_normal'. "
                f"Got distribution='{config.distribution}'."
            )
    else:
        if config.min_value is not None or config.max_value is not None:
            raise ValueError(
                "bounded=False but min_value or max_value is set. "
                f"Got min_value={config.min_value}, max_value={config.max_value}. "
                "Either set bounded=True or remove min_value/max_value."
            )
        if config.distribution != "normal":
            raise ValueError(
                "bounded=False requires distribution='normal'. "
                f"Got distribution='{config.distribution}'."
            )


def _default_lora_target_modules() -> tuple[str, ...]:
    return ("q", "v")


class FreeTextModelConfig(dx.Model):
    """Configuration for free-text generation with LoRA + GLMM support.

    Attributes
    ----------
    model_name : str
        HuggingFace seq2seq model identifier.
    max_input_length : int
        Maximum input sequence length (> 0).
    max_output_length : int
        Maximum output sequence length (> 0).
    num_beams : int
        Beam search width (> 0).
    temperature : float
        Sampling temperature (> 0).
    top_p : float
        Nucleus sampling probability cutoff (0.0-1.0).
    learning_rate : float
        Learning rate (> 0).
    batch_size : int
        Batch size (> 0).
    num_epochs : int
        Training epochs (> 0).
    device : Literal["cpu", "cuda", "mps"]
        Training device.
    lora_rank : int
        LoRA rank (> 0).
    lora_alpha : float
        LoRA scaling factor (> 0).
    lora_dropout : float
        LoRA dropout probability (0.0 <= p < 1.0).
    lora_target_modules : tuple[str, ...]
        Attention modules to apply LoRA to.
    eval_metric : Literal["exact_match", "token_accuracy", "bleu"]
        Evaluation metric.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration.
    """

    model_name: str = "t5-base"
    max_input_length: int = 128
    max_output_length: int = 64
    num_beams: int = 4
    temperature: float = 1.0
    top_p: float = 0.9
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 3
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = dx.field(
        default_factory=_default_lora_target_modules
    )
    eval_metric: Literal["exact_match", "token_accuracy", "bleu"] = "exact_match"
    mixed_effects: dx.Embed[MixedEffectsConfig] = dx.field(
        default_factory=_default_mixed_effects
    )


class ClozeModelConfig(dx.Model):
    """Configuration for cloze (fill-in-the-blank) models.

    Attributes
    ----------
    model_name : str
        HuggingFace masked-LM identifier.
    max_length : int
        Maximum sequence length (> 0).
    learning_rate : float
        Learning rate (> 0).
    batch_size : int
        Batch size (> 0).
    num_epochs : int
        Training epochs (> 0).
    device : Literal["cpu", "cuda", "mps"]
        Training device.
    mask_token : str
        Token used for masking.
    eval_metric : Literal["exact_match", "token_accuracy"]
        Evaluation metric.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration.
    """

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    mask_token: str = "[MASK]"
    eval_metric: Literal["exact_match", "token_accuracy"] = "exact_match"
    mixed_effects: dx.Embed[MixedEffectsConfig] = dx.field(
        default_factory=_default_mixed_effects
    )


def _default_forced_choice_model() -> ForcedChoiceModelConfig:
    return ForcedChoiceModelConfig()


def _default_trainer() -> TrainerConfig:
    return TrainerConfig()


def _default_loop() -> ActiveLearningLoopConfig:
    return ActiveLearningLoopConfig()


def _default_uncertainty_sampler() -> UncertaintySamplerConfig:
    return UncertaintySamplerConfig()


class ActiveLearningConfig(dx.Model):
    """Configuration for the active-learning subsystem.

    Attributes
    ----------
    forced_choice_model : ForcedChoiceModelConfig
        Forced-choice model configuration.
    trainer : TrainerConfig
        Trainer configuration.
    loop : ActiveLearningLoopConfig
        Active-learning loop configuration.
    uncertainty_sampler : UncertaintySamplerConfig
        Uncertainty sampler configuration.
    """

    forced_choice_model: dx.Embed[ForcedChoiceModelConfig] = dx.field(
        default_factory=_default_forced_choice_model
    )
    trainer: dx.Embed[TrainerConfig] = dx.field(default_factory=_default_trainer)
    loop: dx.Embed[ActiveLearningLoopConfig] = dx.field(default_factory=_default_loop)
    uncertainty_sampler: dx.Embed[UncertaintySamplerConfig] = dx.field(
        default_factory=_default_uncertainty_sampler
    )
