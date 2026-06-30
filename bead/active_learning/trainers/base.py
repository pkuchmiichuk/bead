"""Base trainer interface for model training.

This module provides the abstract base class for all trainers and the
ModelMetadata model for tracking training results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from bead.data.base import BeadBaseModel, JsonValue

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel


class ModelMetadata(BeadBaseModel):
    """Training metadata.

    Parameters
    ----------
    model_name : str
        Model identifier.
    framework : str
        Training framework ("huggingface" or "pytorch_lightning").
    training_config : dict[str, str | int | float | bool | Path | None]
        Training configuration used.
    training_data_path : Path
        Path to training data.
    eval_data_path : Path | None
        Path to eval data if used.
    metrics : dict[str, float]
        Final evaluation metrics.
    best_checkpoint : Path | None
        Path to best checkpoint.
    training_time : float
        Total training time in seconds.
    training_timestamp : str
        ISO 8601 timestamp when training completed.

    Attributes
    ----------
    model_name : str
        Model identifier.
    framework : str
        Training framework ("huggingface" or "pytorch_lightning").
    training_config : dict[str, str | int | float | bool | Path | None]
        Training configuration used.
    training_data_path : Path
        Path to training data.
    eval_data_path : Path | None
        Path to eval data if used.
    metrics : dict[str, float]
        Final evaluation metrics.
    best_checkpoint : Path | None
        Path to best checkpoint.
    training_time : float
        Total training time in seconds.
    training_timestamp : str
        ISO 8601 timestamp when training completed.

    Examples
    --------
    >>> from pathlib import Path
    >>> metadata = ModelMetadata(
    ...     model_name="bert-base-uncased",
    ...     framework="huggingface",
    ...     training_config={"epochs": 3},
    ...     training_data_path=Path("train.json"),
    ...     metrics={"accuracy": 0.95},
    ...     training_time=120.5,
    ...     training_timestamp="2025-01-17T00:00:00+00:00"
    ... )
    >>> metadata.framework
    'huggingface'
    >>> metadata.metrics["accuracy"]
    0.95
    """

    model_name: str
    framework: str
    training_data_path: Path
    metrics: dict[str, float]
    training_time: float
    training_timestamp: str
    training_config: dict[str, JsonValue] = {}
    eval_data_path: Path | None = None
    best_checkpoint: Path | None = None


class BaseTrainer(ABC):
    """Base trainer interface.

    All trainers must implement the train, save_model, and load_model methods.
    This provides a consistent interface across different training frameworks.

    Parameters
    ----------
    config : dict[str, str | int | float | bool | Path | None] | BeadBaseModel
        Training configuration (framework-specific).

    Attributes
    ----------
    config : dict[str, str | int | float | bool | Path | None] | BeadBaseModel
        Training configuration.

    Examples
    --------
    >>> class MyTrainer(BaseTrainer):
    ...     def train(self, train_data, eval_data=None):
    ...         return ModelMetadata(
    ...             model_name="test",
    ...             framework="custom",
    ...             training_config={},
    ...             training_data_path=Path("train.json"),
    ...             metrics={},
    ...             training_time=0.0,
    ...             training_timestamp="2025-01-17T00:00:00+00:00"
    ...         )
    ...     def save_model(self, output_dir, metadata):
    ...         pass
    ...     def load_model(self, model_dir):
    ...         return None
    >>> trainer = MyTrainer(config={})
    >>> trainer.config
    {}
    """

    def __init__(
        self, config: dict[str, str | int | float | bool | Path | None] | BeadBaseModel
    ) -> None:
        self.config = config

    @abstractmethod
    def train(
        self,
        train_data: Dataset
        | dict[str, str | int | float | bool | None]
        | list[dict[str, str | int | float | bool | None]],
        eval_data: Dataset
        | dict[str, str | int | float | bool | None]
        | list[dict[str, str | int | float | bool | None]]
        | None = None,
    ) -> ModelMetadata:
        """Train model and return metadata.

        Parameters
        ----------
        train_data : Dataset | dict | list
            Training dataset (framework-specific format).
        eval_data : Dataset | dict | list | None
            Evaluation dataset (framework-specific format).

        Returns
        -------
        ModelMetadata
            Metadata about the training run.

        Examples
        --------
        >>> trainer = MyTrainer(config={})  # doctest: +SKIP
        >>> metadata = trainer.train(train_dataset)  # doctest: +SKIP
        >>> metadata.framework  # doctest: +SKIP
        'custom'
        """
        pass

    @abstractmethod
    def save_model(self, output_dir: Path, metadata: ModelMetadata) -> None:
        """Save model and metadata to directory.

        Parameters
        ----------
        output_dir : Path
            Directory to save model and metadata.
        metadata : ModelMetadata
            Training metadata to save.

        Examples
        --------
        >>> trainer = MyTrainer(config={})  # doctest: +SKIP
        >>> trainer.save_model(Path("output"), metadata)  # doctest: +SKIP
        """
        pass

    @abstractmethod
    def load_model(
        self, model_dir: Path
    ) -> PreTrainedModel | dict[str, str | int | float | bool | None] | BeadBaseModel:
        """Load model from directory.

        Parameters
        ----------
        model_dir : Path
            Directory containing saved model.

        Returns
        -------
        PreTrainedModel | dict[str, str | int | float | bool | None] | BeadBaseModel
            Loaded model (framework-specific type).

        Examples
        --------
        >>> trainer = MyTrainer(config={})  # doctest: +SKIP
        >>> model = trainer.load_model(Path("saved_model"))  # doctest: +SKIP
        """
        pass
