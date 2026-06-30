"""HuggingFace Transformers trainer implementation.

This module provides a trainer that uses the HuggingFace Transformers library
for model training with integrated TensorBoard logging.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from bead.active_learning.trainers.base import BaseTrainer, ModelMetadata
from bead.data.base import BeadBaseModel
from bead.data.timestamps import format_iso8601, now_iso8601

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer


class HuggingFaceTrainer(BaseTrainer):
    """Trainer using HuggingFace Transformers.

    This trainer uses the HuggingFace Transformers library to train models
    for sequence classification and other NLP tasks. It supports TensorBoard
    logging and checkpoint management.

    Parameters
    ----------
    config : dict[str, int | str | float | bool | Path] | BeadBaseModel
        Training configuration with the following expected fields:
        - model_name: str - Base model name/path
        - task_type: str - Task type (classification, regression, etc.)
        - num_labels: int | None - Number of labels for classification
        - output_dir: Path - Directory for outputs
        - num_epochs: int - Number of training epochs
        - batch_size: int - Training batch size
        - learning_rate: float - Learning rate
        - weight_decay: float - Weight decay
        - warmup_steps: int - Warmup steps
        - evaluation_strategy: str - Evaluation strategy (epoch, steps, no)
        - save_strategy: str - Save strategy (epoch, steps, no)
        - load_best_model_at_end: bool - Load best model at end
        - logging_dir: Path | None - Logging directory
        - fp16: bool - Use mixed precision

    Attributes
    ----------
    config : dict[str, int | str | float | bool | Path] | BeadBaseModel
        Training configuration.
    model : PreTrainedModel | None
        The trained model.
    tokenizer : PreTrainedTokenizer | None
        The tokenizer.

    Examples
    --------
    >>> from pathlib import Path
    >>> config = {
    ...     "model_name": "bert-base-uncased",
    ...     "task_type": "classification",
    ...     "num_labels": 2,
    ...     "output_dir": Path("output"),
    ...     "num_epochs": 3,
    ...     "batch_size": 16,
    ...     "learning_rate": 2e-5,
    ...     "weight_decay": 0.01,
    ...     "warmup_steps": 0,
    ...     "evaluation_strategy": "epoch",
    ...     "save_strategy": "epoch",
    ...     "load_best_model_at_end": True,
    ...     "logging_dir": None,
    ...     "fp16": False
    ... }
    >>> trainer = HuggingFaceTrainer(config)
    >>> trainer.model is None
    True
    """

    def __init__(
        self, config: dict[str, int | str | float | bool | Path] | BeadBaseModel
    ) -> None:
        super().__init__(config)
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None

    def _get_config_value(
        self, key: str, default: int | str | float | bool | Path | None = None
    ) -> int | str | float | bool | Path | None:
        """Get configuration value with fallback to default.

        Parameters
        ----------
        key : str
            Configuration key.
        default : int | str | float | bool | Path | None
            Default value if key not found.

        Returns
        -------
        int | str | float | bool | Path | None
            Configuration value.
        """
        if hasattr(self.config, key):
            return getattr(self.config, key)
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return default

    def train(
        self, train_data: Dataset, eval_data: Dataset | None = None
    ) -> ModelMetadata:
        """Train model using HuggingFace Trainer.

        Parameters
        ----------
        train_data : Dataset
            HuggingFace Dataset for training.
        eval_data : Dataset | None
            HuggingFace Dataset for evaluation.

        Returns
        -------
        ModelMetadata
            Training metadata.

        Raises
        ------
        ValueError
            If task type is not supported.

        Examples
        --------
        >>> config = {"model_name": "bert-base-uncased"}  # doctest: +SKIP
        >>> trainer = HuggingFaceTrainer(config)  # doctest: +SKIP
        >>> metadata = trainer.train(train_dataset)  # doctest: +SKIP
        >>> metadata.framework  # doctest: +SKIP
        'huggingface'
        """
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        start_time = time.time()

        # Get config values
        model_name = self._get_config_value("model_name", "bert-base-uncased")
        task_type = self._get_config_value("task_type", "classification")
        num_labels = self._get_config_value("num_labels", 2)
        output_dir = self._get_config_value("output_dir", Path("output"))
        num_epochs = self._get_config_value("num_epochs", 3)
        batch_size = self._get_config_value("batch_size", 16)
        learning_rate = self._get_config_value("learning_rate", 2e-5)
        weight_decay = self._get_config_value("weight_decay", 0.01)
        warmup_steps = self._get_config_value("warmup_steps", 0)
        evaluation_strategy = self._get_config_value("evaluation_strategy", "epoch")
        save_strategy = self._get_config_value("save_strategy", "epoch")
        load_best = self._get_config_value("load_best_model_at_end", True)
        logging_dir = self._get_config_value("logging_dir", None)
        fp16 = self._get_config_value("fp16", False)

        # Load model and tokenizer
        if task_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        else:
            msg = f"Task type not supported: {task_type}"
            raise ValueError(msg)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy=evaluation_strategy,  # type: ignore
            save_strategy=save_strategy,
            load_best_model_at_end=load_best,
            logging_dir=str(logging_dir) if logging_dir else None,
            fp16=fp16,
            report_to=["tensorboard"] if logging_dir else [],
        )

        # Create data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Evaluate
        metrics = {}
        if eval_data is not None:
            eval_results = trainer.evaluate()
            metrics = {k: float(v) for k, v in eval_results.items()}

        training_time = time.time() - start_time

        # Get best checkpoint path
        best_checkpoint = None
        if trainer.state.best_model_checkpoint:
            best_checkpoint = Path(trainer.state.best_model_checkpoint)

        # Create metadata
        # Build a JSON-shaped (Path -> str) view of the training config so
        # ModelMetadata.training_config (typed dict[str, JsonValue]) accepts it.
        if isinstance(self.config, dict):
            raw_dict = self.config
        elif hasattr(self.config, "model_dump"):
            raw_dict = self.config.model_dump()
        else:
            raw_dict = {}

        def _coerce(v: object) -> object:
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, dict):
                return {k: _coerce(x) for k, x in v.items()}
            if isinstance(v, list | tuple):
                return [_coerce(x) for x in v]
            return v

        config_dict = {k: _coerce(v) for k, v in raw_dict.items()}

        metadata = ModelMetadata(
            model_name=model_name,
            framework="huggingface",
            training_config=config_dict,
            training_data_path=Path("train.json"),
            eval_data_path=Path("eval.json") if eval_data else None,
            metrics=metrics,
            best_checkpoint=best_checkpoint,
            training_time=training_time,
            training_timestamp=format_iso8601(now_iso8601()),
        )

        return metadata

    def save_model(self, output_dir: Path, metadata: ModelMetadata) -> None:
        """Save model and metadata.

        Parameters
        ----------
        output_dir : Path
            Directory to save model and metadata.
        metadata : ModelMetadata
            Training metadata to save.

        Examples
        --------
        >>> trainer = HuggingFaceTrainer({})  # doctest: +SKIP
        >>> trainer.save_model(Path("output"), metadata)  # doctest: +SKIP
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model is not None:
            self.model.save_pretrained(output_dir / "model")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir / "model")

        # Save metadata
        with open(output_dir / "metadata.json", "w") as f:
            # Convert Path objects to strings for JSON serialization
            metadata_dict = metadata.model_dump()
            json.dump(metadata_dict, f, indent=2, default=str)

    def load_model(self, model_dir: Path) -> PreTrainedModel:
        """Load model.

        Parameters
        ----------
        model_dir : Path
            Directory containing saved model.

        Returns
        -------
        PreTrainedModel
            Loaded model.

        Examples
        --------
        >>> trainer = HuggingFaceTrainer({})  # doctest: +SKIP
        >>> model = trainer.load_model(Path("saved_model"))  # doctest: +SKIP
        """
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir / "model"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "model")

        return self.model
