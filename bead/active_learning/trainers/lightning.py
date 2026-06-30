"""PyTorch Lightning trainer implementation.

This module provides a trainer that uses PyTorch Lightning for model training
with callbacks for checkpointing and early stopping.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bead.active_learning.trainers.base import BaseTrainer, ModelMetadata
from bead.data.base import BeadBaseModel
from bead.data.timestamps import format_iso8601, now_iso8601

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from torch.nn import Module
    from torch.utils.data import DataLoader


def create_lightning_module(
    model: Module, learning_rate: float = 2e-5
) -> pl.LightningModule:
    """Create a PyTorch Lightning module.

    Parameters
    ----------
    model
        The PyTorch model to wrap in a Lightning module.
    learning_rate
        Learning rate for the AdamW optimizer.

    Returns
    -------
    pl.LightningModule
        Lightning module wrapping the provided model with training,
        validation, and optimizer configuration.
    """
    import pytorch_lightning as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415

    class _LightningModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.learning_rate = learning_rate

        def forward(self, **inputs: Any) -> Any:
            return self.model(**inputs)

        def training_step(self, batch: Any, batch_idx: int) -> Any:
            outputs = self(**batch)
            loss = outputs.loss
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch: Any, batch_idx: int) -> Any:
            outputs = self(**batch)
            loss = outputs.loss
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self) -> Any:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
            return optimizer

    return _LightningModule()


class PyTorchLightningTrainer(BaseTrainer):
    """Trainer using PyTorch Lightning.

    Trains models using PyTorch Lightning with callbacks for checkpointing
    and early stopping.

    Parameters
    ----------
    config
        Training configuration as a dict or config object with the following
        fields:

        - model_name: str, base model name or path
        - num_labels: int, number of output labels
        - num_epochs: int, number of training epochs
        - learning_rate: float, learning rate for optimizer
        - output_dir: Path, directory for outputs and checkpoints
        - logging_dir: Path or None, optional TensorBoard logging directory

    Attributes
    ----------
    config : dict[str, int | str | float | bool | Path] | BeadBaseModel
        Training configuration.
    lightning_module : pl.LightningModule | None
        The Lightning module wrapper, set after training.

    Examples
    --------
    >>> from pathlib import Path
    >>> config = {
    ...     "model_name": "bert-base-uncased",
    ...     "num_labels": 2,
    ...     "num_epochs": 3,
    ...     "learning_rate": 2e-5,
    ...     "output_dir": Path("output"),
    ...     "logging_dir": None
    ... }
    >>> trainer = PyTorchLightningTrainer(config)
    >>> trainer.lightning_module is None
    True
    """

    def __init__(
        self, config: dict[str, int | str | float | bool | Path] | BeadBaseModel
    ) -> None:
        super().__init__(config)
        self.lightning_module: pl.LightningModule | None = None

    def _get_config_value(
        self, key: str, default: int | str | float | bool | Path | None = None
    ) -> int | str | float | bool | Path | None:
        """Get configuration value with fallback to default.

        Parameters
        ----------
        key
            Configuration key to retrieve.
        default
            Default value if key is not found.

        Returns
        -------
        int | str | float | bool | Path | None
            Configuration value for the given key, or default if not found.
        """
        if hasattr(self.config, key):
            return getattr(self.config, key)
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return default

    def train(
        self, train_data: DataLoader, eval_data: DataLoader | None = None
    ) -> ModelMetadata:
        """Train a model using PyTorch Lightning.

        Loads a pretrained model, wraps it in a Lightning module, and trains
        with checkpointing and early stopping callbacks.

        Parameters
        ----------
        train_data
            Training dataloader providing batches for training.
        eval_data
            Optional evaluation dataloader for validation during training.

        Returns
        -------
        ModelMetadata
            Metadata containing model name, framework, training config,
            metrics, checkpoint path, and training time.

        Examples
        --------
        >>> config = {"model_name": "bert-base-uncased"}  # doctest: +SKIP
        >>> trainer = PyTorchLightningTrainer(config)  # doctest: +SKIP
        >>> metadata = trainer.train(train_loader)  # doctest: +SKIP
        >>> metadata.framework  # doctest: +SKIP
        'pytorch_lightning'
        """
        import pytorch_lightning as pl  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification  # noqa: PLC0415

        start_time = time.time()

        # get config values
        model_name = self._get_config_value("model_name", "bert-base-uncased")
        num_labels = self._get_config_value("num_labels", 2)
        num_epochs = self._get_config_value("num_epochs", 3)
        learning_rate = self._get_config_value("learning_rate", 2e-5)
        output_dir = self._get_config_value("output_dir", Path("output"))
        logging_dir = self._get_config_value("logging_dir", None)

        # load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        # create lightning module
        self.lightning_module = create_lightning_module(model, learning_rate)

        # create callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=output_dir,
                filename="best-{epoch:02d}-{val_loss:.2f}",
            ),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        ]

        # create logger
        logger = None
        if logging_dir:
            logger = pl.loggers.TensorBoardLogger(str(logging_dir))

        # create trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            devices="auto",
            logger=logger,
            callbacks=callbacks,
        )

        # train
        trainer.fit(
            self.lightning_module,
            train_dataloaders=train_data,
            val_dataloaders=eval_data,
        )

        # evaluate
        metrics: dict[str, float] = {}
        if eval_data is not None:
            eval_results = trainer.validate(
                self.lightning_module, dataloaders=eval_data
            )
            if eval_results:
                metrics = {k: float(v) for k, v in eval_results[0].items()}

        training_time = time.time() - start_time

        # get best checkpoint path
        best_checkpoint = None
        if hasattr(trainer.checkpoint_callback, "best_model_path"):
            best_checkpoint_str = trainer.checkpoint_callback.best_model_path
            if best_checkpoint_str:
                best_checkpoint = Path(best_checkpoint_str)

        # create metadata; flatten ``self.config`` to a JSON-shaped dict
        # so ``ModelMetadata.training_config`` (typed ``dict[str, JsonValue]``)
        # accepts it. ``model_dump_json`` walks Paths / Enums / etc.
        if isinstance(self.config, dict):
            config_dict = json.loads(json.dumps(self.config, default=str))
        elif hasattr(self.config, "model_dump_json"):
            config_dict = json.loads(self.config.model_dump_json())
        else:
            config_dict = {}

        metadata = ModelMetadata(
            model_name=model_name,
            framework="pytorch_lightning",
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
        """Save model and metadata to disk.

        Saves the Lightning module state dict and training metadata as JSON.

        Parameters
        ----------
        output_dir
            Directory to save model checkpoint and metadata JSON file.
        metadata
            Training metadata to save alongside the model.

        Examples
        --------
        >>> trainer = PyTorchLightningTrainer({})  # doctest: +SKIP
        >>> trainer.save_model(Path("output"), metadata)  # doctest: +SKIP
        """
        import torch  # noqa: PLC0415

        output_dir.mkdir(parents=True, exist_ok=True)

        # save lightning checkpoint
        if self.lightning_module is not None:
            torch.save(
                self.lightning_module.state_dict(),
                output_dir / "lightning_model.pt",
            )

        # save metadata
        with open(output_dir / "metadata.json", "w") as f:
            metadata_dict = metadata.model_dump()
            json.dump(metadata_dict, f, indent=2, default=str)

    def load_model(self, model_dir: Path) -> pl.LightningModule | None:
        """Load a saved model from disk.

        Parameters
        ----------
        model_dir
            Directory containing the saved Lightning model state dict.

        Returns
        -------
        pl.LightningModule | None
            The Lightning module with loaded weights, or None if no module
            has been initialized.

        Examples
        --------
        >>> trainer = PyTorchLightningTrainer({})  # doctest: +SKIP
        >>> model = trainer.load_model(Path("saved_model"))  # doctest: +SKIP
        """
        import torch  # noqa: PLC0415

        if self.lightning_module is not None:
            self.lightning_module.load_state_dict(
                torch.load(model_dir / "lightning_model.pt")
            )
        return self.lightning_module
