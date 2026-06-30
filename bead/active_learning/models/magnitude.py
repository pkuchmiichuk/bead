"""Magnitude model for unbounded and bounded numeric judgments.

Implements continuous regression with support for:
- Unbounded values: Normal distribution N(μ, σ²)
- Bounded values: Truncated Normal distribution N(μ, σ) T[min, max]
Supports GLMM with participant-level random effects (intercepts and slopes).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from bead.active_learning.config import MixedEffectsConfig, VarianceComponents
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.active_learning.trainers.data_collator import MixedEffectsDataCollator
from bead.active_learning.trainers.dataset_utils import items_to_dataset
from bead.active_learning.trainers.metrics import compute_regression_metrics
from bead.active_learning.trainers.model_wrapper import EncoderRegressionWrapper
from bead.config.active_learning import MagnitudeModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["MagnitudeModel"]


class MagnitudeModel(ActiveLearningModel):
    """Model for magnitude tasks with unbounded or bounded continuous responses.

    Uses Normal distribution for unbounded values (e.g., reading time, plausibility)
    or Truncated Normal for bounded values (e.g., confidence on 0-100 scale).
    Supports three modes: fixed effects, random intercepts, random slopes.

    Parameters
    ----------
    config : MagnitudeModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : MagnitudeModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Transformer tokenizer.
    encoder : AutoModel
        Transformer encoder model.
    regression_head : nn.Sequential
        Regression head (fixed effects head) - outputs continuous μ.
    random_effects : RandomEffectsManager
        Manager for participant-level random effects.
    variance_history : list[VarianceComponents]
        Variance component estimates over training (for diagnostics).
    _is_fitted : bool
        Whether model has been trained.

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.items.item import Item
    >>> from bead.config.active_learning import MagnitudeModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"text": f"Sentence {i}"}
    ...     )
    ...     for i in range(10)
    ... ]
    >>> labels = ["250.5", "300.2"] * 5  # Reading times in ms
    >>> config = MagnitudeModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = MagnitudeModel(config=config)  # doctest: +SKIP
    >>> metrics = model.train(items, labels, participant_ids=None)  # doctest: +SKIP
    >>> predictions = model.predict(items[:3], participant_ids=None)  # doctest: +SKIP
    """

    def __init__(
        self,
        config: MagnitudeModelConfig | None = None,
    ) -> None:
        """Initialize magnitude model.

        Parameters
        ----------
        config : MagnitudeModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or MagnitudeModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.encoder = AutoModel.from_pretrained(self.config.model_name)

        self.regression_head: nn.Sequential | None = None
        self._is_fitted = False

        # Initialize random effects manager
        self.random_effects: RandomEffectsManager | None = None
        self.variance_history: list[VarianceComponents] = []

        self.encoder.to(self.config.device)

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "magnitude".
        """
        return ["magnitude"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with magnitude model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "magnitude".
        """
        if item_template.task_type != "magnitude":
            raise ValueError(
                f"Expected task_type 'magnitude', got '{item_template.task_type}'"
            )

    def _initialize_regression_head(self) -> None:
        """Initialize regression head for continuous output μ."""
        hidden_size = self.encoder.config.hidden_size

        # Single output for location parameter μ
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # Output μ (location parameter)
        )
        self.regression_head.to(self.config.device)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode texts using transformer.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.

        Returns
        -------
        torch.Tensor
            Encoded representations of shape (batch_size, hidden_size).
        """
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.config.device) for k, v in encodings.items()}

        outputs = self.encoder(**encodings)
        return outputs.last_hidden_state[:, 0, :]

    def _prepare_inputs(self, items: list[Item]) -> torch.Tensor:
        """Prepare inputs for encoding.

        For magnitude tasks, concatenates all rendered elements.

        Parameters
        ----------
        items : list[Item]
            Items to encode.

        Returns
        -------
        torch.Tensor
            Encoded representations.
        """
        texts = []
        for item in items:
            # Concatenate all rendered elements
            all_text = " ".join(item.rendered_elements.values())
            texts.append(all_text)
        return self._encode_texts(texts)

    def _normal_log_prob(self, y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Compute log probability of normal distribution (unbounded case).

        Parameters
        ----------
        y : torch.Tensor
            Observed values, shape (batch,).
        mu : torch.Tensor
            Location parameters, shape (batch,).

        Returns
        -------
        torch.Tensor
            Log probabilities, shape (batch,).
        """
        dist = Normal(mu.squeeze(), self.config.sigma)
        return dist.log_prob(y)

    def _truncated_normal_log_prob(
        self, y: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of truncated normal distribution (bounded case).

        Uses truncated normal on [min_value, max_value] to properly handle
        bounded responses without arbitrary nudging.

        Parameters
        ----------
        y : torch.Tensor
            Observed values, shape (batch,).
        mu : torch.Tensor
            Location parameters (before truncation), shape (batch,).

        Returns
        -------
        torch.Tensor
            Log probabilities, shape (batch,).
        """
        base_dist = Normal(mu.squeeze(), self.config.sigma)

        # Unnormalized log prob
        log_prob_unnorm = base_dist.log_prob(y)

        # Normalizer: log(Φ((high-μ)/σ) - Φ((low-μ)/σ))
        alpha = (self.config.min_value - mu.squeeze()) / self.config.sigma
        beta = (self.config.max_value - mu.squeeze()) / self.config.sigma
        normalizer = base_dist.cdf(beta) - base_dist.cdf(alpha)

        # Clamp to avoid log(0)
        normalizer = torch.clamp(normalizer, min=1e-8)
        log_normalizer = torch.log(normalizer)

        return log_prob_unnorm - log_normalizer

    def _prepare_training_data(
        self,
        items: list[Item],
        labels: list[str],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> tuple[
        list[Item], list[float], list[str], list[Item] | None, list[float] | None
    ]:
        """Prepare training data for magnitude model.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (continuous values as strings).
        participant_ids : list[str]
            Normalized participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[str] | None
            Validation labels.

        Returns
        -------
        tuple[list[Item], list[float], list[str], list[Item] | None, list[float] | None]
            Prepared items, numeric labels (floats), participant_ids,
            validation_items, numeric validation_labels.
        """
        # Parse labels to floats
        try:
            y_values = [float(label) for label in labels]
        except ValueError as e:
            raise ValueError(
                f"Labels must be numeric strings (e.g., '250.5', '300.2'). "
                f"Got error: {e}"
            ) from e

        # Validate bounds for bounded case
        if self.config.bounded:
            for i, val in enumerate(y_values):
                if not (self.config.min_value <= val <= self.config.max_value):
                    raise ValueError(
                        f"Label at index {i} ({val}) is outside bounds "
                        f"[{self.config.min_value}, {self.config.max_value}]"
                    )

        self._initialize_regression_head()

        # Convert validation labels if provided
        val_y_numeric = None
        if validation_items is not None and validation_labels is not None:
            try:
                val_y_numeric = [float(label) for label in validation_labels]
            except ValueError as e:
                raise ValueError(
                    f"Validation labels must be numeric strings. Got error: {e}"
                ) from e

            # Validate bounds for validation labels
            if self.config.bounded:
                for i, val in enumerate(val_y_numeric):
                    if not (self.config.min_value <= val <= self.config.max_value):
                        raise ValueError(
                            f"Validation label at index {i} ({val}) is outside bounds "
                            f"[{self.config.min_value}, {self.config.max_value}]"
                        )

        return items, y_values, participant_ids, validation_items, val_y_numeric

    def _initialize_random_effects(self, n_classes: int) -> None:
        """Initialize random effects manager.

        Parameters
        ----------
        n_classes : int
            Number of classes (1 for regression).
        """
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects,
            n_classes=n_classes,  # Scalar bias for μ
        )

    def _do_training(
        self,
        items: list[Item],
        labels_numeric: list[float],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels_numeric: list[float] | None,
    ) -> dict[str, float]:
        """Perform magnitude model training.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels_numeric : list[float]
            Numeric labels (continuous values).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels_numeric : list[float] | None
            Numeric validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        # Convert validation_labels_numeric back to string labels for validation metrics
        validation_labels = None
        if validation_items is not None and validation_labels_numeric is not None:
            validation_labels = [str(val) for val in validation_labels_numeric]

        # Use HuggingFace Trainer for fixed and random_intercepts modes
        # random_slopes requires custom loop due to per-participant heads
        use_huggingface_trainer = self.config.mixed_effects.mode in (
            "fixed",
            "random_intercepts",
        )

        if use_huggingface_trainer:
            metrics = self._train_with_huggingface_trainer(
                items,
                labels_numeric,
                participant_ids,
                validation_items,
                validation_labels,
            )
        else:
            # Use custom training loop for random_slopes
            metrics = self._train_with_custom_loop(
                items,
                labels_numeric,
                participant_ids,
                validation_items,
                validation_labels,
            )

        # Add validation MSE if validation data provided and not already computed
        if (
            validation_items is not None
            and validation_labels is not None
            and "val_mse" not in metrics
        ):
            # Validation with placeholder participant_ids for mixed effects
            if self.config.mixed_effects.mode == "fixed":
                val_participant_ids = ["_fixed_"] * len(validation_items)
            else:
                val_participant_ids = ["_validation_"] * len(validation_items)
            val_predictions = self._do_predict(validation_items, val_participant_ids)
            val_pred_values = [float(p.predicted_class) for p in val_predictions]
            val_true_values = [float(label) for label in validation_labels]
            val_mse = np.mean(
                [
                    (pred - true) ** 2
                    for pred, true in zip(val_pred_values, val_true_values, strict=True)
                ]
            )
            metrics["val_mse"] = val_mse

        return metrics

    def _train_with_huggingface_trainer(
        self,
        items: list[Item],
        y_numeric: list[float],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> dict[str, float]:
        """Train using HuggingFace Trainer with mixed effects support for regression.

        Parameters
        ----------
        items : list[Item]
            Training items.
        y_numeric : list[float]
            Numeric labels (continuous values).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[str] | None
            Validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        # Convert items to HuggingFace Dataset
        train_dataset = items_to_dataset(
            items=items,
            labels=y_numeric,
            participant_ids=participant_ids,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        eval_dataset = None
        if validation_items is not None and validation_labels is not None:
            val_y_numeric = [float(label) for label in validation_labels]
            val_participant_ids = (
                ["_validation_"] * len(validation_items)
                if self.config.mixed_effects.mode != "fixed"
                else ["_fixed_"] * len(validation_items)
            )
            eval_dataset = items_to_dataset(
                items=validation_items,
                labels=val_y_numeric,
                participant_ids=val_participant_ids,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )

        # Wrap the encoder and regression head for Trainer
        wrapped_model = EncoderRegressionWrapper(
            encoder=self.encoder, regression_head=self.regression_head
        )

        # Create data collator
        data_collator = MixedEffectsDataCollator(tokenizer=self.tokenizer)

        # Create training arguments with checkpointing
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(checkpoint_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                logging_steps=10,
                eval_strategy="epoch" if eval_dataset is not None else "no",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=False,
                report_to="none",
                remove_unused_columns=False,
                use_cpu=self.config.device == "cpu",
            )

            # Import here to avoid circular import
            from bead.active_learning.trainers.mixed_effects import (  # noqa: PLC0415
                MixedEffectsTrainer,
            )

            # Create trainer
            trainer = MixedEffectsTrainer(
                model=wrapped_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                random_effects_manager=self.random_effects,
                compute_metrics=compute_regression_metrics,
            )

            # Train
            train_result = trainer.train()

            # Get training metrics
            train_metrics = trainer.evaluate(eval_dataset=train_dataset)
            metrics: dict[str, float] = {
                "train_loss": float(train_result.training_loss),
                "train_mse": train_metrics.get("eval_mse", 0.0),
                "train_mae": train_metrics.get("eval_mae", 0.0),
                "train_r2": train_metrics.get("eval_r2", 0.0),
            }

            # Get validation metrics if eval_dataset was provided
            if eval_dataset is not None:
                val_metrics = trainer.evaluate(eval_dataset=eval_dataset)
                metrics.update(
                    {
                        "val_mse": val_metrics.get("eval_mse", 0.0),
                        "val_mae": val_metrics.get("eval_mae", 0.0),
                        "val_r2": val_metrics.get("eval_r2", 0.0),
                    }
                )

        return metrics

    def _train_with_custom_loop(
        self,
        items: list[Item],
        y_numeric: list[float],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> dict[str, float]:
        """Train using custom loop for random_slopes mode.

        Parameters
        ----------
        items : list[Item]
            Training items.
        y_numeric : list[float]
            Numeric labels (continuous values).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[str] | None
            Validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        y = torch.tensor(y_numeric, dtype=torch.float, device=self.config.device)

        # Build optimizer parameters
        params_to_optimize = list(self.encoder.parameters()) + list(
            self.regression_head.parameters()
        )

        # Add random effects parameters for random_slopes
        for head in self.random_effects.slopes.values():
            params_to_optimize.extend(head.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config.learning_rate)

        self.encoder.train()
        self.regression_head.train()

        for _epoch in range(self.config.num_epochs):
            n_batches = (
                len(items) + self.config.batch_size - 1
            ) // self.config.batch_size
            epoch_loss = 0.0
            epoch_mse = 0.0

            for i in range(n_batches):
                start_idx = i * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(items))

                batch_items = items[start_idx:end_idx]
                batch_labels = y[start_idx:end_idx]
                batch_participant_ids = participant_ids[start_idx:end_idx]

                embeddings = self._prepare_inputs(batch_items)

                # Per-participant head for random_slopes
                mu_list = []
                for j, pid in enumerate(batch_participant_ids):
                    participant_head = self.random_effects.get_slopes(
                        pid,
                        fixed_head=self.regression_head,
                        create_if_missing=True,
                    )
                    mu_j = participant_head(embeddings[j : j + 1]).squeeze()
                    mu_list.append(mu_j)
                mu = torch.stack(mu_list)

                # Compute negative log-likelihood
                if self.config.bounded:
                    log_probs = self._truncated_normal_log_prob(batch_labels, mu)
                else:
                    log_probs = self._normal_log_prob(batch_labels, mu)
                loss_nll = -log_probs.mean()

                # Add prior regularization
                loss_prior = self.random_effects.compute_prior_loss()
                loss = loss_nll + loss_prior

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # Also track MSE for interpretability
                mse = ((mu - batch_labels) ** 2).mean().item()
                epoch_mse += mse

            epoch_loss = epoch_loss / n_batches
            epoch_mse = epoch_mse / n_batches

        metrics: dict[str, float] = {
            "train_loss": epoch_loss,
            "train_mse": epoch_mse,
        }

        return metrics

    def _do_predict(
        self, items: list[Item], participant_ids: list[str]
    ) -> list[ModelPrediction]:
        """Perform magnitude model prediction.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        list[ModelPrediction]
            Predictions with predicted_class as string representation of value.
        """
        self.encoder.eval()
        self.regression_head.eval()

        with torch.no_grad():
            embeddings = self._prepare_inputs(items)

            # Forward pass depends on mixed effects mode
            if self.config.mixed_effects.mode == "fixed":
                mu = self.regression_head(embeddings).squeeze(1)

            elif self.config.mixed_effects.mode == "random_intercepts":
                mu = self.regression_head(embeddings).squeeze(1)
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use prior mean (zero bias)
                    bias = self.random_effects.get_intercepts(
                        pid, n_classes=1, param_name="mu", create_if_missing=False
                    )
                    mu[i] = mu[i] + bias.item()

            elif self.config.mixed_effects.mode == "random_slopes":
                mu_list = []
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use fixed head
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.regression_head, create_if_missing=False
                    )
                    mu_i = participant_head(embeddings[i : i + 1]).squeeze()
                    mu_list.append(mu_i)
                mu = torch.stack(mu_list)

            # Clamp predictions to bounds if bounded
            if self.config.bounded:
                mu = torch.clamp(mu, self.config.min_value, self.config.max_value)

            predictions_array = mu.cpu().numpy()

        predictions = []
        for i, item in enumerate(items):
            pred_value = float(predictions_array[i])
            predictions.append(
                ModelPrediction(
                    item_id=str(item.id),
                    probabilities={},  # Not applicable for regression
                    predicted_class=str(pred_value),  # Continuous value as string
                    confidence=1.0,  # Not applicable for regression
                )
            )

        return predictions

    def _do_predict_proba(
        self, items: list[Item], participant_ids: list[str]
    ) -> np.ndarray:
        """Perform magnitude model probability prediction.

        For magnitude regression, returns μ values directly.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        np.ndarray
            Array of shape (n_items, 1) with predicted μ values.
        """
        predictions = self._do_predict(items, participant_ids)
        return np.array([[float(p.predicted_class)] for p in predictions])

    def _save_model_components(self, save_path: Path) -> None:
        """Save model-specific components.

        Parameters
        ----------
        save_path : Path
            Directory to save to.
        """
        self.encoder.save_pretrained(save_path / "encoder")
        self.tokenizer.save_pretrained(save_path / "encoder")

        torch.save(
            self.regression_head.state_dict(),
            save_path / "regression_head.pt",
        )

    def _get_save_state(self) -> dict[str, object]:
        """Get model-specific state to save.

        Returns
        -------
        dict[str, object]
            State dictionary.
        """
        return {}

    def _restore_training_state(self, config_dict: dict[str, object]) -> None:
        """Restore model-specific training state.

        Parameters
        ----------
        config_dict : dict[str, object]
            Configuration dictionary with training state.
        """
        # MagnitudeModel doesn't have additional training state to restore
        pass

    def _load_model_components(
        self, load_path: Path, config_dict: dict[str, object]
    ) -> None:
        """Load model-specific components.

        Parameters
        ----------
        load_path : Path
            Directory to load from.
        config_dict : dict[str, object]
            Schema-only config dict.
        """
        # Reconstruct MixedEffectsConfig if needed
        if "mixed_effects" in config_dict and isinstance(
            config_dict["mixed_effects"], dict
        ):
            config_dict["mixed_effects"] = MixedEffectsConfig(
                **config_dict["mixed_effects"]
            )

        self.config = MagnitudeModelConfig(**config_dict)

        self.encoder = AutoModel.from_pretrained(load_path / "encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "encoder")

        self._initialize_regression_head()
        self.regression_head.load_state_dict(
            torch.load(
                load_path / "regression_head.pt", map_location=self.config.device
            )
        )

        self.encoder.to(self.config.device)
        self.regression_head.to(self.config.device)

    def _get_n_classes_for_random_effects(self) -> int:
        """Get the number of classes for initializing RandomEffectsManager.

        For magnitude models, this is 1 (scalar bias).

        Returns
        -------
        int
            Always 1 for regression.
        """
        return 1

    def _get_random_effects_fixed_head(self) -> torch.nn.Module | None:
        """Get the fixed head for random effects.

        Returns
        -------
        torch.nn.Module | None
            The regression head, or None if not applicable.
        """
        return self.regression_head
