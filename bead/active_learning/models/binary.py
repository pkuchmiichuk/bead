"""Binary model for yes/no or true/false judgments.

Expected architecture: Binary classification with 2-class output.
Different from 2AFC in semantics - represents absolute judgment rather than choice.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from bead.active_learning.config import VarianceComponents
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.active_learning.trainers.data_collator import MixedEffectsDataCollator
from bead.active_learning.trainers.dataset_utils import items_to_dataset
from bead.active_learning.trainers.metrics import compute_binary_metrics
from bead.active_learning.trainers.model_wrapper import EncoderClassifierWrapper
from bead.config.active_learning import BinaryModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["BinaryModel"]


class BinaryModel(ActiveLearningModel):
    """Model for binary tasks (yes/no, true/false judgments).

    Uses true binary classification with a single output unit and sigmoid
    activation (logistic regression). This is more efficient than using
    2-class softmax, as we only need to output P(y=1) and compute
    P(y=0) = 1 - P(y=1).

    Parameters
    ----------
    config : BinaryModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : BinaryModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Transformer tokenizer.
    encoder : AutoModel
        Transformer encoder model.
    classifier_head : nn.Sequential
        Classification head (fixed effects head) - outputs single logit.
    num_classes : int
        Number of output units (always 1 for binary classification).
    label_names : list[str] | None
        Label names (e.g., ["no", "yes"] sorted alphabetically).
    positive_class : str | None
        Which label corresponds to y=1 (second alphabetically).
    random_effects : RandomEffectsManager
        Manager for participant-level random effects (scalar biases).
    variance_history : list[VarianceComponents]
        Variance component estimates over training (for diagnostics).
    _is_fitted : bool
        Whether model has been trained.

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.items.item import Item
    >>> from bead.config.active_learning import BinaryModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"text": f"Sentence {i}"}
    ...     )
    ...     for i in range(10)
    ... ]
    >>> labels = ["yes"] * 5 + ["no"] * 5
    >>> config = BinaryModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = BinaryModel(config=config)  # doctest: +SKIP
    >>> metrics = model.train(items, labels, participant_ids=None)  # doctest: +SKIP
    >>> predictions = model.predict(items[:3], participant_ids=None)  # doctest: +SKIP

    Notes
    -----
    This model uses BCEWithLogitsLoss instead of CrossEntropyLoss, and applies
    sigmoid activation to get probabilities. Random intercepts are scalar values
    (1-dimensional) that shift the logit for each participant.
    """

    def __init__(
        self,
        config: BinaryModelConfig | None = None,
    ) -> None:
        """Initialize binary model.

        Parameters
        ----------
        config : BinaryModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or BinaryModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.encoder = AutoModel.from_pretrained(self.config.model_name)

        self.num_classes: int = 1  # Single output unit for binary classification
        self.label_names: list[str] | None = None
        self.positive_class: str | None = None  # Which label corresponds to 1
        self.classifier_head: nn.Sequential | None = None
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
            List containing "binary".
        """
        return ["binary"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with binary model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "binary".
        """
        if item_template.task_type != "binary":
            raise ValueError(
                f"Expected task_type 'binary', got '{item_template.task_type}'"
            )

    def _initialize_classifier(self) -> None:
        """Initialize classification head for binary classification.

        Outputs a single value (logit) for sigmoid activation.
        """
        hidden_size = self.encoder.config.hidden_size

        # Single output unit for binary classification
        if self.config.encoder_mode == "dual_encoder":
            input_size = hidden_size * 2
        else:
            input_size = hidden_size

        self.classifier_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # Single output unit
        )
        self.classifier_head.to(self.config.device)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode texts using single encoder.

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

        For binary tasks, concatenates all rendered elements.

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

    def _validate_labels(self, labels: list[str]) -> None:
        """Validate that all labels are valid.

        Parameters
        ----------
        labels : list[str]
            Labels to validate.

        Raises
        ------
        ValueError
            If any label is not in label_names.
        """
        if self.label_names is None:
            raise ValueError("label_names not initialized")

        valid_labels = set(self.label_names)
        invalid = [label for label in labels if label not in valid_labels]
        if invalid:
            raise ValueError(
                f"Invalid labels found: {set(invalid)}. "
                f"Labels must be one of {valid_labels}."
            )

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
        """Prepare training data for binary model.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels.
        participant_ids : list[str]
            Normalized participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[str] | None
            Validation labels.

        Returns
        -------
        tuple[list[Item], list[float], list[str], list[Item] | None, list[float] | None]
            Prepared items, numeric labels, participant_ids, validation_items,
            numeric validation_labels.
        """
        # Initialize label names
        unique_labels = sorted(set(labels))
        if len(unique_labels) != 2:
            raise ValueError(
                f"Binary classification requires exactly 2 classes, "
                f"got {len(unique_labels)}: {unique_labels}"
            )
        self.label_names = unique_labels
        # Positive class is the second one alphabetically (index 1)
        self.positive_class = unique_labels[1]

        self._validate_labels(labels)
        self._initialize_classifier()

        # Convert labels to binary (0/1) floats for HuggingFace Trainer
        # Positive class (second alphabetically) = 1, negative = 0
        y_numeric = [1.0 if label == self.positive_class else 0.0 for label in labels]

        # Convert validation labels if provided
        val_y_numeric = None
        if validation_items is not None and validation_labels is not None:
            self._validate_labels(validation_labels)
            if len(validation_items) != len(validation_labels):
                raise ValueError(
                    f"Number of validation items ({len(validation_items)}) "
                    f"must match number of validation labels ({len(validation_labels)})"
                )
            val_y_numeric = [
                1.0 if label == self.positive_class else 0.0
                for label in validation_labels
            ]

        return items, y_numeric, participant_ids, validation_items, val_y_numeric

    def _initialize_random_effects(self, n_classes: int) -> None:
        """Initialize random effects manager.

        Parameters
        ----------
        n_classes : int
            Number of classes (1 for binary).
        """
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects, n_classes=n_classes
        )

    def _do_training(
        self,
        items: list[Item],
        labels_numeric: list[float],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels_numeric: list[float] | None,
    ) -> dict[str, float]:
        """Perform binary model training.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels_numeric : list[float]
            Numeric labels (0.0 or 1.0).
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
            validation_labels = [
                self.positive_class if label == 1.0 else self.label_names[0]
                for label in validation_labels_numeric
            ]

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

        # Add validation accuracy if validation data provided
        if validation_items is not None and validation_labels is not None:
            # Validation with placeholder participant_ids for mixed effects
            # Use _do_predict directly since we're in training
            if self.config.mixed_effects.mode == "fixed":
                val_participant_ids = ["_fixed_"] * len(validation_items)
            else:
                val_participant_ids = ["_validation_"] * len(validation_items)
            val_predictions = self._do_predict(validation_items, val_participant_ids)
            val_pred_labels = [p.predicted_class for p in val_predictions]
            val_acc = sum(
                pred == true
                for pred, true in zip(val_pred_labels, validation_labels, strict=True)
            ) / len(validation_labels)
            metrics["val_accuracy"] = val_acc

        return metrics

    def _train_with_huggingface_trainer(
        self,
        items: list[Item],
        y_numeric: list[float],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> dict[str, float]:
        """Train using HuggingFace Trainer with mixed effects support.

        Parameters
        ----------
        items : list[Item]
            Training items.
        y_numeric : list[float]
            Numeric labels (0.0 or 1.0).
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

        # Create validation dataset if provided
        eval_dataset = None
        if validation_items is not None and validation_labels is not None:
            val_y_numeric = [
                1.0 if label == self.positive_class else 0.0
                for label in validation_labels
            ]
            eval_dataset = items_to_dataset(
                items=validation_items,
                labels=val_y_numeric,
                participant_ids=["_validation_"] * len(validation_items),
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )

        # Create wrapper model for Trainer
        wrapped_model = EncoderClassifierWrapper(
            encoder=self.encoder, classifier_head=self.classifier_head
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
                save_strategy="epoch",  # Save checkpoints every epoch
                save_total_limit=1,  # Keep only the latest checkpoint
                load_best_model_at_end=False,  # Don't auto-load best
                report_to="none",  # Disable wandb/tensorboard
                remove_unused_columns=False,  # Keep participant_id
                use_cpu=self.config.device == "cpu",  # Explicitly use CPU if specified
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
                compute_metrics=compute_binary_metrics,
            )

            # Train (checkpoints are saved automatically by Trainer)
            train_result = trainer.train()

            # Get training metrics using evaluate (Trainer computes metrics during eval)
            train_metrics = trainer.evaluate(eval_dataset=train_dataset)
            # Trainer prefixes eval metrics with "eval_"
            metrics: dict[str, float] = {
                "train_loss": float(train_result.training_loss),
                "train_accuracy": train_metrics.get("eval_accuracy", 0.0),
                "train_precision": train_metrics.get("eval_precision", 0.0),
                "train_recall": train_metrics.get("eval_recall", 0.0),
                "train_f1": train_metrics.get("eval_f1", 0.0),
            }

            # Get validation metrics if eval_dataset was provided
            if eval_dataset is not None:
                val_metrics = trainer.evaluate(eval_dataset=eval_dataset)
                metrics.update(
                    {
                        "val_accuracy": val_metrics.get("eval_accuracy", 0.0),
                        "val_precision": val_metrics.get("eval_precision", 0.0),
                        "val_recall": val_metrics.get("eval_recall", 0.0),
                        "val_f1": val_metrics.get("eval_f1", 0.0),
                    }
                )

        # Estimate variance components
        if self.config.mixed_effects.estimate_variance_components:
            var_comps = self.random_effects.estimate_variance_components()
            if var_comps:
                var_comp = var_comps.get("mu") or var_comps.get("slopes")
                if var_comp:
                    self.variance_history.append(var_comp)
                    metrics["participant_variance"] = var_comp.variance
                    metrics["n_participants"] = var_comp.n_groups

        # Validation metrics (already computed by Trainer if eval_dataset provided)
        if eval_dataset is not None:
            val_metrics = trainer.evaluate(eval_dataset=eval_dataset)
            metrics.update(
                {
                    "val_accuracy": val_metrics.get("eval_accuracy", 0.0),
                    "val_precision": val_metrics.get("eval_precision", 0.0),
                    "val_recall": val_metrics.get("eval_recall", 0.0),
                    "val_f1": val_metrics.get("eval_f1", 0.0),
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
        """Train using custom training loop (for random_slopes mode).

        Parameters
        ----------
        items : list[Item]
            Training items.
        y_numeric : list[float]
            Numeric labels (0.0 or 1.0).
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
        # Convert to tensor
        y = torch.tensor(y_numeric, dtype=torch.float, device=self.config.device)

        # Build optimizer parameters
        params_to_optimize = list(self.encoder.parameters()) + list(
            self.classifier_head.parameters()
        )

        # Add random effects parameters (for random_slopes)
        if self.config.mixed_effects.mode == "random_slopes":
            for head in self.random_effects.slopes.values():
                params_to_optimize.extend(head.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        self.encoder.train()
        self.classifier_head.train()

        for _epoch in range(self.config.num_epochs):
            n_batches = (
                len(items) + self.config.batch_size - 1
            ) // self.config.batch_size
            epoch_loss = 0.0
            epoch_correct = 0

            for i in range(n_batches):
                start_idx = i * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(items))

                batch_items = items[start_idx:end_idx]
                batch_labels = y[start_idx:end_idx]
                batch_participant_ids = participant_ids[start_idx:end_idx]

                embeddings = self._prepare_inputs(batch_items)

                # Random slopes: per-participant head
                logits_list = []
                for j, pid in enumerate(batch_participant_ids):
                    participant_head = self.random_effects.get_slopes(
                        pid,
                        fixed_head=self.classifier_head,
                        create_if_missing=True,
                    )
                    logits_j = participant_head(embeddings[j : j + 1]).squeeze()
                    logits_list.append(logits_j)
                logits = torch.stack(logits_list)

                # Data loss + prior regularization
                loss_bce = criterion(logits, batch_labels)
                loss_prior = self.random_effects.compute_prior_loss()
                loss = loss_bce + loss_prior

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                predictions = (torch.sigmoid(logits) > 0.5).float()
                epoch_correct += (predictions == batch_labels).sum().item()

            epoch_acc = epoch_correct / len(items)
            epoch_loss = epoch_loss / n_batches

        metrics: dict[str, float] = {
            "train_accuracy": epoch_acc,
            "train_loss": epoch_loss,
        }

        # Estimate variance components
        if self.config.mixed_effects.estimate_variance_components:
            var_comps = self.random_effects.estimate_variance_components()
            if var_comps:
                var_comp = var_comps.get("mu") or var_comps.get("slopes")
                if var_comp:
                    self.variance_history.append(var_comp)
                    metrics["participant_variance"] = var_comp.variance
                    metrics["n_participants"] = var_comp.n_groups

        # Validation
        if validation_items is not None and validation_labels is not None:
            val_predictions = self.predict(
                validation_items,
                participant_ids=["_validation_"] * len(validation_items),
            )
            val_pred_labels = [p.predicted_class for p in val_predictions]
            val_acc = sum(
                pred == true
                for pred, true in zip(val_pred_labels, validation_labels, strict=True)
            ) / len(validation_labels)
            metrics["val_accuracy"] = val_acc

        return metrics

    def _do_predict(
        self, items: list[Item], participant_ids: list[str]
    ) -> list[ModelPrediction]:
        """Perform binary model prediction.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        list[ModelPrediction]
            Predictions.
        """
        self.encoder.eval()
        self.classifier_head.eval()

        with torch.no_grad():
            embeddings = self._prepare_inputs(items)

            # Forward pass depends on mixed effects mode
            if self.config.mixed_effects.mode == "fixed":
                logits = self.classifier_head(embeddings).squeeze(1)  # (n_items,)

            elif self.config.mixed_effects.mode == "random_intercepts":
                logits = self.classifier_head(embeddings).squeeze(1)  # (n_items,)
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use prior mean (zero bias)
                    bias = self.random_effects.get_intercepts(
                        pid,
                        n_classes=self.num_classes,
                        param_name="mu",
                        create_if_missing=False,
                    )
                    logits[i] = logits[i] + bias.item()

            elif self.config.mixed_effects.mode == "random_slopes":
                logits_list = []
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use fixed head
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.classifier_head, create_if_missing=False
                    )
                    logits_i = participant_head(embeddings[i : i + 1]).squeeze()
                    logits_list.append(logits_i)
                logits = torch.stack(logits_list)

            # Compute probabilities using sigmoid
            proba_positive = torch.sigmoid(logits).cpu().numpy()  # P(y=1)
            pred_is_positive = proba_positive > 0.5

        predictions = []
        for i, item in enumerate(items):
            # Determine predicted class
            if pred_is_positive[i]:
                pred_label = self.positive_class
            else:
                pred_label = self.label_names[0]

            # Build probability dict: {negative_class: p0, positive_class: p1}
            p1 = float(proba_positive[i])
            p0 = 1.0 - p1
            prob_dict = {
                self.label_names[0]: p0,  # Negative class (first alphabetically)
                self.positive_class: p1,  # Positive class (second alphabetically)
            }

            predictions.append(
                ModelPrediction(
                    item_id=str(item.id),
                    probabilities=prob_dict,
                    predicted_class=pred_label,
                    confidence=max(p0, p1),
                )
            )

        return predictions

    def _do_predict_proba(
        self, items: list[Item], participant_ids: list[str]
    ) -> np.ndarray:
        """Perform binary model probability prediction.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_items, 2).
        """
        self.encoder.eval()
        self.classifier_head.eval()

        with torch.no_grad():
            embeddings = self._prepare_inputs(items)

            # Forward pass depends on mixed effects mode
            if self.config.mixed_effects.mode == "fixed":
                logits = self.classifier_head(embeddings).squeeze(1)

            elif self.config.mixed_effects.mode == "random_intercepts":
                logits = self.classifier_head(embeddings).squeeze(1)
                for i, pid in enumerate(participant_ids):
                    bias = self.random_effects.get_intercepts(
                        pid,
                        n_classes=self.num_classes,
                        param_name="mu",
                        create_if_missing=False,
                    )
                    logits[i] = logits[i] + bias.item()

            elif self.config.mixed_effects.mode == "random_slopes":
                logits_list = []
                for i, pid in enumerate(participant_ids):
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.classifier_head, create_if_missing=False
                    )
                    logits_i = participant_head(embeddings[i : i + 1]).squeeze()
                    logits_list.append(logits_i)
                logits = torch.stack(logits_list)

            # Compute probabilities using sigmoid
            proba_positive = torch.sigmoid(logits).cpu().numpy()  # P(y=1)

        # Return (n_items, 2) array: [P(negative), P(positive)]
        proba = np.stack([1.0 - proba_positive, proba_positive], axis=1)

        return proba

    def _get_save_state(self) -> dict[str, object]:
        """Get model-specific state to save.

        Returns
        -------
        dict[str, object]
            State dictionary.
        """
        return {
            "num_classes": self.num_classes,
            "label_names": self.label_names,
            "positive_class": self.positive_class,
        }

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
            self.classifier_head.state_dict(),
            save_path / "classifier_head.pt",
        )

    def _restore_training_state(self, config_dict: dict[str, object]) -> None:
        """Restore model-specific training state.

        Parameters
        ----------
        config_dict : dict[str, object]
            Configuration dictionary with training state.
        """
        self.num_classes = config_dict.pop("num_classes")
        self.label_names = config_dict.pop("label_names")
        self.positive_class = config_dict.pop("positive_class")

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
            from bead.active_learning.config import MixedEffectsConfig  # noqa: PLC0415

            config_dict["mixed_effects"] = MixedEffectsConfig(
                **config_dict["mixed_effects"]
            )

        self.config = BinaryModelConfig(**config_dict)

        self.encoder = AutoModel.from_pretrained(load_path / "encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "encoder")

        self._initialize_classifier()
        self.classifier_head.load_state_dict(
            torch.load(
                load_path / "classifier_head.pt", map_location=self.config.device
            )
        )
        self.classifier_head.to(self.config.device)

    def _get_random_effects_fixed_head(self) -> torch.nn.Module | None:
        """Get fixed head for random effects loading.

        Returns
        -------
        nn.Module | None
            Fixed head module.
        """
        return self.classifier_head

    def _get_n_classes_for_random_effects(self) -> int:
        """Get number of classes for random effects initialization.

        Returns
        -------
        int
            Number of classes (1 for binary).
        """
        return self.num_classes
