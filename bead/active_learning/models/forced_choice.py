"""Model for forced choice tasks (2AFC, 3AFC, 4AFC, nAFC)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from bead.active_learning.config import MixedEffectsConfig, VarianceComponents
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.active_learning.trainers.data_collator import MixedEffectsDataCollator
from bead.active_learning.trainers.dataset_utils import items_to_dataset
from bead.active_learning.trainers.metrics import compute_multiclass_metrics
from bead.active_learning.trainers.model_wrapper import EncoderClassifierWrapper
from bead.config.active_learning import ForcedChoiceModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["ForcedChoiceModel"]


class ForcedChoiceModel(ActiveLearningModel):
    """Model for forced_choice tasks with n alternatives.

    Supports 2AFC, 3AFC, 4AFC, and general nAFC tasks using any
    HuggingFace transformer model. Provides two encoding strategies:
    single encoder (concatenate options) or dual encoder (separate embeddings).

    Parameters
    ----------
    config : ForcedChoiceModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : ForcedChoiceModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Transformer tokenizer.
    encoder : AutoModel
        Transformer encoder model.
    classifier_head : nn.Sequential
        Classification head (fixed effects head).
    num_classes : int | None
        Number of classes (inferred from training data).
    option_names : list[str] | None
        Option names (e.g., ["option_a", "option_b"]).
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
    >>> from bead.config.active_learning import ForcedChoiceModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"option_a": "sentence A", "option_b": "sentence B"}
    ...     )
    ...     for _ in range(10)
    ... ]
    >>> labels = ["option_a"] * 5 + ["option_b"] * 5
    >>> config = ForcedChoiceModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = ForcedChoiceModel(config=config)  # doctest: +SKIP
    >>> metrics = model.train(items, labels)  # doctest: +SKIP
    >>> predictions = model.predict(items[:3])  # doctest: +SKIP
    """

    def __init__(
        self,
        config: ForcedChoiceModelConfig | None = None,
    ) -> None:
        """Initialize forced choice model.

        Parameters
        ----------
        config : ForcedChoiceModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or ForcedChoiceModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.encoder = AutoModel.from_pretrained(self.config.model_name)

        self.num_classes: int | None = None
        self.option_names: list[str] | None = None
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
            List containing "forced_choice".
        """
        return ["forced_choice"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with forced choice model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "forced_choice".
        ValueError
            If task_spec.options is not defined.
        ValueError
            If item is missing required rendered_elements.
        """
        if item_template.task_type != "forced_choice":
            raise ValueError(
                f"Expected task_type 'forced_choice', got '{item_template.task_type}'"
            )

        if item_template.task_spec.options is None:
            raise ValueError(
                "task_spec.options must be defined for forced_choice tasks"
            )

        for option_name in item_template.task_spec.options:
            if option_name not in item.rendered_elements:
                raise ValueError(
                    f"Item missing required element '{option_name}' "
                    f"from rendered_elements"
                )

    def _initialize_classifier(self, num_classes: int) -> None:
        """Initialize classification head for given number of classes.

        Parameters
        ----------
        num_classes : int
            Number of output classes.
        """
        hidden_size = self.encoder.config.hidden_size

        if self.config.encoder_mode == "dual_encoder":
            input_size = hidden_size * num_classes
        else:
            input_size = hidden_size

        self.classifier_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        self.classifier_head.to(self.config.device)

    def _encode_single(self, texts: list[str]) -> torch.Tensor:
        """Encode texts using single encoder strategy.

        Concatenates all option texts with [SEP] tokens and encodes once.

        Parameters
        ----------
        texts : list[str]
            List of concatenated option texts for each item.

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

    def _encode_dual(self, options_per_item: list[list[str]]) -> torch.Tensor:
        """Encode texts using dual encoder strategy.

        Encodes each option separately and concatenates embeddings.

        Parameters
        ----------
        options_per_item : list[list[str]]
            List of option lists. Each inner list contains option texts for one item.

        Returns
        -------
        torch.Tensor
            Concatenated encodings of shape (batch_size, hidden_size * num_options).
        """
        all_embeddings = []

        for options in options_per_item:
            option_embeddings = []
            for option_text in options:
                encodings = self.tokenizer(
                    [option_text],
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                encodings = {k: v.to(self.config.device) for k, v in encodings.items()}

                outputs = self.encoder(**encodings)
                cls_embedding = outputs.last_hidden_state[0, 0, :]
                option_embeddings.append(cls_embedding)

            concatenated = torch.cat(option_embeddings, dim=0)
            all_embeddings.append(concatenated)

        return torch.stack(all_embeddings)

    def _prepare_inputs(self, items: list[Item]) -> torch.Tensor:
        """Prepare inputs for encoding based on encoder mode.

        Parameters
        ----------
        items : list[Item]
            Items to encode.

        Returns
        -------
        torch.Tensor
            Encoded representations.
        """
        if self.option_names is None:
            raise ValueError("Model not initialized. Call train() first.")

        if self.config.encoder_mode == "single_encoder":
            texts = []
            for item in items:
                option_texts = [
                    item.rendered_elements.get(opt, "") for opt in self.option_names
                ]
                concatenated = " [SEP] ".join(option_texts)
                texts.append(concatenated)
            return self._encode_single(texts)
        else:
            options_per_item = []
            for item in items:
                option_texts = [
                    item.rendered_elements.get(opt, "") for opt in self.option_names
                ]
                options_per_item.append(option_texts)
            return self._encode_dual(options_per_item)

    def _validate_labels(self, labels: list[str]) -> None:
        """Validate that all labels are valid option names.

        Parameters
        ----------
        labels : list[str]
            Labels to validate.

        Raises
        ------
        ValueError
            If any label is not in option_names.
        """
        if self.option_names is None:
            raise ValueError("option_names not initialized")

        valid_labels = set(self.option_names)
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
    ) -> tuple[list[Item], list[int], list[str], list[Item] | None, list[int] | None]:
        """Prepare training data for forced choice model.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (option names).
        participant_ids : list[str]
            Normalized participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[str] | None
            Validation labels.

        Returns
        -------
        tuple[list[Item], list[int], list[str], list[Item] | None, list[int] | None]
            Prepared items, numeric labels, participant_ids, validation_items,
            numeric validation_labels.
        """
        unique_labels = sorted(set(labels))
        self.num_classes = len(unique_labels)
        self.option_names = unique_labels

        self._validate_labels(labels)
        self._initialize_classifier(self.num_classes)

        label_to_idx = {label: idx for idx, label in enumerate(self.option_names)}
        y_numeric = [label_to_idx[label] for label in labels]

        # Convert validation labels if provided
        val_y_numeric = None
        if validation_items is not None and validation_labels is not None:
            self._validate_labels(validation_labels)
            if len(validation_items) != len(validation_labels):
                raise ValueError(
                    f"Number of validation items ({len(validation_items)}) "
                    f"must match number of validation labels ({len(validation_labels)})"
                )
            val_y_numeric = [label_to_idx[label] for label in validation_labels]

        return items, y_numeric, participant_ids, validation_items, val_y_numeric

    def _initialize_random_effects(self, n_classes: int) -> None:
        """Initialize random effects manager.

        Parameters
        ----------
        n_classes : int
            Number of classes.
        """
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects, n_classes=n_classes
        )

    def _do_training(
        self,
        items: list[Item],
        labels_numeric: list[int],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels_numeric: list[int] | None,
    ) -> dict[str, float]:
        """Perform forced choice model training.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels_numeric : list[int]
            Numeric labels (class indices).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels_numeric : list[int] | None
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
                self.option_names[label_idx] for label_idx in validation_labels_numeric
            ]

        # Use HuggingFace Trainer for fixed and random_intercepts modes
        if self.config.mixed_effects.mode in ("fixed", "random_intercepts"):
            metrics = self._train_with_huggingface_trainer(
                items=items,
                y_numeric=labels_numeric,
                participant_ids=participant_ids,
                validation_items=validation_items,
                validation_labels=validation_labels,
            )
        else:
            # Use custom loop for random_slopes mode
            metrics = self._train_with_custom_loop(
                items=items,
                y_numeric=labels_numeric,
                participant_ids=participant_ids,
                validation_items=validation_items,
                validation_labels=validation_labels,
            )

        # Add validation accuracy if validation data provided and not already computed
        if (
            validation_items is not None
            and validation_labels is not None
            and "val_accuracy" not in metrics
        ):
            # Validation with placeholder participant_ids for mixed effects
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
        y_numeric: list[int],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> dict[str, float]:
        """Train using HuggingFace Trainer with mixed effects support.

        Parameters
        ----------
        items : list[Item]
            Training items.
        y_numeric : list[int]
            Numeric labels (class indices).
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
            label_to_idx = {label: idx for idx, label in enumerate(self.option_names)}
            val_y_numeric = [label_to_idx[label] for label in validation_labels]
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

        # Create wrapper model for Trainer
        wrapped_model = EncoderClassifierWrapper(
            encoder=self.encoder, classifier_head=self.classifier_head
        )

        # Create data collator
        data_collator = MixedEffectsDataCollator(tokenizer=self.tokenizer)

        # Create metrics computation function
        def compute_metrics_fn(eval_pred: object) -> dict[str, float]:
            return compute_multiclass_metrics(eval_pred, num_labels=self.num_classes)

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
                compute_metrics=compute_metrics_fn,
            )

            # Train
            train_result = trainer.train()

            # Get training metrics
            train_metrics = trainer.evaluate(eval_dataset=train_dataset)
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

        return metrics

    def _train_with_custom_loop(
        self,
        items: list[Item],
        y_numeric: list[int],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> dict[str, float]:
        """Train using custom training loop (for random_slopes mode).

        Parameters
        ----------
        items : list[Item]
            Training items.
        y_numeric : list[int]
            Numeric labels (class indices).
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
        y = torch.tensor(y_numeric, dtype=torch.long, device=self.config.device)

        # Build optimizer parameters
        params_to_optimize = list(self.encoder.parameters()) + list(
            self.classifier_head.parameters()
        )

        # Add random effects parameters (for random_slopes)
        if self.config.mixed_effects.mode == "random_slopes":
            for head in self.random_effects.slopes.values():
                params_to_optimize.extend(head.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

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

                # Forward pass depends on mixed effects mode
                if self.config.mixed_effects.mode == "fixed":
                    # Standard forward pass
                    logits = self.classifier_head(embeddings)

                elif self.config.mixed_effects.mode == "random_intercepts":
                    # Fixed head + per-participant bias
                    logits = self.classifier_head(embeddings)
                    for j, pid in enumerate(batch_participant_ids):
                        bias = self.random_effects.get_intercepts(
                            pid,
                            n_classes=self.num_classes,
                            param_name="mu",
                            create_if_missing=True,
                        )
                        logits[j] = logits[j] + bias

                elif self.config.mixed_effects.mode == "random_slopes":
                    # Per-participant head
                    logits_list = []
                    for j, pid in enumerate(batch_participant_ids):
                        participant_head = self.random_effects.get_slopes(
                            pid,
                            fixed_head=self.classifier_head,
                            create_if_missing=True,
                        )
                        logits_j = participant_head(embeddings[j : j + 1])
                        logits_list.append(logits_j)
                    logits = torch.cat(logits_list, dim=0)

                # Data loss + prior regularization
                loss_ce = criterion(logits, batch_labels)
                loss_prior = self.random_effects.compute_prior_loss()
                loss = loss_ce + loss_prior

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                epoch_correct += (predictions == batch_labels).sum().item()

            epoch_acc = epoch_correct / len(items)
            epoch_loss = epoch_loss / n_batches

        self._is_fitted = True

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

        if validation_items is not None and validation_labels is not None:
            self._validate_labels(validation_labels)

            if len(validation_items) != len(validation_labels):
                raise ValueError(
                    f"Number of validation items ({len(validation_items)}) "
                    f"must match number of validation labels ({len(validation_labels)})"
                )

            # Validation with placeholder participant_ids for mixed effects
            if self.config.mixed_effects.mode == "fixed":
                val_predictions = self.predict(validation_items, participant_ids=None)
            else:
                val_participant_ids = ["_validation_"] * len(validation_items)
                val_predictions = self.predict(
                    validation_items, participant_ids=val_participant_ids
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
        """Perform forced choice model prediction.

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
                logits = self.classifier_head(embeddings)

            elif self.config.mixed_effects.mode == "random_intercepts":
                logits = self.classifier_head(embeddings)
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use prior mean (zero bias)
                    bias = self.random_effects.get_intercepts(
                        pid,
                        n_classes=self.num_classes,
                        param_name="mu",
                        create_if_missing=False,
                    )
                    logits[i] = logits[i] + bias

            elif self.config.mixed_effects.mode == "random_slopes":
                logits_list = []
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use fixed head
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.classifier_head, create_if_missing=False
                    )
                    logits_i = participant_head(embeddings[i : i + 1])
                    logits_list.append(logits_i)
                logits = torch.cat(logits_list, dim=0)

            proba = torch.softmax(logits, dim=1).cpu().numpy()
            pred_classes = torch.argmax(logits, dim=1).cpu().numpy()

        predictions = []
        for i, item in enumerate(items):
            pred_label = self.option_names[pred_classes[i]]
            prob_dict = {
                opt: float(proba[i, idx]) for idx, opt in enumerate(self.option_names)
            }
            predictions.append(
                ModelPrediction(
                    item_id=str(item.id),
                    probabilities=prob_dict,
                    predicted_class=pred_label,
                    confidence=float(proba[i, pred_classes[i]]),
                )
            )

        return predictions

    def _do_predict_proba(
        self, items: list[Item], participant_ids: list[str]
    ) -> np.ndarray:
        """Perform forced choice model probability prediction.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_items, n_classes).
        """
        self.encoder.eval()
        self.classifier_head.eval()

        with torch.no_grad():
            embeddings = self._prepare_inputs(items)

            # Forward pass depends on mixed effects mode
            if self.config.mixed_effects.mode == "fixed":
                logits = self.classifier_head(embeddings)

            elif self.config.mixed_effects.mode == "random_intercepts":
                logits = self.classifier_head(embeddings)
                for i, pid in enumerate(participant_ids):
                    bias = self.random_effects.get_intercepts(
                        pid,
                        n_classes=self.num_classes,
                        param_name="mu",
                        create_if_missing=False,
                    )
                    logits[i] = logits[i] + bias

            elif self.config.mixed_effects.mode == "random_slopes":
                logits_list = []
                for i, pid in enumerate(participant_ids):
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.classifier_head, create_if_missing=False
                    )
                    logits_i = participant_head(embeddings[i : i + 1])
                    logits_list.append(logits_i)
                logits = torch.cat(logits_list, dim=0)

            proba = torch.softmax(logits, dim=1).cpu().numpy()

        return proba

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

    def _get_save_state(self) -> dict[str, object]:
        """Get model-specific state to save.

        Returns
        -------
        dict[str, object]
            State dictionary.
        """
        return {
            "num_classes": self.num_classes,
            "option_names": self.option_names,
        }

    def _restore_training_state(self, config_dict: dict[str, object]) -> None:
        """Restore model-specific training state.

        Parameters
        ----------
        config_dict : dict[str, object]
            Configuration dictionary with training state.
        """
        self.num_classes = config_dict.pop("num_classes")
        self.option_names = config_dict.pop("option_names")

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

        self.config = ForcedChoiceModelConfig(**config_dict)

        self.encoder = AutoModel.from_pretrained(load_path / "encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "encoder")

        self._initialize_classifier(self.num_classes)
        self.classifier_head.load_state_dict(
            torch.load(
                load_path / "classifier_head.pt", map_location=self.config.device
            )
        )

        self.encoder.to(self.config.device)
        self.classifier_head.to(self.config.device)

    def _get_n_classes_for_random_effects(self) -> int:
        """Get the number of classes for initializing RandomEffectsManager.

        Returns
        -------
        int
            Number of classes.
        """
        return self.num_classes

    def _get_random_effects_fixed_head(self) -> torch.nn.Module | None:
        """Get the fixed head for random effects.

        Returns
        -------
        torch.nn.Module | None
            The classifier head, or None if not applicable.
        """
        return self.classifier_head
