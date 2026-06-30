"""Cloze model for fill-in-the-blank tasks with GLMM support.

Implements masked language modeling with participant-level random effects for
predicting tokens at unfilled slots in partially-filled templates. Supports
three modes: fixed effects, random intercepts, random slopes.

Architecture: Masked LM (BERT/RoBERTa) for token prediction
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments

from bead.active_learning.config import VarianceComponents
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.active_learning.trainers.data_collator import ClozeDataCollator
from bead.active_learning.trainers.dataset_utils import cloze_items_to_dataset
from bead.config.active_learning import ClozeModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["ClozeModel"]


class ClozeModel(ActiveLearningModel):
    """Model for cloze tasks with participant-level random effects.

    Uses masked language modeling (BERT/RoBERTa) to predict tokens at unfilled
    slots in partially-filled templates. Supports three GLMM modes:
    - Fixed effects: Standard MLM
    - Random intercepts: Participant-specific bias on output logits
    - Random slopes: Participant-specific MLM heads

    Parameters
    ----------
    config : ClozeModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : ClozeModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Masked LM tokenizer.
    model : AutoModelForMaskedLM
        Masked language model (BERT or RoBERTa).
    encoder : nn.Module
        Encoder module from the model.
    mlm_head : nn.Module
        MLM prediction head.
    random_effects : RandomEffectsManager
        Manager for participant-level random effects.
    variance_history : list[VarianceComponents]
        Variance component estimates over training.
    _is_fitted : bool
        Whether model has been trained.

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.items.item import Item, UnfilledSlot
    >>> from bead.config.active_learning import ClozeModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"text": "The cat ___."},
    ...         unfilled_slots=[
    ...             UnfilledSlot(slot_name="verb", position=2, constraint_ids=[])
    ...         ]
    ...     )
    ...     for _ in range(6)
    ... ]
    >>> labels = [["ran"], ["jumped"], ["slept"]] * 2  # One token per unfilled slot
    >>> config = ClozeModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = ClozeModel(config=config)  # doctest: +SKIP
    >>> metrics = model.train(items, labels, participant_ids=None)  # doctest: +SKIP
    """

    def __init__(
        self,
        config: ClozeModelConfig | None = None,
    ) -> None:
        """Initialize cloze model.

        Parameters
        ----------
        config : ClozeModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or ClozeModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        # Load tokenizer and masked LM model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)

        # Extract encoder and MLM head
        # BERT-style models use model.bert and model.cls
        # RoBERTa-style models use model.roberta and model.lm_head
        if hasattr(self.model, "bert"):
            self.encoder = self.model.bert
            self.mlm_head = self.model.cls
        elif hasattr(self.model, "roberta"):
            self.encoder = self.model.roberta
            self.mlm_head = self.model.lm_head
        else:
            # Fallback: try to use the base model attribute
            self.encoder = self.model.base_model
            self.mlm_head = self.model.lm_head

        self._is_fitted = False

        # Initialize random effects manager (created during training)
        self.random_effects: RandomEffectsManager | None = None
        self.variance_history: list[VarianceComponents] = []

        self.model.to(self.config.device)

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "cloze".
        """
        return ["cloze"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with cloze model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "cloze".
        ValueError
            If item has no unfilled_slots.
        """
        if item_template.task_type != "cloze":
            raise ValueError(
                f"Expected task_type 'cloze', got '{item_template.task_type}'"
            )

        if not item.unfilled_slots:
            raise ValueError(
                "Cloze items must have at least one unfilled slot. "
                f"Item {item.id} has no unfilled_slots."
            )

    def _prepare_inputs_and_masks(
        self, items: list[Item]
    ) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
        """Prepare tokenized inputs with masked positions.

        Extracts text from items, tokenizes, and replaces tokens at unfilled_slots
        positions with [MASK] token.

        Parameters
        ----------
        items : list[Item]
            Items to prepare.

        Returns
        -------
        tuple[dict[str, torch.Tensor], list[list[int]]]
            - Tokenized inputs (input_ids, attention_mask)
            - List of masked token positions per item (token-level indices)
        """
        texts = []
        n_slots_per_item = []

        for item in items:
            # Get rendered text
            text = item.rendered_elements.get("text", "")
            texts.append(text)
            n_slots_per_item.append(len(item.unfilled_slots))

        # Tokenize all texts
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.config.device)

        mask_token_id = self.tokenizer.mask_token_id

        # Find and replace "___" placeholders with [MASK]
        # Track ONE position per unfilled slot (even if "___" spans multiple tokens)
        token_masked_positions = []
        for i, text in enumerate(texts):
            # Tokenize individually to find "___" positions
            tokens = self.tokenizer.tokenize(text)
            masked_indices = []

            # Track which tokens are part of "___" to avoid duplicates
            in_blank = False
            for j, token in enumerate(tokens):
                # Check if this token is part of a "___" placeholder
                if "_" in token and not in_blank:
                    # Start of a new blank - record this position
                    token_idx = j + 1  # Add 1 for [CLS] token
                    masked_indices.append(token_idx)
                    in_blank = True
                    # Replace with [MASK]
                    if token_idx < tokenized["input_ids"].shape[1]:
                        tokenized["input_ids"][i, token_idx] = mask_token_id
                elif "_" in token and in_blank:
                    # Continuation of current blank - also mask but don't record
                    token_idx = j + 1
                    if token_idx < tokenized["input_ids"].shape[1]:
                        tokenized["input_ids"][i, token_idx] = mask_token_id
                else:
                    # Not a blank token - reset in_blank
                    in_blank = False

            # Verify we found the expected number of masked positions
            expected_slots = n_slots_per_item[i]
            if len(masked_indices) != expected_slots:
                raise ValueError(
                    f"Mismatch between masked positions and unfilled_slots "
                    f"for item {i}: found {len(masked_indices)} '___' "
                    f"placeholders in text but item has {expected_slots} "
                    f"unfilled_slots. Ensure rendered text uses exactly one "
                    f"'___' per unfilled_slot. Text: '{text}'"
                )

            token_masked_positions.append(masked_indices)

        return tokenized, token_masked_positions

    def _prepare_training_data(
        self,
        items: list[Item],
        labels: list[str] | list[list[str]],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | list[list[str]] | None,
    ) -> tuple[
        list[Item],
        list[list[str]],
        list[Item] | None,
        list[list[str]] | None,
    ]:
        """Prepare data for training, including validation of label format.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[list[str]]
            Training labels as list of lists (one token per unfilled slot).
        participant_ids : list[str]
            Participant IDs (already normalized).
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[list[str]] | None
            Validation labels.

        Returns
        -------
        tuple[list[Item], list[list[str]], list[Item] | None, list[list[str]] | None]
            Prepared items, labels, validation items, validation labels.
        """
        # Validate labels format: each label must be a list matching unfilled_slots
        labels_list = list(labels)  # Type: list[list[str]]
        for i, (item, label) in enumerate(zip(items, labels_list, strict=True)):
            if not isinstance(label, list):
                raise ValueError(
                    f"ClozeModel requires labels to be list[list[str]], "
                    f"but got {type(label)} for item {i}"
                )
            if len(label) != len(item.unfilled_slots):
                raise ValueError(
                    f"Label length mismatch for item {i}: "
                    f"expected {len(item.unfilled_slots)} tokens "
                    f"(matching unfilled_slots), got {len(label)} tokens. "
                    f"Ensure each label is a list with one token per unfilled slot."
                )

        val_labels_list: list[list[str]] | None = None
        if validation_items is not None and validation_labels is not None:
            val_labels_list = list(validation_labels)  # Type: list[list[str]]
            for i, (item, label) in enumerate(
                zip(validation_items, val_labels_list, strict=True)
            ):
                if not isinstance(label, list):
                    raise ValueError(
                        f"ClozeModel requires validation_labels to be list[list[str]], "
                        f"but got {type(label)} for validation item {i}"
                    )
                if len(label) != len(item.unfilled_slots):
                    raise ValueError(
                        f"Validation label length mismatch for item {i}: "
                        f"expected {len(item.unfilled_slots)} tokens, "
                        f"got {len(label)} tokens."
                    )

        return items, labels_list, participant_ids, validation_items, val_labels_list

    def _do_training(
        self,
        items: list[Item],
        labels_numeric: list[list[str]],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels_numeric: list[list[str]] | None,
    ) -> dict[str, float]:
        """Perform the actual training logic (HuggingFace Trainer or custom loop).

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels_numeric : list[list[str]]
            Training labels (already validated).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels_numeric : list[list[str]] | None
            Validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        # Use HuggingFace Trainer for fixed and random_intercepts modes
        # random_slopes requires custom loop due to per-participant MLM heads
        use_huggingface_trainer = self.config.mixed_effects.mode in (
            "fixed",
            "random_intercepts",
        )

        if use_huggingface_trainer:
            return self._train_with_huggingface_trainer(
                items,
                labels_numeric,
                participant_ids,
                validation_items,
                validation_labels_numeric,
            )
        else:
            # Use custom training loop for random_slopes
            return self._train_with_custom_loop(
                items,
                labels_numeric,
                participant_ids,
                validation_items,
                validation_labels_numeric,
            )

    def _train_with_huggingface_trainer(
        self,
        items: list[Item],
        labels: list[list[str]],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[list[str]] | None,
    ) -> dict[str, float]:
        """Train using HuggingFace Trainer with mixed effects support for MLM.

        Parameters
        ----------
        items : list[Item]
            Training items with unfilled_slots.
        labels : list[list[str]]
            Training labels as list of lists (one token per unfilled slot).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[list[str]] | None
            Validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        # Convert items to HuggingFace Dataset with masking
        train_dataset = cloze_items_to_dataset(
            items=items,
            labels=labels,
            participant_ids=participant_ids,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        eval_dataset = None
        if validation_items is not None and validation_labels is not None:
            val_participant_ids = (
                ["_validation_"] * len(validation_items)
                if self.config.mixed_effects.mode != "fixed"
                else ["_fixed_"] * len(validation_items)
            )
            eval_dataset = cloze_items_to_dataset(
                items=validation_items,
                labels=validation_labels,
                participant_ids=val_participant_ids,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )

        # Use the model directly (no wrapper needed for MLM models)
        # The model is already compatible with HuggingFace Trainer
        wrapped_model = self.model

        # Create data collator
        data_collator = ClozeDataCollator(tokenizer=self.tokenizer)

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

            # Create metrics computation function
            def compute_metrics_fn(eval_pred: object) -> dict[str, float]:
                from bead.active_learning.trainers.metrics import (  # noqa: PLC0415
                    compute_cloze_metrics,
                )

                return compute_cloze_metrics(eval_pred, tokenizer=self.tokenizer)

            # Import here to avoid circular import
            from bead.active_learning.trainers.mixed_effects import (  # noqa: PLC0415
                ClozeMLMTrainer,
            )

            # Create trainer
            trainer = ClozeMLMTrainer(
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
            }

            # Get validation metrics if eval_dataset was provided
            if eval_dataset is not None:
                val_metrics = trainer.evaluate(eval_dataset=eval_dataset)
                metrics["val_accuracy"] = val_metrics.get("eval_accuracy", 0.0)

        return metrics

    def _train_with_custom_loop(
        self,
        items: list[Item],
        labels: list[list[str]],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[list[str]] | None,
    ) -> dict[str, float]:
        """Train using custom loop for random_slopes mode.

        Parameters
        ----------
        items : list[Item]
            Training items with unfilled_slots.
        labels : list[list[str]]
            Training labels as list of lists.
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[list[str]] | None
            Validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        # Build optimizer parameters
        params_to_optimize = list(self.model.parameters())

        # Add random effects parameters for random_slopes
        for head in self.random_effects.slopes.values():
            params_to_optimize.extend(head.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config.learning_rate)

        self.model.train()

        for _epoch in range(self.config.num_epochs):
            n_batches = (
                len(items) + self.config.batch_size - 1
            ) // self.config.batch_size
            epoch_loss = 0.0

            for i in range(n_batches):
                start_idx = i * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(items))

                batch_items = items[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                batch_participant_ids = participant_ids[start_idx:end_idx]

                # Prepare inputs with masking
                tokenized, masked_positions = self._prepare_inputs_and_masks(
                    batch_items
                )

                # Tokenize labels to get target token IDs
                target_token_ids = []
                for label_list in batch_labels:
                    token_ids = []
                    for token in label_list:
                        tid = self.tokenizer.encode(token, add_special_tokens=False)[0]
                        token_ids.append(tid)
                    target_token_ids.append(token_ids)

                # Use participant-specific MLM heads for random_slopes
                all_logits = []
                for j, pid in enumerate(batch_participant_ids):
                    # Get participant-specific MLM head
                    participant_head = self.random_effects.get_slopes(
                        pid,
                        fixed_head=copy.deepcopy(self.mlm_head),
                        create_if_missing=True,
                    )

                    # Get encoder outputs for this item
                    item_inputs = {k: v[j : j + 1] for k, v in tokenized.items()}
                    encoder_outputs_j = self.encoder(**item_inputs)

                    # Run participant-specific MLM head
                    logits_j = participant_head(encoder_outputs_j.last_hidden_state)
                    all_logits.append(logits_j)

                logits = torch.cat(all_logits, dim=0)

                # Compute loss only on masked positions
                losses = []
                for j, (masked_pos, target_ids) in enumerate(
                    zip(masked_positions, target_token_ids, strict=True)
                ):
                    for pos, target_id in zip(masked_pos, target_ids, strict=True):
                        if pos < logits.shape[1]:
                            loss_j = torch.nn.functional.cross_entropy(
                                logits[j, pos : pos + 1],
                                torch.tensor([target_id], device=self.config.device),
                            )
                            losses.append(loss_j)

                if losses:
                    loss_nll = torch.stack(losses).mean()
                else:
                    loss_nll = torch.tensor(0.0, device=self.config.device)

                # Add prior regularization
                loss_prior = self.random_effects.compute_prior_loss()
                loss = loss_nll + loss_prior

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / n_batches

        metrics: dict[str, float] = {
            "train_loss": epoch_loss,
        }

        # Compute training accuracy
        train_predictions = self._do_predict(items, participant_ids)
        correct = 0
        total = 0
        for pred, label in zip(train_predictions, labels, strict=True):
            # pred.predicted_class is comma-separated tokens
            pred_tokens = pred.predicted_class.split(", ")
            for pt, lt in zip(pred_tokens, label, strict=True):
                if pt.lower() == lt.lower():
                    correct += 1
                total += 1
        if total > 0:
            metrics["train_accuracy"] = correct / total

        return metrics

    def _do_predict(
        self, items: list[Item], participant_ids: list[str]
    ) -> list[ModelPrediction]:
        """Perform cloze model prediction.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        list[ModelPrediction]
            Predictions with predicted_class as comma-separated tokens.
        """
        self.model.eval()

        # Prepare inputs with masking
        tokenized, masked_positions = self._prepare_inputs_and_masks(items)

        with torch.no_grad():
            if self.config.mixed_effects.mode == "fixed":
                # Standard MLM prediction
                outputs = self.model(**tokenized)
                logits = outputs.logits

            elif self.config.mixed_effects.mode == "random_intercepts":
                # Get encoder outputs
                encoder_outputs = self.encoder(**tokenized)
                logits = self.mlm_head(encoder_outputs.last_hidden_state)

                # Add participant-specific bias
                vocab_size = self.tokenizer.vocab_size
                for j, pid in enumerate(participant_ids):
                    bias = self.random_effects.get_intercepts(
                        pid,
                        n_classes=vocab_size,
                        param_name="mu",
                        create_if_missing=False,
                    )
                    # Add to all masked positions
                    for pos in masked_positions[j]:
                        if pos < logits.shape[1]:
                            logits[j, pos] = logits[j, pos] + bias

            elif self.config.mixed_effects.mode == "random_slopes":
                # Use participant-specific MLM heads
                all_logits = []
                for j, pid in enumerate(participant_ids):
                    # Get participant-specific MLM head
                    participant_head = self.random_effects.get_slopes(
                        pid,
                        fixed_head=copy.deepcopy(self.mlm_head),
                        create_if_missing=False,
                    )

                    # Get encoder outputs
                    item_inputs = {k: v[j : j + 1] for k, v in tokenized.items()}
                    encoder_outputs_j = self.encoder(**item_inputs)

                    # Run participant-specific MLM head
                    logits_j = participant_head(encoder_outputs_j.last_hidden_state)
                    all_logits.append(logits_j)

                logits = torch.cat(all_logits, dim=0)

            # Get argmax at masked positions
            predictions = []
            for i, masked_pos in enumerate(masked_positions):
                predicted_tokens = []
                for pos in masked_pos:
                    if pos < logits.shape[1]:
                        # Get token ID with highest probability
                        token_id = torch.argmax(logits[i, pos]).item()
                        # Decode token
                        token = self.tokenizer.decode([token_id])
                        predicted_tokens.append(token.strip())

                # Join with comma for multi-slot items
                predicted_class = ", ".join(predicted_tokens)

                predictions.append(
                    ModelPrediction(
                        item_id=str(items[i].id),
                        probabilities={},  # Not applicable for generation
                        predicted_class=predicted_class,
                        confidence=1.0,  # Not applicable for generation
                    )
                )

        return predictions

    def _do_predict_proba(
        self, items: list[Item], participant_ids: list[str]
    ) -> np.ndarray:
        """Perform cloze model probability prediction.

        For cloze tasks, returns empty array as probabilities are not typically
        used for evaluation.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        np.ndarray
            Empty array of shape (n_items, 0).
        """
        return np.zeros((len(items), 0))

    def _save_model_components(self, save_path: Path) -> None:
        """Save model-specific components (model, tokenizer).

        Parameters
        ----------
        save_path : Path
            Directory to save to.
        """
        self.model.save_pretrained(save_path / "model")
        self.tokenizer.save_pretrained(save_path / "model")

    def _get_save_state(self) -> dict[str, object]:
        """Get model-specific state to save in config.json.

        Returns
        -------
        dict[str, object]
            State dictionary to include in config.json.
        """
        return {}

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

        # Reconstruct ClozeModelConfig
        self.config = ClozeModelConfig(**config_dict)

        # Load model
        self.model = AutoModelForMaskedLM.from_pretrained(load_path / "model")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "model")

        # Re-extract components
        if hasattr(self.model, "bert"):
            self.encoder = self.model.bert
            self.mlm_head = self.model.cls
        elif hasattr(self.model, "roberta"):
            self.encoder = self.model.roberta
            self.mlm_head = self.model.lm_head
        else:
            self.encoder = self.model.base_model
            self.mlm_head = self.model.lm_head

        self.model.to(self.config.device)

    def _restore_training_state(self, config_dict: dict[str, object]) -> None:
        """Restore model-specific training state from config_dict.

        Parameters
        ----------
        config_dict : dict[str, object]
            Configuration dictionary with training state.
        """
        # ClozeModel doesn't have additional training state to restore
        pass

    def _get_n_classes_for_random_effects(self) -> int:
        """Get the number of classes for initializing RandomEffectsManager.

        For cloze models, this is the vocabulary size.

        Returns
        -------
        int
            Vocabulary size.
        """
        return self.tokenizer.vocab_size

    def _initialize_random_effects(self, n_classes: int) -> None:
        """Initialize the RandomEffectsManager.

        Parameters
        ----------
        n_classes : int
            Vocabulary size for cloze models.
        """
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects,
            vocab_size=n_classes,  # For random intercepts (bias on logits)
        )

    def _get_random_effects_fixed_head(self) -> torch.nn.Module | None:
        """Get the fixed head for random effects (classifier_head, etc.).

        For cloze models, this is the MLM head.

        Returns
        -------
        torch.nn.Module | None
            The MLM head, or None if not applicable.
        """
        return self.mlm_head
