"""Free text model for open-ended text generation with GLMM support.

Implements seq2seq generation with participant-level random effects using:
- Random intercepts: Bias on decoder output logits (token probability shifts)
- Random slopes: LoRA adapters on decoder attention layers

Architecture: T5-base or BART-base encoder-decoder model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from bead.active_learning.config import VarianceComponents
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.peft_adapter import create_participant_lora_adapter
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.config.active_learning import FreeTextModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["FreeTextModel"]


class FreeTextModel(ActiveLearningModel):
    """Model for free_text tasks with participant-level random effects.

    Uses seq2seq architecture (T5 or BART) with three modes:
    - Fixed effects: Standard encoder-decoder
    - Random intercepts: Participant-specific bias on output logits
    - Random slopes: Participant-specific LoRA adapters on decoder

    Parameters
    ----------
    config : FreeTextModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : FreeTextModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Seq2seq tokenizer.
    model : AutoModelForSeq2SeqLM
        Base seq2seq model (T5 or BART).
    encoder : nn.Module
        Encoder module.
    base_decoder : nn.Module
        Base decoder module (shared across participants in fixed/random_intercepts).
    lm_head : nn.Module
        Language modeling head (projects decoder output to vocabulary).
    random_effects : RandomEffectsManager
        Manager for participant-level random effects.
    variance_history : list[VarianceComponents]
        Variance component estimates over training.
    _is_fitted : bool
        Whether model has been trained.

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.items.item import Item
    >>> from bead.config.active_learning import FreeTextModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"prompt": "Summarize: The cat sat."}
    ...     )
    ...     for _ in range(10)
    ... ]
    >>> labels = ["Cat sits."] * 10
    >>> config = FreeTextModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = FreeTextModel(config=config)  # doctest: +SKIP
    >>> metrics = model.train(items, labels, participant_ids=None)  # doctest: +SKIP
    """

    def __init__(
        self,
        config: FreeTextModelConfig | None = None,
    ) -> None:
        """Initialize free text model.

        Parameters
        ----------
        config : FreeTextModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or FreeTextModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

        # Extract encoder, decoder, and lm_head for fine-grained control
        self.encoder = self.model.get_encoder()
        self.base_decoder = self.model.get_decoder()
        self.lm_head = self.model.lm_head

        self._is_fitted = False

        # Initialize random effects manager
        self.random_effects: RandomEffectsManager | None = None
        self.variance_history: list[VarianceComponents] = []

        self.model.to(self.config.device)

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "free_text".
        """
        return ["free_text"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with free text model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "free_text".
        """
        if item_template.task_type != "free_text":
            raise ValueError(
                f"Expected task_type 'free_text', got '{item_template.task_type}'"
            )

    def _prepare_inputs(self, items: list[Item]) -> str:
        """Prepare input texts from items.

        For free text tasks, concatenates all rendered elements as prompt.

        Parameters
        ----------
        items : list[Item]
            Items to encode.

        Returns
        -------
        list[str]
            Input texts.
        """
        texts = []
        for item in items:
            # Concatenate all rendered elements as input
            text = " ".join(item.rendered_elements.values())
            texts.append(text)
        return texts

    def _prepare_training_data(
        self,
        items: list[Item],
        labels: list[str],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> tuple[
        list[Item],
        list[str],
        list[str],
        list[Item] | None,
        list[str] | None,
    ]:
        """Prepare data for training, including validation.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (target text strings).
        participant_ids : list[str]
            Participant identifiers.
        validation_items : list[Item] | None
            Optional validation items.
        validation_labels : list[str] | None
            Optional validation labels.

        Returns
        -------
        tuple
            Prepared training data: items, labels, participant_ids,
            validation_items, validation_labels.

        Raises
        ------
        ValueError
            If labels contain empty strings.
        """
        if any(not label for label in labels):
            raise ValueError(
                "labels cannot contain empty strings. "
                "Ensure all labels are non-empty text."
            )

        val_labels_list: list[str] | None = None
        if validation_items is not None and validation_labels is not None:
            if any(not label for label in validation_labels):
                raise ValueError(
                    "validation_labels cannot contain empty strings. "
                    "Ensure all validation labels are non-empty text."
                )
            val_labels_list = validation_labels

        return items, labels, participant_ids, validation_items, val_labels_list

    def _do_training(
        self,
        items: list[Item],
        labels_numeric: list[str],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels_numeric: list[str] | None,
    ) -> dict[str, float]:
        """Perform the actual training logic (custom loop for seq2seq).

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels_numeric : list[str]
            Training labels (target text strings).
        participant_ids : list[str]
            Participant identifiers.
        validation_items : list[Item] | None
            Optional validation items.
        validation_labels_numeric : list[str] | None
            Optional validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        # Prepare inputs
        input_texts = self._prepare_inputs(items)

        # Get actual vocabulary size from lm_head output dimension
        vocab_size = self.lm_head.out_features

        # Build optimizer parameters based on mode
        params_to_optimize = list(self.model.parameters())

        # Add random effects parameters
        if self.config.mixed_effects.mode == "random_intercepts":
            for param_dict in self.random_effects.intercepts.values():
                params_to_optimize.extend(param_dict.values())
        elif self.config.mixed_effects.mode == "random_slopes":
            for adapter in self.random_effects.slopes.values():
                params_to_optimize.extend(adapter.get_lora_parameters())

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

                batch_input_texts = input_texts[start_idx:end_idx]
                batch_labels = labels_numeric[start_idx:end_idx]
                batch_participant_ids = participant_ids[start_idx:end_idx]

                # Tokenize inputs and labels
                inputs = self.tokenizer(
                    batch_input_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_input_length,
                    return_tensors="pt",
                ).to(self.config.device)

                # Tokenize targets (labels)
                targets = self.tokenizer(
                    text_target=batch_labels,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_output_length,
                    return_tensors="pt",
                ).to(self.config.device)

                target_ids = targets["input_ids"]
                # Replace pad token id with -100 for loss computation
                target_ids[target_ids == self.tokenizer.pad_token_id] = -100

                # Forward pass depends on mixed effects mode
                if self.config.mixed_effects.mode == "fixed":
                    # Standard seq2seq training
                    outputs = self.model(
                        **inputs,
                        labels=target_ids,
                    )
                    loss_nll = outputs.loss

                elif self.config.mixed_effects.mode == "random_intercepts":
                    # Get encoder outputs
                    encoder_outputs = self.encoder(**inputs)

                    # Run decoder to get logits
                    decoder_outputs = self.base_decoder(
                        input_ids=targets["input_ids"],
                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                        encoder_attention_mask=inputs["attention_mask"],
                    )

                    # Project to vocabulary
                    logits = self.lm_head(decoder_outputs.last_hidden_state)

                    # Add participant-specific bias to logits
                    for j, pid in enumerate(batch_participant_ids):
                        bias = self.random_effects.get_intercepts(
                            pid,
                            n_classes=vocab_size,
                            param_name="mu",
                            create_if_missing=True,
                        )
                        # bias shape: (vocab_size,)
                        # Add to all positions in sequence
                        logits[j] = logits[j] + bias

                    # Compute cross-entropy loss
                    loss_nll = torch.nn.functional.cross_entropy(
                        logits.view(-1, vocab_size),
                        target_ids.view(-1),
                        ignore_index=-100,
                    )

                elif self.config.mixed_effects.mode == "random_slopes":
                    # Use participant-specific LoRA adapters
                    # Need to process each participant separately
                    losses = []
                    for j, pid in enumerate(batch_participant_ids):
                        # Get participant-specific decoder
                        participant_decoder = self.random_effects.get_slopes(
                            pid,
                            fixed_head=create_participant_lora_adapter(
                                self.base_decoder,
                                rank=self.config.lora_rank,
                                alpha=self.config.lora_alpha,
                                dropout=self.config.lora_dropout,
                                target_modules=self.config.lora_target_modules,
                            ),
                            create_if_missing=True,
                        )

                        # Get encoder outputs for this item
                        item_inputs = {k: v[j : j + 1] for k, v in inputs.items()}
                        encoder_outputs_j = self.encoder(**item_inputs)

                        # Run participant-specific decoder
                        decoder_outputs_j = participant_decoder(
                            input_ids=targets["input_ids"][j : j + 1],
                            encoder_hidden_states=encoder_outputs_j.last_hidden_state,
                            encoder_attention_mask=item_inputs["attention_mask"],
                        )

                        # Project to vocabulary
                        logits_j = self.lm_head(decoder_outputs_j.last_hidden_state)

                        # Compute loss for this item
                        loss_j = torch.nn.functional.cross_entropy(
                            logits_j.view(-1, vocab_size),
                            target_ids[j : j + 1].view(-1),
                            ignore_index=-100,
                        )
                        losses.append(loss_j)

                    loss_nll = torch.stack(losses).mean()

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

        # Estimate variance components
        if self.config.mixed_effects.estimate_variance_components:
            var_comps = self.random_effects.estimate_variance_components()
            if var_comps:
                var_comp = var_comps.get("mu") or var_comps.get("slopes")
                if var_comp:
                    if not hasattr(self, "variance_history"):
                        self.variance_history = []
                    self.variance_history.append(var_comp)
                    metrics["participant_variance"] = var_comp.variance
                    metrics["n_participants"] = var_comp.n_groups

        # Compute training exact match
        train_predictions = self._do_predict(items, participant_ids)
        train_pred_texts = [p.predicted_class for p in train_predictions]
        metrics["train_exact_match"] = self._compute_exact_match(
            train_pred_texts, labels_numeric
        )

        if validation_items is not None and validation_labels_numeric is not None:
            # Validation
            if self.config.mixed_effects.mode == "fixed":
                val_participant_ids = ["_fixed_"] * len(validation_items)
            else:
                val_participant_ids = ["_validation_"] * len(validation_items)
            val_predictions = self._do_predict(validation_items, val_participant_ids)

            val_pred_texts = [p.predicted_class for p in val_predictions]
            metrics["val_exact_match"] = self._compute_exact_match(
                val_pred_texts, validation_labels_numeric
            )

        return metrics

    def _do_predict(
        self, items: list[Item], participant_ids: list[str]
    ) -> list[ModelPrediction]:
        """Generate text for items with participant-specific random effects.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Participant identifiers.

        Returns
        -------
        list[ModelPrediction]
            Predictions with predicted_class as generated text.
        """
        self.model.eval()

        input_texts = self._prepare_inputs(items)

        # Tokenize inputs
        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            if self.config.mixed_effects.mode == "fixed":
                # Standard generation
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_output_length,
                    num_beams=self.config.num_beams,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                generated_texts = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )

            elif self.config.mixed_effects.mode == "random_intercepts":
                # Generate with participant-specific bias
                # For simplicity, use greedy decoding with bias applied at each step
                # (Full beam search with bias is more complex)
                generated_texts = []
                vocab_size = self.lm_head.out_features

                for i, pid in enumerate(participant_ids):
                    # Get encoder outputs for this item
                    item_inputs = {k: v[i : i + 1] for k, v in inputs.items()}
                    encoder_outputs = self.encoder(**item_inputs)

                    # Get participant bias
                    bias = self.random_effects.get_intercepts(
                        pid,
                        n_classes=vocab_size,
                        param_name="mu",
                        create_if_missing=False,
                    )

                    # Greedy decoding with bias
                    decoder_input_ids = torch.tensor(
                        [[self.tokenizer.pad_token_id]], device=self.config.device
                    )
                    generated_ids = []

                    for _ in range(self.config.max_output_length):
                        decoder_outputs = self.base_decoder(
                            input_ids=decoder_input_ids,
                            encoder_hidden_states=encoder_outputs.last_hidden_state,
                            encoder_attention_mask=item_inputs["attention_mask"],
                        )
                        logits = self.lm_head(
                            decoder_outputs.last_hidden_state[:, -1, :]
                        )

                        # Add participant bias (bias is 1D, logits is 2D)
                        logits = logits + bias.unsqueeze(0)

                        # Greedy selection
                        next_token_id = torch.argmax(logits, dim=-1)
                        generated_ids.append(next_token_id.item())

                        # Stop if EOS
                        if next_token_id.item() == self.tokenizer.eos_token_id:
                            break

                        # Append to decoder input (scalar after argmax)
                        decoder_input_ids = torch.cat(
                            [decoder_input_ids, next_token_id.unsqueeze(-1)], dim=1
                        )

                    # Decode generated text
                    text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    generated_texts.append(text)

            elif self.config.mixed_effects.mode == "random_slopes":
                # Generate with participant-specific LoRA decoder
                generated_texts = []

                for i, pid in enumerate(participant_ids):
                    # Get participant-specific decoder
                    participant_decoder = self.random_effects.get_slopes(
                        pid,
                        fixed_head=create_participant_lora_adapter(
                            self.base_decoder,
                            rank=self.config.lora_rank,
                            alpha=self.config.lora_alpha,
                            dropout=self.config.lora_dropout,
                            target_modules=self.config.lora_target_modules,
                        ),
                        create_if_missing=False,
                    )

                    # Get encoder outputs
                    item_inputs = {k: v[i : i + 1] for k, v in inputs.items()}
                    encoder_outputs = self.encoder(**item_inputs)

                    # Greedy decoding with participant decoder
                    decoder_input_ids = torch.tensor(
                        [[self.tokenizer.pad_token_id]], device=self.config.device
                    )
                    generated_ids = []

                    for _ in range(self.config.max_output_length):
                        decoder_outputs = participant_decoder(
                            input_ids=decoder_input_ids,
                            encoder_hidden_states=encoder_outputs.last_hidden_state,
                            encoder_attention_mask=item_inputs["attention_mask"],
                        )
                        logits = self.lm_head(
                            decoder_outputs.last_hidden_state[:, -1, :]
                        )

                        next_token_id = torch.argmax(logits, dim=-1)
                        generated_ids.append(next_token_id.item())

                        if next_token_id.item() == self.tokenizer.eos_token_id:
                            break

                        decoder_input_ids = torch.cat(
                            [decoder_input_ids, next_token_id.unsqueeze(-1)], dim=1
                        )

                    text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    generated_texts.append(text)

        predictions = []
        for i, item in enumerate(items):
            predictions.append(
                ModelPrediction(
                    item_id=str(item.id),
                    probabilities={},  # Not applicable for generation
                    predicted_class=generated_texts[i],  # Generated text
                    confidence=1.0,  # Not applicable for generation
                )
            )

        return predictions

    def _do_predict_proba(
        self, items: list[Item], participant_ids: list[str]
    ) -> np.ndarray:
        """Predict probabilities (not applicable for free text generation).

        For text generation, returns empty array.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Participant identifiers.

        Returns
        -------
        np.ndarray
            Empty array of shape (n_items, 0).
        """
        return np.zeros((len(items), 0))

    def _compute_exact_match(self, predictions: list[str], labels: list[str]) -> float:
        """Compute exact match accuracy.

        Parameters
        ----------
        predictions : list[str]
            Predicted texts.
        labels : list[str]
            Ground truth texts.

        Returns
        -------
        float
            Exact match accuracy (fraction of exact matches).
        """
        return sum(
            p.strip().lower() == label.strip().lower()
            for p, label in zip(predictions, labels, strict=True)
        ) / len(predictions)

    def _save_model_components(self, save_path: Path) -> None:
        """Save model-specific components (model, tokenizer).

        Parameters
        ----------
        save_path : Path
            Directory path to save the model.
        """
        self.model.save_pretrained(save_path / "model")
        self.tokenizer.save_pretrained(save_path / "model")

    def _load_model_components(
        self, load_path: Path, config_dict: dict[str, object]
    ) -> None:
        """Load model-specific components (model, tokenizer).

        Parameters
        ----------
        load_path : Path
            Directory path to load the model from.
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

        from bead.config.active_learning import FreeTextModelConfig  # noqa: PLC0415

        self.config = FreeTextModelConfig(**config_dict)

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path / "model")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "model")

        # Re-extract components
        self.encoder = self.model.get_encoder()
        self.base_decoder = self.model.get_decoder()
        self.lm_head = self.model.lm_head

        self.model.to(self.config.device)

    def _get_save_state(self) -> dict[str, object]:
        """Get model-specific state to save in config.json.

        Returns
        -------
        dict[str, object]
            Model-specific state dictionary.
        """
        return {}

    def _restore_training_state(self, config_dict: dict[str, object]) -> None:
        """Restore model-specific training state from config_dict.

        Parameters
        ----------
        config_dict : dict[str, object]
            Configuration dictionary.
        """
        pass

    def _get_n_classes_for_random_effects(self) -> int:
        """Get the number of classes for initializing RandomEffectsManager.

        For FreeTextModel, this is the vocabulary size.

        Returns
        -------
        int
            Vocabulary size.
        """
        return self.lm_head.out_features

    def _initialize_random_effects(self, n_classes: int, **kwargs: object) -> None:
        """Initialize the RandomEffectsManager.

        Parameters
        ----------
        n_classes : int
            Vocabulary size (for FreeTextModel).
        **kwargs : object
            Additional keyword arguments (not used).
        """
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects,
            vocab_size=n_classes,
        )

    def _get_random_effects_fixed_head(self) -> torch.nn.Module | None:
        """Get the fixed head for random effects.

        For FreeTextModel with random_slopes, returns a template adapter.
        For other modes, returns None.

        Returns
        -------
        torch.nn.Module | None
            Template adapter for random_slopes, None otherwise.
        """
        if self.config.mixed_effects.mode == "random_slopes":
            # For random_slopes, need to provide a template adapter
            return create_participant_lora_adapter(
                self.base_decoder,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
        return None
