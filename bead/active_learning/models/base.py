"""Base interfaces for active learning models with mixed effects support.

This module implements Generalized Linear Mixed Effects Models (GLMMs) following
the standard formulation:

    y = Xβ + Zu + ε

Where:
- Xβ: Fixed effects (population-level parameters, shared across all groups)
- Zu: Random effects (group-specific parameters, e.g., per-participant)
- u ~ N(0, G): Random effects with variance-covariance matrix G
- ε: Residuals

The implementation supports three modeling modes:
1. Fixed effects: Standard model, ignores grouping structure
2. Random intercepts: Per-group biases (Zu = bias vector per group)
3. Random slopes: Per-group model parameters (Zu = separate model head per group)

References
----------
- Bates et al. (2015). "Fitting Linear Mixed-Effects Models using lme4"
- Simchoni & Rosset (2022). "Integrating Random Effects in Deep Neural Networks"
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bead.active_learning.config import (
    MixedEffectsConfig,
    RandomEffectsSpec,
    VarianceComponents,
)
from bead.data.base import BeadBaseModel
from bead.items.item import Item

if TYPE_CHECKING:
    import torch

    from bead.items.item_template import ItemTemplate, TaskType

__all__ = [
    "ActiveLearningModel",
    "ModelPrediction",
    "MixedEffectsConfig",
    "VarianceComponents",
    "RandomEffectsSpec",
]


class ModelPrediction(BeadBaseModel):
    """Prediction output for a single item.

    Attributes
    ----------
    item_id : str
        Unique identifier for the item.
    probabilities : dict[str, float]
        Predicted probabilities for each class/option.
        Keys are option names (e.g., "option_a", "option_b") or class labels.
    predicted_class : str
        The predicted class/option with highest probability.
    confidence : float
        Confidence score (max probability).

    Examples
    --------
    >>> prediction = ModelPrediction(
    ...     item_id="abc123",
    ...     probabilities={"option_a": 0.7, "option_b": 0.3},
    ...     predicted_class="option_a",
    ...     confidence=0.7
    ... )
    >>> prediction.predicted_class
    'option_a'
    """

    item_id: str
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float


class ActiveLearningModel(ABC):
    """Base class for all active learning models with mixed effects support.

    Implements GLMM-based active learning: y = Xβ + Zu + ε

    All models must:
    1. Support mixed effects (fixed, random_intercepts, random_slopes modes)
    2. Accept participant_ids in train/predict/predict_proba (None for fixed effects)
    3. Validate items match supported task types
    4. Track variance components (if estimate_variance_components=True)

    Attributes
    ----------
    config : dict[str, str | int | float | bool | None] | BeadBaseModel
        Model configuration (task-type-specific).
        Must include a `mixed_effects: MixedEffectsConfig` field.
    supported_task_types : list[TaskType]
        List of task types this model can handle.

    Examples
    --------
    >>> class MyModel(ActiveLearningModel):
    ...     def __init__(self, config):
    ...         super().__init__(config)  # Validates mixed_effects field
    ...     @property
    ...     def supported_task_types(self):
    ...         return ["forced_choice"]
    ...     def validate_item_compatibility(self, item, item_template):
    ...         pass
    ...     def train(self, items, labels, participant_ids):
    ...         return {}
    ...     def predict(self, items, participant_ids):
    ...         return []
    ...     def predict_proba(self, items, participant_ids):
    ...         return np.array([])
    ...     def save(self, path):
    ...         pass
    ...     def load(self, path):
    ...         pass
    """

    def __init__(
        self, config: dict[str, str | int | float | bool | None] | BeadBaseModel
    ) -> None:
        """Initialize model with configuration.

        Parameters
        ----------
        config : Any
            Model configuration. Must have a `mixed_effects` field of type
            MixedEffectsConfig.

        Raises
        ------
        ValueError
            If config is invalid or missing required fields.

        Examples
        --------
        >>> from bead.config.active_learning import ForcedChoiceModelConfig
        >>> config = ForcedChoiceModelConfig(
        ...     n_classes=2,
        ...     mixed_effects=MixedEffectsConfig(mode='fixed')
        ... )
        >>> model = ForcedChoiceModel(config)  # doctest: +SKIP
        """
        self.config = config

        # Validate mixed_effects field exists
        if not hasattr(config, "mixed_effects"):
            raise ValueError(
                f"Model config must have a 'mixed_effects' field of type "
                f"MixedEffectsConfig, but {type(config).__name__} has no such field. "
                f"Add: mixed_effects: MixedEffectsConfig = "
                f"Field(default_factory=MixedEffectsConfig)"
            )

        # Validate mixed_effects is correct type
        if not isinstance(config.mixed_effects, MixedEffectsConfig):
            raise ValueError(
                f"config.mixed_effects must be MixedEffectsConfig, but got "
                f"{type(config.mixed_effects).__name__}. "
                f"Ensure the field is properly typed: mixed_effects: MixedEffectsConfig"
            )

    def _validate_items_labels_length(
        self, items: list[Item], labels: list[str]
    ) -> None:
        """Validate that items and labels have the same length.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels.

        Raises
        ------
        ValueError
            If items and labels have different lengths.
        """
        if len(items) != len(labels):
            raise ValueError(
                f"Number of items ({len(items)}) must match "
                f"number of labels ({len(labels)})"
            )

    def _validate_participant_ids_required(
        self, participant_ids: list[str] | None, mode: str
    ) -> None:
        """Validate that participant_ids is provided when required.

        Parameters
        ----------
        participant_ids : list[str] | None
            Participant IDs to validate.
        mode : str
            Mixed effects mode ('fixed', 'random_intercepts', 'random_slopes').

        Raises
        ------
        ValueError
            If participant_ids is None when mode requires it.
        """
        if participant_ids is None and mode != "fixed":
            raise ValueError(
                f"participant_ids is required when mode='{mode}'. "
                f"For fixed effects, set mode='fixed' in config. "
                f"For mixed effects, provide participant_ids as list[str]."
            )

    def _validate_participant_ids_length(
        self, items: list[Item], participant_ids: list[str]
    ) -> None:
        """Validate that items and participant_ids have the same length.

        Parameters
        ----------
        items : list[Item]
            Training items.
        participant_ids : list[str]
            Participant IDs.

        Raises
        ------
        ValueError
            If items and participant_ids have different lengths.
        """
        if len(items) != len(participant_ids):
            raise ValueError(
                f"Length mismatch: {len(items)} items != {len(participant_ids)} "
                f"participant_ids. participant_ids must have same length as items."
            )

    def _validate_participant_ids_not_empty(self, participant_ids: list[str]) -> None:
        """Validate that participant_ids does not contain empty strings.

        Parameters
        ----------
        participant_ids : list[str]
            Participant IDs to validate.

        Raises
        ------
        ValueError
            If participant_ids contains empty strings.
        """
        if any(not pid for pid in participant_ids):
            raise ValueError(
                "participant_ids cannot contain empty strings. "
                "Ensure all participants have valid identifiers."
            )

    def _normalize_participant_ids(
        self,
        participant_ids: list[str] | None,
        items: list[Item],
        mode: str,
    ) -> list[str]:
        """Normalize participant_ids based on mode.

        For fixed mode, replaces participant_ids with dummy values.
        For mixed effects modes, validates and returns participant_ids as-is.

        Parameters
        ----------
        participant_ids : list[str] | None
            Participant IDs (may be None for fixed mode).
        items : list[Item]
            Training items (used to determine length).
        mode : str
            Mixed effects mode ('fixed', 'random_intercepts', 'random_slopes').

        Returns
        -------
        list[str]
            Normalized participant IDs (all "_fixed_" for fixed mode).

        Raises
        ------
        ValueError
            If participant_ids is None when mode requires it.
        ValueError
            If items and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        """
        import warnings  # noqa: PLC0415

        if participant_ids is None:
            if mode != "fixed":
                self._validate_participant_ids_required(participant_ids, mode)
            return ["_fixed_"] * len(items)

        # Validate length and empty strings before normalizing
        self._validate_participant_ids_length(items, participant_ids)
        self._validate_participant_ids_not_empty(participant_ids)

        if mode == "fixed":
            warnings.warn(
                "participant_ids provided but mode='fixed'. "
                "Participant IDs will be ignored.",
                UserWarning,
                stacklevel=3,
            )
            return ["_fixed_"] * len(items)

        return participant_ids

    @property
    @abstractmethod
    def supported_task_types(self) -> list[TaskType]:
        """Get list of task types this model supports.

        Returns
        -------
        list[TaskType]
            List of supported TaskType literals from items.models.

        Examples
        --------
        >>> model.supported_task_types
        ['forced_choice']
        """
        pass

    @abstractmethod
    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate that an item is compatible with this model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If item's task_type is not in supported_task_types.
        ValueError
            If item is missing required elements.
        ValueError
            If item structure is incompatible with model.

        Examples
        --------
        >>> model.validate_item_compatibility(item, template)  # doctest: +SKIP
        """
        pass

    # Hook methods for model-specific implementations
    @abstractmethod
    def _prepare_training_data(
        self,
        items: list[Item],
        labels: list[str],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> tuple[list[Item], list, list[str], list[Item] | None, list | None]:
        """Prepare training data for model-specific training.

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
        tuple[list[Item], list, list[str], list[Item] | None, list | None]
            Items, labels, participant_ids, val_items, val_labels.
        """
        pass

    @abstractmethod
    def _initialize_random_effects(self, n_classes: int) -> None:
        """Initialize random effects manager.

        Parameters
        ----------
        n_classes : int
            Number of classes for random effects.
        """
        pass

    @abstractmethod
    def _do_training(
        self,
        items: list[Item],
        labels_numeric: list,
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels_numeric: list | None,
    ) -> dict[str, float]:
        """Perform model-specific training.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels_numeric : list
            Numeric labels (format depends on model).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels_numeric : list | None
            Numeric validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        pass

    @abstractmethod
    def _do_predict(
        self, items: list[Item], participant_ids: list[str]
    ) -> list[ModelPrediction]:
        """Perform model-specific prediction.

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
        pass

    @abstractmethod
    def _do_predict_proba(
        self, items: list[Item], participant_ids: list[str]
    ) -> np.ndarray:
        """Perform model-specific probability prediction.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        np.ndarray
            Probability array.
        """
        pass

    @abstractmethod
    def _get_save_state(self) -> dict[str, object]:
        """Get model-specific state to save.

        Returns
        -------
        dict[str, object]
            State dictionary to include in config.json.
        """
        pass

    @abstractmethod
    def _save_model_components(self, save_path: Path) -> None:
        """Save model-specific components (encoder, head, etc.).

        Parameters
        ----------
        save_path : Path
            Directory to save to.
        """
        pass

    @abstractmethod
    def _load_model_components(
        self, load_path: Path, config_dict: dict[str, object]
    ) -> None:
        """Load model-specific components.

        Parameters
        ----------
        load_path : Path
            Directory to load from.
        config_dict : dict[str, object]
            Schema-only config dict (model-specific state fields have
            already been popped by :meth:`_restore_training_state`).
            Subclasses use this to reconstruct ``self.config`` without
            re-reading ``config.json`` from disk.
        """
        pass

    @abstractmethod
    def _restore_training_state(self, config_dict: dict[str, object]) -> None:
        """Restore model-specific training state.

        Parameters
        ----------
        config_dict : dict[str, object]
            Configuration dictionary with training state.
        """
        pass

    @abstractmethod
    def _get_random_effects_fixed_head(self) -> torch.nn.Module | None:
        """Get fixed head for random effects loading.

        Returns
        -------
        nn.Module | None
            Fixed head module, or None if not applicable.
        """
        pass

    @abstractmethod
    def _get_n_classes_for_random_effects(self) -> int:
        """Get number of classes for random effects initialization.

        Returns
        -------
        int
            Number of classes.
        """
        pass

    # Common implementations
    def train(
        self,
        items: list[Item],
        labels: list[str] | list[list[str]],
        participant_ids: list[str] | None = None,
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | list[list[str]] | None = None,
    ) -> dict[str, float]:
        """Train model on labeled items with participant identifiers.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (format depends on task type).
        participant_ids : list[str] | None
            Participant identifier for each item.
            - For fixed effects (mode='fixed'): Pass None (automatically handled).
            - For mixed effects (mode='random_intercepts' or 'random_slopes'):
              Must provide list[str] with same length as items.
            Must not contain empty strings.
        validation_items : list[Item] | None
            Optional validation items.
        validation_labels : list[str] | None
            Optional validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics including:
            - "train_accuracy", "train_loss": Standard metrics
            - "participant_variance": σ²_u (if estimate_variance_components=True)
            - "n_participants": Number of unique participants
            - "residual_variance": σ²_ε (if estimated)

        Raises
        ------
        ValueError
            If participant_ids is None when mode is 'random_intercepts'
            or 'random_slopes'.
        ValueError
            If items, labels, and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        ValueError
            If validation data is incomplete.
        ValueError
            If labels are invalid for this task type.
        """
        # Validate input lengths (handle both list[str] and list[list[str]] labels)
        if labels and isinstance(labels[0], list):
            # Cloze model: labels is list[list[str]]
            if len(items) != len(labels):
                raise ValueError(
                    f"Number of items ({len(items)}) must match "
                    f"number of labels ({len(labels)})"
                )
        else:
            # Standard models: labels is list[str]
            self._validate_items_labels_length(items, labels)

        # Validate and normalize participant_ids
        participant_ids = self._normalize_participant_ids(
            participant_ids, items, self.config.mixed_effects.mode
        )

        if (validation_items is None) != (validation_labels is None):
            raise ValueError(
                "Both validation_items and validation_labels must be "
                "provided, or neither"
            )

        # Prepare training data (model-specific)
        (
            prepared_items,
            labels_numeric,
            participant_ids,
            validation_items,
            validation_labels_numeric,
        ) = self._prepare_training_data(
            items, labels, participant_ids, validation_items, validation_labels
        )

        # Initialize random effects
        n_classes = self._get_n_classes_for_random_effects()
        self._initialize_random_effects(n_classes)

        # Register participants for adaptive regularization
        if hasattr(self, "random_effects") and self.random_effects is not None:
            participant_counts = Counter(participant_ids)
            for pid, count in participant_counts.items():
                self.random_effects.register_participant(pid, count)

        # Perform training (model-specific)
        metrics = self._do_training(
            prepared_items,
            labels_numeric,
            participant_ids,
            validation_items,
            validation_labels_numeric,
        )

        self._is_fitted = True

        # Estimate variance components
        if (
            self.config.mixed_effects.estimate_variance_components
            and hasattr(self, "random_effects")
            and self.random_effects is not None
        ):
            var_comps = self.random_effects.estimate_variance_components()
            if var_comps:
                var_comp = var_comps.get("mu") or var_comps.get("slopes")
                if var_comp:
                    if not hasattr(self, "variance_history"):
                        self.variance_history = []
                    self.variance_history.append(var_comp)
                    metrics["participant_variance"] = var_comp.variance
                    metrics["n_participants"] = var_comp.n_groups

        return metrics

    def predict(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> list[ModelPrediction]:
        """Predict class labels for items with participant identifiers.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str] | None
            Participant identifier for each item.
            - For fixed effects (mode='fixed'): Pass None.
            - For mixed effects: Must provide list[str] with same length as items.
            - For unknown participants: Use population mean (prior) for random effects.

        Returns
        -------
        list[ModelPrediction]
            Predictions with probabilities and predicted class for each item.

        Raises
        ------
        ValueError
            If model has not been trained.
        ValueError
            If participant_ids is None when mode requires mixed effects.
        ValueError
            If items and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        ValueError
            If items are incompatible with model.
        """
        if not self._is_fitted:
            raise ValueError("Model not trained. Call train() before predict().")

        # Validate and normalize participant_ids
        participant_ids = self._normalize_participant_ids(
            participant_ids, items, self.config.mixed_effects.mode
        )

        return self._do_predict(items, participant_ids)

    def predict_proba(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> np.ndarray:
        """Predict class probabilities for items with participant identifiers.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str] | None
            Participant identifier for each item.
            - For fixed effects (mode='fixed'): Pass None.
            - For mixed effects: Must provide list[str] with same length as items.

        Returns
        -------
        np.ndarray
            Array of shape (n_items, n_classes) with probabilities.
            Each row sums to 1.0 for classification tasks.

        Raises
        ------
        ValueError
            If model has not been trained.
        ValueError
            If participant_ids is None when mode requires mixed effects.
        ValueError
            If items and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        ValueError
            If items are incompatible with model.
        """
        if not self._is_fitted:
            raise ValueError("Model not trained. Call train() before predict_proba().")

        # Validate and normalize participant_ids
        participant_ids = self._normalize_participant_ids(
            participant_ids, items, self.config.mixed_effects.mode
        )

        return self._do_predict_proba(items, participant_ids)

    def save(self, path: str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str
            File or directory path to save the model.

        Raises
        ------
        ValueError
            If model has not been trained.
        """
        if not self._is_fitted:
            raise ValueError("Model not trained. Call train() before save().")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model-specific components
        self._save_model_components(save_path)

        # Save random effects (includes variance history)
        if hasattr(self, "random_effects") and self.random_effects is not None:
            # Copy variance_history from model to random_effects before saving
            if hasattr(self, "variance_history"):
                self.random_effects.variance_history = self.variance_history.copy()
            self.random_effects.save(save_path / "random_effects")

        # Save config with model-specific state
        config_dict = self.config.model_dump()
        save_state = self._get_save_state()
        config_dict.update(save_state)

        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def load(self, path: str) -> None:
        """Load model from disk.

        Parameters
        ----------
        path : str
            File or directory path to load the model from.

        Raises
        ------
        FileNotFoundError
            If model file/directory does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        with open(load_path / "config.json") as f:
            config_dict = json.load(f)

        # Restore model-specific training state (before reconstructing config)
        self._restore_training_state(config_dict)

        # Load model-specific components (which will reconstruct the config)
        # This must happen before initializing random effects so config is correct
        self._load_model_components(load_path, config_dict)

        # Initialize and load random effects
        n_classes = self._get_n_classes_for_random_effects()
        from bead.active_learning.models.random_effects import (  # noqa: PLC0415
            RandomEffectsManager,
        )

        # Check if model uses vocab_size instead of n_classes (e.g., ClozeModel)
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "vocab_size"):
            # ClozeModel: use vocab_size
            self.random_effects = RandomEffectsManager(
                self.config.mixed_effects, vocab_size=n_classes
            )
        else:
            # Standard models: use n_classes
            self.random_effects = RandomEffectsManager(
                self.config.mixed_effects, n_classes=n_classes
            )
        random_effects_path = load_path / "random_effects"
        if random_effects_path.exists():
            fixed_head = self._get_random_effects_fixed_head()
            self.random_effects.load(random_effects_path, fixed_head=fixed_head)
            # Restore variance history from random_effects
            if hasattr(self.random_effects, "variance_history"):
                if not hasattr(self, "variance_history"):
                    self.variance_history = []
                self.variance_history = self.random_effects.variance_history.copy()

        # Move to device (model-specific)
        if hasattr(self, "encoder"):
            self.encoder.to(self.config.device)
        if hasattr(self, "model"):
            self.model.to(self.config.device)

        self._is_fitted = True
