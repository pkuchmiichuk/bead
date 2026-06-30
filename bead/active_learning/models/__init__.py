"""Active learning models for different task types."""

from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.binary import BinaryModel
from bead.active_learning.models.categorical import CategoricalModel
from bead.active_learning.models.cloze import ClozeModel
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.active_learning.models.free_text import FreeTextModel
from bead.active_learning.models.magnitude import MagnitudeModel
from bead.active_learning.models.multi_select import MultiSelectModel
from bead.active_learning.models.ordinal_scale import OrdinalScaleModel
from bead.active_learning.models.registry import (
    CONFIG_CLASSES,
    MODEL_CLASSES,
    ModelConfig,
    config_class_for_encoding,
    config_class_for_task_type,
    model_class_for_encoding,
    model_class_for_task_type,
)

__all__ = [
    "CONFIG_CLASSES",
    "MODEL_CLASSES",
    "ActiveLearningModel",
    "BinaryModel",
    "CategoricalModel",
    "ClozeModel",
    "ForcedChoiceModel",
    "FreeTextModel",
    "MagnitudeModel",
    "ModelConfig",
    "ModelPrediction",
    "MultiSelectModel",
    "OrdinalScaleModel",
    "config_class_for_encoding",
    "config_class_for_task_type",
    "model_class_for_encoding",
    "model_class_for_task_type",
]
