"""Single canonical registry mapping task types to active-learning models.

bead's eight task types each correspond to exactly one
:class:`~bead.active_learning.models.base.ActiveLearningModel` subclass
and one
:class:`~bead.config.active_learning.BaseEncoderModelConfig`-derived
config class. This module exposes those two mappings as the single
source of truth used by:

- :mod:`bead.cli.models` (CLI training commands)
- :mod:`bead.protocol.items` (protocol-layer integration)
- :func:`model_for_encoding` (protocol-encoding-driven model selection)

There is no other place in the codebase that maps task types to model
or config classes. Adding a new task type requires updating both
mappings here and registering the new model module in
:mod:`bead.active_learning.models`.
"""

from __future__ import annotations

from typing import Final

from bead.active_learning.models.base import ActiveLearningModel
from bead.active_learning.models.binary import BinaryModel
from bead.active_learning.models.categorical import CategoricalModel
from bead.active_learning.models.cloze import ClozeModel
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.active_learning.models.free_text import FreeTextModel
from bead.active_learning.models.magnitude import MagnitudeModel
from bead.active_learning.models.multi_select import MultiSelectModel
from bead.active_learning.models.ordinal_scale import OrdinalScaleModel
from bead.config.active_learning import (
    BinaryModelConfig,
    CategoricalModelConfig,
    ClozeModelConfig,
    ForcedChoiceModelConfig,
    FreeTextModelConfig,
    MagnitudeModelConfig,
    MultiSelectModelConfig,
    OrdinalScaleModelConfig,
)
from bead.items.item_template import TaskType
from bead.protocol.encoding import ResponseEncoding
from bead.protocol.items import scale_type_to_task_type

type ModelConfig = (
    BinaryModelConfig
    | CategoricalModelConfig
    | ClozeModelConfig
    | ForcedChoiceModelConfig
    | FreeTextModelConfig
    | MagnitudeModelConfig
    | MultiSelectModelConfig
    | OrdinalScaleModelConfig
)
"""Union of every active-learning model-config class."""


MODEL_CLASSES: Final[dict[TaskType, type[ActiveLearningModel]]] = {
    "binary": BinaryModel,
    "categorical": CategoricalModel,
    "cloze": ClozeModel,
    "forced_choice": ForcedChoiceModel,
    "free_text": FreeTextModel,
    "magnitude": MagnitudeModel,
    "multi_select": MultiSelectModel,
    "ordinal_scale": OrdinalScaleModel,
}
"""The canonical task-type → model-class mapping.

Add a new task type by appending an entry here and a matching entry
in :data:`CONFIG_CLASSES`. Every keyed task type must be a
``TaskType`` literal (the ``"span_labeling"`` task type has no
active-learning model and is intentionally absent).
"""


CONFIG_CLASSES: Final[dict[TaskType, type[ModelConfig]]] = {
    "binary": BinaryModelConfig,
    "categorical": CategoricalModelConfig,
    "cloze": ClozeModelConfig,
    "forced_choice": ForcedChoiceModelConfig,
    "free_text": FreeTextModelConfig,
    "magnitude": MagnitudeModelConfig,
    "multi_select": MultiSelectModelConfig,
    "ordinal_scale": OrdinalScaleModelConfig,
}
"""The canonical task-type → config-class mapping."""


def model_class_for_task_type(task_type: TaskType) -> type[ActiveLearningModel]:
    """Return the model class registered for ``task_type``.

    Parameters
    ----------
    task_type : TaskType
        Task-type literal.

    Returns
    -------
    type[ActiveLearningModel]
        The registered subclass.

    Raises
    ------
    KeyError
        If ``task_type`` has no registered model (for example,
        ``"span_labeling"``).
    """
    return MODEL_CLASSES[task_type]


def config_class_for_task_type(task_type: TaskType) -> type[ModelConfig]:
    """Return the config class registered for ``task_type``.

    Parameters
    ----------
    task_type : TaskType
        Task-type literal.

    Returns
    -------
    type[ModelConfig]
        The registered config class.

    Raises
    ------
    KeyError
        If ``task_type`` has no registered config.
    """
    return CONFIG_CLASSES[task_type]


def model_class_for_encoding(
    encoding: ResponseEncoding,
) -> type[ActiveLearningModel]:
    """Pick the active-learning model class for a protocol encoding.

    Composes :func:`~bead.protocol.items.scale_type_to_task_type` with
    :func:`model_class_for_task_type`. This is the canonical bridge
    from a :class:`~bead.protocol.ResponseEncoding` to the model
    class that should be trained on responses recorded under that
    encoding.

    Parameters
    ----------
    encoding : ResponseEncoding
        Protocol-side response encoding.

    Returns
    -------
    type[ActiveLearningModel]
        The matching model class.

    Examples
    --------
    >>> from bead.protocol import ResponseSpace, encode_response_space
    >>> rs = ResponseSpace(options=("no", "yes"), is_ordered=False)
    >>> enc = encode_response_space("dynamicity", rs)
    >>> model_class_for_encoding(enc).__name__
    'BinaryModel'
    """
    return model_class_for_task_type(scale_type_to_task_type(encoding.scale_type))


def config_class_for_encoding(
    encoding: ResponseEncoding,
) -> type[ModelConfig]:
    """Pick the active-learning config class for a protocol encoding.

    Parameters
    ----------
    encoding : ResponseEncoding
        Protocol-side response encoding.

    Returns
    -------
    type[ModelConfig]
        The matching config class.
    """
    return config_class_for_task_type(scale_type_to_task_type(encoding.scale_type))
