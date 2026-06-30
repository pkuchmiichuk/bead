# bead.active_learning

Stage 6 of the bead pipeline: active learning with GLMM support and convergence detection.

## Active Learning Loop

::: bead.active_learning.loop
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.selection
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.strategies
    options:
      show_root_heading: true
      show_source: false

## Configuration

::: bead.active_learning.config
    options:
      show_root_heading: true
      show_source: false

## Model Registry

Single canonical task-type → model-class and task-type → config-class
mapping used by the CLI training commands and the protocol-encoding
factory.

::: bead.active_learning.models.registry
    options:
      show_root_heading: true
      show_source: false

## Base Model Interface

::: bead.active_learning.models.base
    options:
      show_root_heading: true
      show_source: false

## Task-Specific Models

::: bead.active_learning.models.forced_choice
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.ordinal_scale
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.binary
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.categorical
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.multi_select
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.magnitude
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.free_text
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.cloze
    options:
      show_root_heading: true
      show_source: false

## Random Effects and LoRA

::: bead.active_learning.models.random_effects
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.lora
    options:
      show_root_heading: true
      show_source: false
