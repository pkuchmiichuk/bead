# bead.deployment

Stage 5 of the bead pipeline: jsPsych 8.x batch experiment generation for JATOS.

## Distribution Strategies

::: bead.deployment.distribution
    options:
      show_root_heading: true
      show_source: false

## jsPsych Experiment Generation

::: bead.deployment.jspsych.generator
    options:
      show_root_heading: true
      show_source: false

::: bead.deployment.jspsych.config
    options:
      show_root_heading: true
      show_source: false

::: bead.deployment.jspsych.trials
    options:
      show_root_heading: true
      show_source: false

::: bead.deployment.jspsych.randomizer
    options:
      show_root_heading: true
      show_source: false

## JATOS Export

::: bead.deployment.jatos.exporter
    options:
      show_root_heading: true
      show_source: false

::: bead.deployment.jatos.api
    options:
      show_root_heading: true
      show_source: false

## Protocol-Layer Bridge

Single canonical bridge from a configured
:class:`~bead.protocol.AnnotationProtocol` and a sequence of
:class:`~bead.protocol.ProtocolContext` records to a flat list of
jsPsych trial dicts.

::: bead.deployment.protocol_trials
    options:
      show_root_heading: true
      show_source: false
