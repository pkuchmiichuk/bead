# bead.dsl

Domain-Specific Language for constraint expressions used in template slot filling and list partitioning.

## Parser

::: bead.dsl.parser
    options:
      show_root_heading: true
      show_source: false

## Evaluator

::: bead.dsl.evaluator
    options:
      show_root_heading: true
      show_source: false

## Standard Library

The standard library includes string, collection, math, type-checking, and
model/simulation builtins, plus **structural-query builtins** that traverse a
dependency parse stored on an `Item` as token-level spans and relations
(`upos`, `xpos`, `lemma_of`, `deprel`, `morph`, `head`, `dependents`,
`has_relation`, `root`, `subtree`, `path_to_root`, `tokens_with_upos`,
`tokens_with_deprel`, `any_deprel`, `filter_upos`). These let a constraint query
syntactic structure, for example:

```text
upos(self, root(self)) == "VERB" and len(dependents(self, root(self), "obj")) > 0
```

which matches sentences whose root is a verb taking a direct object.

::: bead.dsl.stdlib
    options:
      show_root_heading: true
      show_source: false

## Abstract Syntax Tree

::: bead.dsl.ast
    options:
      show_root_heading: true
      show_source: false

## Context

::: bead.dsl.context
    options:
      show_root_heading: true
      show_source: false

## Errors

::: bead.dsl.errors
    options:
      show_root_heading: true
      show_source: false
