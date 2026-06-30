"""Prompt-template label-reference syntax.

bead prompt templates use a single canonical syntax for embedding
references to named spans within a prompt:

- ``[[label]]`` — the prompt is rendered with the span's reconstructed
  text in this position
- ``[[label:text]]`` — the prompt is rendered with the explicit ``text``
  in this position (overrides the reconstructed text)
- ``[[label|transform1|transform2]]`` — the reconstructed text is
  passed through the named transforms before rendering
- ``[[label:text|transform1]]`` — combines the explicit-text and
  transform forms

This module is the single canonical home for that syntax. Drift
validators, item-construction utilities, and the jsPsych deployment
layer all parse references through :func:`parse_label_refs` and never
through their own regular expressions.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from bead.data.base import BeadBaseModel

LABEL_PATTERN: re.Pattern[str] = re.compile(
    r"\[\[([^\]:|]+?)(?::([^\]|]+?))?(?:\|([^\]]+?))?\]\]"
)
"""Compiled regex for ``[[label]]`` / ``[[label:text]]`` / ``[[label|t]]``.

Capture groups: ``(1)`` label name, ``(2)`` optional display text,
``(3)`` optional pipe-separated transform list. The pattern is
non-greedy and rejects ``]``, ``:``, and ``|`` characters inside the
label name.
"""


class LabelRef(BeadBaseModel):
    """A parsed label reference.

    Attributes
    ----------
    label : str
        The label name.
    display_text : str | None
        Explicit display text supplied via ``[[label:text]]``, or
        ``None`` when the reference is bare.
    transforms : tuple[str, ...]
        Transform names supplied via ``[[label|t1|t2]]``, in order.
        Empty when no transforms were supplied.
    start_offset : int
        Inclusive character offset of the matched reference in the
        original prompt.
    end_offset : int
        Exclusive character offset of the matched reference in the
        original prompt.
    """

    label: str
    display_text: str | None = None
    transforms: tuple[str, ...] = ()
    start_offset: int = 0
    end_offset: int = 0


def parse_label_refs(prompt: str) -> tuple[LabelRef, ...]:
    """Parse every label reference in ``prompt``, in order.

    Parameters
    ----------
    prompt : str
        Prompt string potentially containing label references.

    Returns
    -------
    tuple[LabelRef, ...]
        Parsed references in order of appearance. Empty tuple when no
        references match.

    Examples
    --------
    >>> refs = parse_label_refs("Did [[situation|gerund]] happen?")
    >>> refs[0].label
    'situation'
    >>> refs[0].transforms
    ('gerund',)
    """
    refs: list[LabelRef] = []
    for match in LABEL_PATTERN.finditer(prompt):
        raw_transforms = match.group(3)
        if raw_transforms is None:
            transforms: tuple[str, ...] = ()
        else:
            transforms = tuple(
                t.strip() for t in raw_transforms.split("|") if t.strip()
            )
        display_text = match.group(2).strip() if match.group(2) else None
        refs.append(
            LabelRef(
                label=match.group(1).strip(),
                display_text=display_text,
                transforms=transforms,
                start_offset=match.start(),
                end_offset=match.end(),
            )
        )
    return tuple(refs)


def find_label_names(prompt: str) -> frozenset[str]:
    """Return the set of label names referenced in ``prompt``.

    Convenience wrapper around :func:`parse_label_refs` that discards
    everything except the label names. Used by structural drift
    validation, where only the *which-labels-are-present* question
    matters and display text and transforms are irrelevant.

    Parameters
    ----------
    prompt : str
        Prompt string potentially containing label references.

    Returns
    -------
    frozenset[str]
        Distinct label names referenced in the prompt.

    Examples
    --------
    >>> find_label_names("Compare [[a]] and [[b:other]] and [[a|gerund]].")
    frozenset({'a', 'b'})
    """
    return frozenset(ref.label for ref in parse_label_refs(prompt))


def replace_label_refs(
    prompt: str,
    render: Callable[[LabelRef], str],
) -> str:
    """Rewrite ``prompt`` by replacing each reference with rendered text.

    The ``render`` callable is invoked once per reference and must
    return the string that should replace it. Replacements are applied
    right-to-left so earlier matches' offsets remain valid.

    Parameters
    ----------
    prompt : str
        Prompt string potentially containing label references.
    render : Callable[[LabelRef], str]
        Function returning the replacement text for one reference.

    Returns
    -------
    str
        Prompt with every reference replaced.
    """
    refs = parse_label_refs(prompt)
    if not refs:
        return prompt
    result = prompt
    for ref in reversed(refs):
        replacement = render(ref)
        result = result[: ref.start_offset] + replacement + result[ref.end_offset :]
    return result
