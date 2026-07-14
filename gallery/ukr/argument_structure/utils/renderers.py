"""Custom template renderer for the Ukrainian argument structure experiment.

Case is carried in the noun form itself (fusional morphology), so rendering is
plain slot substitution followed by capitalizing the first character of the
sentence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bead.templates.renderers import TemplateRenderer

if TYPE_CHECKING:
    from bead.resources.lexical_item import LexicalItem
    from bead.resources.template import Slot


class UkrainianRenderer(TemplateRenderer):
    """Substitute each slot's form and capitalize the sentence.

    Each placeholder is replaced by the filler's ``form`` (falling back to its
    ``lemma``), then the first character is upper-cased so the sentence begins
    with a capital letter.

    Examples
    --------
    >>> from bead.resources.lexical_item import LexicalItem
    >>> from bead.resources.template import Slot
    >>> renderer = UkrainianRenderer()
    >>> fillers = {
    ...     "subj_nom": LexicalItem(lemma="людина", language_code="ukr"),
    ...     "verb": LexicalItem(lemma="читати", form="читає", language_code="ukr"),
    ...     "obj_acc": LexicalItem(lemma="книга", form="книгу", language_code="ukr"),
    ... }
    >>> slots = {name: Slot(name=name) for name in fillers}
    >>> renderer.render("{subj_nom} {verb} {obj_acc}.", fillers, slots)
    'Людина читає книгу.'
    """

    def render(
        self,
        template_string: str,
        slot_fillers: Mapping[str, LexicalItem],
        template_slots: Mapping[str, Slot],
    ) -> str:
        """Render the template and capitalize the first character.

        Parameters
        ----------
        template_string : str
            Template string with ``{slot_name}`` placeholders.
        slot_fillers : Mapping[str, LexicalItem]
            Mapping from slot names to the items that fill them.
        template_slots : Mapping[str, Slot]
            Mapping from slot names to slot definitions (unused).

        Returns
        -------
        str
            Rendered sentence with a capitalized first character.
        """
        result = template_string
        for slot_name, item in slot_fillers.items():
            placeholder = f"{{{slot_name}}}"
            if placeholder in result:
                surface = item.form if item.form is not None else item.lemma
                result = result.replace(placeholder, surface)
        return result[:1].upper() + result[1:]
