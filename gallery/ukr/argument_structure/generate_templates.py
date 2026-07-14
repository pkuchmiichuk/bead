#!/usr/bin/env python3
"""Generate the Ukrainian sentence frames.

Writes ``templates/generic_frames.jsonl``: one generic, verb-independent frame
per argument structure. Each frame declares its own word order and its noun
slots, constrained by role and case. Verbs and nouns are supplied by the
lexicons when the frames are filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bead.cli.display import (
    display_file_stats,
    print_header,
    print_info,
    print_success,
)
from bead.resources.constraints import Constraint
from bead.resources.template import Slot, Template, slots_match_template
from bead.resources.template_collection import TemplateCollection

LANGUAGE = "ukr"


@dataclass(frozen=True)
class ArgSlot:
    """A noun slot in a frame, constrained by role and case.

    Parameters
    ----------
    name : str
        Slot name, matching the placeholder in the frame's template string.
    role : str
        Argument role the noun fills (e.g. ``"subject"``, ``"object"``).
    case : str
        Case the noun form must carry (e.g. ``"NOM"``, ``"ACC"``).
    """

    name: str
    role: str
    case: str


@dataclass(frozen=True)
class FrameSpec:
    """A sentence frame: a word order plus its argument slots.

    Parameters
    ----------
    name : str
        Frame identifier.
    template_string : str
        Word order with ``{slot}`` placeholders, including the verb slot.
    args : tuple[ArgSlot, ...]
        The frame's noun slots.
    description : str
        Human-readable description.
    """

    name: str
    template_string: str
    args: tuple[ArgSlot, ...]
    description: str


# Reusable argument slots, defined once and shared across frames.
SUBJ_NOM = ArgSlot("subj_nom", "subject", "NOM")
OBJ_ACC = ArgSlot("obj_acc", "object", "ACC")
OBJ_GEN = ArgSlot("obj_gen", "object", "GEN")
OBJ_DAT = ArgSlot("obj_dat", "object", "DAT")
OBJ_INS = ArgSlot("obj_ins", "object", "INS")

FRAME_SPECS: tuple[FrameSpec, ...] = (
    FrameSpec(
        "intransitive",
        "{subj_nom} {verb}.",
        (SUBJ_NOM,),
        "Nominative subject, no object.",
    ),
    FrameSpec(
        "obj_acc",
        "{subj_nom} {verb} {obj_acc}.",
        (SUBJ_NOM, OBJ_ACC),
        "Nominative subject, accusative object.",
    ),
    FrameSpec(
        "obj_gen",
        "{subj_nom} {verb} {obj_gen}.",
        (SUBJ_NOM, OBJ_GEN),
        "Nominative subject, genitive object.",
    ),
    FrameSpec(
        "obj_dat",
        "{subj_nom} {verb} {obj_dat}.",
        (SUBJ_NOM, OBJ_DAT),
        "Nominative subject, dative object.",
    ),
    FrameSpec(
        "obj_ins",
        "{subj_nom} {verb} {obj_ins}.",
        (SUBJ_NOM, OBJ_INS),
        "Nominative subject, instrumental object.",
    ),
)


def _arg_slot(arg: ArgSlot) -> Slot:
    """Return a noun slot constrained to the argument's role and case."""
    expression = (
        "self.features.get('pos') == 'N' "
        f"and self.features.get('role') == '{arg.role}' "
        f"and self.features.get('case') == '{arg.case}'"
    )
    description = f"{arg.role} ({arg.case})"
    return Slot(
        name=arg.name,
        description=description,
        constraints=(Constraint(expression=expression, description=description),),
    )


def _verb_slot() -> Slot:
    """Return a slot constrained to present-tense verbs."""
    expression = (
        "self.features.get('pos') == 'V' and self.features.get('tense') == 'PRS'"
    )
    return Slot(
        name="verb",
        description="present-tense verb",
        constraints=(
            Constraint(expression=expression, description="Present-tense verb"),
        ),
    )


def make_frame(spec: FrameSpec) -> Template:
    """Build the ``Template`` for one frame specification.

    Parameters
    ----------
    spec : FrameSpec
        The frame to build.

    Returns
    -------
    Template
        A frame with a verb slot and one constrained slot per argument.
    """
    slots: dict[str, Slot] = {arg.name: _arg_slot(arg) for arg in spec.args}
    slots["verb"] = _verb_slot()

    template = Template(
        name=spec.name,
        template_string=spec.template_string,
        slots=slots,
        description=spec.description,
        language_code=LANGUAGE,
        tags=("argument_structure",),
    )
    slots_match_template(template)
    return template


def main() -> None:
    """Build the frames and write them to ``templates/generic_frames.jsonl``."""
    print_header("Sentence Frames")

    templates = [make_frame(spec) for spec in FRAME_SPECS]
    for template in templates:
        print_info(f"{template.name}: {template.template_string}")

    collection = TemplateCollection(
        name="generic_frames", language_code=LANGUAGE
    ).with_templates(templates)
    print_success(f"Built {len(templates)} frames")

    output_path = Path(__file__).parent / "templates" / "generic_frames.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collection.to_jsonl(str(output_path))
    display_file_stats(output_path, len(templates), "frames")


if __name__ == "__main__":
    main()
