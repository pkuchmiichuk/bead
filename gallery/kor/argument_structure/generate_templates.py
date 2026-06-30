#!/usr/bin/env python3
"""Generate Template objects for Korean argument structure dataset.

Output: templates/generic_frames.jsonl

Template groups (additive — each builds on base):
  base        Simple intransitive (S-V) and transitive (S-O-V). Always included.
  adjuncts    Frames with oblique arguments. Filtered by --adjuncts case type(s).
  progressive Present/past progressive variants of base frames.
  complement  Complement clause frames (factive, quotative, indirect question).
"""

import argparse

from bead.resources import Constraint, Slot, Template

# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

_NOUN_DESCRIPTIONS: dict[str, str] = {
    "noun_subj":  "subject noun phrase",
    "noun_dobj":  "object noun phrase",
    "noun_pobj":  "oblique noun phrase",
    "noun_pobj2": "second oblique noun phrase",
    "comp_subj":  "complement clause subject",
}

_NOUN   = "self.features.get('pos')=='NOUN'"
_V_FIN  = "self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'"
_V_PTCP = "self.features.get('pos')=='V' and self.features.get('verb_form')=='V.PTCP'"


def noun_slot(name: str) -> Slot:
    """Return a Slot constrained to NOUN items, with a standard description."""
    return Slot(
        name=name,
        description=_NOUN_DESCRIPTIONS.get(name, f"{name} noun phrase"),
        constraints=[Constraint(expression=_NOUN)],
    )


def case_slot(name: str, pos: str, description: str) -> Slot:
    """Return a Slot constrained to the given POS tag (case particle)."""
    return Slot(
        name=name,
        description=description,
        constraints=[Constraint(expression=f"self.features.get('pos')=='{pos}'")],
    )


def verb_slot() -> Slot:
    """Return a Slot for finite main verbs (excludes V.PTCP)."""
    return Slot(
        name="verb",
        description="main verb",
        constraints=[Constraint(expression=_V_FIN)],
    )


def ptcp_verb_slot() -> Slot:
    """Return a Slot for participial verb stems used in progressive constructions."""
    return Slot(
        name="verb",
        description="main verb stem (participial form for progressive)",
        constraints=[Constraint(expression=_V_PTCP)],
    )


def aux_slot(tense: str) -> Slot:
    """Return a progressive auxiliary Slot constrained to the given tense."""
    aux_form = "있다" if tense == "PRS" else "있었다"
    return Slot(
        name="aux",
        description=f"progressive auxiliary ({aux_form})",
        constraints=[Constraint(
            expression=(
                f"self.features.get('pos')=='AUX'"
                f" and self.features.get('tense')=='{tense}'"
            )
        )],
    )


def spatial_noun_slot() -> Slot:
    """Postpositional relational noun slot (NPOST): 위, 아래, 앞, etc."""
    return Slot(
        name="spatial_noun",
        description="spatial/temporal relational noun (postpositional)",
        constraints=[Constraint(expression="self.features.get('pos')=='NPOST'")],
    )


def complex_postp_dat_slot() -> Slot:
    """DAT-governed complex postposition: 에 대해서, 에 관해서, etc."""
    return Slot(
        name="complex_postp_dat",
        description="DAT-governed complex postposition (includes initial particle 에)",
        constraints=[Constraint(expression=(
            "self.features.get('pos')=='POSTP' and self.features.get('gov_case')=='DAT'"
        ))],
    )


def complex_postp_acc_slot() -> Slot:
    """ACC-governed complex postposition: 통해서, 위해서, etc."""
    return Slot(
        name="complex_postp_acc",
        description="ACC-governed complex postposition (preceded by separate 을/를 slot)",  # noqa: E501
        constraints=[Constraint(expression=(
            "self.features.get('pos')=='POSTP' and self.features.get('gov_case')=='ACC'"
        ))],
    )


def fc_agree(noun: str, marker: str) -> Constraint:
    """Final-consonant agreement constraint between a noun slot and its case marker."""
    return Constraint(
        expression=(
            f"{noun}.features.get('final_consonant')"
            f" == {marker}.features.get('final_consonant')"
        )
    )


# ---------------------------------------------------------------------------
# Case marker slot specs  (slot_name, pos_tag, description)
# ---------------------------------------------------------------------------

NOM     = ("nom",      "PART.NOM",      "nominative case marker")
ACC     = ("acc",      "PART.ACC",      "accusative case marker")
DAT     = ("dat",      "PART.DAT",      "dative case marker")
LOC     = ("loc",      "PART.LOC.SRC",  "source locative case marker (에서)")
GOAL    = ("goal",     "PART.LOC.GOAL", "goal locative case marker (에)")
INST    = ("inst",     "PART.INST",     "instrumental case marker")
COM     = ("com",      "PART.COM",      "comitative case marker (와/과)")
TERM    = ("term",     "PART.TERM",     "terminative case marker (까지)")
INIT    = ("init",     "PART.INIT",     "initiative case marker (부터)")
SIM     = ("sim",      "PART.SIM",      "similative case marker (처럼/같이)")
POBJ_ACC = (
    "pobj_acc", "PART.ACC", "accusative case marker for complex postposition object"
)

# Adjunct slot names — anything beyond the core NOM/ACC pair.
_CORE_CASES: frozenset[str] = frozenset({"nom", "acc"})


# ---------------------------------------------------------------------------
# Frame table
# Each row: (name, template_string, noun_slots, case_slots, fc_pairs, description)
# ---------------------------------------------------------------------------

FRAME_SPECS: list[tuple] = [
    # ── Base: Intransitive ────────────────────────────────────────────────────
    ("subj_nom-verb.",
     "{noun_subj}{nom} {verb}.",
     ["noun_subj"],
     [NOM],
     [("noun_subj", "nom")],
     "Intransitive sentence"),

    # ── Base: Transitive ─────────────────────────────────────────────────────
    ("subj_nom-obj_acc-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {verb}.",
     ["noun_subj", "noun_dobj"],
     [NOM, ACC],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence"),

    # ── Adjunct: Dative (에게) ────────────────────────────────────────────────
    ("subj_nom-noun_dat-verb.",
     "{noun_subj}{nom} {noun_pobj}{dat} {verb}.",
     ["noun_subj", "noun_pobj"],
     [NOM, DAT],
     [("noun_subj", "nom")],
     "Intransitive sentence with dative argument"),

    ("subj_nom-obj_acc-noun_dat-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{dat} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"],
     [NOM, ACC, DAT],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with dative argument"),

    ("subj_nom-noun_dat-obj_acc-verb.",
     "{noun_subj}{nom} {noun_pobj}{dat} {noun_dobj}{acc} {verb}.",
     ["noun_subj", "noun_pobj", "noun_dobj"],
     [NOM, DAT, ACC],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with dative-before-accusative scrambled word order"),

    # ── Adjunct: Source locative (에서) ───────────────────────────────────────
    ("subj_nom-noun_loc-verb.",
     "{noun_subj}{nom} {noun_pobj}{loc} {verb}.",
     ["noun_subj", "noun_pobj"],
     [NOM, LOC],
     [("noun_subj", "nom")],
     "Intransitive sentence with source locative argument"),

    ("subj_nom-obj_acc-noun_loc-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{loc} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"],
     [NOM, ACC, LOC],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with source locative argument"),

    ("subj_nom-noun_loc-obj_acc-verb.",
     "{noun_subj}{nom} {noun_pobj}{loc} {noun_dobj}{acc} {verb}.",
     ["noun_subj", "noun_pobj", "noun_dobj"],
     [NOM, LOC, ACC],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with source locative before accusative (scrambled)"),

    # ── Adjunct: Goal locative (에) ───────────────────────────────────────────
    ("subj_nom-noun_goal-verb.",
     "{noun_subj}{nom} {noun_pobj}{goal} {verb}.",
     ["noun_subj", "noun_pobj"],
     [NOM, GOAL],
     [("noun_subj", "nom")],
     "Intransitive sentence with goal locative argument"),

    ("subj_nom-obj_acc-noun_goal-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{goal} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"],
     [NOM, ACC, GOAL],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with goal locative argument"),

    ("subj_nom-noun_goal-obj_acc-verb.",
     "{noun_subj}{nom} {noun_pobj}{goal} {noun_dobj}{acc} {verb}.",
     ["noun_subj", "noun_pobj", "noun_dobj"],
     [NOM, GOAL, ACC],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with goal locative before accusative (scrambled)"),

    # ── Adjunct: Instrumental (으로/로) ────────────────────────────────────────
    ("subj_nom-noun_inst-verb.",
     "{noun_subj}{nom} {noun_pobj}{inst} {verb}.",
     ["noun_subj", "noun_pobj"],
     [NOM, INST],
     [("noun_subj", "nom"), ("noun_pobj", "inst")],
     "Intransitive sentence with instrumental argument"),

    ("subj_nom-obj_acc-noun_inst-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{inst} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"],
     [NOM, ACC, INST],
     [("noun_subj", "nom"), ("noun_dobj", "acc"), ("noun_pobj", "inst")],
     "Transitive sentence with instrumental argument"),

    ("subj_nom-noun_inst-obj_acc-verb.",
     "{noun_subj}{nom} {noun_pobj}{inst} {noun_dobj}{acc} {verb}.",
     ["noun_subj", "noun_pobj", "noun_dobj"],
     [NOM, INST, ACC],
     [("noun_subj", "nom"), ("noun_dobj", "acc"), ("noun_pobj", "inst")],
     "Transitive sentence with instrumental before accusative (scrambled)"),

    # ── Adjunct: Dative + Source locative ────────────────────────────────────
    ("subj_nom-noun_loc-noun_dat-verb.",
     "{noun_subj}{nom} {noun_pobj}{loc} {noun_pobj2}{dat} {verb}.",
     ["noun_subj", "noun_pobj", "noun_pobj2"],
     [NOM, LOC, DAT],
     [("noun_subj", "nom")],
     "Intransitive sentence with source locative and dative arguments"),

    ("subj_nom-obj_acc-noun_loc-noun_dat-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{loc} {noun_pobj2}{dat} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj", "noun_pobj2"],
     [NOM, ACC, LOC, DAT],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with source locative and dative arguments"),

    # ── Adjunct: Dative + Goal locative ──────────────────────────────────────
    ("subj_nom-noun_goal-noun_dat-verb.",
     "{noun_subj}{nom} {noun_pobj}{goal} {noun_pobj2}{dat} {verb}.",
     ["noun_subj", "noun_pobj", "noun_pobj2"],
     [NOM, GOAL, DAT],
     [("noun_subj", "nom")],
     "Intransitive sentence with goal locative and dative arguments"),

    ("subj_nom-obj_acc-noun_goal-noun_dat-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{goal} {noun_pobj2}{dat} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj", "noun_pobj2"],
     [NOM, ACC, GOAL, DAT],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with goal locative and dative arguments"),

    # ── Adjunct: Instrumental + Source locative ───────────────────────────────
    ("subj_nom-noun_inst-noun_loc-verb.",
     "{noun_subj}{nom} {noun_pobj}{inst} {noun_pobj2}{loc} {verb}.",
     ["noun_subj", "noun_pobj", "noun_pobj2"],
     [NOM, INST, LOC],
     [("noun_subj", "nom"), ("noun_pobj", "inst")],
     "Intransitive sentence with instrumental and source locative arguments"),

    ("subj_nom-obj_acc-noun_inst-noun_loc-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{inst} {noun_pobj2}{loc} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj", "noun_pobj2"],
     [NOM, ACC, INST, LOC],
     [("noun_subj", "nom"), ("noun_dobj", "acc"), ("noun_pobj", "inst")],
     "Transitive sentence with instrumental and source locative arguments"),

    # ── Adjunct: Instrumental + Dative ────────────────────────────────────────
    ("subj_nom-noun_inst-noun_dat-verb.",
     "{noun_subj}{nom} {noun_pobj}{inst} {noun_pobj2}{dat} {verb}.",
     ["noun_subj", "noun_pobj", "noun_pobj2"],
     [NOM, INST, DAT],
     [("noun_subj", "nom"), ("noun_pobj", "inst")],
     "Intransitive sentence with instrumental and dative arguments"),

    ("subj_nom-obj_acc-noun_inst-noun_dat-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{inst} {noun_pobj2}{dat} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj", "noun_pobj2"],
     [NOM, ACC, INST, DAT],
     [("noun_subj", "nom"), ("noun_dobj", "acc"), ("noun_pobj", "inst")],
     "Transitive sentence with instrumental and dative arguments"),

    # ── Adjunct: Instrumental + Goal locative ────────────────────────────────
    ("subj_nom-noun_inst-noun_goal-verb.",
     "{noun_subj}{nom} {noun_pobj}{inst} {noun_pobj2}{goal} {verb}.",
     ["noun_subj", "noun_pobj", "noun_pobj2"],
     [NOM, INST, GOAL],
     [("noun_subj", "nom"), ("noun_pobj", "inst")],
     "Intransitive sentence with instrumental and goal locative arguments"),

    ("subj_nom-obj_acc-noun_inst-noun_goal-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{inst} {noun_pobj2}{goal} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj", "noun_pobj2"],
     [NOM, ACC, INST, GOAL],
     [("noun_subj", "nom"), ("noun_dobj", "acc"), ("noun_pobj", "inst")],
     "Transitive sentence with instrumental and goal locative arguments"),

    # ── Adjunct: Comitative (와/과) ────────────────────────────────────────────
    ("subj_nom-noun_com-verb.",
     "{noun_subj}{nom} {noun_pobj}{com} {verb}.",
     ["noun_subj", "noun_pobj"],
     [NOM, COM],
     [("noun_subj", "nom"), ("noun_pobj", "com")],
     "Intransitive sentence with comitative argument"),

    ("subj_nom-obj_acc-noun_com-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{com} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"],
     [NOM, ACC, COM],
     [("noun_subj", "nom"), ("noun_dobj", "acc"), ("noun_pobj", "com")],
     "Transitive sentence with comitative argument"),

    # ── Adjunct: Terminative (까지) ────────────────────────────────────────────
    ("subj_nom-noun_term-verb.",
     "{noun_subj}{nom} {noun_pobj}{term} {verb}.",
     ["noun_subj", "noun_pobj"],
     [NOM, TERM],
     [("noun_subj", "nom")],
     "Intransitive sentence with terminative argument"),

    ("subj_nom-obj_acc-noun_term-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{term} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"],
     [NOM, ACC, TERM],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with terminative argument"),

    # ── Adjunct: Initiative (부터) ─────────────────────────────────────────────
    ("subj_nom-noun_init-verb.",
     "{noun_subj}{nom} {noun_pobj}{init} {verb}.",
     ["noun_subj", "noun_pobj"],
     [NOM, INIT],
     [("noun_subj", "nom")],
     "Intransitive sentence with initiative argument"),

    ("subj_nom-obj_acc-noun_init-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{init} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"],
     [NOM, ACC, INIT],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "Transitive sentence with initiative argument"),
]


# ---------------------------------------------------------------------------
# Spatial noun frame specs
# (name, template_string, noun_names, case_specs, fc_pairs_noun, spatial_fc_case, description)  # noqa: E501
# noun_pobj appears bare (no case marker) — spatial_noun takes the locative marker.
# spatial_fc_case: case slot name requiring fc_agree with spatial_noun (inst only).
# ---------------------------------------------------------------------------

SPATIAL_FRAME_SPECS: list[tuple] = [
    ("subj_nom-spost_goal-verb.",
     "{noun_subj}{nom} {noun_pobj} {spatial_noun}{goal} {verb}.",
     ["noun_subj", "noun_pobj"], [NOM, GOAL], [("noun_subj", "nom")], None,
     "Intransitive with spatial relational noun + goal locative (위에, 앞에, etc.)"),

    ("subj_nom-obj_acc-spost_goal-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj} {spatial_noun}{goal} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"], [NOM, ACC, GOAL],
     [("noun_subj", "nom"), ("noun_dobj", "acc")], None,
     "Transitive with spatial relational noun + goal locative"),

    ("subj_nom-spost_loc-verb.",
     "{noun_subj}{nom} {noun_pobj} {spatial_noun}{loc} {verb}.",
     ["noun_subj", "noun_pobj"], [NOM, LOC], [("noun_subj", "nom")], None,
     "Intransitive with spatial relational noun + source locative (위에서, 앞에서, etc.)"),  # noqa: E501

    ("subj_nom-obj_acc-spost_loc-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj} {spatial_noun}{loc} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"], [NOM, ACC, LOC],
     [("noun_subj", "nom"), ("noun_dobj", "acc")], None,
     "Transitive with spatial relational noun + source locative"),

    ("subj_nom-spost_inst-verb.",
     "{noun_subj}{nom} {noun_pobj} {spatial_noun}{inst} {verb}.",
     ["noun_subj", "noun_pobj"], [NOM, INST], [("noun_subj", "nom")], "inst",
     "Intransitive with spatial relational noun + instrumental (위로, 쪽으로, etc.)"),

    ("subj_nom-obj_acc-spost_inst-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj} {spatial_noun}{inst} {verb}.",
     ["noun_subj", "noun_dobj", "noun_pobj"], [NOM, ACC, INST],
     [("noun_subj", "nom"), ("noun_dobj", "acc")], "inst",
     "Transitive with spatial relational noun + instrumental"),
]


# ---------------------------------------------------------------------------
# Complex postposition frame specs
# (name, template_string, has_dobj, gov_type, description)
# gov_type 'dat': noun_pobj directly precedes complex_postp_dat (no separate case slot).
# gov_type 'acc': noun_pobj takes acc (or pobj_acc if has_dobj), then complex_postp_acc.
# ---------------------------------------------------------------------------

COMPLEX_POSTP_FRAME_SPECS: list[tuple] = [
    ("subj_nom-noun_complex_dat-verb.",
     "{noun_subj}{nom} {noun_pobj}{complex_postp_dat} {verb}.",
     False, "dat",
     "Intransitive with DAT-governed complex postposition (에 대해서, etc.)"),

    ("subj_nom-obj_acc-noun_complex_dat-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{complex_postp_dat} {verb}.",
     True, "dat",
     "Transitive with DAT-governed complex postposition"),

    ("subj_nom-noun_acc_complex-verb.",
     "{noun_subj}{nom} {noun_pobj}{acc} {complex_postp_acc} {verb}.",
     False, "acc",
     "Intransitive with ACC-governed complex postposition (을 통해서, etc.)"),

    ("subj_nom-obj_acc-noun_acc_complex-verb.",
     "{noun_subj}{nom} {noun_dobj}{acc}"
     " {noun_pobj}{pobj_acc} {complex_postp_acc} {verb}.",
     True, "acc",
     "Transitive with ACC-governed complex postposition"),
]


# Progressive specs:
# (name, template_string, noun_slots, case_slots, fc_pairs, tense, description)
PROGRESSIVE_SPECS: list[tuple] = [
    ("subj_nom-verb_prog.",
     "{noun_subj}{nom} {verb}{aux}.",
     ["noun_subj"], [NOM], [("noun_subj", "nom")],
     "PRS", "Present progressive intransitive sentence"),

    ("subj_nom-obj_acc-verb_prog.",
     "{noun_subj}{nom} {noun_dobj}{acc} {verb}{aux}.",
     ["noun_subj", "noun_dobj"], [NOM, ACC],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "PRS", "Present progressive transitive sentence"),

    ("subj_nom-verb_past_prog.",
     "{noun_subj}{nom} {verb}{aux}.",
     ["noun_subj"], [NOM], [("noun_subj", "nom")],
     "PST", "Past progressive intransitive sentence"),

    ("subj_nom-obj_acc-verb_past_prog.",
     "{noun_subj}{nom} {noun_dobj}{acc} {verb}{aux}.",
     ["noun_subj", "noun_dobj"], [NOM, ACC],
     [("noun_subj", "nom"), ("noun_dobj", "acc")],
     "PST", "Past progressive transitive sentence"),
]


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def _adjunct_types(case_specs: list[tuple]) -> frozenset[str]:
    """Return the set of adjunct case types in a frame (anything beyond NOM/ACC)."""
    return frozenset(spec[0] for spec in case_specs if spec[0] not in _CORE_CASES)


def select_frame_specs(
    include_adjuncts: bool,
    adjunct_filter: frozenset[str],
    adjuncts_mode: str = "all",
) -> list[tuple]:
    """Filter FRAME_SPECS by adjunct inclusion rules.

    Base frames (NOM-only or NOM+ACC without obliques) are always included.
    Adjunct frames are filtered by adjunct_filter according to adjuncts_mode:

    - ``"all"`` (default): include a frame only if ALL of its adjunct case
      types are present in adjunct_filter. Use this when testing a single
      adjunct type in isolation — frames with extra adjunct types would
      introduce confounds.
    - ``"any"``: include a frame if ANY of its adjunct case types appears in
      adjunct_filter. Use this for broader coverage when confounds are
      acceptable (e.g., descriptive statistics, corpus-style runs).

    Parameters
    ----------
    include_adjuncts : bool
        Whether to include frames that contain oblique arguments.
    adjunct_filter : frozenset[str]
        Which adjunct case types are permitted.
    adjuncts_mode : str
        Matching strategy: ``"all"`` (strict, default) or ``"any"`` (permissive).

    Returns
    -------
    list[tuple]
        Filtered list of frame spec tuples from FRAME_SPECS.
    """
    selected = []
    for spec in FRAME_SPECS:
        adjuncts = _adjunct_types(spec[3])
        if not adjuncts:
            selected.append(spec)
        elif include_adjuncts:
            if adjuncts_mode == "all":
                if adjuncts.issubset(adjunct_filter):
                    selected.append(spec)
            else:  # "any"
                if adjuncts & adjunct_filter:
                    selected.append(spec)
    return selected


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------

def make_frame(
    name: str,
    template_string: str,
    noun_names: list[str],
    case_specs: list[tuple],
    fc_pairs: list[tuple],
    description: str,
) -> Template:
    """Build a standard NP + case marker Template from a FRAME_SPECS row."""
    slots = {n: noun_slot(n) for n in noun_names}
    slots.update({spec[0]: case_slot(*spec) for spec in case_specs})
    slots["verb"] = verb_slot()
    return Template(
        name=name,
        template_string=template_string,
        slots=slots,
        constraints=[fc_agree(n, m) for n, m in fc_pairs],
        description=description,
        language_code="kor",
    )


def make_progressive(
    name: str,
    template_string: str,
    noun_names: list[str],
    case_specs: list[tuple],
    fc_pairs: list[tuple],
    tense: str,
    description: str,
) -> Template:
    """Build a progressive Template using a participial verb stem and aux slot."""
    slots = {n: noun_slot(n) for n in noun_names}
    slots.update({spec[0]: case_slot(*spec) for spec in case_specs})
    slots["verb"] = ptcp_verb_slot()
    slots["aux"] = aux_slot(tense)
    return Template(
        name=name,
        template_string=template_string,
        slots=slots,
        constraints=[fc_agree(n, m) for n, m in fc_pairs],
        description=description,
        language_code="kor",
    )


def make_spatial_frame(
    name: str,
    template_string: str,
    noun_names: list[str],
    case_specs: list[tuple],
    fc_pairs_noun: list[tuple],
    spatial_fc_case: str | None,
    description: str,
) -> Template:
    """Build a frame with a postpositional spatial/temporal relational noun.

    Parameters
    ----------
    fc_pairs_noun : list[tuple]
        Final-consonant pairs for regular noun-case slots (excludes spatial_noun).
    spatial_fc_case : str | None
        Case slot name to agree with spatial_noun for allomorphy (only "inst");
        None for invariant goal/loc markers.
    """
    slots = {n: noun_slot(n) for n in noun_names}
    slots.update({spec[0]: case_slot(*spec) for spec in case_specs})
    slots["spatial_noun"] = spatial_noun_slot()
    slots["verb"] = verb_slot()
    constraints = [fc_agree(n, m) for n, m in fc_pairs_noun]
    if spatial_fc_case is not None:
        constraints.append(fc_agree("spatial_noun", spatial_fc_case))
    return Template(
        name=name,
        template_string=template_string,
        slots=slots,
        constraints=constraints,
        description=description,
        language_code="kor",
    )


def make_complex_postp_frame(
    name: str,
    template_string: str,
    has_dobj: bool,
    gov_type: str,
    description: str,
) -> Template:
    """Build a frame with a complex multi-morpheme postposition.

    Parameters
    ----------
    has_dobj : bool
        Whether the frame has a direct object (noun_dobj + acc).
    gov_type : str
        ``"dat"``: noun_pobj directly precedes complex_postp_dat (particle 에 is
        built into the postposition form, no separate case slot for noun_pobj).
        ``"acc"``: noun_pobj takes acc (intransitive) or pobj_acc (transitive),
        followed by complex_postp_acc.
    """
    slots: dict[str, Slot] = {
        "noun_subj": noun_slot("noun_subj"),
        "noun_pobj": noun_slot("noun_pobj"),
        "nom": case_slot(*NOM),
        "verb": verb_slot(),
    }
    constraints = [fc_agree("noun_subj", "nom")]

    if has_dobj:
        slots["noun_dobj"] = noun_slot("noun_dobj")
        slots["acc"] = case_slot(*ACC)
        constraints.append(fc_agree("noun_dobj", "acc"))

    if gov_type == "dat":
        slots["complex_postp_dat"] = complex_postp_dat_slot()
    else:  # acc
        if has_dobj:
            slots["pobj_acc"] = case_slot(*POBJ_ACC)
            constraints.append(fc_agree("noun_pobj", "pobj_acc"))
        else:
            slots["acc"] = case_slot(*ACC)
            constraints.append(fc_agree("noun_pobj", "acc"))
        slots["complex_postp_acc"] = complex_postp_acc_slot()

    return Template(
        name=name,
        template_string=template_string,
        slots=slots,
        constraints=constraints,
        description=description,
        language_code="kor",
    )


def comp_verb_adn_slot() -> Slot:
    """Adnominal present verb slot restricted to bleached verbs: 하는, 가는, 오는, etc.

    The source=='bleached' filter avoids the ~1,988 UniMorph V.ADN.PRS entries and
    restricts the slot to the 7 bleached forms in comp_verbs.jsonl, keeping the
    complement template cross-product manageable (~7 × 2,674 × 25 ≈ 469K items).
    """
    return Slot(
        name="comp_verb_adn",
        description=(
            "complement clause verb (adnominal present form, bleached verbs only)"
        ),
        constraints=[Constraint(expression=(
            "self.features.get('verb_form')=='V.ADN.PRS'"
            " and self.features.get('source')=='bleached'"
        ))],
    )


def comp_verb_decl_slot() -> Slot:
    """Present declarative verb slot (bleached verbs only): 한다, 간다, 온다, etc.

    The source=='bleached' filter avoids the ~2,674 UniMorph PRS entries and
    restricts the slot to the 7 bleached forms in comp_verbs.jsonl.
    """
    return Slot(
        name="comp_verb_decl",
        description=(
            "complement clause verb (present declarative form, bleached verbs only)"
        ),
        constraints=[Constraint(expression=(
            "self.features.get('pos')=='V' and self.features.get('tense')=='PRS'"
            " and self.features.get('source')=='bleached'"
        ))],
    )


def _comp_frame(
    name: str,
    template_string: str,
    comp_verb_slot: Slot,
    description: str,
) -> Template:
    """Shared builder for complement clause templates."""
    return Template(
        name=name,
        template_string=template_string,
        slots={
            "noun_subj": noun_slot("noun_subj"),
            "nom":       case_slot("nom", "PART.NOM", "nominative case marker"),
            "comp_subj": noun_slot("comp_subj"),
            "comp_nom":  case_slot(
                "comp_nom", "PART.NOM", "complement clause nominative"
            ),
            comp_verb_slot.name: comp_verb_slot,
            "verb": verb_slot(),
        },
        constraints=[
            fc_agree("noun_subj", "nom"),
            fc_agree("comp_subj", "comp_nom"),
        ],
        description=description,
        language_code="kor",
    )


def make_complement_templates() -> list[Template]:
    """Three complement clause types encoding the presuppositionality contrast.

    - Nominative (-는 것을):    -ing       (알다, 깨닫다, 후회하다)
    - Quotative (-다고):        that      (말하다, 생각하다, 믿다)
    - Indirect Q (-는지):    whether/if   (알다, 궁금하다, 모르다)
    """
    nominative = _comp_frame(
        name="subj-verb-neun_geoseul.",
        template_string="{noun_subj}{nom} {comp_subj}{comp_nom} {comp_verb_adn} 것을 {verb}.",  # noqa: E501
        comp_verb_slot=comp_verb_adn_slot(),
        description="Nominative nominalized complement (-는 것을); truth presupposed",
    )
    quotative = _comp_frame(
        name="subj-verb-dago.",
        template_string="{noun_subj}{nom} {comp_subj}{comp_nom} {comp_verb_decl}고 {verb}.",  # noqa: E501
        comp_verb_slot=comp_verb_decl_slot(),
        description="Quotative complement (-다고); non-factive",
    )
    indirect_q = _comp_frame(
        name="subj-verb-neungi.",
        template_string="{noun_subj}{nom} {comp_subj}{comp_nom} {comp_verb_adn}지 {verb}.",  # noqa: E501
        comp_verb_slot=comp_verb_adn_slot(),
        description="Indirect question complement (-는지); question embedding",
    )
    return [nominative, quotative, indirect_q]


# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------

def main(
    include: list[str] | None = None,
    adjuncts: list[str] | None = None,
    adjuncts_mode: str = "all",
    verb_limit: int | None = None,
) -> None:
    """Generate Template objects and write to templates/generic_frames.jsonl.

    Parameters
    ----------
    include : list[str] | None
        Template groups to add on top of base frames. Options: adjuncts,
        progressive, complement, spatial, complex. None (default) includes all.
    adjuncts : list[str] | None
        Adjunct case types to allow: dat, loc, inst, goal, com, term, init.
        Only relevant when adjuncts is in include. None means all types.
    adjuncts_mode : str
        How to apply the adjuncts filter: ``"all"`` (default, strict — a frame
        is included only if ALL its adjunct types are in the filter) or
        ``"any"`` (permissive — a frame is included if ANY adjunct type matches).
    verb_limit : int | None
        Truncate output to first N templates (for quick sanity checks).
    """
    _all_groups = ["adjuncts", "progressive", "complement", "spatial", "complex"]
    active_groups   = frozenset(include  if include  is not None else _all_groups)
    active_adjuncts = frozenset(
        adjuncts if adjuncts is not None
        else ["dat", "loc", "inst", "goal", "com", "term", "init"]
    )

    frames = select_frame_specs(
        include_adjuncts="adjuncts" in active_groups,
        adjunct_filter=active_adjuncts,
        adjuncts_mode=adjuncts_mode,
    )
    generic_templates: list[Template] = [make_frame(*spec) for spec in frames]

    if "progressive" in active_groups:
        generic_templates += [make_progressive(*spec) for spec in PROGRESSIVE_SPECS]

    if "complement" in active_groups:
        generic_templates += make_complement_templates()

    if "spatial" in active_groups:
        generic_templates += [make_spatial_frame(*spec) for spec in SPATIAL_FRAME_SPECS]

    if "complex" in active_groups:
        generic_templates += [
            make_complex_postp_frame(*spec) for spec in COMPLEX_POSTP_FRAME_SPECS
        ]

    if verb_limit:
        generic_templates = generic_templates[:verb_limit]

    with open("./templates/generic_frames.jsonl", "w") as f:
        for template in generic_templates:
            f.write(template.model_dump_json() + "\n")

    n_base    = sum(1 for spec in FRAME_SPECS if not _adjunct_types(spec[3]))
    n_adj     = len(frames) - n_base
    n_prog    = len(PROGRESSIVE_SPECS) if "progressive" in active_groups else 0
    n_comp    = 3 if "complement" in active_groups else 0
    n_spatial = len(SPATIAL_FRAME_SPECS) if "spatial" in active_groups else 0
    n_complex = len(COMPLEX_POSTP_FRAME_SPECS) if "complex" in active_groups else 0
    print(
        f"Generated {len(generic_templates)} templates "
        f"(base={n_base}, adjuncts={n_adj}, progressive={n_prog}, complement={n_comp}, "
        f"spatial={n_spatial}, complex={n_complex})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Template objects for Korean argument structure dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s                                         # all groups (default)
  %(prog)s --include                               # base only (fast test)
  %(prog)s --include adjuncts                      # base + all adjuncts
  %(prog)s --include adjuncts --adjuncts dat loc   # base + dat/loc adjuncts only
  %(prog)s --include adjuncts progressive spatial  # adjuncts + progressive + spatial
  %(prog)s --include spatial complex               # spatial + complex postpositions
        """,
    )
    parser.add_argument(
        "--include",
        nargs="*",
        choices=["adjuncts", "progressive", "complement", "spatial", "complex"],
        default=["adjuncts", "progressive", "complement", "spatial", "complex"],
        metavar="GROUP",
        help=(
            "Template groups to include on top of base (intransitive + transitive). "
            "Choices: adjuncts, progressive, complement, spatial, complex. "
            "Pass --include with no arguments to generate base only. "
            "Default: all groups."
        ),
    )
    parser.add_argument(
        "--adjuncts",
        nargs="+",
        choices=["dat", "loc", "inst", "goal", "com", "term", "init"],
        default=["dat", "loc", "inst", "goal", "com", "term", "init"],
        help=(
            "Which adjunct case types to include. "
            "Only applies when 'adjuncts' is in --include. Default: all seven. "
            "See --adjuncts-mode for how multi-adjunct frames are handled."
        ),
    )
    parser.add_argument(
        "--adjuncts-mode",
        choices=["all", "any"],
        default="all",
        help=(
            "How to apply --adjuncts when a frame has multiple adjunct types. "
            "'all' (default): include only if ALL adjunct types are in --adjuncts "
            "(strict — avoids confounds when isolating a single case type). "
            "'any': include if ANY adjunct type matches "
            "(permissive — broader coverage, may mix case types)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate output to first N templates (for quick sanity checks).",
    )
    args = parser.parse_args()
    main(
        include=args.include,
        adjuncts=args.adjuncts,
        adjuncts_mode=args.adjuncts_mode,
        verb_limit=args.limit,
    )
