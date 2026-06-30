#!/usr/bin/env python3
"""Generate JSONL lexicon files for argument structure alternation dataset.

Output:
    lexicons/verbs.jsonl
    lexicons/bleached_nouns.jsonl
    lexicons/bleached_verbs.jsonl
    lexicons/bleached_adjectives.jsonl
    lexicons/case_markers.jsonl
    lexicons/auxiliary_verbs.jsonl
    lexicons/spatial_nouns.jsonl
    lexicons/complex_postpositions.jsonl
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
import requests

from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon


def main(verb_limit: int | None = None, save_csv: bool = True) -> None:
    """Generate all lexicon JSONL files for the Korean argument structure dataset.

    Parameters
    ----------
    verb_limit : int | None
        Truncate the verb lexicon to the first N unique lemmas. None means all.
    save_csv : bool
        Whether to also write intermediate CSV files to resources/.
    """
    # Set up paths
    base_dir = Path(__file__).parent
    lexicons_dir = base_dir / "lexicons"
    resources_dir = base_dir / "resources"

    # Ensure directories exist
    lexicons_dir.mkdir(exist_ok=True)
    resources_dir.mkdir(exist_ok=True)

    # Generate verbs lexicon from UniMorph Korean data
    print("=" * 80)
    print("GENERATING VERBS LEXICON FROM UNIMORPH KOR DATA")
    print("=" * 80)

    url = "https://raw.githubusercontent.com/unimorph/kor/master/kor"
    unimorph_kor = requests.get(url)
    data = unimorph_kor.text
    lexicon = Lexicon(name="verbs")
    verbs: list[LexicalItem] = []

    chinese_char_regex = re.compile(r'[\u4e00-\u9fff]')
    romanization_regex = re.compile(r'[a-zA-Z\-]')

    if verb_limit is not None:
        print(f"[TEST MODE] Limiting to first {verb_limit} verbs")

    base_verb: set[str] = set()

    for line in data.splitlines():
        parts = line.strip().split('\t')
        if len(parts) == 3:
            base, form, tags = parts

            if verb_limit is not None and len(base_verb) > verb_limit:
                break

            if '-' in base or chinese_char_regex.search(base):
                continue

            form = romanization_regex.sub('', form).strip()

            base_verb.add(base)

            if 'V;DECL;FIN;PST;FORM' == tags or 'ADJ;DECL;FIN;PST;FORM' == tags:  # past
                verb = LexicalItem(
                    lemma=base,
                    form=form,
                    language_code="kor",
                    features={
                        "pos": "V", "finiteness": "FIN",
                        "tense": "PST", "unimorph_features": tags,
                    },
                    source="UniMorph",
                )
                verbs.append(verb)

            elif (  # present
                'V;DECL;FIN;PRS;FORM' == tags
                or 'ADJ;DECL;FIN;PRS;FORM' == tags
            ):
                verb = LexicalItem(
                    lemma=base,
                    form=form,
                    language_code="kor",
                    features={
                        "pos": "V", "finiteness": "FIN",
                        "tense": "PRS", "unimorph_features": tags,
                    },
                    source="UniMorph",
                )
                verbs.append(verb)

            elif 'V.PTCP;PRS' == tags:  # adnominal present (-는): 가는, 먹는, 아는
                verb = LexicalItem(
                    lemma=base,
                    form=form,
                    language_code="kor",
                    features={
                        "pos": "V", "finiteness": "NFIN",
                        "verb_form": "V.ADN.PRS", "unimorph_features": tags,
                    },
                    source="UniMorph",
                )
                verbs.append(verb)

            elif 'V.CVB;NFIN;CONJ' == tags or 'ADJ.CVB;NFIN;CONJ' == tags:  # gerund
                _ptcp_feats = {
                    "pos": "V", "finiteness": "NFIN",
                    "verb_form": "V.PTCP", "unimorph_features": tags,
                }
                if form.endswith("고"):
                    verb = LexicalItem(
                        lemma=base, form=form,
                        language_code="kor", features=_ptcp_feats,
                        source="UniMorph",
                    )
                    verbs.append(verb)

                # Handling UniMorph annotation inconsistencies
                elif '-' in form:
                    verb = LexicalItem(
                        lemma=base, form=form.split('-')[0],
                        language_code="kor", features=_ptcp_feats,
                        source="UniMorph",
                    )
                    verbs.append(verb)

                elif '\'' in form:
                    verb = LexicalItem(
                        lemma=base, form=form.split('\'')[0] + " 있다",
                        language_code="kor", features=_ptcp_feats,
                        source="UniMorph",
                    )
                    verbs.append(verb)

                elif form.endswith("면"):
                    verb = LexicalItem(
                        lemma=base, form=form[:-1] + "고",
                        language_code="kor", features=_ptcp_feats,
                        source="UniMorph",
                    )
                    verbs.append(verb)
                    

    lexicon.add_many(verbs)

    print(f"Total base verbs found: {len(base_verb)}")
    print(f"Total verbs found: {len(verbs)}")
    
    lexicon.to_jsonl("./lexicons/verbs.jsonl")

    # 2. Generate bleached nouns, verbs, adjectives, and case markers lexicons
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED NOUNS, VERBS, ADJECTIVES, AND CASE MARKERS LEXICONS")
    print("=" * 80)

    # Generate nominative Case Markers csv and jsonl
    case_markers = pd.DataFrame(columns=['marker', 'case', 'final_consonant'])

    marker = ['이', '가', '을', '를', '으로', '로', '에게', '에서', '에',
              '와', '과', '의', '까지', '부터', '처럼', '같이', '없이']
              # TODO: remove '없이' — adverbial postposition without a noun
    case = ['NOM', 'NOM', 'ACC', 'ACC', 'INST', 'INST', 'DAT', 'LOC.SRC', 'LOC.GOAL',
            'COM', 'COM', 'GEN', 'TERM', 'INIT', 'SIM', 'SIM', 'PRIV']
    final_consonant = ['yes', 'no', 'yes', 'no', 'yes', 'no', None, None, None,
                       'no', 'yes', None, None, None, None, None, None]

    case_markers['marker'] = marker
    case_markers['case'] = case
    case_markers['final_consonant'] = final_consonant

    if save_csv:
        case_markers.to_csv('./resources/case_markers.csv', index=False)

    lexicon = Lexicon(name="case_markers")
    with open("resources/case_markers.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row["case"]
            fc = row["final_consonant"]
            pos_map = {
                "NOM": "PART.NOM", "ACC": "PART.ACC", "INST": "PART.INST",
                "DAT": "PART.DAT",
                "LOC.SRC": "PART.LOC.SRC", "LOC.GOAL": "PART.LOC.GOAL",
            }
            if case in pos_map:
                item = LexicalItem(
                    lemma=row["marker"],
                    language_code="kor",
                    features={
                        "pos": pos_map[case], "case": case, "final_consonant": fc,
                    },
                )
                lexicon.add(item)
            remaining_map = {
                "COM": "PART.COM", "GEN": "PART.GEN", "TERM": "PART.TERM",
                "INIT": "PART.INIT", "SIM": "PART.SIM", "PRIV": "PART.PRIV",
            }
            if case in remaining_map:
                item = LexicalItem(
                    lemma=row["marker"],
                    language_code="kor",
                    features={
                        "pos": remaining_map[case], "case": case, "final_consonant": fc,
                    },
                )
                lexicon.add(item)
    lexicon.to_jsonl("./lexicons/case_markers.jsonl")

    print(f"Created {len(case_markers)} case markers.")

    # Generate bleached nouns csv and jsonl
    bleached_nouns = pd.DataFrame(
        columns=['word', 'semantic_class', 'number', 'countability', 'final_consonant']
    )

    # Five singular bleached nouns — one per core semantic class.
    # Plurals excluded: Korean 들 suffix produces multi-subword tokens under
    # klue/bert-base.
    # Five entries keep exhaustive cross-products manageable for test runs.
    word = ['사람', '단체', '물건', '장소', '사건']
    semantic_class = ['animate', 'animate', 'inanimate_object', 'location', 'event']
    number = ['singular'] * len(word)
    countability = ['countable'] * len(word)
    final_consonant = [
        'yes',  
        'no',   
        'yes',  
        'no',   
        'yes',  
    ]

    bleached_nouns['word'] = word
    bleached_nouns['semantic_class'] = semantic_class
    bleached_nouns['number'] = number
    bleached_nouns['countability'] = countability
    bleached_nouns['final_consonant'] = final_consonant

    if save_csv:
        bleached_nouns.to_csv('./resources/bleached_nouns.csv', index=False)

    lexicon = Lexicon(name="bleached_nouns")

    with open("resources/bleached_nouns.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="kor",
                features={
                    "pos": "NOUN",
                    "number": row["number"],
                    "countability": row["countability"],
                    "final_consonant": row["final_consonant"],
                },
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/bleached_nouns.jsonl")

    print(f"Created {len(bleached_nouns)} bleached nouns.")


    # Generate bleached verbs csv and jsonl
    bleached_verbs = pd.DataFrame(columns=[
        'word', 'semantic_class', 'aspect', 'valency',
        'tenseless_vp', 'gerund', 'tenseless_clause',
        'infinitival_clause', 'tensed_clause',
    ])

    # 'be' functions differently in Korean
    word = ['하다', '가지다', '가다', '갖다', '만들다', '일어나다', '오다']
    semantic_class = [
        'activity', 'state', 'change', 'change', 'causation', 'event', 'change',
    ]
    aspect = [
        'dynamic', 'stative', 'dynamic', 'dynamic', 'dynamic', 'dynamic', 'dynamic',
    ]
    valency = [
        'transitive', 'transitive', 'intransitive',
        'transitive', 'transitive', 'intransitive', 'intransitive',
    ]
    tenseless_vp = [
        '{{ object }} 하다', '{{ object }} 가지다', '가다',
        '{{ object }} 갖다', '{{ object }} 만들다', '일어나다', '오다',
    ]
    gerund = [
        '{{ object }} 하고', '{{ object }} 가지고', '가고',
        '{{ object }} 갖고', '{{ object }} 만들고', '일어나고', '오고',
    ]
    tenseless_clause = [
        '{{ subject }} {{ object }} 하다',
        '{{ subject }} {{ object }} 가지다',
        '{{ subject }} 가다',
        '{{ subject }} {{ object }} 갖다',
        '{{ subject }} {{ object }} 만들다',
        '{{ subject }} 일어나다',
        '{{ subject }} 오다',
    ]
    tensed_clause = [
        '{{ subject }} {{ object }} 했다',
        '{{ subject }} {{ object }} 가졌다',
        '{{ subject }} 갔다',
        '{{ subject }} {{ object }} 갖았다',
        '{{ subject }} {{ object }} 만들었다',
        '{{ subject }} 일어났다',
        '{{ subject }} 왔다',
    ]

    bleached_verbs['word'] = word
    bleached_verbs['semantic_class'] = semantic_class
    bleached_verbs['aspect'] = aspect
    bleached_verbs['valency'] = valency
    bleached_verbs['tenseless_vp'] = tenseless_vp
    bleached_verbs['gerund'] = gerund
    bleached_verbs['tenseless_clause'] = tenseless_clause
    # same as tenseless_clause in Korean
    bleached_verbs['infinitival_clause'] = tenseless_clause
    bleached_verbs['tensed_clause'] = tensed_clause

    if save_csv:
        bleached_verbs.to_csv("./resources/bleached_verbs.csv", index=False)

    lexicon = Lexicon(name="bleached_verbs")

    with open("resources/bleached_verbs.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="kor",
                features={
                    "pos": "V", "tense": "", "semantic_class": row["semantic_class"],
                }
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/bleached_verbs.jsonl")

    print(f"Created {len(bleached_verbs)} bleached verbs.")

    # Generate comp_verbs lexicon (bleached verb forms for complement templates)
    # Each of the 7 bleached verbs contributes two forms:
    #   - V.ADN.PRS: adnominal present (-는) for 것을 (factive) and 는지 (indirect Q)
    #   - PRS:       present declarative for 다고 (quotative) template
    # These are exhaustive slots — only 14 entries total, cross-products manageable.
    # The source="bleached" feature distinguishes them from ~10k UniMorph V.ADN.PRS.
    comp_verb_entries = [
        # (lemma,       adn_form,    decl_form,   semantic_class)
        ("하다",        "하는",       "한다",       "activity"),
        ("가지다",      "가지는",     "가진다",     "state"),
        ("가다",        "가는",       "간다",       "change"),
        ("갖다",        "갖는",       "갖는다",     "change"),
        ("만들다",      "만드는",     "만든다",     "causation"),  # ㄹ drops before ㄴ
        ("일어나다",    "일어나는",   "일어난다",   "event"),
        ("오다",        "오는",       "온다",       "change"),
    ]

    lexicon = Lexicon(name="comp_verbs")
    for lemma, adn_form, decl_form, sem_class in comp_verb_entries:
        lexicon.add(LexicalItem(
            lemma=lemma,
            form=adn_form,
            language_code="kor",
            features={"pos": "V", "verb_form": "V.ADN.PRS", "source": "bleached",
                      "semantic_class": sem_class},
            source="bleached",
        ))
        lexicon.add(LexicalItem(
            lemma=lemma,
            form=decl_form,
            language_code="kor",
            features={"pos": "V", "tense": "PRS", "source": "bleached",
                      "semantic_class": sem_class},
            source="bleached",
        ))

    lexicon.to_jsonl("./lexicons/comp_verbs.jsonl")
    n_comp_verbs = len(comp_verb_entries) * 2
    n_adn = len(comp_verb_entries)
    print(
        f"Created {n_comp_verbs} comp verb forms "
        f"({n_adn} V.ADN.PRS + {n_adn} PRS declarative)."
    )

    # Generate bleached adjectives csv and jsonl
    bleached_adjectives = pd.DataFrame(
        columns=[
            'word', 'semantic_class', 'gradability', 'stage_vs_individual', 'notes',
        ]
    )

    # '다른'/'같은' = different/same in Korean
    word = [
        '좋은', '나쁜', '맞는', '틀린', '괜찮은',
        '확실한', '준비된', '끝난', '다른', '같은',
    ]
    semantic_class = [
        'evaluation', 'evaluation', 'evaluation', 'evaluation', 'evaluation',
        'epistemic', 'aspectual', 'aspectual', 'comparison', 'comparison',
    ]
    gradability = [
        'gradable', 'gradable', 'non-gradable', 'non-gradable', 'gradable',
        'non-gradable', 'non-gradable', 'non-gradable', 'gradable', 'gradable',
    ]
    stage_vs_individual = [
        'individual', 'individual', 'stage', 'stage', 'stage',
        'stage', 'stage', 'stage', 'individual', 'individual',
    ]
    notes = ['', '', '', '', '', '', '', '', '', '']

    bleached_adjectives['word'] = word
    bleached_adjectives['semantic_class'] = semantic_class
    bleached_adjectives['gradability'] = gradability
    bleached_adjectives['stage_vs_individual'] = stage_vs_individual
    bleached_adjectives['notes'] = notes

    if save_csv:
        bleached_adjectives.to_csv("./resources/bleached_adjectives.csv", index=False)

    lexicon = Lexicon(name="bleached_adjectives")

    with open("resources/bleached_adjectives.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="kor",
                features={"pos": "ADJ", "semantic_class": row["semantic_class"]}
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/bleached_adjectives.jsonl")

    print(f"Created {len(bleached_adjectives)} bleached adjectives.")

    # Generate auxiliary verbs (for progressive) csv and jsonl
    auxiliary_verbs = pd.DataFrame(columns=['lemma', 'form', 'pos', 'tense'])

    lemma = ['있다', '있다']
    form = ['있다', '있었다']
    pos = ['AUX', 'AUX']
    tense = ['PRS', 'PST']

    auxiliary_verbs['lemma'] = lemma
    auxiliary_verbs['form'] = form
    auxiliary_verbs['pos'] = pos
    auxiliary_verbs['tense'] = tense

    if save_csv:
        auxiliary_verbs.to_csv("./resources/auxiliary_verbs.csv", index=False)

    lexicon = Lexicon(name="auxiliary_verbs")

    with open("resources/auxiliary_verbs.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["lemma"],
                form=row["form"],
                language_code="kor",
                features={"pos": row["pos"], "tense": row["tense"]}
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/auxiliary_verbs.jsonl")

    print(f"Created {len(auxiliary_verbs)} auxiliary verbs.")

    # Generate spatial nouns (postpositional relational nouns, 의존명사)
    print("\n" + "=" * 80)
    print("GENERATING SPATIAL NOUNS LEXICON")
    print("=" * 80)

    spatial_nouns = pd.DataFrame(
        columns=[
            'word', 'spatial_class', 'final_consonant', 'eng_gloss', 'compatible_cases',
        ]
    )

    sn_word = [
        '위', '아래', '밑', '앞', '뒤', '옆', '안', '밖', '사이',
        '주위', '건너편', '너머', '가운데', '근처', '쪽', '전', '후', '동안', '이후',
    ]
    sn_class = [
        'vertical_up', 'vertical_down', 'vertical_down', 'front', 'back', 'side',
        'interior', 'exterior', 'medial', 'surround', 'across', 'beyond', 'medial',
        'proximal', 'directional', 'temporal', 'temporal', 'temporal', 'temporal',
    ]
    sn_fc = ['no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no',
             'no', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no']
    sn_gloss = [
        'above/on/over/upon/up', 'below/under/down', 'beneath/underneath/under',
        'in front of/before (spatial)', 'behind/back/after (spatial)',
        'beside/by/next to',
        'inside/within/in', 'outside/out/off', 'between/among', 'around/round',
        'across/opposite side', 'beyond', 'among/in the middle of', 'near/nearby',
        'toward/towards', 'before (temporal)', 'after (temporal)',
        'during/for (duration)', 'since/after (temporal)',
    ]
    sn_cases = [
        'goal,loc,inst', 'goal,loc,inst', 'goal,loc,inst', 'goal,loc', 'goal,loc',
        'goal,loc', 'goal,loc,inst', 'goal,loc', 'goal,loc', 'goal,loc',
        'goal', 'goal', 'goal,loc', 'goal', 'inst', 'goal', 'goal', 'goal', 'goal,inst',
    ]

    spatial_nouns['word'] = sn_word
    spatial_nouns['spatial_class'] = sn_class
    spatial_nouns['final_consonant'] = sn_fc
    spatial_nouns['eng_gloss'] = sn_gloss
    spatial_nouns['compatible_cases'] = sn_cases

    if save_csv:
        spatial_nouns.to_csv('./resources/spatial_nouns.csv', index=False)

    lexicon = Lexicon(name="spatial_nouns")

    with open("resources/spatial_nouns.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="kor",
                features={
                    "pos": "NPOST",
                    "spatial_class": row["spatial_class"],
                    "final_consonant": row["final_consonant"],
                    "eng_gloss": row["eng_gloss"],
                    "compatible_cases": row["compatible_cases"],
                }
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/spatial_nouns.jsonl")
    print(f"Created {len(spatial_nouns)} spatial nouns.")

    # Generate complex postpositions (다단어 후치사)
    print("\n" + "=" * 80)
    print("GENERATING COMPLEX POSTPOSITIONS LEXICON")
    print("=" * 80)

    complex_postpositions = pd.DataFrame(columns=['form', 'gov_case', 'eng_gloss'])

    cp_form = [
        # DAT-governed: initial particle 에 is built into the form
        '에 대해서', '에 관해서', '에도 불구하고',
        '에 의해서', '에 반해서', '에 걸쳐서',
        # ACC-governed: preceded by separate 을/를 slot
        '통해서', '위해서', '따라', '지나서', '제외하고',
    ]
    cp_gov = ['DAT', 'DAT', 'DAT', 'DAT', 'DAT', 'DAT',
              'ACC', 'ACC', 'ACC', 'ACC', 'ACC']
    cp_gloss = [
        'about/concerning/regarding', 'concerning/regarding', 'despite/in spite of',
        'by (passive agent)', 'against/contrary to', 'throughout/across',
        'through/via', 'for (beneficiary)', 'along/according to',
        'past/after passing', 'except/excluding',
    ]

    complex_postpositions['form'] = cp_form
    complex_postpositions['gov_case'] = cp_gov
    complex_postpositions['eng_gloss'] = cp_gloss

    if save_csv:
        complex_postpositions.to_csv(
            './resources/complex_postpositions.csv', index=False
        )

    lexicon = Lexicon(name="complex_postpositions")

    with open("resources/complex_postpositions.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["form"],
                form=row["form"],
                language_code="kor",
                features={
                    "pos": "POSTP",
                    "gov_case": row["gov_case"],
                    "eng_gloss": row["eng_gloss"],
                }
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/complex_postpositions.jsonl")
    print(f"Created {len(complex_postpositions)} complex postpositions.")

    # Summary
    print("\n" + "=" * 80)
    print("LEXICON GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {9} lexicon files:")
    print(f"  1. verbs.jsonl:                  {len(verbs)} entries")
    print(f"  2. bleached_nouns.jsonl:          {len(bleached_nouns)} entries")
    print(f"  3. bleached_verbs.jsonl:          {len(bleached_verbs)} entries")
    print(f"  4. bleached_adjectives.jsonl:     {len(bleached_adjectives)} entries")
    print(f"  5. case_markers.jsonl:            {len(case_markers)} entries")
    print(f"  6. auxiliary_verbs.jsonl:         {len(auxiliary_verbs)} entries")
    print(f"  7. spatial_nouns.jsonl:           {len(spatial_nouns)} entries")
    print(f"  8. complex_postpositions.jsonl:   {len(complex_postpositions)} entries")
    print(f"  9. comp_verbs.jsonl:              {n_comp_verbs} entries")
    print(f"\nAll files saved to: {lexicons_dir}/")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Generate JSONL lexicon files for argument structure dataset"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of VerbNet verbs to process (for testing)",
    )

    parser.add_argument(
        "--save_csv",
        type=bool,
        default=True,
        help=(
            "Whether to save CSV files during processing "
            "(default: True, must run if first time)"
        ),
    )
    args = parser.parse_args()

    main(verb_limit=args.limit, save_csv=args.save_csv)