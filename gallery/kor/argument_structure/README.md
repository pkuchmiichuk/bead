# Korean Argument Structure

## Quick Start

```bash
make data          # Quick test run (100 verbs, 50 filled templates, 2 lists)
make data-full     # Production run (all verbs, all pairs)
make deployment    # Generate jsPsych deployment for 2 lists (local test)
make help          # Show all available targets
```

## Pipeline Stages

| Step | Script | Make target |
|------|--------|-------------|
| 1. Lexicons | `generate_lexicons.py` | `make lexicons` |
| 2. Templates | `generate_templates.py` | `make templates` / `make templates-full` |
| 3. Fill templates | `fill_templates.py` | `make fill-templates` / `make fill-templates-full` |
| 4. 2AFC pairs | `create_2afc_pairs.py` | `make 2afc-pairs` / `make 2afc-pairs-full` |
| 5. Lists | `generate_lists.py` | `make lists` |
| 6. Deployment | `generate_deployment.py` | `make deployment` / `make deployment-full` |

---

### Stage 1 — Lexicons (`make lexicons`)

Generates 9 JSONL files under `lexicons/`:

| File | Entries | Contents |
|------|---------|----------|
| `verbs.jsonl` | ~10k | UniMorph Korean verbs in PST/PRS/ADN/PTCP forms |
| `bleached_nouns.jsonl` | 5 | One placeholder noun per semantic class (animate, inanimate, location, event, abstract) |
| `bleached_verbs.jsonl` | 7 | Semantically light verbs (하다, 가다, 오다, …) |
| `bleached_adjectives.jsonl` | 10 | Light adjectives |
| `case_markers.jsonl` | 17 | NOM/ACC/DAT/INST/LOC.SRC/LOC.GOAL + COM/GEN/TERM/INIT/SIM/PRIV particles |
| `auxiliary_verbs.jsonl` | 2 | Progressive auxiliary 있다 (PRS/PST) |
| `spatial_nouns.jsonl` | 19 | Relational nouns (위, 아래, 앞, 뒤, 안, 밖, …) |
| `complex_postpositions.jsonl` | 11 | Multi-morpheme postpositions (DAT-governed: 6, ACC-governed: 5) |
| `comp_verbs.jsonl` | 14 | Complement-clause verb forms (7 ADN.PRS + 7 PRS declarative) |

Case markers follow Korean allomorphy: e.g. NOM is 이 after a consonant-final noun, 가 after a vowel-final noun. This is handled automatically during template filling via the `fc_agree` constraint.

---

### Stage 2 — Templates (`make templates-full`)

Generates `templates/generic_frames.jsonl` with 47 abstract frames. Each frame is a sentence schema with labeled slots (e.g. `{noun_subj}`, `{nom}`, `{verb}`).

**Base frames (2)** — intransitive and transitive:
```
{noun_subj}{nom} {verb}.
{noun_subj}{nom} {noun_dobj}{acc} {verb}.
```

**Adjunct frames (28)** — base frames extended with one or two adjuncts. Adjunct types:

| Type | Example slot added | Example frame |
|------|--------------------|---------------|
| `dat` | `{noun_iobj}{dat}` | `{noun_subj}{nom} {noun_iobj}{dat} {verb}.` |
| `loc` | `{noun_loc}{loc}` | `{noun_subj}{nom} {noun_loc}{loc} {verb}.` |
| `inst` | `{noun_inst}{inst}` | `{noun_subj}{nom} {noun_inst}{inst} {verb}.` |
| `goal` | `{noun_goal}{goal}` | `{noun_subj}{nom} {noun_goal}{goal} {verb}.` |
| `com` | `{noun_com}{com}` | `{noun_subj}{nom} {noun_com}{com} {verb}.` |
| `term` | `{noun_term}{term}` | `{noun_subj}{nom} {noun_term}{term} {verb}.` |
| `init` | `{noun_init}{init}` | `{noun_subj}{nom} {noun_init}{init} {verb}.` |

To generate only a subset of adjunct types:
```bash
uv run python generate_templates.py --include adjuncts --adjuncts dat loc
```

**Progressive frames (4)**: intransitive/transitive × PRS/PST auxiliary:
```
{noun_subj}{nom} {verb_stem} {aux}.
```

**Complement frames (3)**: that-clause complements using ADN verb forms:
```
{noun_subj}{nom} {comp_verb_adn} {noun_comp}{acc} {verb}.
```

**Spatial frames (6)**: relational noun postpositions (위, 아래, …):
```
{noun_subj}{nom} {noun_pobj} {spatial_noun}{loc} {verb}.
```

**Complex postposition frames (4)**: multi-morpheme postpositions (DAT-governed and ACC-governed):
```
{noun_subj}{nom} {noun_pobj}{dat} {complex_postp_dat} {verb}.
{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{pobj_acc} {complex_postp_acc} {verb}.
```

---

### Stage 3 — Fill templates (`make fill-templates-full`)

Fills each template slot exhaustively from the lexicons. Outputs one JSONL file per template group under `filled_templates/`.

- `make fill-templates` limits to 100 verbs (fast, for testing)
- `make fill-templates-full` uses all ~10k verb forms (slow)

---

### Stage 4 — 2AFC pairs (`make 2afc-pairs-full`)

Pairs filled templates into two-alternative forced-choice items. Each pair contrasts two sentences that differ in argument structure (e.g. transitive vs. intransitive frame for the same verb). Pairs are scored with `EleutherAI/polyglot-ko-1.3b` to rank by LM plausibility.

- `make 2afc-pairs` limits to 50 filled templates (fast; still generates ~1,000+ pairs from those)
- `make 2afc-pairs-full` generates all pairs (slow; downloads the language model on first run)

Output: `items/2afc_pairs.jsonl`

---

### Stages 5–6 — Lists and Deployment

```bash
make lists             # Partition pairs into 2 lists (quick test; config default is 16)
make deployment        # Build jsPsych experiment for 2 lists (local test)
make deployment-full   # Build jsPsych experiment for all 16 lists
```

---

## Utility targets

```bash
make show-stats    # Count entries at each pipeline stage
make show-config   # Print key config values
make clean         # Remove generated items and model cache
make clean-all     # Remove all generated files
```
