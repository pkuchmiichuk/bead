# Ukrainian Argument Structure

## Quick Start

```bash
make data-quick    # Smoke test over 20 verbs
make data          # The experiment (100 verbs)
make deployment    # Build the jsPsych experiments and package them for JATOS
make help          # Show all available targets
```


## Pipeline Stages

| Step | Script | Make target |
|------|--------|-------------|
| 1. Lexicons | `generate_lexicons.py` | `make lexicons` |
| 2. Templates | `generate_templates.py` | `make templates` |
| 3. Fill templates | `fill_templates.py` | `make fill-templates` / `make fill-templates-quick` |
| 4. 2AFC pairs | `create_2afc_pairs.py` | `make 2afc-pairs` |
| 5. Lists | `generate_lists.py` | `make lists` |
| 6. Deployment | `generate_deployment.py` | `make deployment` / `make deployment-quick` |

---

### Stage 1: Lexicons (`make lexicons`)

Writes two files under `lexicons/`.

`verbs.jsonl` holds present-tense verb forms. `bleached_nouns.jsonl` holds one
semantically light noun per class, in each of the cases the frames need:

| Lemma | Class | Role | Gloss |
|-------|-------|------|-------|
| людина | animate | subject, object | person |
| гурт | group | object | band, group |
| інструмент | inanimate_object | object | instrument |
| зразок | abstract | object | sample |
| двір | location | object | yard |
| день | temporal | object | day |
| шматок | quantity | object | piece |
| випадок | event | object | incident |


Ukrainian case syncretism means a single form often realizes several cells. When
a case has more than one candidate form, the generator prefers a form that
realizes only its own cell. Two collisions survive that filter and are recorded
on the item as `case_collides_with`, so downstream stages can see them: `людини`
also realizes `ACC.PL`, and `випадку` also realizes `DAT.SG`.

---

### Stage 2: Templates (`make templates`)

Writes `templates/generic_frames.jsonl` with five frames:

| Frame | Slots |
|-------|-------|
| `intransitive` | `subj_nom`, `verb` |
| `obj_acc` | `subj_nom`, `obj_acc`, `verb` |
| `obj_gen` | `subj_nom`, `obj_gen`, `verb` |
| `obj_dat` | `subj_nom`, `obj_dat`, `verb` |
| `obj_ins` | `subj_nom`, `obj_ins`, `verb` |


The verb slot carries a stoplist read from `template.verb_stoplist` in
`config.yaml`.

---

### Stage 3: Fill templates (`make fill-templates`)

Fills every slot exhaustively and writes `items/filled.jsonl`. Filling is
dispatched through bead's `MixedFillingStrategy`, so an individual slot can be
switched to masked-LM selection in `config.yaml` without touching the script.
With the current bleached noun set every slot is exhaustive, and no model is
loaded.

`--limit N` caps the number of distinct verb lemmas. Pass `--by-frequency`
alongside it, as the Makefile does, to keep the *most frequent* lemmas. VESUM is
ordered alphabetically, so slicing off the head would otherwise yield rare verbs
rather than everyday ones.

---

### Stage 4: 2AFC pairs (`make 2afc-pairs`)

Scores every object-bearing sentence with a language model, then builds two
kinds of pair and writes them to `items/2afc_pairs.jsonl`.

**Case contrasts** group sentences that share a verb and an object noun, and
pair them across cases. Because the noun and verb are held fixed, the two
sentences differ only in the object's case form. Pairs whose two sentences
render identically are dropped, which is how the remaining syncretism from stage
1 is handled.

**Anchor contrasts** pair a test verb against a verb whose government is already
known, in the same frame and with the same noun:

| Case | Anchor verb | Gloss |
|------|-------------|-------|
| ACC | вивчати | study |
| GEN | стосуватися | concern |
| DAT | радіти | rejoice |
| INS | цікавитися | to be interested in |

Anchors put every verb on a scale shared across verbs, and they expose verbs
that fit no frame at all, which a purely within-verb contrast cannot detect.
The four were chosen to combine sensibly with all of the bleached nouns.

Both pair types record the score difference between their two sentences, and are
then assigned difficulty quantiles stratified by contrast type, so that easy and
hard pairs are distinguishable when lists are built.

`config.yaml` offers a masked and a causal model; masked models are scored
by pseudo-log-likelihood and causal ones by sentence log probability.

---

### Stages 5-6: Lists and Deployment

`make lists` partitions pairs into lists;
`make deployment-quick` builds two lists for local testing;
`make deployment-local` skips JATOS packaging entirely.

---

## Configuration

Everything lives in `config.yaml`.

`resources/verb_frequencies.csv` is a committed resource, derived from
`wordfreq`, that fixes the verb ranking across runs. Regenerate it with
`make frequencies`, which supplies `wordfreq` for that one call so the pipeline
itself does not depend on it. Frequencies are taken from the infinitive rather
than summed over the paradigm, because summing lets a paradigm absorb unrelated
homographs (колоти would absorb коли, and маяти would absorb має).

## Utility Targets

```bash
make show-stats    # Counts and sizes for the generated files
make show-config   # Key configuration values
make lint          # Ruff over the pipeline scripts
make clean         # Remove generated items, lists, and deployment
make clean-cache   # Discard the cached model scores
make clean-all     # Remove every generated file and the cached scores
```

`clean-cache` and `clean-all` throw away the cached sentence scores, which means
the next `make 2afc-pairs` rescores every sentence from scratch.

