# Korean Argument Structure

## Quick Start

```bash
make data          # Quick test run (100 verbs, 50 pairs)
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

### Stage notes

- **Lexicons**: extracts verbs from UniMorph; generates bleached nouns, verbs, adjectives, case markers, spatial nouns, complex postpositions, and complement verbs
- **Templates**: generates 47 generic frames (base=2, adjuncts=28, progressive=4, complement=3, spatial=6, complex=4); use `make templates` for base frames only (quick test)
- **Fill templates**: `make fill-templates` limits to 100 verbs; `make fill-templates-full` uses all ~10k verb forms
- **2AFC pairs**: scored with `EleutherAI/polyglot-ko-1.3b`; full run is slow

## Utility targets

```bash
make show-stats    # Count entries at each pipeline stage
make show-config   # Print key config values
make clean         # Remove generated items and model cache
make clean-all     # Remove all generated files
```
