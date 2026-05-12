# Configuration System

The configuration system orchestrates multi-stage pipelines through YAML files. Configuration files eliminate the need for custom scripts by specifying resources, strategies, constraints, and training parameters in a single document.

## Configuration Structure

Configuration files map to the eight main sections of `BeadConfig`:

```yaml
profile: "default"

paths:
  data_dir: "data"
  output_dir: "output"
  cache_dir: ".cache"

resources:
  lexicons:
    - path: "lexicons/verbs.jsonl"
      name: "verbs"
  templates:
    - path: "templates/frames.jsonl"
      name: "frames"

templates:
  filling_strategy: "exhaustive"
  output_path: "filled_templates/filled.jsonl"

items:
  judgment_type: "forced_choice"
  n_alternatives: 2

lists:
  strategy: "balanced"
  n_lists: 10
  items_per_list: 50

deployment:
  platform: "jatos"
  distribution_strategy:
    strategy_type: "balanced"
  experiment:
    title: "Experiment Title"

active_learning:
  strategy: "uncertainty_sampling"
  budget_per_iteration: 200
  max_iterations: 20

logging:
  level: "INFO"
  file:
    enabled: true
    path: "pipeline.log"
```

## Loading Configurations

Load YAML configurations using the Python API:

```python
from bead.config import load_config

# Load from file
config = load_config("config.yaml")

# Access nested fields
print(config.paths.data_dir)  # PosixPath('data')
print(config.items.judgment_type)  # 'forced_choice'
print(config.active_learning.strategy)  # 'uncertainty_sampling'
```

From environment variables:

```python
from bead.config import load_from_env

# Override fields from environment
# BEAD_PATHS__DATA_DIR=/custom/path
# BEAD_LOGGING__LEVEL=DEBUG
config = load_from_env(config)
```

Compose multiple configurations and apply CLI-style overrides:

```python
from bead.config import load_config

# extra files overlay after the primary YAML
# overrides are dotted-key=value strings (YAML-parsed for typing)
config = load_config(
    "config.yaml",
    extra=["overlays/local.yaml"],
    overrides=["paths.data_dir=/tmp/data", "logging.level=DEBUG"],
)
```

Configs may also reference each other through a top-level
`defaults:` list (paths resolve next to the primary YAML; bare
names resolve to `.yaml` or `.toml`):

```yaml
defaults:
  - protocol/argument_structure   # protocol/argument_structure.yaml
  - logging/verbose
paths:
  data_dir: "${oc.env:BEAD_DATA,/tmp/data}"
  out_dir: "${paths.data_dir}/out"
```

Interpolation follows the OmegaConf grammar:
`${section.field}` absolute references, `${.x}` / `${..y}`
relative references, `${a.b[0]}` and `${a.b.0}` list indexing,
`${a.${b}}` nested expressions, `\${literal}` escape, and the
built-in resolvers (`oc.env`, `oc.select`, `oc.decode`,
`oc.deprecated`, `oc.create`, `oc.dict.keys`, `oc.dict.values`).
Register custom resolvers with
`bead.config.compose.register_resolver(name, fn)`.

TOML configs (`.toml`) load the same way as YAML.

## Configuration Profiles

Use predefined profiles for different environments:

```python
from bead.config import get_profile, list_profiles

# Development profile (verbose logging, small datasets)
dev_config = get_profile("dev")

# Production profile (optimized settings)
prod_config = get_profile("prod")

# Test profile (minimal settings for fast tests)
test_config = get_profile("test")

# List available profiles
profiles = list_profiles()  # ['default', 'dev', 'prod', 'test']
```

Profile configurations are in `bead/config/profiles.py`:

- **dev**: Verbose logging (DEBUG), small batch sizes, frequent checkpoints
- **prod**: Optimized for performance, INFO logging, larger batches
- **test**: Minimal settings for CI/CD, WARNING logging

## Configuration Sections

### PathsConfig

```yaml
paths:
  data_dir: "data"
  output_dir: "output"
  cache_dir: ".cache"
  temp_dir: null  # Optional temporary directory
```

All paths support relative or absolute paths. Relative paths resolve from the current working directory.

### ResourceConfig

```yaml
resources:
  lexicons:
    - path: "lexicons/verbs.jsonl"
      name: "verbs"
      description: "Verb lexicon from VerbNet"
    - path: "lexicons/nouns.jsonl"
      name: "nouns"

  templates:
    - path: "templates/transitive.jsonl"
      name: "transitive"
    - path: "templates/ditransitive.jsonl"
      name: "ditransitive"

  constraints:
    - path: "constraints/animacy.jsonl"
      description: "Animacy constraints for subject"
```

### TemplateConfig

```yaml
templates:
  filling_strategy: "mixed"  # exhaustive, random, stratified, mlm, mixed
  output_path: "filled_templates/filled.jsonl"

  # MLM settings (for mlm or mixed strategy)
  mlm:
    model_name: "bert-base-uncased"
    beam_size: 5
    top_k: 10
    device: "cpu"
    cache_enabled: true

  # Slot-specific strategies (for mixed strategy)
  slot_strategies:
    verb:
      strategy: "exhaustive"
      description: "All verbs for cross-product design"

    noun_subj:
      strategy: "mlm"
      description: "MLM selection of subject nouns"
      max_fills: 5
      enforce_unique: true
```

### ItemConfig

```yaml
items:
  judgment_type: "forced_choice"  # forced_choice, acceptability_rating, inference
  n_alternatives: 2  # For forced choice

  models:
    - name: "gpt2"
      provider: "huggingface"
      device: "cpu"
      use_for_scoring: true

  construction:
    create_minimal_pairs: true
    pair_types:
      - "same_verb"
      - "different_verb"

    score_filtering:
      enabled: true
      min_score_diff: 0.5

  preserve_metadata:
    - "verb_lemma"
    - "template_id"
    - "pair_type"
```

### ListConfig

```yaml
lists:
  strategy: "quantile_balanced_minimal_pairs"  # balanced, random, stratified, custom
  n_lists: 16
  items_per_list: 50

  # List constraints (apply to each list individually)
  constraints:
    - type: "balance"
      property_expression: "item.metadata.condition"
      target_counts:
        control: 25
        experimental: 25
      description: "Balanced conditions per list"

    - type: "uniqueness"
      property_expression: "item.metadata.verb"
      description: "No verb appears twice in same list"

    - type: "grouped_quantile"
      property_expression: "item.metadata.score_diff"
      group_by_expression: "item.metadata.pair_type"
      n_quantiles: 10
      items_per_quantile: 5
      description: "Stratified difficulty within pair types"

  # Batch constraints (apply across all lists)
  batch_constraints:
    - type: "coverage"
      property_expression: "item['template_id']"
      target_values: [0, 1, 2, 3, 4, 5]
      min_coverage: 1.0
      description: "All templates appear somewhere"

    - type: "balance"
      property_expression: "item['pair_type']"
      target_distribution:
        same_verb: 0.5
        different_verb: 0.5
      tolerance: 0.05
      description: "50/50 pair type split across batch"
```

### DeploymentConfig

```yaml
deployment:
  platform: "jatos"  # jatos, prolific, mturk

  # Distribution strategy for assigning participants to lists
  distribution_strategy:
    strategy_type: "balanced"  # random, sequential, balanced, latin_square, etc.
    max_participants: 400  # Optional cap
    error_on_exhaustion: false
    debug_mode: false  # Force single list for testing
    debug_list_index: 0

  n_lists_to_deploy: 20
  random_seed: 42
  output_dir: "deployment"

  experiment:
    title: "Sentence Acceptability Judgments"
    description: "Rate which sentence sounds more natural"
    estimated_duration_minutes: 15

  jspsych:
    version: "8.0.0"
    use_jatos: true
    prolific_completion_code: null  # Set for Prolific integration

    trial:
      type: "html-button-response"
      stimulus_duration: null
      trial_duration: null

    choices:
      - "Sentence A"
      - "Sentence B"

    randomize_order: true
    randomize_choices: true

  # Behavioral capture with slopit (optional)
  slopit:
    enabled: false  # Set to true to enable behavioral capture
    keystroke:
      enabled: true
    focus:
      enabled: true
    paste:
      enabled: true
      prevent: false  # Set true to block paste events
    target_selectors:
      likert_rating: ".bead-rating-button"
      forced_choice: ".bead-choice-button"
      cloze: ".bead-cloze-field"

  participants:
    n_per_list: 30
    qualifications:
      - "Native English speaker"
      - "Age 18+"
    payment_usd: 2.50
```

### ActiveLearningConfig

```yaml
active_learning:
  strategy: "uncertainty_sampling"  # uncertainty_sampling, query_by_committee, random
  method: "entropy"  # entropy, least_confidence, margin

  budget_per_iteration: 200
  max_iterations: 20
  stopping_criterion: "convergence"  # max_iterations, performance_threshold, convergence

  initial_training_size: 500
  batch_mode: true
  promote_diversity: true
  diversity_lambda: 0.1

  # Model configuration
  model:
    architecture: "transformer"
    model_name: "bert-base-uncased"
    learning_rate: 2e-5
    batch_size: 16
    epochs_per_iteration: 3
    warmup_steps: 100
    device: "cpu"

  # Convergence detection
  convergence:
    metric: "krippendorff_alpha"  # krippendorff_alpha, fleiss_kappa, cohens_kappa
    threshold: 0.05  # Stop when model within 0.05 of human agreement
    min_iterations: 3
    alpha: 0.05  # Statistical significance level
```

### LoggingConfig

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  file:
    enabled: true
    path: "pipeline.log"
    max_bytes: 10485760  # 10 MB
    backup_count: 5

  console:
    enabled: true
    colored: true
```

## Validation

Validate configuration files before running pipelines:

```python
from bead.config import validate_config, load_config

config = load_config("config.yaml")

# Validate configuration
errors = validate_config(config, strict=True)

if errors:
    for error in errors:
        print(f"ERROR: {error}")
else:
    print("Configuration valid")
```

Validate paths exist:

```python
# Check all path fields
path_errors = config.validate_paths()

if path_errors:
    for error in path_errors:
        print(f"PATH ERROR: {error}")
```

Validation checks:

- Required fields present
- File paths exist (when absolute)
- Strategy names valid
- Numeric fields in valid ranges
- Distribution strategy configuration complete
- Task type matches item configuration
- Constraint expressions parse correctly

## Serialization

Export configurations to YAML:

```python
from bead.config import save_yaml, to_yaml

# Convert to YAML string
yaml_str = to_yaml(config, include_defaults=False)
print(yaml_str)

# Save to file
save_yaml(config, "exported_config.yaml", include_defaults=False)
```

Exclude default values with `include_defaults=False` for cleaner output.

Convert to dictionary:

```python
# Pydantic model_dump()
config_dict = config.to_dict()

# Access nested values
print(config_dict["paths"]["data_dir"])
```

## Environment Variables

Override configuration fields using environment variables with double-underscore notation:

```bash
# Override paths.data_dir
export BEAD_PATHS__DATA_DIR=/custom/data

# Override logging.level
export BEAD_LOGGING__LEVEL=DEBUG

# Override active_learning.budget_per_iteration
export BEAD_ACTIVE_LEARNING__BUDGET_PER_ITERATION=500
```

Load environment overrides:

```python
from bead.config import load_config, load_from_env

# Load base config
config = load_config("config.yaml")

# Apply environment overrides
config = load_from_env(config)
```

Environment variables take precedence over YAML values.

## Complete Example

Example configuration for a forced-choice experiment with active learning:

```yaml
# ============================================================================
# Forced-Choice Experiment with Active Learning
# ============================================================================

profile: "prod"

# Paths
# ============================================================================
paths:
  data_dir: "data"
  output_dir: "output"
  cache_dir: ".cache/bead"
  temp_dir: null

# Resources
# ============================================================================
resources:
  lexicons:
    - path: "lexicons/verbs.jsonl"
      name: "verbs"
      description: "VerbNet verbs with frame information"
    - path: "lexicons/nouns.jsonl"
      name: "nouns"
      description: "Semantically light nouns"

  templates:
    - path: "templates/frames.jsonl"
      name: "frames"
      description: "Generic frame structures"

  constraints:
    - path: "constraints/animacy.jsonl"
      description: "Animacy constraints for subjects"

# Template Filling
# ============================================================================
templates:
  filling_strategy: "exhaustive"
  output_path: "filled_templates/frames_filled.jsonl"

# Items
# ============================================================================
items:
  judgment_type: "forced_choice"
  n_alternatives: 2

  construction:
    create_minimal_pairs: true
    pair_types:
      - "same_verb"
      - "different_verb"

  preserve_metadata:
    - "verb_lemma"
    - "template_id"
    - "pair_type"

# Lists
# ============================================================================
lists:
  strategy: "balanced"
  n_lists: 10
  items_per_list: 50

  constraints:
    - type: "uniqueness"
      property_expression: "item.metadata.verb"
      description: "Unique verbs per list"

    - type: "balance"
      property_expression: "item.metadata.pair_type"
      target_counts:
        same_verb: 25
        different_verb: 25
      description: "Balanced pair types"

  batch_constraints:
    - type: "coverage"
      property_expression: "item['template_id']"
      target_values: [0, 1, 2, 3, 4, 5]
      min_coverage: 1.0
      description: "All templates represented"

# Deployment
# ============================================================================
deployment:
  platform: "jatos"

  distribution_strategy:
    strategy_type: "balanced"

  experiment:
    title: "Sentence Acceptability Study"
    description: "Choose the more natural sentence"
    estimated_duration_minutes: 15

  jspsych:
    version: "8.0.0"
    use_jatos: true

    choices:
      - "Sentence A"
      - "Sentence B"

    randomize_order: true
    randomize_choices: true

  participants:
    n_per_list: 30

# Active Learning
# ============================================================================
active_learning:
  strategy: "uncertainty_sampling"
  method: "entropy"

  budget_per_iteration: 100
  max_iterations: 15
  stopping_criterion: "convergence"

  initial_training_size: 200

  model:
    architecture: "transformer"
    model_name: "bert-base-uncased"
    learning_rate: 2e-5
    batch_size: 16
    epochs_per_iteration: 3
    device: "cpu"

  convergence:
    metric: "krippendorff_alpha"
    threshold: 0.05
    min_iterations: 3

# Logging
# ============================================================================
logging:
  level: "INFO"

  file:
    enabled: true
    path: "experiment.log"
    max_bytes: 10485760
    backup_count: 3

  console:
    enabled: true
    colored: true
```

## Using Configurations

Load and use configuration in Python scripts:

```python
from bead.config import load_config
from bead.resources import Lexicon, TemplateCollection
from bead.templates import TemplateFiller
from bead.items.forced_choice import create_forced_choice_items_from_groups
from bead.lists import ListPartitioner
from bead.deployment.jspsych import JsPsychExperimentGenerator

# Load configuration
config = load_config("config.yaml")

# Stage 1: Load resources
lexicon = Lexicon.from_jsonl(config.resources.lexicons[0].path)
templates = TemplateCollection.from_jsonl(config.resources.templates[0].path)

# Stage 2: Fill templates
filler = TemplateFiller(templates, lexicon)
filled = filler.fill(strategy=config.templates.filling_strategy)

# Stage 3: Create items
items = create_forced_choice_items_from_groups(
    filled, group_by=lambda t: t.template_id, n_alternatives=config.items.n_alternatives
)

# Stage 4: Partition lists
partitioner = ListPartitioner()
lists = partitioner.partition(
    items, n_lists=config.lists.n_lists, strategy=config.lists.strategy
)

# Stage 5: Generate experiment
from bead.deployment.jspsych.config import ExperimentConfig
from bead.deployment.distribution import ListDistributionStrategy

exp_config = ExperimentConfig(
    experiment_type="forced_choice",
    title=config.deployment.experiment.title,
    description=config.deployment.experiment.description,
    distribution_strategy=ListDistributionStrategy(
        strategy_type=config.deployment.distribution_strategy.strategy_type
    ),
)

generator = JsPsychExperimentGenerator(exp_config, output_dir="experiment/")
generator.generate(lists, items_dict, templates_dict)
```

## CLI Commands

Manage configurations via the command line:

### Validate Configuration

```bash
uv run bead config validate --config config.yaml --strict
```

Validates configuration structure, required fields, and path existence.

### Merge Configurations

```bash
uv run bead config merge --base base.yaml --override overrides.yaml --output merged.yaml
```

Merges multiple configuration files (later configs override earlier ones).

### Create Model Configuration

```bash
uv run bead config create-model --task-type forced_choice --base-model bert-base-uncased
```

Generates model configuration section with task-specific defaults.

### Create Active Learning Configuration

```bash
uv run bead config create-active-learning --selection-strategy uncertainty --budget 1000
```

Generates active learning configuration section.

### Future Commands

The following command is planned but not yet implemented:

```bash
# Planned (not yet available)
uv run bead config create-deployment --distribution-strategy balanced
```

## Next Steps

After creating configuration files:

1. [Load resources](cli/resources.md) specified in configuration
2. [Fill templates](cli/templates.md) using configured strategies
3. [Create items](cli/items.md) with configured task types
4. [Partition lists](cli/lists.md) using configured constraints
5. [Deploy experiments](cli/deployment.md) with configured distribution strategies
6. [Train models](cli/training.md) using active learning configuration

For complete API documentation, see [bead.config API reference](../api/config.md).
