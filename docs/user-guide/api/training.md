# Training Module

The `bead.active_learning` and `bead.evaluation` modules provide active learning loops with convergence detection.

## Active Learning Loop

Orchestrates iterative training with human annotation:

```python
from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.selection import UncertaintySampler
from bead.config.active_learning import (
    ActiveLearningLoopConfig,
    UncertaintySamplerConfig,
)

# Create selector configuration
selector_config = UncertaintySamplerConfig(
    method="entropy",
    batch_size=25,
)

# Create selector
selector = UncertaintySampler(config=selector_config)

# Create loop configuration
loop_config = ActiveLearningLoopConfig(
    max_iterations=10,
    budget_per_iteration=25,
)

# Create active learning loop
loop = ActiveLearningLoop(
    item_selector=selector,
    config=loop_config,
)
```

## Uncertainty Sampling

The `UncertaintySampler` selects items with highest uncertainty:

```python
from pathlib import Path

import numpy as np
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.active_learning.selection import UncertaintySampler
from bead.config.active_learning import UncertaintySamplerConfig
from bead.data.serialization import read_jsonlines
from bead.items.item import Item

# Load items for pool
remaining_items = read_jsonlines(Path("items/2afc_pairs.jsonl"), Item)

# Create trained model (placeholder)
trained_model = None  # In practice, this would be a trained model

# Create sampler
config = UncertaintySamplerConfig(method="entropy")
sampler = UncertaintySampler(config=config)


# Selection function
def predict_fn(model: ForcedChoiceModel, item: Item) -> np.ndarray:
    """Return probability distribution over classes."""
    # For 2AFC: array of shape (2,) with probabilities
    return np.array([0.6, 0.4])


# Select items
selected = sampler.select(
    items=remaining_items,
    model=trained_model,
    predict_fn=predict_fn,
    budget=25,
)
```

**Uncertainty methods**:

- `"entropy"`: maximum entropy (most uncertain)
- `"margin"`: minimum margin between top-2 classes
- `"least_confidence"`: minimum confidence in top class

## Convergence Detection

The `ConvergenceDetector` monitors agreement between model and humans:

```python
from bead.evaluation.convergence import ConvergenceDetector

detector = ConvergenceDetector(
    human_agreement_metric="krippendorff_alpha",
    convergence_threshold=0.05,
    min_iterations=3,
    statistical_test=True,
    alpha=0.05,
)

# Compute human baseline first
human_ratings = {
    "rater1": [0, 1, 0],  # Rater 1's ratings for 3 items
    "rater2": [0, 1, 0],  # Rater 2's ratings
    "rater3": [1, 1, 0],  # Rater 3's ratings
}
baseline = detector.compute_human_baseline(human_ratings)

# Check convergence after each iteration
model_accuracy = 0.85  # Model's accuracy on validation set
is_converged = detector.check_convergence(
    model_accuracy=model_accuracy,
    iteration=5,
    human_agreement=baseline,
)

if is_converged:
    print("Model has converged to human agreement!")
```

**Supported metrics**:

- `"krippendorff_alpha"`: Krippendorff's alpha (handles missing data)
- `"fleiss_kappa"`: Fleiss' kappa (multiple raters)
- `"cohens_kappa"`: Cohen's kappa (two raters)
- `"percentage_agreement"`: Simple agreement rate

## Model Training

Train models on human judgments:

```python
from pathlib import Path

from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.config.active_learning import ForcedChoiceModelConfig
from bead.data.serialization import read_jsonlines
from bead.items.item import Item

# Load training data
training_items = read_jsonlines(Path("items/2afc_pairs.jsonl"), Item)[:5]

# Create model config
config = ForcedChoiceModelConfig(
    model_name="bert-base-uncased",
    num_epochs=10,
    batch_size=16,
    learning_rate=2e-5,
)

# Create model
model = ForcedChoiceModel(config=config)

# Prepare training data
labels = [0, 1, 0, 1, 0]  # Human judgments (0 or 1 for 2AFC)

# Train model (fixed-effects mode does not use participant_ids)
model.train(items=training_items, labels=labels)

# After training, predict on new items
print(f"Model trained on {len(training_items)} items")
print("Model ready for predictions")
```

## Mixed Effects Models

For modeling participant and item variability:

```python
from pathlib import Path

from bead.active_learning.config import (
    MixedEffectsConfig,
    RandomEffectsSpec,
    VarianceComponents,
)
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.config.active_learning import ForcedChoiceModelConfig
from bead.data.serialization import read_jsonlines
from bead.items.item import Item

# Load data
training_items = read_jsonlines(Path("items/2afc_pairs.jsonl"), Item)[:5]
labels = [0, 1, 0, 1, 0]
participant_ids = ["p1", "p1", "p2", "p2", "p1"]

# Configure mixed effects
config = ForcedChoiceModelConfig(
    model_name="bert-base-uncased",
    mixed_effects=MixedEffectsConfig(
        mode="random_intercepts",  # "fixed", "random_intercepts", "random_slopes"
        prior_mean=0.0,
        prior_variance=1.0,
        regularization_strength=0.01,
    ),
)

model = ForcedChoiceModel(config=config)

# Train with participant IDs (required for mixed effects)
model.train(
    items=training_items,
    labels=labels,
    participant_ids=participant_ids,
)

print("Mixed effects model trained with random participant intercepts")
```

## Agreement Metrics

Compute inter-annotator agreement:

```python
from bead.evaluation.interannotator import InterAnnotatorMetrics
import numpy as np

# Krippendorff's alpha (handles missing data)
reliability_data = {
    "rater1": [0, 1, 0],
    "rater2": [0, 1, 0],
    "rater3": [1, None, 0],  # Missing value for item 1
}

alpha = InterAnnotatorMetrics.krippendorff_alpha(reliability_data)
print(f"Krippendorff's alpha: {alpha:.3f}")

# Fleiss' kappa (requires matrix of counts)
# Matrix shape: (n_items, n_categories)
# Element [i, j] = number of raters who assigned item i to category j
ratings_matrix = np.array(
    [
        [2, 1],  # Item 0: 2 raters chose 0, 1 chose 1
        [0, 3],  # Item 1: 0 raters chose 0, 3 chose 1
        [3, 0],  # Item 2: 3 raters chose 0, 0 chose 1
    ]
)

kappa = InterAnnotatorMetrics.fleiss_kappa(ratings_matrix)
print(f"Fleiss' kappa: {kappa:.3f}")
```

## Complete Example

Configuration-based active learning workflow:

```python
from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.selection import UncertaintySampler
from bead.config.active_learning import (
    ActiveLearningLoopConfig,
    UncertaintySamplerConfig,
)
from bead.evaluation.convergence import ConvergenceDetector

# Create config dict (normally loaded from YAML)
config = {
    "training": {
        "convergence": {
            "metric": "krippendorff_alpha",
            "threshold": 0.05,
            "min_iterations": 3,
            "alpha": 0.05,
        }
    },
    "active_learning": {
        "method": "entropy",
        "batch_size": 25,
        "max_iterations": 10,
    },
}

# Create convergence detector
conv_config = config["training"]["convergence"]
detector = ConvergenceDetector(
    human_agreement_metric=conv_config["metric"],
    convergence_threshold=conv_config["threshold"],
    min_iterations=conv_config["min_iterations"],
    alpha=conv_config.get("alpha", 0.05),
)

print("Convergence detector initialized:")
print(f"  Metric: {conv_config['metric']}")
print(f"  Threshold: {conv_config['threshold']}")

# Create selector
al_config = config["active_learning"]
selector_config = UncertaintySamplerConfig(
    method=al_config["method"],
    batch_size=al_config["batch_size"],
)
selector = UncertaintySampler(config=selector_config)

print(f"Active learning strategy: {al_config['method']}")

# Create loop config
loop_config = ActiveLearningLoopConfig(
    max_iterations=al_config["max_iterations"],
    budget_per_iteration=al_config["batch_size"],
)

# Create active learning loop
loop = ActiveLearningLoop(
    item_selector=selector,
    config=loop_config,
)

print("Active learning loop initialized")
```

## Design Principles

1. **Configuration-Driven**: All parameters in config objects
2. **Participant Tracking**: All models require `participant_ids` parameter
3. **Mixed Effects Support**: Random intercepts and interaction terms
4. **Convergence Detection**: Statistical comparison to human agreement

## Configuration Summary

**ActiveLearningLoopConfig**:

| Parameter | Purpose |
|-----------|---------|
| `max_iterations` | Maximum AL iterations |
| `budget_per_iteration` | Items selected per iteration |

**UncertaintySamplerConfig**:

| Parameter | Purpose |
|-----------|---------|
| `method` | Uncertainty method (entropy, margin, least_confidence) |
| `batch_size` | Items to select |

**ConvergenceDetector**:

| Parameter | Purpose |
|-----------|---------|
| `human_agreement_metric` | Baseline metric (krippendorff_alpha, fleiss_kappa, etc.) |
| `convergence_threshold` | Threshold for convergence |
| `min_iterations` | Minimum iterations before checking |

## Next Steps

- [Workflows](workflows.md): Complete end-to-end pipeline
- [CLI reference](../cli/training.md): Command-line equivalents
- [Gallery example](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/run_pipeline.py): Full working script
