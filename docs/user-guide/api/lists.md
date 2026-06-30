# Lists Module

The `bead.lists` module provides list partitioning with constraint satisfaction for balanced experimental designs.

## Basic List Partitioning

Partition items into balanced lists:

```python
from uuid import uuid4

from bead.lists.partitioner import ListPartitioner

# Create partitioner
partitioner = ListPartitioner(random_seed=42)

# Partition items (use real UUIDs)
item_uuids = [uuid4() for _ in range(100)]
lists = partitioner.partition(
    items=item_uuids,
    n_lists=10,
    strategy="balanced",
)

print(f"Created {len(lists)} lists")
```

**Partitioning strategies**:

- `"balanced"`: Equal list sizes
- `"random"`: Random assignment
- `"stratified"`: Balance across strata

## List Constraints

Constraints apply to each list individually. Available constraint types:

**UniquenessConstraint**: no duplicate values

```python
from bead.lists import UniquenessConstraint

constraint = UniquenessConstraint(
    constraint_type="uniqueness",
    property_expression="item.metadata.verb_lemma",
)
```

**BalanceConstraint**: balanced distribution

```python
from bead.lists import BalanceConstraint

constraint = BalanceConstraint(
    constraint_type="balance",
    property_expression="item.metadata.transitivity",
    target_counts={"transitive": 5, "intransitive": 5},
)
```

**DiversityConstraint**: minimum unique values

```python
from bead.lists import DiversityConstraint

constraint = DiversityConstraint(
    constraint_type="diversity",
    property_expression="item.metadata.template_id",
    min_unique_values=5,
)
```

**Using constraints**:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.lists import DiversityConstraint, UniquenessConstraint
from bead.lists.partitioner import ListPartitioner

# Load items from fixtures
items = read_jsonlines(Path("items/cross_product_items.jsonl"), Item)

partitioner = ListPartitioner(random_seed=42)

# Build metadata dict
metadata = {item.id: {"metadata": dict(item.item_metadata)} for item in items}

# Partition with constraints
lists = partitioner.partition(
    items=[item.id for item in items],
    n_lists=10,
    constraints=[
        UniquenessConstraint(
            constraint_type="uniqueness", property_expression="item.metadata.verb_lemma"
        ),
        DiversityConstraint(
            constraint_type="diversity",
            property_expression="item.metadata.template_id",
            min_unique_values=5,
        ),
    ],
    metadata=metadata,
)
```

## Batch Constraints

Batch constraints apply across all lists. Available types:

**BatchCoverageConstraint**: ensure values appear somewhere

```python
from bead.lists.constraints import BatchCoverageConstraint

constraint = BatchCoverageConstraint(
    constraint_type="coverage",
    property_expression="item.metadata.template_id",
    target_values=list(range(26)),  # All 26 templates
    min_coverage=1.0,  # 100% coverage
)
```

**BatchBalanceConstraint**: balanced distribution across lists

```python
from bead.lists.constraints import BatchBalanceConstraint

constraint = BatchBalanceConstraint(
    constraint_type="balance",
    property_expression="item.metadata.condition",
    target_distribution={"A": 0.5, "B": 0.5},
    tolerance=0.1,
)
```

**BatchDiversityConstraint**: prevent values from appearing in too many lists

```python
from bead.lists.constraints import BatchDiversityConstraint

constraint = BatchDiversityConstraint(
    constraint_type="diversity",
    property_expression="item.metadata.verb_lemma",
    max_lists_per_value=3,  # No verb appears in more than 3 lists
)
```

**BatchMinOccurrenceConstraint**: minimum occurrences per value

```python
from bead.lists.constraints import BatchMinOccurrenceConstraint

constraint = BatchMinOccurrenceConstraint(
    constraint_type="min_occurrence",
    property_expression="item.metadata.template_id",
    min_occurrences=5,  # Each template appears at least 5 times
)
```

**Using batch constraints**:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.lists import UniquenessConstraint
from bead.lists.constraints import BatchCoverageConstraint
from bead.lists.partitioner import ListPartitioner

# Load items
items = read_jsonlines(Path("items/cross_product_items.jsonl"), Item)
item_uuids = [item.id for item in items]
metadata = {item.id: {"metadata": dict(item.item_metadata)} for item in items}

partitioner = ListPartitioner(random_seed=42)

lists = partitioner.partition_with_batch_constraints(
    items=item_uuids,
    n_lists=10,
    list_constraints=[
        UniquenessConstraint(
            constraint_type="uniqueness", property_expression="item.metadata.verb_lemma"
        ),
    ],
    batch_constraints=[
        BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item.metadata.template_id",
            target_values=list(range(26)),
            min_coverage=1.0,
        ),
    ],
    metadata=metadata,
)
```

## Grouped Quantile Constraint

Partition into quantiles grouped by a property:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.lists.constraints import GroupedQuantileConstraint
from bead.lists.partitioner import ListPartitioner

# Load items
items = read_jsonlines(Path("items/cross_product_items.jsonl"), Item)
item_uuids = [item.id for item in items]
metadata = {item.id: {"metadata": dict(item.item_metadata)} for item in items}

partitioner = ListPartitioner(random_seed=42)

constraint = GroupedQuantileConstraint(
    constraint_type="grouped_quantile",
    property_expression="item.metadata.lm_score",
    group_by_expression="item.metadata.verb_lemma",
    n_quantiles=4,
)

lists = partitioner.partition(
    items=item_uuids,
    n_lists=10,
    constraints=[constraint],
    metadata=metadata,
)
```

## Assigning Quantiles

Assign quantile labels to items before partitioning:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.lists.stratification import assign_quantiles

# Load items with numeric scores
items = read_jsonlines(Path("items/2afc_pairs.jsonl"), Item)
metadata = {item.id: {"metadata": dict(item.item_metadata)} for item in items}

# Assign quantiles within groups
quantile_assignments = assign_quantiles(
    items=[item.id for item in items],
    property_getter=lambda item_id: metadata[item_id]["metadata"]["lm_score_a"],
    n_quantiles=4,
    stratify_by=lambda item_id: metadata[item_id]["metadata"]["group_key"],
)

# Add quantile assignments to metadata
for item_id, quantile in quantile_assignments.items():
    metadata[item_id]["metadata"]["assigned_quantile"] = quantile

# Check assignments
for item_id in list(metadata.keys())[:3]:
    q = metadata[item_id]["metadata"]["assigned_quantile"]
    print(f"Item quantile: {q}")
```

## List Collections

Save and load list collections:

```python
from pathlib import Path
from uuid import uuid4

from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.lists import DiversityConstraint, ListCollection, UniquenessConstraint
from bead.lists.partitioner import ListPartitioner

# Load items and partition
items = read_jsonlines(Path("items/cross_product_items.jsonl"), Item)
metadata = {item.id: {"metadata": dict(item.item_metadata)} for item in items}

partitioner = ListPartitioner(random_seed=42)
lists = partitioner.partition(
    items=[item.id for item in items],
    n_lists=10,
    constraints=[
        UniquenessConstraint(
            constraint_type="uniqueness", property_expression="item.metadata.verb_lemma"
        ),
        DiversityConstraint(
            constraint_type="diversity",
            property_expression="item.metadata.template_id",
            min_unique_values=5,
        ),
    ],
    metadata=metadata,
)

# Create collection
collection = ListCollection(
    name="experiment_lists",
    source_items_id=uuid4(),
    lists=lists,
    partitioning_strategy="balanced",
    partitioning_config={
        "n_lists": len(lists),
        "n_list_constraints": 2,
        "n_batch_constraints": 0,
    },
    partitioning_stats={
        "total_items": sum(len(lst.item_refs) for lst in lists),
    },
)

print(f"Created collection with {len(collection.lists)} lists")
```

## Complete Example

From [gallery/eng/argument_structure/generate_lists.py](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/generate_lists.py):

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.lists import DiversityConstraint, UniquenessConstraint
from bead.lists.constraints import (
    BatchCoverageConstraint,
    GroupedQuantileConstraint,
)
from bead.lists.partitioner import ListPartitioner

# Load items
items = read_jsonlines(Path("items/cross_product_items.jsonl"), Item)

# Build metadata dict
metadata = {item.id: {"metadata": dict(item.item_metadata)} for item in items}

# Define constraints
list_constraints = [
    UniquenessConstraint(
        constraint_type="uniqueness", property_expression="item.metadata.verb_lemma"
    ),
    DiversityConstraint(
        constraint_type="diversity",
        property_expression="item.metadata.template_id",
        min_unique_values=5,
    ),
    GroupedQuantileConstraint(
        constraint_type="grouped_quantile",
        property_expression="item.metadata.lm_score",
        group_by_expression="item.metadata.verb_lemma",
        n_quantiles=4,
    ),
]

batch_constraints = [
    BatchCoverageConstraint(
        constraint_type="coverage",
        property_expression="item.metadata.template_id",
        target_values=list(range(26)),
        min_coverage=1.0,
    ),
]

# Partition
partitioner = ListPartitioner(random_seed=42)
lists = partitioner.partition_with_batch_constraints(
    items=[item.id for item in items],
    n_lists=10,
    list_constraints=list_constraints,
    batch_constraints=batch_constraints,
    metadata=metadata,
)

total_items = sum(len(lst.item_refs) for lst in lists)
print(f"Created {len(lists)} lists with {total_items} total items")
```

## Design Principles

1. **Stand-off Annotation**: Lists contain item UUIDs, not full items
2. **Metadata-Driven**: Constraints evaluate against metadata dict
3. **DSL Expressions**: Flexible property extraction via DSL
4. **Iterative Refinement**: Batch constraints use swap-based refinement

## Constraint Summary

**List Constraints** (per-list):

| Constraint | Purpose |
|------------|---------|
| `UniquenessConstraint` | No duplicate values |
| `BalanceConstraint` | Balanced distribution |
| `DiversityConstraint` | Minimum unique values |
| `GroupedQuantileConstraint` | Quantile-based grouping |

**Batch Constraints** (across all lists):

| Constraint | Purpose |
|------------|---------|
| `BatchCoverageConstraint` | All values appear somewhere |
| `BatchBalanceConstraint` | Balanced distribution across batch |
| `BatchDiversityConstraint` | Minimum diversity across batch |
| `BatchMinOccurrenceConstraint` | Minimum occurrences per value |

## Next Steps

- [Deployment module](deployment.md): Generate jsPsych experiments from lists
- [CLI reference](../cli/lists.md): Command-line equivalents
- [Gallery example](https://github.com/FACTSlab/bead/blob/main/gallery/eng/argument_structure/generate_lists.py): Full working script
