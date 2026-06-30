#!/usr/bin/env python3
"""Simulate the complete active learning pipeline with synthetic judgments.

This script demonstrates the bead.simulation framework on the argument structure
project. It:
1. Loads 2AFC pairs from items/2afc_pairs.jsonl
2. Simulates human judgments using the bead.simulation framework
3. Trains model on simulated data
4. Uses active learning to select next batch
5. Repeats until convergence

The simulation uses the LMBasedAnnotator with temperature noise to generate
probabilistic judgments based on language model scores.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score

from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.active_learning.selection import UncertaintySampler
from bead.cli.display import (
    console,
    create_summary_table,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.config.active_learning import (
    ActiveLearningLoopConfig,
    ForcedChoiceModelConfig,
    UncertaintySamplerConfig,
)
from bead.config.simulation import NoiseModelConfig, SimulatedAnnotatorConfig
from bead.evaluation.convergence import ConvergenceDetector
from bead.evaluation.interannotator import InterAnnotatorMetrics
from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.protocol.items import family_to_item_template
from bead.simulation.annotators.base import SimulatedAnnotator

from protocol import acceptability_family, build_protocol


def load_2afc_pairs(path: Path, limit: int | None = None, skip: int = 0) -> list[Item]:
    """Load 2AFC pairs from JSONL.

    Parameters
    ----------
    path : Path
        Path to JSONL file
    limit : int | None
        Maximum number of items to load
    skip : int
        Number of items to skip at start

    Returns
    -------
    list[Item]
        List of items
    """
    items: list[Item] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if limit and (i - skip) >= limit:
                break
            items.append(Item.model_validate_json(line))
    return items


_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def get_forced_choice_template() -> ItemTemplate:
    """Build the 2AFC ItemTemplate from the configured protocol.

    The prompt and task-type come from the
    ``protocol.families[].anchor`` declaration in ``config.yaml``
    via :func:`bead.protocol.items.family_to_item_template`. The
    canonical bridge leaves ``task_spec.options`` unset because the
    per-item alternatives (the two sentences) live on each
    :class:`Item`; the simulator however samples from response-space
    labels (``"first"`` / ``"second"``), so we splice those onto
    ``task_spec.options`` here.
    """
    family = acceptability_family(build_protocol(_CONFIG_PATH))
    template = family_to_item_template(family, judgment_type="acceptability")
    response_options = tuple(family.anchor.response_space.options)
    return template.with_(
        task_spec=template.task_spec.with_(options=response_options)
    )


def run_simulation(
    initial_size: int = 50,
    budget_per_iteration: int = 20,
    max_iterations: int = 10,
    convergence_threshold: float = 0.05,
    temperature: float = 1.0,
    random_state: int | None = None,
    output_dir: Path | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Run complete simulation of active learning pipeline.

    Parameters
    ----------
    initial_size : int
        Initial training set size
    budget_per_iteration : int
        Items to annotate per iteration
    max_iterations : int
        Maximum AL iterations
    convergence_threshold : float
        Convergence threshold for stopping
    temperature : float
        Temperature for simulated judgments (higher = more noise)
    random_state : int | None
        Random seed
    output_dir : Path | None
        Directory to save simulation results
    max_items : int | None
        Maximum total items to use (for quick testing)

    Returns
    -------
    dict[str, Any]
        Simulation results including convergence metrics
    """
    print_header("Argument Structure Active Learning Pipeline Simulation")

    config_table = create_summary_table(
        {
            "Initial size": str(initial_size),
            "Budget/iteration": str(budget_per_iteration),
            "Max iterations": str(max_iterations),
            "Temperature": str(temperature),
            "Random state": str(random_state),
        },
        title="Configuration",
    )
    console.print(config_table)
    console.print()

    # setup output directory
    if output_dir is None:
        output_dir = Path("simulation_output")
    output_dir.mkdir(exist_ok=True)

    # set random seed
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # [1/7] Load data
    print_header("[1/7] Loading Data")
    pairs_path = Path("items/2afc_pairs.jsonl")

    if not pairs_path.exists():
        print_error(f"2AFC pairs not found: {pairs_path}")
        print_info("Run: make 2afc-pairs")
        raise FileNotFoundError(f"2AFC pairs not found: {pairs_path}")

    # load and sample data
    if max_items is None:
        max_items = initial_size + budget_per_iteration * max_iterations

    all_pairs = load_2afc_pairs(pairs_path, limit=max_items)

    if len(all_pairs) < initial_size:
        print_error(f"Not enough items: need {initial_size}, found {len(all_pairs)}")
        raise ValueError(
            f"Not enough items: need {initial_size}, found {len(all_pairs)}"
        )

    # shuffle and split
    random.shuffle(all_pairs)
    initial_items = all_pairs[:initial_size]
    unlabeled_pool = all_pairs[initial_size:]

    print_success(f"Loaded {len(all_pairs)} 2AFC pairs")
    console.print(f"  - Initial set: {len(initial_items)}")
    console.print(f"  - Unlabeled pool: {len(unlabeled_pool)}")
    console.print()

    # [2/7] Setup simulated annotator
    print_header("[2/7] Setting Up Simulated Annotator")

    # create annotator configuration using bead.simulation framework
    annotator_config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(
            noise_type="temperature",
            temperature=temperature,
        ),
        random_state=random_state,
        fallback_to_random=True,
    )

    # create annotator from configuration
    annotator = SimulatedAnnotator.from_config(annotator_config)

    print_success("Simulated annotator initialized")
    console.print("  - Strategy: lm_score")
    console.print(f"  - Temperature: {temperature}")
    console.print(f"  - Random state: {random_state}")
    console.print()

    # [3/7] Generate initial annotations
    print_header("[3/7] Generating Initial Annotations")

    # create ItemTemplate for the simulation
    item_template = get_forced_choice_template()

    # generate initial annotations using the simulation framework
    human_ratings = annotator.annotate_batch(initial_items, item_template)
    print_success(f"Generated {len(human_ratings)} initial annotations")

    # compute simulated human agreement (sample twice with different seeds)
    # create two new annotators with different random states for agreement calculation
    annotator_sample1 = SimulatedAnnotator.from_config(
        annotator_config.with_(random_state=(random_state or 0) + 1000)
    )
    annotator_sample2 = SimulatedAnnotator.from_config(
        annotator_config.with_(random_state=(random_state or 0) + 2000)
    )

    sample1 = annotator_sample1.annotate_batch(initial_items, item_template)
    sample2 = annotator_sample2.annotate_batch(initial_items, item_template)

    labels1 = [sample1[str(item.id)] for item in initial_items]
    labels2 = [sample2[str(item.id)] for item in initial_items]

    inter_annotator = InterAnnotatorMetrics()
    human_agreement = inter_annotator.cohens_kappa(labels1, labels2)

    kappa_msg = f"  - Simulated human agreement (Cohen's kappa): {human_agreement:.3f}"
    console.print(kappa_msg)
    console.print()

    # [4/7] Setup convergence detection
    print_header("[4/7] Setting Up Convergence Detection")
    convergence_detector = ConvergenceDetector(
        human_agreement_metric="percentage_agreement",  # Using agreement as proxy
        convergence_threshold=convergence_threshold,
        min_iterations=2,
        alpha=0.05,
    )
    print_success("Convergence detector initialized")
    console.print(f"  - Convergence threshold: {convergence_threshold}")
    console.print()

    # [5/7] Setup active learning
    print_header("[5/7] Setting Up Active Learning")

    # create model with configuration
    model_config = ForcedChoiceModelConfig(
        model_name="bert-base-uncased",
        num_epochs=3,
        batch_size=16,
        device="cpu",
    )
    model = ForcedChoiceModel(config=model_config)

    # create selector with configuration
    selector_config = UncertaintySamplerConfig(method="entropy")
    item_selector = UncertaintySampler(config=selector_config)

    # create loop with configuration
    loop_config = ActiveLearningLoopConfig(
        max_iterations=max_iterations,
        budget_per_iteration=budget_per_iteration,
    )
    ActiveLearningLoop(
        item_selector=item_selector,
        config=loop_config,
    )

    print_success("Active learning components initialized")
    console.print("  - Strategy: Uncertainty sampling (entropy)")
    console.print("  - Model: ForcedChoiceModel (BERT-based)")
    console.print()

    # [6/7] Run active learning loop
    print_header("[6/7] Running Active Learning Loop")
    console.print()

    iteration_results = []
    current_labeled = initial_items.copy()
    current_unlabeled = unlabeled_pool.copy()
    converged = False

    for iteration in range(max_iterations):
        console.print(f"  Iteration {iteration + 1}/{max_iterations}")
        console.print("  " + "-" * 70)

        # extract labels
        labels = [human_ratings[str(item.id)] for item in current_labeled]

        # train model
        console.print(f"    Training on {len(current_labeled)} items...")
        train_metrics = model.train(current_labeled, labels)
        console.print(f"    Train accuracy: {train_metrics['train_accuracy']:.3f}")

        # evaluate on held-out data
        # sample from unlabeled pool for testing
        test_size = min(50, len(current_unlabeled))
        if test_size > 0:
            test_items = random.sample(current_unlabeled, test_size)
            test_annotations = annotator.annotate_batch(test_items, item_template)
            test_labels = [test_annotations[str(item.id)] for item in test_items]

            predictions = model.predict(test_items)
            pred_labels = [p.predicted_class for p in predictions]

            test_accuracy = accuracy_score(test_labels, pred_labels)
            console.print(f"    Test accuracy: {test_accuracy:.3f}")
        else:
            # no unlabeled items left, use training accuracy
            test_accuracy = train_metrics["train_accuracy"]
            console.print(f"    Test accuracy: {test_accuracy:.3f} (using train)")

        # store results
        iteration_results.append(
            {
                "iteration": iteration + 1,
                "train_accuracy": train_metrics["train_accuracy"],
                "test_accuracy": test_accuracy,
                "n_labeled": len(current_labeled),
                "n_unlabeled": len(current_unlabeled),
            }
        )

        # check convergence
        converged = convergence_detector.check_convergence(
            model_accuracy=test_accuracy,
            iteration=iteration + 1,
            human_agreement=human_agreement,
        )

        gap = abs(test_accuracy - human_agreement)
        console.print(f"    Agreement gap: {gap:.3f}")

        if converged:
            print_success("Converged!")
            break

        # select next batch
        if not current_unlabeled:
            print_info("No more unlabeled items")
            break

        n_select = min(budget_per_iteration, len(current_unlabeled))
        console.print(f"    Selecting {n_select} items for annotation...")

        selected_items = item_selector.select(
            items=current_unlabeled,
            model=model,
            predict_fn=lambda m, i: m.predict_proba([i])[0],  # Return 1D array
            budget=n_select,
        )

        # simulate annotations for selected items using the simulation framework
        new_annotations = annotator.annotate_batch(selected_items, item_template)
        human_ratings.update(new_annotations)

        # update sets
        current_labeled.extend(selected_items)
        current_unlabeled = [
            item
            for item in current_unlabeled
            if str(item.id) not in {str(s.id) for s in selected_items}
        ]

        console.print()

    # [7/7] Summary
    print_header("[7/7] Simulation Complete")

    final_accuracy = iteration_results[-1]["test_accuracy"]
    final_gap = abs(final_accuracy - human_agreement)
    summary_table = create_summary_table(
        {
            "Iterations completed": str(len(iteration_results)),
            "Total annotations": str(len(human_ratings)),
            "Final test accuracy": f"{final_accuracy:.3f}",
            "Simulated human agreement": f"{human_agreement:.3f}",
            "Final gap": f"{final_gap:.3f}",
            "Status": "CONVERGED" if converged else "MAX ITERATIONS REACHED",
        },
        title="Summary",
    )
    console.print(summary_table)

    if converged:
        print_success("Pipeline converged successfully")
    else:
        print_warning("Max iterations reached without convergence")

    # save results
    results = {
        "config": {
            "initial_size": initial_size,
            "budget_per_iteration": budget_per_iteration,
            "max_iterations": max_iterations,
            "convergence_threshold": convergence_threshold,
            "temperature": temperature,
            "random_state": random_state,
        },
        "human_agreement": human_agreement,
        "iterations": iteration_results,
        "converged": converged,
        "total_annotations": len(human_ratings),
    }

    results_path = output_dir / "simulation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print_success(f"Results saved to: {results_path}")
    console.print()

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate active learning pipeline with synthetic judgments"
    )
    parser.add_argument(
        "--initial-size",
        type=int,
        default=50,
        help="Initial training set size (default: 50)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Items to annotate per iteration (default: 20)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum AL iterations (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Convergence threshold (default: 0.05)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Judgment noise temperature (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_output"),
        help="Output directory (default: simulation_output)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum total items to use (default: None = use all needed)",
    )

    args = parser.parse_args()

    run_simulation(
        initial_size=args.initial_size,
        budget_per_iteration=args.budget,
        max_iterations=args.max_iterations,
        convergence_threshold=args.threshold,
        temperature=args.temperature,
        random_state=args.seed,
        output_dir=args.output_dir,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()
