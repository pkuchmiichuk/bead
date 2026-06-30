"""Simulation runner for orchestrating multi-annotator simulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bead.config.simulation import SimulationRunnerConfig
    from bead.items.item import Item
    from bead.items.item_template import ItemTemplate


class SimulationRunner:
    """Orchestrates multi-annotator simulation.

    Can simulate:
    - Multiple independent annotators
    - Correlated annotators (shared noise component)
    - Mixed strategies (some LM-based, some random)

    Parameters
    ----------
    config
        Configuration for simulation.

    Examples
    --------
    >>> from bead.config.simulation import (  # doctest: +SKIP
    ...     SimulationRunnerConfig,
    ...     SimulatedAnnotatorConfig,
    ... )
    >>> config = SimulationRunnerConfig(  # doctest: +SKIP
    ...     annotator_configs=[
    ...         SimulatedAnnotatorConfig(strategy="lm_score", random_state=1),
    ...         SimulatedAnnotatorConfig(strategy="lm_score", random_state=2),
    ...     ],
    ...     n_annotators=2
    ... )
    >>> runner = SimulationRunner(config)
    >>> # results = runner.run(items, templates)
    """

    def __init__(self, config: SimulationRunnerConfig) -> None:
        self.config = config

        # create annotators from configs
        from bead.simulation.annotators.base import (  # noqa: PLC0415
            SimulatedAnnotator,
        )

        self.annotators = [
            SimulatedAnnotator.from_config(cfg) for cfg in config.annotator_configs
        ]

        # if n_annotators > len(annotator_configs), replicate first config
        if config.n_annotators > len(self.annotators):
            base_config = config.annotator_configs[0]
            for i in range(len(self.annotators), config.n_annotators):
                # create new config with different seed
                new_config = base_config.with_(
                    random_state=(base_config.random_state or 0) + i
                )
                self.annotators.append(SimulatedAnnotator.from_config(new_config))

    def run(
        self,
        items: list[Item],
        templates: list[ItemTemplate] | ItemTemplate,
    ) -> dict[str, list[str | int | float | list[str]]]:
        """Run simulation with all annotators.

        Parameters
        ----------
        items : list[Item]
            Items to annotate.
        templates : list[ItemTemplate] | ItemTemplate
            Templates (one per item or shared).

        Returns
        -------
        dict[str, list[str | int | float | list[str]]]
            Results: {
                "item_ids": [...],
                "annotator_0": [...],
                "annotator_1": [...],
                ...
            }
        """
        # collect annotations from each annotator
        results: dict[str, list[str | int | float | list[str]]] = {
            "item_ids": [str(item.id) for item in items]
        }

        for i, annotator in enumerate(self.annotators):
            annotations = annotator.annotate_batch(items, templates)
            results[f"annotator_{i}"] = [annotations[str(item.id)] for item in items]

        # save if configured
        if self.config.save_path:
            self.save_results(results)

        return results

    def save_results(
        self, results: dict[str, list[str | int | float | list[str]]]
    ) -> None:
        """Save results to file.

        Parameters
        ----------
        results : dict[str, list[str | int | float | list[str]]]
            Simulation results.
        """
        if self.config.save_path is None:
            msg = "save_path not configured"
            raise ValueError(msg)

        path = Path(self.config.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.output_format == "jsonl":
            # write JSONL format
            with open(path, "w") as f:
                for i in range(len(results["item_ids"])):
                    row = {
                        "item_id": results["item_ids"][i],
                        **{
                            key: results[key][i] for key in results if key != "item_ids"
                        },
                    }
                    f.write(json.dumps(row) + "\n")

        elif self.config.output_format == "dict":
            # write JSON format
            with open(path, "w") as f:
                json.dump(results, f, indent=2)

        elif self.config.output_format == "dataframe":
            # write CSV format (optional dependency)
            import pandas as pd  # noqa: PLC0415

            df = pd.DataFrame(results)
            df.to_csv(path, index=False)
