"""Tests for generated JavaScript randomizer execution.

These tests execute the generated JavaScript code using Node.js to verify
that the constraint enforcement logic works correctly at runtime.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from bead.deployment.jspsych.randomizer import generate_randomizer_function
from bead.lists.constraints import OrderingConstraint, OrderingPair


def check_node_available() -> bool:
    """Check if Node.js is available on the system.

    Returns
    -------
    bool
        True if Node.js is available, False otherwise.
    """
    return shutil.which("node") is not None


requires_node = pytest.mark.skipif(
    not check_node_available(), reason="Node.js not available"
)


@requires_node
class TestJavaScriptExecution:
    """Tests for JavaScript execution with Node.js."""

    def test_basic_randomization(self, tmp_path: Path) -> None:
        """Test basic randomization without constraints."""
        item_ids = [uuid4() for _ in range(5)]
        constraints: list[OrderingConstraint] = []
        metadata = {item_id: {"condition": "A"} for item_id in item_ids}

        js_code = generate_randomizer_function(item_ids, constraints, metadata)

        # Create test script
        test_script = self._create_test_script(tmp_path, js_code, item_ids, seed=42)

        # Execute with Node.js
        result = subprocess.run(
            ["node", str(test_script)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output) == 5

    def test_precedence_constraint(self, tmp_path: Path) -> None:
        """Test precedence constraint enforcement."""
        item_ids = [uuid4() for _ in range(5)]
        constraint = OrderingConstraint(
            constraint_type="ordering",
            precedence_pairs=(OrderingPair(before=item_ids[0], after=item_ids[4]),),
        )
        metadata = {item_id: {"condition": "A"} for item_id in item_ids}

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        # Run multiple times with different seeds
        for seed in range(10):
            test_script = self._create_test_script(
                tmp_path, js_code, item_ids, seed=seed
            )

            result = subprocess.run(
                ["node", str(test_script)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0
            output = json.loads(result.stdout)

            # Find positions
            pos_a = next(
                i for i, t in enumerate(output) if t["item_id"] == str(item_ids[0])
            )
            pos_b = next(
                i for i, t in enumerate(output) if t["item_id"] == str(item_ids[4])
            )

            # item_ids[0] must come before item_ids[4]
            assert pos_a < pos_b

    def test_no_adjacent_constraint(self, tmp_path: Path) -> None:
        """Test no-adjacent constraint enforcement."""
        item_ids = [uuid4() for _ in range(6)]
        constraint = OrderingConstraint(
            constraint_type="ordering", no_adjacent_property="condition"
        )
        metadata = {
            item_ids[0]: {"condition": "A"},
            item_ids[1]: {"condition": "B"},
            item_ids[2]: {"condition": "A"},
            item_ids[3]: {"condition": "B"},
            item_ids[4]: {"condition": "A"},
            item_ids[5]: {"condition": "B"},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        # Run multiple times to check constraint is satisfied
        for seed in range(10):
            test_script = self._create_test_script(
                tmp_path, js_code, item_ids, seed=seed
            )

            result = subprocess.run(
                ["node", str(test_script)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0
            output = json.loads(result.stdout)

            # Check no adjacent items have same condition
            for i in range(len(output) - 1):
                item_id_a = output[i]["item_id"]
                item_id_b = output[i + 1]["item_id"]

                # Find UUIDs
                uuid_a = next(uid for uid in item_ids if str(uid) == item_id_a)
                uuid_b = next(uid for uid in item_ids if str(uid) == item_id_b)

                condition_a = metadata[uuid_a]["condition"]
                condition_b = metadata[uuid_b]["condition"]

                assert condition_a != condition_b

    def test_practice_items_first(self, tmp_path: Path) -> None:
        """Test practice items appear first."""
        item_ids = [uuid4() for _ in range(5)]
        constraint = OrderingConstraint(
            constraint_type="ordering", practice_item_property="is_practice"
        )
        metadata = {
            item_ids[0]: {"is_practice": False},
            item_ids[1]: {"is_practice": True},
            item_ids[2]: {"is_practice": False},
            item_ids[3]: {"is_practice": True},
            item_ids[4]: {"is_practice": False},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        # Run multiple times
        for seed in range(10):
            test_script = self._create_test_script(
                tmp_path, js_code, item_ids, seed=seed
            )

            result = subprocess.run(
                ["node", str(test_script)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0
            output = json.loads(result.stdout)

            # Find practice items
            practice_positions = []
            for i, trial in enumerate(output):
                uuid = next(uid for uid in item_ids if str(uid) == trial["item_id"])
                if metadata[uuid]["is_practice"]:
                    practice_positions.append(i)

            # All practice items should come before non-practice items
            if practice_positions:
                max_practice_pos = max(practice_positions)
                min_main_pos = min(
                    i
                    for i, trial in enumerate(output)
                    if not metadata[
                        next(uid for uid in item_ids if str(uid) == trial["item_id"])
                    ]["is_practice"]
                )
                assert max_practice_pos < min_main_pos

    def test_blocking_constraint(self, tmp_path: Path) -> None:
        """Test blocking creates contiguous blocks."""
        item_ids = [uuid4() for _ in range(6)]
        constraint = OrderingConstraint(
            constraint_type="ordering",
            block_by_property="block_type",
            randomize_within_blocks=False,
        )
        metadata = {
            item_ids[0]: {"block_type": "A"},
            item_ids[1]: {"block_type": "B"},
            item_ids[2]: {"block_type": "A"},
            item_ids[3]: {"block_type": "B"},
            item_ids[4]: {"block_type": "A"},
            item_ids[5]: {"block_type": "B"},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        test_script = self._create_test_script(tmp_path, js_code, item_ids, seed=42)

        result = subprocess.run(
            ["node", str(test_script)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)

        # Get block sequence
        block_sequence = []
        for trial in output:
            uuid = next(uid for uid in item_ids if str(uid) == trial["item_id"])
            block_sequence.append(metadata[uuid]["block_type"])

        # Check contiguity - find transitions
        transitions = 0
        for i in range(len(block_sequence) - 1):
            if block_sequence[i] != block_sequence[i + 1]:
                transitions += 1

        # Should have exactly 1 transition (2 blocks)
        assert transitions == 1

    def test_min_distance_constraint(self, tmp_path: Path) -> None:
        """Test minimum distance constraint enforcement."""
        item_ids = [uuid4() for _ in range(8)]
        constraint = OrderingConstraint(
            constraint_type="ordering", no_adjacent_property="condition", min_distance=2
        )
        metadata = {
            item_ids[0]: {"condition": "A"},
            item_ids[1]: {"condition": "B"},
            item_ids[2]: {"condition": "A"},
            item_ids[3]: {"condition": "B"},
            item_ids[4]: {"condition": "A"},
            item_ids[5]: {"condition": "B"},
            item_ids[6]: {"condition": "C"},
            item_ids[7]: {"condition": "C"},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        # Run multiple times
        for seed in range(10):
            test_script = self._create_test_script(
                tmp_path, js_code, item_ids, seed=seed
            )

            result = subprocess.run(
                ["node", str(test_script)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0
            output = json.loads(result.stdout)

            # Check minimum distance between same conditions
            condition_positions = {}
            for i, trial in enumerate(output):
                uuid = next(uid for uid in item_ids if str(uid) == trial["item_id"])
                condition = metadata[uuid]["condition"]

                if condition not in condition_positions:
                    condition_positions[condition] = []
                condition_positions[condition].append(i)

            # Check distances
            for positions in condition_positions.values():
                for i in range(len(positions) - 1):
                    distance = positions[i + 1] - positions[i] - 1
                    assert distance >= 2

    def _create_test_script(
        self, tmp_path: Path, js_code: str, item_ids: list, seed: int
    ) -> Path:
        """Create a Node.js test script.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory for the script.
        js_code : str
            Generated JavaScript code.
        item_ids : list
            List of item UUIDs.
        seed : int
            Random seed.

        Returns
        -------
        Path
            Path to the test script.
        """
        # Add seedrandom library (simple implementation for testing)
        seedrandom_code = """
        Math.seedrandom = function(seed) {
            var m = 0x80000000;
            var a = 1103515245;
            var c = 12345;
            var state = seed ? seed : Math.floor(Math.random() * (m - 1));

            return function() {
                state = (a * state + c) % m;
                return state / m;
            };
        };
        """

        # Create test data
        trials = [{"item_id": str(item_id)} for item_id in item_ids]

        test_code = f"""
        {seedrandom_code}

        {js_code}

        const trials = {json.dumps(trials)};
        const seed = {seed};

        const randomized = randomizeTrials(trials, seed);
        console.log(JSON.stringify(randomized));
        """

        script_path = tmp_path / f"test_script_{seed}.js"
        script_path.write_text(test_code)
        return script_path
