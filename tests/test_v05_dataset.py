import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embody.dataset import TEACHER_POLICY_STATUS, generate_heuristic_demonstration_dataset
from scripts.run_v04_policy_comparison import comparison_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V05DatasetTest(unittest.TestCase):
    def test_generate_v05_dataset_outputs_files_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "v0_5"
            summary = generate_heuristic_demonstration_dataset(
                comparison_cases(),
                output_dir,
            )

            expert_path = output_dir / "expert_demonstrations.csv"
            train_path = output_dir / "train.csv"
            test_path = output_dir / "test.csv"
            summary_path = output_dir / "dataset_summary.json"

            self.assertTrue(expert_path.exists())
            self.assertTrue(train_path.exists())
            self.assertTrue(test_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertGreater(summary["total_rows"], 0)

            rows = self._read_rows(expert_path)
            first = rows[0]
            for field in [
                "x",
                "y",
                "direction",
                "next_x",
                "next_y",
                "next_direction",
                "front_blocked",
                "left_blocked",
                "right_blocked",
                "true_front_blocked",
                "true_left_blocked",
                "true_right_blocked",
                "distance_to_goal",
                "dx_to_goal",
                "dy_to_goal",
                "action",
            ]:
                self.assertIn(field, first)

            self.assertEqual(first["dataset_type"], "heuristic_demonstration")
            self.assertEqual(first["policy_name"], "obstacle_aware_goal")
            self.assertEqual(first["policy_version"], "v0.4.2")
            self.assertTrue(all(row["is_demonstration_accepted"] == "true" for row in rows))

            self.assertFalse(self._forward_true_blocked_moves(rows))

            train_rows = self._read_rows(train_path)
            test_rows = self._read_rows(test_path)
            self.assertGreater(len(train_rows), 0)
            self.assertGreater(len(test_rows), 0)

            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_summary["total_rows"], len(rows))
            self.assertEqual(loaded_summary["train_rows"], len(train_rows))
            self.assertEqual(loaded_summary["test_rows"], len(test_rows))
            self.assertEqual(
                loaded_summary["teacher_policy_status"],
                TEACHER_POLICY_STATUS,
            )

    def test_unreachable_case_is_not_in_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "v0_5"
            summary = generate_heuristic_demonstration_dataset(
                comparison_cases(),
                output_dir,
            )

            rows = self._read_rows(output_dir / "expert_demonstrations.csv")
            case_names = {row["case_name"] for row in rows}
            skipped = {item["case_name"] for item in summary["skipped_cases"]}

            self.assertNotIn("blocked_unreachable_6x6", case_names)
            self.assertIn("blocked_unreachable_6x6", skipped)

    def test_dataset_observation_fields_are_pre_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "v0_5"
            generate_heuristic_demonstration_dataset(
                comparison_cases(),
                output_dir,
            )

            rows = self._read_rows(output_dir / "expert_demonstrations.csv")
            sample = next(
                row
                for row in rows
                if row["case_name"] == "no_obstacle_8x8" and row["step"] == "6"
            )

            self.assertEqual(sample["x"], "6")
            self.assertEqual(sample["y"], "0")
            self.assertEqual(sample["direction"], "east")
            self.assertEqual(sample["next_x"], "7")
            self.assertEqual(sample["next_y"], "0")
            self.assertEqual(sample["next_direction"], "east")
            self.assertEqual(sample["front_blocked"], "0")
            self.assertEqual(sample["true_front_blocked"], "0")

    def test_generate_v05_script_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/generate_v05_dataset.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("dataset_type: heuristic_demonstration", completed.stdout)
        self.assertTrue((PROJECT_ROOT / "data/datasets/v0_5/expert_demonstrations.csv").exists())

    def test_existing_entrypoints_still_run(self) -> None:
        commands = [
            ["scripts/run_v01.py", "reached_goal: True"],
            ["scripts/run_v02_experiments.py", "total_runs: 3"],
            ["scripts/run_v03_uncertainty.py", "total_runs: 4"],
            ["scripts/run_v04_policy_comparison.py", "best_policy_by_success_rate"],
            ["scripts/cleanup_results.py", "Mode: dry-run"],
        ]

        for script, expected in commands:
            with self.subTest(script=script):
                completed = subprocess.run(
                    [sys.executable, script],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn(expected, completed.stdout)

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as csv_file:
            return list(csv.DictReader(csv_file))

    def _forward_true_blocked_moves(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        anomalies = []
        for row in rows:
            if row["action"] != "forward" or row["true_front_blocked"] != "1":
                continue
            moved = (
                row["x"],
                row["y"],
                row["direction"],
            ) != (
                row["next_x"],
                row["next_y"],
                row["next_direction"],
            )
            if moved:
                anomalies.append(row)
        return anomalies


if __name__ == "__main__":
    unittest.main()
