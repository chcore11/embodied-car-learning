import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.train_v065_turn_focused_dataset_coverage import turn_focused_cases
from scripts.train_v066_turn_focused_behavior_cloning import (
    build_dataset_coverage_audit,
    run_v066_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V066TurnFocusedBehaviorCloningTest(unittest.TestCase):
    def test_new_dataset_files_can_be_generated_without_overwriting_v05(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            v05_dir = root / "v0_5"
            v05_dir.mkdir()
            sentinel = v05_dir / "train.csv"
            sentinel.write_text("sentinel", encoding="utf-8")

            summary = run_v066_pipeline(
                dataset_dir=root / "v0_6_6",
                model_dir=root / "models",
                results_dir=root / "results",
            )

            self.assertTrue((root / "v0_6_6/train.csv").exists())
            self.assertTrue((root / "v0_6_6/test.csv").exists())
            self.assertTrue((root / "v0_6_6/dataset_summary.json").exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel")
            self.assertIn("dataset_coverage", summary)

    def test_coverage_gate_fields_exist(self) -> None:
        audit = build_dataset_coverage_audit([], [], [])

        self.assertIn("action_distribution", audit)
        self.assertIn("train_action_distribution", audit)
        self.assertIn("test_action_distribution", audit)
        self.assertIn("front_blocked_state_count", audit)
        self.assertIn("front_blocked_action_distribution", audit)
        self.assertTrue(audit["dataset_not_ready_for_behavior_cloning"])

    def test_dataset_not_ready_skips_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = run_v066_pipeline(
                dataset_dir=root / "dataset",
                model_dir=root / "models",
                results_dir=root / "results",
                cases=turn_focused_cases()[:1],
            )

            self.assertTrue(summary["training_skipped_due_to_dataset_not_ready"])
            self.assertFalse((root / "models/knn_majority.json").exists())

    def test_dataset_ready_generates_evaluation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = run_v066_pipeline(
                dataset_dir=root / "dataset",
                model_dir=root / "models",
                results_dir=root / "results",
            )

            self.assertFalse(summary["training_skipped_due_to_dataset_not_ready"])
            self.assertTrue((root / "results/behavior_cloning_evaluation_summary.json").exists())
            self.assertTrue((root / "results/prediction_distribution_summary.json").exists())
            self.assertIn("knn_majority", summary["prediction_distribution"])
            self.assertIn(
                "predicted_action_distribution",
                summary["prediction_distribution"]["knn_majority"],
            )

    def test_v066_script_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/train_v066_turn_focused_behavior_cloning.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("dataset_ready_for_behavior_cloning:", completed.stdout)
        summary_path = PROJECT_ROOT / "experiments/v0_6_6/results/v066_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("prediction_distribution", summary)


if __name__ == "__main__":
    unittest.main()
