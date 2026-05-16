import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embody.behavior_cloning import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    KNNBehaviorCloningPolicy,
    evaluate_offline,
    load_rows,
    train_knn_policy,
)
from embody.dataset import generate_heuristic_demonstration_dataset
from scripts.run_v04_policy_comparison import comparison_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V062BehaviorCloningBalancedTest(unittest.TestCase):
    def test_class_balanced_knn_can_predict_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "v0_5"
            generate_heuristic_demonstration_dataset(comparison_cases(), dataset_dir)
            policy = train_knn_policy(
                dataset_dir / "train.csv",
                k=5,
                per_class_k=3,
                voting_mode="class_balanced",
                name="behavior_cloning_knn_class_balanced",
            )

            action = policy.predict_action(load_rows(dataset_dir / "test.csv")[0])

            self.assertIn(action.value, {"forward", "turn_left", "turn_right"})
            self.assertEqual(policy.voting_mode, "class_balanced")

    def test_class_balanced_knn_excludes_leakage_features(self) -> None:
        leakage_columns = {
            "true_front_blocked",
            "true_left_blocked",
            "true_right_blocked",
            "next_x",
            "next_y",
            "next_direction",
            "reward",
            "done",
            "case_name",
            "episode_id",
            "step",
        }

        self.assertTrue(leakage_columns.issubset(set(EXCLUDED_COLUMNS)))
        self.assertFalse(leakage_columns.intersection(FEATURE_COLUMNS))

    def test_class_balanced_model_can_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "v0_5"
            model_path = Path(tmp_dir) / "class_balanced.json"
            generate_heuristic_demonstration_dataset(comparison_cases(), dataset_dir)
            policy = train_knn_policy(
                dataset_dir / "train.csv",
                k=5,
                per_class_k=3,
                voting_mode="class_balanced",
                name="behavior_cloning_knn_class_balanced",
            )
            test_row = load_rows(dataset_dir / "test.csv")[0]

            policy.save(model_path)
            loaded = KNNBehaviorCloningPolicy.load(model_path)

            self.assertEqual(loaded.voting_mode, "class_balanced")
            self.assertEqual(loaded.per_class_k, 3)
            self.assertEqual(loaded.predict_action(test_row), policy.predict_action(test_row))

    def test_balanced_metrics_include_distribution_and_action_accuracies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "v0_5"
            generate_heuristic_demonstration_dataset(comparison_cases(), dataset_dir)
            policy = train_knn_policy(
                dataset_dir / "train.csv",
                k=5,
                per_class_k=3,
                voting_mode="class_balanced",
            )

            metrics = evaluate_offline(policy, load_rows(dataset_dir / "test.csv"))

            self.assertIn("predicted_action_distribution", metrics)
            self.assertIn("forward_only_baseline_accuracy", metrics)
            self.assertIn("macro_action_accuracy", metrics)
            self.assertIn("balanced_action_accuracy", metrics)

    def test_v062_comparison_summary_can_be_generated(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/train_v062_behavior_cloning_balanced.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("knn_class_balanced_predicted_action_distribution", completed.stdout)
        summary_path = PROJECT_ROOT / "experiments/v0_6_2/results/comparison_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(
            summary["models_compared"],
            ["knn_majority", "knn_class_balanced"],
        )
        self.assertIn("predicted_action_distribution", summary["offline_metrics"]["knn_majority"])
        self.assertIn(
            "balanced_action_accuracy",
            summary["offline_metrics"]["knn_class_balanced"],
        )
        self.assertTrue(
            (
                PROJECT_ROOT
                / "experiments/v0_6_2/results/failure_analysis_class_balanced.json"
            ).exists()
        )
        distance_summary_path = (
            PROJECT_ROOT / "experiments/v0_6_2/results/class_distance_summary.json"
        )
        debug_csv_path = (
            PROJECT_ROOT
            / "experiments/v0_6_2/results/class_balanced_debug_turn_cases.csv"
        )
        self.assertTrue(distance_summary_path.exists())
        self.assertTrue(debug_csv_path.exists())
        distance_summary = json.loads(distance_summary_path.read_text(encoding="utf-8"))
        self.assertIn("smallest_distance_counts", distance_summary)
        self.assertIn("actual_turn_left_rows_smallest_distance_class", distance_summary)
        self.assertIn("actual_turn_right_rows_smallest_distance_class", distance_summary)
        self.assertIn(distance_summary["final_determination"], {"A", "B", "C"})


if __name__ == "__main__":
    unittest.main()
