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
    build_behavior_cloning_summary,
    build_failure_analysis,
    evaluate_offline,
    evaluate_rollouts,
    load_rows,
    train_knn_policy,
)
from embody.dataset import generate_heuristic_demonstration_dataset
from scripts.run_v04_policy_comparison import comparison_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V06BehaviorCloningTest(unittest.TestCase):
    def test_train_and_test_csv_can_be_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "v0_5"
            generate_heuristic_demonstration_dataset(comparison_cases(), dataset_dir)

            train_rows = load_rows(dataset_dir / "train.csv")
            test_rows = load_rows(dataset_dir / "test.csv")

            self.assertGreater(len(train_rows), 0)
            self.assertGreater(len(test_rows), 0)

    def test_feature_columns_exclude_leakage_columns(self) -> None:
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

    def test_knn_policy_predicts_and_round_trips_model_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "v0_5"
            model_path = Path(tmp_dir) / "model.json"
            generate_heuristic_demonstration_dataset(comparison_cases(), dataset_dir)

            policy = train_knn_policy(dataset_dir / "train.csv")
            test_row = load_rows(dataset_dir / "test.csv")[0]
            action = policy.predict_action(test_row)
            self.assertIn(action.value, {"forward", "turn_left", "turn_right"})

            policy.save(model_path)
            loaded = KNNBehaviorCloningPolicy.load(model_path)
            self.assertEqual(loaded.predict_action(test_row), action)

    def test_offline_evaluation_generates_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "v0_5"
            predictions_path = Path(tmp_dir) / "offline_predictions.csv"
            generate_heuristic_demonstration_dataset(comparison_cases(), dataset_dir)
            policy = train_knn_policy(dataset_dir / "train.csv")

            metrics = evaluate_offline(
                policy,
                load_rows(dataset_dir / "test.csv"),
                predictions_path,
            )

            self.assertIn("action_accuracy", metrics)
            self.assertIn("predicted_action_distribution", metrics)
            self.assertIn("forward_only_baseline_accuracy", metrics)
            self.assertIn("macro_action_accuracy", metrics)
            self.assertIn("balanced_action_accuracy", metrics)
            self.assertGreaterEqual(metrics["action_accuracy"], 0.0)
            self.assertLessEqual(metrics["action_accuracy"], 1.0)
            self.assertTrue(predictions_path.exists())

    def test_behavior_cloning_summary_and_rollouts_can_be_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "v0_5"
            results_dir = root / "results"
            generate_heuristic_demonstration_dataset(comparison_cases(), dataset_dir)
            policy = train_knn_policy(dataset_dir / "train.csv")

            train_rows = load_rows(dataset_dir / "train.csv")
            offline_metrics = evaluate_offline(policy, load_rows(dataset_dir / "test.csv"))
            rollout_metrics = evaluate_rollouts(policy, comparison_cases(), results_dir)
            summary = build_behavior_cloning_summary(
                train_rows,
                offline_metrics,
                rollout_metrics,
            )
            failure_analysis = build_failure_analysis(
                train_rows,
                offline_metrics,
                rollout_metrics,
            )

            self.assertEqual(summary["model_type"], "knn_behavior_cloning")
            self.assertIn("train_action_distribution", summary)
            self.assertIn("predicted_action_distribution", summary)
            self.assertIn("offline_metrics", summary)
            self.assertIn("rollout_metrics", summary)
            self.assertIn("rollout_failure_cases", failure_analysis)
            self.assertIn("whether_policy_collapsed_to_forward", failure_analysis)
            self.assertTrue((results_dir / "rollout_summary.json").exists())

    def test_train_v06_script_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/train_v06_behavior_cloning.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("model_type: knn_behavior_cloning", completed.stdout)
        self.assertTrue((PROJECT_ROOT / "models/v0_6/behavior_cloning_policy.json").exists())
        summary_path = PROJECT_ROOT / "experiments/v0_6/results/behavior_cloning_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("action_accuracy", summary["offline_metrics"])
        self.assertIn("macro_action_accuracy", summary)
        failure_analysis_path = PROJECT_ROOT / "experiments/v0_6/results/failure_analysis.json"
        self.assertTrue(failure_analysis_path.exists())
        failure_analysis = json.loads(failure_analysis_path.read_text(encoding="utf-8"))
        self.assertIn("per_case_first_collision_step", failure_analysis)

    def test_existing_entrypoints_still_run(self) -> None:
        commands = [
            ["scripts/run_v01.py", "reached_goal: True"],
            ["scripts/run_v02_experiments.py", "total_runs: 3"],
            ["scripts/run_v03_uncertainty.py", "total_runs: 4"],
            ["scripts/run_v04_policy_comparison.py", "best_policy_by_success_rate"],
            ["scripts/generate_v05_dataset.py", "dataset_type: heuristic_demonstration"],
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


if __name__ == "__main__":
    unittest.main()
