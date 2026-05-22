import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embody.behavior_cloning import (
    DERIVED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    KNNBehaviorCloningPolicy,
    encode_row_features,
    load_rows,
    train_knn_policy,
)
from embody.dataset import generate_heuristic_demonstration_dataset
from scripts.run_v04_policy_comparison import comparison_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V064DerivedFeaturesBehaviorCloningTest(unittest.TestCase):
    def test_derived_feature_generation_runs(self) -> None:
        row = {
            "x": "1",
            "y": "2",
            "direction": "east",
            "front_blocked": "1",
            "left_blocked": "0",
            "right_blocked": "1",
            "distance_to_goal": "5",
            "dx_to_goal": "3",
            "dy_to_goal": "-2",
        }

        base = encode_row_features(row, feature_mode="base")
        derived = encode_row_features(row, feature_mode="derived")

        self.assertEqual(len(base), len(FEATURE_COLUMNS))
        self.assertEqual(len(derived), len(DERIVED_FEATURE_COLUMNS))
        self.assertGreater(len(derived), len(base))

    def test_derived_feature_columns_exclude_leakage(self) -> None:
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
            "action",
        }

        self.assertFalse(leakage_columns.intersection(DERIVED_FEATURE_COLUMNS))

    def test_feature_mode_base_keeps_original_feature_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "v0_5"
            generate_heuristic_demonstration_dataset(comparison_cases(), dataset_dir)
            policy = train_knn_policy(dataset_dir / "train.csv", feature_mode="base")

            self.assertEqual(policy.feature_mode, "base")
            self.assertEqual(policy.feature_columns, FEATURE_COLUMNS)
            self.assertEqual(len(policy.samples[0]["features"]), len(FEATURE_COLUMNS))

    def test_derived_policy_predicts_and_round_trips_feature_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "v0_5"
            model_path = Path(tmp_dir) / "derived_model.json"
            generate_heuristic_demonstration_dataset(comparison_cases(), dataset_dir)
            policy = train_knn_policy(
                dataset_dir / "train.csv",
                feature_mode="derived",
                voting_mode="class_balanced",
            )
            row = load_rows(dataset_dir / "test.csv")[0]

            action = policy.predict_action(row)
            policy.save(model_path)
            loaded = KNNBehaviorCloningPolicy.load(model_path)

            self.assertIn(action.value, {"forward", "turn_left", "turn_right"})
            self.assertEqual(loaded.feature_mode, "derived")
            self.assertEqual(loaded.voting_mode, "class_balanced")
            self.assertEqual(loaded.predict_action(row), action)

    def test_v064_comparison_summary_can_be_generated(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/train_v064_derived_features_behavior_cloning.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("derived_class_balanced", completed.stdout)
        summary_path = PROJECT_ROOT / "experiments/v0_6_4/results/comparison_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(
            summary["models_compared"],
            [
                "base_majority",
                "base_class_balanced",
                "derived_majority",
                "derived_class_balanced",
            ],
        )
        self.assertIn("derived_feature_columns", summary)
        self.assertIn("feature_space_comparison", summary)
        self.assertTrue(
            (
                PROJECT_ROOT
                / "experiments/v0_6_4/results/feature_space_summary_derived.json"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
