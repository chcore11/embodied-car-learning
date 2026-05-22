import json
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "v0_6_7" / "results"


class V067BehaviorCloningErrorDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            [sys.executable, "scripts/train_v067_behavior_cloning_error_diagnosis.py"],
            cwd=PROJECT_ROOT,
            check=True,
        )

    def read_json(self, name):
        return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))

    def test_required_outputs_are_generated(self):
        expected_files = [
            "confusion_matrix_summary.json",
            "front_blocked_error_cases.json",
            "turn_left_error_analysis.json",
            "rollout_collision_summary.json",
            "collision_heavy_cases.json",
            "model_difference_summary.json",
            "v067_summary.json",
        ]
        for filename in expected_files:
            with self.subTest(filename=filename):
                self.assertTrue((RESULTS_DIR / filename).exists())

    def test_confusion_summary_tracks_turn_left_errors(self):
        summary = self.read_json("confusion_matrix_summary.json")
        for model_name in ["knn_majority", "knn_class_balanced"]:
            model_summary = summary[model_name]
            self.assertIn("turn_left_as_forward_count", model_summary)
            self.assertIn("turn_left_as_turn_right_count", model_summary)
            self.assertIn("turn_left_recall", model_summary)

    def test_front_blocked_forward_errors_are_reported(self):
        summary = self.read_json("front_blocked_error_cases.json")
        for model_name in ["knn_majority", "knn_class_balanced"]:
            model_summary = summary[model_name]
            self.assertIn("front_blocked_predicted_forward_count", model_summary)
            self.assertIn("front_blocked_predicted_forward_cases", model_summary)
            self.assertIn("main_source", model_summary)

    def test_rollout_collision_summary_has_clean_success_metrics(self):
        summary = self.read_json("rollout_collision_summary.json")
        for model_name in ["knn_majority", "knn_class_balanced"]:
            model_summary = summary[model_name]
            self.assertIn("clean_success_count", model_summary)
            self.assertIn("clean_success_rate", model_summary)
            self.assertIn("average_collisions_per_success", model_summary)
            self.assertIn("cases", model_summary)

    def test_v067_summary_answers_required_questions(self):
        summary = self.read_json("v067_summary.json")
        self.assertIn("main_bottleneck", summary)
        self.assertIn("next_priority", summary)
        self.assertIn("class_balanced_stability_assessment", summary)
        self.assertIn("rollout_collision_overview", summary)
        self.assertIn("answers", summary)


if __name__ == "__main__":
    unittest.main()
