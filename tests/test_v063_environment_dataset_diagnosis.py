import json
import subprocess
import sys
import unittest
from pathlib import Path

from embody.behavior_cloning import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V063EnvironmentDatasetDiagnosisTest(unittest.TestCase):
    def test_v063_diagnosis_script_generates_outputs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/train_v063_environment_dataset_diagnosis.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("final_determination:", completed.stdout)
        results_dir = PROJECT_ROOT / "experiments/v0_6_3/results"
        expected_files = [
            "dataset_coverage_summary.json",
            "turn_feature_space_summary.json",
            "turn_samples.csv",
            "ambiguous_state_summary.json",
            "critical_state_summary.json",
            "case_difficulty_summary.json",
            "environment_diagnosis_summary.json",
        ]
        for filename in expected_files:
            with self.subTest(filename=filename):
                self.assertTrue((results_dir / filename).exists())

    def test_v063_feature_columns_exclude_leakage_columns(self) -> None:
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

        self.assertFalse(leakage_columns.intersection(FEATURE_COLUMNS))

    def test_v063_environment_summary_contains_final_determination(self) -> None:
        subprocess.run(
            [sys.executable, "scripts/train_v063_environment_dataset_diagnosis.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        summary_path = (
            PROJECT_ROOT
            / "experiments/v0_6_3/results/environment_diagnosis_summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertIn(summary["final_determination"], {"A", "B", "C", "D"})
        self.assertIn("dataset_coverage_result", summary)
        self.assertIn("feature_space_result", summary)


if __name__ == "__main__":
    unittest.main()
