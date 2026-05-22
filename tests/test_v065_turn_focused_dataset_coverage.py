import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.train_v065_turn_focused_dataset_coverage import (
    build_coverage_audit,
    run_turn_focused_coverage,
    turn_focused_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V065TurnFocusedDatasetCoverageTest(unittest.TestCase):
    def test_turn_focused_cases_can_be_generated(self) -> None:
        cases = turn_focused_cases()
        case_names = {case.name for case in cases}

        self.assertIn("left_turn_required", case_names)
        self.assertIn("right_turn_required", case_names)
        self.assertIn("front_blocked_left_open", case_names)
        self.assertIn("front_blocked_right_open", case_names)
        self.assertIn("corridor_left_turn", case_names)
        self.assertIn("corridor_right_turn", case_names)
        self.assertIn("consecutive_turns", case_names)

    def test_coverage_audit_marks_not_ready_when_gate_fails(self) -> None:
        audit = build_coverage_audit([], [])

        self.assertEqual(audit["action_distribution"]["forward"], 0)
        self.assertEqual(audit["action_distribution"]["turn_left"], 0)
        self.assertEqual(audit["action_distribution"]["turn_right"], 0)
        self.assertIn("front_blocked_state_count", audit)
        self.assertTrue(audit["dataset_not_ready_for_behavior_cloning"])
        self.assertFalse(audit["dataset_ready_for_behavior_cloning"])

    def test_v065_outputs_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "v0_6_5"
            summary = run_turn_focused_coverage(results_dir)

            self.assertIn("action_distribution", summary)
            self.assertIn("front_blocked_state_summary", summary)
            self.assertIn("dataset_not_ready_for_behavior_cloning", summary)
            expected_files = [
                "turn_focused_case_summary.json",
                "dataset_coverage_audit.json",
                "action_distribution_by_case.json",
                "front_blocked_state_summary.json",
                "v065_summary.json",
            ]
            for filename in expected_files:
                with self.subTest(filename=filename):
                    self.assertTrue((results_dir / filename).exists())

    def test_v065_script_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/train_v065_turn_focused_dataset_coverage.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("dataset_ready_for_behavior_cloning:", completed.stdout)
        summary_path = PROJECT_ROOT / "experiments/v0_6_5/results/v065_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("front_blocked_state_summary", summary)


if __name__ == "__main__":
    unittest.main()
