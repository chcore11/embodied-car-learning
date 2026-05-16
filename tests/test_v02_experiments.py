import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embody.experiments import (
    ExperimentCase,
    build_overall_summary,
    default_cases,
    run_experiments,
)
from embody.grid_world import Direction, RobotState


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V02ExperimentsTest(unittest.TestCase):
    def test_run_experiments_writes_case_outputs_and_overall_summary(self) -> None:
        cases = [
            ExperimentCase(
                name="tiny_success",
                width=3,
                height=3,
                start=RobotState(x=0, y=0, direction=Direction.EAST),
                goal=(2, 0),
                obstacles=set(),
                max_steps=8,
                expected_reachable=True,
            ),
            ExperimentCase(
                name="tiny_blocked",
                width=3,
                height=3,
                start=RobotState(x=0, y=0, direction=Direction.EAST),
                goal=(2, 2),
                obstacles={(1, 0), (0, 1)},
                max_steps=4,
                expected_reachable=False,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            overall = run_experiments(cases, results_dir)

            self.assertEqual(overall["total_runs"], 2)
            self.assertEqual(overall["success_runs"], 1)
            self.assertEqual(overall["failed_runs"], 1)
            self.assertEqual(overall["success_rate"], 0.5)
            self.assertEqual(overall["reachable_runs"], 1)
            self.assertEqual(overall["reachable_success_runs"], 1)
            self.assertEqual(overall["reachable_failed_runs"], 0)
            self.assertEqual(overall["reachable_success_rate"], 1.0)
            self.assertEqual(overall["expected_unreachable_runs"], 1)
            self.assertEqual(overall["unexpected_success_runs"], 0)
            self.assertEqual(overall["policy_failed_runs"], 0)
            self.assertIn("average_steps", overall)
            self.assertIn("average_reward", overall)

            overall_path = results_dir / "overall_summary.json"
            self.assertTrue(overall_path.exists())
            self.assertEqual(json.loads(overall_path.read_text(encoding="utf-8")), overall)

            for case in cases:
                case_dir = results_dir / case.name
                self.assertTrue((case_dir / "run_log.csv").exists())
                self.assertTrue((case_dir / "trajectory.png").exists())
                self.assertTrue((case_dir / "summary.json").exists())

                summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["case_name"], case.name)
                self.assertEqual(summary["expected_reachable"], case.expected_reachable)
                self.assertIn("steps", summary)
                self.assertIn("reached_goal", summary)
                self.assertIn("total_reward", summary)
                self.assertEqual(Path(summary["csv_path"]).name, "run_log.csv")
                self.assertEqual(Path(summary["trajectory_path"]).name, "trajectory.png")
                self.assertIn("failure_reason", summary)

            success_summary = json.loads(
                (results_dir / "tiny_success" / "summary.json").read_text(encoding="utf-8")
            )
            blocked_summary = json.loads(
                (results_dir / "tiny_blocked" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(success_summary["result_type"], "success")
            self.assertEqual(blocked_summary["result_type"], "expected_unreachable")

    def test_default_cases_include_expected_reachable_flags(self) -> None:
        cases = default_cases()

        self.assertTrue(all(isinstance(case.expected_reachable, bool) for case in cases))
        self.assertEqual(
            {case.name: case.expected_reachable for case in cases},
            {
                "no_obstacle_8x8": True,
                "obstacle_reachable_8x8": True,
                "blocked_unreachable_6x6": False,
            },
        )

    def test_build_overall_summary_handles_empty_input(self) -> None:
        overall = build_overall_summary([])

        self.assertEqual(overall["total_runs"], 0)
        self.assertEqual(overall["success_runs"], 0)
        self.assertEqual(overall["failed_runs"], 0)
        self.assertEqual(overall["success_rate"], 0.0)
        self.assertEqual(overall["reachable_runs"], 0)
        self.assertEqual(overall["reachable_success_runs"], 0)
        self.assertEqual(overall["reachable_failed_runs"], 0)
        self.assertEqual(overall["reachable_success_rate"], 0.0)
        self.assertEqual(overall["expected_unreachable_runs"], 0)
        self.assertEqual(overall["unexpected_success_runs"], 0)
        self.assertEqual(overall["policy_failed_runs"], 0)
        self.assertEqual(overall["average_steps"], 0.0)
        self.assertEqual(overall["average_reward"], 0.0)

    def test_v01_demo_script_still_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/run_v01.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("reached_goal: True", completed.stdout)


if __name__ == "__main__":
    unittest.main()
