import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embody.grid_world import Action, Direction, GridWorld, RobotState
from embody.policy import GreedyGridPolicy, ObstacleAwareGoalPolicy
from scripts.run_v04_policy_comparison import (
    build_paired_metrics,
    comparison_cases,
    comparison_policies,
    run_policy_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V04PolicyComparisonTest(unittest.TestCase):
    def test_baseline_policy_can_select_action(self) -> None:
        world = GridWorld(width=4, height=4, obstacles=set(), goal=(3, 0))
        observation = world.observe(RobotState(x=0, y=0, direction=Direction.EAST))

        action = GreedyGridPolicy().select_action(observation)

        self.assertIn(action, {Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT})

    def test_obstacle_aware_policy_can_select_action(self) -> None:
        world = GridWorld(width=4, height=4, obstacles={(1, 0)}, goal=(3, 3))
        observation = world.observe(RobotState(x=0, y=0, direction=Direction.EAST))

        action = ObstacleAwareGoalPolicy().select_action(observation)

        self.assertIn(action, {Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT})
        self.assertNotEqual(action, Action.FORWARD)

    def test_observation_contains_goal_delta(self) -> None:
        world = GridWorld(width=4, height=4, obstacles=set(), goal=(3, 2))
        observation = world.observe(RobotState(x=1, y=0, direction=Direction.EAST))

        self.assertEqual(observation.x, 1)
        self.assertEqual(observation.y, 0)
        self.assertEqual(observation.direction, Direction.EAST)
        self.assertEqual(observation.dx_to_goal, 2)
        self.assertEqual(observation.dy_to_goal, 2)

    def test_v04_runner_writes_policy_comparison_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            summary = run_policy_comparison(results_dir)

            summary_path = results_dir / "policy_comparison_summary.json"
            self.assertTrue(summary_path.exists())
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8")),
                summary,
            )
            self.assertEqual(
                {item["policy_name"] for item in summary["policies"]},
                {"baseline_greedy", "obstacle_aware_goal"},
            )

            for item in summary["policies"]:
                self.assertIn("total_runs", item)
                self.assertIn("success_runs", item)
                self.assertIn("failed_runs", item)
                self.assertIn("success_rate", item)
                self.assertIn("average_steps_all_runs", item)
                self.assertIn("average_reward", item)
                self.assertIn("total_collisions", item)
                self.assertIn("total_action_failures", item)

            self.assertIn("best_policy_by_success_rate", summary)
            self.assertIn("best_policy_by_average_steps", summary)
            self.assertIn("per_case_comparison", summary)
            self.assertIn("common_success_cases", summary)
            self.assertIn("baseline_common_success_average_steps", summary)
            self.assertIn("improved_common_success_average_steps", summary)
            self.assertIn("common_success_average_step_delta", summary)
            self.assertIn("improved_extra_solved_cases", summary)
            self.assertIn("baseline_extra_solved_cases", summary)
            self.assertIn("both_success_cases", summary)
            self.assertIn("both_failed_cases", summary)
            self.assertIn("tie_by_success_rate", summary)
            self.assertIn("tie_by_average_steps", summary)
            self.assertIn("tie_by_common_success_average_steps", summary)
            self.assertIsInstance(summary["best_policy_by_success_rate"], list)
            self.assertIsInstance(summary["best_policy_by_average_steps"], list)

    def test_v04_runs_multiple_policies_and_cases(self) -> None:
        self.assertGreaterEqual(len(comparison_policies()), 2)
        self.assertGreaterEqual(len(comparison_cases()), 5)

    def test_case_summary_contains_policy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            run_policy_comparison(results_dir)

            summary_path = (
                results_dir
                / "baseline_greedy"
                / "turn_choice_8x8"
                / "summary.json"
            )
            case_summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(case_summary["policy_name"], "baseline_greedy")
            self.assertIn("action_counts", case_summary)
            self.assertIn("forward", case_summary["action_counts"])
            self.assertIn("turn_left", case_summary["action_counts"])
            self.assertIn("turn_right", case_summary["action_counts"])
            self.assertIn("unique_positions", case_summary)
            self.assertIn("repeated_position_count", case_summary)

    def test_policy_comparison_summary_supports_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            summary = run_policy_comparison(results_dir)

            if summary["tie_by_success_rate"]:
                self.assertGreater(len(summary["best_policy_by_success_rate"]), 1)
            if summary["tie_by_average_steps"]:
                self.assertGreater(len(summary["best_policy_by_average_steps"]), 1)

    def test_per_case_comparison_step_delta_only_for_both_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            summary = run_policy_comparison(results_dir)

            for item in summary["per_case_comparison"]:
                if item["result_type"] == "both_success":
                    self.assertIsInstance(item["step_delta"], int)
                else:
                    self.assertIsNone(item["step_delta"])
                self.assertIn("reward_delta", item)

    def test_paired_metrics_records_improved_extra_solved_cases(self) -> None:
        metrics = build_paired_metrics(
            [
                {"case_name": "case_a", "reached_goal": False, "steps": 10, "total_reward": -1.0},
                {"case_name": "case_b", "reached_goal": True, "steps": 8, "total_reward": 5.0},
            ],
            [
                {"case_name": "case_a", "reached_goal": True, "steps": 12, "total_reward": 4.0},
                {"case_name": "case_b", "reached_goal": True, "steps": 7, "total_reward": 6.0},
            ],
        )

        self.assertEqual(metrics["improved_extra_solved_cases"], ["case_a"])
        self.assertEqual(metrics["both_success_cases"], ["case_b"])
        self.assertEqual(metrics["common_success_cases"], ["case_b"])
        self.assertEqual(metrics["baseline_common_success_average_steps"], 8.0)
        self.assertEqual(metrics["improved_common_success_average_steps"], 7.0)
        self.assertEqual(metrics["common_success_average_step_delta"], -1.0)

    def test_best_policy_by_average_steps_uses_common_success_only(self) -> None:
        metrics = build_paired_metrics(
            [
                {"case_name": "shared", "reached_goal": True, "steps": 5, "total_reward": 4.0},
                {"case_name": "baseline_only", "reached_goal": True, "steps": 1, "total_reward": 2.0},
            ],
            [
                {"case_name": "shared", "reached_goal": True, "steps": 4, "total_reward": 5.0},
                {"case_name": "baseline_only", "reached_goal": False, "steps": 20, "total_reward": -2.0},
            ],
        )

        self.assertEqual(metrics["common_success_cases"], ["shared"])
        self.assertEqual(metrics["best_policy_by_average_steps"], ["obstacle_aware_goal"])
        self.assertFalse(metrics["tie_by_common_success_average_steps"])

    def test_existing_entrypoints_still_run(self) -> None:
        commands = [
            ["scripts/run_v01.py", "reached_goal: True"],
            ["scripts/run_v02_experiments.py", "total_runs: 3"],
            ["scripts/run_v03_uncertainty.py", "total_runs: 4"],
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


if __name__ == "__main__":
    unittest.main()
