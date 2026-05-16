import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embody.grid_world import Action, Direction, GridWorld, RobotState, run_episode
from embody.policy import GreedyGridPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AlwaysForwardPolicy:
    def choose_action(self, world, state, observation):
        return Action.FORWARD


class V03UncertaintyTest(unittest.TestCase):
    def test_action_fail_zero_keeps_default_step_behavior(self) -> None:
        clean_world = GridWorld(width=4, height=4, obstacles=set(), goal=(3, 0))
        explicit_world = GridWorld(
            width=4,
            height=4,
            obstacles=set(),
            goal=(3, 0),
            action_fail_prob=0.0,
            sensor_noise_prob=0.0,
            random_seed=1,
        )
        state = RobotState(x=0, y=0, direction=Direction.EAST)

        clean_result = clean_world.step(state, Action.FORWARD)
        explicit_result = explicit_world.step(state, Action.FORWARD)

        self.assertEqual(clean_result.state, explicit_result.state)
        self.assertEqual(clean_result.reward, explicit_result.reward)
        self.assertTrue(explicit_result.action_success)
        self.assertFalse(explicit_result.collision)

    def test_action_fail_one_keeps_robot_in_place(self) -> None:
        world = GridWorld(
            width=4,
            height=4,
            obstacles=set(),
            goal=(3, 0),
            action_fail_prob=1.0,
            random_seed=1,
        )
        state = RobotState(x=0, y=0, direction=Direction.EAST)

        result = world.step(state, Action.FORWARD)

        self.assertEqual(result.state, state)
        self.assertFalse(result.action_success)
        self.assertEqual(result.event, "action_failed")
        self.assertEqual(result.reward, -0.5)

    def test_sensor_noise_zero_observation_equals_true_observation(self) -> None:
        world = GridWorld(
            width=3,
            height=3,
            obstacles={(1, 0)},
            goal=(2, 2),
            sensor_noise_prob=0.0,
            random_seed=1,
        )
        state = RobotState(x=0, y=0, direction=Direction.EAST)

        observation = world.observe(state)

        self.assertEqual(observation.front_blocked, observation.true_front_blocked)
        self.assertEqual(observation.left_blocked, observation.true_left_blocked)
        self.assertEqual(observation.right_blocked, observation.true_right_blocked)

    def test_same_random_seed_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = self._run_seeded_episode(Path(tmp_dir) / "first")
            second = self._run_seeded_episode(Path(tmp_dir) / "second")

        first_trace = [(record.x, record.y, record.event) for record in first.records]
        second_trace = [(record.x, record.y, record.event) for record in second.records]
        self.assertEqual(first_trace, second_trace)
        self.assertEqual(first.action_failure_count, second.action_failure_count)
        self.assertEqual(first.collision_count, second.collision_count)

    def test_collision_is_recorded(self) -> None:
        world = GridWorld(width=3, height=3, obstacles={(1, 0)}, goal=(2, 2))
        state = RobotState(x=0, y=0, direction=Direction.EAST)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_episode(
                world=world,
                start=state,
                policy=AlwaysForwardPolicy(),
                output_dir=Path(tmp_dir),
                run_id="collision_case",
                max_steps=1,
            )

        self.assertEqual(result.collision_count, 1)
        self.assertEqual(result.records[0].collision, True)
        self.assertEqual(result.records[0].event, "collision")

    def test_v01_and_v02_scripts_still_run(self) -> None:
        v01 = subprocess.run(
            [sys.executable, "scripts/run_v01.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        v02 = subprocess.run(
            [sys.executable, "scripts/run_v02_experiments.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("reached_goal: True", v01.stdout)
        self.assertIn("total_runs: 3", v02.stdout)

    def _run_seeded_episode(self, output_dir: Path):
        world = GridWorld(
            width=8,
            height=8,
            obstacles={(2, 1), (2, 2), (2, 3), (4, 4), (5, 4), (5, 5)},
            goal=(7, 7),
            action_fail_prob=0.25,
            sensor_noise_prob=0.25,
            random_seed=42,
        )
        return run_episode(
            world=world,
            start=RobotState(x=0, y=0, direction=Direction.EAST),
            policy=GreedyGridPolicy(),
            output_dir=output_dir,
            run_id="seeded",
            max_steps=20,
        )


if __name__ == "__main__":
    unittest.main()
