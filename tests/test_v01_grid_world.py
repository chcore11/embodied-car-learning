import tempfile
import unittest
from pathlib import Path

from embody.grid_world import Action, Direction, GridWorld, RobotState, create_default_world, run_episode
from embody.policy import GreedyGridPolicy


class GridWorldTest(unittest.TestCase):
    def test_observation_reports_blocked_cells_and_distance(self) -> None:
        world = GridWorld(width=3, height=3, obstacles={(1, 0)}, goal=(2, 2))
        state = RobotState(x=0, y=0, direction=Direction.EAST)

        observation = world.observe(state)

        self.assertTrue(observation.front_blocked)
        self.assertTrue(observation.left_blocked)
        self.assertFalse(observation.right_blocked)
        self.assertEqual(observation.distance_to_goal, 4)

    def test_forward_collision_keeps_state_and_penalizes(self) -> None:
        world = GridWorld(width=3, height=3, obstacles={(1, 0)}, goal=(2, 2))
        state = RobotState(x=0, y=0, direction=Direction.EAST)

        result = world.step(state, Action.FORWARD)

        self.assertEqual(result.state, state)
        self.assertEqual(result.event, "collision")
        self.assertLess(result.reward, 0)

    def test_default_episode_writes_csv_and_png(self) -> None:
        world, start = create_default_world()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_episode(
                world=world,
                start=start,
                policy=GreedyGridPolicy(),
                output_dir=Path(tmp_dir),
                run_id="test_run",
            )

            self.assertTrue(result.csv_path.exists())
            self.assertTrue(result.trajectory_path.exists())
            self.assertGreater(len(result.records), 0)
            self.assertEqual(result.trajectory_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
