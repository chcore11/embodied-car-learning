from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.grid_world import create_default_world, run_episode
from embody.policy import GreedyGridPolicy


def main() -> None:
    world, start = create_default_world()
    result = run_episode(
        world=world,
        start=start,
        policy=GreedyGridPolicy(),
        output_dir=Path("data/runs"),
        run_id="v0_1_demo",
    )

    print(f"run_id: {result.run_id}")
    print(f"steps: {len(result.records)}")
    print(f"reached_goal: {result.reached_goal}")
    print(f"total_reward: {result.total_reward:.2f}")
    print(f"csv: {result.csv_path}")
    print(f"trajectory: {result.trajectory_path}")


if __name__ == "__main__":
    main()
