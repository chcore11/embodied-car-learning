from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.experiments import ExperimentCase, run_experiments
from embody.grid_world import Direction, RobotState


def uncertainty_cases() -> list[ExperimentCase]:
    base = {
        "width": 8,
        "height": 8,
        "start": RobotState(x=0, y=0, direction=Direction.EAST),
        "goal": (7, 7),
        "obstacles": {(2, 1), (2, 2), (2, 3), (4, 4), (5, 4), (5, 5)},
        "max_steps": 64,
        "expected_reachable": True,
    }
    return [
        ExperimentCase(
            name="clean_baseline_8x8",
            action_fail_prob=0.0,
            sensor_noise_prob=0.0,
            random_seed=300,
            **base,
        ),
        ExperimentCase(
            name="action_noise_8x8",
            action_fail_prob=0.15,
            sensor_noise_prob=0.0,
            random_seed=301,
            **base,
        ),
        ExperimentCase(
            name="sensor_noise_8x8",
            action_fail_prob=0.0,
            sensor_noise_prob=0.15,
            random_seed=302,
            **base,
        ),
        ExperimentCase(
            name="combined_noise_8x8",
            action_fail_prob=0.1,
            sensor_noise_prob=0.1,
            random_seed=303,
            **base,
        ),
    ]


def main() -> None:
    results_dir = Path("experiments/v0_3/results")
    overall = run_experiments(uncertainty_cases(), results_dir)

    print(f"results_dir: {results_dir}")
    print(f"total_runs: {overall['total_runs']}")
    print(f"success_runs: {overall['success_runs']}")
    print(f"failed_runs: {overall['failed_runs']}")
    print(f"success_rate: {overall['success_rate']:.2f}")
    print(f"average_steps: {overall['average_steps']:.2f}")
    print(f"average_reward: {overall['average_reward']:.2f}")
    print(f"total_collisions: {overall['total_collisions']}")
    print(f"total_action_failures: {overall['total_action_failures']}")


if __name__ == "__main__":
    main()
