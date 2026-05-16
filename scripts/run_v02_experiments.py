from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.experiments import default_cases, run_experiments


def main() -> None:
    results_dir = Path("experiments/v0_2/results")
    overall = run_experiments(default_cases(), results_dir)

    print(f"results_dir: {results_dir}")
    print(f"total_runs: {overall['total_runs']}")
    print(f"success_runs: {overall['success_runs']}")
    print(f"failed_runs: {overall['failed_runs']}")
    print(f"success_rate: {overall['success_rate']:.2f}")
    print(f"reachable_runs: {overall['reachable_runs']}")
    print(f"reachable_success_runs: {overall['reachable_success_runs']}")
    print(f"reachable_failed_runs: {overall['reachable_failed_runs']}")
    print(f"reachable_success_rate: {overall['reachable_success_rate']:.2f}")
    print(f"expected_unreachable_runs: {overall['expected_unreachable_runs']}")
    print(f"unexpected_success_runs: {overall['unexpected_success_runs']}")
    print(f"policy_failed_runs: {overall['policy_failed_runs']}")
    print(f"average_steps: {overall['average_steps']:.2f}")
    print(f"average_reward: {overall['average_reward']:.2f}")


if __name__ == "__main__":
    main()
