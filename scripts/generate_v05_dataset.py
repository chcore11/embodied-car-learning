from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.dataset import generate_heuristic_demonstration_dataset
from scripts.run_v04_policy_comparison import comparison_cases


def main() -> None:
    output_dir = Path("data/datasets/v0_5")
    summary = generate_heuristic_demonstration_dataset(
        cases=comparison_cases(),
        output_dir=output_dir,
    )

    print(f"output_dir: {output_dir}")
    print(f"dataset_type: {summary['dataset_type']}")
    print(f"teacher_policy: {summary['teacher_policy']}")
    print(f"teacher_policy_status: {summary['teacher_policy_status']}")
    print(f"total_rows: {summary['total_rows']}")
    print(f"train_rows: {summary['train_rows']}")
    print(f"test_rows: {summary['test_rows']}")
    print(f"accepted_episodes: {summary['accepted_episodes']}")
    print(f"rejected_episodes: {summary['rejected_episodes']}")


if __name__ == "__main__":
    main()
