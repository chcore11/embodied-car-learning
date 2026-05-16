from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.behavior_cloning import (
    build_behavior_cloning_summary,
    build_failure_analysis,
    evaluate_offline,
    evaluate_rollouts,
    load_rows,
    train_knn_policy,
)
from embody.experiments import write_json
from scripts.run_v04_policy_comparison import comparison_cases


def main() -> None:
    train_csv = Path("data/datasets/v0_5/train.csv")
    test_csv = Path("data/datasets/v0_5/test.csv")
    model_path = Path("models/v0_6/behavior_cloning_policy.json")
    results_dir = Path("experiments/v0_6/results")
    predictions_path = results_dir / "offline_predictions.csv"
    summary_path = results_dir / "behavior_cloning_summary.json"
    failure_analysis_path = results_dir / "failure_analysis.json"

    policy = train_knn_policy(train_csv)
    policy.save(model_path)

    train_rows = load_rows(train_csv)
    test_rows = load_rows(test_csv)
    offline_metrics = evaluate_offline(policy, test_rows, predictions_path)
    rollout_metrics = evaluate_rollouts(policy, comparison_cases(), results_dir)
    summary = build_behavior_cloning_summary(train_rows, offline_metrics, rollout_metrics)
    failure_analysis = build_failure_analysis(train_rows, offline_metrics, rollout_metrics)
    write_json(summary_path, summary)
    write_json(failure_analysis_path, failure_analysis)

    print(f"model_path: {model_path}")
    print(f"results_dir: {results_dir}")
    print(f"model_type: {summary['model_type']}")
    print(f"offline_action_accuracy: {offline_metrics['action_accuracy']:.4f}")
    print(f"macro_action_accuracy: {offline_metrics['macro_action_accuracy']:.4f}")
    print(f"balanced_action_accuracy: {offline_metrics['balanced_action_accuracy']:.4f}")
    print(
        "whether_policy_collapsed_to_forward: "
        f"{failure_analysis['whether_policy_collapsed_to_forward']}"
    )
    print(f"rollout_success_rate: {rollout_metrics['success_rate']:.4f}")
    print(f"total_rollout_runs: {rollout_metrics['total_runs']}")


if __name__ == "__main__":
    main()
