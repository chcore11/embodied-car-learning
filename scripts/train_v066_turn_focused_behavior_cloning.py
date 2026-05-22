from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.behavior_cloning import (
    ACTIONS,
    build_failure_analysis,
    evaluate_offline,
    evaluate_rollouts,
    load_rows,
    train_knn_policy,
)
from embody.dataset import generate_heuristic_demonstration_dataset
from embody.experiments import write_json
from scripts.train_v065_turn_focused_dataset_coverage import COVERAGE_GATE, turn_focused_cases


DATASET_DIR = Path("data/datasets/v0_6_6")
MODEL_DIR = Path("models/v0_6_6")
RESULTS_DIR = Path("experiments/v0_6_6/results")


def main() -> None:
    summary = run_v066_pipeline()
    print(f"dataset_dir: {DATASET_DIR}")
    print(f"results_dir: {RESULTS_DIR}")
    print(f"action_distribution: {summary['dataset_coverage']['action_distribution']}")
    print(f"dataset_ready_for_behavior_cloning: {summary['dataset_ready_for_behavior_cloning']}")
    print(
        "training_skipped_due_to_dataset_not_ready: "
        f"{summary['training_skipped_due_to_dataset_not_ready']}"
    )
    if not summary["training_skipped_due_to_dataset_not_ready"]:
        for model_name, metrics in summary["prediction_distribution"].items():
            print(
                f"{model_name}: predicted={metrics['predicted_action_distribution']}, "
                f"turn_left_recall={metrics['turn_left_recall']}, "
                f"turn_right_recall={metrics['turn_right_recall']}, "
                f"success_rate={summary['rollout_metrics'][model_name]['success_rate']}"
            )


def run_v066_pipeline(
    dataset_dir: Path = DATASET_DIR,
    model_dir: Path = MODEL_DIR,
    results_dir: Path = RESULTS_DIR,
    cases=None,
) -> dict:
    selected_cases = cases or turn_focused_cases()
    dataset_summary = generate_heuristic_demonstration_dataset(selected_cases, dataset_dir)
    train_rows = load_rows(dataset_dir / "train.csv")
    test_rows = load_rows(dataset_dir / "test.csv")
    all_rows = load_rows(dataset_dir / "expert_demonstrations.csv")

    coverage = build_dataset_coverage_audit(all_rows, train_rows, test_rows)
    write_json(results_dir / "dataset_coverage_audit.json", coverage)

    if not coverage["dataset_ready_for_behavior_cloning"]:
        skipped_summary = {
            "version": "v0.6.6",
            "dataset_summary": dataset_summary,
            "dataset_coverage": coverage,
            "dataset_ready_for_behavior_cloning": False,
            "training_skipped_due_to_dataset_not_ready": True,
            "prediction_distribution": {},
            "rollout_metrics": {},
            "always_forward_collapse": {},
        }
        write_json(results_dir / "v066_summary.json", skipped_summary)
        write_json(results_dir / "behavior_cloning_evaluation_summary.json", skipped_summary)
        write_json(results_dir / "prediction_distribution_summary.json", {})
        write_json(results_dir / "front_blocked_prediction_summary.json", {})
        write_json(results_dir / "rollout_summary.json", {})
        return skipped_summary

    model_specs = [
        ("knn_majority", "majority"),
        ("knn_class_balanced", "class_balanced"),
    ]
    evaluation = {}
    prediction_summary = {}
    front_blocked_summary = {}
    rollout_summary = {}
    collapse_summary = {}

    for model_name, voting_mode in model_specs:
        policy = train_knn_policy(
            dataset_dir / "train.csv",
            k=5,
            per_class_k=3,
            voting_mode=voting_mode,
            name=f"v066_{model_name}",
        )
        model_path = model_dir / f"{model_name}.json"
        policy.save(model_path)

        offline = evaluate_offline(
            policy,
            test_rows,
            results_dir / f"offline_predictions_{model_name}.csv",
        )
        rollout = evaluate_rollouts(
            policy,
            selected_cases,
            results_dir,
            summary_name=f"rollout_summary_{model_name}.json",
        )
        failure = build_failure_analysis(train_rows, offline, rollout)
        rollout_clean = {key: value for key, value in rollout.items() if key != "case_analyses"}
        prediction = build_prediction_distribution(offline)
        front_blocked = build_front_blocked_prediction_summary(policy, test_rows)

        evaluation[model_name] = {
            "model_path": str(model_path),
            "offline_metrics": offline,
            "front_blocked_prediction_summary": front_blocked,
            "rollout_metrics": rollout_clean,
            "whether_policy_collapsed_to_forward": failure["whether_policy_collapsed_to_forward"],
        }
        prediction_summary[model_name] = prediction
        front_blocked_summary[model_name] = front_blocked
        rollout_summary[model_name] = rollout_clean
        collapse_summary[model_name] = failure["whether_policy_collapsed_to_forward"]

    final_summary = {
        "version": "v0.6.6",
        "dataset_summary": dataset_summary,
        "dataset_coverage": coverage,
        "dataset_ready_for_behavior_cloning": True,
        "training_skipped_due_to_dataset_not_ready": False,
        "prediction_distribution": prediction_summary,
        "front_blocked_prediction_summary": front_blocked_summary,
        "rollout_metrics": rollout_summary,
        "always_forward_collapse": collapse_summary,
    }

    write_json(results_dir / "behavior_cloning_evaluation_summary.json", evaluation)
    write_json(results_dir / "prediction_distribution_summary.json", prediction_summary)
    write_json(results_dir / "front_blocked_prediction_summary.json", front_blocked_summary)
    write_json(results_dir / "rollout_summary.json", rollout_summary)
    write_json(results_dir / "v066_summary.json", final_summary)
    return final_summary


def build_dataset_coverage_audit(
    rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> dict:
    action_counts = action_distribution(rows)
    total = sum(action_counts.values())
    forward_ratio = round(action_counts["forward"] / total, 4) if total else 0.0
    front_blocked_rows = [row for row in rows if row["front_blocked"] == "1"]
    gate_failures = []

    if action_counts["turn_left"] < COVERAGE_GATE["min_turn_left"]:
        gate_failures.append("turn_left_below_minimum")
    if action_counts["turn_right"] < COVERAGE_GATE["min_turn_right"]:
        gate_failures.append("turn_right_below_minimum")
    if len(front_blocked_rows) < COVERAGE_GATE["min_front_blocked_states"]:
        gate_failures.append("front_blocked_states_below_minimum")
    if forward_ratio > COVERAGE_GATE["max_forward_ratio"]:
        gate_failures.append("forward_ratio_too_high")

    return {
        "coverage_gate": COVERAGE_GATE,
        "action_distribution": action_counts,
        "train_action_distribution": action_distribution(train_rows),
        "test_action_distribution": action_distribution(test_rows),
        "forward_ratio": forward_ratio,
        "turn_left_count": action_counts["turn_left"],
        "turn_right_count": action_counts["turn_right"],
        "front_blocked_state_count": len(front_blocked_rows),
        "front_blocked_action_distribution": action_distribution(front_blocked_rows),
        "gate_failures": gate_failures,
        "dataset_ready_for_behavior_cloning": not gate_failures,
        "dataset_not_ready_for_behavior_cloning": bool(gate_failures),
    }


def build_prediction_distribution(offline_metrics: dict) -> dict:
    per_action = offline_metrics["per_action_accuracy"]
    return {
        "overall_accuracy": offline_metrics["action_accuracy"],
        "predicted_action_distribution": offline_metrics["predicted_action_distribution"],
        "turn_left_recall": per_action["turn_left"],
        "turn_right_recall": per_action["turn_right"],
        "balanced_action_accuracy": offline_metrics["balanced_action_accuracy"],
    }


def build_front_blocked_prediction_summary(policy, rows: list[dict[str, str]]) -> dict:
    front_blocked_rows = [row for row in rows if row["front_blocked"] == "1"]
    predictions = {action: 0 for action in ACTIONS}
    actual = action_distribution(front_blocked_rows)
    correct = 0
    for row in front_blocked_rows:
        predicted = policy.predict_action(row).value
        predictions[predicted] += 1
        correct += int(predicted == row["action"])
    total = len(front_blocked_rows)
    return {
        "front_blocked_test_rows": total,
        "front_blocked_actual_action_distribution": actual,
        "front_blocked_prediction_distribution": predictions,
        "front_blocked_accuracy": round(correct / total, 4) if total else None,
    }


def action_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {action: 0 for action in ACTIONS}
    for row in rows:
        action = row["action"]
        if action in counts:
            counts[action] += 1
    return counts


if __name__ == "__main__":
    main()
