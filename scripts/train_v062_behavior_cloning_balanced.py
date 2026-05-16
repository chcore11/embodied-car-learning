from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.behavior_cloning import (
    ACTIONS,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    KNNBehaviorCloningPolicy,
    build_failure_analysis,
    encode_row_features,
    evaluate_offline,
    evaluate_rollouts,
    load_rows,
    train_knn_policy,
)
from embody.dataset import TEACHER_POLICY, TEACHER_POLICY_STATUS
from embody.experiments import write_json
from scripts.run_v04_policy_comparison import comparison_cases


DATASET_SOURCE = "data/datasets/v0_5"


def main() -> None:
    train_csv = Path(DATASET_SOURCE) / "train.csv"
    test_csv = Path(DATASET_SOURCE) / "test.csv"
    model_dir = Path("models/v0_6_2")
    results_dir = Path("experiments/v0_6_2/results")

    train_rows = load_rows(train_csv)
    test_rows = load_rows(test_csv)
    models = [
        {
            "key": "knn_majority",
            "policy_name": "behavior_cloning_knn_majority",
            "model_path": model_dir / "behavior_cloning_knn_majority.json",
            "predictions_path": results_dir / "offline_predictions_majority.csv",
            "rollout_summary_name": "rollout_summary_majority.json",
            "failure_analysis_path": results_dir / "failure_analysis_majority.json",
            "voting_mode": "majority",
        },
        {
            "key": "knn_class_balanced",
            "policy_name": "behavior_cloning_knn_class_balanced",
            "model_path": model_dir / "behavior_cloning_knn_class_balanced.json",
            "predictions_path": results_dir / "offline_predictions_class_balanced.csv",
            "rollout_summary_name": "rollout_summary_class_balanced.json",
            "failure_analysis_path": results_dir / "failure_analysis_class_balanced.json",
            "voting_mode": "class_balanced",
        },
    ]

    offline_metrics = {}
    rollout_metrics = {}
    failure_analyses = {}
    trained_policies = {}
    implementation_checks = {}

    for spec in models:
        policy = train_knn_policy(
            train_csv,
            k=5,
            per_class_k=3,
            voting_mode=spec["voting_mode"],
            name=spec["policy_name"],
        )
        policy.save(spec["model_path"])
        loaded_policy = KNNBehaviorCloningPolicy.load(spec["model_path"])
        implementation_checks[spec["key"]] = {
            "saved_voting_mode": policy.voting_mode,
            "loaded_voting_mode": loaded_policy.voting_mode,
            "predict_uses_class_balanced_branch": loaded_policy.voting_mode == "class_balanced",
        }
        trained_policies[spec["key"]] = policy

        offline = evaluate_offline(policy, test_rows, spec["predictions_path"])
        rollout = evaluate_rollouts(
            policy,
            comparison_cases(),
            results_dir,
            summary_name=spec["rollout_summary_name"],
        )
        failure_analysis = build_failure_analysis(train_rows, offline, rollout)
        write_json(spec["failure_analysis_path"], failure_analysis)

        offline_metrics[spec["key"]] = offline
        rollout_metrics[spec["key"]] = {
            key: value for key, value in rollout.items() if key != "case_analyses"
        }
        failure_analyses[spec["key"]] = {
            "whether_policy_collapsed_to_forward": failure_analysis[
                "whether_policy_collapsed_to_forward"
            ],
            "per_case_dominant_predicted_action": failure_analysis[
                "per_case_dominant_predicted_action"
            ],
        }

    distance_summary = build_class_distance_diagnostics(
        train_rows=train_rows,
        test_rows=test_rows,
        majority_policy=trained_policies["knn_majority"],
        class_balanced_policy=trained_policies["knn_class_balanced"],
        debug_csv_path=results_dir / "class_balanced_debug_turn_cases.csv",
        implementation_checks=implementation_checks,
    )
    write_json(results_dir / "class_distance_summary.json", distance_summary)

    comparison = {
        "dataset_source": DATASET_SOURCE,
        "teacher_policy": TEACHER_POLICY,
        "teacher_policy_status": TEACHER_POLICY_STATUS,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "models_compared": ["knn_majority", "knn_class_balanced"],
        "offline_metrics": offline_metrics,
        "rollout_metrics": rollout_metrics,
        "failure_analysis_overview": failure_analyses,
        "class_distance_summary": distance_summary,
        "limitations": [
            "The dataset is still small and imbalanced.",
            "Class-balanced KNN is only a baseline correction, not a final policy.",
            "Improved offline turn prediction does not guarantee online rollout success.",
            "The teacher policy is heuristic, not optimal.",
            "true_* and post-action fields are excluded from model inputs to avoid leakage.",
        ],
    }
    write_json(results_dir / "comparison_summary.json", comparison)

    majority = offline_metrics["knn_majority"]
    balanced = offline_metrics["knn_class_balanced"]
    majority_rollout = rollout_metrics["knn_majority"]
    balanced_rollout = rollout_metrics["knn_class_balanced"]
    print(f"results_dir: {results_dir}")
    print(f"knn_majority_offline_accuracy: {majority['action_accuracy']:.4f}")
    print(f"knn_majority_balanced_accuracy: {majority['balanced_action_accuracy']:.4f}")
    print(
        "knn_majority_predicted_action_distribution: "
        f"{majority['predicted_action_distribution']}"
    )
    print(f"knn_majority_rollout_success_rate: {majority_rollout['success_rate']:.4f}")
    print(f"knn_class_balanced_offline_accuracy: {balanced['action_accuracy']:.4f}")
    print(
        "knn_class_balanced_balanced_accuracy: "
        f"{balanced['balanced_action_accuracy']:.4f}"
    )
    print(
        "knn_class_balanced_predicted_action_distribution: "
        f"{balanced['predicted_action_distribution']}"
    )
    print(
        "knn_class_balanced_rollout_success_rate: "
        f"{balanced_rollout['success_rate']:.4f}"
    )
    print(f"class_distance_final_determination: {distance_summary['final_determination']}")


def build_class_distance_diagnostics(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    majority_policy: KNNBehaviorCloningPolicy,
    class_balanced_policy: KNNBehaviorCloningPolicy,
    debug_csv_path: Path,
    implementation_checks: dict,
) -> dict:
    debug_rows = []
    smallest_distance_counts = {action: 0 for action in ACTIONS}
    actual_turn_left_smallest = {action: 0 for action in ACTIONS}
    actual_turn_right_smallest = {action: 0 for action in ACTIONS}

    for row_index, row in enumerate(test_rows):
        features = encode_row_features(row)
        scores = class_balanced_policy.class_distance_scores(features)
        chosen_class = choose_class_by_distance(scores)
        if chosen_class:
            smallest_distance_counts[chosen_class] += 1
        if row["action"] == "turn_left" and chosen_class:
            actual_turn_left_smallest[chosen_class] += 1
        if row["action"] == "turn_right" and chosen_class:
            actual_turn_right_smallest[chosen_class] += 1

        if row["action"] in {"turn_left", "turn_right"}:
            majority_prediction = majority_policy.predict_action(row).value
            balanced_prediction = class_balanced_policy.predict_action(row).value
            debug_rows.append(
                build_turn_debug_row(
                    row_index,
                    row,
                    scores,
                    chosen_class,
                    majority_prediction,
                    balanced_prediction,
                )
            )

    write_turn_debug_csv(debug_csv_path, debug_rows)
    class_balanced_works = (
        implementation_checks["knn_class_balanced"]["loaded_voting_mode"]
        == "class_balanced"
    )
    final_determination = (
        "B"
        if class_balanced_works
        and smallest_distance_counts["forward"] >= smallest_distance_counts["turn_left"]
        and smallest_distance_counts["forward"] >= smallest_distance_counts["turn_right"]
        else "C"
    )
    if not class_balanced_works:
        final_determination = "A"

    return {
        "implementation_checks": implementation_checks,
        "train_action_distribution": count_actions(train_rows),
        "test_action_distribution": count_actions(test_rows),
        "smallest_distance_counts": smallest_distance_counts,
        "actual_turn_left_rows_smallest_distance_class": actual_turn_left_smallest,
        "actual_turn_right_rows_smallest_distance_class": actual_turn_right_smallest,
        "debug_turn_case_count": len(debug_rows),
        "final_determination": final_determination,
        "final_determination_label": {
            "A": "class-balanced implementation bug",
            "B": "class-balanced implementation works, but current feature distance still favors forward",
            "C": "evidence insufficient",
        }[final_determination],
    }


def build_turn_debug_row(
    row_index: int,
    row: dict[str, str],
    scores: dict,
    chosen_class: str | None,
    majority_prediction: str,
    balanced_prediction: str,
) -> dict:
    return {
        "row_index": row_index,
        "actual_action": row["action"],
        "predicted_action_majority": majority_prediction,
        "predicted_action_class_balanced": balanced_prediction,
        "x": row["x"],
        "y": row["y"],
        "direction": row["direction"],
        "front_blocked": row["front_blocked"],
        "left_blocked": row["left_blocked"],
        "right_blocked": row["right_blocked"],
        "distance_to_goal": row["distance_to_goal"],
        "dx_to_goal": row["dx_to_goal"],
        "dy_to_goal": row["dy_to_goal"],
        "forward_nearest_distance": scores["forward"]["nearest_distance"],
        "forward_average_distance": scores["forward"]["average_distance"],
        "turn_left_nearest_distance": scores["turn_left"]["nearest_distance"],
        "turn_left_average_distance": scores["turn_left"]["average_distance"],
        "turn_right_nearest_distance": scores["turn_right"]["nearest_distance"],
        "turn_right_average_distance": scores["turn_right"]["average_distance"],
        "chosen_class_by_distance": chosen_class or "",
        "explanation": (
            f"class-balanced chose {chosen_class} because it has the smallest "
            "per-class average distance"
        ),
    }


def choose_class_by_distance(scores: dict) -> str | None:
    candidates = [
        (score["average_distance"], ACTIONS.index(action), action)
        for action, score in scores.items()
        if score["average_distance"] is not None
    ]
    if not candidates:
        return None
    return min(candidates)[2]


def write_turn_debug_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "actual_action",
        "predicted_action_majority",
        "predicted_action_class_balanced",
        "x",
        "y",
        "direction",
        "front_blocked",
        "left_blocked",
        "right_blocked",
        "distance_to_goal",
        "dx_to_goal",
        "dy_to_goal",
        "forward_nearest_distance",
        "forward_average_distance",
        "turn_left_nearest_distance",
        "turn_left_average_distance",
        "turn_right_nearest_distance",
        "turn_right_average_distance",
        "chosen_class_by_distance",
        "explanation",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_actions(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {action: 0 for action in ACTIONS}
    for row in rows:
        if row["action"] in counts:
            counts[row["action"]] += 1
    return counts


if __name__ == "__main__":
    main()
