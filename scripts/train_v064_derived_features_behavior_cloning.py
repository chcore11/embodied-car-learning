from pathlib import Path
import json
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.behavior_cloning import (
    ACTIONS,
    BASE_FEATURE_COLUMNS,
    DERIVED_FEATURE_COLUMNS,
    EXCLUDED_COLUMNS,
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
    model_dir = Path("models/v0_6_4")
    results_dir = Path("experiments/v0_6_4/results")
    train_rows = load_rows(train_csv)
    test_rows = load_rows(test_csv)
    all_rows = load_rows(Path(DATASET_SOURCE) / "expert_demonstrations.csv")

    specs = [
        ("base_majority", "base", "majority"),
        ("base_class_balanced", "base", "class_balanced"),
        ("derived_majority", "derived", "majority"),
        ("derived_class_balanced", "derived", "class_balanced"),
    ]
    offline_metrics = {}
    rollout_metrics = {}
    failure_overview = {}

    for key, feature_mode, voting_mode in specs:
        policy_name = f"knn_{key}"
        policy = train_knn_policy(
            train_csv,
            k=5,
            per_class_k=3,
            voting_mode=voting_mode,
            feature_mode=feature_mode,
            name=policy_name,
        )
        policy.save(model_dir / f"{policy_name}.json")

        offline = evaluate_offline(
            policy,
            test_rows,
            results_dir / f"offline_predictions_{key}.csv",
        )
        rollout = evaluate_rollouts(
            policy,
            comparison_cases(),
            results_dir,
            summary_name=f"rollout_summary_{key}.json",
        )
        failure = build_failure_analysis(train_rows, offline, rollout)

        offline_metrics[key] = offline
        rollout_metrics[key] = {
            item_key: value
            for item_key, value in rollout.items()
            if item_key != "case_analyses"
        }
        failure_overview[key] = {
            "whether_policy_collapsed_to_forward": failure[
                "whether_policy_collapsed_to_forward"
            ]
        }

    feature_space_base = build_feature_space_summary(all_rows, "base")
    feature_space_derived = build_feature_space_summary(all_rows, "derived")
    write_json(results_dir / "feature_space_summary_base.json", feature_space_base)
    write_json(results_dir / "feature_space_summary_derived.json", feature_space_derived)

    comparison = {
        "dataset_source": DATASET_SOURCE,
        "teacher_policy": TEACHER_POLICY,
        "teacher_policy_status": TEACHER_POLICY_STATUS,
        "models_compared": [key for key, _, _ in specs],
        "base_feature_columns": BASE_FEATURE_COLUMNS,
        "derived_feature_columns": DERIVED_FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "offline_metrics": offline_metrics,
        "rollout_metrics": rollout_metrics,
        "feature_space_comparison": {
            "base": feature_space_base,
            "derived": feature_space_derived,
        },
        "failure_overview": failure_overview,
        "limitations": [
            "Derived features are hand-designed and only use pre-action observations.",
            "No new benchmark cases were added in v0.6.4.",
            "Improved feature-space separation does not guarantee rollout success.",
            "The teacher policy is heuristic, not optimal.",
            "true_* and post-action fields are excluded from model inputs to avoid leakage.",
        ],
    }
    write_json(results_dir / "comparison_summary.json", comparison)

    print(f"results_dir: {results_dir}")
    for key, _, _ in specs:
        offline = offline_metrics[key]
        rollout = rollout_metrics[key]
        print(
            f"{key}: action_accuracy={offline['action_accuracy']:.4f}, "
            f"balanced_action_accuracy={offline['balanced_action_accuracy']:.4f}, "
            f"predicted={offline['predicted_action_distribution']}, "
            f"success_rate={rollout['success_rate']:.4f}, "
            f"collisions={rollout['total_collisions']}"
        )
    print(f"base_turn_left_closer_to_forward: {feature_space_base['turn_left_samples_closer_to_forward_than_turn_left_count']}")
    print(f"derived_turn_left_closer_to_forward: {feature_space_derived['turn_left_samples_closer_to_forward_than_turn_left_count']}")
    print(f"base_turn_right_closer_to_forward: {feature_space_base['turn_right_samples_closer_to_forward_than_turn_right_count']}")
    print(f"derived_turn_right_closer_to_forward: {feature_space_derived['turn_right_samples_closer_to_forward_than_turn_right_count']}")


def build_feature_space_summary(rows: list[dict[str, str]], feature_mode: str) -> dict:
    by_action = {
        action: [row for row in rows if row["action"] == action]
        for action in ACTIONS
    }
    turn_left_forward_closer = 0
    turn_right_forward_closer = 0
    for row in by_action["turn_left"]:
        if is_less(
            nearest_distance(row, by_action["forward"], feature_mode),
            nearest_distance(row, by_action["turn_left"], feature_mode, exclude_self=True),
        ):
            turn_left_forward_closer += 1
    for row in by_action["turn_right"]:
        if is_less(
            nearest_distance(row, by_action["forward"], feature_mode),
            nearest_distance(row, by_action["turn_right"], feature_mode, exclude_self=True),
        ):
            turn_right_forward_closer += 1
    return {
        "feature_mode": feature_mode,
        "turn_left_samples_closer_to_forward_than_turn_left_count": turn_left_forward_closer,
        "turn_right_samples_closer_to_forward_than_turn_right_count": turn_right_forward_closer,
        "average_nearest_forward_distance_for_turn_left": average(
            nearest_distance(row, by_action["forward"], feature_mode)
            for row in by_action["turn_left"]
        ),
        "average_nearest_same_action_distance_for_turn_left": average(
            nearest_distance(row, by_action["turn_left"], feature_mode, exclude_self=True)
            for row in by_action["turn_left"]
        ),
        "average_nearest_forward_distance_for_turn_right": average(
            nearest_distance(row, by_action["forward"], feature_mode)
            for row in by_action["turn_right"]
        ),
        "average_nearest_same_action_distance_for_turn_right": average(
            nearest_distance(row, by_action["turn_right"], feature_mode, exclude_self=True)
            for row in by_action["turn_right"]
        ),
    }


def nearest_distance(
    row: dict[str, str],
    candidates: list[dict[str, str]],
    feature_mode: str,
    exclude_self: bool = False,
) -> float | None:
    distances = []
    row_features = encode_row_features(row, feature_mode=feature_mode)
    for candidate in candidates:
        if exclude_self and candidate is row:
            continue
        candidate_features = encode_row_features(candidate, feature_mode=feature_mode)
        distances.append(
            math.sqrt(
                sum((left - right) ** 2 for left, right in zip(row_features, candidate_features))
            )
        )
    return min(distances) if distances else None


def average(values) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 6)


def is_less(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left < right


if __name__ == "__main__":
    main()
