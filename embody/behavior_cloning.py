from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import TEACHER_POLICY, TEACHER_POLICY_STATUS, TEACHER_POLICY_VERSION
from .experiments import ExperimentCase, build_case_summary, build_overall_summary, write_json
from .grid_world import Action, Direction, Observation, TURN_LEFT, TURN_RIGHT, run_episode


POLICY_NAME = "behavior_cloning_knn"
MODEL_TYPE = "knn_behavior_cloning"
DATASET_SOURCE = "data/datasets/v0_5"
ACTIONS = ["forward", "turn_left", "turn_right"]
BASE_FEATURE_COLUMNS = [
    "x",
    "y",
    "direction",
    "front_blocked",
    "left_blocked",
    "right_blocked",
    "distance_to_goal",
    "dx_to_goal",
    "dy_to_goal",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
DERIVED_FEATURE_COLUMNS = [
    "x",
    "y",
    "direction_north",
    "direction_east",
    "direction_south",
    "direction_west",
    "front_blocked",
    "left_blocked",
    "right_blocked",
    "distance_to_goal",
    "dx_to_goal",
    "dy_to_goal",
    "goal_relative_front",
    "goal_relative_left",
    "goal_relative_right",
    "goal_relative_behind",
    "goal_relative_aligned",
    "goal_on_left",
    "goal_on_right",
    "goal_ahead",
    "goal_behind",
    "front_blocked_and_left_free",
    "front_blocked_and_right_free",
    "front_blocked_and_both_sides_free",
    "front_blocked_and_no_side_free",
    "would_forward_reduce_distance",
    "would_left_turn_face_goal",
    "would_right_turn_face_goal",
]
EXCLUDED_COLUMNS = [
    "true_front_blocked",
    "true_left_blocked",
    "true_right_blocked",
    "next_x",
    "next_y",
    "next_direction",
    "reward",
    "done",
    "episode_success",
    "is_demonstration_accepted",
    "case_name",
    "episode_id",
    "step",
]


class KNNBehaviorCloningPolicy:
    name = POLICY_NAME

    def __init__(
        self,
        samples: list[dict[str, Any]] | None = None,
        feature_columns: list[str] | None = None,
        k: int = 3,
        per_class_k: int = 3,
        voting_mode: str = "majority",
        feature_mode: str = "base",
        name: str | None = None,
    ) -> None:
        self.samples = samples or []
        self.feature_mode = feature_mode
        if self.feature_mode not in {"base", "derived"}:
            raise ValueError("feature_mode must be base or derived")
        self.feature_columns = feature_columns or feature_columns_for_mode(feature_mode)
        self.k = k
        self.per_class_k = per_class_k
        self.voting_mode = voting_mode
        self.name = name or POLICY_NAME
        if self.voting_mode not in {"majority", "class_balanced"}:
            raise ValueError("voting_mode must be majority or class_balanced")

    def fit(self, rows: list[dict[str, str]]) -> None:
        self.samples = [
            {
                "features": encode_row_features(row, feature_mode=self.feature_mode),
                "action": row["action"],
            }
            for row in rows
        ]

    def reset(self) -> None:
        pass

    def predict_action(self, observation: Observation | dict[str, str]) -> Action:
        if not self.samples:
            raise ValueError("KNNBehaviorCloningPolicy has no training samples")

        features = encode_observation_features(observation, feature_mode=self.feature_mode)
        if self.voting_mode == "class_balanced":
            return self._predict_class_balanced(features)

        return self._predict_majority(features)

    def _predict_majority(self, features: list[float]) -> Action:
        neighbors = sorted(
            (
                (_euclidean_distance(features, sample["features"]), index, sample["action"])
                for index, sample in enumerate(self.samples)
            ),
            key=lambda item: (item[0], item[1]),
        )[: self.k]

        votes = Counter(action for _, _, action in neighbors)
        best_count = max(votes.values())
        tied_actions = [action for action, count in votes.items() if count == best_count]
        if len(tied_actions) == 1:
            return Action(tied_actions[0])

        best_by_distance = min(
            tied_actions,
            key=lambda action: (
                min(distance for distance, _, neighbor_action in neighbors if neighbor_action == action),
                ACTIONS.index(action),
            ),
        )
        return Action(best_by_distance)

    def _predict_class_balanced(self, features: list[float]) -> Action:
        scores = self.class_distance_scores(features)
        candidates = [
            (score["average_distance"], ACTIONS.index(action), action)
            for action, score in scores.items()
            if score["average_distance"] is not None
        ]
        if not candidates:
            return self._predict_majority(features)
        return Action(min(candidates)[2])

    def class_distance_scores(
        self,
        features: list[float],
    ) -> dict[str, dict[str, float | int | None]]:
        scores = {}
        for action in ACTIONS:
            distances = sorted(
                _euclidean_distance(features, sample["features"])
                for sample in self.samples
                if sample["action"] == action
            )
            nearest = distances[: self.per_class_k]
            scores[action] = {
                "sample_count": len(distances),
                "nearest_distance": round(nearest[0], 6) if nearest else None,
                "average_distance": (
                    round(sum(nearest) / len(nearest), 6) if nearest else None
                ),
            }
        return scores

    def select_action(self, observation: Observation) -> Action:
        return self.predict_action(observation)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "policy_name": self.name,
            "model_type": MODEL_TYPE,
            "k": self.k,
            "per_class_k": self.per_class_k,
            "voting_mode": self.voting_mode,
            "feature_mode": self.feature_mode,
            "feature_columns": self.feature_columns,
            "actions": ACTIONS,
            "samples": self.samples,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KNNBehaviorCloningPolicy":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            samples=list(data["samples"]),
            feature_columns=list(data["feature_columns"]),
            k=int(data["k"]),
            per_class_k=int(data.get("per_class_k", 3)),
            voting_mode=str(data.get("voting_mode", "majority")),
            feature_mode=str(data.get("feature_mode", "base")),
            name=str(data.get("policy_name", POLICY_NAME)),
        )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def train_knn_policy(
    train_csv: Path,
    k: int = 3,
    per_class_k: int = 3,
    voting_mode: str = "majority",
    feature_mode: str = "base",
    name: str | None = None,
) -> KNNBehaviorCloningPolicy:
    rows = load_rows(train_csv)
    policy = KNNBehaviorCloningPolicy(
        k=k,
        per_class_k=per_class_k,
        voting_mode=voting_mode,
        feature_mode=feature_mode,
        name=name,
    )
    policy.fit(rows)
    return policy


def evaluate_offline(
    policy: KNNBehaviorCloningPolicy,
    test_rows: list[dict[str, str]],
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    predictions = []
    correct = 0
    confusion = _empty_confusion_matrix()
    per_action_counts = {action: {"total": 0, "correct": 0} for action in ACTIONS}

    for row in test_rows:
        actual = row["action"]
        predicted = policy.predict_action(row).value
        is_correct = predicted == actual
        correct += int(is_correct)
        confusion[actual][predicted] += 1
        per_action_counts[actual]["total"] += 1
        per_action_counts[actual]["correct"] += int(is_correct)
        predictions.append(
            {
                "case_name": row.get("case_name", ""),
                "episode_id": row.get("episode_id", ""),
                "step": row.get("step", ""),
                "actual_action": actual,
                "predicted_action": predicted,
                "correct": int(is_correct),
            }
        )

    if predictions_path is not None:
        write_predictions(predictions_path, predictions)

    total = len(test_rows)
    per_action_accuracy = {
        action: (
            round(counts["correct"] / counts["total"], 4)
            if counts["total"]
            else None
        )
        for action, counts in per_action_counts.items()
    }
    return {
        "total_test_rows": total,
        "correct_predictions": correct,
        "action_accuracy": round(correct / total, 4) if total else 0.0,
        "test_action_distribution": action_distribution(test_rows),
        "predicted_action_distribution": _prediction_distribution(predictions),
        "forward_only_baseline_accuracy": _forward_only_accuracy(test_rows),
        "macro_action_accuracy": _macro_accuracy(per_action_accuracy, include_missing=True),
        "balanced_action_accuracy": _macro_accuracy(per_action_accuracy, include_missing=False),
        "confusion_matrix": confusion,
        "per_action_accuracy": per_action_accuracy,
    }


def evaluate_rollouts(
    policy: KNNBehaviorCloningPolicy,
    cases: list[ExperimentCase],
    results_dir: Path,
    summary_name: str = "rollout_summary.json",
) -> dict[str, Any]:
    policy_name = policy.name
    policy_dir = results_dir / policy_name
    summaries = []
    case_analyses = []
    for case in cases:
        case_dir = policy_dir / case.name
        result = run_episode(
            world=case.build_world(),
            start=case.start,
            policy=policy,
            output_dir=case_dir,
            run_id=f"{policy_name}_{case.name}",
            max_steps=case.max_steps,
            csv_name="run_log.csv",
            trajectory_name="trajectory.png",
        )
        summary = build_case_summary(case, result)
        summary["policy_name"] = policy_name
        write_json(case_dir / "summary.json", summary)
        summaries.append(summary)
        case_analyses.append(analyze_rollout_case(case.name, summary, result.records))

    overall = build_overall_summary(summaries)
    rollout_summary = {
        "policy_name": policy_name,
        "total_runs": overall["total_runs"],
        "success_runs": overall["success_runs"],
        "failed_runs": overall["failed_runs"],
        "success_rate": overall["success_rate"],
        "average_steps_all_runs": overall["average_steps"],
        "average_reward": overall["average_reward"],
        "total_collisions": overall["total_collisions"],
        "total_action_failures": overall["total_action_failures"],
        "case_analyses": case_analyses,
    }
    write_json(
        results_dir / summary_name,
        {key: value for key, value in rollout_summary.items() if key != "case_analyses"},
    )
    return rollout_summary


def build_behavior_cloning_summary(
    train_rows: list[dict[str, str]],
    offline_metrics: dict[str, Any],
    rollout_metrics: dict[str, Any],
) -> dict[str, Any]:
    rollout_metrics_for_summary = {
        key: value for key, value in rollout_metrics.items() if key != "case_analyses"
    }
    return {
        "dataset_source": DATASET_SOURCE,
        "teacher_policy": TEACHER_POLICY,
        "teacher_policy_version": TEACHER_POLICY_VERSION,
        "teacher_policy_status": TEACHER_POLICY_STATUS,
        "model_type": MODEL_TYPE,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "train_action_distribution": action_distribution(train_rows),
        "test_action_distribution": offline_metrics["test_action_distribution"],
        "predicted_action_distribution": offline_metrics["predicted_action_distribution"],
        "forward_only_baseline_accuracy": offline_metrics["forward_only_baseline_accuracy"],
        "macro_action_accuracy": offline_metrics["macro_action_accuracy"],
        "balanced_action_accuracy": offline_metrics["balanced_action_accuracy"],
        "offline_metrics": offline_metrics,
        "rollout_metrics": rollout_metrics_for_summary,
        "limitations": [
            "The model imitates a heuristic teacher policy, not an optimal expert.",
            "Offline action accuracy does not guarantee online navigation success.",
            "The dataset is small and limited to current benchmark cases.",
            "true_* and post-action fields are excluded from model inputs to avoid leakage.",
        ],
    }


def build_failure_analysis(
    train_rows: list[dict[str, str]],
    offline_metrics: dict[str, Any],
    rollout_metrics: dict[str, Any],
) -> dict[str, Any]:
    case_analyses = rollout_metrics.get("case_analyses", [])
    collapsed_to_forward = _collapsed_to_forward(
        offline_metrics["predicted_action_distribution"]
    )
    no_obstacle = next(
        (item for item in case_analyses if item["case_name"] == "no_obstacle_8x8"),
        None,
    )
    return {
        "action_distribution_summary": {
            "train_action_distribution": action_distribution(train_rows),
            "test_action_distribution": offline_metrics["test_action_distribution"],
            "predicted_action_distribution": offline_metrics["predicted_action_distribution"],
            "forward_only_baseline_accuracy": offline_metrics["forward_only_baseline_accuracy"],
            "macro_action_accuracy": offline_metrics["macro_action_accuracy"],
            "balanced_action_accuracy": offline_metrics["balanced_action_accuracy"],
        },
        "confusion_matrix": offline_metrics["confusion_matrix"],
        "per_action_accuracy": offline_metrics["per_action_accuracy"],
        "rollout_failure_cases": [
            item for item in case_analyses if not item["reached_goal"]
        ],
        "per_case_first_collision_step": {
            item["case_name"]: item["first_collision_step"] for item in case_analyses
        },
        "per_case_first_repeated_action_pattern": {
            item["case_name"]: item["first_repeated_action_pattern"]
            for item in case_analyses
        },
        "per_case_dominant_predicted_action": {
            item["case_name"]: item["most_common_action"] for item in case_analyses
        },
        "whether_policy_collapsed_to_forward": collapsed_to_forward,
        "no_obstacle_8x8_boundary_forward_check": (
            no_obstacle["boundary_forward_check"] if no_obstacle else None
        ),
    }


def analyze_rollout_case(
    case_name: str,
    summary: dict[str, Any],
    records,
) -> dict[str, Any]:
    action_counts = {action: 0 for action in ACTIONS}
    positions = []
    seen_positions = set()
    repeated_position_count = 0
    first_collision_step = None
    first_10_actions = []

    for record in records:
        action = record.action.value
        action_counts[action] += 1
        if len(first_10_actions) < 10:
            first_10_actions.append(action)
        if record.collision and first_collision_step is None:
            first_collision_step = record.step

        position = (record.x, record.y)
        positions.append(position)
        if position in seen_positions:
            repeated_position_count += 1
        else:
            seen_positions.add(position)

    most_common_action = _most_common_action(action_counts)
    return {
        "case_name": case_name,
        "reached_goal": summary["reached_goal"],
        "failure_reason": summary["failure_reason"],
        "collision_count": summary["collision_count"],
        "first_collision_step": first_collision_step,
        "most_common_action": most_common_action,
        "action_counts": action_counts,
        "repeated_position_count": repeated_position_count,
        "first_10_actions": first_10_actions,
        "first_repeated_action_pattern": _first_repeated_action_pattern(records),
        "boundary_forward_check": _boundary_forward_check(records),
    }


def action_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {action: 0 for action in ACTIONS}
    for row in rows:
        action = row["action"]
        if action in counts:
            counts[action] += 1
    return counts


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_name",
        "episode_id",
        "step",
        "actual_action",
        "predicted_action",
        "correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def feature_columns_for_mode(feature_mode: str) -> list[str]:
    if feature_mode == "base":
        return list(BASE_FEATURE_COLUMNS)
    if feature_mode == "derived":
        return list(DERIVED_FEATURE_COLUMNS)
    raise ValueError("feature_mode must be base or derived")


def encode_row_features(row: dict[str, str], feature_mode: str = "base") -> list[float]:
    if feature_mode == "derived":
        return encode_derived_features(
            x=int(row["x"]),
            y=int(row["y"]),
            direction=Direction(row["direction"]),
            front_blocked=_as_bool(row["front_blocked"]),
            left_blocked=_as_bool(row["left_blocked"]),
            right_blocked=_as_bool(row["right_blocked"]),
            distance_to_goal=int(row["distance_to_goal"]),
            dx_to_goal=int(row["dx_to_goal"]),
            dy_to_goal=int(row["dy_to_goal"]),
        )
    return [
        float(row["x"]),
        float(row["y"]),
        _encode_direction(row["direction"]),
        float(row["front_blocked"]),
        float(row["left_blocked"]),
        float(row["right_blocked"]),
        float(row["distance_to_goal"]),
        float(row["dx_to_goal"]),
        float(row["dy_to_goal"]),
    ]


def encode_observation_features(
    observation: Observation | dict[str, str],
    feature_mode: str = "base",
) -> list[float]:
    if isinstance(observation, dict):
        return encode_row_features(observation, feature_mode=feature_mode)
    if feature_mode == "derived":
        return encode_derived_features(
            x=observation.x,
            y=observation.y,
            direction=observation.direction,
            front_blocked=observation.front_blocked,
            left_blocked=observation.left_blocked,
            right_blocked=observation.right_blocked,
            distance_to_goal=observation.distance_to_goal,
            dx_to_goal=observation.dx_to_goal,
            dy_to_goal=observation.dy_to_goal,
        )
    return [
        float(observation.x),
        float(observation.y),
        _encode_direction(observation.direction.value),
        float(observation.front_blocked),
        float(observation.left_blocked),
        float(observation.right_blocked),
        float(observation.distance_to_goal),
        float(observation.dx_to_goal),
        float(observation.dy_to_goal),
    ]


def encode_derived_features(
    x: int,
    y: int,
    direction: Direction,
    front_blocked: bool,
    left_blocked: bool,
    right_blocked: bool,
    distance_to_goal: int,
    dx_to_goal: int,
    dy_to_goal: int,
) -> list[float]:
    relative = goal_relative_direction(direction, dx_to_goal, dy_to_goal)
    left_direction = TURN_LEFT[direction]
    right_direction = TURN_RIGHT[direction]
    left_free = not left_blocked
    right_free = not right_blocked
    return [
        float(x),
        float(y),
        *one_hot(direction.value, [item.value for item in Direction]),
        float(front_blocked),
        float(left_blocked),
        float(right_blocked),
        float(distance_to_goal),
        float(dx_to_goal),
        float(dy_to_goal),
        *one_hot(relative, ["front", "left", "right", "behind", "aligned"]),
        float(relative == "left"),
        float(relative == "right"),
        float(relative == "front"),
        float(relative == "behind"),
        float(front_blocked and left_free),
        float(front_blocked and right_free),
        float(front_blocked and left_free and right_free),
        float(front_blocked and not left_free and not right_free),
        float(_direction_progress(direction, dx_to_goal, dy_to_goal) > 0),
        float(_direction_progress(left_direction, dx_to_goal, dy_to_goal) > 0),
        float(_direction_progress(right_direction, dx_to_goal, dy_to_goal) > 0),
    ]


def goal_relative_direction(
    direction: Direction,
    dx_to_goal: int,
    dy_to_goal: int,
) -> str:
    if dx_to_goal == 0 and dy_to_goal == 0:
        return "aligned"
    scores = {
        "front": _direction_progress(direction, dx_to_goal, dy_to_goal),
        "left": _direction_progress(TURN_LEFT[direction], dx_to_goal, dy_to_goal),
        "right": _direction_progress(TURN_RIGHT[direction], dx_to_goal, dy_to_goal),
        "behind": -_direction_progress(direction, dx_to_goal, dy_to_goal),
    }
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], ["front", "left", "right", "behind"].index(item[0])),
    )[0][0]


def one_hot(value: str, choices: list[str]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _direction_progress(
    direction: Direction,
    dx_to_goal: int,
    dy_to_goal: int,
) -> int:
    if direction == Direction.EAST:
        return dx_to_goal
    if direction == Direction.WEST:
        return -dx_to_goal
    if direction == Direction.SOUTH:
        return dy_to_goal
    if direction == Direction.NORTH:
        return -dy_to_goal
    return 0


def _as_bool(value: str) -> bool:
    return value in {"1", "true", "True"}


def _encode_direction(direction: str) -> float:
    return {
        Direction.NORTH.value: 0.0,
        Direction.EAST.value: 1.0,
        Direction.SOUTH.value: 2.0,
        Direction.WEST.value: 3.0,
    }[direction]


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {
        actual: {predicted: 0 for predicted in ACTIONS}
        for actual in ACTIONS
    }


def _prediction_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {action: 0 for action in ACTIONS}
    for row in rows:
        predicted = row["predicted_action"]
        if predicted in counts:
            counts[predicted] += 1
    return counts


def _forward_only_accuracy(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    forward_count = sum(1 for row in rows if row["action"] == Action.FORWARD.value)
    return round(forward_count / len(rows), 4)


def _macro_accuracy(
    per_action_accuracy: dict[str, float | None],
    include_missing: bool,
) -> float:
    values = []
    for action in ACTIONS:
        value = per_action_accuracy[action]
        if value is None and include_missing:
            values.append(0.0)
        elif value is not None:
            values.append(value)
    return round(sum(values) / len(values), 4) if values else 0.0


def _collapsed_to_forward(predicted_distribution: dict[str, int]) -> bool:
    total = sum(predicted_distribution.values())
    if total == 0:
        return False
    return predicted_distribution.get(Action.FORWARD.value, 0) / total >= 0.95


def _most_common_action(action_counts: dict[str, int]) -> str | None:
    if not action_counts or not any(action_counts.values()):
        return None
    return sorted(action_counts.items(), key=lambda item: (-item[1], ACTIONS.index(item[0])))[0][0]


def _first_repeated_action_pattern(records) -> dict[str, Any] | None:
    previous_action = None
    start_step = None
    run_length = 0
    for record in records:
        action = record.action.value
        if action == previous_action:
            run_length += 1
        else:
            previous_action = action
            start_step = record.step
            run_length = 1
        if run_length >= 5:
            return {
                "action": action,
                "start_step": start_step,
                "length": run_length,
            }
    return None


def _boundary_forward_check(records) -> dict[str, Any]:
    collision_steps = [
        record.step
        for record in records
        if record.action == Action.FORWARD and record.collision
    ]
    consecutive_after_first = 0
    if collision_steps:
        expected = collision_steps[0]
        for step in collision_steps:
            if step == expected:
                consecutive_after_first += 1
                expected += 1
            elif step > expected:
                break
    return {
        "forward_collision_steps": collision_steps[:20],
        "first_forward_collision_step": collision_steps[0] if collision_steps else None,
        "consecutive_forward_collisions_after_first": consecutive_after_first,
        "kept_predicting_forward_after_boundary": consecutive_after_first >= 3,
    }
