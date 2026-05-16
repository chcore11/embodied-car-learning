from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import TEACHER_POLICY, TEACHER_POLICY_STATUS, TEACHER_POLICY_VERSION
from .experiments import ExperimentCase, build_case_summary, build_overall_summary, write_json
from .grid_world import Action, Direction, Observation, run_episode


POLICY_NAME = "behavior_cloning_knn"
MODEL_TYPE = "knn_behavior_cloning"
DATASET_SOURCE = "data/datasets/v0_5"
ACTIONS = ["forward", "turn_left", "turn_right"]
FEATURE_COLUMNS = [
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
    ) -> None:
        self.samples = samples or []
        self.feature_columns = feature_columns or list(FEATURE_COLUMNS)
        self.k = k

    def fit(self, rows: list[dict[str, str]]) -> None:
        self.samples = [
            {
                "features": encode_row_features(row),
                "action": row["action"],
            }
            for row in rows
        ]

    def reset(self) -> None:
        pass

    def predict_action(self, observation: Observation | dict[str, str]) -> Action:
        if not self.samples:
            raise ValueError("KNNBehaviorCloningPolicy has no training samples")

        features = encode_observation_features(observation)
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

    def select_action(self, observation: Observation) -> Action:
        return self.predict_action(observation)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "policy_name": self.name,
            "model_type": MODEL_TYPE,
            "k": self.k,
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
        )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def train_knn_policy(train_csv: Path, k: int = 3) -> KNNBehaviorCloningPolicy:
    rows = load_rows(train_csv)
    policy = KNNBehaviorCloningPolicy(k=k)
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
    return {
        "total_test_rows": total,
        "correct_predictions": correct,
        "action_accuracy": round(correct / total, 4) if total else 0.0,
        "confusion_matrix": confusion,
        "per_action_accuracy": {
            action: (
                round(counts["correct"] / counts["total"], 4)
                if counts["total"]
                else None
            )
            for action, counts in per_action_counts.items()
        },
    }


def evaluate_rollouts(
    policy: KNNBehaviorCloningPolicy,
    cases: list[ExperimentCase],
    results_dir: Path,
) -> dict[str, Any]:
    policy_dir = results_dir / POLICY_NAME
    summaries = []
    for case in cases:
        case_dir = policy_dir / case.name
        result = run_episode(
            world=case.build_world(),
            start=case.start,
            policy=policy,
            output_dir=case_dir,
            run_id=f"{POLICY_NAME}_{case.name}",
            max_steps=case.max_steps,
            csv_name="run_log.csv",
            trajectory_name="trajectory.png",
        )
        summary = build_case_summary(case, result)
        summary["policy_name"] = POLICY_NAME
        write_json(case_dir / "summary.json", summary)
        summaries.append(summary)

    overall = build_overall_summary(summaries)
    rollout_summary = {
        "policy_name": POLICY_NAME,
        "total_runs": overall["total_runs"],
        "success_runs": overall["success_runs"],
        "failed_runs": overall["failed_runs"],
        "success_rate": overall["success_rate"],
        "average_steps_all_runs": overall["average_steps"],
        "average_reward": overall["average_reward"],
        "total_collisions": overall["total_collisions"],
        "total_action_failures": overall["total_action_failures"],
    }
    write_json(results_dir / "rollout_summary.json", rollout_summary)
    return rollout_summary


def build_behavior_cloning_summary(
    offline_metrics: dict[str, Any],
    rollout_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_source": DATASET_SOURCE,
        "teacher_policy": TEACHER_POLICY,
        "teacher_policy_version": TEACHER_POLICY_VERSION,
        "teacher_policy_status": TEACHER_POLICY_STATUS,
        "model_type": MODEL_TYPE,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "offline_metrics": offline_metrics,
        "rollout_metrics": rollout_metrics,
        "limitations": [
            "The model imitates a heuristic teacher policy, not an optimal expert.",
            "Offline action accuracy does not guarantee online navigation success.",
            "The dataset is small and limited to current benchmark cases.",
            "true_* and post-action fields are excluded from model inputs to avoid leakage.",
        ],
    }


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


def encode_row_features(row: dict[str, str]) -> list[float]:
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


def encode_observation_features(observation: Observation | dict[str, str]) -> list[float]:
    if isinstance(observation, dict):
        return encode_row_features(observation)
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
