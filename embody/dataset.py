from __future__ import annotations

import csv
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from .experiments import ExperimentCase
from .grid_world import run_episode
from .policy import ObstacleAwareGoalPolicy


DATASET_TYPE = "heuristic_demonstration"
TEACHER_POLICY = "obstacle_aware_goal"
TEACHER_POLICY_VERSION = "v0.4.2"
TEACHER_POLICY_STATUS = "current_best_heuristic_not_optimal"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_RANDOM_SEED = 42

FIELDNAMES = [
    "dataset_type",
    "policy_name",
    "policy_version",
    "case_name",
    "episode_id",
    "step",
    "x",
    "y",
    "direction",
    "next_x",
    "next_y",
    "next_direction",
    "front_blocked",
    "left_blocked",
    "right_blocked",
    "true_front_blocked",
    "true_left_blocked",
    "true_right_blocked",
    "distance_to_goal",
    "dx_to_goal",
    "dy_to_goal",
    "action",
    "reward",
    "done",
    "episode_success",
    "is_demonstration_accepted",
]


def generate_heuristic_demonstration_dataset(
    cases: list[ExperimentCase],
    output_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    used_cases: list[str] = []
    skipped_cases: list[dict[str, str]] = []
    successful_episodes = 0
    failed_episodes = 0
    accepted_episodes = 0
    rejected_episodes = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for index, case in enumerate(cases):
            episode_id = f"v0_5_{case.name}_{index}"
            if not case.expected_reachable:
                skipped_cases.append({"case_name": case.name, "reason": "expected_unreachable"})
                rejected_episodes += 1
                continue

            result = run_episode(
                world=case.build_world(),
                start=case.start,
                policy=ObstacleAwareGoalPolicy(),
                output_dir=tmp_root / case.name,
                run_id=episode_id,
                max_steps=case.max_steps,
            )

            if not result.reached_goal:
                skipped_cases.append({"case_name": case.name, "reason": "episode_failed"})
                failed_episodes += 1
                rejected_episodes += 1
                continue

            successful_episodes += 1
            accepted_episodes += 1
            used_cases.append(case.name)
            rows.extend(_records_to_dataset_rows(case.name, episode_id, result.records))

    train_rows, test_rows = split_rows(rows, train_ratio, random_seed)
    expert_path = output_dir / "expert_demonstrations.csv"
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    summary_path = output_dir / "dataset_summary.json"

    write_rows(expert_path, rows)
    write_rows(train_path, train_rows)
    write_rows(test_path, test_rows)

    summary = {
        "dataset_type": DATASET_TYPE,
        "teacher_policy": TEACHER_POLICY,
        "teacher_policy_version": TEACHER_POLICY_VERSION,
        "teacher_policy_status": TEACHER_POLICY_STATUS,
        "total_cases": len(cases),
        "used_cases": used_cases,
        "skipped_cases": skipped_cases,
        "total_episodes": len(cases),
        "successful_episodes": successful_episodes,
        "failed_episodes": failed_episodes,
        "accepted_episodes": accepted_episodes,
        "rejected_episodes": rejected_episodes,
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_ratio": train_ratio,
        "random_seed": random_seed,
        "output_files": {
            "expert_demonstrations": str(expert_path),
            "train": str(train_path),
            "test": str(test_path),
            "summary": str(summary_path),
        },
        "limitations": [
            "The teacher policy is a heuristic policy, not an optimal expert.",
            "The dataset only contains successful episodes.",
            "The dataset quality is limited by current benchmark coverage.",
            "This dataset is intended to validate the behavior cloning pipeline, not to represent final expert behavior.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def split_rows(
    rows: list[dict[str, Any]],
    train_ratio: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(random_seed).shuffle(shuffled)
    if not shuffled:
        return [], []

    split_index = int(len(shuffled) * train_ratio)
    split_index = max(1, split_index)
    if len(shuffled) > 1:
        split_index = min(len(shuffled) - 1, split_index)
    return shuffled[:split_index], shuffled[split_index:]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _records_to_dataset_rows(case_name: str, episode_id: str, records) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "dataset_type": DATASET_TYPE,
                "policy_name": TEACHER_POLICY,
                "policy_version": TEACHER_POLICY_VERSION,
                "case_name": case_name,
                "episode_id": episode_id,
                "step": record.step,
                "x": record.pre_x,
                "y": record.pre_y,
                "direction": record.pre_direction.value if record.pre_direction else "",
                "next_x": record.x,
                "next_y": record.y,
                "next_direction": record.direction.value,
                "front_blocked": int(record.pre_front_blocked),
                "left_blocked": int(record.pre_left_blocked),
                "right_blocked": int(record.pre_right_blocked),
                "true_front_blocked": int(record.pre_true_front_blocked),
                "true_left_blocked": int(record.pre_true_left_blocked),
                "true_right_blocked": int(record.pre_true_right_blocked),
                "distance_to_goal": record.pre_distance_to_goal,
                "dx_to_goal": record.pre_dx_to_goal,
                "dy_to_goal": record.pre_dy_to_goal,
                "action": record.action.value,
                "reward": f"{record.reward:.2f}",
                "done": int(record.done),
                "episode_success": 1,
                "is_demonstration_accepted": "true",
            }
        )
    return rows
