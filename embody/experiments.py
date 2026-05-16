from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grid_world import Direction, GridWorld, RobotState, RunResult, run_episode
from .policy import GreedyGridPolicy


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    width: int
    height: int
    start: RobotState
    goal: tuple[int, int]
    obstacles: set[tuple[int, int]]
    max_steps: int
    expected_reachable: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentCase":
        start = data["start"]
        return cls(
            name=str(data["name"]),
            width=int(data["width"]),
            height=int(data["height"]),
            start=RobotState(
                x=int(start["x"]),
                y=int(start["y"]),
                direction=Direction(str(start["direction"])),
            ),
            goal=_cell(data["goal"]),
            obstacles={_cell(item) for item in data.get("obstacles", [])},
            max_steps=int(data["max_steps"]),
            expected_reachable=bool(data["expected_reachable"]),
        )

    def build_world(self) -> GridWorld:
        return GridWorld(
            width=self.width,
            height=self.height,
            obstacles=self.obstacles,
            goal=self.goal,
        )


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "name": "no_obstacle_8x8",
        "width": 8,
        "height": 8,
        "start": {"x": 0, "y": 0, "direction": "east"},
        "goal": [7, 7],
        "obstacles": [],
        "max_steps": 32,
        "expected_reachable": True,
    },
    {
        "name": "obstacle_reachable_8x8",
        "width": 8,
        "height": 8,
        "start": {"x": 0, "y": 0, "direction": "east"},
        "goal": [7, 7],
        "obstacles": [[3, 0], [3, 1], [3, 2], [3, 4], [3, 5], [3, 6]],
        "max_steps": 64,
        "expected_reachable": True,
    },
    {
        "name": "blocked_unreachable_6x6",
        "width": 6,
        "height": 6,
        "start": {"x": 0, "y": 0, "direction": "east"},
        "goal": [5, 5],
        "obstacles": [[1, 0], [0, 1], [4, 5], [5, 4]],
        "max_steps": 24,
        "expected_reachable": False,
    },
]


def default_cases() -> list[ExperimentCase]:
    return [ExperimentCase.from_dict(data) for data in DEFAULT_CASES]


def run_experiments(
    cases: list[ExperimentCase],
    results_dir: Path,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for case in cases:
        case_dir = results_dir / case.name
        summary = run_case(case, case_dir)
        summaries.append(summary)

    overall = build_overall_summary(summaries)
    write_json(results_dir / "overall_summary.json", overall)
    return overall


def run_case(case: ExperimentCase, case_dir: Path) -> dict[str, Any]:
    result = run_episode(
        world=case.build_world(),
        start=case.start,
        policy=GreedyGridPolicy(),
        output_dir=case_dir,
        run_id=case.name,
        max_steps=case.max_steps,
        csv_name="run_log.csv",
        trajectory_name="trajectory.png",
    )
    summary = build_case_summary(case, result)
    write_json(case_dir / "summary.json", summary)
    return summary


def build_case_summary(case: ExperimentCase, result: RunResult) -> dict[str, Any]:
    failure_reason = None
    if not result.reached_goal:
        failure_reason = _failure_reason(result, case.max_steps)
    result_type = classify_result(case.expected_reachable, result.reached_goal)

    return {
        "case_name": case.name,
        "expected_reachable": case.expected_reachable,
        "result_type": result_type,
        "steps": len(result.records),
        "reached_goal": result.reached_goal,
        "total_reward": round(result.total_reward, 2),
        "csv_path": str(result.csv_path),
        "trajectory_path": str(result.trajectory_path),
        "failure_reason": failure_reason,
    }


def build_overall_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total_runs = len(summaries)
    success_runs = sum(1 for item in summaries if item["reached_goal"])
    failed_runs = total_runs - success_runs
    reachable_runs = sum(1 for item in summaries if item["expected_reachable"])
    reachable_success_runs = sum(
        1
        for item in summaries
        if item["expected_reachable"] and item["reached_goal"]
    )
    reachable_failed_runs = reachable_runs - reachable_success_runs
    expected_unreachable_runs = sum(
        1 for item in summaries if item["result_type"] == "expected_unreachable"
    )
    unexpected_success_runs = sum(
        1 for item in summaries if item["result_type"] == "unexpected_success"
    )
    policy_failed_runs = sum(
        1 for item in summaries if item["result_type"] == "policy_failed"
    )
    average_steps = _average(float(item["steps"]) for item in summaries)
    average_reward = _average(float(item["total_reward"]) for item in summaries)

    return {
        "total_runs": total_runs,
        "success_runs": success_runs,
        "failed_runs": failed_runs,
        "success_rate": round(success_runs / total_runs, 4) if total_runs else 0.0,
        "reachable_runs": reachable_runs,
        "reachable_success_runs": reachable_success_runs,
        "reachable_failed_runs": reachable_failed_runs,
        "reachable_success_rate": (
            round(reachable_success_runs / reachable_runs, 4)
            if reachable_runs
            else 0.0
        ),
        "expected_unreachable_runs": expected_unreachable_runs,
        "unexpected_success_runs": unexpected_success_runs,
        "policy_failed_runs": policy_failed_runs,
        "average_steps": round(average_steps, 2),
        "average_reward": round(average_reward, 2),
    }


def classify_result(expected_reachable: bool, reached_goal: bool) -> str:
    if expected_reachable and reached_goal:
        return "success"
    if expected_reachable and not reached_goal:
        return "policy_failed"
    if not expected_reachable and not reached_goal:
        return "expected_unreachable"
    return "unexpected_success"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _cell(value: Any) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _failure_reason(result: RunResult, max_steps: int) -> str:
    if len(result.records) >= max_steps:
        return "max_steps"
    if result.records and result.records[-1].event == "collision":
        return "collision_loop"
    return "not_reached_goal"


def _average(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)
