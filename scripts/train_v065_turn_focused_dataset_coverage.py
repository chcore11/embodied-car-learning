from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.experiments import ExperimentCase, build_case_summary, write_json
from embody.grid_world import Direction, RobotState, run_episode
from embody.policy import ObstacleAwareGoalPolicy


ACTIONS = ["forward", "turn_left", "turn_right"]
RESULTS_DIR = Path("experiments/v0_6_5/results")
COVERAGE_GATE = {
    "min_turn_left": 20,
    "min_turn_right": 20,
    "min_front_blocked_states": 20,
    "max_forward_ratio": 0.75,
}


def turn_focused_cases() -> list[ExperimentCase]:
    cases = [
        ExperimentCase(
            name="left_turn_required",
            width=6,
            height=6,
            start=RobotState(2, 3, Direction.EAST),
            goal=(2, 0),
            obstacles={(3, 3), (3, 2), (3, 1)},
            max_steps=24,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="right_turn_required",
            width=6,
            height=6,
            start=RobotState(2, 2, Direction.EAST),
            goal=(2, 5),
            obstacles={(3, 2), (3, 3), (3, 4)},
            max_steps=24,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="front_blocked_left_open",
            width=7,
            height=7,
            start=RobotState(3, 3, Direction.EAST),
            goal=(3, 0),
            obstacles={(4, 3), (4, 2), (2, 3), (3, 4)},
            max_steps=30,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="front_blocked_right_open",
            width=7,
            height=7,
            start=RobotState(3, 3, Direction.EAST),
            goal=(3, 6),
            obstacles={(4, 3), (4, 4), (2, 3), (3, 2)},
            max_steps=30,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="corridor_left_turn",
            width=8,
            height=8,
            start=RobotState(1, 6, Direction.EAST),
            goal=(6, 1),
            obstacles={(2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (6, 6), (0, 5)},
            max_steps=50,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="corridor_right_turn",
            width=8,
            height=8,
            start=RobotState(1, 1, Direction.EAST),
            goal=(6, 6),
            obstacles={(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 1), (0, 2)},
            max_steps=50,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="consecutive_turns",
            width=8,
            height=8,
            start=RobotState(1, 4, Direction.EAST),
            goal=(6, 4),
            obstacles={(2, 4), (2, 3), (3, 3), (4, 3), (4, 4), (4, 5), (5, 5)},
            max_steps=60,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="left_turn_required_mirror",
            width=6,
            height=6,
            start=RobotState(3, 2, Direction.WEST),
            goal=(3, 5),
            obstacles={(2, 2), (2, 3), (2, 4)},
            max_steps=24,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="right_turn_required_mirror",
            width=6,
            height=6,
            start=RobotState(3, 3, Direction.WEST),
            goal=(3, 0),
            obstacles={(2, 3), (2, 2), (2, 1)},
            max_steps=24,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="front_blocked_left_open_south",
            width=7,
            height=7,
            start=RobotState(3, 3, Direction.SOUTH),
            goal=(6, 3),
            obstacles={(3, 4), (4, 4), (3, 2), (2, 3)},
            max_steps=30,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="front_blocked_right_open_north",
            width=7,
            height=7,
            start=RobotState(3, 3, Direction.NORTH),
            goal=(6, 3),
            obstacles={(3, 2), (4, 2), (3, 4), (2, 3)},
            max_steps=30,
            expected_reachable=True,
        ),
    ]
    cases.extend(generated_forced_turn_cases())
    return cases


def generated_forced_turn_cases() -> list[ExperimentCase]:
    cases = []
    for index in range(12):
        y = 2 + (index % 2)
        cases.append(
            ExperimentCase(
                name=f"generated_left_turn_blocked_{index + 1:02d}",
                width=6,
                height=6,
                start=RobotState(2, y, Direction.EAST),
                goal=(2, y - 1),
                obstacles={(3, y)},
                max_steps=6,
                expected_reachable=True,
            )
        )
    for index in range(12):
        y = 2 + (index % 2)
        cases.append(
            ExperimentCase(
                name=f"generated_right_turn_blocked_{index + 1:02d}",
                width=6,
                height=6,
                start=RobotState(2, y, Direction.EAST),
                goal=(2, y + 1),
                obstacles={(3, y)},
                max_steps=6,
                expected_reachable=True,
            )
        )
    return cases


def run_turn_focused_coverage(results_dir: Path = RESULTS_DIR) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    case_summaries = []
    all_records = []

    for case in turn_focused_cases():
        case_dir = results_dir / "rollouts" / case.name
        result = run_episode(
            world=case.build_world(),
            start=case.start,
            policy=ObstacleAwareGoalPolicy(),
            output_dir=case_dir,
            run_id=f"v0_6_5_{case.name}",
            max_steps=case.max_steps,
            csv_name="run_log.csv",
            trajectory_name="trajectory.png",
        )
        summary = build_case_summary(case, result)
        summary["action_distribution"] = action_distribution(result.records)
        summary["front_blocked_state_count"] = front_blocked_state_count(result.records)
        summary["front_blocked_action_distribution"] = front_blocked_action_distribution(result.records)
        write_json(case_dir / "summary.json", summary)
        case_summaries.append(summary)
        all_records.extend(result.records)

    audit = build_coverage_audit(case_summaries, all_records)
    action_by_case = {
        item["case_name"]: item["action_distribution"] for item in case_summaries
    }
    front_blocked = build_front_blocked_summary(all_records)
    final_summary = build_final_summary(case_summaries, audit, front_blocked)

    write_json(results_dir / "turn_focused_case_summary.json", {"cases": case_summaries})
    write_json(results_dir / "dataset_coverage_audit.json", audit)
    write_json(results_dir / "action_distribution_by_case.json", action_by_case)
    write_json(results_dir / "front_blocked_state_summary.json", front_blocked)
    write_json(results_dir / "v065_summary.json", final_summary)
    return final_summary


def build_coverage_audit(case_summaries: list[dict], records) -> dict:
    action_counts = action_distribution(records)
    total_actions = sum(action_counts.values())
    forward_ratio = round(action_counts["forward"] / total_actions, 4) if total_actions else 0.0
    front_blocked_count = front_blocked_state_count(records)
    gate_failures = []

    if action_counts["turn_left"] < COVERAGE_GATE["min_turn_left"]:
        gate_failures.append("turn_left_below_minimum")
    if action_counts["turn_right"] < COVERAGE_GATE["min_turn_right"]:
        gate_failures.append("turn_right_below_minimum")
    if front_blocked_count < COVERAGE_GATE["min_front_blocked_states"]:
        gate_failures.append("front_blocked_states_below_minimum")
    if action_counts["turn_left"] == 0:
        gate_failures.append("turn_left_is_zero")
    if action_counts["turn_right"] == 0:
        gate_failures.append("turn_right_is_zero")
    if forward_ratio > COVERAGE_GATE["max_forward_ratio"]:
        gate_failures.append("forward_ratio_too_high")

    return {
        "coverage_gate": COVERAGE_GATE,
        "total_cases": len(case_summaries),
        "total_steps": total_actions,
        "action_distribution": action_counts,
        "forward_ratio": forward_ratio,
        "front_blocked_state_count": front_blocked_count,
        "front_blocked_action_distribution": front_blocked_action_distribution(records),
        "gate_failures": gate_failures,
        "dataset_ready_for_behavior_cloning": not gate_failures,
        "dataset_not_ready_for_behavior_cloning": bool(gate_failures),
    }


def build_front_blocked_summary(records) -> dict:
    front_blocked_records = [record for record in records if is_pre_front_blocked(record)]
    return {
        "front_blocked_state_count": len(front_blocked_records),
        "front_blocked_action_distribution": action_distribution(front_blocked_records),
        "front_blocked_turn_count": sum(
            1 for record in front_blocked_records if record.action.value in {"turn_left", "turn_right"}
        ),
        "front_blocked_forward_count": sum(
            1 for record in front_blocked_records if record.action.value == "forward"
        ),
    }


def build_final_summary(case_summaries: list[dict], audit: dict, front_blocked: dict) -> dict:
    return {
        "version": "v0.6.5",
        "purpose": "turn-focused dataset coverage improvement audit",
        "policy_used": "obstacle_aware_goal",
        "total_cases": audit["total_cases"],
        "case_names": [item["case_name"] for item in case_summaries],
        "action_distribution": audit["action_distribution"],
        "forward_ratio": audit["forward_ratio"],
        "front_blocked_state_summary": front_blocked,
        "coverage_gate": audit["coverage_gate"],
        "gate_failures": audit["gate_failures"],
        "dataset_ready_for_behavior_cloning": audit["dataset_ready_for_behavior_cloning"],
        "dataset_not_ready_for_behavior_cloning": audit["dataset_not_ready_for_behavior_cloning"],
        "recommendation": (
            "ready_to_regenerate_behavior_cloning_dataset"
            if audit["dataset_ready_for_behavior_cloning"]
            else "continue_adding_turn_focused_data"
        ),
    }


def action_distribution(records) -> dict[str, int]:
    counts = {action: 0 for action in ACTIONS}
    for record in records:
        counts[record.action.value] += 1
    return counts


def front_blocked_action_distribution(records) -> dict[str, int]:
    return action_distribution([record for record in records if is_pre_front_blocked(record)])


def front_blocked_state_count(records) -> int:
    return sum(1 for record in records if is_pre_front_blocked(record))


def is_pre_front_blocked(record) -> bool:
    if record.pre_front_blocked is not None:
        return bool(record.pre_front_blocked)
    return bool(record.true_front_blocked)


def main() -> None:
    summary = run_turn_focused_coverage()
    print(f"results_dir: {RESULTS_DIR}")
    print(f"total_cases: {summary['total_cases']}")
    print(f"action_distribution: {summary['action_distribution']}")
    print(
        "front_blocked_state_count: "
        f"{summary['front_blocked_state_summary']['front_blocked_state_count']}"
    )
    print(
        "dataset_ready_for_behavior_cloning: "
        f"{summary['dataset_ready_for_behavior_cloning']}"
    )
    print(
        "dataset_not_ready_for_behavior_cloning: "
        f"{summary['dataset_not_ready_for_behavior_cloning']}"
    )
    print(f"gate_failures: {summary['gate_failures']}")


if __name__ == "__main__":
    main()
