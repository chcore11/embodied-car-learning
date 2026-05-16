from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.experiments import ExperimentCase, build_case_summary, build_overall_summary, default_cases, write_json
from embody.grid_world import Direction, RobotState, run_episode
from embody.policy import GreedyGridPolicy, ObstacleAwareGoalPolicy


BASELINE_POLICY_NAME = "baseline_greedy"
IMPROVED_POLICY_NAME = "obstacle_aware_goal"


def comparison_policies():
    return [GreedyGridPolicy(), ObstacleAwareGoalPolicy()]


def comparison_cases() -> list[ExperimentCase]:
    return default_cases() + [
        ExperimentCase(
            name="turn_choice_8x8",
            width=8,
            height=8,
            start=RobotState(x=3, y=3, direction=Direction.NORTH),
            goal=(7, 3),
            obstacles={(3, 2), (2, 3), (2, 2), (2, 4), (4, 2)},
            max_steps=40,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="detour_required_8x8",
            width=8,
            height=8,
            start=RobotState(x=0, y=3, direction=Direction.EAST),
            goal=(7, 3),
            obstacles={(2, 3), (3, 3), (4, 3), (4, 2), (4, 4)},
            max_steps=60,
            expected_reachable=True,
        ),
        ExperimentCase(
            name="loop_trap_8x8",
            width=8,
            height=8,
            start=RobotState(x=1, y=1, direction=Direction.EAST),
            goal=(6, 1),
            obstacles={(2, 1), (3, 1), (4, 1), (2, 0), (2, 2), (4, 0), (4, 2)},
            max_steps=64,
            expected_reachable=True,
        ),
    ]


def run_policy_comparison(results_dir: Path = Path("experiments/v0_4/results")) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    policy_summaries = []
    case_summaries_by_policy = {}

    for policy in comparison_policies():
        case_summaries = []
        for case in comparison_cases():
            case_dir = results_dir / policy.name / case.name
            result = run_episode(
                world=case.build_world(),
                start=case.start,
                policy=policy,
                output_dir=case_dir,
                run_id=f"{policy.name}_{case.name}",
                max_steps=case.max_steps,
                csv_name="run_log.csv",
                trajectory_name="trajectory.png",
            )
            summary = build_case_summary(case, result)
            summary["policy_name"] = policy.name
            summary.update(build_policy_run_metrics(result))
            write_json(case_dir / "summary.json", summary)
            case_summaries.append(summary)

        overall = build_overall_summary(case_summaries)
        case_summaries_by_policy[policy.name] = case_summaries
        policy_summaries.append(
            {
                "policy_name": policy.name,
                "total_runs": overall["total_runs"],
                "success_runs": overall["success_runs"],
                "failed_runs": overall["failed_runs"],
                "success_rate": overall["success_rate"],
                "average_steps_all_runs": overall["average_steps"],
                "average_reward": overall["average_reward"],
                "total_collisions": overall["total_collisions"],
                "total_action_failures": overall["total_action_failures"],
            }
        )

    paired_metrics = build_paired_metrics(
        case_summaries_by_policy[BASELINE_POLICY_NAME],
        case_summaries_by_policy[IMPROVED_POLICY_NAME],
    )
    comparison = {
        "policies": policy_summaries,
        "best_policy_by_success_rate": _best_policies(policy_summaries, "success_rate", reverse=True),
        "best_policy_by_average_steps": paired_metrics["best_policy_by_average_steps"],
        "tie_by_success_rate": _has_tie(policy_summaries, "success_rate", reverse=True),
        "tie_by_average_steps": paired_metrics["tie_by_common_success_average_steps"],
        **paired_metrics,
    }
    write_json(results_dir / "policy_comparison_summary.json", comparison)
    return comparison


def build_paired_metrics(
    baseline_summaries: list[dict],
    improved_summaries: list[dict],
) -> dict:
    baseline_by_case = {item["case_name"]: item for item in baseline_summaries}
    improved_by_case = {item["case_name"]: item for item in improved_summaries}
    per_case = []
    common_success_cases = []
    improved_extra_solved_cases = []
    baseline_extra_solved_cases = []
    both_success_cases = []
    both_failed_cases = []

    for case_name in sorted(baseline_by_case):
        baseline = baseline_by_case[case_name]
        improved = improved_by_case[case_name]
        baseline_success = bool(baseline["reached_goal"])
        improved_success = bool(improved["reached_goal"])

        if baseline_success and improved_success:
            result_type = "both_success"
            both_success_cases.append(case_name)
            common_success_cases.append(case_name)
            step_delta = improved["steps"] - baseline["steps"]
        elif baseline_success and not improved_success:
            result_type = "baseline_only_success"
            baseline_extra_solved_cases.append(case_name)
            step_delta = None
        elif not baseline_success and improved_success:
            result_type = "improved_only_success"
            improved_extra_solved_cases.append(case_name)
            step_delta = None
        else:
            result_type = "both_failed"
            both_failed_cases.append(case_name)
            step_delta = None

        reward_delta = round(improved["total_reward"] - baseline["total_reward"], 2)
        per_case.append(
            {
                "case_name": case_name,
                BASELINE_POLICY_NAME: {
                    "reached_goal": baseline_success,
                    "steps": baseline["steps"],
                    "total_reward": baseline["total_reward"],
                },
                IMPROVED_POLICY_NAME: {
                    "reached_goal": improved_success,
                    "steps": improved["steps"],
                    "total_reward": improved["total_reward"],
                },
                "result_type": result_type,
                "step_delta": step_delta,
                "reward_delta": reward_delta,
            }
        )

    baseline_common_average = _average(
        baseline_by_case[name]["steps"] for name in common_success_cases
    )
    improved_common_average = _average(
        improved_by_case[name]["steps"] for name in common_success_cases
    )
    if common_success_cases:
        common_delta = round(improved_common_average - baseline_common_average, 2)
        best_by_steps = _best_by_common_success_steps(
            baseline_common_average,
            improved_common_average,
        )
        tie_by_common_steps = len(best_by_steps) > 1
    else:
        common_delta = None
        best_by_steps = []
        tie_by_common_steps = False

    return {
        "per_case_comparison": per_case,
        "common_success_cases": common_success_cases,
        "baseline_common_success_average_steps": (
            round(baseline_common_average, 2) if common_success_cases else None
        ),
        "improved_common_success_average_steps": (
            round(improved_common_average, 2) if common_success_cases else None
        ),
        "common_success_average_step_delta": common_delta,
        "improved_extra_solved_cases": improved_extra_solved_cases,
        "baseline_extra_solved_cases": baseline_extra_solved_cases,
        "both_success_cases": both_success_cases,
        "both_failed_cases": both_failed_cases,
        "tie_by_common_success_average_steps": tie_by_common_steps,
        "best_policy_by_average_steps": best_by_steps,
    }


def build_policy_run_metrics(result) -> dict:
    action_counts = {
        "forward": 0,
        "turn_left": 0,
        "turn_right": 0,
    }
    positions = []
    seen_positions = set()
    repeated_position_count = 0

    for record in result.records:
        action_counts[record.action.value] += 1
        position = (record.x, record.y)
        positions.append(position)
        if position in seen_positions:
            repeated_position_count += 1
        else:
            seen_positions.add(position)

    return {
        "action_counts": action_counts,
        "unique_positions": len(seen_positions),
        "repeated_position_count": repeated_position_count,
    }


def main() -> None:
    summary = run_policy_comparison()

    print("results_dir: experiments\\v0_4\\results")
    for policy_summary in summary["policies"]:
        print(
            f"{policy_summary['policy_name']}: "
            f"success_rate={policy_summary['success_rate']:.2f}, "
            f"average_steps_all_runs={policy_summary['average_steps_all_runs']:.2f}, "
            f"average_reward={policy_summary['average_reward']:.2f}"
        )
    print(f"best_policy_by_success_rate: {summary['best_policy_by_success_rate']}")
    print(f"tie_by_success_rate: {summary['tie_by_success_rate']}")
    print(f"best_policy_by_average_steps: {summary['best_policy_by_average_steps']}")
    print(f"tie_by_average_steps: {summary['tie_by_average_steps']}")


def _best_policies(policy_summaries: list[dict], metric: str, reverse: bool) -> list[str]:
    if not policy_summaries:
        return []
    best_value = sorted(policy_summaries, key=lambda item: item[metric], reverse=reverse)[0][metric]
    return [item["policy_name"] for item in policy_summaries if item[metric] == best_value]


def _has_tie(policy_summaries: list[dict], metric: str, reverse: bool) -> bool:
    return len(_best_policies(policy_summaries, metric, reverse)) > 1


def _best_by_common_success_steps(
    baseline_average: float,
    improved_average: float,
) -> list[str]:
    if baseline_average == improved_average:
        return [BASELINE_POLICY_NAME, IMPROVED_POLICY_NAME]
    if baseline_average < improved_average:
        return [BASELINE_POLICY_NAME]
    return [IMPROVED_POLICY_NAME]


def _average(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


if __name__ == "__main__":
    main()
