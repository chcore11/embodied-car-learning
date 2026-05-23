from pathlib import Path
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.behavior_cloning import KNNBehaviorCloningPolicy
from embody.experiments import ExperimentCase, build_case_summary, build_overall_summary, default_cases, write_json
from embody.grid_world import run_episode, write_png
from embody.policy import ObstacleAwareGoalPolicy
from scripts.train_v065_turn_focused_dataset_coverage import turn_focused_cases


RESULTS_DIR = Path("experiments/v0_7_final")
ORIGINAL_BC_MODEL = Path("models/v0_6/behavior_cloning_policy.json")
IMPROVED_BC_MODEL = Path("models/v0_6_6/knn_class_balanced.json")
ACTIONS = ["forward", "turn_left", "turn_right"]
POLICY_SPECS = [
    ("teacher", "obstacle_aware_goal"),
    ("original_bc", "v0_6_behavior_cloning_knn"),
    ("improved_bc", "v0_6_6_knn_class_balanced"),
]


def main() -> None:
    summary = run_final_comparison()
    print(f"results_dir: {RESULTS_DIR}")
    print(f"benchmark_cases: {[case['name'] for case in summary['benchmark_cases']]}")
    for row in summary["comparison_table"]:
        print(
            f"{row['policy_id']}: success_rate={row['success_rate']}, "
            f"average_steps_all_runs={row['average_steps_all_runs']}, "
            f"average_reward={row['average_reward']}, "
            f"total_collisions={row['total_collisions']}"
        )


def run_final_comparison(results_dir: Path = RESULTS_DIR) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    cases = final_benchmark_cases()
    policies = load_policies()

    write_json(results_dir / "benchmark_cases.json", {"cases": [case_to_dict(case) for case in cases]})

    comparison_table = []
    case_level_rows = []

    for policy_id, policy_label, policy in policies:
        policy_dir = results_dir / "trajectories" / policy_id
        summaries = []
        for case in cases:
            case_dir = policy_dir / case.name
            result = run_episode(
                world=case.build_world(),
                start=case.start,
                policy=policy,
                output_dir=case_dir,
                run_id=f"v0_7_{policy_id}_{case.name}",
                max_steps=case.max_steps,
                csv_name="run_log.csv",
                trajectory_name="trajectory.png",
            )
            summary = build_case_summary(case, result)
            summary["policy_id"] = policy_id
            summary["policy_label"] = policy_label
            write_json(case_dir / "summary.json", summary)
            summaries.append(summary)
            case_level_rows.append(case_result_row(policy_id, policy_label, summary))

        overall = build_overall_summary(summaries)
        comparison_table.append(policy_result_row(policy_id, policy_label, overall, summaries))

    write_csv(results_dir / "final_comparison_table.csv", comparison_table)
    write_csv(results_dir / "case_level_results.csv", case_level_rows)
    write_bar_chart(
        results_dir / "final_success_rate_comparison.png",
        [(row["policy_id"], float(row["success_rate"])) for row in comparison_table],
        max_value=1.0,
    )
    write_action_distribution_chart(
        results_dir / "dataset_action_distribution_comparison.png"
    )

    final_summary = {
        "version": "v0.7_final",
        "purpose": "final unified comparison without retraining",
        "benchmark_cases": [case_to_dict(case) for case in cases],
        "model_sources": {
            "original_bc": str(ORIGINAL_BC_MODEL),
            "improved_bc": str(IMPROVED_BC_MODEL),
        },
        "evaluation_rules": {
            "same_benchmark_cases": True,
            "same_environment_parameters_per_case": True,
            "same_reward_logic": True,
            "same_collision_counting": True,
            "models_retrained": False,
        },
        "comparison_table": comparison_table,
        "output_files": {
            "benchmark_cases": str(results_dir / "benchmark_cases.json"),
            "final_comparison_summary": str(results_dir / "final_comparison_summary.json"),
            "final_comparison_table": str(results_dir / "final_comparison_table.csv"),
            "case_level_results": str(results_dir / "case_level_results.csv"),
            "success_rate_chart": str(results_dir / "final_success_rate_comparison.png"),
            "dataset_action_distribution_chart": str(
                results_dir / "dataset_action_distribution_comparison.png"
            ),
        },
        "representative_trajectories": representative_trajectories(results_dir, case_level_rows),
        "known_limitations": [
            "The teacher is a heuristic policy, not an optimal expert.",
            "The final behavior cloning model still fails on some turn-heavy cases.",
            "Collision-heavy failures remain visible in the unified benchmark.",
            "This is a compact 2D grid learning project, not a real robot or VLA system.",
        ],
    }
    write_json(results_dir / "final_comparison_summary.json", final_summary)
    return final_summary


def final_benchmark_cases() -> list[ExperimentCase]:
    no_obstacle = next(case for case in default_cases() if case.name == "no_obstacle_8x8")
    turn_cases_by_name = {case.name: case for case in turn_focused_cases()}
    selected_names = [
        "left_turn_required",
        "right_turn_required",
        "front_blocked_left_open",
        "front_blocked_right_open",
        "corridor_left_turn",
        "corridor_right_turn",
        "consecutive_turns",
    ]
    return [no_obstacle] + [turn_cases_by_name[name] for name in selected_names]


def load_policies() -> list[tuple[str, str, object]]:
    original_bc = KNNBehaviorCloningPolicy.load(ORIGINAL_BC_MODEL)
    original_bc.name = "original_bc"
    improved_bc = KNNBehaviorCloningPolicy.load(IMPROVED_BC_MODEL)
    improved_bc.name = "improved_bc"
    return [
        ("teacher", "obstacle_aware_goal", ObstacleAwareGoalPolicy()),
        ("original_bc", "v0.6 behavior_cloning_knn", original_bc),
        ("improved_bc", "v0.6.6 knn_class_balanced", improved_bc),
    ]


def case_to_dict(case: ExperimentCase) -> dict:
    return {
        "name": case.name,
        "width": case.width,
        "height": case.height,
        "start": {
            "x": case.start.x,
            "y": case.start.y,
            "direction": case.start.direction.value,
        },
        "goal": list(case.goal),
        "obstacles": [list(item) for item in sorted(case.obstacles)],
        "max_steps": case.max_steps,
        "expected_reachable": case.expected_reachable,
        "action_fail_prob": case.action_fail_prob,
        "sensor_noise_prob": case.sensor_noise_prob,
        "random_seed": case.random_seed,
    }


def policy_result_row(policy_id: str, policy_label: str, overall: dict, summaries: list[dict]) -> dict:
    return {
        "policy_id": policy_id,
        "policy_label": policy_label,
        "total_runs": overall["total_runs"],
        "success_runs": overall["success_runs"],
        "success_rate": overall["success_rate"],
        "average_steps_all_runs": overall["average_steps"],
        "average_reward": overall["average_reward"],
        "total_collisions": overall["total_collisions"],
        "failed_cases": ";".join(item["case_name"] for item in summaries if not item["reached_goal"]),
    }


def case_result_row(policy_id: str, policy_label: str, summary: dict) -> dict:
    return {
        "policy_id": policy_id,
        "policy_label": policy_label,
        "case_name": summary["case_name"],
        "reached_goal": summary["reached_goal"],
        "result_type": summary["result_type"],
        "steps": summary["steps"],
        "total_reward": summary["total_reward"],
        "collision_count": summary["collision_count"],
        "failure_reason": summary["failure_reason"],
        "trajectory_path": summary["trajectory_path"],
        "csv_path": summary["csv_path"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bar_chart(path: Path, values: list[tuple[str, float]], max_value: float) -> None:
    width = 520
    height = 300
    pixels = [[(248, 248, 248) for _ in range(width)] for _ in range(height)]
    margin_left = 60
    margin_bottom = 45
    chart_width = width - margin_left - 30
    chart_height = height - 40 - margin_bottom
    draw_rect(pixels, margin_left, 40, margin_left + chart_width, 40 + chart_height, (238, 238, 238))
    colors = [(80, 150, 220), (220, 120, 80), (90, 170, 100)]
    bar_width = max(30, chart_width // (len(values) * 2))
    for index, (_, value) in enumerate(values):
        bar_height = int(chart_height * (value / max_value)) if max_value else 0
        x1 = margin_left + 35 + index * (bar_width * 2)
        y1 = 40 + chart_height - bar_height
        draw_rect(pixels, x1, y1, x1 + bar_width, 40 + chart_height, colors[index % len(colors)])
    draw_line(pixels, margin_left, 40, margin_left, 40 + chart_height, (80, 80, 80))
    draw_line(pixels, margin_left, 40 + chart_height, margin_left + chart_width, 40 + chart_height, (80, 80, 80))
    write_png(path, pixels)


def write_action_distribution_chart(path: Path) -> None:
    distributions = [
        ("v0_5", action_distribution_from_csv(Path("data/datasets/v0_5/expert_demonstrations.csv"))),
        ("v0_6_6", action_distribution_from_csv(Path("data/datasets/v0_6_6/expert_demonstrations.csv"))),
    ]
    width = 560
    height = 320
    pixels = [[(248, 248, 248) for _ in range(width)] for _ in range(height)]
    max_count = max(max(distribution.values()) for _, distribution in distributions)
    margin_left = 55
    chart_top = 35
    chart_height = 230
    colors = {
        "forward": (80, 150, 220),
        "turn_left": (220, 120, 80),
        "turn_right": (90, 170, 100),
    }
    bar_width = 34
    x = margin_left + 25
    for _, distribution in distributions:
        for action in ACTIONS:
            count = distribution[action]
            bar_height = int(chart_height * (count / max_count)) if max_count else 0
            draw_rect(
                pixels,
                x,
                chart_top + chart_height - bar_height,
                x + bar_width,
                chart_top + chart_height,
                colors[action],
            )
            x += bar_width + 8
        x += 55
    draw_line(pixels, margin_left, chart_top, margin_left, chart_top + chart_height, (80, 80, 80))
    draw_line(pixels, margin_left, chart_top + chart_height, width - 30, chart_top + chart_height, (80, 80, 80))
    write_png(path, pixels)


def action_distribution_from_csv(path: Path) -> dict[str, int]:
    counts = {action: 0 for action in ACTIONS}
    if not path.exists():
        return counts
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            action = row.get("action")
            if action in counts:
                counts[action] += 1
    return counts


def representative_trajectories(results_dir: Path, case_level_rows: list[dict]) -> dict:
    rows_by_policy_case = {
        (row["policy_id"], row["case_name"]): row for row in case_level_rows
    }

    def path_for(policy_id: str, case_name: str) -> str | None:
        row = rows_by_policy_case.get((policy_id, case_name))
        return None if row is None else row["trajectory_path"]

    return {
        "teacher_success": path_for("teacher", "corridor_left_turn"),
        "original_bc_failure": path_for("original_bc", "front_blocked_left_open"),
        "improved_bc_success": path_for("improved_bc", "front_blocked_right_open"),
        "improved_bc_failure_or_collision": path_for("improved_bc", "consecutive_turns"),
    }


def draw_rect(
    pixels: list[list[tuple[int, int, int]]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    for y in range(max(0, y1), min(height, y2)):
        for x in range(max(0, x1), min(width, x2)):
            pixels[y][x] = color


def draw_line(
    pixels: list[list[tuple[int, int, int]]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    for index in range(steps + 1):
        t = index / steps
        x = round(x1 + (x2 - x1) * t)
        y = round(y1 + (y2 - y1) * t)
        if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
            pixels[y][x] = color


if __name__ == "__main__":
    main()
