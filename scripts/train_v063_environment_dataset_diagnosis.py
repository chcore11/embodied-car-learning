from pathlib import Path
import csv
import json
import math
import sys
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.behavior_cloning import ACTIONS, EXCLUDED_COLUMNS, FEATURE_COLUMNS, encode_row_features
from scripts.run_v04_policy_comparison import comparison_cases


DATASET_SOURCE = Path("data/datasets/v0_5")
RESULTS_DIR = Path("experiments/v0_6_3/results")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(DATASET_SOURCE / "train.csv", "train")
    test_rows = load_rows(DATASET_SOURCE / "test.csv", "test")
    all_rows = load_rows(DATASET_SOURCE / "expert_demonstrations.csv", "all")
    case_lookup = {case.name: case for case in comparison_cases()}

    coverage = build_dataset_coverage_summary(train_rows, test_rows, all_rows)
    feature_space, turn_rows = build_turn_feature_space_summary(all_rows)
    ambiguous = build_ambiguous_state_summary(all_rows)
    critical = build_critical_state_summary(all_rows, case_lookup)
    case_difficulty = build_case_difficulty_summary(all_rows, case_lookup)
    final = build_environment_diagnosis_summary(
        coverage,
        feature_space,
        ambiguous,
        critical,
        case_difficulty,
    )

    write_json(RESULTS_DIR / "dataset_coverage_summary.json", coverage)
    write_json(RESULTS_DIR / "turn_feature_space_summary.json", feature_space)
    write_turn_samples_csv(RESULTS_DIR / "turn_samples.csv", turn_rows)
    write_json(RESULTS_DIR / "ambiguous_state_summary.json", ambiguous)
    write_json(RESULTS_DIR / "critical_state_summary.json", critical)
    write_json(RESULTS_DIR / "case_difficulty_summary.json", case_difficulty)
    write_json(RESULTS_DIR / "environment_diagnosis_summary.json", final)

    print(f"results_dir: {RESULTS_DIR}")
    print(f"turn_left_rows_count: {coverage['turn_left_rows_count']}")
    print(f"turn_right_rows_count: {coverage['turn_right_rows_count']}")
    print(f"dataset_coverage_result: {final['dataset_coverage_result']}")
    print(f"feature_space_result: {final['feature_space_result']}")
    print(f"final_determination: {final['final_determination']}")


def load_rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in rows:
        row["split"] = split
    return rows


def build_dataset_coverage_summary(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    all_rows: list[dict[str, str]],
) -> dict:
    per_case = {
        case: action_distribution(rows)
        for case, rows in group_by(all_rows, "case_name").items()
    }
    per_action_case = {
        action: count_by([row for row in all_rows if row["action"] == action], "case_name")
        for action in ACTIONS
    }
    turn_left_count = count_action(all_rows, "turn_left")
    turn_right_count = count_action(all_rows, "turn_right")
    forward_count = count_action(all_rows, "forward")
    per_case_turn_count = {
        case: counts["turn_left"] + counts["turn_right"]
        for case, counts in per_case.items()
    }
    cases_with_no_turn_left = sorted(
        case for case, counts in per_case.items() if counts["turn_left"] == 0
    )
    cases_with_no_turn_right = sorted(
        case for case, counts in per_case.items() if counts["turn_right"] == 0
    )
    cases_with_no_turn_actions = sorted(
        case for case, count in per_case_turn_count.items() if count == 0
    )
    forward_ratio = round(forward_count / len(all_rows), 4) if all_rows else 0.0

    issues = []
    if forward_ratio >= 0.75:
        issues.append("Dataset is dominated by forward actions.")
    if turn_right_count < 5:
        issues.append("turn_right has very low coverage.")
    if turn_left_count + turn_right_count < max(10, len(all_rows) * 0.2):
        issues.append("Turn action coverage is low.")
    if cases_with_no_turn_right:
        issues.append("Some cases have no turn_right samples.")

    return {
        "train_action_distribution": action_distribution(train_rows),
        "test_action_distribution": action_distribution(test_rows),
        "all_action_distribution": action_distribution(all_rows),
        "per_case_action_distribution": per_case,
        "per_action_case_distribution": per_action_case,
        "per_action_direction_distribution": distribution_by_action(all_rows, "direction"),
        "per_action_front_blocked_distribution": distribution_by_action(all_rows, "front_blocked"),
        "per_action_left_blocked_distribution": distribution_by_action(all_rows, "left_blocked"),
        "per_action_right_blocked_distribution": distribution_by_action(all_rows, "right_blocked"),
        "turn_left_rows_count": turn_left_count,
        "turn_right_rows_count": turn_right_count,
        "forward_rows_count": forward_count,
        "forward_rows_with_front_blocked_count": count_matching(
            all_rows, action="forward", field="front_blocked", value="1"
        ),
        "turn_rows_with_front_blocked_count": sum(
            1
            for row in all_rows
            if row["action"] in {"turn_left", "turn_right"} and row["front_blocked"] == "1"
        ),
        "per_case_turn_count": per_case_turn_count,
        "cases_with_no_turn_left": cases_with_no_turn_left,
        "cases_with_no_turn_right": cases_with_no_turn_right,
        "cases_with_no_turn_actions": cases_with_no_turn_actions,
        "dataset_coverage_assessment": " ".join(issues) if issues else "Coverage is acceptable for this stage.",
    }


def build_turn_feature_space_summary(all_rows: list[dict[str, str]]) -> tuple[dict, list[dict]]:
    by_action = {action: [row for row in all_rows if row["action"] == action] for action in ACTIONS}
    turn_samples = []
    turn_left_forward_closer = 0
    turn_right_forward_closer = 0

    for row in all_rows:
        if row["action"] not in {"turn_left", "turn_right"}:
            continue
        distances = nearest_distances_by_action(row, by_action)
        closest_action = closest_action_class(distances)
        same_action_distance = distances[row["action"]]
        forward_distance = distances["forward"]
        if row["action"] == "turn_left" and is_less(forward_distance, same_action_distance):
            turn_left_forward_closer += 1
        if row["action"] == "turn_right" and is_less(forward_distance, same_action_distance):
            turn_right_forward_closer += 1
        turn_samples.append(
            {
                "split": row["split"],
                "case_name": row["case_name"],
                "episode_id": row["episode_id"],
                "step": row["step"],
                "action": row["action"],
                "x": row["x"],
                "y": row["y"],
                "direction": row["direction"],
                "front_blocked": row["front_blocked"],
                "left_blocked": row["left_blocked"],
                "right_blocked": row["right_blocked"],
                "distance_to_goal": row["distance_to_goal"],
                "dx_to_goal": row["dx_to_goal"],
                "dy_to_goal": row["dy_to_goal"],
                "nearest_forward_distance": format_distance(forward_distance),
                "nearest_same_action_distance": format_distance(same_action_distance),
                "nearest_turn_left_distance": format_distance(distances["turn_left"]),
                "nearest_turn_right_distance": format_distance(distances["turn_right"]),
                "closest_action_class": closest_action,
                "diagnosis_note": build_turn_diagnosis_note(row, forward_distance, same_action_distance),
            }
        )

    summary = {
        "average_nearest_same_class_distance_forward": average_nearest_same_class(by_action["forward"]),
        "average_nearest_same_class_distance_turn_left": average_nearest_same_class(by_action["turn_left"]),
        "average_nearest_same_class_distance_turn_right": average_nearest_same_class(by_action["turn_right"]),
        "average_nearest_forward_distance_for_turn_left": average(
            nearest_distances_by_action(row, by_action)["forward"]
            for row in by_action["turn_left"]
        ),
        "average_nearest_forward_distance_for_turn_right": average(
            nearest_distances_by_action(row, by_action)["forward"]
            for row in by_action["turn_right"]
        ),
        "average_nearest_same_action_distance_for_turn_left": average(
            nearest_distances_by_action(row, by_action)["turn_left"]
            for row in by_action["turn_left"]
        ),
        "average_nearest_same_action_distance_for_turn_right": average(
            nearest_distances_by_action(row, by_action)["turn_right"]
            for row in by_action["turn_right"]
        ),
        "turn_left_samples_closer_to_forward_than_turn_left_count": turn_left_forward_closer,
        "turn_right_samples_closer_to_forward_than_turn_right_count": turn_right_forward_closer,
        "feature_space_assessment": feature_space_assessment(
            len(by_action["turn_left"]),
            len(by_action["turn_right"]),
            turn_left_forward_closer,
            turn_right_forward_closer,
        ),
    }
    return summary, turn_samples


def build_ambiguous_state_summary(all_rows: list[dict[str, str]]) -> dict:
    exact_groups = defaultdict(list)
    for row in all_rows:
        exact_groups[feature_key(row)].append(row)
    ambiguous_exact = [
        rows for rows in exact_groups.values() if len({row["action"] for row in rows}) > 1
    ]
    near_groups = defaultdict(list)
    for row in all_rows:
        near_groups[near_feature_key(row)].append(row)
    ambiguous_near = [
        rows for rows in near_groups.values() if len({row["action"] for row in rows}) > 1
    ]
    return {
        "exact_duplicate_feature_groups_count": sum(
            1 for rows in exact_groups.values() if len(rows) > 1
        ),
        "exact_duplicate_feature_groups_with_multiple_actions_count": len(ambiguous_exact),
        "examples_exact_ambiguous_groups": ambiguous_examples(ambiguous_exact),
        "near_duplicate_feature_groups_with_multiple_actions_count": len(ambiguous_near),
        "examples_near_ambiguous_groups": ambiguous_examples(ambiguous_near),
        "ambiguity_assessment": (
            "Some similar observations map to multiple actions."
            if ambiguous_exact or ambiguous_near
            else "No strong label ambiguity found in exact or coarse feature groups."
        ),
    }


def build_critical_state_summary(all_rows: list[dict[str, str]], case_lookup: dict) -> dict:
    front_blocked_rows = [row for row in all_rows if row["front_blocked"] == "1"]
    boundary_rows = [row for row in all_rows if is_boundary_state(row, case_lookup)]
    corner_rows = [row for row in all_rows if is_corner_state(row, case_lookup)]
    turn_rows = [row for row in all_rows if row["action"] in {"turn_left", "turn_right"}]
    forward_rows = [row for row in all_rows if row["action"] == "forward"]
    front_blocked_turn_count = sum(1 for row in front_blocked_rows if row["action"] in {"turn_left", "turn_right"})
    front_blocked_forward_count = sum(1 for row in front_blocked_rows if row["action"] == "forward")
    return {
        "front_blocked_action_distribution": action_distribution(front_blocked_rows),
        "boundary_state_action_distribution": action_distribution(boundary_rows),
        "corner_state_action_distribution": action_distribution(corner_rows),
        "critical_turn_state_count": len(turn_rows),
        "critical_forward_state_count": len(forward_rows),
        "front_blocked_forward_count": front_blocked_forward_count,
        "front_blocked_turn_count": front_blocked_turn_count,
        "critical_state_assessment": (
            "There are few turn-critical samples where front_blocked is true."
            if front_blocked_turn_count < 10
            else "Critical turn states have some coverage."
        ),
    }


def build_case_difficulty_summary(all_rows: list[dict[str, str]], case_lookup: dict) -> dict:
    cases = []
    for case_name, rows in sorted(group_by(all_rows, "case_name").items()):
        counts = action_distribution(rows)
        turn_count = counts["turn_left"] + counts["turn_right"]
        turn_ratio = round(turn_count / len(rows), 4) if rows else 0.0
        case = case_lookup.get(case_name)
        has_obstacle = bool(case and case.obstacles)
        cases.append(
            {
                "case_name": case_name,
                "total_rows": len(rows),
                "forward_count": counts["forward"],
                "turn_left_count": counts["turn_left"],
                "turn_right_count": counts["turn_right"],
                "turn_ratio": turn_ratio,
                "has_obstacle": has_obstacle,
                "has_turn_left": counts["turn_left"] > 0,
                "has_turn_right": counts["turn_right"] > 0,
                "estimated_case_difficulty": estimate_case_difficulty(has_obstacle, turn_ratio, len(rows)),
                "diagnosis_note": case_difficulty_note(counts, has_obstacle),
            }
        )
    return {
        "cases": cases,
        "case_difficulty_result": (
            "Several cases are forward-heavy or miss one turn direction."
            if any(not item["has_turn_right"] or item["turn_ratio"] < 0.2 for item in cases)
            else "Case action mix is acceptable for this stage."
        ),
    }


def build_environment_diagnosis_summary(
    coverage: dict,
    feature_space: dict,
    ambiguous: dict,
    critical: dict,
    case_difficulty: dict,
) -> dict:
    coverage_problem = (
        coverage["forward_rows_count"] >= 0.75 * sum(coverage["all_action_distribution"].values())
        or coverage["turn_right_rows_count"] < 5
    )
    feature_problem = (
        feature_space["turn_left_samples_closer_to_forward_than_turn_left_count"] > 0
        or feature_space["turn_right_samples_closer_to_forward_than_turn_right_count"] > 0
    )
    if coverage_problem and feature_problem:
        final = "C"
    elif coverage_problem:
        final = "A"
    elif feature_problem:
        final = "B"
    else:
        final = "A"
    return {
        "dataset_source": str(DATASET_SOURCE),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "key_findings": [
            coverage["dataset_coverage_assessment"],
            feature_space["feature_space_assessment"],
            ambiguous["ambiguity_assessment"],
            critical["critical_state_assessment"],
            case_difficulty["case_difficulty_result"],
        ],
        "dataset_coverage_result": "problem" if coverage_problem else "acceptable",
        "feature_space_result": "problem" if feature_problem else "acceptable",
        "ambiguity_result": ambiguous["ambiguity_assessment"],
        "critical_state_result": critical["critical_state_assessment"],
        "case_difficulty_result": case_difficulty["case_difficulty_result"],
        "final_determination": final,
        "final_determination_label": {
            "A": "Mostly dataset coverage problem",
            "B": "Mostly feature representation problem",
            "C": "Both dataset coverage and feature representation problem",
            "D": "Environment representation limitation",
        }[final],
    }


def action_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return {action: count_action(rows, action) for action in ACTIONS}


def distribution_by_action(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, int]]:
    return {action: count_by([row for row in rows if row["action"] == action], field) for action in ACTIONS}


def count_action(rows: list[dict[str, str]], action: str) -> int:
    return sum(1 for row in rows if row["action"] == action)


def count_by(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        counts[row[field]] += 1
    return dict(sorted(counts.items()))


def count_matching(rows: list[dict[str, str]], action: str, field: str, value: str) -> int:
    return sum(1 for row in rows if row["action"] == action and row[field] == value)


def group_by(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return groups


def distance(left: dict[str, str], right: dict[str, str]) -> float:
    left_features = encode_row_features(left)
    right_features = encode_row_features(right)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left_features, right_features)))


def nearest_distance(row: dict[str, str], candidates: list[dict[str, str]], exclude_self: bool) -> float | None:
    distances = []
    for candidate in candidates:
        if exclude_self and candidate is row:
            continue
        distances.append(distance(row, candidate))
    return min(distances) if distances else None


def nearest_distances_by_action(row: dict[str, str], by_action: dict[str, list[dict[str, str]]]) -> dict[str, float | None]:
    return {
        action: nearest_distance(row, rows, exclude_self=(action == row["action"]))
        for action, rows in by_action.items()
    }


def closest_action_class(distances: dict[str, float | None]) -> str:
    candidates = [(value, ACTIONS.index(action), action) for action, value in distances.items() if value is not None]
    return min(candidates)[2] if candidates else ""


def average_nearest_same_class(rows: list[dict[str, str]]) -> float | None:
    return average(nearest_distance(row, rows, exclude_self=True) for row in rows)


def average(values) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 6)


def is_less(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left < right


def format_distance(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def build_turn_diagnosis_note(row: dict[str, str], forward_distance: float | None, same_distance: float | None) -> str:
    if is_less(forward_distance, same_distance):
        return f"{row['action']} sample is closer to forward than to same-action samples."
    return f"{row['action']} sample is closest enough to same-action samples."


def feature_space_assessment(
    turn_left_total: int,
    turn_right_total: int,
    turn_left_forward_closer: int,
    turn_right_forward_closer: int,
) -> str:
    if turn_left_forward_closer or turn_right_forward_closer:
        return "Some turn samples are closer to forward samples than to same-action turn samples."
    if turn_left_total < 5 or turn_right_total < 5:
        return "Turn sample counts are too small to confidently assess separability."
    return "Turn samples are not obviously closer to forward in this dataset."


def feature_key(row: dict[str, str]) -> tuple:
    return tuple(row[column] for column in FEATURE_COLUMNS)


def near_feature_key(row: dict[str, str]) -> tuple:
    return (
        row["direction"],
        row["front_blocked"],
        row["left_blocked"],
        row["right_blocked"],
        sign(int(row["dx_to_goal"])),
        sign(int(row["dy_to_goal"])),
    )


def sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def ambiguous_examples(groups: list[list[dict[str, str]]]) -> list[dict]:
    examples = []
    for rows in groups[:5]:
        examples.append(
            {
                "feature_key": list(feature_key(rows[0])),
                "actions": sorted({row["action"] for row in rows}),
                "examples": [
                    {
                        "case_name": row["case_name"],
                        "step": row["step"],
                        "action": row["action"],
                    }
                    for row in rows[:5]
                ],
            }
        )
    return examples


def is_boundary_state(row: dict[str, str], case_lookup: dict) -> bool:
    case = case_lookup.get(row["case_name"])
    if not case:
        return False
    x = int(row["x"])
    y = int(row["y"])
    return x == 0 or y == 0 or x == case.width - 1 or y == case.height - 1


def is_corner_state(row: dict[str, str], case_lookup: dict) -> bool:
    case = case_lookup.get(row["case_name"])
    if not case:
        return False
    x = int(row["x"])
    y = int(row["y"])
    return x in {0, case.width - 1} and y in {0, case.height - 1}


def estimate_case_difficulty(has_obstacle: bool, turn_ratio: float, row_count: int) -> str:
    if has_obstacle and turn_ratio >= 0.25:
        return "medium"
    if has_obstacle:
        return "medium_forward_heavy"
    if row_count <= 16 and turn_ratio < 0.2:
        return "easy_forward_heavy"
    return "easy"


def case_difficulty_note(counts: dict[str, int], has_obstacle: bool) -> str:
    notes = []
    if counts["turn_right"] == 0:
        notes.append("No turn_right samples.")
    if counts["turn_left"] == 0:
        notes.append("No turn_left samples.")
    if counts["forward"] > counts["turn_left"] + counts["turn_right"] * 3:
        notes.append("Forward dominates the case.")
    if has_obstacle:
        notes.append("Case has obstacles.")
    return " ".join(notes) if notes else "Case has some turn coverage."


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_turn_samples_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "case_name",
        "episode_id",
        "step",
        "action",
        "x",
        "y",
        "direction",
        "front_blocked",
        "left_blocked",
        "right_blocked",
        "distance_to_goal",
        "dx_to_goal",
        "dy_to_goal",
        "nearest_forward_distance",
        "nearest_same_action_distance",
        "nearest_turn_left_distance",
        "nearest_turn_right_distance",
        "closest_action_class",
        "diagnosis_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
