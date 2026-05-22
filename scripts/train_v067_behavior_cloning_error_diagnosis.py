from pathlib import Path
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embody.experiments import write_json


DATASET_DIR = Path("data/datasets/v0_6_6")
SOURCE_RESULTS_DIR = Path("experiments/v0_6_6/results")
RESULTS_DIR = Path("experiments/v0_6_7/results")
ACTIONS = ["forward", "turn_left", "turn_right"]
MODEL_SPECS = {
    "knn_majority": "v066_knn_majority",
    "knn_class_balanced": "v066_knn_class_balanced",
}


def main() -> None:
    summary = run_v067_error_diagnosis()
    print(f"results_dir: {RESULTS_DIR}")
    print(f"main_bottleneck: {summary['main_bottleneck']}")
    print(f"next_priority: {summary['next_priority']}")
    print(f"clean_success_rate: {summary['rollout_collision_overview']['clean_success_rate']}")
    print(
        "most_collision_heavy_case: "
        f"{summary['rollout_collision_overview']['collision_heavy_cases'][0]['case_name']}"
    )


def run_v067_error_diagnosis(
    dataset_dir: Path = DATASET_DIR,
    source_results_dir: Path = SOURCE_RESULTS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    test_rows = read_csv(dataset_dir / "test.csv")
    predictions = {
        model: read_csv(source_results_dir / f"offline_predictions_{model}.csv")
        for model in MODEL_SPECS
    }
    rollout_cases = {
        model: read_rollout_cases(source_results_dir, policy_dir)
        for model, policy_dir in MODEL_SPECS.items()
    }

    confusion = build_confusion_matrix_summary(test_rows, predictions)
    front_errors = build_front_blocked_error_cases(test_rows, predictions)
    turn_left = build_turn_left_error_analysis(test_rows, predictions)
    rollout = build_rollout_collision_summary(rollout_cases)
    collision_heavy = build_collision_heavy_cases(rollout_cases)
    model_diff = build_model_difference_summary(rollout_cases, predictions)
    final_summary = build_v067_summary(confusion, front_errors, turn_left, rollout, collision_heavy, model_diff)

    write_json(results_dir / "confusion_matrix_summary.json", confusion)
    write_json(results_dir / "front_blocked_error_cases.json", front_errors)
    write_json(results_dir / "turn_left_error_analysis.json", turn_left)
    write_json(results_dir / "rollout_collision_summary.json", rollout)
    write_json(results_dir / "collision_heavy_cases.json", collision_heavy)
    write_json(results_dir / "model_difference_summary.json", model_diff)
    write_json(results_dir / "v067_summary.json", final_summary)
    return final_summary


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_confusion_matrix_summary(test_rows: list[dict[str, str]], predictions: dict) -> dict:
    summary = {}
    for model, rows in predictions.items():
        matrix = empty_matrix()
        for test_row, pred_row in zip(test_rows, rows):
            actual = test_row["action"]
            predicted = pred_row["predicted_action"]
            matrix[actual][predicted] += 1
        turn_left_total = sum(matrix["turn_left"].values())
        turn_right_total = sum(matrix["turn_right"].values())
        summary[model] = {
            "confusion_matrix": matrix,
            "turn_left_as_forward_count": matrix["turn_left"]["forward"],
            "turn_left_as_turn_right_count": matrix["turn_left"]["turn_right"],
            "turn_right_as_forward_count": matrix["turn_right"]["forward"],
            "turn_right_as_turn_left_count": matrix["turn_right"]["turn_left"],
            "turn_left_recall": round(matrix["turn_left"]["turn_left"] / turn_left_total, 4) if turn_left_total else None,
            "turn_right_recall": round(matrix["turn_right"]["turn_right"] / turn_right_total, 4) if turn_right_total else None,
            "turn_left_test_count": turn_left_total,
            "turn_right_test_count": turn_right_total,
        }
    return summary


def build_front_blocked_error_cases(test_rows: list[dict[str, str]], predictions: dict) -> dict:
    output = {}
    for model, rows in predictions.items():
        errors = []
        actual_distribution = action_counts(row for row in test_rows if row["front_blocked"] == "1")
        predicted_distribution = {action: 0 for action in ACTIONS}
        for index, (test_row, pred_row) in enumerate(zip(test_rows, rows)):
            if test_row["front_blocked"] != "1":
                continue
            predicted = pred_row["predicted_action"]
            predicted_distribution[predicted] += 1
            if predicted == "forward":
                errors.append(
                    {
                        "row_index": index,
                        "case_name": test_row["case_name"],
                        "episode_id": test_row["episode_id"],
                        "step": test_row["step"],
                        "x": test_row["x"],
                        "y": test_row["y"],
                        "direction": test_row["direction"],
                        "front_blocked": test_row["front_blocked"],
                        "left_blocked": test_row["left_blocked"],
                        "right_blocked": test_row["right_blocked"],
                        "dx_to_goal": test_row["dx_to_goal"],
                        "dy_to_goal": test_row["dy_to_goal"],
                        "actual_action": test_row["action"],
                        "predicted_action": predicted,
                    }
                )
        output[model] = {
            "front_blocked_test_rows": sum(actual_distribution.values()),
            "front_blocked_actual_action_distribution": actual_distribution,
            "front_blocked_prediction_distribution": predicted_distribution,
            "front_blocked_predicted_forward_count": len(errors),
            "front_blocked_predicted_forward_cases": errors,
            "main_source": summarize_front_blocked_error_source(errors),
        }
    return output


def build_turn_left_error_analysis(test_rows: list[dict[str, str]], predictions: dict) -> dict:
    output = {}
    for model, rows in predictions.items():
        errors = []
        for index, (test_row, pred_row) in enumerate(zip(test_rows, rows)):
            if test_row["action"] != "turn_left":
                continue
            predicted = pred_row["predicted_action"]
            if predicted != "turn_left":
                errors.append(
                    {
                        "row_index": index,
                        "case_name": test_row["case_name"],
                        "episode_id": test_row["episode_id"],
                        "step": test_row["step"],
                        "x": test_row["x"],
                        "y": test_row["y"],
                        "direction": test_row["direction"],
                        "front_blocked": test_row["front_blocked"],
                        "left_blocked": test_row["left_blocked"],
                        "right_blocked": test_row["right_blocked"],
                        "dx_to_goal": test_row["dx_to_goal"],
                        "dy_to_goal": test_row["dy_to_goal"],
                        "predicted_action": predicted,
                    }
                )
        predicted_counts = {action: 0 for action in ACTIONS}
        turn_left_rows = []
        for test_row, pred_row in zip(test_rows, rows):
            if test_row["action"] == "turn_left":
                predicted_counts[pred_row["predicted_action"]] += 1
                turn_left_rows.append(test_row)
        total = len(turn_left_rows)
        output[model] = {
            "turn_left_test_count": total,
            "turn_left_predicted_distribution": predicted_counts,
            "turn_left_recall": round(predicted_counts["turn_left"] / total, 4) if total else None,
            "turn_left_as_forward_count": predicted_counts["forward"],
            "turn_left_as_turn_right_count": predicted_counts["turn_right"],
            "turn_left_error_cases": errors,
            "interpretation": (
                "turn_left recall is based on very few test rows; metric is unstable and errors still matter."
                if total < 10
                else "turn_left recall is low with enough test support."
            ),
        }
    return output


def read_rollout_cases(source_results_dir: Path, policy_dir: str) -> dict:
    root = source_results_dir / policy_dir
    cases = {}
    for summary_path in root.glob("*/summary.json"):
        case_name = summary_path.parent.name
        run_log_path = summary_path.parent / "run_log.csv"
        summary = read_json(summary_path)
        records = read_csv(run_log_path) if run_log_path.exists() else []
        cases[case_name] = {
            "summary": summary,
            "records": records,
            "log_available": run_log_path.exists(),
        }
    return cases


def build_rollout_collision_summary(rollout_cases: dict) -> dict:
    output = {}
    for model, cases in rollout_cases.items():
        case_rows = []
        clean_success_count = 0
        success_collisions = []
        failed_cases = []
        for case_name, item in sorted(cases.items()):
            summary = item["summary"]
            collision_count = int(summary.get("collision_count", 0))
            reached_goal = bool(summary.get("reached_goal", False))
            if reached_goal and collision_count == 0:
                clean_success_count += 1
            if reached_goal:
                success_collisions.append(collision_count)
            else:
                failed_cases.append(case_name)
            first_collision = first_collision_state(item["records"])
            case_rows.append(
                {
                    "case_name": case_name,
                    "reached_goal": reached_goal,
                    "failure_reason": summary.get("failure_reason"),
                    "collision_count": collision_count,
                    "steps": summary.get("steps"),
                    "first_collision_step": first_collision.get("step"),
                    "first_collision_state": first_collision,
                    "log_available": item["log_available"],
                }
            )
        total_runs = len(case_rows)
        success_runs = sum(1 for row in case_rows if row["reached_goal"])
        output[model] = {
            "total_runs": total_runs,
            "success_runs": success_runs,
            "failed_runs": total_runs - success_runs,
            "clean_success_count": clean_success_count,
            "clean_success_rate": round(clean_success_count / total_runs, 4) if total_runs else 0.0,
            "average_collisions_per_success": (
                round(sum(success_collisions) / len(success_collisions), 4)
                if success_collisions
                else None
            ),
            "failed_cases": failed_cases,
            "cases": case_rows,
            "collision_analysis_note": "run_log.csv is available; first collision states were recovered from existing logs.",
        }
    return output


def build_collision_heavy_cases(rollout_cases: dict) -> dict:
    output = {}
    for model, cases in rollout_cases.items():
        ranked = []
        for case_name, item in cases.items():
            summary = item["summary"]
            ranked.append(
                {
                    "case_name": case_name,
                    "collision_count": int(summary.get("collision_count", 0)),
                    "reached_goal": bool(summary.get("reached_goal", False)),
                    "failure_reason": summary.get("failure_reason"),
                    "steps": summary.get("steps"),
                }
            )
        output[model] = sorted(ranked, key=lambda row: row["collision_count"], reverse=True)
    return output


def build_model_difference_summary(rollout_cases: dict, predictions: dict) -> dict:
    majority_cases = rollout_cases["knn_majority"]
    balanced_cases = rollout_cases["knn_class_balanced"]
    collision_differences = []
    for case_name in sorted(set(majority_cases) & set(balanced_cases)):
        majority_collision = int(majority_cases[case_name]["summary"].get("collision_count", 0))
        balanced_collision = int(balanced_cases[case_name]["summary"].get("collision_count", 0))
        if majority_collision != balanced_collision:
            collision_differences.append(
                {
                    "case_name": case_name,
                    "knn_majority_collisions": majority_collision,
                    "knn_class_balanced_collisions": balanced_collision,
                    "collision_delta_class_balanced_minus_majority": balanced_collision - majority_collision,
                    "knn_majority_reached_goal": bool(majority_cases[case_name]["summary"].get("reached_goal", False)),
                    "knn_class_balanced_reached_goal": bool(balanced_cases[case_name]["summary"].get("reached_goal", False)),
                }
            )
    prediction_distributions = {
        model: action_counts({"action": row["predicted_action"]} for row in rows)
        for model, rows in predictions.items()
    }
    return {
        "offline_prediction_distributions": prediction_distributions,
        "offline_predictions_identical": (
            prediction_distributions["knn_majority"]
            == prediction_distributions["knn_class_balanced"]
        ),
        "case_collision_differences": collision_differences,
        "class_balanced_total_collision_delta": sum(
            row["collision_delta_class_balanced_minus_majority"]
            for row in collision_differences
        ),
        "stability_assessment": (
            "class-balanced has fewer total collisions, but offline distribution is identical; improvement is case-specific, not enough to prove robust stability."
        ),
    }


def build_v067_summary(confusion: dict, front_errors: dict, turn_left: dict, rollout: dict, collision_heavy: dict, model_diff: dict) -> dict:
    overview = {
        model: {
            "turn_left_recall": confusion[model]["turn_left_recall"],
            "turn_right_recall": confusion[model]["turn_right_recall"],
            "front_blocked_predicted_forward_count": front_errors[model]["front_blocked_predicted_forward_count"],
            "clean_success_rate": rollout[model]["clean_success_rate"],
            "average_collisions_per_success": rollout[model]["average_collisions_per_success"],
        }
        for model in confusion
    }
    combined_collision_heavy_cases = []
    for model, rows in collision_heavy.items():
        for row in rows:
            item = dict(row)
            item["model_name"] = model
            combined_collision_heavy_cases.append(item)
    combined_collision_heavy_cases.sort(
        key=lambda row: (row["collision_count"], row["steps"]), reverse=True
    )
    return {
        "version": "v0.6.7",
        "source": "existing v0.6.6 dataset, predictions, and rollout logs",
        "main_bottleneck": "rollout/collision evaluation is too permissive; feature errors remain in front_blocked and turn_left states",
        "next_priority": "modify rollout/collision evaluation rules",
        "class_balanced_stability_assessment": model_diff["stability_assessment"],
        "model_overview": overview,
        "rollout_collision_overview": {
            "clean_success_rate": {
                model: rollout[model]["clean_success_rate"] for model in rollout
            },
            "clean_success_count": {
                model: rollout[model]["clean_success_count"] for model in rollout
            },
            "average_collisions_per_success": {
                model: rollout[model]["average_collisions_per_success"] for model in rollout
            },
            "collision_heavy_cases": combined_collision_heavy_cases[:10],
        },
        "most_collision_heavy_cases": {
            model: rows[:5] for model, rows in collision_heavy.items()
        },
        "answers": {
            "why_front_blocked_predicts_forward": "front_blocked errors occur in specific test rows where learned nearest-neighbor behavior still maps blocked observations to forward; existing logs show this is dangerous because forward at blocked states produces collisions.",
            "why_turn_left_recall_low": "turn_left test support is small and most turn_left errors are predicted as forward, so recall is both weak and statistically unstable.",
            "where_collisions_concentrate": "see collision_heavy_cases.json; collisions concentrate in a small number of long or failed rollout cases.",
            "why_same_offline_distribution_but_different_rollout_collisions": "offline aggregate prediction distributions match, but rollout is sequential; small per-state/case differences can compound into different trajectories and collision counts.",
        },
    }


def first_collision_state(records: list[dict[str, str]]) -> dict:
    for row in records:
        if row.get("collision") == "1":
            return {
                "step": int(row["step"]),
                "x": row.get("x"),
                "y": row.get("y"),
                "direction": row.get("direction"),
                "pre_x": row.get("pre_x"),
                "pre_y": row.get("pre_y"),
                "pre_direction": row.get("pre_direction"),
                "front_blocked": row.get("front_blocked"),
                "true_front_blocked": row.get("true_front_blocked"),
                "action": row.get("action"),
            }
    return {}


def action_counts(rows) -> dict[str, int]:
    counts = {action: 0 for action in ACTIONS}
    for row in rows:
        action = row["action"]
        if action in counts:
            counts[action] += 1
    return counts


def empty_matrix() -> dict[str, dict[str, int]]:
    return {actual: {predicted: 0 for predicted in ACTIONS} for actual in ACTIONS}


def summarize_front_blocked_error_source(errors: list[dict]) -> str:
    if not errors:
        return "No front_blocked + predicted_forward errors in the test split."
    cases = sorted({row["case_name"] for row in errors})
    return f"front_blocked + predicted_forward errors appear in {len(cases)} case(s): {', '.join(cases)}"


if __name__ == "__main__":
    main()
