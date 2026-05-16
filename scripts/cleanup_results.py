from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path("experiments/v0_2/results")
PROTECTED_FILENAMES = {"summary.json", "overall_summary.json"}


def build_cleanup_plan(results_dir: Path) -> dict[str, list[dict[str, str]]]:
    files_to_delete: list[dict[str, str]] = []
    files_to_keep: list[dict[str, str]] = []
    dirs_to_delete: list[dict[str, str]] = []
    dirs_to_keep: list[dict[str, str]] = []

    if not results_dir.exists():
        return {
            "files_to_delete": files_to_delete,
            "files_to_keep": files_to_keep,
            "dirs_to_delete": dirs_to_delete,
            "dirs_to_keep": dirs_to_keep,
        }

    overall_summary = results_dir / "overall_summary.json"
    if overall_summary.exists():
        files_to_keep.append(_entry(overall_summary, "protected overall_summary.json"))

    for child in sorted(results_dir.iterdir()):
        if child.name == "overall_summary.json":
            continue
        if _is_curated(child):
            _keep_tree(child, files_to_keep, dirs_to_keep, "curated")
            continue
        if child.is_file():
            files_to_keep.append(_entry(child, "top-level file"))
            continue
        if not child.is_dir():
            continue

        case_files = sorted(path for path in child.rglob("*") if path.is_file())
        case_dirs = sorted(
            (path for path in child.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )

        if not case_files and not case_dirs:
            dirs_to_delete.append(_entry(child, "empty directory"))
            continue

        summary_path = child / "summary.json"
        if not summary_path.exists():
            _keep_case_tree(
                child,
                case_files,
                case_dirs,
                files_to_keep,
                dirs_to_keep,
                "missing summary.json",
            )
            continue

        summary = _read_json(summary_path)
        result_type = summary.get("result_type") if isinstance(summary, dict) else None
        if not result_type:
            _keep_case_tree(
                child,
                case_files,
                case_dirs,
                files_to_keep,
                dirs_to_keep,
                "missing result_type",
            )
            continue

        for file_path in case_files:
            if _is_curated(file_path):
                files_to_keep.append(_entry(file_path, "curated"))
            elif file_path.name in PROTECTED_FILENAMES:
                files_to_keep.append(_entry(file_path, f"protected {file_path.name}"))
            elif result_type == "success" and file_path.name in {"run_log.csv", "trajectory.png"}:
                files_to_delete.append(_entry(file_path, "success case artifact"))
            else:
                files_to_keep.append(_entry(file_path, f"result_type={result_type}"))

        for dir_path in case_dirs:
            if _is_curated(dir_path):
                dirs_to_keep.append(_entry(dir_path, "curated"))
            elif not any(dir_path.iterdir()):
                dirs_to_delete.append(_entry(dir_path, "empty directory"))

    return {
        "files_to_delete": files_to_delete,
        "files_to_keep": files_to_keep,
        "dirs_to_delete": dirs_to_delete,
        "dirs_to_keep": dirs_to_keep,
    }


def apply_cleanup(plan: dict[str, list[dict[str, str]]]) -> None:
    for item in plan["files_to_delete"]:
        path = Path(item["path"])
        if path.exists() and path.is_file():
            path.unlink()

    for item in sorted(
        plan["dirs_to_delete"],
        key=lambda entry: len(Path(entry["path"]).parts),
        reverse=True,
    ):
        path = Path(item["path"])
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def format_plan(plan: dict[str, list[dict[str, str]]], apply: bool) -> str:
    mode = "apply" if apply else "dry-run"
    lines = [
        f"Mode: {mode}",
        "No files were deleted" if not apply else "Apply mode completed",
        "files_to_delete:",
    ]
    lines.extend(_format_entries(plan["files_to_delete"]))
    lines.append("files_to_keep:")
    lines.extend(_format_entries(plan["files_to_keep"]))
    lines.append("empty_dirs_to_delete:")
    lines.extend(_format_entries(plan["dirs_to_delete"]))
    lines.append(f"total_delete_count: {len(plan['files_to_delete']) + len(plan['dirs_to_delete'])}")
    lines.append(f"total_keep_count: {len(plan['files_to_keep']) + len(plan['dirs_to_keep'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely clean low-value v0.2 experiment results.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files from the cleanup plan.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Experiment results directory to scan.",
    )
    args = parser.parse_args()

    plan = build_cleanup_plan(args.results_dir)
    if args.apply:
        apply_cleanup(plan)
    print(format_plan(plan, apply=args.apply))


def _keep_tree(
    root: Path,
    files_to_keep: list[dict[str, str]],
    dirs_to_keep: list[dict[str, str]],
    reason: str,
) -> None:
    if root.is_file():
        files_to_keep.append(_entry(root, reason))
        return
    dirs_to_keep.append(_entry(root, reason))
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files_to_keep.append(_entry(path, reason))
        elif path.is_dir():
            dirs_to_keep.append(_entry(path, reason))


def _keep_case_tree(
    case_dir: Path,
    case_files: list[Path],
    case_dirs: list[Path],
    files_to_keep: list[dict[str, str]],
    dirs_to_keep: list[dict[str, str]],
    reason: str,
) -> None:
    dirs_to_keep.append(_entry(case_dir, reason))
    for file_path in case_files:
        files_to_keep.append(_entry(file_path, reason if not _is_curated(file_path) else "curated"))
    for dir_path in case_dirs:
        dirs_to_keep.append(_entry(dir_path, reason if not _is_curated(dir_path) else "curated"))


def _format_entries(entries: list[dict[str, str]]) -> list[str]:
    if not entries:
        return ["  - none"]
    return [f"  - {item['path']} | reason: {item['reason']}" for item in entries]


def _entry(path: Path, reason: str) -> dict[str, str]:
    return {"path": str(path), "reason": reason}


def _is_curated(path: Path) -> bool:
    return "curated" in path.parts


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    main()
