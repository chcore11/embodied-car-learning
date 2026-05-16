from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_RESULTS_DIR = Path("experiments")
DELETE_SUFFIXES = {".csv", ".png"}


def build_cleanup_plan(results_dir: Path) -> dict[str, list[dict[str, str]]]:
    files_to_delete: list[dict[str, str]] = []
    files_to_keep: list[dict[str, str]] = []

    if not results_dir.exists():
        return {"files_to_delete": files_to_delete, "files_to_keep": files_to_keep}

    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue

        if _is_curated(path):
            files_to_keep.append(_entry(path, "curated"))
        elif path.suffix.lower() in DELETE_SUFFIXES:
            files_to_delete.append(_entry(path, f"{path.suffix.lower()} artifact"))
        else:
            files_to_keep.append(_entry(path, "not csv/png artifact"))

    return {"files_to_delete": files_to_delete, "files_to_keep": files_to_keep}


def apply_cleanup(plan: dict[str, list[dict[str, str]]]) -> None:
    for item in plan["files_to_delete"]:
        path = Path(item["path"])
        if path.exists() and path.is_file():
            path.unlink()


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
    lines.append(f"total_delete_count: {len(plan['files_to_delete'])}")
    lines.append(f"total_keep_count: {len(plan['files_to_keep'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely clean low-value experiment artifacts.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files from the cleanup plan.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Experiment directory to scan.",
    )
    args = parser.parse_args()

    plan = build_cleanup_plan(args.results_dir)
    if args.apply:
        apply_cleanup(plan)
    print(format_plan(plan, apply=args.apply))


def _format_entries(entries: list[dict[str, str]]) -> list[str]:
    if not entries:
        return ["  - none"]
    return [f"  - {item['path']} | reason: {item['reason']}" for item in entries]


def _entry(path: Path, reason: str) -> dict[str, str]:
    return {"path": str(path), "reason": reason}


def _is_curated(path: Path) -> bool:
    return "curated" in path.parts


if __name__ == "__main__":
    main()
