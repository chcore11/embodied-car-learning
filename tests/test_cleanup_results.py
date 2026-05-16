import json
import tempfile
import unittest
from pathlib import Path

from scripts.cleanup_results import apply_cleanup, build_cleanup_plan


class CleanupResultsTest(unittest.TestCase):
    def test_dry_run_does_not_delete_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            self._write_case(results_dir, "success_case", "success")

            plan = build_cleanup_plan(results_dir)

            self.assertEqual(len(plan["files_to_delete"]), 2)
            self.assertTrue((results_dir / "success_case" / "run_log.csv").exists())
            self.assertTrue((results_dir / "success_case" / "trajectory.png").exists())

    def test_apply_deletes_success_case_log_and_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            self._write_case(results_dir, "success_case", "success")

            plan = build_cleanup_plan(results_dir)
            apply_cleanup(plan)

            self.assertFalse((results_dir / "success_case" / "run_log.csv").exists())
            self.assertFalse((results_dir / "success_case" / "trajectory.png").exists())
            self.assertTrue((results_dir / "success_case" / "summary.json").exists())

    def test_policy_failed_case_files_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            self._write_case(results_dir, "policy_failed_case", "policy_failed")

            plan = build_cleanup_plan(results_dir)

            self.assertEqual(plan["files_to_delete"], [])
            keep_paths = self._paths(plan["files_to_keep"])
            self.assertIn(str(results_dir / "policy_failed_case" / "run_log.csv"), keep_paths)
            self.assertIn(str(results_dir / "policy_failed_case" / "trajectory.png"), keep_paths)

    def test_expected_unreachable_case_files_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            self._write_case(results_dir, "unreachable_case", "expected_unreachable")

            plan = build_cleanup_plan(results_dir)

            self.assertEqual(plan["files_to_delete"], [])
            keep_paths = self._paths(plan["files_to_keep"])
            self.assertIn(str(results_dir / "unreachable_case" / "run_log.csv"), keep_paths)
            self.assertIn(str(results_dir / "unreachable_case" / "trajectory.png"), keep_paths)

    def test_case_missing_summary_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            case_dir = results_dir / "missing_summary"
            case_dir.mkdir(parents=True)
            (case_dir / "run_log.csv").write_text("log", encoding="utf-8")
            (case_dir / "trajectory.png").write_bytes(b"png")

            plan = build_cleanup_plan(results_dir)

            self.assertEqual(plan["files_to_delete"], [])
            reasons = {item["reason"] for item in plan["files_to_keep"]}
            self.assertIn("missing summary.json", reasons)

    def test_overall_summary_is_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            self._write_case(results_dir, "success_case", "success")
            (results_dir / "overall_summary.json").write_text("{}", encoding="utf-8")

            plan = build_cleanup_plan(results_dir)
            apply_cleanup(plan)

            self.assertTrue((results_dir / "overall_summary.json").exists())

    def test_curated_directory_contents_are_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            self._write_case(results_dir, "success_case", "success")
            curated_dir = results_dir / "success_case" / "curated"
            curated_dir.mkdir()
            curated_file = curated_dir / "trajectory.png"
            curated_file.write_bytes(b"curated")

            plan = build_cleanup_plan(results_dir)
            apply_cleanup(plan)

            self.assertTrue(curated_file.exists())

    def _write_case(self, results_dir: Path, case_name: str, result_type: str) -> None:
        case_dir = results_dir / case_name
        case_dir.mkdir(parents=True)
        (case_dir / "summary.json").write_text(
            json.dumps({"case_name": case_name, "result_type": result_type}),
            encoding="utf-8",
        )
        (case_dir / "run_log.csv").write_text("step,event\n0,move\n", encoding="utf-8")
        (case_dir / "trajectory.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    def _paths(self, entries: list[dict[str, str]]) -> set[str]:
        return {item["path"] for item in entries}


if __name__ == "__main__":
    unittest.main()
