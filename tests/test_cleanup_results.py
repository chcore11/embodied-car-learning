import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.cleanup_results import apply_cleanup, build_cleanup_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CleanupResultsTest(unittest.TestCase):
    def test_dry_run_does_not_delete_csv_or_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiments_dir = Path(tmp_dir) / "experiments"
            csv_path = experiments_dir / "v0_2" / "results" / "case_a" / "run_log.csv"
            png_path = experiments_dir / "v0_2" / "results" / "case_a" / "trajectory.png"
            self._write_file(csv_path, "csv")
            self._write_file(png_path, "png")

            plan = build_cleanup_plan(experiments_dir)

            self.assertEqual(len(plan["files_to_delete"]), 2)
            self.assertTrue(csv_path.exists())
            self.assertTrue(png_path.exists())

    def test_apply_deletes_csv_under_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiments_dir = Path(tmp_dir) / "experiments"
            csv_path = experiments_dir / "v0_2" / "results" / "case_a" / "run_log.csv"
            self._write_file(csv_path, "csv")

            apply_cleanup(build_cleanup_plan(experiments_dir))

            self.assertFalse(csv_path.exists())

    def test_apply_deletes_png_under_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiments_dir = Path(tmp_dir) / "experiments"
            png_path = experiments_dir / "v0_2" / "results" / "case_a" / "trajectory.png"
            self._write_file(png_path, "png")

            apply_cleanup(build_cleanup_plan(experiments_dir))

            self.assertFalse(png_path.exists())

    def test_apply_keeps_json_under_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiments_dir = Path(tmp_dir) / "experiments"
            summary_path = experiments_dir / "v0_2" / "results" / "case_a" / "summary.json"
            overall_path = experiments_dir / "v0_2" / "results" / "overall_summary.json"
            self._write_file(summary_path, "{}")
            self._write_file(overall_path, "{}")

            apply_cleanup(build_cleanup_plan(experiments_dir))

            self.assertTrue(summary_path.exists())
            self.assertTrue(overall_path.exists())

    def test_apply_keeps_curated_csv_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiments_dir = Path(tmp_dir) / "experiments"
            curated_csv = experiments_dir / "v0_2" / "results" / "case_a" / "curated" / "run_log.csv"
            curated_png = experiments_dir / "v0_2" / "results" / "case_a" / "curated" / "trajectory.png"
            self._write_file(curated_csv, "csv")
            self._write_file(curated_png, "png")

            apply_cleanup(build_cleanup_plan(experiments_dir))

            self.assertTrue(curated_csv.exists())
            self.assertTrue(curated_png.exists())

    def test_non_experiments_csv_and_png_are_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            experiments_dir = root / "experiments"
            outside_csv = root / "data" / "run_log.csv"
            outside_png = root / "data" / "trajectory.png"
            self._write_file(outside_csv, "csv")
            self._write_file(outside_png, "png")

            apply_cleanup(build_cleanup_plan(experiments_dir))

            self.assertTrue(outside_csv.exists())
            self.assertTrue(outside_png.exists())

    def test_v01_and_v02_scripts_still_run(self) -> None:
        v01 = subprocess.run(
            [sys.executable, "scripts/run_v01.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        v02 = subprocess.run(
            [sys.executable, "scripts/run_v02_experiments.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("reached_goal: True", v01.stdout)
        self.assertIn("total_runs: 3", v02.stdout)

    def _write_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
