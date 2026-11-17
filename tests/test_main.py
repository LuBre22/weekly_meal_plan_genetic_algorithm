import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIMITS = [
    "--daily-min-protein", "0", "--daily-min-fat", "0", "--daily-min-carbs", "0",
    "--weekly-max-protein", "2000", "--weekly-max-fat", "2000", "--weekly-max-carbs", "2000",
    "--daily-min-calories", "0", "--weekly-max-calories", "20000",
]


class MainTests(unittest.TestCase):
    def test_main_runs_from_outside_project(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "main.py"), *LIMITS],
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Constraints satisfied; maximizing weekly protein.", result.stdout)
        self.assertIn("Weekly meal plan (fitness:", result.stdout)
        for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
            self.assertIn(day, result.stdout)
        self.assertIn("Weekly total:", result.stdout)
        self.assertIn("Daily average:", result.stdout)
        self.assertNotIn("Run statistics", result.stdout)

    def test_settings_can_be_adjusted_and_statistics_enabled(self):
        script = """
import main
main.POPULATION_SIZE = 10
main.ELITE_COUNT = 1
main.IMMIGRANT_COUNT = 2
main.TOURNAMENT_SIZE = 2
main.GENERATIONS = 3
main.MIN_EXPLORATION_GENERATIONS = 1
main.MAX_EXPLORATION_GENERATIONS = 2
main.INCLUDE_STATISTICS = True
raise SystemExit(main.main())
"""
        result = subprocess.run(
            [sys.executable, "-c", script, *LIMITS],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Run statistics", result.stdout)
        self.assertIn("Population size: 10", result.stdout)
        self.assertIn("Generations: 3", result.stdout)

    def test_infeasible_and_invalid_input_are_reported(self):
        impossible = [
            "--daily-min-protein", "0", "--daily-min-fat", "0", "--daily-min-carbs", "0",
            "--weekly-max-protein", "0", "--weekly-max-fat", "0", "--weekly-max-carbs", "0",
            "--daily-min-calories", "0", "--weekly-max-calories", "0",
        ]
        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *impossible],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("No feasible plan found", result.stdout)
        lines = result.stdout.splitlines()
        weekly_index = next(i for i, line in enumerate(lines) if line.startswith("Weekly total:"))
        self.assertTrue(lines[weekly_index + 1].startswith("Daily shortfalls (sum over week):"))
        self.assertRegex(lines[weekly_index + 2], r"Weekly excess: calories [\d.]+ kcal \| protein [\d.]+ g")
        self.assertTrue(lines[weekly_index + 3].startswith("Daily average:"))
        result = subprocess.run(
            [sys.executable, "-c", "import main; main.DAILY_MIN_CALORIES = None; main.main()"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required", result.stderr)

    def test_file_settings_work_and_cli_overrides_them(self):
        script = """
import main
main.DAILY_MIN_PROTEIN = 0
main.DAILY_MIN_FAT = 0
main.DAILY_MIN_CARBS = 0
main.WEEKLY_MAX_PROTEIN = 2000
main.WEEKLY_MAX_FAT = 2000
main.WEEKLY_MAX_CARBS = 2000
main.DAILY_MIN_CALORIES = 0
main.WEEKLY_MAX_CALORIES = 20000
main.POPULATION_SIZE = 10
main.ELITE_COUNT = 1
main.IMMIGRANT_COUNT = 2
main.TOURNAMENT_SIZE = 2
main.GENERATIONS = 3
main.MIN_EXPLORATION_GENERATIONS = 1
main.MAX_EXPLORATION_GENERATIONS = 2
raise SystemExit(main.main())
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Constraints satisfied", result.stdout)
        result = subprocess.run(
            [sys.executable, "-c", script, "--weekly-max-protein", "0"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertRegex(result.stdout, r"Weekly excess: calories 0 kcal \| protein [1-9]\d*(?:\.\d+)? g")

    def test_partial_file_settings_can_be_combined_with_cli(self):
        script = """
import main
main.DAILY_MIN_PROTEIN = 0
main.DAILY_MIN_FAT = 0
main.DAILY_MIN_CARBS = 0
main.DAILY_MIN_CALORIES = 0
main.POPULATION_SIZE = 10
main.ELITE_COUNT = 1
main.IMMIGRANT_COUNT = 2
main.TOURNAMENT_SIZE = 2
main.GENERATIONS = 3
main.MIN_EXPLORATION_GENERATIONS = 1
main.MAX_EXPLORATION_GENERATIONS = 2
raise SystemExit(main.main())
"""
        result = subprocess.run(
            [sys.executable, "-c", script, *LIMITS[6:]],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Constraints satisfied", result.stdout)

    def test_calorie_violations_report_amounts_in_kcal(self):
        limits = [
            "--daily-min-protein", "0", "--daily-min-fat", "0", "--daily-min-carbs", "0",
            "--weekly-max-protein", "2000", "--weekly-max-fat", "2000", "--weekly-max-carbs", "2000",
            "--daily-min-calories", "5000", "--weekly-max-calories", "35000",
        ]
        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *limits],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"Daily shortfalls \(sum over week\): calories [1-9][\d.]* kcal")
        limits[-3:] = ["0", "--weekly-max-calories", "0"]
        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *limits],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertRegex(result.stdout, r"Weekly excess: calories [1-9][\d.]* kcal")


if __name__ == "__main__":
    unittest.main()
