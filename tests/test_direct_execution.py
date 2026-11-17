import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DirectExecutionTests(unittest.TestCase):
    def test_test_files_run_directly_from_outside_project(self):
        test_directory = Path(__file__).resolve().parent
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as working_directory:
            for name in ("test_strategy.py", "test_genetic_algorithm.py", "test_report.py"):
                with self.subTest(file=name):
                    result = subprocess.run(
                        [sys.executable, str(test_directory / name)],
                        cwd=working_directory,
                        env=environment,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("OK", result.stderr)


if __name__ == "__main__":
    unittest.main()
