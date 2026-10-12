"""The hidden --selftest-parallel check used to verify packaged builds."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ParallelSelfTestTests(unittest.TestCase):
    def test_gui_main_selftest_runs_a_parallel_batch_and_reports(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            report = Path(tmp) / "report.txt"
            result = subprocess.run(
                [sys.executable, "-B", "gui_main.py",
                 "--selftest-parallel", str(report)],
                cwd=PROJECT_ROOT,
                env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                capture_output=True, text=True, timeout=300,
            )
            text = report.read_text(encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(text.startswith("selftest-parallel: OK"), text)
        self.assertIn("results=2", text)


if __name__ == "__main__":
    unittest.main()
