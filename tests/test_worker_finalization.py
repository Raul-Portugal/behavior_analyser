import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

try:
    from gui.worker import AnalysisWorker
    from mazes.epm import EPM
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class WorkerFinalizationTests(unittest.TestCase):
    def test_empty_run_writes_versioned_manifest_and_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = AnalysisWorker({}, Path(directory), EPM())
            outcomes = []
            worker.finished.connect(outcomes.append)

            worker.run()

            self.assertEqual(len(outcomes), 1)
            manifest = json.loads(
                (worker.output_dir / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["application"]["version"], "3.0.0")

    def test_manifest_failure_is_reported_without_losing_finished_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = AnalysisWorker({}, Path(directory), EPM())
            worker._write_run_manifest = Mock(
                side_effect=OSError("simulated manifest failure")
            )
            outcomes = []
            worker.finished.connect(outcomes.append)

            worker.run()

            self.assertEqual(len(outcomes), 1)
            self.assertIn(
                "simulated manifest failure",
                outcomes[0].failures[0][1],
            )


if __name__ == "__main__":
    unittest.main()
