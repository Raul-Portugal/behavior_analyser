import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from gui.worker import AnalysisWorker
    from mazes.epm import EPM
    from core.models import BatchSettings, DetectionConfig
    from core.version import APP_VERSION
    from roi_manager import ROIManager
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
            self.assertEqual(manifest["application"]["version"], APP_VERSION)

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

    def test_output_directory_failure_is_a_per_video_failure(self):
        real_mkdir = Path.mkdir

        def failing_mkdir(path, *args, **kwargs):
            if path.name == "data":
                raise OSError("simulated mkdir failure")
            return real_mkdir(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "missing_video.mp4"
            settings = BatchSettings(
                ROIManager(source_dimensions=(100, 100)),
                DetectionConfig(),
                0.0,
            )
            worker = AnalysisWorker(
                {video_path: settings}, Path(directory), EPM()
            )
            outcomes = []
            worker.finished.connect(outcomes.append)

            with patch.object(Path, "mkdir", failing_mkdir):
                worker.run()

            self.assertEqual(len(outcomes), 1)
            self.assertEqual(len(outcomes[0].failures), 1)
            failed_path, message = outcomes[0].failures[0]
            self.assertEqual(failed_path, video_path)
            self.assertIn("simulated mkdir failure", message)
            self.assertEqual(outcomes[0].results, [])
            self.assertTrue((worker.output_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
