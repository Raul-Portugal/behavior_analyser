"""Run manifest, schema 3: per-video hashes, status, metadata and summary."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from checkpoint_manager import CheckpointManager
    from core.models import BatchSettings
    from core.run_outputs import (
        MANIFEST_SCHEMA_VERSION, RUN_PLAN_FILENAME, write_run_manifest,
    )
    from core.sample_sheet import load_sample_sheet, metadata_for_videos
    from gui.worker import AnalysisWorker
    from mazes.freestyle import Freestyle
    from tests import worker_support as support
    from tests.worker_support import FRAME_COUNTS
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

FREESTYLE = "Freestyle / Open Field"


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class ManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.video_dir = self.root / "videos"
        self.video_dir.mkdir()
        self.videos = [
            support.write_tracking_video(self.video_dir, frames, name).resolve()
            for name, frames in FRAME_COUNTS.items()
        ]

    def _run(self, label, videos=None, **options):
        out = self.root / label
        out.mkdir()
        plan = {}
        for video in videos or self.videos:
            settings = support.tracking_settings(create_timelapse=False)
            settings.paradigm_id = FREESTYLE
            settings.algorithm_parameters = Freestyle().get_algorithm_parameters()
            plan[video] = settings
        cache = str(self.root / f"cache_{label}")
        with patch.dict(os.environ, {"LOCALAPPDATA": cache,
                                     "XDG_CACHE_HOME": cache}):
            worker = AnalysisWorker(plan, out, Freestyle(), **options)
            outcomes = []
            worker.finished.connect(outcomes.append)
            worker.run()
        manifest = json.loads(
            (worker.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        return worker, outcomes[0], manifest

    def test_public_hash_helpers_keep_their_private_aliases(self):
        manager = CheckpointManager(self.root / "ckpt", create_dir=False)
        settings = support.tracking_settings()
        self.assertEqual(
            manager.get_settings_hash(settings),
            manager._get_settings_hash(settings),
        )
        video = self.videos[0]
        self.assertEqual(
            CheckpointManager.get_video_fingerprint(video),
            CheckpointManager._get_video_fingerprint(video),
        )

    def test_a_completed_run_records_hashes_status_and_summary_paths(self):
        worker, outcome, manifest = self._run("plain")
        self.assertEqual(outcome.failures, [])

        self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 3)
        self.assertIsNone(manifest["sample_sheet"])
        self.assertIsNone(manifest["time_bin_seconds"])
        self.assertEqual(
            [Path(v["source"]) for v in manifest["videos"]], self.videos
        )
        hasher = CheckpointManager(self.root / "plain", create_dir=False)
        for entry, video in zip(manifest["videos"], self.videos):
            self.assertEqual(entry["status"], "completed")
            self.assertEqual(entry["output_folder"], video.stem)
            self.assertEqual(
                entry["settings_hash"],
                hasher.get_settings_hash(worker.analysis_plan[video]),
            )
            self.assertEqual(
                entry["video_fingerprint"],
                CheckpointManager.get_video_fingerprint(video),
            )
            self.assertEqual(entry["metadata"], {})
            self.assertEqual(
                entry["summary_json"],
                f"{video.stem}/data/{video.stem}_summary.json",
            )
            self.assertTrue(
                (worker.output_dir / entry["summary_json"]).is_file()
            )

    def test_settings_hash_matches_the_reloaded_run_plan(self):
        worker, _, manifest = self._run("roundtrip")
        plan = BatchSettings.load_batch_plan(worker.output_dir / RUN_PLAN_FILENAME)
        hasher = CheckpointManager(self.root / "roundtrip", create_dir=False)
        hashes = {hasher.get_settings_hash(settings) for settings in plan.values()}
        self.assertEqual(
            {entry["settings_hash"] for entry in manifest["videos"]}, hashes
        )

    def test_failed_videos_are_listed_and_stay_in_the_run_plan(self):
        broken = self.video_dir / "broken.avi"
        broken.write_bytes(b"not a video")
        videos = [self.videos[0], broken.resolve(), self.videos[1]]
        worker, outcome, manifest = self._run("failed", videos=videos)

        self.assertEqual(len(outcome.failures), 1)
        entries = {Path(v["source"]).name: v for v in manifest["videos"]}
        self.assertEqual(
            [Path(v["source"]).name for v in manifest["videos"]],
            ["mouse_a.avi", "broken.avi", "mouse_b.avi"],
        )
        failed = entries["broken.avi"]
        self.assertEqual(failed["status"], "failed")
        self.assertIsNotNone(failed["settings_hash"])
        self.assertIsNone(failed["summary_json"])
        self.assertEqual(entries["mouse_a.avi"]["status"], "completed")
        self.assertEqual(manifest["completed_videos"], 2)
        self.assertEqual(len(manifest["failures"]), 1)

        # The failed video can be re-run from the run plan: its entry is
        # there and hashes to the value the manifest records.
        plan = BatchSettings.load_batch_plan(worker.output_dir / RUN_PLAN_FILENAME)
        self.assertEqual(len(plan), 3)
        self.assertIn("broken.avi", " ".join(plan))
        hasher = CheckpointManager(self.root / "failed", create_dir=False)
        key = next(key for key in plan if key.endswith("broken.avi"))
        self.assertEqual(
            hasher.get_settings_hash(plan[key]), failed["settings_hash"]
        )

    def test_metadata_sheet_and_bin_length_are_recorded(self):
        sheet_path = self.root / "sheet.csv"
        sheet_path.write_text(
            "video,animal_id\n"
            + "".join(
                f"videos/{v.name},A{n}\n" for n, v in enumerate(self.videos, 1)
            ),
            encoding="utf-8",
        )
        metadata = metadata_for_videos(
            load_sample_sheet(sheet_path), sheet_path, self.videos
        ).metadata
        _, _, manifest = self._run(
            "meta", sample_metadata=metadata, bin_seconds=1.5
        )
        self.assertEqual(Path(manifest["sample_sheet"]), sheet_path)
        self.assertEqual(manifest["time_bin_seconds"], 1.5)
        self.assertEqual(
            [v["metadata"] for v in manifest["videos"]],
            [{"animal_id": "A1"}, {"animal_id": "A2"}, {"animal_id": "A3"}],
        )

    def test_without_a_plan_or_hasher_the_new_fields_are_null(self):
        run_dir = self.root / "bare"
        run_dir.mkdir()
        video = self.root / "gone.avi"
        write_run_manifest(
            run_dir, Freestyle(), "start", 1, {video: "gone"},
            [], [(video, "boom")], [], False,
        )
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        (entry,) = manifest["videos"]
        self.assertEqual(entry["status"], "failed")
        self.assertIsNone(entry["settings_hash"])
        self.assertIsNone(entry["video_fingerprint"])
        self.assertIsNone(entry["summary_json"])


if __name__ == "__main__":
    unittest.main()
