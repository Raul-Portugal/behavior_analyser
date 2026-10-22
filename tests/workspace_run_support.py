"""Helpers for the Run / Results tab tests: real synthetic videos in a project."""
import contextlib
import io
import os
import time
from pathlib import Path
from typing import Callable, List
from unittest.mock import patch

from tests import worker_support as support
from tests.workspace_support import HAS_RUNTIME, WorkspaceFixture

if HAS_RUNTIME:
    from PyQt6.QtCore import QCoreApplication, QThread
    from core import cli
    from core.models import BatchSettings
    from gui.workspace import run_controller
    from mazes.freestyle import Freestyle

FREESTYLE = "Freestyle / Open Field"


def freestyle_settings(create_timelapse: bool = True) -> "BatchSettings":
    settings = support.tracking_settings(create_timelapse=create_timelapse)
    settings.paradigm_id = FREESTYLE
    settings.algorithm_parameters = Freestyle().get_algorithm_parameters()
    return settings


def wait_for(predicate: Callable[[], bool], timeout: float = 90.0,
             what: str = "the condition") -> None:
    """Run the Qt event loop until ``predicate()`` holds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"Timed out waiting for {what}")
        QCoreApplication.processEvents()
        QThread.msleep(5)
    QCoreApplication.processEvents()


class RunFixture(WorkspaceFixture):
    """A workspace with a saved project of three real synthetic videos."""

    FRAMES = support.FRAME_COUNTS         # name -> frames

    def setUp(self):
        super().setUp()
        self.addCleanup(self.drain_threads)
        self.real_dir = self.root / "real"
        self.real_dir.mkdir()
        self.real_videos: List[Path] = [
            support.write_tracking_video(self.real_dir, frames, name).resolve()
            for name, frames in self.FRAMES.items()
        ]
        self.new_project_at()
        self.project = self.window.project
        self.add_videos(self.real_videos)
        self.project.set_template(freestyle_settings())
        self.review = self.window.stage("review")
        self.results = self.window.stage("results")
        self.review.controller.FORCE_QUIT_DELAY_MS = 60_000
        self.review.set_output_dir(self.root / "ws_out")
        # Quiet defaults: one process, so the tests are fast and deterministic.
        self.project.set_run_options(parallel_workers=1)
        self.window.go_to_stage("review")
        self.addCleanup(self.results.shutdown)

    def drain_threads(self):
        stage = getattr(self, "review", None)
        if stage is not None and stage.is_running:
            stage.controller._cancel_worker()
        wait_for(lambda: not run_controller._live_threads, 60.0,
                 "the run thread to end")

    # -- helpers ---------------------------------------------------------
    def select(self, *names):
        keys = [v.key for v in self.project.videos if v.name in names]
        self.window.video_panel.select_keys(keys)
        self.window.video_panel.flush_selection()
        return keys

    def start_and_wait(self, accept_warnings: bool = True, timeout: float = 120.0):
        """Start the run (accepting the pre-flight warnings) and wait for it."""
        if accept_warnings:
            self.prompts.say("confirm", True)
        finished = []
        self.review.run_finished.connect(finished.append)
        self.assertTrue(self.review.start_run())
        wait_for(lambda: bool(finished), timeout, "the run to finish")
        return finished[0]

    def isolated_environment(self, label: str):
        cache = str(self.root / f"cache_{label}")
        return patch.dict(
            os.environ, {"LOCALAPPDATA": cache, "XDG_CACHE_HOME": cache}
        )

    def run_cli(self, label: str, *extra, settings_for=None) -> Path:
        """The same videos and settings through the headless CLI.

        ``settings_for(video_path)`` gives the plan entry of a video (default:
        the template of the fixture).
        """
        plan = self.root / f"{label}_plan.json"
        BatchSettings.save_batch_plan(
            {f"real/{video.name}": (
                freestyle_settings() if settings_for is None
                else settings_for(video)
            )
             for video in self.real_videos},
            plan,
        )
        out = self.root / f"{label}_out"
        with self.isolated_environment(label), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main([
                "run", "--plan", str(plan), "--out", str(out), "--quiet", *extra,
            ])
        self.assertEqual(code, cli.EXIT_OK)
        (run_dir,) = sorted(out.glob("analysis_run_*"))
        return run_dir


SHEET_TEXT = (
    "video,animal_id,group\n"
    "mouse_a.avi,1,ctl\nmouse_b.avi,2,ctl\nmouse_c.avi,3,trt\n"
)


def make_cli_run(root: Path, bin_seconds: float = 1.0,
                 timelapse: bool = True, sheet: bool = True) -> Path:
    """A real run folder made by the CLI from three synthetic videos.

    The videos are in ``root/real``; returns the ``analysis_run_*`` folder.
    """
    root = Path(root)
    real = root / "real"
    real.mkdir(parents=True, exist_ok=True)
    for name, frames in support.FRAME_COUNTS.items():
        support.write_tracking_video(real, frames, name)
    plan = root / "plan.json"
    BatchSettings.save_batch_plan(
        {f"real/{name}": freestyle_settings(timelapse)
         for name in support.FRAME_COUNTS},
        plan,
    )
    args = ["run", "--plan", str(plan), "--out", str(root / "cli_out"),
            "--quiet", "--workers", "1"]
    if bin_seconds:
        args += ["--bin-seconds", str(bin_seconds)]
    if sheet:
        (root / "sheet.csv").write_text(SHEET_TEXT, encoding="utf-8")
        args += ["--sample-sheet", str(root / "sheet.csv")]
    cache = str(root / "cache")
    with patch.dict(os.environ, {"LOCALAPPDATA": cache, "XDG_CACHE_HOME": cache}), \
            contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        code = cli.main(args)
    if code != cli.EXIT_OK:
        raise RuntimeError(f"The CLI run failed with exit code {code}")
    (run_dir,) = sorted((root / "cli_out").glob("analysis_run_*"))
    return run_dir
