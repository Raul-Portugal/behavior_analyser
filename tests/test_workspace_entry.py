"""gui_main: the workspace is the default window, --classic keeps the old one."""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import gui_main
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


class FakeWindow:
    instances = []

    def __init__(self, *args):
        type(self).instances.append(self)
        self.calls = []

    def show(self):
        self.calls.append("show")

    def close(self):
        self.calls.append("close")

    def schedule_start(self, path):
        self.calls.append(("schedule_start", path))


class FakeWorkspace(FakeWindow):
    instances = []


class FakeClassic(FakeWindow):
    instances = []


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class EntryPointTests(unittest.TestCase):
    def setUp(self):
        FakeWorkspace.instances = []
        FakeClassic.instances = []
        self.app = MagicMock()
        self.app.exec.return_value = 7
        patches = [
            patch.object(gui_main, "QApplication", return_value=self.app),
            patch.object(gui_main, "WorkspaceWindow", FakeWorkspace),
            patch.object(gui_main, "MainWindow", FakeClassic),
            patch.object(sys, "excepthook", sys.excepthook),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def run_main(self, *extra):
        return gui_main.main(["gui_main.py", *extra])

    def test_default_is_the_workspace(self):
        self.assertEqual(self.run_main(), 7)
        self.assertEqual(len(FakeWorkspace.instances), 1)
        self.assertEqual(FakeClassic.instances, [])
        self.assertEqual(
            FakeWorkspace.instances[0].calls, ["show", ("schedule_start", None)]
        )

    def test_classic_flag_opens_the_classic_window(self):
        self.assertEqual(self.run_main("--classic"), 7)
        self.assertEqual(FakeWorkspace.instances, [])
        self.assertEqual(FakeClassic.instances[0].calls, ["show"])

    def test_flags_are_not_passed_to_qt(self):
        self.run_main("--classic")
        (qt_arguments,), _ = gui_main.QApplication.call_args
        self.assertEqual(qt_arguments, ["gui_main.py"])

    def test_a_project_path_on_the_command_line_is_opened(self):
        project = PROJECT_ROOT / "readme.md"      # any existing file
        self.run_main(str(project))
        self.assertEqual(
            FakeWorkspace.instances[0].calls[-1], ("schedule_start", project)
        )

    def test_a_missing_path_is_ignored(self):
        self.run_main(str(PROJECT_ROOT / "no" / "such.bmaproj"))
        self.assertEqual(
            FakeWorkspace.instances[0].calls[-1], ("schedule_start", None)
        )

    def test_startup_check_builds_and_closes_both_windows(self):
        self.assertEqual(self.run_main("--startup-check"), 0)
        self.assertEqual(FakeWorkspace.instances[0].calls, ["close"])
        self.assertEqual(FakeClassic.instances[0].calls, ["close"])
        self.app.exec.assert_not_called()

    def test_startup_check_with_classic_still_checks_both(self):
        self.assertEqual(self.run_main("--startup-check", "--classic"), 0)
        self.assertEqual(len(FakeWorkspace.instances), 1)
        self.assertEqual(len(FakeClassic.instances), 1)

    def test_path_argument_helper(self):
        self.assertIsNone(gui_main._path_argument(["-platform", "offscreen"]))
        existing = str(PROJECT_ROOT / "readme.md")
        self.assertEqual(
            gui_main._path_argument(["--x", existing]), Path(existing)
        )


class StartupCheckProcessTests(unittest.TestCase):
    def run_check(self, *flags):
        return subprocess.run(
            [sys.executable, "-B", "gui_main.py", "--startup-check", *flags],
            cwd=PROJECT_ROOT,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True, text=True, timeout=300,
        )

    def test_the_real_startup_check_exits_cleanly(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
