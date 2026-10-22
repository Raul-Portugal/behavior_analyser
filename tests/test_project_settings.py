"""Recent projects and the last project in AppSettings."""
import tempfile
import unittest
from pathlib import Path

try:
    from PyQt6.QtWidgets import QApplication
    from gui.app_settings import MAX_RECENT_PROJECTS, AppSettings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RecentProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.ini = self.root / "settings.ini"
        self.settings = AppSettings.for_file(self.ini)

    def _project(self, name: str) -> Path:
        path = self.root / f"{name}.bmaproj"
        path.write_text("{}", encoding="utf-8")
        return path

    def _reopen(self) -> AppSettings:
        self.settings.sync()
        return AppSettings.for_file(self.ini)

    def test_the_limit_is_eight(self):
        self.assertEqual(MAX_RECENT_PROJECTS, 8)

    def test_recent_projects_are_most_recent_first_and_deduplicated(self):
        a, b, c = (self._project(name) for name in "abc")
        for path in (a, b, c, a):
            self.settings.add_recent_project(path)
        self.assertEqual(self.settings.recent_projects(), [a, c, b])
        self.assertEqual(self._reopen().recent_projects(), [a, c, b])

    def test_only_eight_are_kept(self):
        paths = [self._project(f"p{n}") for n in range(11)]
        for path in paths:
            self.settings.add_recent_project(path)
        self.assertEqual(
            self.settings.recent_projects(), list(reversed(paths))[:8]
        )
        self.assertEqual(len(self.settings.recent_projects(existing_only=False)), 8)

    def test_missing_files_are_hidden_unless_asked_for(self):
        a, b = self._project("a"), self._project("b")
        self.settings.add_recent_project(a)
        self.settings.add_recent_project(b)
        b.unlink()
        self.assertEqual(self.settings.recent_projects(), [a])
        self.assertEqual(
            self.settings.recent_projects(existing_only=False), [b, a]
        )

    def test_last_project_follows_the_latest_add(self):
        a, b = self._project("a"), self._project("b")
        self.assertIsNone(self.settings.last_project())
        self.settings.add_recent_project(a)
        self.settings.add_recent_project(b)
        self.assertEqual(self._reopen().last_project(), b)
        self.settings.add_recent_project(a)
        self.assertEqual(self.settings.last_project(), a)

    def test_last_project_is_none_once_the_file_is_gone(self):
        a = self._project("a")
        self.settings.add_recent_project(a)
        a.unlink()
        self.assertIsNone(self.settings.last_project())

    def test_clear_last_project_keeps_the_recent_list(self):
        a = self._project("a")
        self.settings.add_recent_project(a)
        self.settings.clear_last_project()
        self.assertIsNone(self.settings.last_project())
        self.assertEqual(self.settings.recent_projects(), [a])

    def test_remove_and_clear(self):
        a, b = self._project("a"), self._project("b")
        self.settings.add_recent_project(a)
        self.settings.add_recent_project(b)
        self.settings.remove_recent_project(b)
        self.assertEqual(self.settings.recent_projects(), [a])
        self.assertIsNone(self.settings.last_project())
        self.settings.clear_recent_projects()
        self.assertEqual(self.settings.recent_projects(), [])

    def test_projects_and_templates_have_separate_recent_lists(self):
        project = self._project("a")
        template = self.root / "template.json"
        template.write_text("{}", encoding="utf-8")
        self.settings.add_recent_project(project)
        self.settings.add_recent_file(template)
        self.assertEqual(self.settings.recent_projects(), [project])
        self.assertEqual(self.settings.recent_files(), [template])

    def test_corrupt_values_fall_back_to_empty(self):
        self.settings._qs.setValue(self.settings.KEY_RECENT_PROJECTS, "{broken")
        self.settings._qs.setValue(self.settings.KEY_LAST_PROJECT, 5)
        self.assertEqual(self.settings.recent_projects(), [])
        self.assertIsNone(self.settings.last_project())


if __name__ == "__main__":
    unittest.main()
