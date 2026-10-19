import unittest
from unittest.mock import patch

try:
    from PyQt6.QtWidgets import QApplication
    from gui.main_window import MainWindow
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ValidationMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)

    def test_validation_menu_lists_the_three_tools(self):
        menus = {action.text(): action.menu()
                 for action in self.window.menuBar().actions()}
        self.assertIn("&Validation", menus)
        texts = [action.text() for action in menus["&Validation"].actions()]
        self.assertEqual(texts, ["&Score a video manually...",
                                 "Tracking s&pot-check...",
                                 "&Build validation report..."])

    def test_actions_open_their_dialogs_with_the_window_and_settings(self):
        cases = (
            (self.window.manual_scoring_action,
             "gui.manual_scoring_dialog.open_manual_scoring"),
            (self.window.position_spotcheck_action,
             "gui.position_spotcheck_dialog.open_position_spotcheck"),
            (self.window.validation_report_action,
             "gui.validation_report_dialog.open_validation_report"),
        )
        for action, target in cases:
            with self.subTest(target=target):
                with patch(target) as opener:
                    action.trigger()
                opener.assert_called_once_with(
                    self.window, self.window.app_settings)


if __name__ == "__main__":
    unittest.main()
