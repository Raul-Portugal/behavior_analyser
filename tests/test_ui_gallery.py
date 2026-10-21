"""tools/ui_gallery.py renders every kit widget without errors, off screen."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication
    from gui.workspace import theme
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def load_gallery():
    spec = importlib.util.spec_from_file_location(
        "ui_gallery_tool", PROJECT_ROOT / "tools" / "ui_gallery.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class GalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.gallery = load_gallery()

    def setUp(self):
        self.addCleanup(theme.reset_theme, self.app)
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "gallery"

    def test_renders_every_page_to_a_png(self):
        paths = self.gallery.render_gallery(self.out, app=self.app)
        self.assertEqual({p.stem for p in paths}, set(self.gallery.PAGES))
        for path in paths:
            image = QImage(str(path))
            self.assertFalse(image.isNull(), path)
            self.assertGreaterEqual(image.width(), self.gallery.PAGE_WIDTH - 2)
            self.assertGreater(image.height(), 100)

    def test_covers_every_kit_widget(self):
        source = (PROJECT_ROOT / "tools" / "ui_gallery.py").read_text(encoding="utf-8")
        for name in ("Stepper", "StageHeader", "FooterBar", "PrimaryButton", "SecondaryButton",
                     "GhostButton", "DangerButton", "Card", "CollapsibleSection", "StatusChip",
                     "EmptyState", "DropZone", "ParadigmPicker", "Filmstrip", "InlineHelp",
                     "Callout", "KeyHint"):
            self.assertIn(name, source)

    def test_dark_pages_get_a_suffix_and_look_different(self):
        light = self.gallery.render_gallery(self.out, app=self.app)
        dark = self.gallery.render_gallery(self.out, dark=True, app=self.app)
        self.assertTrue(all(p.stem.endswith("_dark") for p in dark))
        light_image = QImage(str(next(p for p in light if p.stem == "stage_mockup")))
        dark_image = QImage(str(next(p for p in dark if p.stem == "stage_mockup_dark")))
        self.assertNotEqual(light_image.pixelColor(2, 200), dark_image.pixelColor(2, 200))

    def test_command_line_entry_point(self):
        self.assertEqual(self.gallery.main(["--out", str(self.out)]), 0)
        self.assertTrue((self.out / "stepper.png").is_file())
        self.assertIsNone(theme.get_controller(self.app))     # the theme is put back

    def test_nothing_is_shown_on_screen(self):
        page, _layout = self.gallery._page()
        from PyQt6.QtCore import Qt
        self.assertTrue(page.testAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen))


if __name__ == "__main__":
    unittest.main()
