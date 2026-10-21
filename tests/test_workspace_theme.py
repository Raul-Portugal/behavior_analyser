"""The workspace design system: palettes, style sheet, theme switching, icons."""
import unittest

try:
    from PyQt6.QtCore import QSize, Qt, qInstallMessageHandler
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QTableView
    from gui.workspace import theme
    from gui.workspace.icons import ICON_NAMES, icon_pixmap, icon_svg
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

REQUIRED_ICONS = (
    "video", "folder", "plus", "trash", "play", "pause", "stop", "check",
    "warning", "error", "info", "chevron-left", "chevron-right", "settings",
    "zones", "target", "clock", "chart", "list", "search", "eye", "undo",
    "redo", "zoom-in", "zoom-out", "fit", "align", "reset", "ruler", "help",
)

TEXT_PAIRS = (
    ("text", "background"), ("text", "surface"), ("text", "surface_alt"),
    ("text_muted", "background"), ("text_muted", "surface"),
    ("text_muted", "surface_alt"), ("primary_text", "primary"),
    ("primary_text", "primary_hover"), ("text", "selection"),
    ("primary", "background"), ("primary", "surface"),
    ("success", "surface"), ("warning", "surface"), ("danger", "surface"),
    ("info", "surface"), ("success", "background"), ("warning", "background"),
    ("danger", "background"), ("info", "background"),
)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class PaletteTests(unittest.TestCase):
    def test_text_pairs_reach_wcag_aa(self):
        for palette in (theme.LIGHT, theme.DARK):
            for foreground, background in TEXT_PAIRS:
                ratio = theme.contrast_ratio(
                    getattr(palette, foreground), getattr(palette, background)
                )
                self.assertGreaterEqual(
                    ratio, 4.5, f"{palette.name}: {foreground} on {background} = {ratio:.2f}"
                )

    def test_status_colors_are_readable_on_their_own_tint(self):
        for palette in (theme.LIGHT, theme.DARK):
            for base in ("surface", "background"):
                for kind in ("success", "warning", "danger", "info", "primary"):
                    tint = palette.tint(kind, 0.10, over=base)
                    ratio = theme.contrast_ratio(getattr(palette, kind), tint)
                    self.assertGreaterEqual(ratio, 4.5, f"{palette.name} {kind}/{base}")

    def test_kind_colors_cover_every_chip_kind(self):
        for palette in (theme.LIGHT, theme.DARK):
            for kind in ("ok", "warning", "error", "info", "neutral", "running"):
                foreground, soft = palette.kind_colors(kind)
                self.assertGreaterEqual(theme.contrast_ratio(foreground, soft), 4.5, kind)

    def test_controls_and_focus_ring_are_visible(self):
        for palette in (theme.LIGHT, theme.DARK):
            for background in (palette.surface, palette.background):
                self.assertGreaterEqual(
                    theme.contrast_ratio(palette.focus_ring, background), 3.0
                )
            self.assertGreaterEqual(
                theme.contrast_ratio(palette.border_strong, palette.surface), 2.9
            )

    def test_contrast_ratio_known_values(self):
        self.assertAlmostEqual(theme.contrast_ratio("#000000", "#FFFFFF"), 21.0, places=1)
        self.assertAlmostEqual(theme.contrast_ratio("#777777", "#777777"), 1.0, places=2)

    def test_mix(self):
        self.assertEqual(theme.mix("#000000", "#FFFFFF", 0.0), "#FFFFFF")
        self.assertEqual(theme.mix("#000000", "#FFFFFF", 1.0), "#000000")

    def test_palettes_have_distinct_light_and_dark_backgrounds(self):
        self.assertFalse(theme.LIGHT.is_dark)
        self.assertTrue(theme.DARK.is_dark)
        self.assertGreater(
            theme.contrast_ratio(theme.LIGHT.background, theme.DARK.background), 10
        )

    def test_scales(self):
        self.assertEqual(theme.Spacing.SCALE, (4, 8, 12, 16, 24, 32, 48))
        self.assertEqual(
            theme.TypeScale.SIZES,
            {"display": 22, "title": 18, "subtitle": 14, "body": 13, "caption": 11},
        )

    def test_palette_for(self):
        self.assertIs(theme.palette_for("light", Qt.ColorScheme.Dark), theme.LIGHT)
        self.assertIs(theme.palette_for("dark", Qt.ColorScheme.Light), theme.DARK)
        self.assertIs(theme.palette_for("auto", Qt.ColorScheme.Dark), theme.DARK)
        self.assertIs(theme.palette_for("auto", Qt.ColorScheme.Light), theme.LIGHT)
        self.assertIs(theme.palette_for("auto", Qt.ColorScheme.Unknown), theme.LIGHT)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class StyleSheetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_selectors_for_every_required_widget(self):
        sheet = theme.stylesheet(theme.LIGHT)
        for selector in (
            "QMainWindow", 'QPushButton[class="primary"]', 'QPushButton[class="secondary"]',
            'QPushButton[class="ghost"]', 'QPushButton[class="danger"]', "QLineEdit",
            "QComboBox", "QSpinBox", "QDoubleSpinBox", "QTableView", "QHeaderView::section",
            "QTabBar::tab", "QScrollBar:vertical", "QSlider::groove:horizontal",
            "QCheckBox::indicator", "QRadioButton::indicator", "QToolTip", "QMenu",
            "QMenuBar", "QStatusBar", ":focus", ":disabled", 'invalid="true"',
        ):
            self.assertIn(selector, sheet, selector)

    def test_every_token_is_substituted(self):
        for palette in (theme.LIGHT, theme.DARK):
            sheet = theme.stylesheet(palette)
            self.assertNotIn("$", sheet)
            self.assertIn(palette.primary, sheet)
            self.assertIn(palette.surface_alt, sheet)

    def test_images_referenced_by_the_sheet_exist(self):
        import re
        from pathlib import Path
        sheet = theme.stylesheet(theme.DARK)
        urls = re.findall(r"url\(([^)]+)\)", sheet)
        self.assertTrue(urls)
        for url in urls:
            self.assertTrue(Path(url).is_file(), url)

    def test_sheet_parses_without_qt_warnings(self):
        messages = []
        previous = qInstallMessageHandler(lambda kind, context, text: messages.append(text))
        try:
            label = QLabel("x")
            label.setStyleSheet(theme.stylesheet(theme.LIGHT))
            label.ensurePolished()
        finally:
            qInstallMessageHandler(previous)
        self.assertEqual([m for m in messages if "stylesheet" in m.lower()], [])

    def test_set_class_and_role_update_properties(self):
        button = QPushButton("x")
        theme.set_class(button, "primary")
        self.assertEqual(button.property("class"), "primary")
        label = QLabel("x")
        theme.set_role(label, "title")
        self.assertEqual(label.property("role"), "title")

    def test_style_table_view(self):
        view = QTableView()
        theme.style_table_view(view)
        self.assertEqual(view.verticalHeader().defaultSectionSize(), 32)
        self.assertFalse(view.showGrid())
        self.assertTrue(view.alternatingRowColors())
        self.assertFalse(view.verticalHeader().isVisible())


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ApplyThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.addCleanup(theme.reset_theme, self.app)
        self.before_sheet = self.app.styleSheet()
        self.before_style = self.app.style().objectName()

    def test_apply_dark_sets_palette_sheet_and_current(self):
        palette = theme.apply_theme(self.app, "dark")
        self.assertIs(palette, theme.DARK)
        self.assertIs(theme.current_palette(), theme.DARK)
        self.assertEqual(
            self.app.palette().color(self.app.palette().ColorRole.Window).name().upper(),
            theme.DARK.background,
        )
        self.assertIn(theme.DARK.primary, self.app.styleSheet())
        self.assertTrue(theme.get_controller(self.app)._fusion_set)

    def test_switching_mode_changes_everything(self):
        theme.apply_theme(self.app, "dark")
        theme.apply_theme(self.app, "light")
        self.assertIs(theme.current_palette(), theme.LIGHT)
        self.assertIn(theme.LIGHT.primary, self.app.styleSheet())
        self.assertNotIn(theme.DARK.primary, self.app.styleSheet())

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            theme.apply_theme(self.app, "sepia")

    def test_auto_follows_the_os_scheme_live(self):
        theme.apply_theme(self.app, "auto")
        controller = theme.get_controller(self.app)
        seen = []
        controller.palette_changed.connect(seen.append)
        controller._on_scheme_changed(Qt.ColorScheme.Dark)
        self.assertIs(theme.current_palette(), theme.DARK)
        controller._on_scheme_changed(Qt.ColorScheme.Light)
        self.assertIs(theme.current_palette(), theme.LIGHT)
        self.assertEqual(seen, [theme.DARK, theme.LIGHT])

    def test_pinned_mode_ignores_the_os_scheme(self):
        theme.apply_theme(self.app, "light")
        controller = theme.get_controller(self.app)
        controller._on_scheme_changed(Qt.ColorScheme.Dark)
        self.assertIs(theme.current_palette(), theme.LIGHT)

    def test_one_controller_per_application(self):
        theme.apply_theme(self.app, "dark")
        first = theme.get_controller(self.app)
        theme.apply_theme(self.app, "light")
        self.assertIs(theme.get_controller(self.app), first)
        self.assertEqual(len(self.app.findChildren(theme.ThemeController)), 1)

    def test_reset_restores_the_application(self):
        theme.apply_theme(self.app, "dark")
        theme.reset_theme(self.app)
        self.assertEqual(self.app.styleSheet(), self.before_sheet)
        self.assertEqual(self.app.style().objectName(), self.before_style)
        self.assertIsNone(theme.get_controller(self.app))
        self.assertIs(theme.current_palette(), theme.LIGHT)

    def test_body_font_size_follows_the_type_scale(self):
        theme.apply_theme(self.app, "light")
        self.assertEqual(self.app.font().pixelSize(), theme.TypeScale.BODY)

    def test_theme_mode_setting_round_trips(self):
        settings = temp_app_settings(self)
        self.assertEqual(settings.theme_mode(), "auto")
        settings.set_theme_mode("dark")
        self.assertEqual(settings.theme_mode(), "dark")
        with self.assertRaises(ValueError):
            settings.set_theme_mode("blue")

    def test_ui_flag_setting_round_trips(self):
        settings = temp_app_settings(self)
        self.assertFalse(settings.ui_flag("x"))
        self.assertTrue(settings.ui_flag("x", True))
        settings.set_ui_flag("x", True)
        self.assertTrue(settings.ui_flag("x"))
        settings.set_ui_flag("x", False)
        self.assertFalse(settings.ui_flag("x", True))


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class IconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        theme.reset_theme(self.app)

    def opaque_pixels(self, pixmap):
        image = pixmap.toImage()
        return sum(
            1 for x in range(image.width()) for y in range(image.height())
            if image.pixelColor(x, y).alpha() > 40
        )

    def test_the_required_icons_exist(self):
        self.assertGreaterEqual(len(ICON_NAMES), 25)
        for name in REQUIRED_ICONS:
            self.assertIn(name, ICON_NAMES)

    def test_every_icon_renders_something(self):
        for name in ICON_NAMES:
            pixmap = theme.icon(name).pixmap(QSize(24, 24))
            self.assertFalse(pixmap.isNull(), name)
            self.assertGreater(self.opaque_pixels(pixmap), 10, name)

    def test_icon_svg_is_valid_and_uses_the_color(self):
        self.assertIn('stroke="#ff0000"', icon_svg("video", "#ff0000"))
        with self.assertRaises(KeyError):
            icon_svg("nope")
        with self.assertRaises(KeyError):
            theme.icon("nope")

    def test_color_can_be_hex_qcolor_or_token(self):
        theme.apply_theme(self.app, "light")
        red = theme.icon("plus", "#ff0000", 24).pixmap(QSize(24, 24)).toImage()
        qred = theme.icon("plus", QColor("#ff0000")).pixmap(QSize(24, 24)).toImage()
        self.assertEqual(red.pixelColor(12, 12).red(), 255)
        self.assertEqual(qred.pixelColor(12, 12).red(), 255)
        token = icon_pixmap("plus", "primary", 24).toImage().pixelColor(12, 12)
        self.assertEqual(token.name().upper(), theme.LIGHT.primary)

    def test_token_colors_follow_the_theme(self):
        engine_icon = theme.icon("plus", "primary", 24)
        theme.apply_theme(self.app, "light")
        light = engine_icon.pixmap(QSize(24, 24)).toImage().pixelColor(12, 12).name()
        theme.apply_theme(self.app, "dark")
        dark = engine_icon.pixmap(QSize(24, 24)).toImage().pixelColor(12, 12).name()
        self.assertEqual(light.upper(), theme.LIGHT.primary)
        self.assertEqual(dark.upper(), theme.DARK.primary)

    def test_fixed_size_and_disabled_mode(self):
        fixed = theme.icon("check", "#000000", 16)
        self.assertEqual(fixed.pixmap(QSize(64, 64)).size(), QSize(16, 16))
        normal = fixed.pixmap(QSize(16, 16), mode=fixed.Mode.Normal).toImage()
        disabled = fixed.pixmap(QSize(16, 16), mode=fixed.Mode.Disabled).toImage()
        brightest = lambda image: max(  # noqa: E731
            image.pixelColor(x, y).alpha() for x in range(16) for y in range(16)
        )
        self.assertLess(brightest(disabled), brightest(normal))

    def test_icon_can_be_set_on_a_button(self):
        button = QPushButton("Go")
        button.setIcon(theme.icon("play"))
        self.assertFalse(button.icon().isNull())


if __name__ == "__main__":
    unittest.main()
