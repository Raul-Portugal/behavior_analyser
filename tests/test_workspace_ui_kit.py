"""The workspace UI kit: signals, states, painting and persistence."""
import tempfile
import unittest
from pathlib import Path

from tests.gui_support import temp_app_settings
from tests.synthetic_video import write_indexed_video

try:
    from PyQt6.QtCore import (QCoreApplication, QEvent, QMimeData, QPoint, QPointF, QUrl,
                              Qt)
    from PyQt6.QtGui import (QDragEnterEvent, QDragLeaveEvent, QDropEvent, QFocusEvent,
                             QFontMetrics)
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget
    from gui.workspace import theme
    from gui.workspace.ui_kit import (Callout, Card, CollapsibleSection, DangerButton,
                                      DropZone, EmptyState, Filmstrip, FilmstripItem,
                                      FooterBar, GhostButton, InlineHelp, KeyHint, Label,
                                      ParadigmPicker, PrimaryButton, SecondaryButton,
                                      StageHeader, StatTile, StatusChip, StepInfo,
                                      Stepper)
    from mazes import AVAILABLE_MAZES
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


class KitCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAS_RUNTIME:
            raise unittest.SkipTest("GUI runtime is not installed")
        cls.app = QApplication.instance() or QApplication([])
        theme.apply_theme(cls.app, "light")

    @classmethod
    def tearDownClass(cls):
        theme.reset_theme(cls.app)

    def setUp(self):
        self._widgets = []

    def tearDown(self):
        for widget in self._widgets:
            widget.close()
            widget.deleteLater()
        QCoreApplication.processEvents()

    def show(self, widget, width=None, height=None):
        """Lay a widget out and paint it without showing anything on screen."""
        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        if width and height:
            widget.resize(width, height)
        elif width:
            widget.resize(width, widget.sizeHint().height())
        widget.show()
        QCoreApplication.processEvents()
        self._widgets.append(widget)
        return widget

    def grab(self, widget):
        QCoreApplication.processEvents()
        return widget.grab().toImage()

    @staticmethod
    def hex_at(image, x, y):
        return image.pixelColor(int(x), int(y)).name().upper()


class ButtonTests(KitCase):
    def test_variants_set_the_class_property(self):
        for factory, name in ((PrimaryButton, "primary"), (SecondaryButton, "secondary"),
                              (GhostButton, "ghost"), (DangerButton, "danger")):
            button = factory("Go")
            self.assertEqual(button.property("class"), name)
            self.assertEqual(button.text(), "Go")

    def test_icon_name(self):
        button = PrimaryButton("Add", "plus")
        self.assertEqual(button.icon_name(), "plus")
        self.assertFalse(button.icon().isNull())
        button.set_icon_name(None)
        self.assertTrue(button.icon().isNull())

    def test_buttons_emit_clicked(self):
        button = SecondaryButton("x")
        seen = []
        button.clicked.connect(lambda: seen.append(1))
        button.click()
        self.assertEqual(seen, [1])

    def test_primary_button_is_painted_with_the_primary_colour(self):
        button = self.show(PrimaryButton("Go"), 120, 40)
        self.assertEqual(self.hex_at(self.grab(button), 8, 20), theme.LIGHT.primary)

    def test_label_role(self):
        self.assertEqual(Label("x", "title").property("role"), "title")


class StatusChipTests(KitCase):
    def test_kinds_and_text(self):
        for kind in ("ok", "warning", "error", "info", "neutral", "running"):
            chip = StatusChip(kind, "Hello")
            self.assertEqual(chip.kind(), kind)
            self.assertEqual(chip.text(), "Hello")
        with self.assertRaises(ValueError):
            StatusChip("purple", "x")

    def test_size_follows_the_text(self):
        short, long = StatusChip("ok", "Hi"), StatusChip("ok", "A much longer status")
        self.assertGreater(long.sizeHint().width(), short.sizeHint().width())
        self.assertEqual(short.sizeHint().height(), 22)

    def test_chip_is_painted_with_its_soft_colour(self):
        for kind in ("ok", "warning", "error", "info"):
            chip = self.show(StatusChip(kind, "Status"))
            soft = theme.LIGHT.kind_colors(kind)[1].upper()
            self.assertEqual(self.hex_at(self.grab(chip), chip.width() - 4, 11), soft, kind)

    def test_running_chip_animates_only_while_visible(self):
        chip = StatusChip("running", "Working")
        self.assertFalse(chip.is_animating())
        self.show(chip)
        self.assertTrue(chip.is_animating())
        chip.set_kind("ok")
        self.assertFalse(chip.is_animating())
        chip.set_kind("running")
        self.assertTrue(chip.is_animating())
        chip.hide()
        self.assertFalse(chip.is_animating())

    def test_set_status(self):
        chip = StatusChip("neutral", "")
        chip.set_status("error", "Failed")
        self.assertEqual((chip.kind(), chip.text()), ("error", "Failed"))


class SmallWidgetTests(KitCase):
    def test_key_hint(self):
        hint = KeyHint("Ctrl+S")
        self.assertEqual(hint.text(), "Ctrl+S")
        self.assertGreater(hint.sizeHint().width(), KeyHint("S").sizeHint().width())
        self.assertGreaterEqual(KeyHint("S").sizeHint().width(), 22)
        self.show(hint)
        self.assertFalse(self.grab(hint).isNull())

    def test_callout_kinds(self):
        for kind in ("info", "tip", "warning", "error", "success"):
            callout = Callout(kind, "Text")
            self.assertEqual(callout.kind(), kind)
        with self.assertRaises(ValueError):
            Callout("fun", "x")

    def test_callout_text_title_and_accent(self):
        callout = Callout("warning", "Careful", title="Heads up")
        self.assertEqual((callout.text(), callout.title()), ("Careful", "Heads up"))
        callout.set_text("Changed")
        self.assertEqual(callout.text(), "Changed")
        self.show(callout, 400)
        image = self.grab(callout)
        self.assertEqual(self.hex_at(image, 1, callout.height() // 2),
                         theme.LIGHT.warning)

    def test_callout_changes_kind(self):
        callout = Callout("info", "x")
        callout.set_kind("error")
        self.assertEqual(callout.kind(), "error")
        with self.assertRaises(ValueError):
            callout.set_kind("nope")

    def test_inline_help_can_be_dismissed(self):
        help_box = InlineHelp("A tip", dismissible=True)
        self.show(help_box, 400)
        seen = []
        help_box.dismissed.connect(lambda: seen.append(True))
        self.assertTrue(help_box.close_button.isVisibleTo(help_box))
        help_box.close_button.click()
        self.assertEqual(seen, [True])
        self.assertFalse(help_box.isVisible())

    def test_inline_help_is_not_dismissible_by_default(self):
        help_box = InlineHelp("A tip")
        self.show(help_box, 400)
        self.assertFalse(help_box.close_button.isVisibleTo(help_box))


class CardTests(KitCase):
    def test_title_header_and_body(self):
        card = Card("Title", "Subtitle")
        self.show(card, 300)
        self.assertEqual(card.title(), "Title")
        self.assertTrue(card.title_label.isVisibleTo(card))
        self.assertTrue(card.subtitle_label.isVisibleTo(card))
        card.add_widget(QLabel("content"))
        self.assertEqual(card.body.count(), 1)

    def test_untitled_card_has_no_header(self):
        card = Card()
        self.show(card, 300)
        self.assertFalse(card.title_label.isVisibleTo(card))
        card.add_header_widget(QPushButton("x"))
        self.assertTrue(card.title_label.isVisibleTo(card))

    def test_card_surface_and_border_colours(self):
        card = self.show(Card("T"), 200, 120)
        image = self.grab(card)
        self.assertEqual(self.hex_at(image, 100, 100), theme.LIGHT.surface)
        self.assertEqual(self.hex_at(image, 100, 0), theme.LIGHT.border)


class StatTileTests(KitCase):
    def test_caption_value_and_unit(self):
        tile = StatTile("Total distance", "123.4", "px")
        self.show(tile, 220)
        self.assertEqual(tile.caption(), "Total distance")
        self.assertEqual(tile.value(), "123.4")
        self.assertEqual(tile.unit(), "px")
        self.assertTrue(tile._unit.isVisibleTo(tile))
        self.assertIn("Total distance: 123.4 px", tile.toolTip())

    def test_the_unit_is_optional(self):
        tile = StatTile("Grade", "Excellent")
        self.show(tile, 220)
        self.assertFalse(tile._unit.isVisibleTo(tile))
        tile.set_stat("Speed", "36.8", "px/s")
        self.assertEqual((tile.caption(), tile.value(), tile.unit()),
                         ("Speed", "36.8", "px/s"))

    def test_it_paints_a_quiet_tile_in_the_theme_colours(self):
        tile = self.show(StatTile("a", "b"), 200, 80)
        image = self.grab(tile)
        self.assertEqual(self.hex_at(image, 100, 70), theme.LIGHT.surface_alt)
        self.assertEqual(self.hex_at(image, 100, 0), theme.LIGHT.border)

    def test_a_long_value_does_not_widen_the_tile(self):
        tile = StatTile("Caption", "x" * 200)
        self.show(tile, 160)
        self.assertLessEqual(tile.width(), 160)


class CollapsibleSectionTests(KitCase):
    def make(self, **kwargs):
        kwargs.setdefault("animated", False)
        section = CollapsibleSection("Advanced", **kwargs)
        section.body.addWidget(QLabel("body"))
        self.show(section, 300)
        return section

    def test_collapsed_by_default_and_toggles(self):
        section = self.make()
        self.assertFalse(section.is_expanded())
        self.assertFalse(section._body.isVisibleTo(section))
        seen = []
        section.toggled.connect(seen.append)
        section.toggle()
        self.assertTrue(section.is_expanded())
        self.assertTrue(section._body.isVisibleTo(section))
        section.set_expanded(False)
        self.assertEqual(seen, [True, False])
        section.set_expanded(False)
        self.assertEqual(seen, [True, False])

    def test_clicking_the_header_toggles(self):
        section = self.make()
        QTest.mouseClick(section._header, Qt.MouseButton.LeftButton)
        self.assertTrue(section.is_expanded())

    def test_chevron_follows_the_state(self):
        section = self.make(expanded=True)
        self.assertEqual(section._header.icon_name(), "chevron-down")
        section.set_expanded(False)
        self.assertEqual(section._header.icon_name(), "chevron-right")

    def test_summary(self):
        section = self.make(summary="2 changed")
        self.assertTrue(section._summary.isVisibleTo(section))
        section.set_summary("")
        self.assertFalse(section._summary.isVisibleTo(section))

    def test_state_is_remembered_in_the_settings(self):
        settings = temp_app_settings(self)
        first = self.make(settings=settings, settings_key="detection/advanced")
        self.assertFalse(first.is_expanded())
        first.set_expanded(True)
        second = self.make(settings=settings, settings_key="detection/advanced")
        self.assertTrue(second.is_expanded())
        second.set_expanded(False)
        third = self.make(settings=settings, settings_key="detection/advanced")
        self.assertFalse(third.is_expanded())

    def test_first_run_uses_the_default(self):
        settings = temp_app_settings(self)
        section = self.make(expanded=True, settings=settings, settings_key="x")
        self.assertTrue(section.is_expanded())

    def test_animated_toggle_finishes_in_the_right_state(self):
        section = self.make(animated=True)
        section.set_expanded(True)
        for _ in range(100):
            QCoreApplication.processEvents()
            if not section.animation_running():
                break
            QTest.qWait(10)
        self.assertFalse(section.animation_running())
        self.assertTrue(section._body.isVisibleTo(section))
        self.assertEqual(section._body.maximumHeight(), 16777215)
        section.set_expanded(False)
        for _ in range(100):
            if not section.animation_running():
                break
            QTest.qWait(10)
        self.assertFalse(section._body.isVisibleTo(section))


class EmptyStateTests(KitCase):
    def test_texts_and_actions(self):
        state = EmptyState("video", "No videos", "Add some")
        self.show(state, 500, 400)
        self.assertEqual((state.title(), state.body()), ("No videos", "Add some"))
        first = state.add_action("Add", icon_name="plus")
        second = state.add_action("Open")
        self.assertEqual(first.property("class"), "primary")
        self.assertEqual(second.property("class"), "secondary")
        with self.assertRaises(ValueError):
            state.add_action("Third")
        seen = []
        first.clicked.connect(lambda: seen.append("add"))
        first.click()
        self.assertEqual(seen, ["add"])

    def test_body_text_is_fully_visible_when_it_wraps(self):
        body = "word " * 60
        state = EmptyState("video", "Title", body)
        self.show(state, 420, 500)
        label = state.body_label
        self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))

    def test_clear_actions(self):
        state = EmptyState("video", "T")
        state.add_action("A")
        state.clear_actions()
        self.assertEqual(state.buttons, [])
        state.add_action("B")


class StageHeaderTests(KitCase):
    def test_texts_and_help(self):
        header = StageHeader("Videos", "Add your videos", "video", show_help=True)
        self.show(header, 600)
        self.assertEqual((header.title(), header.subtitle()), ("Videos", "Add your videos"))
        seen = []
        header.help_requested.connect(lambda: seen.append(True))
        header.help_button.click()
        self.assertEqual(seen, [True])

    def test_help_is_optional(self):
        header = StageHeader("Videos")
        self.show(header, 600)
        self.assertFalse(header.help_button.isVisibleTo(header))
        header.set_help_visible(True)
        self.assertTrue(header.help_button.isVisibleTo(header))

    def test_action_slot_is_right_of_the_title(self):
        header = StageHeader("Videos", "x")
        button = QPushButton("Act")
        header.add_action(button)
        self.show(header, 800)
        self.assertGreater(button.x(), header.title_label.x() + 100)
        self.assertGreater(button.x() + button.width(), 600)

    def test_set_texts(self):
        header = StageHeader("A", "b")
        header.set_texts("C", "")
        self.assertEqual(header.title(), "C")
        self.show(header, 400)
        self.assertFalse(header.subtitle_label.isVisibleTo(header))


class FooterBarTests(KitCase):
    def test_signals(self):
        footer = self.show(FooterBar(), 800)
        seen = []
        footer.back_clicked.connect(lambda: seen.append("back"))
        footer.next_clicked.connect(lambda: seen.append("next"))
        footer.back_button.click()
        footer.next_button.click()
        self.assertEqual(seen, ["back", "next"])
        self.assertEqual(footer.back_button.text(), "← Back")
        self.assertEqual(footer.back_button.property("class"), "ghost")
        self.assertEqual(footer.next_button.property("class"), "primary")

    def test_next_label(self):
        footer = self.show(FooterBar(), 800)
        footer.set_next("Detection")
        self.assertEqual(footer.next_button.text(), "Next: Detection →")
        self.assertTrue(footer.next_button.isEnabled())
        self.assertEqual(footer.next_button.toolTip(), "")

    def test_disabled_next_carries_a_reason(self):
        footer = self.show(FooterBar(), 800)
        footer.set_next("Detection", enabled=False, reason="Add a video first")
        self.assertFalse(footer.next_button.isEnabled())
        self.assertEqual(footer.next_button.toolTip(), "Add a video first")
        footer.set_next("Detection", enabled=True, reason="ignored when enabled")
        self.assertEqual(footer.next_button.toolTip(), "")

    def test_ampersand_in_a_label_is_not_a_mnemonic(self):
        footer = self.show(FooterBar(), 800)
        footer.set_next("Arena & zones")
        self.assertEqual(footer.next_text(), "Next: Arena & zones →")
        self.assertIn("&&", footer.next_button.text())

    def test_last_stage_wording_and_hiding(self):
        footer = self.show(FooterBar(), 800)
        footer.set_next("Run analysis", prefix="", arrow=False)
        self.assertEqual(footer.next_text(), "Run analysis")
        footer.set_next(None)
        self.assertFalse(footer.next_button.isVisibleTo(footer))

    def test_status_text_and_chip(self):
        footer = self.show(FooterBar(), 800)
        self.assertFalse(footer.chip.isVisibleTo(footer))
        footer.set_status("3 videos", ("ok", "Ready"))
        self.assertEqual(footer.status_label.text(), "3 videos")
        self.assertTrue(footer.chip.isVisibleTo(footer))
        self.assertEqual((footer.chip.kind(), footer.chip.text()), ("ok", "Ready"))
        footer.set_status("plain")
        self.assertFalse(footer.chip.isVisibleTo(footer))

    def test_back_can_be_hidden_or_disabled(self):
        footer = self.show(FooterBar(), 800)
        footer.set_back(visible=True, enabled=False)
        self.assertFalse(footer.back_button.isEnabled())
        footer.set_back(visible=False)
        self.assertFalse(footer.back_button.isVisibleTo(footer))

    def test_extra_button_sits_left_of_next(self):
        footer = FooterBar()
        extra = QPushButton("Save")
        footer.add_extra(extra)
        self.show(footer, 900)
        self.assertLess(extra.x(), footer.next_button.x())


def steps(*statuses):
    names = [("videos", "Videos", "video"), ("arena", "Arena & zones", "zones"),
             ("detection", "Detection", "target"), ("trials", "Trials", "clock"),
             ("review", "Review & run", "list"), ("results", "Results", "chart")]
    return [StepInfo(i, label, icon, status, f"reason {i}")
            for (i, label, icon), status in zip(names, statuses)]


class StepperTests(KitCase):
    def make(self, statuses=("done", "current", "attention", "todo", "disabled", "disabled"),
             width=1100):
        stepper = Stepper(steps(*statuses))
        self.show(stepper, width, 60)
        return stepper

    def click(self, stepper, step_id):
        center = stepper.step_rect(step_id).center().toPoint()
        QTest.mouseClick(stepper, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center)

    def test_content_and_lookup(self):
        stepper = self.make()
        self.assertEqual(stepper.step_ids(),
                         ["videos", "arena", "detection", "trials", "review", "results"])
        self.assertEqual(stepper.status("arena"), "current")
        self.assertEqual(stepper.current_id(), "arena")
        self.assertEqual(stepper.label("trials"), "Trials")
        with self.assertRaises(KeyError):
            stepper.status("nope")

    def test_status_validation(self):
        stepper = self.make()
        with self.assertRaises(ValueError):
            stepper.set_status("videos", "sparkly")
        with self.assertRaises(ValueError):
            Stepper([StepInfo("a", "A", status="bogus")])

    def test_clicking_a_step_emits_its_id(self):
        stepper = self.make()
        seen = []
        stepper.step_clicked.connect(seen.append)
        for step_id in ("videos", "arena", "detection", "trials"):
            self.click(stepper, step_id)
        self.assertEqual(seen, ["videos", "arena", "detection", "trials"])

    def test_disabled_steps_are_not_clickable(self):
        stepper = self.make()
        seen = []
        stepper.step_clicked.connect(seen.append)
        self.click(stepper, "review")
        self.click(stepper, "results")
        self.assertEqual(seen, [])

    def test_clicking_between_steps_does_nothing(self):
        stepper = self.make()
        seen = []
        stepper.step_clicked.connect(seen.append)
        videos, arena = stepper.step_rect("videos"), stepper.step_rect("arena")
        gap = QPoint(int((videos.right() + arena.left()) / 2), 30)
        QTest.mouseClick(stepper, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, gap)
        self.assertEqual(seen, [])

    def test_set_status_changes_what_is_clickable(self):
        stepper = self.make()
        seen = []
        stepper.step_clicked.connect(seen.append)
        stepper.set_status("review", "todo", "now ready")
        self.click(stepper, "review")
        self.assertEqual(seen, ["review"])
        self.assertEqual(stepper.reason("review"), "now ready")

    def test_set_current_moves_the_marker(self):
        stepper = self.make()
        stepper.set_current("detection")
        self.assertEqual(stepper.current_id(), "detection")
        self.assertEqual(stepper.status("arena"), "todo")

    def test_tooltip_states_status_and_reason(self):
        stepper = self.make()
        text = stepper.tooltip_for("detection")
        self.assertIn("Detection", text)
        self.assertIn("Needs attention", text)
        self.assertIn("reason detection", text)
        self.assertIn("Done", stepper.tooltip_for("videos"))
        self.assertIn("Not available", stepper.tooltip_for("results"))
        self.assertIn("You are here", stepper.tooltip_for("arena"))
        stepper.set_status("videos", "done", "")
        self.assertEqual(stepper.tooltip_for("videos"), "Videos: Done")

    def test_statuses_are_drawn_differently(self):
        stepper = self.make()
        palette = theme.LIGHT
        image = self.grab(stepper)

        def badge_fill(step_id):
            rect = stepper.badge_rect(step_id)
            return self.hex_at(image, rect.left() + 4, rect.center().y())

        self.assertEqual(badge_fill("arena"), palette.primary)                    # current
        self.assertEqual(badge_fill("videos"), palette.tint("success", 0.16))     # done
        self.assertEqual(badge_fill("detection"), palette.tint("warning", 0.16))  # attention
        self.assertEqual(badge_fill("trials"), palette.surface)                   # todo
        self.assertEqual(badge_fill("review"), palette.surface)                   # disabled
        # The current step also gets a tinted pill; a todo step does not.
        arena, trials = stepper.step_rect("arena"), stepper.step_rect("trials")
        self.assertEqual(self.hex_at(image, arena.right() - 8, arena.center().y()),
                         palette.tint("primary", 0.12))
        self.assertEqual(self.hex_at(image, trials.right() - 8, trials.center().y()),
                         palette.background)

    def test_done_connector_is_green_and_others_are_neutral(self):
        stepper = self.make(("done", "current", "todo", "todo", "todo", "todo"))
        image = self.grab(stepper)
        videos, arena, detection = (stepper.step_rect(i) for i in ("videos", "arena", "detection"))
        y = int(videos.center().y())
        self.assertEqual(self.hex_at(image, (videos.right() + arena.left()) / 2, y),
                         theme.LIGHT.success)
        self.assertEqual(self.hex_at(image, (arena.right() + detection.left()) / 2, y),
                         theme.LIGHT.border)

    def test_six_steps_fit_1100_px_with_full_labels(self):
        stepper = self.make(width=1100)
        self.assertEqual(stepper.layout_mode(), "full")
        metrics = QFontMetrics(stepper._label_font())
        if metrics.horizontalAdvance("Arena & zones") < 120:     # a real UI font
            for step_id in stepper.step_ids():
                self.assertEqual(stepper.displayed_label(step_id), stepper.label(step_id))
        rects = [stepper.step_rect(i) for i in stepper.step_ids()]
        self.assertGreaterEqual(rects[0].left(), 0)
        self.assertLessEqual(rects[-1].right(), 1100)
        for left, right in zip(rects, rects[1:]):
            self.assertLessEqual(left.right(), right.left())

    def test_labels_are_elided_when_narrow_and_dropped_when_tiny(self):
        medium = self.make(width=800)
        self.assertEqual(medium.layout_mode(), "label")
        labels = [medium.displayed_label(i) for i in medium.step_ids()]
        self.assertTrue(any("…" in label for label in labels))
        self.assertTrue(all(len(label) <= len(medium.label(i))
                            for label, i in zip(labels, medium.step_ids())))
        tiny = self.make(width=400)
        self.assertEqual(tiny.layout_mode(), "badge")
        self.assertEqual(tiny.displayed_label("videos"), "")
        # Even then every step stays clickable.
        seen = []
        tiny.step_clicked.connect(seen.append)
        self.click(tiny, "trials")
        self.assertEqual(seen, ["trials"])

    def test_minimum_width_is_small(self):
        self.assertLess(Stepper(steps(*["todo"] * 6)).minimumSizeHint().width(), 300)

    def test_keyboard_navigation_and_activation(self):
        stepper = self.make()
        seen = []
        stepper.step_clicked.connect(seen.append)
        stepper.setFocus()
        stepper.focusInEvent(QFocusEvent(QEvent.Type.FocusIn))   # (no window focus offscreen)
        self.assertEqual(stepper.focused_id(), "arena")        # the current step
        QTest.keyClick(stepper, Qt.Key.Key_Right)
        self.assertEqual(stepper.focused_id(), "detection")
        QTest.keyClick(stepper, Qt.Key.Key_Right)
        self.assertEqual(stepper.focused_id(), "trials")
        QTest.keyClick(stepper, Qt.Key.Key_Right)               # review is disabled: stays
        self.assertEqual(stepper.focused_id(), "trials")
        QTest.keyClick(stepper, Qt.Key.Key_Return)
        QTest.keyClick(stepper, Qt.Key.Key_Left)
        QTest.keyClick(stepper, Qt.Key.Key_Space)
        self.assertEqual(seen, ["trials", "detection"])

    def test_theme_change_repaints(self):
        stepper = self.make()
        light = self.grab(stepper)
        theme.apply_theme(self.app, "dark")
        try:
            dark = self.grab(stepper)
            rect = stepper.badge_rect("arena")
            self.assertEqual(self.hex_at(dark, rect.left() + 4, rect.center().y()),
                             theme.DARK.primary)
            self.assertNotEqual(self.hex_at(light, 2, 2), self.hex_at(dark, 2, 2))
        finally:
            theme.apply_theme(self.app, "light")


_KEEP_ALIVE = []     # events do not own their mime data; a dead one crashes Qt


def make_drop(paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    _KEEP_ALIVE.append(mime)
    return mime


def drop_event(mime):
    return QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                      Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


def enter_event(mime):
    return QDragEnterEvent(QPoint(10, 10), Qt.DropAction.CopyAction, mime,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


class DropZoneTests(KitCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.video = self.root / "a.mp4"
        self.video.write_bytes(b"x")
        self.note = self.root / "notes.txt"
        self.note.write_text("hi")
        self.folder = self.root / "clips"
        (self.folder / "deep").mkdir(parents=True)
        (self.folder / "b.avi").write_bytes(b"x")
        (self.folder / "deep" / "c.mov").write_bytes(b"x")

    def test_defaults_text(self):
        zone = DropZone()
        self.assertEqual(zone.title_label.text(), "Drop videos or folders here")
        self.assertTrue(zone.acceptDrops())

    def test_dropping_videos_and_folders_emits_the_videos(self):
        zone = self.show(DropZone(), 500, 260)
        dropped, ignored = [], []
        zone.paths_dropped.connect(dropped.append)
        zone.paths_ignored.connect(ignored.append)
        event = drop_event(make_drop([self.video, self.folder, self.note]))
        zone.dropEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertEqual(len(dropped), 1)
        self.assertEqual({p.name for p in dropped[0]}, {"a.mp4", "b.avi"})
        self.assertEqual([p.name for p in ignored[0]], ["notes.txt"])

    def test_recursive_drops_search_sub_folders(self):
        zone = self.show(DropZone(), 500, 260)
        dropped = []
        zone.paths_dropped.connect(dropped.append)
        zone.set_recursive(True)
        self.assertTrue(zone.recursive())
        zone.dropEvent(drop_event(make_drop([self.folder])))
        self.assertEqual({p.name for p in dropped[0]}, {"b.avi", "c.mov"})

    def test_dropping_only_other_files_reports_them_and_adds_nothing(self):
        zone = self.show(DropZone(), 500, 260)
        dropped, ignored = [], []
        zone.paths_dropped.connect(dropped.append)
        zone.paths_ignored.connect(ignored.append)
        zone.dropEvent(drop_event(make_drop([self.note])))
        self.assertEqual(dropped, [])
        self.assertEqual(len(ignored), 1)

    def test_highlight_while_dragging(self):
        zone = self.show(DropZone(), 500, 260)
        self.assertFalse(zone.is_dragging())
        before = self.grab(zone)
        event = enter_event(make_drop([self.video]))
        zone.dragEnterEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertTrue(zone.is_dragging())
        during = self.grab(zone)
        self.assertEqual(self.hex_at(before, 40, 40), theme.LIGHT.surface)
        self.assertEqual(self.hex_at(during, 40, 40), theme.LIGHT.tint("primary", 0.12))
        zone.dragLeaveEvent(QDragLeaveEvent())
        self.assertFalse(zone.is_dragging())

    def test_a_drop_clears_the_highlight(self):
        zone = self.show(DropZone(), 500, 260)
        zone.dragEnterEvent(enter_event(make_drop([self.video])))
        zone.dropEvent(drop_event(make_drop([self.video])))
        self.assertFalse(zone.is_dragging())

    def test_drags_without_files_are_refused(self):
        zone = self.show(DropZone(), 500, 260)
        mime = QMimeData()
        mime.setText("just text")
        _KEEP_ALIVE.append(mime)
        event = enter_event(mime)
        event.setAccepted(False)
        zone.dragEnterEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertFalse(zone.is_dragging())
        dropped = []
        zone.paths_dropped.connect(dropped.append)
        zone.dropEvent(drop_event(mime))
        self.assertEqual(dropped, [])

    def test_clicks_and_buttons_request_dialogs(self):
        zone = self.show(DropZone(), 500, 260)
        seen = []
        zone.browse_requested.connect(lambda: seen.append("browse"))
        zone.folder_requested.connect(lambda: seen.append("folder"))
        zone.browse_button.click()
        zone.folder_button.click()
        QTest.mouseClick(zone, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(20, 20))
        QTest.keyClick(zone, Qt.Key.Key_Return)
        self.assertEqual(seen, ["browse", "folder", "browse", "browse"])

    def test_compact_mode_hides_the_illustration(self):
        zone = self.show(DropZone(), 500, 260)
        zone.set_compact(True)
        self.assertFalse(zone.hint_label.isVisibleTo(zone))
        zone.set_compact(False)
        self.assertTrue(zone.hint_label.isVisibleTo(zone))

    def test_content_does_not_overlap_at_the_natural_height(self):
        zone = self.show(DropZone(), 600)
        badge_bottom = zone._badge.geometry().bottom()
        self.assertLess(badge_bottom, zone.title_label.geometry().top())
        self.assertLessEqual(zone.folder_button.geometry().bottom(), zone.height())


class ParadigmPickerTests(KitCase):
    def test_one_card_per_registered_paradigm(self):
        picker = ParadigmPicker()
        self.assertEqual(picker.names(), list(AVAILABLE_MAZES))

    def test_descriptions_are_the_agreed_one_liners(self):
        picker = ParadigmPicker()
        self.assertEqual(picker.card("Y-Maze").description(),
                         "Spontaneous alternation: arm entries and alternation %")
        self.assertEqual(picker.card("Elevated Plus Maze").description(),
                         "Anxiety: time and entries in open vs closed arms")
        self.assertEqual(picker.card("Freestyle / Open Field").description(),
                         "Locomotion, zones, center vs periphery (thigmotaxis)")
        self.assertEqual(picker.card("Tail Suspension Test").description(),
                         "Immobility, passive swing and active escape per mouse")

    def test_clicking_a_card_selects_and_emits_once(self):
        picker = self.show(ParadigmPicker(), 1000, 200)
        chosen = []
        picker.paradigm_chosen.connect(chosen.append)
        card = picker.card("Elevated Plus Maze")
        QTest.mouseClick(card, Qt.MouseButton.LeftButton)
        self.assertEqual(chosen, ["Elevated Plus Maze"])
        self.assertEqual(picker.selected(), "Elevated Plus Maze")
        self.assertTrue(card.isChecked())
        QTest.mouseClick(card, Qt.MouseButton.LeftButton)       # already chosen
        self.assertEqual(chosen, ["Elevated Plus Maze"])
        self.assertTrue(card.isChecked())

    def test_only_one_card_is_selected(self):
        picker = self.show(ParadigmPicker(), 1000, 200)
        QTest.mouseClick(picker.card("Y-Maze"), Qt.MouseButton.LeftButton)
        QTest.mouseClick(picker.card("Tail Suspension Test"), Qt.MouseButton.LeftButton)
        checked = [n for n in picker.names() if picker.card(n).isChecked()]
        self.assertEqual(checked, ["Tail Suspension Test"])

    def test_set_selected_is_silent_unless_asked(self):
        picker = ParadigmPicker()
        chosen = []
        picker.paradigm_chosen.connect(chosen.append)
        picker.set_selected("Y-Maze")
        self.assertEqual(chosen, [])
        picker.set_selected("Tail Suspension Test", emit=True)
        self.assertEqual(chosen, ["Tail Suspension Test"])
        picker.set_selected(None)
        self.assertIsNone(picker.selected())
        self.assertFalse(any(picker.card(n).isChecked() for n in picker.names()))
        with self.assertRaises(KeyError):
            picker.set_selected("Radial Arm Maze")

    def test_keyboard_selection(self):
        picker = self.show(ParadigmPicker(), 1000, 200)
        chosen = []
        picker.paradigm_chosen.connect(chosen.append)
        first = picker.card("Y-Maze")
        first.setFocus()
        QTest.keyClick(first, Qt.Key.Key_Return)
        self.assertEqual(chosen, ["Y-Maze"])
        QTest.keyClick(first, Qt.Key.Key_Right)
        self.assertEqual(chosen, ["Y-Maze", "Elevated Plus Maze"])
        QTest.keyClick(picker.card("Elevated Plus Maze"), Qt.Key.Key_Down)
        QTest.keyClick(picker.card("Freestyle / Open Field"), Qt.Key.Key_Left)
        self.assertEqual(chosen[-1], "Elevated Plus Maze")
        QTest.keyClick(picker.card("Elevated Plus Maze"), Qt.Key.Key_Left)
        QTest.keyClick(picker.card("Y-Maze"), Qt.Key.Key_Left)    # the edge: no change
        self.assertEqual(picker.selected(), "Y-Maze")

    def test_space_selects_the_focused_card(self):
        picker = self.show(ParadigmPicker(), 1000, 200)
        chosen = []
        picker.paradigm_chosen.connect(chosen.append)
        card = picker.card("Tail Suspension Test")
        card.setFocus()
        QTest.keyClick(card, Qt.Key.Key_Space)
        self.assertEqual(chosen, ["Tail Suspension Test"])

    def test_cards_reflow_with_the_width(self):
        picker = self.show(ParadigmPicker(), 1100, 160)
        self.assertEqual(picker.columns(), 4)
        picker.resize(560, 300)
        QCoreApplication.processEvents()
        self.assertEqual(picker.columns(), 2)
        picker.resize(300, 600)
        QCoreApplication.processEvents()
        self.assertEqual(picker.columns(), 1)

    def test_unknown_paradigms_still_get_a_card(self):
        picker = ParadigmPicker(["Y-Maze", "Radial Arm"])
        self.assertEqual(picker.card("Radial Arm").description(), "Custom paradigm")

    def test_selected_card_is_painted_differently(self):
        picker = self.show(ParadigmPicker(), 1100, 160)
        picker.set_selected("Y-Maze")
        image = self.grab(picker)
        card = picker.card("Y-Maze")
        other = picker.card("Elevated Plus Maze")
        self.assertEqual(self.hex_at(image, card.x() + card.width() // 2, card.y() + card.height() - 7),
                         theme.LIGHT.tint("primary", 0.08))
        self.assertEqual(self.hex_at(image, other.x() + other.width() // 2,
                                     other.y() + other.height() - 7), theme.LIGHT.surface)


class FilmstripTests(KitCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.paths = [write_indexed_video(self.root / f"m{i}.avi", frame_count=4)
                      for i in range(3)]
        self.items = [FilmstripItem(f"k{i}", path) for i, path in enumerate(self.paths)]

    def make(self, **kwargs):
        strip = Filmstrip(**kwargs)
        self.addCleanup(strip.thumbnail_loader.shutdown)
        strip.set_items(self.items)
        self.show(strip, 700, 260 if strip.orientation() == Qt.Orientation.Vertical else 200)
        return strip

    def click_row(self, strip, row):
        rect = strip.view.visualRect(strip._model.index(row))
        QTest.mouseClick(strip.view.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, rect.center())

    def test_items_and_defaults(self):
        strip = self.make()
        self.assertEqual(strip.keys(), ["k0", "k1", "k2"])
        self.assertEqual(strip.items()[0].name, "m0.avi")
        self.assertIsNone(strip.selected_key())
        self.assertEqual(strip.count_label.text(), "(3)")

    def test_clicking_a_video_selects_and_emits(self):
        strip = self.make()
        chosen = []
        strip.video_selected.connect(chosen.append)
        self.click_row(strip, 1)
        self.assertEqual(chosen, ["k1"])
        self.assertEqual(strip.selected_key(), "k1")
        self.click_row(strip, 2)
        self.assertEqual(chosen, ["k1", "k2"])

    def test_select_is_silent_unless_asked(self):
        strip = self.make()
        chosen = []
        strip.video_selected.connect(chosen.append)
        strip.select("k2")
        self.assertEqual((chosen, strip.selected_key()), ([], "k2"))
        strip.select("k0", emit=True)
        self.assertEqual(chosen, ["k0"])
        strip.select(None)
        self.assertIsNone(strip.selected_key())
        strip.select("missing")
        self.assertIsNone(strip.selected_key())

    def test_selection_survives_a_refresh(self):
        strip = self.make()
        strip.select("k1")
        chosen = []
        strip.video_selected.connect(chosen.append)
        strip.set_items(self.items[1:])
        self.assertEqual(strip.selected_key(), "k1")
        self.assertEqual(chosen, [])
        strip.set_items(self.items[:1])
        self.assertIsNone(strip.selected_key())

    def test_badges(self):
        strip = self.make()
        self.assertTrue(strip.set_badge("k1", "Done", "ok"))
        item = strip.items()[1]
        self.assertEqual((item.badge_text, item.badge_kind), ("Done", "ok"))
        self.assertFalse(strip.set_badge("zzz", "x", "ok"))
        with self.assertRaises(ValueError):
            strip.set_badge("k1", "x", "purple")

    def test_thumbnails_load_lazily_for_visible_items(self):
        strip = self.make()
        loader = strip.thumbnail_loader
        self.assertIsNotNone(self.grab(strip))
        self.assertTrue(loader.wait_idle())
        for path in self.paths:
            preview = loader.get(path)
            self.assertIsNotNone(preview, path)
            self.assertTrue(preview.ok)

    def test_a_shared_loader_is_used(self):
        from gui.workspace.thumbnails import ThumbnailLoader
        loader = ThumbnailLoader()
        self.addCleanup(loader.shutdown)
        strip = Filmstrip(loader=loader)
        self.addCleanup(strip.thumbnail_loader.shutdown)
        self.assertIs(strip.thumbnail_loader, loader)

    def test_thumbnail_arrival_repaints(self):
        strip = self.make()
        before = self.grab(strip)
        strip.thumbnail_loader.wait_idle()
        self.grab(strip)
        strip.thumbnail_loader.wait_idle()
        after = self.grab(strip)
        self.assertNotEqual(before, after)

    def test_missing_files_do_not_break_painting(self):
        strip = Filmstrip()
        self.addCleanup(strip.thumbnail_loader.shutdown)
        strip.set_items([FilmstripItem("x", self.root / "gone.mp4", badge_text="Missing",
                                       badge_kind="error")])
        self.show(strip, 500, 200)
        self.assertFalse(self.grab(strip).isNull())

    def test_compact_toggle(self):
        strip = self.make()
        seen = []
        strip.compact_changed.connect(seen.append)
        self.assertFalse(strip.is_compact())
        strip.toggle_button.click()
        self.assertTrue(strip.is_compact())
        strip.set_compact(True)
        strip.toggle_button.click()
        self.assertEqual(seen, [True, False])
        self.assertFalse(strip.is_compact())

    def test_compact_rows_are_shorter_than_thumbnail_cards(self):
        strip = self.make()
        card_height = strip.view.visualRect(strip._model.index(0)).height()
        strip.set_compact(True)
        QCoreApplication.processEvents()
        row_height = strip.view.visualRect(strip._model.index(0)).height()
        self.assertLess(row_height, card_height)
        self.assertEqual(row_height, 32)

    def test_compact_state_is_remembered(self):
        settings = temp_app_settings(self)
        first = self.make(settings=settings, settings_key="videos")
        first.set_compact(True)
        second = Filmstrip(settings=settings, settings_key="videos")
        self.addCleanup(second.thumbnail_loader.shutdown)
        self.assertTrue(second.is_compact())

    def test_horizontal_cards_flow_left_to_right_and_vertical_top_to_bottom(self):
        horizontal = self.make()
        first = horizontal.view.visualRect(horizontal._model.index(0))
        second = horizontal.view.visualRect(horizontal._model.index(1))
        self.assertGreater(second.x(), first.x())
        self.assertEqual(second.y(), first.y())
        vertical = self.make(orientation=Qt.Orientation.Vertical)
        first = vertical.view.visualRect(vertical._model.index(0))
        second = vertical.view.visualRect(vertical._model.index(1))
        self.assertGreater(second.y(), first.y())
        self.assertEqual(second.x(), first.x())

    def test_keyboard_navigation_changes_the_selection(self):
        strip = self.make()
        chosen = []
        strip.video_selected.connect(chosen.append)
        self.click_row(strip, 0)
        QTest.keyClick(strip.view, Qt.Key.Key_Right)
        self.assertEqual(chosen, ["k0", "k1"])

    def test_header_can_be_hidden(self):
        strip = Filmstrip(show_header=False)
        self.addCleanup(strip.thumbnail_loader.shutdown)
        self.show(strip, 400, 200)
        self.assertFalse(strip.title_label.isVisibleTo(strip))

    def test_selected_card_is_highlighted(self):
        strip = self.make()
        strip.select("k0")
        image = self.grab(strip)
        rect = strip.view.visualRect(strip._model.index(0))
        x = strip.view.mapTo(strip, rect.topLeft()).x() + rect.width() // 2
        y = strip.view.mapTo(strip, rect.topLeft()).y() + rect.height() - 6
        self.assertEqual(self.hex_at(image, x, y), theme.LIGHT.tint("primary", 0.10))


class ThemeFollowTests(KitCase):
    def test_widgets_are_dark_after_a_switch(self):
        chip = self.show(StatusChip("ok", "Done"))
        theme.apply_theme(self.app, "dark")
        try:
            soft = theme.DARK.kind_colors("ok")[1].upper()
            self.assertEqual(self.hex_at(self.grab(chip), chip.width() - 4, 11), soft)
            card = self.show(Card("x"), 200, 100)
            self.assertEqual(self.hex_at(self.grab(card), 100, 90), theme.DARK.surface)
        finally:
            theme.apply_theme(self.app, "light")


class CalloutTitleTests(KitCase):
    def test_the_title_can_be_set_and_cleared(self):
        callout = Callout("warning", "text")
        self.show(callout, 300)
        self.assertEqual(callout.title(), "")
        callout.set_title("Quality warnings")
        self.assertEqual(callout.title(), "Quality warnings")
        self.assertTrue(callout._title.isVisibleTo(callout))
        callout.set_title("")
        self.assertFalse(callout._title.isVisibleTo(callout))


class ModuleSanityTests(KitCase):
    def test_the_module_does_not_use_qobject_sender(self):
        source = (Path(__file__).resolve().parent.parent / "gui" / "workspace" / "ui_kit.py"
                  ).read_text(encoding="utf-8")
        self.assertNotIn(".sender()", source)

    def test_every_public_widget_can_be_built_without_arguments_it_does_not_need(self):
        for factory in (StatusChip, lambda: KeyHint("S"), lambda: Callout("info", "x"),
                        Card, lambda: CollapsibleSection("t"), lambda: EmptyState("video", "t"),
                        lambda: StageHeader("t"), FooterBar, Stepper, DropZone, ParadigmPicker,
                        Filmstrip, StatTile, QWidget):
            widget = factory()
            self.assertIsNotNone(widget)
            widget.deleteLater()


if __name__ == "__main__":
    unittest.main()
