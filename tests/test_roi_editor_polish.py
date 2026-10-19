"""ROI editor polish: point snapping, the shortcut help, the status bar."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

try:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from gui.roi_editor import TOOL_DRAW, TOOL_EDIT, TOOL_LINE
    from gui.roi_editor.help import SHORTCUT_GROUPS, help_html
    from roi_manager import ROIManager
    from tests.test_roi_editor import LEFT, SQUARE, _EditorCase, _square
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False
    _EditorCase = unittest.TestCase


# ----------------------------------------------------------------------
# Snapping
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class SnappingTests(_EditorCase):
    def _neighbour(self, zoom=1.0):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_a", np.array(SQUARE))
        widget = self.editor(manager=manager, zoom=zoom)
        widget.set_current_zone("arm_b")
        self.assertEqual(widget.tool(), TOOL_DRAW)
        return widget

    def test_a_new_point_snaps_to_a_vertex_within_six_screen_pixels(self):
        widget = self._neighbour()
        self.click(widget, 203, 98)               # about 3.5 px from (200, 100)
        self.assertEqual(widget.pending_points(), [(200, 100)])
        self.assertEqual(widget.session.snap_radius(), 6.0)

    def test_a_point_outside_the_radius_is_left_alone(self):
        widget = self._neighbour()
        self.click(widget, 209, 100)
        self.assertEqual(widget.pending_points(), [(209, 100)])

    def test_the_radius_is_in_screen_pixels_so_zoom_changes_it(self):
        close = (202, 101)                        # about 2.2 source px away
        zoomed_in = self._neighbour(zoom=4.0)
        self.click(zoomed_in, *close)
        self.assertNotEqual(zoomed_in.pending_points()[0], (200, 100))
        self.assertAlmostEqual(zoomed_in.session.snap_radius(), 1.5)
        zoomed_out = self._neighbour(zoom=1.0)
        self.click(zoomed_out, *close)
        self.assertEqual(zoomed_out.pending_points()[0], (200, 100))

    def test_adjacent_zones_share_an_exact_border(self):
        widget = self._neighbour()
        # Next to arm A's right edge, clicked a few pixels off its vertices.
        self.draw(widget, [(198, 103), (260, 99), (260, 200), (203, 197)])
        shape = self.shapes(widget, "arm_b")[0]
        self.assertEqual(shape.points[0], (200, 100))
        self.assertEqual(shape.points[3], (200, 200))
        # The two zones now meet along x = 200 with no gap or overlap.
        arm_a = self.shapes(widget, "arm_a")[0].points
        shared = [p for p in shape.points if p in arm_a]
        self.assertEqual(sorted(shared), [(200, 100), (200, 200)])

    def test_dragging_a_vertex_snaps_to_a_neighbour_and_alt_turns_it_off(self):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_a", np.array(SQUARE))
        manager.add_roi("arm_b", np.array(_square(220, 100, 80)))
        widget = self.editor(manager=manager)
        widget.set_tool(TOOL_EDIT)
        session = widget.session

        session.press((220, 100))
        session.move((203, 98))
        self.assertEqual(self.shapes(widget, "arm_b")[0].points[0], (200, 100))
        self.assertEqual(session.snap_target, (200.0, 100.0))
        session.release((203, 98))
        self.assertIsNone(session.snap_target)
        widget.undo()

        session.press((220, 100))
        session.move((203, 98), Qt.KeyboardModifier.AltModifier)
        session.release((203, 98), Qt.KeyboardModifier.AltModifier)
        self.assertEqual(self.shapes(widget, "arm_b")[0].points[0], (203, 98))

    def test_a_dragged_vertex_does_not_snap_to_itself(self):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_a", np.array(SQUARE))
        widget = self.editor(manager=manager)
        widget.set_tool(TOOL_EDIT)
        widget.session.press((100, 100))
        widget.session.move((104, 103))
        widget.session.release((104, 103))
        self.assertEqual(self.shapes(widget)[0].points[0], (104, 103))

    def test_snapping_can_be_switched_off(self):
        widget = self._neighbour()
        widget.session.snapping = False
        self.click(widget, 203, 98)
        self.assertEqual(widget.pending_points(), [(203, 98)])

    def test_holding_alt_while_clicking_does_not_snap(self):
        widget = self._neighbour()
        self.click(widget, 203, 98, modifiers=Qt.KeyboardModifier.AltModifier)
        self.assertEqual(widget.pending_points(), [(203, 98)])

    def test_reference_line_ends_snap_to_vertices_too(self):
        widget = self._neighbour()
        widget.set_tool(TOOL_LINE)
        self.click(widget, 103, 98)
        self.click(widget, 198, 203)
        (x1, y1), (x2, y2) = widget.line_points()
        # Snapped onto the pixel centres of (100, 100) and (200, 200).
        self.assertEqual((x1, y1), (100.5, 100.5))
        self.assertEqual((x2, y2), (200.5, 200.5))


# ----------------------------------------------------------------------
# Help overlay and status bar
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class HelpAndStatusTests(_EditorCase):
    def test_question_mark_and_the_button_toggle_the_shortcut_overlay(self):
        widget = self.editor()
        self.assertFalse(widget.help_visible())
        QTest.keyClick(widget.canvas, Qt.Key.Key_Question,
                       Qt.KeyboardModifier.ShiftModifier)
        self.assertTrue(widget.help_visible())
        self.assertTrue(widget.canvas.help_overlay.isVisible())
        QTest.keyClick(widget.canvas, Qt.Key.Key_F1)
        self.assertFalse(widget.help_visible())
        widget.help_button.click()
        self.assertTrue(widget.help_visible())
        text = widget.canvas.help_overlay.text()
        for fragment in ("Ctrl+Z", "Enter", "Space", "Alt", "Delete", "Ctrl + wheel"):
            self.assertIn(fragment, text)

    def test_help_lists_every_group_and_is_not_empty(self):
        html = help_html()
        for heading, entries in SHORTCUT_GROUPS:
            self.assertIn(heading, html)
            self.assertTrue(entries)

    def test_escape_closes_the_help_before_cancelling_a_shape(self):
        widget = self.editor()
        widget.add_point(100, 100)
        widget.toggle_help()
        QTest.keyClick(widget.canvas, Qt.Key.Key_Escape)
        self.assertFalse(widget.help_visible())
        self.assertEqual(len(widget.pending_points()), 1)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Escape)
        self.assertEqual(widget.pending_points(), [])

    def test_clicking_the_overlay_closes_it(self):
        widget = self.editor()
        widget.toggle_help()
        QTest.mouseClick(widget.canvas.help_overlay, LEFT)
        self.assertFalse(widget.help_visible())

    def test_status_bar_shows_cursor_position_in_source_pixels_and_zoom(self):
        widget = self.editor(size=(640, 480), zoom=2.0)
        self.assertEqual(widget.zoom_label.text(), "Zoom 200%")
        viewport = widget.canvas.viewport()
        QTest.mouseMove(viewport, self.vp(widget, 150, 100))
        self.app.processEvents()
        self.assertIn("x: 150", widget.cursor_label.text())
        self.assertIn("y: 100", widget.cursor_label.text())
        QTest.mouseMove(viewport, QPoint(900, 650))          # outside the frame
        self.app.processEvents()
        self.assertEqual(widget.cursor_label.text(), "x: -  y: -")

    def test_fit_button_refits_and_reports_the_zoom(self):
        widget = self.editor(size=(2000, 1500), zoom=1.0)
        widget.fit_button.click()
        self.assertLess(widget.zoom_percent(), 100.0)
        self.assertEqual(widget.zoom_label.text(),
                         f"Zoom {widget.zoom_percent():.0f}%")
        widget.actual_button.click()
        self.assertEqual(widget.zoom_label.text(), "Zoom 100%")

    def test_issue_summary_follows_the_zones(self):
        widget = self.editor()
        self.assertEqual(widget.issue_label.text(), "4 to fix")
        for category, x in (("arm_a", 10), ("arm_b", 100), ("arm_c", 190),
                            ("center", 280)):
            widget.set_current_zone(category)
            self.draw(widget, _square(x, 10))
        self.assertEqual(widget.issue_label.text(), "All zones complete")


if __name__ == "__main__":
    unittest.main()
