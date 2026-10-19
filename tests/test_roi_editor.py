"""The ROI editor widget: drawing, editing, undo, zones, line, zoom, load/save."""
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

try:
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from gui.roi_editor import (MIN_ROI_AREA_PX, RoiEditorWidget, RoiRule,
                                TOOL_DRAW, TOOL_EDIT, TOOL_LINE)
    from gui.roi_editor.canvas import to_model, to_scene
    from gui.roi_editor.rules import (STATUS_DONE, STATUS_MISSING,
                                      STATUS_VIOLATED, distance_to_segment,
                                      evaluate_zone, point_in_polygon,
                                      polygon_area, rotate_about, scale_about,
                                      snap_point)
    from mazes import create_maze
    from mazes.freestyle import Freestyle
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

LEFT = Qt.MouseButton.LeftButton if HAS_RUNTIME else None
SQUARE = [(100, 100), (200, 100), (200, 200), (100, 200)]


def _square(x, y, size=80):
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]


class _EditorCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def editor(self, maze=None, size=(400, 300), manager=None, zoom=1.0):
        widget = RoiEditorWidget(
            maze if maze is not None else create_maze("Y-Maze"),
            np.zeros((size[1], size[0], 3), dtype=np.uint8),
            initial_roi_manager=manager,
        )
        widget.resize(1000, 700)
        # Top-left alignment keeps scene and viewport pixels aligned, so a
        # click lands on an exact source pixel at 100%.
        widget.canvas.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        widget.show()
        widget.set_zoom(zoom)
        self.app.processEvents()
        self.addCleanup(self._dispose, widget)
        return widget

    @staticmethod
    def _dispose(widget):
        # Hidden at once: a lingering window would steal later mouse events.
        widget.hide()
        widget.deleteLater()

    # -- real mouse events, in source pixels ---------------------------
    def vp(self, widget, x, y):
        return widget.canvas.view_pos_for_model((x, y))

    def click(self, widget, x, y, button=LEFT, modifiers=Qt.KeyboardModifier.NoModifier):
        QTest.mouseClick(widget.canvas.viewport(), button, modifiers,
                         self.vp(widget, x, y))

    def double_click(self, widget, x, y):
        # What a real double-click delivers: press, release, then the
        # double-click event (QTest's mouseDClick sends only the latter).
        self.click(widget, x, y)
        QTest.mouseDClick(widget.canvas.viewport(), LEFT,
                          Qt.KeyboardModifier.NoModifier, self.vp(widget, x, y))

    def drag(self, widget, start, end):
        viewport = widget.canvas.viewport()
        QTest.mousePress(viewport, LEFT, Qt.KeyboardModifier.NoModifier,
                         self.vp(widget, *start))
        QTest.mouseMove(viewport, self.vp(widget, *end))
        QTest.mouseRelease(viewport, LEFT, Qt.KeyboardModifier.NoModifier,
                           self.vp(widget, *end))

    def draw(self, widget, points, close=True):
        for x, y in points:
            self.click(widget, x, y)
        if close:
            QTest.keyClick(widget.canvas, Qt.Key.Key_Enter)

    def shapes(self, widget, category=None):
        shapes = widget.session.document.shapes
        return [s for s in shapes if category is None or s.category == category]


# ----------------------------------------------------------------------
# Pure rules and geometry
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RuleTests(unittest.TestCase):
    def test_zone_status_done_missing_violated(self):
        rule = RoiRule(1, 1)
        self.assertEqual(evaluate_zone("m", "Mouse 1", rule, []).status, STATUS_MISSING)
        self.assertEqual(evaluate_zone("m", "Mouse 1", rule, [500.0]).status, STATUS_DONE)
        two = evaluate_zone("m", "Mouse 1", rule, [500.0, 500.0])
        self.assertEqual(two.status, STATUS_VIOLATED)
        self.assertIn("exactly 1 polygon", two.issues[0])
        small = evaluate_zone("m", "Mouse 1", rule, [MIN_ROI_AREA_PX - 1])
        self.assertEqual(small.status, STATUS_VIOLATED)
        self.assertIn("too small", small.issues[0])

    def test_open_ended_zone_accepts_many_polygons(self):
        status = evaluate_zone("a", "Arm A", RoiRule(1, None), [200.0, 300.0, 400.0])
        self.assertTrue(status.ok)
        self.assertEqual(status.count, 3)

    def test_geometry_helpers(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertEqual(polygon_area(square), 100.0)
        self.assertTrue(point_in_polygon((5, 5), square))
        self.assertFalse(point_in_polygon((15, 5), square))
        distance, closest = distance_to_segment((5, 4), (0, 0), (10, 0))
        self.assertEqual((distance, closest), (4.0, (5.0, 0.0)))
        rotated = rotate_about([(10, 0)], (0, 0), math.pi / 2)
        self.assertAlmostEqual(rotated[0][0], 0.0)
        self.assertAlmostEqual(rotated[0][1], 10.0)
        self.assertEqual(scale_about([(10, 0)], (0, 0), 0.5), [(5.0, 0.0)])

    def test_snap_point_picks_nearest_within_radius(self):
        candidates = [(10, 10), (12, 10)]
        self.assertEqual(snap_point((11.4, 10), candidates, 3.0), (12.0, 10.0))
        self.assertEqual(snap_point((50, 50), candidates, 3.0), (50, 50))


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class DrawingTests(_EditorCase):
    def test_click_points_and_enter_make_a_polygon(self):
        widget = self.editor()
        self.assertEqual(widget.tool(), TOOL_DRAW)
        self.assertEqual(widget.current_zone(), "arm_a")
        self.draw(widget, SQUARE[:3], close=False)
        self.assertEqual(len(widget.pending_points()), 3)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Return)
        self.assertEqual(widget.pending_points(), [])
        shapes = self.shapes(widget, "arm_a")
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0].points, SQUARE[:3])

    def test_double_click_closes_the_shape(self):
        widget = self.editor()
        self.click(widget, 100, 100)
        self.click(widget, 200, 100)
        self.click(widget, 200, 200)
        self.double_click(widget, 100, 200)
        shapes = self.shapes(widget, "arm_a")
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0].points, SQUARE)

    def test_right_click_also_closes_a_pending_shape(self):
        widget = self.editor()
        self.draw(widget, SQUARE[:3], close=False)
        self.click(widget, 150, 150, Qt.MouseButton.RightButton)
        self.assertEqual(len(self.shapes(widget, "arm_a")), 1)

    def test_escape_cancels_only_the_shape_in_progress(self):
        widget = self.editor()
        self.draw(widget, SQUARE[:3], close=False)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Escape)
        self.assertEqual(widget.pending_points(), [])
        self.assertEqual(self.shapes(widget), [])
        self.assertFalse(widget.has_unsaved_work())

    def test_escape_with_nothing_pending_is_left_for_the_dialog(self):
        widget = self.editor()
        self.assertFalse(widget.session.key_press(Qt.Key.Key_Escape,
                                                  Qt.KeyboardModifier.NoModifier))

    def test_backspace_and_ctrl_z_remove_the_last_pending_point(self):
        widget = self.editor()
        self.draw(widget, SQUARE, close=False)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Backspace)
        self.assertEqual(len(widget.pending_points()), 3)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(len(widget.pending_points()), 2)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(len(widget.pending_points()), 3)

    def test_too_small_shape_is_refused_with_a_hint(self):
        widget = self.editor()
        self.draw(widget, [(10, 10), (15, 10), (15, 15)])
        self.assertEqual(self.shapes(widget), [])
        self.assertIn("too small", widget.hint_label.text())
        self.assertEqual(len(widget.pending_points()), 3)   # still editable
        self.assertGreater(MIN_ROI_AREA_PX, 12.5)

    def test_two_points_are_not_enough(self):
        widget = self.editor()
        self.draw(widget, SQUARE[:2])
        self.assertEqual(self.shapes(widget), [])
        self.assertIn("at least three points", widget.hint_label.text())

    def test_zones_can_be_drawn_in_any_order(self):
        widget = self.editor()
        self.assertTrue(widget.set_current_zone("arm_c"))
        self.draw(widget, _square(10, 10))
        self.assertTrue(widget.set_current_zone("arm_a"))
        self.draw(widget, _square(150, 10))
        self.assertEqual(self.shapes(widget, "arm_c")[0].points, _square(10, 10))
        self.assertEqual(self.shapes(widget, "arm_a")[0].points, _square(150, 10))
        self.assertEqual(widget.zone_status("arm_b").status, STATUS_MISSING)

        # The ROIManager follows the maze's zone order, not the drawing order.
        manager = widget.roi_manager()
        self.assertEqual(list(manager.rois), ["arm_a", "arm_c"])

    def test_picking_a_zone_with_a_shape_in_progress_closes_or_refuses(self):
        widget = self.editor()
        self.draw(widget, SQUARE, close=False)
        self.assertTrue(widget.set_current_zone("arm_b"))     # closed into arm_a
        self.assertEqual(len(self.shapes(widget, "arm_a")), 1)
        self.draw(widget, [(5, 5), (8, 5)], close=False)
        self.assertFalse(widget.set_current_zone("arm_c"))    # too few points
        self.assertEqual(widget.current_zone(), "arm_b")
        self.assertIn("Finish the shape", widget.hint_label.text())

    def test_enter_with_nothing_pending_moves_to_the_next_missing_zone(self):
        widget = self.editor()
        self.draw(widget, SQUARE)
        self.assertEqual(widget.current_zone(), "arm_a")
        QTest.keyClick(widget.canvas, Qt.Key.Key_Return)
        self.assertEqual(widget.current_zone(), "arm_b")
        self.assertEqual(widget.tool(), TOOL_DRAW)

    def test_open_ended_zone_takes_several_polygons(self):
        widget = self.editor()
        self.draw(widget, _square(10, 10))
        self.assertTrue(widget.set_tool(TOOL_DRAW))
        self.draw(widget, _square(150, 10))
        self.assertEqual(len(self.shapes(widget, "arm_a")), 2)
        self.assertTrue(widget.zone_status("arm_a").ok)

    def test_points_are_clamped_to_the_frame(self):
        widget = self.editor(size=(200, 150))
        widget.add_point(-30, 400)
        self.assertEqual(widget.pending_points(), [(0, 149)])


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class TstRuleTests(_EditorCase):
    def _tst(self, mice):
        maze = create_maze("Tail Suspension Test")
        maze.configure_mice(mice)
        return maze

    def test_one_polygon_per_mouse_and_advance_to_the_next_mouse(self):
        widget = self.editor(self._tst(2))
        self.assertEqual(widget.current_zone(), "mouse_1")
        self.draw(widget, _square(10, 10))
        self.assertEqual(widget.current_zone(), "mouse_2")
        self.assertTrue(widget.zone_status("mouse_1").ok)

    def test_a_second_polygon_for_one_mouse_is_refused(self):
        widget = self.editor(self._tst(1))
        self.draw(widget, _square(10, 10))
        self.assertTrue(widget.set_tool(TOOL_DRAW))
        self.assertFalse(widget.add_point(150, 150))
        self.assertEqual(widget.pending_points(), [])
        self.assertIn("complete", widget.hint_label.text())

    def test_loaded_zero_or_many_polygons_are_flagged(self):
        manager = ROIManager((400, 300))
        manager.add_roi("mouse_1", np.array(_square(10, 10)))
        manager.add_roi("mouse_1", np.array(_square(150, 10)))
        widget = self.editor(self._tst(2), manager=manager)
        self.assertEqual(widget.zone_status("mouse_1").status, STATUS_VIOLATED)
        self.assertEqual(widget.zone_status("mouse_2").status, STATUS_MISSING)
        self.assertFalse(widget.is_valid())
        self.assertEqual(len(widget.issues()), 2)
        # No calibration row for a paradigm without distances.
        self.assertNotIn(TOOL_LINE, widget.tool_buttons)

    def test_an_edited_shape_that_becomes_too_small_is_flagged(self):
        widget = self.editor(self._tst(1))
        self.draw(widget, _square(10, 10, 40))
        self.assertTrue(widget.is_valid())
        shape = self.shapes(widget)[0]
        widget.session.nudge(0, 0)
        widget.set_tool(TOOL_EDIT)
        widget.session.select(shape.id, 2)
        widget.session.press((50, 50))          # on vertex 2 (50, 50)
        widget.session.move((12, 12))
        widget.session.release((12, 12))
        self.assertFalse(widget.is_valid())
        self.assertIn("too small", widget.issues()[0])


# ----------------------------------------------------------------------
# Editing and undo
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class EditingTests(_EditorCase):
    def _with_square(self, **kwargs):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_a", np.array(SQUARE))
        widget = self.editor(manager=manager, **kwargs)
        self.assertEqual(widget.tool(), TOOL_DRAW)   # other zones still missing
        widget.set_tool(TOOL_EDIT)
        return widget

    def test_dragging_a_vertex_is_one_undo_step(self):
        widget = self._with_square()
        self.drag(widget, (200, 200), (230, 240))
        shape = self.shapes(widget)[0]
        self.assertEqual(shape.points, [(100, 100), (200, 100), (230, 240), (100, 200)])
        self.assertEqual(widget.undo_stack.count(), 1)
        widget.undo()
        self.assertEqual(self.shapes(widget)[0].points, SQUARE)
        widget.redo()
        self.assertEqual(self.shapes(widget)[0].points[2], (230, 240))

    def test_dragging_the_body_moves_the_whole_polygon(self):
        widget = self._with_square()
        self.drag(widget, (150, 150), (170, 130))
        self.assertEqual(self.shapes(widget)[0].points, _square(120, 80, 100))
        self.assertEqual(widget.undo_stack.count(), 1)

    def test_moving_a_polygon_cannot_leave_the_frame(self):
        widget = self._with_square()
        widget.session.select(self.shapes(widget)[0].id)
        widget.session.press((150, 150))
        widget.session.move((900, 900))
        widget.session.release((900, 900))
        points = self.shapes(widget)[0].points
        self.assertEqual(max(p[0] for p in points), 399)
        self.assertEqual(max(p[1] for p in points), 299)
        self.assertEqual(points[2][0] - points[0][0], 100)   # shape unchanged

    def test_a_click_without_movement_is_not_an_edit(self):
        widget = self._with_square()
        self.click(widget, 150, 150)
        self.click(widget, 200, 200)
        self.assertEqual(widget.undo_stack.count(), 0)
        self.assertFalse(widget.has_unsaved_work())

    def test_double_click_on_an_edge_inserts_a_vertex(self):
        widget = self._with_square()
        self.double_click(widget, 150, 100)
        points = self.shapes(widget)[0].points
        self.assertEqual(len(points), 5)
        self.assertEqual(points[1], (150, 100))
        widget.undo()
        self.assertEqual(len(self.shapes(widget)[0].points), 4)

    def test_right_click_deletes_a_vertex_then_the_polygon(self):
        widget = self._with_square()
        self.click(widget, 200, 200, Qt.MouseButton.RightButton)
        self.assertEqual(len(self.shapes(widget)[0].points), 3)
        # Three vertices is the minimum: another vertex delete is refused.
        self.click(widget, 100, 100, Qt.MouseButton.RightButton)
        self.assertEqual(len(self.shapes(widget)[0].points), 3)
        self.assertIn("three vertices", widget.hint_label.text())
        # Right-click inside the polygon removes it.
        self.click(widget, 130, 150, Qt.MouseButton.RightButton)
        self.assertEqual(self.shapes(widget), [])
        widget.undo()
        self.assertEqual(len(self.shapes(widget)), 1)

    def test_delete_key_removes_the_selected_vertex_or_polygon(self):
        widget = self._with_square()
        self.click(widget, 200, 100)                         # select vertex 1
        QTest.keyClick(widget.canvas, Qt.Key.Key_Delete)
        self.assertEqual(len(self.shapes(widget)[0].points), 3)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Delete)     # now the polygon
        self.assertEqual(self.shapes(widget), [])

    def test_arrow_keys_nudge_the_selection(self):
        widget = self._with_square()
        self.click(widget, 150, 150)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Right)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.shapes(widget)[0].points[0], (101, 110))

    def test_rotate_handle_rotates_about_the_centre(self):
        widget = self._with_square()
        session = widget.session
        session.select(self.shapes(widget)[0].id)
        handle = session.transform_handles(self.shapes(widget)[0])["rotate"]
        centre = (150.0, 150.0)
        session.press(handle)
        # Quarter turn clockwise: from straight above to the right of the centre.
        session.move((centre[0] + 60, centre[1]))
        session.release((centre[0] + 60, centre[1]))
        points = self.shapes(widget)[0].points
        self.assertEqual(sorted(points), sorted(
            [(200, 100), (200, 200), (100, 200), (100, 100)]))
        self.assertEqual(widget.undo_stack.count(), 1)
        self.assertEqual(widget.undo_stack.undoText(), "Rotate polygon")

    def test_scale_handle_scales_about_the_centre(self):
        widget = self._with_square()
        session = widget.session
        shape = self.shapes(widget)[0]
        session.select(shape.id)
        handle = session.transform_handles(shape)["scale"]
        start = math.hypot(handle[0] - 150, handle[1] - 150)
        session.press(handle)
        target = (150 + (handle[0] - 150) * 2, 150 + (handle[1] - 150) * 2)
        session.move(target)
        session.release(target)
        self.assertGreater(start, 0)
        points = self.shapes(widget)[0].points
        self.assertEqual(points[0], (50, 50))
        self.assertEqual(points[2], (250, 250))

    def test_the_smallest_polygon_under_the_cursor_is_picked(self):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_a", np.array(_square(150, 100, 40)))
        manager.set_arena(np.array([(20, 20), (380, 20), (380, 280), (20, 280)]))
        widget = self.editor(manager=manager)
        widget.set_tool(TOOL_EDIT)
        self.click(widget, 170, 120)
        selected = widget.session.document.shape(widget.session.selected)
        self.assertEqual(selected.category, "arm_a")
        self.click(widget, 60, 60)
        selected = widget.session.document.shape(widget.session.selected)
        self.assertEqual(selected.category, "arena")

    def test_clearing_a_zone_is_a_single_undo_step(self):
        widget = self.editor()
        self.draw(widget, _square(10, 10))
        widget.set_tool(TOOL_DRAW)
        self.draw(widget, _square(150, 10))
        self.assertEqual(widget.set_current_zone("arm_a"), True)
        self.assertEqual(widget.clear_current_zone(), 2)
        self.assertEqual(self.shapes(widget), [])
        widget.undo()
        self.assertEqual(len(self.shapes(widget, "arm_a")), 2)

    def test_undo_redo_cover_every_kind_of_edit(self):
        widget = self.editor()
        self.draw(widget, SQUARE)
        widget.set_tool(TOOL_EDIT)
        self.drag(widget, (200, 200), (210, 210))
        self.double_click(widget, 150, 100)
        self.click(widget, 100, 100, Qt.MouseButton.RightButton)
        widget.set_line((10, 10), (110, 10))
        steps = widget.undo_stack.count()
        self.assertEqual(steps, 5)
        for _ in range(steps):
            self.assertTrue(widget.undo())
        self.assertEqual(self.shapes(widget), [])
        self.assertIsNone(widget.line_points())
        for _ in range(steps):
            self.assertTrue(widget.redo())
        self.assertEqual(widget.line_points(), ((10.0, 10.0), (110.0, 10.0)))
        QTest.keyClick(widget.canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        self.assertIsNone(widget.line_points())
        QTest.keyClick(widget.canvas, Qt.Key.Key_Z,
                       Qt.KeyboardModifier.ControlModifier
                       | Qt.KeyboardModifier.ShiftModifier)
        self.assertIsNotNone(widget.line_points())

    def test_unsaved_work_follows_the_undo_stack(self):
        widget = self.editor()
        self.assertFalse(widget.has_unsaved_work())
        self.draw(widget, SQUARE)
        self.assertTrue(widget.has_unsaved_work())
        widget.undo()
        self.assertFalse(widget.has_unsaved_work())
        widget.redo()
        widget.mark_saved()
        self.assertFalse(widget.has_unsaved_work())
        widget.undo()
        self.assertTrue(widget.has_unsaved_work())

    def test_edit_mode_selecting_a_shape_makes_its_zone_current(self):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_a", np.array(_square(10, 10)))
        manager.add_roi("arm_b", np.array(_square(150, 10)))
        widget = self.editor(manager=manager)
        widget.set_tool(TOOL_EDIT)
        self.click(widget, 180, 40)
        self.assertEqual(widget.current_zone(), "arm_b")


# ----------------------------------------------------------------------
# Zone list
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ZoneListTests(_EditorCase):
    def test_one_row_per_zone_plus_the_reference_line(self):
        widget = self.editor()
        keys = [widget.zone_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(widget.zone_list.count())]
        self.assertEqual(keys[:4], ["arm_a", "arm_b", "arm_c", "center"])
        self.assertEqual(widget.zone_list.count(), 5)
        self.assertIn("missing", widget.zone_list.item(0).text())

    def test_clicking_a_row_makes_that_zone_current_at_any_time(self):
        widget = self.editor()
        item = widget.zone_list.item(2)
        widget.zone_list.setCurrentItem(item)
        widget.zone_list.itemClicked.emit(item)
        self.assertEqual(widget.current_zone(), "arm_c")
        self.assertEqual(widget.tool(), TOOL_DRAW)
        self.assertEqual(widget.zone_list.currentRow(), 2)
        self.draw(widget, _square(20, 20))
        self.assertIn("1 polygon", widget.zone_list.item(2).text())

    def test_done_zone_opens_for_editing_and_the_reference_row_picks_the_tool(self):
        manager = ROIManager((400, 300))
        for category in ("arm_a", "arm_b", "arm_c", "center"):
            manager.add_roi(category, np.array(_square(10 + 60 * ("abc".find(category[-1]) + 1), 10)))
        widget = self.editor(manager=manager)
        self.assertEqual(widget.tool(), TOOL_EDIT)
        self.assertTrue(widget.is_valid())
        row = widget.zone_list.item(4)
        widget.zone_list.itemClicked.emit(row)
        self.assertEqual(widget.tool(), TOOL_LINE)
        self.assertEqual(widget.zone_list.currentRow(), 4)

    def test_rows_show_rule_violations(self):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_a", np.array([(10, 10), (14, 10), (14, 14)]))
        widget = self.editor(manager=manager)
        self.assertIn("too small", widget.zone_list.item(0).text())
        self.assertEqual(widget.zone_status("arm_a").status, STATUS_VIOLATED)

    def test_validity_signal_fires_only_when_the_result_changes(self):
        widget = self.editor()
        seen = []
        widget.validity_changed.connect(lambda ok, issues: seen.append((ok, list(issues))))
        for category, x in (("arm_a", 10), ("arm_b", 100), ("arm_c", 190)):
            widget.set_current_zone(category)
            self.draw(widget, _square(x, 10))
        self.assertFalse(widget.is_valid())
        widget.set_current_zone("center")
        self.draw(widget, _square(280, 10))
        self.assertTrue(widget.is_valid())
        self.assertEqual(seen[-1], (True, []))
        self.assertEqual(sum(1 for ok, _ in seen if ok), 1)
        widget.undo()
        self.assertEqual(seen[-1][0], False)
        self.assertIn("Center", seen[-1][1][0])

    def test_changed_signal_fires_for_edits(self):
        widget = self.editor()
        count = []
        widget.changed.connect(lambda: count.append(1))
        self.draw(widget, SQUARE)
        self.assertGreaterEqual(len(count), 1)


# ----------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class LoadSaveTests(_EditorCase):
    def _manager(self):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_b", np.array(_square(150, 10)))
        manager.add_roi("arm_a", np.array(_square(10, 10)))
        manager.add_roi("arm_a", np.array(_square(10, 120)))
        manager.set_reference_line((10.5, 20.25), (110.5, 20.25), "My line")
        return manager

    def test_roi_manager_is_a_new_object_with_int32_points_and_dimensions(self):
        widget = self.editor(manager=self._manager())
        first, second = widget.roi_manager(), widget.roi_manager()
        self.assertIsNot(first, second)
        self.assertEqual(first.source_dimensions, (400, 300))
        for rois in first.rois.values():
            for roi in rois:
                self.assertEqual(roi.points.dtype, np.int32)

    def test_load_then_save_round_trips(self):
        original = self._manager()
        widget = self.editor(manager=original)
        saved = widget.roi_manager()
        # Definition order (arm_a first), polygons in their original order.
        self.assertEqual(list(saved.rois), ["arm_a", "arm_b"])
        np.testing.assert_array_equal(
            saved.rois["arm_a"][1].points, original.rois["arm_a"][1].points)
        self.assertEqual(saved.reference_line, ((10.5, 20.25), (110.5, 20.25)))
        self.assertEqual(saved.reference_name, "My line")
        self.assertAlmostEqual(saved.reference_length_pixels, 100.0)

    def test_loading_does_not_count_as_unsaved_work(self):
        widget = self.editor(manager=self._manager())
        self.assertFalse(widget.has_unsaved_work())
        self.assertEqual(widget.undo_stack.count(), 0)

    def test_load_roi_manager_replaces_the_content_and_clears_history(self):
        widget = self.editor()
        self.draw(widget, SQUARE)
        widget.load_roi_manager(self._manager())
        self.assertEqual(len(self.shapes(widget)), 3)
        self.assertFalse(widget.undo())
        widget.load_roi_manager(None)
        self.assertEqual(self.shapes(widget), [])

    def test_arena_is_kept_apart_from_zones(self):
        maze = Freestyle()
        maze.configure_zones([("zone_a", "Zone A")])
        maze.configure_arena(True, "distance_cm")
        widget = self.editor(maze)
        self.assertEqual([r.category for r in widget.session.rows()], ["arena", "zone_a"])
        self.assertEqual(widget.current_zone(), "arena")
        self.draw(widget, _square(20, 20, 200))
        self.assertEqual(widget.current_zone(), "zone_a")   # arena takes exactly one
        self.draw(widget, _square(60, 60))
        manager = widget.roi_manager()
        self.assertEqual(list(manager.rois), ["zone_a"])
        self.assertIsNotNone(manager.arena)
        np.testing.assert_array_equal(manager.arena.points, np.array(_square(20, 20, 200)))
        # The arena round-trips through a reload.
        again = self.editor(maze, manager=manager).roi_manager()
        np.testing.assert_array_equal(again.arena.points, manager.arena.points)

    def test_area_fraction_center_preview_is_carried_on_the_manager(self):
        maze = Freestyle()
        maze.configure_zones([])
        maze.configure_arena(True, "area_fraction")
        widget = self.editor(maze, size=(800, 600))
        self.draw(widget, [(100, 100), (500, 100), (500, 500), (100, 500)])
        overlay = widget.roi_manager().center_overlay
        self.assertIsNotNone(overlay)
        np.testing.assert_allclose(overlay.min(axis=0), [200, 200], atol=2.5)
        np.testing.assert_allclose(overlay.max(axis=0), [400, 400], atol=2.5)
        widget.undo()
        self.assertIsNone(widget.roi_manager().center_overlay)

    def test_distance_center_has_no_preview_before_calibration(self):
        maze = Freestyle()
        maze.configure_zones([])
        maze.configure_arena(True, "distance_cm")
        widget = self.editor(maze, size=(800, 600))
        self.draw(widget, [(100, 100), (500, 100), (500, 500), (100, 500)])
        self.assertIsNone(widget.roi_manager().center_overlay)

    def test_a_calibration_without_endpoints_survives_the_round_trip(self):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_a", np.array(SQUARE))
        manager.set_reference_length(123.5, "Legacy")
        widget = self.editor(manager=manager)
        saved = widget.roi_manager()
        self.assertAlmostEqual(saved.reference_length_pixels, 123.5)
        self.assertEqual(saved.reference_name, "Legacy")
        self.assertIsNone(saved.reference_line)
        # Drawing a line replaces it.
        widget.set_line((0, 0), (30, 40))
        self.assertAlmostEqual(widget.roi_manager().reference_length_pixels, 50.0)

    def test_categories_outside_the_maze_definition_are_preserved(self):
        manager = ROIManager((400, 300))
        manager.add_roi("arm_a", np.array(SQUARE))
        manager.add_roi("legacy_zone", np.array(_square(250, 10)))
        widget = self.editor(manager=manager)
        self.assertIn("legacy_zone", [r.category for r in widget.session.rows()])
        self.assertEqual(list(widget.roi_manager().rois), ["arm_a", "legacy_zone"])

    def test_dimension_mismatch_is_pointed_out(self):
        manager = ROIManager((640, 480))
        manager.add_roi("arm_a", np.array(SQUARE))
        widget = self.editor(manager=manager)
        self.assertIn("640x480", widget.hint_label.text())


# ----------------------------------------------------------------------
# Reference line
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ReferenceLineTests(_EditorCase):
    def test_two_clicks_draw_the_line_and_show_its_length(self):
        widget = self.editor()
        widget.set_tool(TOOL_LINE)
        self.click(widget, 20, 50)
        self.assertIsNone(widget.line_points())
        self.click(widget, 140, 50)
        self.assertEqual(widget.line_points(), ((20.0, 50.0), (140.0, 50.0)))
        self.assertAlmostEqual(widget.line_length(), 120.0)
        self.assertIn("120.0 px", widget.zone_list.item(4).text())
        manager = widget.roi_manager()
        self.assertEqual(manager.reference_line, ((20.0, 50.0), (140.0, 50.0)))
        self.assertAlmostEqual(manager.reference_length_pixels, 120.0)
        self.assertEqual(manager.reference_name, "User-drawn reference line")

    def test_dragging_draws_the_line_in_one_gesture(self):
        widget = self.editor()
        widget.set_tool(TOOL_LINE)
        self.drag(widget, (30, 30), (130, 80))
        points = widget.line_points()
        self.assertEqual((points[0], points[1]), ((30.0, 30.0), (130.0, 80.0)))
        self.assertEqual(widget.undo_stack.count(), 1)

    def test_an_endpoint_can_be_dragged_and_undone(self):
        widget = self.editor()
        widget.set_line((20, 50), (140, 50))
        widget.set_tool(TOOL_LINE)
        self.drag(widget, (140, 50), (140, 90))
        self.assertEqual(widget.line_points(), ((20.0, 50.0), (140.0, 90.0)))
        self.assertEqual(widget.undo_stack.count(), 2)
        widget.undo()
        self.assertEqual(widget.line_points(), ((20.0, 50.0), (140.0, 50.0)))

    def test_right_click_or_delete_removes_the_line(self):
        widget = self.editor()
        widget.set_line((20, 50), (140, 50))
        widget.set_tool(TOOL_LINE)
        self.click(widget, 80, 50, Qt.MouseButton.RightButton)
        self.assertIsNone(widget.line_points())
        widget.undo()
        QTest.keyClick(widget.canvas, Qt.Key.Key_Delete)
        self.assertIsNone(widget.line_points())

    def test_degenerate_line_is_refused(self):
        widget = self.editor()
        self.assertFalse(widget.set_line((5, 5), (5, 5)))
        self.assertIsNone(widget.line_points())

    def test_escape_cancels_a_half_drawn_line(self):
        widget = self.editor()
        widget.set_tool(TOOL_LINE)
        self.click(widget, 20, 50)
        QTest.keyClick(widget.canvas, Qt.Key.Key_Escape)
        self.click(widget, 140, 50)       # starts a new line instead of finishing
        self.assertIsNone(widget.line_points())

    def test_line_only_widget_has_no_zone_panel(self):
        widget = RoiEditorWidget(None, np.zeros((100, 100, 3), np.uint8), line_only=True)
        self.addCleanup(widget.deleteLater)
        self.assertEqual(widget.tool(), TOOL_LINE)
        self.assertEqual(list(widget.tool_buttons), [TOOL_LINE])
        self.assertTrue(widget.is_valid())


# ----------------------------------------------------------------------
# Zoom, pan and the coordinate guarantee
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ZoomTests(_EditorCase):
    def test_scene_units_are_source_pixels(self):
        widget = self.editor(size=(640, 480))
        rect = widget.canvas.scene().sceneRect()
        self.assertEqual((rect.width(), rect.height()), (640.0, 480.0))
        self.assertEqual(widget.source_size, (640, 480))
        self.assertEqual(to_model(to_scene((12, 34))), (12.0, 34.0))

    def test_fit_actual_size_and_wheel_zoom(self):
        widget = self.editor(size=(2000, 1500))
        widget.zoom_fit()
        fitted = widget.zoom_percent()
        self.assertLess(fitted, 100)
        widget.zoom_actual()
        self.assertAlmostEqual(widget.zoom_percent(), 100.0)
        self.assertEqual(widget.zoom_label.text(), "Zoom 100%")

        viewport = widget.canvas.viewport()
        center = QPointF(viewport.rect().center())
        event = QWheelEvent(
            center, viewport.mapToGlobal(center), QPoint(0, 0), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase, False)
        self.app.sendEvent(viewport, event)
        self.assertGreater(widget.zoom_percent(), 100.0)
        widget.zoom_fit()
        self.assertAlmostEqual(widget.zoom_percent(), fitted, places=3)

    def test_plain_wheel_does_not_zoom(self):
        widget = self.editor(size=(2000, 1500))
        before = widget.zoom_percent()
        viewport = widget.canvas.viewport()
        center = QPointF(viewport.rect().center())
        event = QWheelEvent(
            center, viewport.mapToGlobal(center), QPoint(0, 0), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False)
        self.app.sendEvent(viewport, event)
        self.assertEqual(widget.zoom_percent(), before)

    def test_clicks_land_on_the_right_source_pixel_at_any_zoom(self):
        # The guarantee the old display-to-source mapping test protected:
        # whatever the zoom or scroll position, the vertex that is stored is
        # the pixel under the cursor (to within one screen pixel).
        for zoom in (0.35, 0.5, 1.0, 2.0, 3.7, 8.0):
            widget = self.editor(size=(1920, 1080), zoom=zoom)
            canvas = widget.canvas
            canvas.horizontalScrollBar().setValue(canvas.horizontalScrollBar().maximum() // 3)
            canvas.verticalScrollBar().setValue(canvas.verticalScrollBar().maximum() // 2)
            self.app.processEvents()
            viewport_rect = canvas.viewport().rect()
            for fraction in ((0.25, 0.3), (0.6, 0.55), (0.8, 0.8)):
                view_point = QPoint(int(viewport_rect.width() * fraction[0]),
                                    int(viewport_rect.height() * fraction[1]))
                expected = canvas.model_point(view_point)
                inside = (0 <= expected[0] < 1920 and 0 <= expected[1] < 1080)
                if not inside:
                    continue
                widget.session.cancel_draft()
                QTest.mouseClick(canvas.viewport(), LEFT,
                                 Qt.KeyboardModifier.NoModifier, view_point)
                stored = widget.pending_points()[-1]
                self.assertLessEqual(abs(stored[0] - expected[0]), 0.5 + 1e-6,
                                     f"x at zoom {zoom}")
                self.assertLessEqual(abs(stored[1] - expected[1]), 0.5 + 1e-6,
                                     f"y at zoom {zoom}")
                # And it is drawn back under the cursor, within a screen pixel.
                back = canvas.view_pos_for_model(stored)
                tolerance = zoom * 0.5 + 1.5
                self.assertLessEqual(abs(back.x() - view_point.x()), tolerance)
                self.assertLessEqual(abs(back.y() - view_point.y()), tolerance)

    def test_middle_drag_pans_the_view(self):
        widget = self.editor(size=(2000, 1500), zoom=1.0)
        canvas = widget.canvas
        bar = canvas.horizontalScrollBar()
        bar.setValue(200)
        viewport = canvas.viewport()
        QTest.mousePress(viewport, Qt.MouseButton.MiddleButton,
                         Qt.KeyboardModifier.NoModifier, QPoint(300, 200))
        QTest.mouseMove(viewport, QPoint(250, 200))
        QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton,
                           Qt.KeyboardModifier.NoModifier, QPoint(250, 200))
        self.assertEqual(bar.value(), 250)
        self.assertEqual(self.shapes(widget), [])
        self.assertEqual(widget.pending_points(), [])

    def test_space_drag_pans_instead_of_drawing(self):
        widget = self.editor(size=(2000, 1500), zoom=1.0)
        canvas = widget.canvas
        bar = canvas.horizontalScrollBar()
        bar.setValue(200)
        QTest.keyPress(canvas, Qt.Key.Key_Space)
        viewport = canvas.viewport()
        QTest.mousePress(viewport, LEFT, Qt.KeyboardModifier.NoModifier, QPoint(300, 200))
        QTest.mouseMove(viewport, QPoint(260, 200))
        QTest.mouseRelease(viewport, LEFT, Qt.KeyboardModifier.NoModifier, QPoint(260, 200))
        QTest.keyRelease(canvas, Qt.Key.Key_Space)
        self.assertEqual(bar.value(), 240)
        self.assertEqual(widget.pending_points(), [])

    def test_keyboard_shortcuts_for_tools_and_zoom(self):
        widget = self.editor(size=(2000, 1500))
        QTest.keyClick(widget.canvas, Qt.Key.Key_E)
        self.assertEqual(widget.tool(), TOOL_EDIT)
        QTest.keyClick(widget.canvas, Qt.Key.Key_L)
        self.assertEqual(widget.tool(), TOOL_LINE)
        QTest.keyClick(widget.canvas, Qt.Key.Key_D)
        self.assertEqual(widget.tool(), TOOL_DRAW)
        QTest.keyClick(widget.canvas, Qt.Key.Key_1)
        self.assertAlmostEqual(widget.zoom_percent(), 100.0)
        QTest.keyClick(widget.canvas, Qt.Key.Key_F)
        self.assertLess(widget.zoom_percent(), 100.0)


if __name__ == "__main__":
    unittest.main()
