"""The workspace video panel: table, filters, editing, context menu, drops."""
import unittest

from tests.project_support import make_settings
from tests.workspace_support import HAS_RUNTIME, WorkspaceFixture

if HAS_RUNTIME:
    from PyQt6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
    from PyQt6.QtGui import QDropEvent
    from PyQt6.QtTest import QTest
    from gui.project_table_model import KeyRole
    from gui.workspace.theme import current_palette


class PanelFixture(WorkspaceFixture):
    def setUp(self):
        super().setUp()
        self.new_project_at()
        self.panel = self.window.video_panel
        self.project = self.window.project
        self.selections = []
        self.panel.selection_changed.connect(self.selections.append)

    def names(self):
        return [v.name for v in self.project.videos]

    def shown(self):
        proxy = self.panel.proxy
        return [proxy.index(r, 1).data() for r in range(proxy.rowCount())]

    def select(self, *names):
        keys = [v.key for v in self.project.videos if v.name in names]
        self.panel.select_keys(keys)
        self.panel.flush_selection()
        return keys

    def action(self, menu, text):
        for action in menu.actions():
            if action.text() == text:
                return action
        raise AssertionError(f"No {text!r} in {[a.text() for a in menu.actions()]}")


class AddingTests(PanelFixture):
    def test_add_videos_dialog(self):
        self.prompts.say("open_files", self.videos[:2])
        result = self.panel.add_videos_dialog()
        self.assertEqual([v.name for v in result.added], ["m1.mp4", "m2.mp4"])
        self.assertEqual(self.names(), ["m1.mp4", "m2.mp4"])
        self.assertEqual(self.panel.model.rowCount(), 2)
        self.assertEqual(self.settings.last_video_dir(), self.videos[0].parent)

    def test_cancelling_the_dialog_adds_nothing(self):
        self.prompts.say("open_files", [])
        self.assertIsNone(self.panel.add_videos_dialog())
        self.assertEqual(self.names(), [])

    def test_add_folder_dialog_takes_the_videos_of_the_folder(self):
        nested = self.write_video("videos/deep/m9.mp4")
        self.prompts.say("choose_folder", self.root / "videos")
        self.panel.add_folder_dialog()
        self.assertEqual(self.names(), ["m1.mp4", "m2.mp4", "m3.mp4"])
        self.assertNotIn(nested, [v.path for v in self.project.videos])

    def test_include_subfolders_is_remembered_and_used(self):
        self.write_video("videos/deep/m9.mp4")
        self.panel.recursive_check.setChecked(True)
        self.assertTrue(self.settings.add_folder_recursive())
        self.prompts.say("choose_folder", self.root / "videos")
        self.panel.add_folder_dialog()
        self.assertIn("m9.mp4", self.names())
        self.assertTrue(self.panel.model.recursive_drop)

    def test_an_empty_folder_is_explained(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.prompts.say("choose_folder", empty)
        self.assertIsNone(self.panel.add_folder_dialog())
        self.assertEqual(len(self.prompts.called("info")), 1)

    def test_duplicates_are_reported_not_added(self):
        messages = []
        self.panel.status_message.connect(messages.append)
        self.panel.add_paths(self.videos[:2])
        self.panel.add_paths(self.videos)
        self.assertEqual(self.names(), ["m1.mp4", "m2.mp4", "m3.mp4"])
        self.assertIn("2 already in the project", messages[-1])

    def test_added_videos_get_selected(self):
        self.panel.add_paths(self.videos)
        self.panel.flush_selection()
        self.assertEqual(self.panel.selected_keys(), self.project.keys())
        self.assertEqual(self.selections[-1], self.project.keys())

    def test_toolbar_actions_are_enabled_with_a_project(self):
        self.assertTrue(self.window.add_videos_action.isEnabled())
        self.prompts.say("open_files", self.videos[:1])
        self.window.add_videos_action.trigger()
        self.assertEqual(self.names(), ["m1.mp4"])
        self.prompts.say("choose_folder", self.root / "videos")
        self.window.add_folder_action.trigger()
        self.assertEqual(len(self.names()), 3)


class RemovingAndMovingTests(PanelFixture):
    def setUp(self):
        super().setUp()
        self.panel.add_paths(self.videos)
        self.panel.clear_selection()
        self.panel.flush_selection()

    def test_delete_key_removes_after_confirmation(self):
        self.select("m2.mp4")
        self.prompts.say("confirm", True)
        self.panel.view.setFocus()
        QTest.keyClick(self.panel.view, Qt.Key.Key_Delete)
        self.assertEqual(self.names(), ["m1.mp4", "m3.mp4"])
        (title, text), = self.prompts.called("confirm")
        self.assertIn("m2.mp4", text)
        self.assertEqual(self.panel.model.rowCount(), 2)

    def test_declining_keeps_the_videos(self):
        self.select("m1.mp4", "m2.mp4")
        self.prompts.say("confirm", False)
        QTest.keyClick(self.panel.view, Qt.Key.Key_Delete)
        self.assertEqual(len(self.names()), 3)
        self.assertIn("2 videos", self.prompts.called("confirm")[0][1])

    def test_delete_with_nothing_selected_does_nothing(self):
        QTest.keyClick(self.panel.view, Qt.Key.Key_Delete)
        self.assertEqual(self.prompts.called("confirm"), [])

    def test_toolbar_remove_follows_the_selection(self):
        self.assertFalse(self.window.remove_action.isEnabled())
        self.select("m1.mp4")
        self.assertTrue(self.window.remove_action.isEnabled())
        self.prompts.say("confirm", True)
        self.window.remove_action.trigger()
        self.assertEqual(self.names(), ["m2.mp4", "m3.mp4"])

    def test_after_removing_a_neighbour_is_selected(self):
        self.select("m2.mp4")
        self.prompts.say("confirm", True)
        self.panel.remove_selected()
        self.panel.flush_selection()
        self.assertEqual(len(self.panel.selected_keys()), 1)

    def test_alt_down_and_up_move_the_selection(self):
        self.select("m1.mp4")
        QTest.keyClick(self.panel.view, Qt.Key.Key_Down,
                       Qt.KeyboardModifier.AltModifier)
        self.assertEqual(self.names(), ["m2.mp4", "m1.mp4", "m3.mp4"])
        self.assertEqual(self.panel.selected_keys(),
                         [self.project.videos[1].key])
        QTest.keyClick(self.panel.view, Qt.Key.Key_Up,
                       Qt.KeyboardModifier.AltModifier)
        self.assertEqual(self.names(), ["m1.mp4", "m2.mp4", "m3.mp4"])

    def test_a_block_moves_together_and_stops_at_the_edge(self):
        self.select("m2.mp4", "m3.mp4")
        self.panel.move_selected(1)
        self.assertEqual(self.names(), ["m1.mp4", "m2.mp4", "m3.mp4"])
        self.panel.move_selected(-1)
        self.assertEqual(self.names(), ["m2.mp4", "m3.mp4", "m1.mp4"])
        self.assertEqual(len(self.panel.selected_keys()), 2)

    def test_the_table_follows_the_project_order(self):
        self.select("m3.mp4")
        self.panel.move_selected(-2)
        self.assertEqual(self.shown(), ["m3.mp4", "m1.mp4", "m2.mp4"])


class SelectionTests(PanelFixture):
    def setUp(self):
        super().setUp()
        self.panel.add_paths(self.videos)

    def test_selection_changed_reports_keys_in_table_order(self):
        self.select("m3.mp4", "m1.mp4")
        self.assertEqual(
            [self.project.entry(k).name for k in self.selections[-1]],
            ["m1.mp4", "m3.mp4"],
        )

    def test_clearing_the_selection_reports_an_empty_list(self):
        self.select("m1.mp4")
        self.panel.clear_selection()
        self.panel.flush_selection()
        self.assertEqual(self.selections[-1], [])

    def test_the_window_forwards_the_selection(self):
        seen = []
        self.window.selection_changed.connect(seen.append)
        keys = self.select("m2.mp4")
        self.assertEqual(seen[-1], keys)
        self.assertEqual(self.window.selected_keys(), keys)
        self.assertEqual(
            [e.name for e in self.window.selected_entries()], ["m2.mp4"]
        )

    def test_same_selection_is_not_reported_twice(self):
        self.select("m2.mp4")
        count = len(self.selections)
        self.panel.flush_selection()
        self.panel.flush_selection()
        self.assertEqual(len(self.selections), count)


class FilterTests(PanelFixture):
    def setUp(self):
        super().setUp()
        self.panel.add_paths(self.videos)
        self.project.add_metadata_column("group")
        for entry, group in zip(self.project.videos, ("ctl", "trt", "ctl")):
            self.project.set_metadata_value(entry.key, "group", group)

    def test_search_filters_by_name_and_metadata(self):
        self.panel.search_box.setText("m2")
        self.assertEqual(self.shown(), ["m2.mp4"])
        self.panel.search_box.setText("ctl")
        self.assertEqual(self.shown(), ["m1.mp4", "m3.mp4"])
        self.panel.search_box.setText("ctl m3")
        self.assertEqual(self.shown(), ["m3.mp4"])
        self.panel.search_box.clear()
        self.assertEqual(len(self.shown()), 3)

    def test_status_filter(self):
        self.project.videos[0].last_status = "done"
        self.panel.model.refresh()
        index = self.panel.status_combo.findData("done")
        self.panel.status_combo.setCurrentIndex(index)
        self.assertEqual(self.shown(), ["m1.mp4"])
        self.panel.status_combo.setCurrentIndex(0)
        self.assertEqual(len(self.shown()), 3)

    def test_footer_shows_how_many_are_shown(self):
        self.panel.search_box.setText("m2")
        self.assertIn("3 videos (1 shown)", self.panel.footer.text())

    def test_filtered_out_selection_is_reported(self):
        self.select("m1.mp4")
        self.panel.search_box.setText("m2")
        self.panel.flush_selection()
        self.assertEqual(self.selections[-1], [])

    def test_changing_project_resets_the_filters(self):
        self.panel.search_box.setText("zzz")
        self.window.save()
        self.window.close_project()
        self.assertEqual(self.panel.search_box.text(), "")
        self.assertFalse(self.panel.search_box.isEnabled())


class FooterTests(PanelFixture):
    def test_no_project(self):
        self.window.close_project()
        self.assertEqual(self.panel.footer.text(), "No project open")

    def test_counts(self):
        self.assertIn("0 videos", self.panel.footer.text())
        self.project.set_template(make_settings())
        self.panel.add_paths(self.videos)
        self.assertIn("3 videos", self.panel.footer.text())
        self.assertIn("3 configured", self.panel.footer.text())
        self.assertIn("0 verified", self.panel.footer.text())
        self.project.set_verified(self.project.keys()[:2], True)
        self.assertIn("2 verified", self.panel.footer.text())

    def test_issues_count_and_color(self):
        self.panel.add_paths(self.videos)       # unconfigured: 3 warnings
        self.assertIn("3 issues", self.panel.footer.text())
        self.assertNotIn("color", self.panel.footer.text())
        self.videos[0].unlink()
        self.panel.model.refresh()
        self.assertIn("4 issues", self.panel.footer.text())
        self.assertIn("color", self.panel.footer.text())

    def test_the_drop_zone_stands_in_for_an_empty_table(self):
        panel = self.panel
        self.assertTrue(panel.drop_zone.isVisibleTo(panel))
        self.assertFalse(panel.view.isVisibleTo(panel))
        self.assertFalse(panel.add_bar.isVisibleTo(panel))
        self.assertTrue(panel.empty_state_visible)
        panel.add_paths(self.videos[:1])
        self.assertFalse(panel.drop_zone.isVisibleTo(panel))
        self.assertTrue(panel.view.isVisibleTo(panel))
        self.assertTrue(panel.add_bar.isVisibleTo(panel))
        self.assertFalse(panel.empty_state_visible)
        panel.remove_keys(self.project.keys(), confirm=False)
        self.assertTrue(panel.drop_zone.isVisibleTo(panel))


class MetadataTests(PanelFixture):
    def setUp(self):
        super().setUp()
        self.panel.add_paths(self.videos)

    def test_add_column_button(self):
        self.prompts.say("ask_text", "dose")
        self.assertEqual(self.panel.add_column(), "dose")
        self.assertEqual(self.project.metadata_columns(), ["dose"])
        self.assertIn("dose", self.panel.model.headers())

    def test_add_column_cancel_blank_and_duplicate(self):
        self.prompts.say("ask_text", None, "  ", "dose", "dose")
        self.assertIsNone(self.panel.add_column())
        self.assertIsNone(self.panel.add_column())
        self.assertEqual(self.panel.add_column(), "dose")
        self.assertIsNone(self.panel.add_column())
        self.assertEqual(len(self.prompts.called("info")), 1)

    def test_metadata_is_editable_in_place(self):
        self.project.add_metadata_column("group")
        self.panel.model.refresh()
        column = self.panel.model.column_for_metadata("group")
        index = self.panel.proxy.index(1, column)
        self.assertTrue(index.flags() & Qt.ItemFlag.ItemIsEditable)
        self.assertTrue(self.panel.proxy.setData(index, "treated"))
        self.assertEqual(self.project.videos[1].metadata["group"], "treated")
        self.assertTrue(self.window.has_unsaved_changes)

    def test_video_column_is_not_editable(self):
        index = self.panel.proxy.index(0, 1)
        self.assertFalse(index.flags() & Qt.ItemFlag.ItemIsEditable)

    def test_edit_metadata_column_sets_a_value_for_the_selection(self):
        self.project.add_metadata_column("group")
        keys = [self.project.videos[0].key, self.project.videos[2].key]
        self.prompts.say("ask_item", "group")
        self.prompts.say("ask_text", "ctl")
        self.assertEqual(self.panel.edit_metadata_column(keys), 2)
        self.assertEqual(
            [v.metadata["group"] for v in self.project.videos],
            ["ctl", "", "ctl"],
        )

    def test_edit_metadata_column_for_a_given_column_skips_the_picker(self):
        self.project.add_metadata_column("group")
        self.prompts.say("ask_text", "x")
        self.panel.edit_metadata_column(self.project.keys(), "group")
        self.assertEqual(self.prompts.called("ask_item"), [])

    def test_edit_metadata_without_columns_offers_to_create_one(self):
        self.prompts.say("ask_text", "group", "ctl")
        self.assertEqual(self.panel.edit_metadata_column(self.project.keys()), 3)
        self.assertEqual(self.project.metadata_columns(), ["group"])

    def test_remove_column(self):
        self.project.add_metadata_column("group")
        self.prompts.say("confirm", False, True)
        self.assertFalse(self.panel.remove_column("group"))
        self.assertTrue(self.panel.remove_column("group"))
        self.assertEqual(self.project.metadata_columns(), [])

    def test_load_sample_sheet_fills_metadata_and_reports(self):
        sheet = self.root / "sheet.csv"
        sheet.write_text(
            "video,animal,group\nm1.mp4,A1,ctl\nm2.mp4,A2,trt\n",
            encoding="utf-8",
        )
        self.prompts.say("open_file", sheet)
        match = self.panel.load_sample_sheet()
        self.assertIsNotNone(match)
        self.assertEqual(self.project.videos[0].metadata["group"], "ctl")
        self.assertEqual(self.project.run_options.sample_sheet, sheet.resolve())
        self.assertEqual(len(self.prompts.called("info")), 1)
        self.assertIn("group", self.panel.model.headers())

    def test_a_bad_sample_sheet_is_reported(self):
        bad = self.root / "bad.csv"
        bad.write_text("name,group\nm1.mp4,ctl\n", encoding="utf-8")
        self.prompts.say("open_file", bad)
        self.assertIsNone(self.panel.load_sample_sheet())
        self.assertEqual(len(self.prompts.called("error")), 1)
        self.assertEqual(self.project.metadata_columns(), [])


class ContextMenuTests(PanelFixture):
    def setUp(self):
        super().setUp()
        self.project.set_template(make_settings())
        self.panel.add_paths(self.videos)
        self.panel.clear_selection()
        self.panel.flush_selection()

    def menu(self, *names, column=0):
        self.select(*names)
        index = self.panel.proxy.index(0, column)
        return self.panel.build_context_menu(index)

    def test_entries(self):
        menu = self.menu("m1.mp4")
        texts = [a.text() for a in menu.actions() if a.text()]
        self.assertEqual(texts, [
            "Details", "Remove", "Reveal in folder", "Copy path", "Mark verified",
            "Mark unverified", "Apply template settings",
            "Edit metadata column...", "Load sample sheet...",
        ])

    def test_entries_depend_on_the_selection(self):
        menu = self.menu()
        enabled = {a.text(): a.isEnabled() for a in menu.actions() if a.text()}
        self.assertFalse(enabled["Remove"])
        self.assertFalse(enabled["Copy path"])
        self.assertTrue(enabled["Load sample sheet..."])
        menu = self.menu("m1.mp4", "m2.mp4")
        enabled = {a.text(): a.isEnabled() for a in menu.actions() if a.text()}
        self.assertFalse(enabled["Reveal in folder"])
        self.assertTrue(enabled["Copy path"])

    def test_apply_template_needs_a_template(self):
        self.project.template_settings = None
        menu = self.menu("m1.mp4")
        self.assertFalse(self.action(menu, "Apply template settings").isEnabled())

    def test_mark_verified_and_unverified(self):
        menu = self.menu("m1.mp4", "m2.mp4")
        self.action(menu, "Mark verified").trigger()
        self.assertEqual([v.verified for v in self.project.videos],
                         [True, True, False])
        self.action(menu, "Mark unverified").trigger()
        self.assertEqual([v.verified for v in self.project.videos],
                         [False, False, False])

    def test_unconfigured_videos_cannot_be_verified(self):
        self.project.videos[0].settings = None
        self.assertEqual(self.panel.mark_verified([self.project.keys()[0]]), 0)
        self.assertEqual(len(self.prompts.called("info")), 1)
        self.assertFalse(self.project.videos[0].verified)

    def test_reveal_and_copy_path(self):
        menu = self.menu("m2.mp4")
        self.action(menu, "Reveal in folder").trigger()
        self.assertEqual(self.prompts.called("reveal"), [(self.videos[1],)])
        menu = self.menu("m1.mp4", "m3.mp4")
        self.action(menu, "Copy path").trigger()
        self.assertEqual(
            self.prompts.clipboard[-1], f"{self.videos[0]}\n{self.videos[2]}"
        )

    def test_remove_from_the_menu_asks(self):
        menu = self.menu("m1.mp4")
        self.prompts.say("confirm", True)
        self.action(menu, "Remove").trigger()
        self.assertEqual(self.names(), ["m2.mp4", "m3.mp4"])

    def test_apply_template_overwrites_custom_settings_after_confirming(self):
        entry = self.project.videos[0]
        self.project.customize(entry.key, make_settings(offset=7))
        menu = self.menu("m1.mp4", "m2.mp4")
        self.prompts.say("confirm", False, True)
        self.action(menu, "Apply template settings").trigger()
        self.assertEqual(entry.settings_source, "custom")
        self.action(menu, "Apply template settings").trigger()
        self.assertEqual(entry.settings_source, "template")
        self.assertEqual(len(self.prompts.called("confirm")), 2)

    def test_apply_template_without_custom_videos_does_not_ask(self):
        menu = self.menu("m1.mp4")
        self.action(menu, "Apply template settings").trigger()
        self.assertEqual(self.prompts.called("confirm"), [])

    def test_metadata_column_entries_appear_on_a_metadata_cell(self):
        self.project.add_metadata_column("group")
        self.panel.model.refresh()
        column = self.panel.model.column_for_metadata("group")
        menu = self.menu("m1.mp4", column=column)
        texts = [a.text() for a in menu.actions()]
        self.assertIn("Remove column “group”...", texts)
        self.prompts.say("ask_text", "ctl")
        self.action(menu, "Edit metadata column...").trigger()
        self.assertEqual(self.project.videos[0].metadata["group"], "ctl")
        self.assertEqual(self.prompts.called("ask_item"), [])

    def test_load_sample_sheet_entry(self):
        sheet = self.root / "s.csv"
        sheet.write_text("video,group\nm1.mp4,a\n", encoding="utf-8")
        menu = self.menu()
        self.prompts.say("open_file", sheet)
        self.action(menu, "Load sample sheet...").trigger()
        self.assertEqual(self.project.videos[0].metadata["group"], "a")


class MissingFileTests(PanelFixture):
    def setUp(self):
        super().setUp()
        self.moved = self.write_video("old/day1/a.mp4", b"a")
        self.moved2 = self.write_video("old/day1/b.mp4", b"bb")
        self.panel.add_paths([self.moved, self.moved2])

    def move_data(self):
        new = self.root / "new" / "day1"
        new.mkdir(parents=True)
        for path in (self.moved, self.moved2):
            path.rename(new / path.name)
        self.panel.model.refresh()
        return new

    def test_missing_videos_are_drawn_in_red(self):
        proxy = self.panel.proxy
        role = Qt.ItemDataRole.ForegroundRole
        self.assertIsNone(proxy.index(0, 1).data(role))
        self.moved.unlink()
        self.panel.model.refresh()
        brush = proxy.index(0, 1).data(role)
        self.assertEqual(brush.color().name(), current_palette().danger.lower())
        self.assertIsNone(proxy.index(1, 1).data(role))

    def test_menu_offers_relink_only_for_missing_files(self):
        keys = self.select("a.mp4")
        texts = [a.text() for a in self.panel.build_context_menu().actions()]
        self.assertNotIn("Relink...", texts)
        self.moved.unlink()
        self.panel.model.refresh()
        texts = [a.text() for a in self.panel.build_context_menu().actions()]
        self.assertIn("Relink...", texts)
        self.assertTrue(self.panel.has_missing(keys))

    def test_relink_repoints_the_whole_data_folder(self):
        new = self.move_data()
        self.assertEqual(len(self.project.missing_files()), 2)
        key = self.project.videos[0].key
        self.prompts.say("choose_folder", new)
        relinked = self.panel.relink(key)
        self.assertEqual(set(relinked), set(self.project.keys()))
        self.assertEqual(self.project.videos[0].path, (new / "a.mp4").resolve())
        self.assertEqual(self.project.videos[0].key, key)
        self.assertEqual(self.project.missing_files(), [])
        role = Qt.ItemDataRole.ForegroundRole
        self.assertIsNone(self.panel.proxy.index(0, 1).data(role))

    def test_relink_into_a_folder_without_the_video_is_refused(self):
        self.move_data()
        wrong = self.root / "elsewhere"
        wrong.mkdir()
        self.prompts.say("choose_folder", wrong)
        self.assertEqual(self.panel.relink(self.project.keys()[0]), [])
        self.assertEqual(len(self.prompts.called("warn")), 1)
        self.assertEqual(len(self.project.missing_files()), 2)

    def test_relink_cancelled(self):
        self.move_data()
        self.prompts.say("choose_folder", None)
        self.assertEqual(self.panel.relink(self.project.keys()[0]), [])


class DropTests(PanelFixture):
    def drop(self, widget, paths):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
        event = QDropEvent(
            QPointF(QPoint(5, 5)), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        widget.dropEvent(event)
        return event

    def test_the_panel_reports_dropped_paths(self):
        got = []
        self.panel.paths_dropped.connect(got.append)
        event = self.drop(self.panel, [self.videos[0], self.root / "videos"])
        self.assertTrue(event.isAccepted())
        self.assertEqual(got, [[self.videos[0], self.root / "videos"]])

    def test_the_window_adds_what_the_panel_reports(self):
        self.drop(self.panel, [self.root / "videos"])
        self.assertEqual(len(self.names()), 3)

    def test_the_panel_accepts_drags_only_when_editable(self):
        from PyQt6.QtGui import QDragEnterEvent
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(self.videos[0]))])

        def enter():
            event = QDragEnterEvent(
                QPoint(5, 5), Qt.DropAction.CopyAction, mime,
                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
            )
            self.panel.dragEnterEvent(event)
            return event.isAccepted()

        self.assertTrue(enter())
        self.panel.set_read_only(True)
        self.assertFalse(enter())
        self.panel.set_read_only(False)
        self.window.close_project()
        self.assertFalse(enter())

    def test_the_table_model_drop_path_also_reports_videos(self):
        got = []
        self.panel.model.pathsDropped.connect(got.append)
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(self.root / "videos"))])
        self.assertTrue(self.panel.model.dropMimeData(
            mime, Qt.DropAction.CopyAction, -1, -1, self.panel.model.index(-1, -1)
        ))
        self.assertEqual(len(got[0]), 3)


class ReadOnlyTests(PanelFixture):
    def setUp(self):
        super().setUp()
        self.panel.add_paths(self.videos)
        self.panel.set_read_only(True)

    def test_every_edit_is_refused(self):
        keys = self.project.keys()
        self.select("m1.mp4")
        self.prompts.calls.clear()
        self.assertIsNone(self.panel.add_paths(self.videos))
        self.assertEqual(self.panel.remove_keys(keys), [])
        self.assertFalse(self.panel.move_selected(1))
        self.assertEqual(self.panel.mark_verified(keys), 0)
        self.assertEqual(self.panel.apply_template(keys), [])
        self.assertIsNone(self.panel.add_column())
        self.assertEqual(self.panel.edit_metadata_column(keys, "g"), 0)
        self.assertIsNone(self.panel.load_sample_sheet(self.root / "x.csv"))
        self.assertEqual(self.prompts.calls, [])
        self.assertEqual(len(self.names()), 3)

    def test_cells_are_not_editable_while_read_only(self):
        self.assertEqual(
            self.panel.view.editTriggers(),
            self.panel.view.EditTrigger.NoEditTriggers,
        )
        self.panel.set_read_only(False)
        self.assertNotEqual(
            self.panel.view.editTriggers(),
            self.panel.view.EditTrigger.NoEditTriggers,
        )

    def test_the_window_locks_and_unlocks_the_panel(self):
        self.panel.set_read_only(False)
        self.window.set_busy(True)
        self.assertTrue(self.panel.is_read_only())
        self.assertFalse(self.window.add_videos_action.isEnabled())
        self.window.set_busy(False)
        self.assertFalse(self.panel.is_read_only())
        self.assertTrue(self.window.add_videos_action.isEnabled())


class KeyRoleTests(PanelFixture):
    def test_rows_carry_the_project_keys(self):
        self.panel.add_paths(self.videos)
        self.assertEqual(
            [self.panel.proxy.index(r, 0).data(KeyRole) for r in range(3)],
            self.project.keys(),
        )


if __name__ == "__main__":
    unittest.main()
