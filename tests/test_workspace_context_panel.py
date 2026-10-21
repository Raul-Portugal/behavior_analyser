"""The workspace context panel and the thumbnail loader."""
import unittest
from unittest.mock import patch

from tests.project_support import make_settings
from tests.synthetic_video import write_indexed_video
from tests.workspace_support import HAS_RUNTIME, WorkspaceFixture

if HAS_RUNTIME:
    from PyQt6.QtTest import QTest
    from gui.workspace import thumbnails
    from gui.workspace.context_panel import (ACTION_APPLY_TEMPLATE,
                                             ACTION_REMOVE, ACTION_RELINK,
                                             ACTION_UNVERIFY, ACTION_VERIFY,
                                             ContextPanel)


class ContextFixture(WorkspaceFixture):
    def setUp(self):
        super().setUp()
        self.real = write_indexed_video(self.root / "real.avi")
        self.new_project_at()
        self.project = self.window.project
        self.panel = self.window.context_panel
        self.table = self.window.video_panel
        self.window.drawer.open(animated=False)   # a closed drawer does not redraw

    def add(self, *paths):
        self.table.add_paths(list(paths) or self.videos)
        self.table.clear_selection()
        self.table.flush_selection()

    def select(self, *names):
        keys = [v.key for v in self.project.videos if v.name in names]
        self.table.select_keys(keys)
        self.table.flush_selection()
        return keys

    @staticmethod
    def plain(text):
        return text.replace("​", "")

    def settle(self):
        self.panel.loader.wait_idle()
        self.app.processEvents()


class EmptyTests(ContextFixture):
    def test_hint_without_a_selection(self):
        self.assertEqual(self.panel.pages.currentIndex(), ContextPanel.PAGE_EMPTY)
        self.assertIn("Select a video", self.panel.empty_label.text())

    def test_without_a_project(self):
        self.window.close_project()
        self.assertEqual(self.panel.pages.currentIndex(), ContextPanel.PAGE_EMPTY)
        self.assertIn("No project", self.panel.empty_label.text())

    def test_the_panel_has_no_project_box_any_more(self):
        # Name, file and paradigm moved to the Videos stage.
        for attribute in ("project_box", "paradigm_combo", "paradigm_chosen"):
            self.assertFalse(hasattr(self.panel, attribute), attribute)


class SingleVideoTests(ContextFixture):
    def test_shows_state_source_and_path(self):
        self.project.set_template(make_settings())
        self.add()
        self.select("m2.mp4")
        panel = self.panel
        self.assertEqual(panel.pages.currentIndex(), ContextPanel.PAGE_SINGLE)
        self.assertEqual(self.plain(panel.name_label.text()), "m2.mp4")
        self.assertEqual(self.plain(panel.path_label.text()), str(self.videos[1]))
        self.assertEqual(panel.path_label.toolTip(), str(self.videos[1]))
        self.assertEqual(panel.configured_label.text(), "Configured")
        self.assertEqual(panel.source_label.text(), "Project template")
        self.project.set_verified([self.project.videos[1].key], True)
        self.assertEqual(panel.configured_label.text(), "Configured and verified")

    def test_shows_the_chip_trial_and_zones(self):
        self.project.set_template(make_settings())
        self.add()
        self.select("m2.mp4")
        panel = self.panel
        self.assertEqual(panel.status_chip.text(), "Set up")
        self.assertEqual(panel.trial_label.text(), "whole video")
        self.assertEqual(panel.zones_label.text(), "The project template's")
        self.project.set_verified([self.project.videos[1].key], True)
        self.assertEqual(panel.status_chip.text(), "Verified")
        self.assertEqual(panel.verify_button.text(), "Mark unverified")

    def test_the_actions_announce_themselves(self):
        self.project.set_template(make_settings())
        self.add()
        self.select("m2.mp4")
        got = []
        self.panel.action_requested.connect(got.append)
        self.panel.verify_button.click()
        self.panel.reveal_button.click()
        self.assertEqual(got, ["verify", "reveal"])
        self.assertEqual(self.prompts.called("reveal"), [(self.videos[1],)])
        self.assertEqual(
            [v.verified for v in self.project.videos], [False, True, False]
        )
        self.panel.verify_button.click()
        self.assertEqual(
            [v.verified for v in self.project.videos], [False, False, False]
        )

    def test_updates_when_the_project_changes(self):
        self.add()
        self.select("m1.mp4")
        self.assertEqual(self.panel.configured_label.text(), "Not configured")
        self.assertEqual(self.panel.source_label.text(), "—")
        self.project.customize(self.project.videos[0].key, make_settings(offset=3))
        self.assertEqual(self.panel.configured_label.text(), "Configured")
        self.assertEqual(
            self.panel.source_label.text(), "Customised for this video"
        )

    def test_follows_the_selection(self):
        self.add()
        self.select("m1.mp4")
        self.assertEqual(self.plain(self.panel.name_label.text()), "m1.mp4")
        self.select("m3.mp4")
        self.assertEqual(self.plain(self.panel.name_label.text()), "m3.mp4")
        self.table.clear_selection()
        self.table.flush_selection()
        self.assertEqual(self.panel.pages.currentIndex(), ContextPanel.PAGE_EMPTY)

    def test_lists_the_issues(self):
        self.add()
        self.select("m1.mp4")
        items = [self.panel.issues_list.item(i).text()
                 for i in range(self.panel.issues_list.count())]
        self.assertEqual(items, ["m1.mp4 has no settings"])
        self.assertTrue(self.panel.has_issues())
        self.project.set_template(make_settings())
        self.assertEqual(self.panel.issues_list.item(0).text(), "No issues")
        self.assertFalse(self.panel.has_issues())

    def test_missing_file(self):
        self.add()
        self.videos[0].unlink()
        self.table.model.refresh()
        self.select("m1.mp4")
        self.assertEqual(self.panel.thumbnail_label.text(), "File not found")
        self.assertFalse(self.panel.relink_button.isHidden())
        texts = [self.panel.issues_list.item(i).text()
                 for i in range(self.panel.issues_list.count())]
        self.assertTrue(any("not found" in t for t in texts))
        got = []
        self.panel.action_requested.connect(got.append)
        self.prompts.say("choose_folder", None)
        self.panel.relink_button.click()
        self.assertEqual(got, [ACTION_RELINK])
        self.assertEqual(len(self.prompts.called("choose_folder")), 1)

    def test_existing_file_hides_the_relink_button(self):
        self.add()
        self.select("m2.mp4")
        self.assertTrue(self.panel.relink_button.isHidden())

    def test_last_run_status_and_open_run_folder(self):
        self.add()
        run_dir = self.root / "analysis_run_1"
        run_dir.mkdir()
        entry = self.project.videos[0]
        self.select("m1.mp4")
        self.assertEqual(self.panel.last_run_label.text(), "Not analysed yet")
        self.assertFalse(self.panel.open_run_button.isEnabled())
        entry.last_status = "done"
        entry.last_run_dir = run_dir
        self.project.set_video_notes(entry.key, "x")      # any change
        self.assertIn("Done", self.panel.last_run_label.text())
        self.assertIn(str(run_dir), self.plain(self.panel.last_run_label.text()))
        self.assertTrue(self.panel.open_run_button.isEnabled())
        self.panel.open_run_button.click()
        self.assertEqual(self.prompts.called("open_folder"), [(run_dir,)])

    def test_open_run_folder_that_vanished_warns(self):
        self.add()
        entry = self.project.videos[0]
        entry.last_status = "failed"
        entry.last_run_dir = self.root / "gone"
        self.select("m1.mp4")
        self.assertFalse(self.panel.open_run_button.isEnabled())
        with patch.object(self.prompts, "open_folder", return_value=False):
            self.assertFalse(self.panel.open_run_folder())
        self.assertEqual(len(self.prompts.called("warn")), 1)


class ThumbnailTests(ContextFixture):
    def test_first_frame_and_facts_arrive_off_thread(self):
        self.add(self.real)
        self.select("real.avi")
        self.assertEqual(self.panel.thumbnail_label.text(), "Loading preview...")
        self.assertEqual(self.panel.resolution_label.text(), "—")
        self.settle()
        self.assertEqual(self.panel.thumbnail_label.text(), "")
        pixmap = self.panel.thumbnail_label.pixmap()
        self.assertFalse(pixmap.isNull())
        self.assertEqual(pixmap.width(), 160)
        self.assertEqual(self.panel.resolution_label.text(), "160 × 120")
        self.assertEqual(self.panel.fps_label.text(), "25.00 fps")
        self.assertEqual(self.panel.duration_label.text(), "3s")

    def test_previews_are_cached(self):
        self.add(self.real, self.videos[0])
        # Read both previews first; re-selecting them must not read again.
        self.select("real.avi")
        self.settle()
        self.select("m1.mp4")
        self.settle()
        with patch.object(thumbnails, "read_preview",
                          side_effect=AssertionError("read again")) as read:
            self.select("m1.mp4")
            self.select("real.avi")
        read.assert_not_called()
        self.assertEqual(self.panel.resolution_label.text(), "160 × 120")

    def test_an_edited_video_is_read_again(self):
        self.add(self.real)
        self.select("real.avi")
        self.settle()
        loader = self.panel.loader
        self.assertIsNotNone(loader.get(self.real))
        with open(self.real, "ab") as handle:
            handle.write(b"\0" * 16)
        self.assertIsNone(loader.get(self.real))
        self.assertTrue(loader.request(self.real))

    def test_an_unreadable_video_shows_the_reason(self):
        self.add()
        self.select("m1.mp4")
        self.settle()
        self.assertIn("cannot be opened", self.panel.thumbnail_label.text())
        self.assertEqual(self.panel.resolution_label.text(), "—")

    def test_a_late_preview_for_another_selection_is_ignored(self):
        self.add(self.real, self.videos[0])
        self.select("real.avi")
        self.select("m1.mp4")
        self.settle()
        self.assertIn("cannot be opened", self.panel.thumbnail_label.text())

    def test_request_is_deduplicated_and_ignores_missing_files(self):
        loader = self.panel.loader
        self.assertTrue(loader.request(self.real))
        self.assertFalse(loader.request(self.real))
        self.assertFalse(loader.request(self.root / "nope.avi"))
        loader.wait_idle()
        self.assertFalse(loader.request(self.real))

    def test_shutdown_ignores_results_and_stops_new_work(self):
        loader = self.panel.loader
        loader.request(self.real)
        loader.shutdown()
        QTest.qWait(30)
        self.assertFalse(loader.request(self.videos[0]))

    def test_cache_is_bounded(self):
        loader = self.panel.loader
        with patch.object(thumbnails, "CACHE_SIZE", 2):
            for number in range(4):
                path = self.write_video(f"c{number}.mp4")
                loader.request(path)
                loader.wait_idle()
        self.assertLessEqual(len(loader._cache), 2)

    def test_read_preview_of_a_real_video(self):
        preview = thumbnails.read_preview(self.real, max_width=80)
        self.assertTrue(preview.ok)
        self.assertEqual((preview.width, preview.height), (160, 120))
        self.assertEqual(preview.image.width(), 80)
        self.assertEqual(preview.fps, 25.0)
        self.assertAlmostEqual(preview.duration, 3.2, places=1)


class NotesTests(ContextFixture):
    def setUp(self):
        super().setUp()
        self.add()
        self.entry = self.project.videos[0]

    def test_typing_commits_after_a_pause(self):
        self.select("m1.mp4")
        self.panel.notes_edit.setPlainText("jumpy animal")
        self.assertTrue(self.panel._notes_timer.isActive())
        self.assertEqual(self.entry.notes, "")
        self.panel._notes_timer.timeout.emit()
        self.assertEqual(self.entry.notes, "jumpy animal")
        self.assertTrue(self.window.has_unsaved_changes)
        notes_column = self.table.model.headers().index("Notes")
        self.assertEqual(
            self.table.model.index(0, notes_column).data(), "jumpy animal"
        )

    def test_switching_video_commits_first(self):
        self.select("m1.mp4")
        self.panel.notes_edit.setPlainText("for m1")
        self.select("m2.mp4")
        self.assertEqual(self.entry.notes, "for m1")
        self.assertEqual(self.panel.notes_edit.toPlainText(), "")
        self.assertEqual(self.project.videos[1].notes, "")

    def test_saving_commits_pending_notes(self):
        self.select("m1.mp4")
        self.panel.notes_edit.setPlainText("last words")
        self.window.save()
        self.assertEqual(self.entry.notes, "last words")

    def test_notes_edited_in_the_table_reach_the_panel(self):
        self.select("m1.mp4")
        notes_column = self.table.model.headers().index("Notes")
        index = self.table.proxy.index(0, notes_column)
        self.table.proxy.setData(index, "from the table")
        self.assertEqual(self.panel.notes_edit.toPlainText(), "from the table")

    def test_typing_is_not_overwritten_by_a_refresh(self):
        self.select("m1.mp4")
        self.panel.notes_edit.setPlainText("typing...")
        self.project.set_verified([self.entry.key], True)    # refreshes
        self.assertEqual(self.panel.notes_edit.toPlainText(), "typing...")

    def test_programmatic_refresh_does_not_start_the_timer(self):
        self.entry.notes = "set elsewhere"
        self.select("m1.mp4")
        self.assertFalse(self.panel._notes_timer.isActive())
        self.assertEqual(self.panel.notes_edit.toPlainText(), "set elsewhere")


class MultiSelectionTests(ContextFixture):
    def setUp(self):
        super().setUp()
        self.project.set_template(make_settings())
        self.add()

    def test_summary_counts(self):
        self.project.set_verified([self.project.videos[0].key], True)
        self.videos[2].unlink()
        self.table.model.refresh()
        self.select("m1.mp4", "m2.mp4", "m3.mp4")
        self.assertEqual(self.panel.pages.currentIndex(), ContextPanel.PAGE_MULTI)
        text = self.panel.summary_label.text()
        self.assertIn("3 videos selected", text)
        self.assertIn("3 configured, 1 verified", text)
        self.assertIn("1 with issues", text)
        self.assertIn("1 with a missing file", text)
        self.assertFalse(self.panel.bulk_buttons[ACTION_RELINK].isHidden())

    def test_relink_button_hidden_without_missing_files(self):
        self.select("m1.mp4", "m2.mp4")
        self.assertTrue(self.panel.bulk_buttons[ACTION_RELINK].isHidden())

    def test_apply_template_needs_a_template(self):
        self.select("m1.mp4", "m2.mp4")
        button = self.panel.bulk_buttons[ACTION_APPLY_TEMPLATE]
        self.assertTrue(button.isEnabled())
        self.project.template_settings = None
        self.panel.refresh()
        self.assertFalse(button.isEnabled())

    def test_buttons_announce_their_action(self):
        self.select("m1.mp4", "m2.mp4")
        got = []
        self.panel.action_requested.connect(got.append)
        self.prompts.say("confirm", False)     # the Remove question
        for action in (ACTION_VERIFY, ACTION_UNVERIFY, ACTION_APPLY_TEMPLATE,
                       ACTION_REMOVE):
            self.panel.bulk_buttons[action].click()
        self.assertEqual(
            got, [ACTION_VERIFY, ACTION_UNVERIFY, ACTION_APPLY_TEMPLATE,
                  ACTION_REMOVE]
        )

    def test_the_window_runs_the_bulk_actions_on_the_selection(self):
        self.select("m1.mp4", "m2.mp4")
        self.panel.bulk_buttons[ACTION_VERIFY].click()
        self.assertEqual([v.verified for v in self.project.videos],
                         [True, True, False])
        self.panel.bulk_buttons[ACTION_UNVERIFY].click()
        self.assertEqual([v.verified for v in self.project.videos],
                         [False, False, False])
        self.prompts.say("confirm", True)
        self.panel.bulk_buttons[ACTION_REMOVE].click()
        self.assertEqual([v.name for v in self.project.videos], ["m3.mp4"])

    def test_bulk_relink_goes_through_the_window(self):
        old = self.write_video("old/a.mp4")
        self.table.add_paths([old])
        new = self.root / "new"
        new.mkdir()
        old.rename(new / "a.mp4")
        self.table.model.refresh()
        self.select("a.mp4", "m1.mp4")
        self.prompts.say("choose_folder", new)
        self.panel.bulk_buttons[ACTION_RELINK].click()
        self.assertEqual(self.project.find(new / "a.mp4").name, "a.mp4")

    def test_selection_shrinking_to_one_shows_the_single_page(self):
        self.select("m1.mp4", "m2.mp4")
        self.select("m2.mp4")
        self.assertEqual(self.panel.pages.currentIndex(), ContextPanel.PAGE_SINGLE)

    def test_removed_videos_leave_the_selection(self):
        self.select("m1.mp4", "m2.mp4")
        self.table.remove_keys([self.project.videos[0].key], confirm=False)
        self.table.flush_selection()
        self.assertEqual(self.panel.pages.currentIndex(), ContextPanel.PAGE_SINGLE)


class ShutdownTests(ContextFixture):
    def test_shutdown_commits_notes_and_stops_the_reader(self):
        self.add()
        self.select("m1.mp4")
        self.panel.notes_edit.setPlainText("saved on exit")
        self.panel.shutdown()
        self.assertEqual(self.project.videos[0].notes, "saved on exit")
        # The window's loader is shared and stopped by the window itself.
        self.assertTrue(self.panel.loader.request(self.real))

    def test_a_panel_with_its_own_reader_stops_it(self):
        alone = ContextPanel(self.prompts)
        self.addCleanup(alone.deleteLater)
        alone.shutdown()
        self.assertFalse(alone.loader.request(self.real))

    def test_closing_the_window_shuts_the_panel_down(self):
        self.window.close()
        self.assertFalse(self.panel.loader.request(self.real))


if __name__ == "__main__":
    unittest.main()
