"""The Qt table adapter over Project.videos."""
import unittest

try:
    from PyQt6.QtCore import (QMimeData, QModelIndex, Qt, QUrl,
                              qInstallMessageHandler)
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QAbstractItemModelTester
    from core.project import START_PER_VIDEO, ZONE_ALIGNED, Project
    from gui.project_table_model import (COL_TRIAL, COL_ZONES, LEADING_COLUMNS,
                                         KeyRole, PathRole, SortRole,
                                         format_clock, format_trial,
                                         ProjectFilterProxyModel,
                                         ProjectTableModel, paths_from_mime,
                                         selected_keys)
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

from tests.project_support import ProjectFixture, make_settings


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class TableFixture(ProjectFixture):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.messages = []
        previous = qInstallMessageHandler(
            lambda mode, context, message: self.messages.append(message)
        )
        self.addCleanup(qInstallMessageHandler, previous)
        self.project = Project(name="P")
        self.project.file_path = self.root / "p.bmaproj"
        self.project.set_template(make_settings())
        self.paths = [
            self.write_video(f"videos/m{n}.mp4", b"x" * n) for n in (1, 2, 3, 4)
        ]
        self.project.add_videos(self.paths)
        for number, entry in enumerate(self.project.videos):
            entry.metadata["group"] = "ctl" if number % 2 == 0 else "trt"
        self.model = ProjectTableModel(self.project)
        self.tester = QAbstractItemModelTester(
            self.model, QAbstractItemModelTester.FailureReportingMode.Warning
        )
        self.changes = []
        self.model.projectChanged.connect(lambda: self.changes.append(1))

    def tearDown(self):
        self.model.set_project(None)
        # QAbstractItemModelTester reports problems as Qt warnings.
        self.assertEqual(
            [m for m in self.messages if "FAIL" in m or "QAbstractItemModelTester" in m],
            [],
        )

    def column(self, header):
        return self.model.headers().index(header)

    def cell(self, row, header, role=Qt.ItemDataRole.DisplayRole):
        return self.model.index(row, self.column(header)).data(role)


class ModelStructureTests(TableFixture):
    def test_columns_follow_the_project_metadata(self):
        self.assertEqual(
            self.model.headers(),
            ["Status", "Video", "Configured", "Settings", "Trial", "Zones", "group",
             "Last run", "Notes", "Issues"],
        )
        self.assertEqual(self.model.rowCount(), 4)
        self.assertEqual(self.model.columnCount(), 10)
        self.assertEqual(self.model.rowCount(self.model.index(0, 0)), 0)
        self.assertEqual(self.model.column_for_metadata("group"), 6)
        self.assertEqual(self.model.column_for_metadata("nope"), -1)

    def test_cells(self):
        self.assertEqual(self.cell(0, "Video"), "m1.mp4")
        self.assertEqual(self.cell(0, "Status"), "Configured")
        self.assertEqual(self.cell(0, "Configured"), "Configured")
        self.assertEqual(self.cell(0, "Settings"), "Template")
        self.assertEqual(self.cell(1, "group"), "trt")
        self.assertEqual(self.cell(0, "Last run"), "")
        self.assertEqual(self.cell(0, "Issues"), "")
        self.assertEqual(
            self.cell(0, "Video", Qt.ItemDataRole.ToolTipRole), str(self.paths[0])
        )
        self.assertIsNotNone(self.cell(0, "Status", Qt.ItemDataRole.DecorationRole))
        self.assertEqual(self.cell(0, "Video", KeyRole), self.project.videos[0].key)
        self.assertEqual(self.cell(0, "Video", PathRole), self.paths[0])

    def test_unconfigured_verified_and_run_cells(self):
        entry = self.project.videos[1]
        entry.settings = None
        entry.last_status = "new"
        self.project.videos[2].verified = True
        self.project.videos[0].last_run_dir = self.root / "out" / "analysis_run_9"
        self.model.refresh()
        self.assertEqual(self.cell(1, "Configured"), "Not configured")
        self.assertEqual(self.cell(1, "Settings"), "")
        self.assertEqual(self.cell(1, "Status"), "New")
        self.assertEqual(self.cell(2, "Configured"), "Verified")
        self.assertEqual(self.cell(0, "Last run"), "analysis_run_9")

    def test_issues_column_comes_from_validate(self):
        self.paths[2].unlink()
        self.project.videos[1].settings = None
        self.model.refresh_issues()
        self.assertIn("no settings", self.cell(1, "Issues"))
        self.assertIn("not found", self.cell(2, "Issues"))
        self.assertIn("not found", self.cell(2, "Issues", Qt.ItemDataRole.ToolTipRole))
        self.assertEqual(self.cell(0, "Issues"), "")
        self.assertEqual([i.code for i in self.model.issues_for(
            self.project.videos[2].key)], ["missing_file"])

    def test_project_level_issues(self):
        self.project.run_options.sample_sheet = self.root / "no.csv"
        self.assertEqual(
            [i.code for i in self.model.project_issues()], ["sample_sheet_missing"]
        )

    def test_empty_and_projectless_models(self):
        empty = ProjectTableModel()
        self.assertEqual(empty.rowCount(), 0)
        self.assertEqual(empty.columnCount(), 9)
        self.assertEqual(empty.add_videos([self.paths[0]]).added, [])
        tester = QAbstractItemModelTester(
            empty, QAbstractItemModelTester.FailureReportingMode.Warning
        )
        empty.set_project(Project())
        self.assertEqual(empty.rowCount(), 0)
        del tester

    def test_sort_roles(self):
        self.project.videos[0].last_status = "failed"
        self.model.refresh()
        self.assertLess(
            self.cell(1, "Status", SortRole), self.cell(0, "Status", SortRole)
        )
        self.assertEqual(self.cell(0, "Video", SortRole), "m1.mp4")


class EditingTests(TableFixture):
    def test_only_metadata_and_notes_are_editable(self):
        editable = []
        for column in range(self.model.columnCount()):
            flags = self.model.flags(self.model.index(0, column))
            if flags & Qt.ItemFlag.ItemIsEditable:
                editable.append(self.model.headers()[column])
        self.assertEqual(editable, ["group", "Notes"])
        self.assertFalse(self.model.setData(self.model.index(0, 1), "x"))
        self.assertEqual(self.project.videos[0].name, "m1.mp4")

    def test_editing_metadata_updates_the_project_and_signals(self):
        index = self.model.index(0, self.column("group"))
        seen = []
        self.model.dataChanged.connect(lambda a, b, r=None: seen.append((a.row(), a.column())))
        self.assertTrue(self.model.setData(index, "treated"))
        self.assertEqual(self.project.videos[0].metadata["group"], "treated")
        self.assertEqual(index.data(), "treated")
        self.assertEqual(seen, [(0, 6)])
        self.assertEqual(len(self.changes), 1)

    def test_editing_notes(self):
        index = self.model.index(2, self.column("Notes"))
        self.assertTrue(self.model.setData(index, "good run"))
        self.assertEqual(self.project.videos[2].notes, "good run")
        self.assertEqual(len(self.changes), 1)

    def test_unchanged_value_emits_nothing(self):
        index = self.model.index(0, self.column("group"))
        self.assertTrue(self.model.setData(index, "ctl"))
        self.assertEqual(self.changes, [])

    def test_wrong_role_is_refused(self):
        index = self.model.index(0, self.column("group"))
        self.assertFalse(
            self.model.setData(index, "x", Qt.ItemDataRole.DisplayRole)
        )

    def test_adding_and_removing_a_metadata_column(self):
        self.assertTrue(self.model.add_metadata_column("sex"))
        self.assertFalse(self.model.add_metadata_column("sex"))
        self.assertEqual(self.model.headers()[6:8], ["group", "sex"])
        self.assertEqual(self.model.columnCount(), 11)
        self.assertTrue(self.model.setData(
            self.model.index(1, self.model.column_for_metadata("sex")), "F"
        ))
        self.assertEqual(self.project.videos[1].metadata["sex"], "F")
        self.assertTrue(self.model.remove_metadata_column("group"))
        self.assertFalse(self.model.remove_metadata_column("group"))
        self.assertEqual(self.model.headers()[6], "sex")
        self.assertNotIn("group", self.project.videos[0].metadata)


class RowOperationTests(TableFixture):
    def test_add_videos_appends_rows_and_reports_duplicates(self):
        new = self.write_video("videos/m9.mp4")
        inserted = []
        self.model.rowsInserted.connect(lambda p, a, b: inserted.append((a, b)))
        result = self.model.add_videos([new, self.paths[0]])
        self.assertEqual([v.name for v in result.added], ["m9.mp4"])
        self.assertEqual(result.duplicates, [self.paths[0]])
        self.assertEqual(inserted, [(4, 4)])
        self.assertEqual(self.cell(4, "Video"), "m9.mp4")
        self.assertEqual(self.cell(4, "Status"), "Configured")
        self.assertEqual(len(self.changes), 1)
        self.assertEqual(self.model.add_videos([self.paths[0]]).added, [])
        self.assertEqual(len(self.changes), 1)

    def test_remove_keys_removes_contiguous_blocks(self):
        removed_blocks = []
        self.model.rowsRemoved.connect(lambda p, a, b: removed_blocks.append((a, b)))
        keys = self.model.keys_for_rows([0, 1, 3])
        removed = self.model.remove_keys(keys)
        self.assertEqual(len(removed), 3)
        self.assertEqual(removed_blocks, [(3, 3), (0, 1)])
        self.assertEqual(self.model.rowCount(), 1)
        self.assertEqual(self.cell(0, "Video"), "m3.mp4")
        self.assertEqual(self.model.remove_keys(["nope"]), [])
        self.assertEqual(len(self.changes), 1)

    def test_removing_the_last_video_with_a_column_drops_that_column(self):
        self.project.videos[3].metadata["only_here"] = "x"
        self.model.refresh()
        self.assertIn("only_here", self.model.headers())
        self.model.remove_keys([self.project.videos[3].key])
        self.assertNotIn("only_here", self.model.headers())

    def test_move_keys_keeps_persistent_indexes_on_their_video(self):
        key = self.project.videos[0].key
        persistent = [self.model.index(0, 1)]
        from PyQt6.QtCore import QPersistentModelIndex
        tracked = QPersistentModelIndex(persistent[0])
        self.assertTrue(self.model.move_keys([key], 2))
        self.assertEqual(self.cell(2, "Video"), "m1.mp4")
        self.assertEqual(tracked.row(), 2)
        self.assertEqual([v.name for v in self.project.videos],
                         ["m2.mp4", "m3.mp4", "m1.mp4", "m4.mp4"])
        self.assertFalse(self.model.move_keys([self.model.key_at(0)], -1))
        self.assertEqual(len(self.changes), 1)

    def test_selection_helpers(self):
        keys = [v.key for v in self.project.videos]
        self.assertEqual(self.model.keys_for_rows([3, 1, 1, 99, -1]), [keys[1], keys[3]])
        self.assertEqual(self.model.rows_for_keys([keys[2], keys[0]]), [0, 2])
        self.assertEqual(self.model.row_for_key(keys[2]), 2)
        self.assertEqual(self.model.row_for_key("nope"), -1)
        indexes = [self.model.index(2, 0), self.model.index(2, 3),
                   self.model.index(0, 1), QModelIndex()]
        self.assertEqual(self.model.keys_for_indexes(indexes), [keys[0], keys[2]])
        self.assertEqual(selected_keys(indexes), [keys[2], keys[0]])
        self.assertEqual(
            [e.key for e in self.model.entries_for_keys([keys[3], keys[1]])],
            [keys[1], keys[3]],
        )
        self.assertEqual(self.model.entry_at(1).key, keys[1])


class ExternalChangeTests(TableFixture):
    def test_changes_made_on_the_project_refresh_the_model(self):
        resets = []
        self.model.modelReset.connect(lambda: resets.append(1))
        self.project.add_videos([self.write_video("videos/m7.mp4")])
        self.assertEqual(self.model.rowCount(), 5)
        self.assertEqual(resets, [1])
        self.assertEqual(len(self.changes), 1)

    def test_same_shape_changes_only_report_data_changed(self):
        resets, changed = [], []
        self.model.modelReset.connect(lambda: resets.append(1))
        self.model.dataChanged.connect(lambda a, b, r=None: changed.append((a.row(), b.row())))
        self.project.set_video_notes(self.project.videos[1].key, "outside")
        self.assertEqual(resets, [])
        self.assertEqual(changed, [(0, 3)])
        self.assertEqual(self.cell(1, "Notes"), "outside")

    def test_new_metadata_column_from_the_project_shows_up(self):
        self.project.add_metadata_column("sex")
        self.assertIn("sex", self.model.headers())

    def test_switching_projects_detaches_the_listener(self):
        other = Project()
        self.model.set_project(other)
        self.assertEqual(self.model.rowCount(), 0)
        self.project.add_videos([self.write_video("videos/m8.mp4")])
        self.assertEqual(self.model.rowCount(), 0)
        self.assertIs(self.model.project(), other)

    def test_a_deleted_model_removes_its_listener_on_the_next_change(self):
        from PyQt6 import sip
        extra = ProjectTableModel(self.project)
        before = len(self.project._listeners)
        sip.delete(extra)
        self.project.set_video_notes(self.project.videos[0].key, "x")
        self.assertEqual(len(self.project._listeners), before - 1)

    def test_deleting_the_model_does_not_leave_a_live_listener(self):
        before = len(self.project._listeners)
        extra = ProjectTableModel(self.project)
        self.assertEqual(len(self.project._listeners), before + 1)
        extra.set_project(None)
        self.assertEqual(len(self.project._listeners), before)


class ProxyTests(TableFixture):
    def setUp(self):
        super().setUp()
        self.proxy = ProjectFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.names = lambda: [
            self.proxy.index(r, 1).data() for r in range(self.proxy.rowCount())
        ]

    def test_text_filter_searches_metadata_notes_and_names(self):
        self.project.set_video_notes(self.project.videos[3].key, "excellent")
        self.proxy.set_text_filter("TRT")
        self.assertEqual(self.names(), ["m2.mp4", "m4.mp4"])
        self.proxy.set_text_filter("trt excellent")
        self.assertEqual(self.names(), ["m4.mp4"])
        self.proxy.set_text_filter("m3")
        self.assertEqual(self.names(), ["m3.mp4"])
        self.proxy.set_text_filter("")
        self.assertEqual(len(self.names()), 4)

    def test_status_filter(self):
        self.project.videos[0].last_status = "failed"
        self.project.videos[1].last_status = "done"
        self.model.refresh()
        self.proxy.set_status_filter(["failed", "done"])
        self.assertEqual(self.names(), ["m1.mp4", "m2.mp4"])
        self.proxy.set_text_filter("m2")
        self.assertEqual(self.names(), ["m2.mp4"])
        self.proxy.clear_filters()
        self.assertEqual(len(self.names()), 4)

    def test_sorting_uses_the_sort_role(self):
        self.project.videos[2].last_status = "failed"
        self.model.refresh()
        self.proxy.sort(1, Qt.SortOrder.DescendingOrder)
        self.assertEqual(self.names(), ["m4.mp4", "m3.mp4", "m2.mp4", "m1.mp4"])
        self.proxy.sort(0, Qt.SortOrder.DescendingOrder)
        self.assertEqual(self.names()[0], "m3.mp4")  # failed sorts above configured

    def test_selection_through_the_proxy_reads_the_same_keys(self):
        self.proxy.sort(1, Qt.SortOrder.DescendingOrder)
        keys = [v.key for v in self.project.videos]
        picked = [self.proxy.index(0, 2), self.proxy.index(1, 0)]
        self.assertEqual(selected_keys(picked), [keys[3], keys[2]])
        self.assertEqual(self.proxy.keys_for_rows([1, 0, 0, 9]), [keys[3], keys[2]])
        self.assertEqual(self.proxy.all_keys(), keys[::-1])
        self.assertEqual(self.proxy.row_for_key(keys[0]), 3)
        self.assertEqual(self.proxy.row_for_key("nope"), -1)

    def test_a_table_view_selection_yields_keys_and_survives_moves(self):
        from PyQt6.QtCore import QItemSelectionModel
        from PyQt6.QtWidgets import QAbstractItemView, QTableView
        view = QTableView()
        self.addCleanup(view.deleteLater)
        view.setModel(self.proxy)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        flags = (QItemSelectionModel.SelectionFlag.Select
                 | QItemSelectionModel.SelectionFlag.Rows)
        for row in (0, 2):
            view.selectionModel().select(self.proxy.index(row, 0), flags)
        keys = [v.key for v in self.project.videos]
        picked = selected_keys(view.selectionModel().selectedRows())
        self.assertEqual(sorted(picked), sorted([keys[0], keys[2]]))
        # Move the first selected video down: the selection follows it.
        self.assertTrue(self.model.move_keys([keys[0]], 1))
        picked = selected_keys(view.selectionModel().selectedRows())
        self.assertEqual(sorted(picked), sorted([keys[0], keys[2]]))
        self.assertEqual(self.proxy.row_for_key(keys[0]), 1)

    def test_edits_through_the_proxy_reach_the_project(self):
        self.proxy.sort(1, Qt.SortOrder.DescendingOrder)
        index = self.proxy.index(0, self.column("group"))  # m4
        self.assertTrue(self.proxy.setData(index, "edited"))
        self.assertEqual(self.project.videos[3].metadata["group"], "edited")


class DropTests(TableFixture):
    def _mime(self, *paths):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
        return mime

    def test_files_and_folders_become_video_paths(self):
        folder = self.root / "incoming"
        folder.mkdir()
        (folder / "x.mp4").write_bytes(b"v")
        (folder / "y.avi").write_bytes(b"v")
        (folder / "readme.txt").write_bytes(b"t")
        sub = folder / "sub"
        sub.mkdir()
        (sub / "z.mov").write_bytes(b"v")
        loose = self.write_video("loose/l.mkv")
        notes = self.write_video("loose/notes.txt")
        flat, ignored = paths_from_mime(self._mime(folder, loose, notes))
        self.assertEqual([p.name for p in flat], ["x.mp4", "y.avi", "l.mkv"])
        self.assertEqual(ignored, [notes])
        deep, _ = paths_from_mime(self._mime(folder), recursive=True)
        self.assertEqual(sorted(p.name for p in deep), ["x.mp4", "y.avi", "z.mov"])
        self.assertEqual(paths_from_mime(QMimeData()), ([], []))

    def test_drop_emits_paths_for_the_shell_and_adds_nothing(self):
        loose = self.write_video("loose/l.mkv")
        notes = self.write_video("loose/notes.txt")
        dropped, ignored = [], []
        self.model.pathsDropped.connect(dropped.append)
        self.model.pathsIgnored.connect(ignored.append)
        mime = self._mime(loose, notes)
        self.assertEqual(self.model.mimeTypes(), ["text/uri-list"])
        self.assertTrue(self.model.canDropMimeData(
            mime, Qt.DropAction.CopyAction, -1, -1, QModelIndex()))
        self.assertTrue(self.model.dropMimeData(
            mime, Qt.DropAction.CopyAction, -1, -1, QModelIndex()))
        self.assertEqual(dropped, [[loose]])
        self.assertEqual(ignored, [[notes]])
        self.assertEqual(self.model.rowCount(), 4)

    def test_unusable_drops_are_refused(self):
        notes = self.write_video("loose/notes.txt")
        self.assertFalse(self.model.dropMimeData(
            self._mime(notes), Qt.DropAction.CopyAction, -1, -1, QModelIndex()))
        self.assertFalse(self.model.dropMimeData(
            QMimeData(), Qt.DropAction.CopyAction, -1, -1, QModelIndex()))
        self.assertFalse(self.model.canDropMimeData(
            self._mime(notes), Qt.DropAction.MoveAction, -1, -1, QModelIndex()))

    def test_drop_through_the_proxy(self):
        proxy = ProjectFilterProxyModel()
        proxy.setSourceModel(self.model)
        loose = self.write_video("loose/l.mkv")
        dropped = []
        self.model.pathsDropped.connect(dropped.append)
        self.assertTrue(proxy.dropMimeData(
            self._mime(loose), Qt.DropAction.CopyAction, -1, -1, QModelIndex()))
        self.assertEqual(dropped, [[loose]])

    def test_the_model_accepts_drops_on_empty_space_only(self):
        self.assertTrue(
            self.model.flags(QModelIndex()) & Qt.ItemFlag.ItemIsDropEnabled
        )
        self.assertEqual(self.model.supportedDropActions(), Qt.DropAction.CopyAction)


class TrialAndZoneColumnTests(TableFixture):
    def test_the_new_columns_sit_after_settings(self):
        self.assertEqual(self.model.headers()[COL_TRIAL], "Trial")
        self.assertEqual(self.model.headers()[COL_ZONES], "Zones")
        self.assertEqual(LEADING_COLUMNS, COL_ZONES + 1)
        self.assertEqual(self.model.column_for_metadata("group"), LEADING_COLUMNS)

    def test_a_template_follower_shows_the_whole_video(self):
        self.assertEqual(self.cell(0, "Trial"), "whole video")
        self.assertEqual(self.cell(0, "Zones"), "Template")
        self.assertEqual(self.cell(0, "Configured"), "Configured")

    def test_trial_times_are_shown_as_a_range(self):
        self.project.set_trial(self.project.videos[0].key, start=12, end=312)
        self.project.set_trial(self.project.videos[1].key, start=65.5)
        self.project.set_trial(
            self.project.videos[2].key, start=3700, end=3790.0
        )
        self.assertEqual(self.cell(0, "Trial"), "0:12 – 5:12")
        self.assertEqual(self.cell(1, "Trial"), "1:05.5 – end")
        self.assertEqual(self.cell(2, "Trial"), "1:01:40 – 1:03:10")
        self.assertEqual(self.cell(3, "Trial"), "whole video")
        self.assertEqual(self.cell(0, "Configured"), "Configured")

    def test_the_protocol_duration_shows_in_the_trial_column(self):
        self.project.set_trial_protocol(duration_s=300.0)
        self.assertEqual(self.cell(3, "Trial"), "0:00 – 5:00")

    def test_a_missing_start_is_flagged_in_both_columns(self):
        self.project.set_trial_protocol(start_mode=START_PER_VIDEO)
        self.project.set_trial(self.project.videos[0].key, start=0.0)
        # 0 is a deliberate start; the others still need theirs.
        self.assertEqual(self.cell(0, "Trial"), "whole video")
        self.assertEqual(self.cell(0, "Configured"), "Configured")
        self.assertEqual(self.cell(1, "Trial"), "⚠ set start")
        self.assertEqual(self.cell(1, "Configured"), "Needs start")
        self.project.videos[1].verified = True
        self.assertEqual(self.cell(1, "Configured"), "Needs start")

    def test_invalid_times_are_flagged(self):
        self.project.set_trial(self.project.videos[0].key, start=50, end=20)
        self.assertEqual(self.cell(0, "Trial"), "⚠ invalid times")
        self.assertEqual(self.cell(0, "Configured"), "Invalid trial")
        self.assertIn(
            "end time", self.cell(0, "Trial", Qt.ItemDataRole.ToolTipRole)
        )
        self.assertIn("trial_invalid", [
            i.code for i in self.model.issues_for(self.project.videos[0].key)
        ])

    def test_zone_status_labels(self):
        first, second, third, fourth = (v.key for v in self.project.videos)
        self.project.set_zone_override(first, make_settings(offset=7).roi_manager)
        self.project.set_zone_override(
            second, make_settings(offset=9).roi_manager, source=ZONE_ALIGNED
        )
        self.project.customize(third, make_settings(offset=3))
        self.assertEqual(
            [self.cell(row, "Zones") for row in range(4)],
            ["Adjusted", "Aligned", "Custom", "Template"],
        )
        self.assertEqual(fourth, self.project.keys()[3])
        self.assertTrue(self.cell(0, "Zones", Qt.ItemDataRole.ToolTipRole))

    def test_unconfigured_videos_have_empty_cells(self):
        self.project.videos[1].settings = None
        self.model.refresh()
        self.assertEqual(self.cell(1, "Trial"), "")
        self.assertEqual(self.cell(1, "Zones"), "")
        self.assertEqual(self.cell(1, "Configured"), "Not configured")
        self.assertIsNone(self.cell(1, "Trial", Qt.ItemDataRole.ToolTipRole))

    def test_columns_are_read_only(self):
        for column in (COL_TRIAL, COL_ZONES):
            index = self.model.index(0, column)
            self.assertFalse(self.model.flags(index) & Qt.ItemFlag.ItemIsEditable)
            self.assertFalse(self.model.setData(index, "x"))

    def test_the_cells_follow_changes_made_to_the_project(self):
        seen = []
        self.model.dataChanged.connect(lambda a, b, r=None: seen.append(1))
        self.project.set_trial(self.project.videos[0].key, start=30.0)
        self.assertTrue(seen)
        self.assertEqual(self.cell(0, "Trial"), "0:30 – end")

    def test_sorting_puts_problems_first_then_by_start(self):
        proxy = ProjectFilterProxyModel()
        proxy.setSourceModel(self.model)
        self.project.set_trial_protocol(start_mode=START_PER_VIDEO)
        a, b, c, d = (v.key for v in self.project.videos)
        self.project.set_trial(a, start=40.0)
        self.project.set_trial(b, start=10.0)
        self.project.set_trial(c, start=5.0, end=1.0)            # invalid
        proxy.sort(COL_TRIAL)
        self.assertEqual(proxy.all_keys(), [c, d, b, a])

    def test_filter_text_finds_the_new_cells(self):
        proxy = ProjectFilterProxyModel()
        proxy.setSourceModel(self.model)
        self.project.set_zone_override(
            self.project.videos[2].key, make_settings(offset=7).roi_manager
        )
        proxy.set_text_filter("adjusted")
        self.assertEqual(proxy.all_keys(), [self.project.videos[2].key])

    def test_clock_formatting(self):
        self.assertEqual(format_clock(0), "0:00")
        self.assertEqual(format_clock(59.96), "1:00")
        self.assertEqual(format_clock(12.5), "0:12.5")
        self.assertEqual(format_clock(3600), "1:00:00")
        self.assertEqual(format_clock(-5), "-0:05")
        self.assertEqual(format_trial(0, None), "whole video")
        self.assertEqual(format_trial(10, None), "0:10 – end")
        self.assertEqual(format_trial(0, 60), "0:00 – 1:00")


if __name__ == "__main__":
    unittest.main()
