import unittest

from mazes.epm import EPM, EPMAnalysisResult
from mazes.y_maze import YMaze, YMazeAnalysisResult

try:
    from PyQt6.QtWidgets import QApplication
    from gui.arm_entry_dialog import ArmEntryParametersDialog
    HAS_QT = True
except ModuleNotFoundError:
    HAS_QT = False

FPS = 10.0


def _populate(result, labels):
    """Mimic Analyzer.finalize_result for a 1 px/frame walk along x."""
    result.fps = FPS
    result.positions = [(float(i), 0.0) for i in range(len(labels))]
    result.timestamps = [i / FPS for i in range(len(labels))]
    result.overlapping_zones = [
        ['undetected'] if label == 'undetected' else [label]
        for label in labels
    ]
    for index, label in enumerate(labels):
        if label == 'undetected':
            continue
        result.time_in_roi[label] = result.time_in_roi.get(label, 0.0) + 1 / FPS
        if index > 0 and labels[index - 1] != 'undetected':
            result.distance_in_roi[label] = (
                result.distance_in_roi.get(label, 0.0) + 1.0
            )


class EPMMetricTests(unittest.TestCase):
    LABELS = [
        'center', 'center',
        'open_arm_1', 'open_arm_1', 'open_arm_1',
        'center',
        'closed_arm_1', 'closed_arm_1', 'closed_arm_1', 'closed_arm_1',
        'center',
        'open_arm_2', 'open_arm_2',
    ]

    def _result(self, labels=None):
        result = EPMAnalysisResult()
        _populate(result, labels or self.LABELS)
        maze = EPM()
        maze.resolve_tracking_labels(result)
        maze.calculate_metrics(result)
        return maze, result

    def test_time_distance_entries_and_percentages(self):
        _, result = self._result()
        self.assertAlmostEqual(result.time_in_open_arms_s, 0.5)
        self.assertAlmostEqual(result.time_in_closed_arms_s, 0.4)
        self.assertAlmostEqual(result.time_in_center_s, 0.4)
        self.assertAlmostEqual(result.distance_in_open_arms, 5.0)
        self.assertAlmostEqual(result.distance_in_closed_arms, 4.0)
        self.assertAlmostEqual(result.distance_in_center, 3.0)
        self.assertEqual(result.entries_to_open_arms, 2)
        self.assertEqual(result.entries_to_closed_arms, 1)
        self.assertEqual(result.total_arm_entries, 3)
        self.assertAlmostEqual(result.percent_time_in_open, 0.5 / 0.9 * 100)
        self.assertAlmostEqual(result.percent_entries_to_open, 2 / 3 * 100)
        self.assertEqual(result.entries_possibly_missed, 0)

    def test_gap_between_arms_is_flagged_in_metrics_and_batch_row(self):
        labels = [
            'center', 'open_arm_1', 'open_arm_1',
            'undetected', 'undetected',
            'closed_arm_1', 'closed_arm_1',
        ]
        maze, result = self._result(labels)
        self.assertEqual(result.total_arm_entries, 1)
        self.assertEqual(result.entries_possibly_missed, 1)
        headers = maze.get_batch_summary_headers()
        row = maze.get_batch_summary_row(result)
        self.assertEqual(len(headers), len(row))
        self.assertEqual(headers[-1], 'entries_possibly_missed')
        self.assertEqual(row[-1], 1)
        self.assertEqual(len(maze.get_batch_summary_row(object())), len(headers))

    def test_zero_percentages_without_arm_time(self):
        _, result = self._result(['center', 'center'])
        self.assertEqual(result.percent_time_in_open, 0)
        self.assertEqual(result.percent_entries_to_open, 0)

    def test_initial_arm_option_changes_entries(self):
        labels = ['open_arm_1', 'open_arm_1', 'center', 'closed_arm_1']
        result = EPMAnalysisResult()
        _populate(result, labels)
        maze = EPM()
        maze.configure_algorithm_parameters({'count_initial_arm': False})
        maze.resolve_tracking_labels(result)
        maze.calculate_metrics(result)
        self.assertEqual(result.total_arm_entries, 1)


class YMazeEntryTests(unittest.TestCase):
    def _result(self, labels):
        result = YMazeAnalysisResult()
        _populate(result, labels)
        maze = YMaze()
        maze.resolve_tracking_labels(result)
        maze.calculate_metrics(result)
        return maze, result

    def test_alternation_with_flagged_gap(self):
        labels = [
            'arm_a', 'center', 'arm_b', 'center', 'arm_c',
            'undetected', 'arm_a', 'center', 'arm_b',
        ]
        maze, result = self._result(labels)
        # The A after the gap is not counted; the triplet stays conservative.
        self.assertEqual(result.arm_sequence, 'ABCB')
        self.assertEqual(result.entries_possibly_missed, 1)
        headers = maze.get_batch_summary_headers()
        row = maze.get_batch_summary_row(result)
        self.assertEqual(len(headers), len(row))
        self.assertEqual(row[headers.index('entries_possibly_missed')], 1)

    def test_clean_run_reports_zero_missed(self):
        _, result = self._result(
            ['arm_a', 'center', 'arm_b', 'center', 'arm_c']
        )
        self.assertEqual(result.arm_sequence, 'ABC')
        self.assertEqual(result.entries_possibly_missed, 0)
        self.assertEqual(result.correct_alternations, 1)

    def test_parameters_round_trip(self):
        maze = YMaze()
        maze.configure_algorithm_parameters(
            {'min_dwell_seconds': 0.4, 'count_initial_arm': False}
        )
        other = YMaze()
        other.configure_algorithm_parameters(maze.get_algorithm_parameters())
        self.assertEqual(other.min_dwell_seconds, 0.4)
        self.assertFalse(other.count_initial_arm)


@unittest.skipUnless(HAS_QT, "GUI runtime is not installed")
class ArmEntryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_edits_maze_parameters(self):
        for maze in (YMaze(), EPM()):
            dialog = ArmEntryParametersDialog(maze)
            self.assertAlmostEqual(dialog.dwell_spin.value(), 0.10)
            self.assertTrue(dialog.initial_check.isChecked())
            dialog.dwell_spin.setValue(0.35)
            dialog.initial_check.setChecked(False)
            maze.configure_algorithm_parameters(dialog.get_parameters())
            self.assertAlmostEqual(maze.min_dwell_seconds, 0.35)
            self.assertFalse(maze.count_initial_arm)
            dialog.close()


if __name__ == "__main__":
    unittest.main()
