import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.batch_plan_keys import (build_plan_by_key, match_videos_to_plan,
                                  plan_key_for)


class BatchPlanKeyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.plan_file = self.root / "plans" / "batch_plan.json"
        self.a = self.root / "day1" / "mouse1.mp4"
        self.b = self.root / "day2" / "mouse1.mp4"
        self.c = self.root / "day2" / "mouse2.mp4"

    def test_keys_are_relative_posix_paths(self):
        self.assertEqual(
            plan_key_for(self.a, self.plan_file), "../day1/mouse1.mp4"
        )
        inside = self.root / "plans" / "clips" / "m.mp4"
        self.assertEqual(plan_key_for(inside, self.plan_file), "clips/m.mp4")

    def test_absolute_fallback_on_another_drive(self):
        with patch("core.batch_plan_keys.os.path.relpath",
                   side_effect=ValueError("different drive")):
            key = plan_key_for(self.a, self.plan_file)
        self.assertEqual(key, self.a.as_posix())

    def test_duplicate_names_get_distinct_keys(self):
        plan = build_plan_by_key({self.a: "A", self.b: "B"}, self.plan_file)
        self.assertEqual(len(plan), 2)
        self.assertEqual(set(plan.values()), {"A", "B"})

    def test_same_names_in_different_folders_match_their_own_entry(self):
        keys = [plan_key_for(self.a, self.plan_file),
                plan_key_for(self.b, self.plan_file)]
        result = match_videos_to_plan([self.b, self.a], keys, self.plan_file)
        self.assertEqual(result.unmatched, [])
        self.assertEqual(result.matched[self.a], keys[0])
        self.assertEqual(result.matched[self.b], keys[1])

    def test_absolute_keys_match_exactly(self):
        keys = [self.a.as_posix()]
        result = match_videos_to_plan([self.a], keys, self.plan_file)
        self.assertEqual(result.matched, {self.a: keys[0]})

    def test_old_name_keyed_plan_matches_when_unambiguous(self):
        result = match_videos_to_plan(
            [self.a, self.c], ["mouse1.mp4", "mouse2.mp4"], self.plan_file
        )
        self.assertEqual(result.unmatched, [])
        self.assertEqual(result.matched[self.a], "mouse1.mp4")
        self.assertEqual(result.matched[self.c], "mouse2.mp4")

    def test_old_plan_is_ambiguous_for_two_selected_duplicates(self):
        result = match_videos_to_plan(
            [self.a, self.b], ["mouse1.mp4"], self.plan_file
        )
        self.assertEqual(result.matched, {})
        self.assertEqual({v for v, _ in result.unmatched}, {self.a, self.b})

    def test_ambiguous_plan_entries_are_not_guessed(self):
        other = self.root / "day3" / "mouse1.mp4"
        keys = [plan_key_for(self.a, self.plan_file),
                plan_key_for(self.b, self.plan_file)]
        result = match_videos_to_plan([other], keys, self.plan_file)
        self.assertEqual(result.matched, {})
        self.assertEqual([v for v, _ in result.unmatched], [other])

    def test_video_missing_from_plan_is_unmatched(self):
        result = match_videos_to_plan(
            [self.a, self.c], ["mouse1.mp4"], self.plan_file
        )
        self.assertEqual(result.matched, {self.a: "mouse1.mp4"})
        self.assertEqual([v for v, _ in result.unmatched], [self.c])

    def test_exact_match_does_not_leak_into_filename_fallback(self):
        # b is matched exactly; the other same-named video must not inherit
        # the remaining entry just because the filename is the same.
        other = self.root / "day3" / "mouse1.mp4"
        keys = [plan_key_for(self.b, self.plan_file)]
        result = match_videos_to_plan([self.b, other], keys, self.plan_file)
        self.assertEqual(list(result.matched), [self.b])
        self.assertEqual([v for v, _ in result.unmatched], [other])


if __name__ == "__main__":
    unittest.main()
