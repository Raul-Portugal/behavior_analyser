"""The welcome page of the Videos stage: the 3.x import card, recent projects, routing."""
import json
import unittest
from unittest.mock import patch

from tests.project_support import make_settings
from tests.workspace_support import HAS_RUNTIME, WorkspaceFixture

if HAS_RUNTIME:
    from gui.workspace.welcome import recent_import_sources
    from gui.workspace.window import WorkspaceWindow


class WelcomePageTests(WorkspaceFixture):
    def write_template(self, name="tpl.json"):
        path = self.root / name
        make_settings().save_to_file(path)
        return path

    def write_plan(self, name="plan.json"):
        path = self.root / name
        path.write_text(json.dumps({"type": "batch_plan", "plan": {}}),
                        encoding="utf-8")
        return path

    def card_visible(self):
        """Is the 3.x import offer showing on the welcome page?"""
        welcome = self.window.welcome
        return welcome.import_card.isVisibleTo(welcome)

    def test_hidden_without_recent_configurations(self):
        self.assertFalse(self.window.start())
        self.assertFalse(self.card_visible())

    def test_appears_with_recent_templates_and_plans(self):
        self.settings.add_recent_file(self.write_template())
        self.settings.add_recent_file(self.write_plan())
        self.assertFalse(self.window.start())
        card = self.window.welcome
        self.assertTrue(self.card_visible())
        self.assertEqual(
            [kind for _path, kind in card.sources], ["batch_plan", "template"]
        )
        self.assertEqual(len(card.source_buttons), 2)
        self.assertEqual(card.create_button.text(), "Start a new study")
        self.assertEqual(card.open_button.text(), "Open project")
        self.assertEqual(card.classic_button.text(), "Use classic interface")
        self.assertIn("Import 3.x configuration", card.import_button.text())
        self.assertEqual(self.prompts.calls, [])

    def test_lists_at_most_three_and_skips_missing_files(self):
        for n in range(5):
            self.settings.add_recent_file(self.write_template(f"t{n}.json"))
        self.assertEqual(len(recent_import_sources(self.settings)), 3)
        (self.root / "t4.json").unlink()   # the most recent one is gone
        found = recent_import_sources(self.settings)
        self.assertEqual(
            [p.name for p, _ in found], ["t3.json", "t2.json", "t1.json"]
        )

    def test_not_shown_when_there_is_a_last_project(self):
        self.settings.add_recent_file(self.write_template())
        path = self.new_project_at()
        self.window.close_project()
        self.settings.add_recent_project(path)
        self.prompts.say("confirm", False)      # decline "Reopen last project"
        self.window.start()
        self.assertFalse(self.card_visible())

    def test_not_shown_when_a_path_is_given(self):
        self.settings.add_recent_file(self.write_template())
        path = self.new_project_at()
        self.window.close_project()
        self.window.start(path)
        self.assertFalse(self.card_visible())

    def test_dismissal_persists(self):
        self.settings.add_recent_file(self.write_template())
        self.window.start()
        self.assertTrue(self.card_visible())
        self.window.welcome.dismiss_button.click()
        self.assertFalse(self.card_visible())
        self.assertTrue(self.settings.welcome_dismissed())
        # A fresh window on the same settings does not show it again.
        self.dispose_window()
        self.window = WorkspaceWindow(self.settings, self.prompts)
        self.assertFalse(self.window.start())
        self.assertFalse(self.card_visible())

    def test_opening_a_project_dismisses_it(self):
        self.settings.add_recent_file(self.write_template())
        self.window.start()
        self.new_project_at()
        self.assertFalse(self.card_visible())
        self.assertTrue(self.settings.welcome_dismissed())

    def test_create_and_open_buttons_use_the_window(self):
        self.settings.add_recent_file(self.write_template())
        self.window.start()
        card = self.window.welcome
        with patch.object(self.window, "new_project") as new, \
                patch.object(self.window, "open_project") as opened:
            card.create_button.click()
            card.open_button.click()
        new.assert_called_once_with()
        opened.assert_called_once_with()

    def test_listed_template_goes_through_import_routing(self):
        template = self.write_template()
        self.settings.add_recent_file(template)
        self.window.start()
        with patch.object(self.window, "import_template") as routed:
            self.window.welcome.source_buttons[0].click()
        routed.assert_called_once_with(template)

    def test_listed_plan_goes_through_import_routing(self):
        plan = self.write_plan()
        self.settings.add_recent_file(plan)
        self.window.start()
        with patch.object(self.window, "import_batch_plan") as routed:
            self.window.welcome.source_buttons[0].click()
        routed.assert_called_once_with(plan)

    def test_importing_for_real_creates_the_project_and_dismisses(self):
        template = self.write_template()
        self.settings.add_recent_file(template)
        self.window.start()
        target = self.root / "migrated.bmaproj"
        self.prompts.say("save_file", target)
        self.window.welcome.source_buttons[0].click()
        self.assertEqual(self.window.project.file_path, target)
        self.assertIsNotNone(self.window.project.template_settings)
        self.assertFalse(self.card_visible())
        self.assertTrue(self.settings.welcome_dismissed())

    def test_import_button_picks_a_file_and_routes_by_kind(self):
        self.settings.add_recent_file(self.write_template())
        self.window.start()
        plan = self.write_plan("other_plan.json")
        self.prompts.say("open_file", plan)
        with patch.object(self.window, "import_batch_plan") as routed:
            self.window.welcome.import_button.click()
        routed.assert_called_once_with(plan)

    def test_import_button_cancelled_does_nothing(self):
        self.settings.add_recent_file(self.write_template())
        self.window.start()
        self.prompts.say("open_file", None)
        self.window.welcome.import_button.click()
        self.assertTrue(self.card_visible())
        self.assertFalse(self.settings.welcome_dismissed())

    def test_classic_button_switches_and_dismisses(self):
        self.settings.add_recent_file(self.write_template())
        self.window.start()
        with patch.object(self.window, "switch_to_classic",
                          return_value=True) as switched:
            self.window.welcome.classic_button.click()
        switched.assert_called_once_with()
        self.assertTrue(self.settings.welcome_dismissed())

    def test_failed_switch_keeps_the_card(self):
        self.settings.add_recent_file(self.write_template())
        self.window.start()
        with patch.object(self.window, "switch_to_classic", return_value=False):
            self.window.welcome.classic_button.click()
        self.assertTrue(self.card_visible())
        self.assertFalse(self.settings.welcome_dismissed())

    def test_the_start_page_offers_a_new_study_and_opening_one(self):
        from gui.workspace.stages.videos_stage import PAGE_WELCOME
        stage = self.window.videos_stage
        self.assertEqual(stage.page, PAGE_WELCOME)
        self.assertFalse(stage.header.isVisibleTo(stage))
        welcome = self.window.welcome
        self.assertEqual(welcome.hero.title(), "Start a new study")
        self.assertEqual(welcome.open_button.text(), "Open project")
        self.assertFalse(self.card_visible())
        ok, reason = stage.can_advance()
        self.assertFalse(ok)
        self.assertIn("project", reason)

    def test_recent_projects_are_cards_that_open_them(self):
        first = self.new_project_at("a.bmaproj")
        self.window.close_project()
        self.new_project_at("b.bmaproj")
        self.window.close_project()
        welcome = self.window.welcome
        names = [card.path.name for card in welcome.recent_cards]
        self.assertEqual(names, ["b.bmaproj", "a.bmaproj"])
        self.assertFalse(welcome.recent_header.isHidden())
        welcome.recent_cards[1].click()
        self.assertEqual(self.window.project.file_path, first)

    def test_without_recent_projects_there_are_no_cards(self):
        welcome = self.window.welcome
        self.assertEqual(welcome.recent_cards, [])
        self.assertTrue(welcome.recent_header.isHidden())

    def test_at_most_three_recent_projects_are_listed(self):
        for number in range(6):
            self.new_project_at(f"p{number}.bmaproj")
            self.window.close_project()
        self.assertEqual(len(self.window.welcome.recent_cards), 3)

    def test_a_vanished_recent_project_is_reported_and_forgotten(self):
        path = self.new_project_at("gone.bmaproj")
        self.window.close_project()
        card = self.window.welcome.recent_cards[0]
        path.unlink()
        card.click()
        self.assertEqual(len(self.prompts.called("error")), 1)
        self.assertEqual(self.window.welcome.recent_cards, [])


if __name__ == "__main__":
    unittest.main()
