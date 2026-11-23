from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


class ParlayAutomaticHistoryTests(unittest.TestCase):
    def test_best_effort_is_removed_from_automatic_ui(self):
        self.assertNotIn('id="best-effort-mode"', HTML)
        self.assertIn("best_effort:false", HTML)

    def test_auto_history_runs_when_any_fixture_league_lacks_history(self):
        self.assertIn("const hasMissingFixtureHistory=fixtureLabelsForTopUp.some", HTML)
        self.assertIn("data.provider==='OddsPapi'&&hasMissingFixtureHistory", HTML)
        self.assertIn("autoTopUpHistoryForFixtures(data)", HTML)
        self.assertIn("data.fixture_competitions_in_window", HTML)

    def test_auto_history_uses_sportapi_box_scores_and_retries_analysis(self):
        self.assertIn("item.box_scores!==false", HTML)
        self.assertIn("include_players:'false'", HTML)
        self.assertIn("data=await requestRecommendation(games,teamStats)", HTML)
        self.assertIn("minimum_history_games:12", HTML)


if __name__ == "__main__":
    unittest.main()
