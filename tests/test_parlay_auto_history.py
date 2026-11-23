from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


class ParlayAutomaticHistoryTests(unittest.TestCase):
    def test_best_effort_is_disabled_by_default(self):
        self.assertIn(
            '<input id="best-effort-mode" type="checkbox">',
            HTML,
        )
        self.assertNotIn(
            '<input id="best-effort-mode" type="checkbox" checked>',
            HTML,
        )

    def test_auto_history_runs_only_when_odds_fixtures_have_no_history_match(self):
        self.assertIn(
            "data.provider==='OddsPapi'&&Number(data.matches_matching_imported_competitions||0)===0",
            HTML,
        )
        self.assertIn("autoTopUpHistoryForFixtures(data)", HTML)
        self.assertIn("data.fixture_competitions_in_window", HTML)

    def test_auto_history_uses_sportapi_box_scores_and_retries_analysis(self):
        self.assertIn("item.box_scores!==false", HTML)
        self.assertIn("include_players:'false'", HTML)
        self.assertIn("data=await requestRecommendation(games,teamStats)", HTML)
        self.assertIn("minimum_history_games:12", HTML)


if __name__ == "__main__":
    unittest.main()
