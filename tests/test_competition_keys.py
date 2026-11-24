import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oddspapi_provider

try:
    import app as app_module
except ModuleNotFoundError:  # fastapi not installed in this environment
    app_module = None


class CompetitionKeyTests(unittest.TestCase):
    def test_odds_provider_keeps_summer_league_out_of_nba(self):
        key = oddspapi_provider._competition_key
        self.assertEqual(key("NBA"), "nba")
        for label in ("NBA Summer League", "NBA California Classic", "Salt Lake City Summer League"):
            self.assertNotEqual(key(label), "nba", label)
        self.assertNotEqual(key("NBA G League"), "nba")

    @unittest.skipIf(app_module is None, "fastapi is not installed")
    def test_laketown_squadron_reuses_birmingham_franchise_history(self):
        normalise = app_module._normalise_team_label
        self.assertEqual(normalise("Laketown Squadron"), "laketownsquadron")
        self.assertEqual(normalise("Birmingham Squadron"), "laketownsquadron")
        self.assertEqual(
            normalise("Birmingham Squadron"),
            normalise("Laketown Squadron"),
        )
        self.assertNotEqual(normalise("Birmingham Squadron 2"), "laketownsquadron")

    @unittest.skipIf(app_module is None, "fastapi is not installed")
    def test_history_normaliser_keeps_summer_league_out_of_nba(self):
        norm = app_module._normalise_competition
        self.assertEqual(norm("NBA"), "nba")
        self.assertEqual(norm("NBA Men"), "nba")
        for label in ("NBA Summer League", "NBA California Classic", "Salt Lake City Summer League",
                      "NBA G League"):
            self.assertNotEqual(norm(label), "nba", label)
        self.assertEqual(norm("WNBA"), "wnba")


if __name__ == "__main__":
    unittest.main()
