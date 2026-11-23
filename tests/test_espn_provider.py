from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

import espn_provider


def _event(start):
    return {
        "id": "event-123",
        "date": start.isoformat(),
        "season": {"year": start.year},
        "status": {"type": {"completed": True, "state": "post"}},
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"id": "1", "displayName": "Home Team"}, "score": "105"},
                {"homeAway": "away", "team": {"id": "2", "displayName": "Away Team"}, "score": "99"},
            ]
        }],
    }


def _summary():
    def team(team_id):
        return {
            "team": {"id": team_id},
            "statistics": [
                {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": "40-85"},
                {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "12-15"},
                {"name": "offensiveRebounds", "displayValue": "10"},
                {"name": "totalTurnovers", "displayValue": "12"},
            ],
        }
    return {"boxscore": {"teams": [team("1"), team("2")]}}


class ESPNProviderTests(unittest.TestCase):
    def test_competition_aliases_are_explicit_and_keep_women_separate(self):
        self.assertEqual(espn_provider._competition_key("NBA"), "nba")
        self.assertEqual(espn_provider._competition_key("NBA G League Tip-Off Tournament"), "nbagleague")
        self.assertEqual(espn_provider._competition_key("NCAA, Regular Season"), "ncaadivision1men")
        self.assertEqual(espn_provider._competition_key("NCAA Women, Regular Season"), "ncaadivision1women")
        self.assertEqual(espn_provider._competition_key("NBL"), "nbl")
        self.assertNotEqual(
            espn_provider._competition_key("NCAA, Regular Season"),
            espn_provider._competition_key("NCAA Women, Regular Season"),
        )

    def test_team_boxscore_requires_actual_attempts_rebounds_and_turnovers(self):
        entry = _summary()["boxscore"]["teams"][0]
        stats = espn_provider._team_stats(entry, 105.0)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["fga"], 85.0)
        self.assertEqual(stats["fta"], 15.0)
        self.assertEqual(stats["oreb"], 10.0)
        self.assertEqual(stats["tov"], 12.0)
        self.assertAlmostEqual(stats["possessions"], 93.6)
        incomplete = {"statistics": [{"name": "points", "displayValue": "105"}]}
        self.assertIsNone(espn_provider._team_stats(incomplete, 105.0))

    def test_sync_imports_only_completed_games_with_two_complete_team_boxscores(self):
        played = datetime.now(timezone.utc) - timedelta(days=20)
        event = _event(played)

        def fake_get(path, params=None, ttl=3600):
            if path.endswith("/scoreboard"):
                return {"events": [event]}
            if path.endswith("/summary"):
                return _summary()
            raise AssertionError(path)

        with patch.object(espn_provider, "_get_json", side_effect=fake_get):
            result = espn_provider.sync("NBA", days=30, max_games_per_league=10)

        self.assertEqual(result["provider"], "ESPN public JSON")
        self.assertEqual(result["games_received"], 1)
        self.assertEqual(result["complete_boxscore_games"], 1)
        self.assertEqual(result["team_stats_rows"], 2)
        self.assertEqual(result["games"][0]["event_id"], "espn-event-123")
        self.assertEqual({row["team"] for row in result["team_stats"]}, {"Home Team", "Away Team"})

    def test_unsupported_competition_is_not_silently_mapped(self):
        result = espn_provider.sync("EuroLeague", days=10, max_games_per_league=10)
        self.assertEqual(result["games_received"], 0)
        self.assertEqual(result["competitions_supported"], [])


if __name__ == "__main__":
    unittest.main()
