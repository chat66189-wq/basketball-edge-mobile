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

    def test_sync_keeps_good_months_when_one_scoreboard_returns_invalid_json(self):
        now = datetime.now(timezone.utc)
        played = now - timedelta(days=10)
        bad_month = "190001"
        good_month = played.strftime("%Y%m")
        event = _event(played)

        def fake_get(path, params=None, ttl=3600):
            if path.endswith("/scoreboard"):
                if params.get("dates") == bad_month:
                    raise espn_provider.ESPNError("invalid json", 502)
                if params.get("dates") == good_month:
                    return {"events": [event]}
                return {"events": []}
            if path.endswith("/summary"):
                return _summary()
            raise AssertionError(path)

        with patch.object(espn_provider, "_month_keys", return_value=[bad_month, good_month]):
            with patch.object(espn_provider, "_get_json", side_effect=fake_get):
                result = espn_provider.sync("NBA", days=40, max_games_per_league=10)

        self.assertEqual(result["games_received"], 1)
        self.assertEqual(result["team_stats_rows"], 2)
        self.assertEqual(result["failed_scoreboards"], 1)
        self.assertTrue(any("calendarios mensuales no se pudieron leer" in warning
                            for warning in result["warnings"]))

    def test_sync_accepts_a_bounded_date_chunk_and_excludes_other_dates(self):
        now = datetime.now(timezone.utc)
        inside = now - timedelta(days=4)
        outside = now - timedelta(days=18)
        included = _event(inside)
        included["id"] = "included-event"
        excluded = _event(outside)
        excluded["id"] = "excluded-event"

        def fake_get(path, params=None, ttl=3600):
            if path.endswith("/scoreboard"):
                return {"events": [included, excluded]}
            if path.endswith("/summary"):
                return _summary()
            raise AssertionError(path)

        with patch.object(espn_provider, "_get_json", side_effect=fake_get):
            result = espn_provider.sync(
                "NBA",
                start_date=inside.date().isoformat(),
                end_date=now.date().isoformat(),
                max_games_per_league=10,
            )

        self.assertEqual(result["games_received"], 1)
        self.assertEqual(result["start_date"], inside.date().isoformat())
        self.assertEqual(result["end_date"], now.date().isoformat())
        self.assertEqual(result["games"][0]["event_id"], "espn-included-event")

    def test_sync_rejects_unpaired_or_oversized_date_chunks(self):
        with self.assertRaises(espn_provider.ESPNError) as unpaired:
            espn_provider.sync("NBA", start_date="2026-11-01")
        self.assertEqual(unpaired.exception.status, 422)

        with self.assertRaises(espn_provider.ESPNError) as oversized:
            espn_provider.sync("NBA", start_date="2026-10-01", end_date="2026-11-01")
        self.assertEqual(oversized.exception.status, 422)

    def test_unsupported_competition_is_not_silently_mapped(self):
        result = espn_provider.sync("EuroLeague", days=10, max_games_per_league=10)
        self.assertEqual(result["games_received"], 0)
        self.assertEqual(result["competitions_supported"], [])


if __name__ == "__main__":
    unittest.main()
