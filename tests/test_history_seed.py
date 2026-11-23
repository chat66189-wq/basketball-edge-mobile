import json
import unittest
from collections import defaultdict, Counter
from pathlib import Path


SEED_PATH = Path(__file__).resolve().parents[1] / "data" / "history-seed-2026-11-23.json"


class HistorySeedBackupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with SEED_PATH.open("r", encoding="utf-8") as handle:
            cls.payload = json.load(handle)
        cls.records = cls.payload["records"]
        cls.games = [row["data"] for row in cls.records if row.get("kind") == "dataset" and row.get("type") == "games"]
        cls.stats = [row["data"] for row in cls.records if row.get("kind") == "dataset" and row.get("type") == "team_stats"]

    def test_snapshot_format_and_record_counts(self):
        self.assertEqual(self.payload["app"], "Basketball Edge Analyst v20 iPhone UI")
        self.assertEqual(len(self.records), 3186)
        self.assertEqual(len(self.games), 1064)
        self.assertEqual(len(self.stats), 2122)
        self.assertEqual(
            Counter(row.get("type") for row in self.records),
            Counter({"games": 1064, "team_stats": 2122}),
        )

    def test_game_ids_and_team_box_scores_are_consistent(self):
        game_ids = [str(game["event_id"]) for game in self.games]
        self.assertEqual(len(game_ids), len(set(game_ids)))
        stats_by_event = defaultdict(list)
        for row in self.stats:
            stats_by_event[str(row["event_id"])].append(row)

        complete = 0
        missing = 0
        for game in self.games:
            self.assertEqual(game.get("status"), "finished")
            rows = stats_by_event.get(str(game["event_id"]), [])
            if not rows:
                missing += 1
                continue
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {str(row["team"]) for row in rows},
                {str(game["home_team"]), str(game["away_team"])},
            )
            complete += 1
        self.assertEqual(complete, 1061)
        self.assertEqual(missing, 3)

    def test_nba_seed_is_not_misrepresented_as_twelve_game_team_history(self):
        nba = [game for game in self.games if game.get("competition") == "NBA"]
        appearances = Counter()
        for game in nba:
            appearances[game["home_team"]] += 1
            appearances[game["away_team"]] += 1
        self.assertEqual(len(nba), 61)
        self.assertEqual(len(appearances), 30)
        self.assertLess(max(appearances.values()), 12)


if __name__ == "__main__":
    unittest.main()
