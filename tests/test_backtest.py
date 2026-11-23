import datetime as dt
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import app as app_module
except ModuleNotFoundError:  # fastapi not installed in this environment
    app_module = None


def synthetic_league(seed=7, teams=8, games=260):
    rng = random.Random(seed)
    strength = {f"T{i}": rng.uniform(-8, 8) for i in range(teams)}
    names = list(strength)
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rows, stats = [], []
    for g in range(games):
        home, away = rng.sample(names, 2)
        diff = strength[home] - strength[away] + 2.5
        pos = 96.0
        hp = round(rng.gauss(108 + diff / 2, 9))
        ap = round(rng.gauss(108 - diff / 2, 9))
        if hp == ap:
            hp += 1
        eid = f"syn-{g}"
        rows.append({
            "event_id": eid, "competition": "NBA", "season": "2025-26",
            "event_time": (start + dt.timedelta(days=g * 1.2)).isoformat(),
            "home_team": home, "away_team": away, "status": "finished",
        })
        stats.append({"event_id": eid, "team": home, "points": hp, "possessions": pos})
        stats.append({"event_id": eid, "team": away, "points": ap, "possessions": pos})
    return rows, stats


@unittest.skipIf(app_module is None, "fastapi is not installed")
class BacktestTests(unittest.TestCase):
    def test_runs_and_reports_calibration(self):
        games, stats = synthetic_league()
        result = app_module._run_backtest(games, stats, 1000, 8, 120, "", 60.0)
        summary = result["summary"]
        self.assertGreater(result["games_evaluated"], 40)
        self.assertIn("brier_score", summary)
        self.assertEqual(len(summary["by_threshold"]), len(app_module.BACKTEST_THRESHOLDS))
        # Teams differ a lot here, so the model should beat "always pick home".
        self.assertTrue(summary["brier_beats_baseline"])
        self.assertGreater(summary["favourite_accuracy"], 0.55)

    def test_future_results_do_not_change_past_predictions(self):
        games, stats = synthetic_league()
        cutoff = 200
        early_ids = {g["event_id"] for g in games[:cutoff]}

        def early_summary(stat_rows):
            return app_module._run_backtest(
                games[:cutoff], [r for r in stat_rows if r["event_id"] in early_ids],
                1000, 8, 120, "", 60.0,
            )["summary"]

        baseline = early_summary(stats)
        tampered = []
        for row in stats:
            row = dict(row)
            if row["event_id"] not in early_ids:
                row["points"] = 999  # absurd future results must be ignored
            tampered.append(row)
        # Full run with tampered future rows, restricted to the same early games.
        after = app_module._run_backtest(
            games[:cutoff], tampered, 1000, 8, 120, "", 60.0)["summary"]
        self.assertEqual(baseline["brier_score"], after["brier_score"])
        self.assertEqual(baseline["games"], after["games"])

    def test_first_games_without_history_are_skipped(self):
        games, stats = synthetic_league(games=30)
        result = app_module._run_backtest(games, stats, 1000, 12, 30, "", 60.0)
        self.assertLess(result["games_evaluated"], 30)
        self.assertGreater(sum(result["skipped"].values()), 0)

    def test_empty_input(self):
        result = app_module._run_backtest([], [], 1000, 12, 50, "", 10.0)
        self.assertEqual(result["games_evaluated"], 0)
        self.assertEqual(result["summary"], {"games": 0})

    def test_wilson_lower_bound(self):
        self.assertIsNone(app_module._wilson_lower_bound(0, 0))
        low = app_module._wilson_lower_bound(70, 100)
        self.assertTrue(0.60 < low < 0.70)


if __name__ == "__main__":
    unittest.main()
