from pathlib import Path
import unittest

from parlay_optimizer import MIN_PARLAY_JOINT_PROBABILITY, best_automatic_parlay


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
ESPN = (ROOT / "espn_provider.py").read_text(encoding="utf-8")
OPTIMIZER = (ROOT / "parlay_optimizer.py").read_text(encoding="utf-8")


def pick(event_id, probability, odds):
    return {
        "event_id": event_id,
        "game": event_id + " game",
        "market": "moneyline",
        "side": "home",
        "line": None,
        "selection": "home",
        "bookmaker": "Book A",
        "odds": odds,
        "probability": probability,
        "passes_strict_filters": True,
        "edge_percentage_points": 4.0,
        "score": 1.0,
    }


class ScopeAndLegProbabilityTests(unittest.TestCase):
    def test_each_leg_must_be_at_least_seventy_percent_but_combo_need_not_be(self):
        result, _ = best_automatic_parlay([
            pick("event-1", 0.70, 2.00),
            pick("event-2", 0.70, 2.00),
        ], 4.00)

        self.assertIsNotNone(result)
        self.assertTrue(all(leg["probability"] >= 0.70 for leg in result["legs"]))
        self.assertAlmostEqual(result["estimated_hit_probability"], 0.49, places=6)
        self.assertLess(result["estimated_hit_probability"], 0.70)
        self.assertGreaterEqual(
            result["estimated_hit_probability"], MIN_PARLAY_JOINT_PROBABILITY
        )
        self.assertEqual(result["minimum_leg_probability"], 0.70)

    def test_a_single_leg_below_seventy_percent_is_rejected(self):
        result, search = best_automatic_parlay([
            pick("event-1", 0.69, 2.00),
            pick("event-2", 0.95, 2.00),
        ], 4.00)

        self.assertIsNone(result)
        self.assertEqual(search["strict_candidate_selections"], 1)

    def test_history_top_up_uses_real_odds_fixtures_not_global_catalogue(self):
        self.assertIn('"fixture_competitions_in_window": sorted({', APP)
        self.assertNotIn('feed.get("available_competitions", [])[:40]', APP)
        self.assertIn("matches_with_odds_loaded", HTML)
        self.assertIn("Partidos compatibles con el historial", HTML)

    def test_birmingham_and_laketown_names_share_one_franchise_history_key(self):
        self.assertIn('if label in {"birminghamsquadron", "laketownsquadron"}', APP)
        self.assertIn('return "laketownsquadron"', APP)

    def test_team_level_history_deficits_are_reported_for_targeted_top_up(self):
        self.assertIn('"history_deficit_examples": history_deficit_examples', APP)
        self.assertIn('"home_history_games": home_games', APP)
        self.assertIn('"away_history_games": away_games', APP)
        self.assertIn("hasTeamHistoryDeficit", HTML)
        self.assertIn("teamDeficitLabels", HTML)

    def test_modelable_leagues_can_be_refreshed_when_fixture_team_is_short(self):
        self.assertIn("const teamDeficitLabels=labels.filter", HTML)
        self.assertIn("const fallbackDeficitLabels=hasShortHistory&&!deficits.length", HTML)
        self.assertIn("Volviendo a consultar cuotas y analizar los equipos", HTML)

    def test_team_names_are_normalized_without_dropping_ascii_characters(self):
        self.assertIn("replace(/[\\u0300-\\u036f]/g", HTML)

    def test_odds_diagnostics_expose_quota_and_exact_history_deficits(self):
        self.assertIn('"odds_requests_made": feed.get("odds_requests_made")', APP)
        self.assertIn('"quota_remaining_after_scan": feed.get("quota_remaining_after_scan")', APP)
        self.assertIn("Equipos por debajo del mínimo", HTML)
        self.assertIn("quotaAndProvider", HTML)

    def test_probability_threshold_scope_is_explicit(self):
        self.assertIn('"minimum_estimated_leg_probability": MIN_ESTIMATED_LEG_PROBABILITY', APP)
        self.assertIn('"minimum_probability_scope": "each_parlay_leg_individually"', APP)
        self.assertIn('"minimum_joint_probability_scope": "whole_parlay_product_estimate"', APP)
        self.assertIn("CADA pick individual de la combinada", HTML)
        self.assertIn("el conjunto no necesita 70%", HTML)

    def test_espn_history_imports_are_processed_in_small_batches(self):
        self.assertIn("MAX_SCOREBOARD_WORKERS = 2", ESPN)
        self.assertIn("MAX_WORKERS = 2", ESPN)
        self.assertIn("for month_offset in range(0, len(months), MAX_SCOREBOARD_WORKERS)", ESPN)
        self.assertIn("for event_offset in range(0, len(chosen), MAX_WORKERS)", ESPN)
        self.assertIn("month_futures.clear()", ESPN)
        self.assertIn("futures.clear()", ESPN)


if __name__ == "__main__":
    unittest.main()
