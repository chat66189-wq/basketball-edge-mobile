from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
SERVICE_WORKER = (ROOT / "sw.js").read_text(encoding="utf-8")
ODDSPAPI = (ROOT / "oddspapi_provider.py").read_text(encoding="utf-8")
OPTIMIZER = (ROOT / "parlay_optimizer.py").read_text(encoding="utf-8")


class StrictParlayPolicyTests(unittest.TestCase):
    def test_automatic_parlays_cannot_enable_best_effort(self):
        self.assertIn("best_effort_requested = False", APP)
        self.assertNotIn('id="best-effort-mode"', HTML)
        self.assertIn("best_effort:false", HTML)

    def test_each_leg_keeps_seventy_percent_conservative_minimum(self):
        self.assertIn("MIN_ESTIMATED_LEG_PROBABILITY = 0.70", APP)
        self.assertIn("MIN_RAW_MODEL_PROBABILITY = 0.72", APP)
        self.assertIn('"conservative_probability_below_70pct"', APP)

    def test_automatic_search_uses_alternative_aware_grouped_optimizer(self):
        self.assertIn("strict_best_parlay, parlay_search_diagnostics = best_automatic_parlay(", APP)
        self.assertIn('"parlay_search": parlay_search_diagnostics', APP)
        self.assertIn("grouped_log_odds_dynamic_programming", OPTIMIZER)

    def test_two_bookmaker_paired_consensus_is_required(self):
        self.assertIn("MIN_AUTOMATIC_BOOKMAKER_CONSENSUS = 2", APP)
        self.assertIn('reject("fewer_than_2_bookmakers")', APP)
        self.assertIn("MAX_MARKET_DISPERSION = 0.04", APP)

    def test_uncertainty_haircut_is_dynamic_but_capped(self):
        self.assertIn("PROBABILITY_UNCERTAINTY_HAIRCUT = 0.02", APP)
        self.assertIn("MAX_DYNAMIC_UNCERTAINTY_ADJUSTMENT = 0.005", APP)
        self.assertIn("effective_haircut, haircut_adjustments = _uncertainty_haircut(", APP)
        self.assertIn('"probability_uncertainty_adjustments_pp"', APP)

    def test_joint_parlay_estimate_has_a_thirty_percent_floor(self):
        self.assertIn("MIN_PARLAY_JOINT_PROBABILITY = 0.30", APP)
        self.assertIn("if hit_probability < MIN_PARLAY_JOINT_PROBABILITY:", OPTIMIZER)
        self.assertIn('"minimum_joint_probability": MIN_PARLAY_JOINT_PROBABILITY', OPTIMIZER)

    def test_consensus_breakdown_is_reported(self):
        self.assertIn("bookmaker_groups_1_house", APP)
        self.assertIn("bookmaker_groups_2_houses", APP)
        self.assertIn("bookmaker_groups_3plus_houses", APP)
        self.assertIn("fewer_than_2_bookmakers", HTML)

    def test_app_attempts_history_for_missing_leagues_even_when_some_games_match(self):
        self.assertIn("const hasMissingFixtureHistory=fixtureLabelsForTopUp.some", HTML)
        self.assertIn("if(data.provider==='OddsPapi'&&(hasMissingFixtureHistory||hasTeamHistoryDeficit))", HTML)
        self.assertNotIn("data.matches_matching_imported_competitions||0)===0", HTML)

    def test_ncaam_and_g_league_fixture_aliases_are_normalized_consistently(self):
        self.assertIn(') else "men"', APP)
        self.assertIn('return "nbagleague"', APP)
        self.assertIn(') else "men"', ODDSPAPI)
        self.assertIn('return "nbagleague"', ODDSPAPI)
        self.assertIn("const ncaaGender=(label.includes('women')", HTML)
        self.assertIn("if(label.startsWith('nbagleague'))return 'nbagleague'", HTML)

    def test_service_worker_refreshes_navigation_and_invalidates_prior_cache(self):
        self.assertIn("basketball-edge-v38-team-history-topup", SERVICE_WORKER)
        self.assertIn("event.request.mode==='navigate'", SERVICE_WORKER)


if __name__ == "__main__":
    unittest.main()
