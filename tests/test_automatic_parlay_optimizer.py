import unittest

from parlay_optimizer import MIN_PARLAY_JOINT_PROBABILITY, best_automatic_parlay


def _pick(event_id, market, probability, odds, *, strict=True):
    return {
        "event_id": event_id,
        "game": event_id + " game",
        "market": market,
        "side": "home",
        "line": None,
        "selection": market + " " + event_id,
        "bookmaker": "Book A",
        "odds": odds,
        "probability": probability,
        "passes_strict_filters": strict,
        "edge_percentage_points": 4.0,
        "score": 1.0,
    }


class AutomaticParlayOptimizerTests(unittest.TestCase):
    def test_optimizer_keeps_higher_price_alternatives_per_event(self):
        picks = [
            _pick("event-1", "moneyline", 0.90, 1.15),
            _pick("event-1", "spread", 0.82, 2.00),
            _pick("event-2", "moneyline", 0.90, 1.15),
            _pick("event-2", "spread", 0.82, 2.00),
        ]

        result, search = best_automatic_parlay(picks, 4.00)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["combined_odds"], 4.00, places=6)
        self.assertAlmostEqual(result["estimated_hit_probability"], 0.82 ** 2, places=6)
        self.assertEqual(len(result["legs"]), 2)
        self.assertEqual(len({leg["event_id"] for leg in result["legs"]}), 2)
        self.assertTrue(all(leg["market"] == "spread" for leg in result["legs"]))
        self.assertEqual(result["bookmaker"], "Book A")
        self.assertGreaterEqual(result["estimated_hit_probability"], MIN_PARLAY_JOINT_PROBABILITY)
        self.assertGreaterEqual(search["alternative_selections_considered"], 4)

    def test_optimizer_never_uses_candidate_that_fails_strict_filters(self):
        picks = [_pick("event-1", "moneyline", 0.99, 4.00, strict=False)]
        result, search = best_automatic_parlay(picks, 4.00)
        self.assertIsNone(result)
        self.assertEqual(search["strict_candidate_selections"], 0)

    def test_optimizer_never_combines_two_markets_from_the_same_event(self):
        picks = [
            _pick("event-1", "moneyline", 0.95, 2.00),
            _pick("event-1", "spread", 0.94, 2.00),
            _pick("event-2", "moneyline", 0.80, 2.00),
        ]
        result, _ = best_automatic_parlay(picks, 4.00)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["legs"]), 2)
        self.assertEqual(len({leg["event_id"] for leg in result["legs"]}), 2)
        self.assertEqual({leg["event_id"] for leg in result["legs"]}, {"event-1", "event-2"})


if __name__ == "__main__":
    unittest.main()
