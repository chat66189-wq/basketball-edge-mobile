from datetime import datetime, timedelta, timezone
import unittest

from oddspapi_provider import (
    _market_family,
    _normalise_odds_fixture,
    _parse_label,
)


class OddsPapiBasketballNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 23, 4, 0, tzinfo=timezone.utc)
        self.captured = (self.now - timedelta(minutes=25)).isoformat()
        self.fixture = {
            "fixtureId": "test-fixture",
            "participant1Name": "Home Club",
            "participant2Name": "Away Club",
            "tournamentName": "EuroLeague",
        }
        self.market_catalogue = {
            "111": {
                "family": "moneyline",
                "handicap": 0,
                # Deliberately opaque labels: the bookmaker's outcome IDs are
                # numeric and must fall back to OddsPapi's documented ordering.
                "outcomes": {"111": "selection alpha", "112": "selection beta"},
            },
            "22530": {
                "family": "total",
                "handicap": 220.5,
                "outcomes": {"22530": "Over", "22531": "Under"},
            },
            "22531": {
                "family": "spread",
                "handicap": -4.5,
                "outcomes": {"22532": "Home", "22533": "Away"},
            },
        }

    def _price(self, price, bookmaker_outcome_id):
        return {
            "active": True,
            "price": price,
            "bookmakerOutcomeId": bookmaker_outcome_id,
            "changedAt": self.captured,
        }

    def _bookmaker(self, suffix):
        return {
            "bookmakerIsActive": True,
            "suspended": False,
            "markets": {
                "111": {
                    "marketActive": True,
                    "outcomes": {
                        "111": {"players": {"0": self._price(1.80 + suffix / 100, "981723641")}},
                        "112": {"players": {"0": self._price(2.00 + suffix / 100, "981723642")}},
                    },
                },
                "22530": {
                    "marketActive": True,
                    "outcomes": {
                        "22530": {"players": {"0": self._price(1.91, "220.5/over")}},
                        "22531": {"players": {"0": self._price(1.91, "220.5/under")}},
                    },
                },
                "22531": {
                    "marketActive": True,
                    "outcomes": {
                        "22532": {"players": {"0": self._price(1.91, "-4.5/home")}},
                        "22533": {"players": {"0": self._price(1.91, "+4.5/away")}},
                    },
                },
            },
        }

    def test_market_names_match_documented_basketball_families(self):
        self.assertEqual(_market_family("Winner (incl. overtime)"), "moneyline")
        self.assertEqual(_market_family("Over Under (incl. overtime)"), "total")
        self.assertEqual(_market_family("Handicap (incl. overtime)"), "spread")
        self.assertIsNone(_market_family("Over Under First Quarter"))

    def test_bookmaker_labels_decode_spreads_and_totals(self):
        self.assertEqual(_parse_label("-4.5/home"), ("home", -4.5))
        self.assertEqual(_parse_label("+4.5/away"), ("away", 4.5))
        self.assertEqual(_parse_label("220.5/over"), ("over", 220.5))
        self.assertEqual(_parse_label("220.5/under"), ("under", 220.5))

    def test_opaque_moneyline_ids_use_documented_home_away_order_and_pair_all_markets(self):
        odds = {
            "updatedAt": self.captured,
            "bookmakerOdds": {
                "book-one": self._bookmaker(0),
                "book-two": self._bookmaker(1),
                "book-three": self._bookmaker(2),
            },
        }
        normalized = _normalise_odds_fixture(
            self.fixture, odds, self.market_catalogue, self.now
        )

        self.assertEqual(normalized["odds_audit"]["unparsed_outcomes"], 0)
        self.assertEqual(normalized["odds_audit"]["outcomes_mapped_by_position"], 6)
        self.assertEqual(normalized["odds_audit"]["paired_quotes"], 9)
        self.assertEqual(normalized["odds_audit"]["normalized_market_groups"], 9)
        self.assertEqual(normalized["odds_audit"]["fresh_prices"], 18)
        self.assertEqual(normalized["odds_audit"]["source_changes_within_1h"], 18)
        self.assertEqual(normalized["odds_audit"]["missing_market_line"], 0)

        ft = normalized["periods"]["ft"]
        self.assertEqual(len(ft), 3)
        for bookmaker, markets in ft.items():
            self.assertEqual(len(markets["moneyline"]), 1)
            self.assertEqual(len(markets["total"]), 1)
            self.assertEqual(len(markets["spread"]), 1)
            self.assertGreater(markets["moneyline"][0]["home"], 1.0)
            self.assertGreater(markets["moneyline"][0]["away"], 1.0)
            self.assertEqual(markets["moneyline"][0]["captured_at"], self.now.isoformat())
            self.assertEqual(markets["total"][0]["line"], 220.5)
            self.assertEqual(markets["spread"][0]["line"], -4.5)


if __name__ == "__main__":
    unittest.main()
