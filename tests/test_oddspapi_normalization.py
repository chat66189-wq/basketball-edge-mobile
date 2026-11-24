from datetime import datetime, timedelta, timezone
import io
import json
import unittest
from unittest.mock import patch

from oddspapi_provider import (
    _competition_key,
    _is_full_game_period,
    _market_family,
    _market_metadata,
    _normalise_odds_fixture,
    _parse_label,
    _read_basketball_market_array,
)


class OddsPapiBasketballNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 23, 4, 0, tzinfo=timezone.utc)
        self.captured = (self.now - timedelta(minutes=5)).isoformat()
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
                "outcomes": {"111": "Home", "112": "Away"},
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
                        "111": {"players": {"0": self._price(1.80 + suffix / 100, "opaque-1")}},
                        "112": {"players": {"0": self._price(2.00 + suffix / 100, "opaque-2")}},
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

    @staticmethod
    def _catalogue_row(market_id, name, handicap, outcomes, *, period="result",
                       player_prop=False, market_length=None):
        return {
            "sportId": 11,
            "marketId": market_id,
            "marketName": name,
            "marketLength": market_length if market_length is not None else len(outcomes),
            "playerProp": player_prop,
            "handicap": handicap,
            "period": period,
            "marketType": "match",
            "outcomes": [
                {"outcomeId": outcome_id, "outcomeName": label}
                for outcome_id, label in outcomes
            ],
        }

    def test_market_names_match_exact_documented_basketball_families(self):
        self.assertEqual(_market_family("Winner (incl. overtime)"), "moneyline")
        self.assertEqual(_market_family("Over Under (incl. overtime)"), "total")
        self.assertEqual(_market_family("Handicap (incl. overtime)"), "spread")
        self.assertIsNone(_market_family("Over Under First Quarter"))
        self.assertIsNone(_market_family("Regular Time Result"))
        self.assertIsNone(_market_family("Winner"))

    def test_only_result_period_is_full_game_for_includes_overtime_markets(self):
        self.assertTrue(_is_full_game_period("result"))
        self.assertFalse(_is_full_game_period("fulltime"))
        self.assertFalse(_is_full_game_period("p1"))
        self.assertFalse(_is_full_game_period(""))
        self.assertFalse(_is_full_game_period("first half"))

    def test_catalogue_stream_parser_preserves_line_and_outcome_metadata(self):
        rows = [
            self._catalogue_row(111, "Winner (incl. overtime)", 0,
                                [(111, "Home"), (112, "Away")]),
            self._catalogue_row(222, "Over Under (incl. overtime)", 220.5,
                                [(222, "Over"), (223, "Under")]),
            self._catalogue_row(333, "Handicap (incl. overtime)", -4.5,
                                [(333, "Home"), (334, "Away")]),
            self._catalogue_row(444, "Over Under (incl. overtime)", 45.5,
                                [(444, "Over"), (445, "Under")], period="p1"),
            self._catalogue_row(555, "Over Under (incl. overtime)", 12.5,
                                [(555, "Over"), (556, "Under")], player_prop=True),
            {"sportId": 10, "marketId": 666, "marketName": "Over Under (incl. overtime)"},
            self._catalogue_row(777, "Winner (incl. overtime)", 0,
                                [(777, "Home"), (778, "Draw"), (779, "Away")]),
        ]
        selected = _read_basketball_market_array(io.BytesIO(json.dumps(rows).encode("utf-8")))
        self.assertEqual([row["marketId"] for row in selected], [111, 222, 333])
        total = selected[1]
        self.assertEqual(total["handicap"], 220.5)
        self.assertEqual(total["marketLength"], 2)
        self.assertEqual(total["period"], "result")
        self.assertEqual(total["outcomes"][0]["outcomeName"], "Over")

    def test_market_metadata_resolves_catalogue_outcome_ids_and_lines(self):
        rows = [
            self._catalogue_row(111, "Winner (incl. overtime)", 0,
                                [(111, "Home"), (112, "Away")]),
            self._catalogue_row(222, "Over Under (incl. overtime)", 220.5,
                                [(222, "Over"), (223, "Under")]),
            self._catalogue_row(333, "Handicap (incl. overtime)", -4.5,
                                [(333, "Home"), (334, "Away")]),
            self._catalogue_row(444, "Over Under (incl. overtime)", None,
                                [(444, "Over"), (445, "Under")], period="result"),
        ]
        with patch("oddspapi_provider._get", return_value=rows):
            metadata = _market_metadata()
        self.assertEqual(set(metadata), {"111", "222", "333"})
        self.assertEqual(metadata["222"]["handicap"], 220.5)
        self.assertEqual(metadata["222"]["outcomes"], {"222": "Over", "223": "Under"})
        self.assertEqual(metadata["333"]["handicap"], -4.5)

    def test_provider_competition_aliases_match_local_history_labels(self):
        self.assertEqual(_competition_key("NCAA, Regular Season"), "ncaadivision1men")
        self.assertEqual(_competition_key("NCAA Women, Regular Season"), "ncaadivision1women")
        self.assertEqual(_competition_key("NBA G League Tip-Off Tournament"), "nbagleague")

    def test_bookmaker_labels_decode_spreads_and_totals(self):
        self.assertEqual(_parse_label("-4.5/home"), ("home", -4.5))
        self.assertEqual(_parse_label("+4.5/away"), ("away", 4.5))
        self.assertEqual(_parse_label("220.5/over"), ("over", 220.5))
        self.assertEqual(_parse_label("220.5/under"), ("under", 220.5))

    def test_catalogue_outcome_ids_pair_moneyline_spread_and_total_without_position_fallback(self):
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

        audit = normalized["odds_audit"]
        self.assertEqual(audit["unparsed_outcomes"], 0)
        self.assertEqual(audit["outcomes_mapped_by_catalogue"], 18)
        self.assertEqual(audit["outcomes_mapped_by_position"], 0)
        self.assertEqual(audit["paired_quotes"], 9)
        self.assertEqual(audit["normalized_market_groups"], 9)
        self.assertEqual(audit["fresh_prices"], 18)
        self.assertEqual(audit["source_changes_within_1h"], 18)
        self.assertEqual(audit["missing_market_line"], 0)
        self.assertEqual(audit["line_label_conflicts"], 0)

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

    def test_outcome_mapping_is_independent_of_provider_dictionary_order(self):
        bookmaker = self._bookmaker(0)
        outcomes = bookmaker["markets"]["22530"]["outcomes"]
        bookmaker["markets"]["22530"]["outcomes"] = {
            "22531": outcomes["22531"],
            "22530": outcomes["22530"],
        }
        normalized = _normalise_odds_fixture(
            self.fixture,
            {"bookmakerOdds": {"book-one": bookmaker}},
            self.market_catalogue,
            self.now,
        )
        market = normalized["periods"]["ft"]["book-one"]["total"][0]
        self.assertEqual(market["over"], 1.91)
        self.assertEqual(market["under"], 1.91)

    def test_unknown_outcome_id_is_not_guessed_from_array_position(self):
        bookmaker = self._bookmaker(0)
        del bookmaker["markets"]["111"]["outcomes"]["112"]
        normalized = _normalise_odds_fixture(
            self.fixture,
            {"bookmakerOdds": {"book-one": bookmaker}},
            self.market_catalogue,
            self.now,
        )
        self.assertNotIn("moneyline", normalized["periods"]["ft"].get("book-one", {}))
        self.assertGreaterEqual(normalized["odds_audit"]["unparsed_outcomes"], 0)
        self.assertEqual(normalized["odds_audit"]["outcomes_mapped_by_position"], 0)

    def test_bookmaker_line_conflict_does_not_pair_different_totals(self):
        bookmaker = self._bookmaker(0)
        bookmaker["markets"]["22530"]["outcomes"]["22531"]["players"]["0"]["bookmakerOutcomeId"] = "221.5/under"
        normalized = _normalise_odds_fixture(
            self.fixture,
            {"bookmakerOdds": {"book-one": bookmaker}},
            self.market_catalogue,
            self.now,
        )
        self.assertNotIn("total", normalized["periods"]["ft"].get("book-one", {}))
        self.assertEqual(normalized["odds_audit"]["line_label_conflicts"], 1)

    def test_catalogue_and_bookmaker_outcome_side_conflict_is_rejected(self):
        bookmaker = self._bookmaker(0)
        bookmaker["markets"]["111"]["outcomes"]["111"]["players"]["0"]["bookmakerOutcomeId"] = "away"
        normalized = _normalise_odds_fixture(
            self.fixture,
            {"bookmakerOdds": {"book-one": bookmaker}},
            self.market_catalogue,
            self.now,
        )
        self.assertNotIn("moneyline", normalized["periods"]["ft"].get("book-one", {}))
        self.assertEqual(normalized["odds_audit"]["outcome_label_conflicts"], 1)


if __name__ == "__main__":
    unittest.main()
