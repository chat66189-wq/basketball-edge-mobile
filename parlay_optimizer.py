"""Pure, deterministic optimizer for same-bookmaker automatic basketball parlays."""
from __future__ import annotations

import math

MIN_ESTIMATED_LEG_PROBABILITY = 0.70
MIN_PARLAY_JOINT_PROBABILITY = 0.30
PARLAY_ODDS_TOLERANCE_FRACTION = 0.05


def _parlay_odds_band(target_odds: float) -> tuple[float, float]:
    return (
        target_odds * (1.0 - PARLAY_ODDS_TOLERANCE_FRACTION),
        target_odds * (1.0 + PARLAY_ODDS_TOLERANCE_FRACTION),
    )


def _parlay_rank(hit_probability: float, combined_odds: float,
                 target_odds: float, leg_count: int) -> tuple[float, float, int]:
    relative_distance = abs(combined_odds - target_odds) / target_odds
    return (hit_probability, -relative_distance, -leg_count)


def best_automatic_parlay(source: list[dict], target_odds: float) -> tuple[dict | None, dict]:
    """Search strict picks while preserving multiple market alternatives per event."""
    target_min, target_max = _parlay_odds_band(target_odds)
    target_log = math.log(target_odds)
    minimum_log = math.log(target_min)
    maximum_log = math.log(target_max)
    by_bookmaker: dict[str, dict[str, dict[tuple, dict]]] = {}

    def candidate_rank(item: dict) -> tuple[float, float, float, float]:
        return (
            float(item.get("probability") or 0.0),
            float(item.get("edge_percentage_points") or 0.0),
            float(item.get("score") or 0.0),
            float(item.get("odds") or 0.0),
        )

    for item in source:
        if not isinstance(item, dict) or not item.get("passes_strict_filters", False):
            continue
        bookmaker = str(item.get("bookmaker") or "").strip()
        event_id = str(item.get("event_id") or "").strip()
        if not bookmaker or not event_id:
            continue
        try:
            odds = float(item.get("odds"))
            probability = float(item.get("probability"))
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(odds) or not math.isfinite(probability)
            or odds <= 1.0 or odds > target_max or not 0.0 < probability <= 1.0
            or probability < MIN_ESTIMATED_LEG_PROBABILITY
        ):
            continue
        selection_key = (
            str(item.get("market") or ""),
            str(item.get("side") or ""),
            item.get("line"),
        )
        event_selections = by_bookmaker.setdefault(bookmaker, {}).setdefault(event_id, {})
        previous = event_selections.get(selection_key)
        if previous is None or candidate_rank(item) > candidate_rank(previous):
            event_selections[selection_key] = item

    best = None
    search = {
        "strict_candidate_selections": sum(
            len(selections)
            for events in by_bookmaker.values()
            for selections in events.values()
        ),
        "bookmakers_with_candidates": len(by_bookmaker),
        "event_groups_available": sum(len(events) for events in by_bookmaker.values()),
        "event_groups_considered": 0,
        "alternative_selections_considered": 0,
        "maximum_event_groups_per_bookmaker": 40,
        "maximum_alternatives_per_event": 24,
        "log_odds_bucket_step": 0.05,
        "dynamic_programming_log_odds_resolution": 0.0005,
    }
    step = 0.0005
    for bookmaker, events in by_bookmaker.items():
        event_groups = []
        for event_id, selections in events.items():
            by_price_band = {}
            for item in selections.values():
                odds = float(item["odds"])
                price_band = int(round(math.log(odds) / 0.05))
                previous = by_price_band.get(price_band)
                if previous is None or candidate_rank(item) > candidate_rank(previous):
                    by_price_band[price_band] = item
            options = sorted(by_price_band.values(), key=candidate_rank, reverse=True)[:24]
            if options:
                event_groups.append((event_id, options))
        event_groups.sort(
            key=lambda row: max(candidate_rank(item) for item in row[1]),
            reverse=True,
        )
        event_groups = event_groups[:40]
        search["event_groups_considered"] += len(event_groups)
        search["alternative_selections_considered"] += sum(len(options) for _, options in event_groups)
        if not event_groups:
            continue

        # State: log probability, exact log odds, and selected legs.
        dp: list[dict[int, tuple[float, float, tuple[dict, ...]]]] = [{} for _ in range(9)]
        dp[0][0] = (0.0, 0.0, ())
        for _, options in event_groups:
            previous_states = [tuple(dp[count].items()) for count in range(8)]
            option_logs = [
                (item, math.log(float(item["odds"])), math.log(float(item["probability"])))
                for item in options
            ]
            for count, states in enumerate(previous_states):
                if not states:
                    continue
                for _, state in states:
                    log_probability, log_odds, legs = state
                    for item, item_log_odds, item_log_probability in option_logs:
                        combined_log_odds = log_odds + item_log_odds
                        if combined_log_odds > maximum_log + step:
                            continue
                        bucket = int(round(combined_log_odds / step))
                        next_state = (
                            log_probability + item_log_probability,
                            combined_log_odds,
                            legs + (item,),
                        )
                        old = dp[count + 1].get(bucket)
                        if (
                            old is None
                            or next_state[0] > old[0] + 1e-12
                            or (
                                abs(next_state[0] - old[0]) <= 1e-12
                                and abs(next_state[1] - target_log) < abs(old[1] - target_log)
                            )
                        ):
                            dp[count + 1][bucket] = next_state

        for count in range(1, 9):
            for log_probability, log_odds, legs in dp[count].values():
                if not minimum_log <= log_odds <= maximum_log:
                    continue
                hit_probability = math.exp(log_probability)
                if hit_probability < MIN_PARLAY_JOINT_PROBABILITY:
                    continue
                combined_odds = math.exp(log_odds)
                rank = _parlay_rank(hit_probability, combined_odds, target_odds, count)
                if best is None or rank > best["rank"]:
                    best = {
                        "rank": rank,
                        "bookmaker": bookmaker,
                        "combined_odds": combined_odds,
                        "target_odds": target_odds,
                        "target_odds_min": target_min,
                        "target_odds_max": target_max,
                        "odds_deviation": combined_odds - target_odds,
                        "odds_deviation_percent": ((combined_odds / target_odds) - 1.0) * 100.0,
                        "estimated_hit_probability": hit_probability,
                        "minimum_leg_probability": MIN_ESTIMATED_LEG_PROBABILITY,
                        "minimum_joint_probability": MIN_PARLAY_JOINT_PROBABILITY,
                        "probability_scope": "each_leg_minimum_and_separate_joint_floor",
                        "joint_probability_method": "product_of_leg_estimates_assuming_approximate_independence",
                        "optimization_method": "grouped_log_odds_dynamic_programming",
                        "legs": list(legs),
                    }
    return best, search


