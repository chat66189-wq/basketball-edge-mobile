from __future__ import annotations

import math
import os
import random
import re
import unicodedata
from difflib import SequenceMatcher
from itertools import combinations
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from parlay_optimizer import best_automatic_parlay

BASE_DIR = Path(__file__).resolve().parent
ENGINE_BUNDLE = BASE_DIR / "engine_bundle.zip"
ENGINE_DIR = Path(os.environ.get("ENGINE_DIR", "/tmp/bea_v20_engine"))


def load_engine() -> None:
    """Extract the included, unmodified v15-v20 Python modules once per instance."""
    marker = ENGINE_DIR / "scripts" / "edge_v15" / "team_model.py"
    if not marker.exists():
        if not ENGINE_BUNDLE.exists():
            raise RuntimeError("engine_bundle.zip is missing beside app.py")
        ENGINE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ENGINE_BUNDLE) as archive:
            archive.extractall(ENGINE_DIR)
    if str(ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_DIR))


load_engine()
from scripts.edge_v15.team_model import (  # noqa: E402
    TeamGame, LeaguePriors, GameSimulation, fit_team_rates, fit_opponent_adjusted_ratings,
    project_game, simulate_game,
)
from scripts.edge_v15.market_pricing import price_game_market, ev_per_unit  # noqa: E402

app = FastAPI(
    title="Basketball Edge Analyst v20 API",
    description="Adaptador web para el motor de proyección y simulación BEA. Las salidas son exploratorias, no picks validados.",
    version="20.5.0-batched-history-correct-fixture-scope",
)

frontend_origin = os.environ.get("FRONTEND_ORIGIN", "https://chat66189-wq.github.io")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://localhost", "http://127.0.0.1"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

TERMINAL_STATUSES = {"final", "finished", "complete", "completed", "ended", "ft", "post", "closed"}
SUPPORTED_MARKETS = {"moneyline", "spread", "total", "home_total", "away_total"}

# Conservative candidate policy: screening thresholds, not a claim of calibration.
MIN_ESTIMATED_LEG_PROBABILITY = 0.70
MIN_RAW_MODEL_PROBABILITY = 0.72
MIN_MARKET_PROBABILITY = 0.60
MAX_MODEL_MARKET_GAP = 0.12
MIN_AUTOMATIC_BOOKMAKER_CONSENSUS = 2
MAX_MARKET_DISPERSION = 0.04
MAX_QUOTE_AGE_SECONDS = 15 * 60
MIN_HISTORICAL_GAMES = 12
PROBABILITY_UNCERTAINTY_HAIRCUT = 0.02
MAX_DYNAMIC_UNCERTAINTY_ADJUSTMENT = 0.005
MIN_PARLAY_JOINT_PROBABILITY = 0.30
PARLAY_ODDS_TOLERANCE_FRACTION = 0.05


def _uncertainty_haircut(
    history_games: int,
    bookmaker_count: int,
    market_dispersion_pp: float,
    quote_age_seconds: float,
) -> tuple[float, dict[str, float]]:
    """Transparent small risk adjustment on top of the 2pp base haircut."""
    history_penalty = min(
        0.0025,
        max(0.0, (24.0 - float(history_games)) / 12.0) * 0.0025,
    )
    consensus_penalty = 0.001 if bookmaker_count == 2 else 0.0
    dispersion_penalty = 0.001 * min(1.0, max(0.0, float(market_dispersion_pp)) / 4.0)
    freshness_penalty = 0.0005 * min(
        1.0, max(0.0, float(quote_age_seconds)) / MAX_QUOTE_AGE_SECONDS
    )
    adjustments = {
        "limited_history": history_penalty,
        "two_bookmaker_consensus": consensus_penalty,
        "market_dispersion": dispersion_penalty,
        "quote_age": freshness_penalty,
    }
    total_adjustment = min(MAX_DYNAMIC_UNCERTAINTY_ADJUSTMENT, sum(adjustments.values()))
    return PROBABILITY_UNCERTAINTY_HAIRCUT + total_adjustment, adjustments

# Best-effort is a visibly weaker fallback, never a substitute for strict screening.
BEST_EFFORT_MIN_RAW_MODEL_PROBABILITY = 0.60
BEST_EFFORT_MIN_MARKET_PROBABILITY = 0.55
BEST_EFFORT_MAX_MODEL_MARKET_GAP = 0.15
BEST_EFFORT_MIN_ESTIMATED_LEG_PROBABILITY = 0.58
BEST_EFFORT_MIN_EXPECTED_VALUE = -0.08


def _parlay_odds_band(target_odds: float) -> tuple[float, float]:
    """Allow a narrow band around the requested combined price, not a minimum-only floor."""
    return (
        target_odds * (1.0 - PARLAY_ODDS_TOLERANCE_FRACTION),
        target_odds * (1.0 + PARLAY_ODDS_TOLERANCE_FRACTION),
    )


def _parlay_rank(hit_probability: float, combined_odds: float,
                 target_odds: float, leg_count: int) -> tuple[float, float, int]:
    """Prefer the highest joint estimate, then the closest price to target."""
    relative_distance = abs(combined_odds - target_odds) / target_odds
    return (hit_probability, -relative_distance, -leg_count)



def _simulate_game_for_settlement(projection, *, simulations: int, seed: int,
                                  overtime_rule: str) -> GameSimulation:
    """Simulate regulation, and add repeated five-minute OTs only for tied games."""
    if overtime_rule not in {"regulation_only", "includes_overtime"}:
        raise ValueError("settlement_rule_must_be_confirmed")
    base = simulate_game(projection, simulations=simulations, seed=seed)
    if overtime_rule == "regulation_only":
        return base

    rng = random.Random(seed ^ 0x5A17C0DE)
    home_scores = list(base.home_scores)
    away_scores = list(base.away_scores)
    ot_pace_mean = projection.pace * (5.0 / 48.0)
    ot_pace_sd = 3.0 * (5.0 / 48.0)
    corr = 0.10
    orth = math.sqrt(1.0 - corr * corr)
    for index in range(simulations):
        periods = 0
        while home_scores[index] == away_scores[index]:
            periods += 1
            if periods > 20:
                raise RuntimeError("overtime_simulation_exceeded_20_periods")
            pace = max(0.1, rng.gauss(ot_pace_mean, ot_pace_sd))
            z1, z2 = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
            home_ppp = max(0.3, projection.home_ppp + 0.075 * z1)
            away_ppp = max(0.3, projection.away_ppp + 0.075 * (corr * z1 + orth * z2))
            home_scores[index] += max(0, int(math.floor(pace * home_ppp + 0.5)))
            away_scores[index] += max(0, int(math.floor(pace * away_ppp + 0.5)))
    return GameSimulation(projection, home_scores, away_scores, simulations, seed, False)


def _aware_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}_required")
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name}_invalid_iso8601") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name}_timezone_required")
    return result.astimezone(timezone.utc)


def _number(value: Any, name: str, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}_must_be_numeric")
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name}_must_be_finite_and_at_least_{minimum}")
    return result


def _historical_datetime(value: Any, name: str) -> datetime:
    """Parse provider event timestamps; naive provider dates are interpreted as UTC."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name}_invalid_timestamp")
    if isinstance(value, (int, float)):
        stamp = float(value)
        if not math.isfinite(stamp):
            raise ValueError(f"{name}_invalid_timestamp")
        # Providers may send Unix seconds or milliseconds.
        if abs(stamp) >= 1e12:
            stamp /= 1000.0
        try:
            return datetime.fromtimestamp(stamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"{name}_invalid_timestamp") from exc

    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{name}_required")
    # Some feeds serialize Unix timestamps as strings.
    try:
        numeric_stamp = float(raw)
    except ValueError:
        numeric_stamp = None
    if numeric_stamp is not None:
        return _historical_datetime(numeric_stamp, name)

    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name}_invalid_iso8601") from exc
    # SportAPI event times can be ISO strings without an offset. Treat those as UTC
    # rather than silently dropping every such historical row.
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _build_history(payload: dict[str, Any], as_of: datetime) -> tuple[list[TeamGame], dict[str, int]]:
    games = payload.get("games")
    stats = payload.get("team_stats")
    if not isinstance(games, list) or not isinstance(stats, list):
        raise ValueError("games_and_team_stats_must_be_arrays")
    if len(games) > 20000 or len(stats) > 40000:
        raise ValueError("too_many_rows_for_one_analysis")

    competition = str(payload.get("competition", "")).strip()
    season = str(payload.get("season", "")).strip()
    if not competition or not season:
        raise ValueError("competition_and_season_are_required_exact_scope_fields")

    stats_by_event: dict[str, dict[str, dict[str, Any]]] = {}
    for row in stats:
        if not isinstance(row, dict):
            continue
        event_id = str(row.get("event_id", "")).strip()
        team = str(row.get("team", "")).strip()
        if not event_id or not team:
            continue
        stats_by_event.setdefault(event_id, {})[team] = row

    history: list[TeamGame] = []
    counters = {"games_received": len(games), "games_in_scope": 0,
                "games_used": 0, "games_skipped": 0}
    for game in games:
        if not isinstance(game, dict):
            counters["games_skipped"] += 1
            continue
        if str(game.get("competition", "")).strip() != competition or str(game.get("season", "")).strip() != season:
            continue
        counters["games_in_scope"] += 1
        if str(game.get("status", "")).strip().lower() not in TERMINAL_STATUSES:
            counters["games_skipped"] += 1
            continue
        try:
            played = _historical_datetime(game.get("event_time"), "event_time")
            # The v20 games CSV template has no official result-publication timestamp.
            # Use a conservative +24h availability proxy and disclose it in the response.
            available = played + timedelta(hours=24)
            if available >= as_of:
                counters["games_skipped"] += 1
                continue
            event_id = str(game.get("event_id", "")).strip()
            home = str(game.get("home_team", "")).strip()
            away = str(game.get("away_team", "")).strip()
            if not event_id or not home or not away or home == away:
                counters["games_skipped"] += 1
                continue
            per_team = stats_by_event.get(event_id, {})
            home_row, away_row = per_team.get(home), per_team.get(away)
            if not home_row or not away_row:
                counters["games_skipped"] += 1
                continue
            hp = _number(home_row.get("points"), "home_points")
            ap = _number(away_row.get("points"), "away_points")
            hpos = _number(home_row.get("possessions"), "home_possessions", 1.0)
            apos = _number(away_row.get("possessions"), "away_possessions", 1.0)
            day = played.isoformat()
            available_at = available.isoformat()
            history.append(TeamGame(day, home, away, hp, ap, hpos, True, available_at))
            history.append(TeamGame(day, away, home, ap, hp, apos, False, available_at))
            counters["games_used"] += 1
        except (ValueError, TypeError):
            counters["games_skipped"] += 1
    return history, counters


@app.get("/")
def root():
    return {"service": "Basketball Edge Analyst v20 API", "health": "/api/health", "capabilities": "/api/capabilities"}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "basketball-edge-analyst-v20", "version": app.version}


@app.get("/api/capabilities")
def capabilities():
    return {
        "status": "ready",
        "engine_version": "v20 source package with v15 projection and simulation modules",
        "supported_markets": sorted(SUPPORTED_MARKETS),
        "stateless_requests": True,
        "persistent_server_storage": False,
        "validation_status": "EXPLORATORY_NOT_VALIDATED",
        "note": "No authenticated GanamosVIP/Azar247 odds feed is configured. The API accepts manually captured bookmaker quote pairs for supported full-game markets; all output remains exploratory.",
    }


@app.post("/api/analyze-game")
async def analyze_game(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request_body_must_be_object")
        as_of = _aware_datetime(payload.get("as_of"), "as_of")
        home = str(payload.get("home_team", "")).strip()
        away = str(payload.get("away_team", "")).strip()
        if not home or not away or home == away:
            raise ValueError("home_and_away_teams_must_be_distinct")

        simulations = int(payload.get("simulations", 25000))
        if not 1000 <= simulations <= 100000:
            raise ValueError("simulations_must_be_between_1000_and_100000")

        history, counters = _build_history(payload, as_of)
        if not history:
            raise ValueError("no_eligible_completed_games_with_both_team_box_scores_in_exact_competition_season_scope")

        available_history = [r for r in history if _aware_datetime(r.available_at, "available_at") < as_of]
        if not available_history:
            raise ValueError("no_point_in_time_history_available_before_as_of")
        pace = sum(r.possessions for r in available_history) / len(available_history)
        total_possessions = sum(r.possessions for r in available_history)
        ppp = sum(r.points_for for r in available_history) / total_possessions
        priors = LeaguePriors(pace=pace, points_per_possession=ppp)

        home_rates = fit_team_rates(history, home, as_of=as_of, priors=priors)
        away_rates = fit_team_rates(history, away, as_of=as_of, priors=priors)
        if home_rates.games < 3 or away_rates.games < 3:
            raise ValueError(
                f"insufficient_team_history_minimum_3_games_each_required:home={home_rates.games},away={away_rates.games}"
            )
        adjusted = None
        try:
            adjusted = fit_opponent_adjusted_ratings(history, as_of=as_of,
                league_ppp=priors.points_per_possession)
        except ValueError:
            adjusted = None
        projection = project_game(home_rates, away_rates, priors, adjusted_ratings=adjusted)
        simulation = simulate_game(projection, simulations=simulations,
                                   seed=int(payload.get("seed", 20261120)))

        raw_markets = payload.get("markets", [])
        if not isinstance(raw_markets, list) or len(raw_markets) > 8:
            raise ValueError("markets_must_be_a_list_with_at_most_8_items")
        market_results = []
        for item in raw_markets:
            if not isinstance(item, dict):
                raise ValueError("each_market_must_be_an_object")
            market = str(item.get("market", "")).lower().strip()
            side = str(item.get("side", "")).lower().strip()
            if market not in SUPPORTED_MARKETS:
                raise ValueError(f"unsupported_market:{market}")
            line = None if market == "moneyline" else _number(item.get("line"), "market_line", -10000)
            result = price_game_market(simulation, market, side=side, line=line, overtime_rule="excluded")
            output = asdict(result)
            odds = item.get("decimal_odds")
            if odds not in (None, ""):
                odds_value = _number(odds, "decimal_odds", 1.0000001)
                output["decimal_odds"] = odds_value
                output["expected_net_units_per_unit_stake"] = ev_per_unit(result, odds_value)
            market_results.append(output)

        warnings = [
            "EXPLORATORY_NOT_VALIDATED: no approved model card or exact-scope walk-forward/holdout validation was found.",
            "The v15 baseline simulator is regulation-only; check the bookmaker's settlement rules before comparing probabilities.",
            "The v20 games CSV template has no result-publication timestamp; historical availability was conservatively approximated as event_time + 24 hours.",
            "The market probability is a model simulation output, not a calibrated probability, betting recommendation, or guarantee.",
        ]
        return {
            "status": "EXPLORATORY_NOT_VALIDATED",
            "decision": "NO_APPROVED_PICK_MODEL_REQUIRED",
            "scope": {"competition": payload["competition"], "season": payload["season"],
                      "home_team": home, "away_team": away, "as_of": as_of.isoformat()},
            "data_audit": counters,
            "projection": {
                "pace": projection.pace,
                "home_ppp": projection.home_ppp,
                "away_ppp": projection.away_ppp,
                "home_points_mean": projection.home_points_mean,
                "away_points_mean": projection.away_points_mean,
                "total_points_mean": projection.home_points_mean + projection.away_points_mean,
                "home_minus_away_mean": projection.home_points_mean - projection.away_points_mean,
                "home_games": home_rates.games,
                "away_games": away_rates.games,
                "home_effective_games": home_rates.effective_games,
                "away_effective_games": away_rates.effective_games,
                "rating_method": projection.rating_method,
                "model_version": projection.model_version,
            },
            "simulation": {"n": simulation.simulation_count, "seed": simulation.seed,
                           "regulation_only": True},
            "markets": market_results,
            "warnings": warnings,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid returning stack traces or local file paths to public clients.
        raise HTTPException(status_code=500, detail=f"analysis_failed:{type(exc).__name__}") from exc

@app.get("/api/provider/status")
def provider_status():
 try:
  from provider_api import status
  return status()
 except Exception as e:
  from provider_api import ProviderError
  if isinstance(e,ProviderError): raise HTTPException(status_code=e.status,detail=str(e))
  raise HTTPException(status_code=500,detail="provider_status_failed")
@app.get("/api/provider/leagues")
def provider_leagues(season: str, search: str | None = None):
 try:
  from provider_api import leagues
  return leagues(season,search)
 except Exception as e:
  from provider_api import ProviderError
  if isinstance(e,ProviderError): raise HTTPException(status_code=e.status,detail=str(e))
  raise HTTPException(status_code=500,detail="provider_leagues_failed")
@app.get("/api/provider/sync")
def provider_sync(league: int, season: str, date: str | None = None, limit: int = 40):
 try:
  from provider_api import sync
  return sync(league,season,date,limit)
 except Exception as e:
  from provider_api import ProviderError
  if isinstance(e,ProviderError): raise HTTPException(status_code=e.status,detail=str(e))
  raise HTTPException(status_code=500,detail="provider_sync_failed")


@app.get("/api/sportapi/status")
def sportapi_status():
    try:
        from sportapi_provider import status
        return status()
    except Exception as exc:
        from sportapi_provider import SportAPIError
        if isinstance(exc, SportAPIError):
            raise HTTPException(status_code=exc.status, detail=str(exc))
        raise HTTPException(status_code=500, detail="sportapi_status_failed")


@app.get("/api/sportapi/leagues")
def sportapi_leagues(search: str | None = None):
    try:
        from sportapi_provider import leagues
        return leagues(search)
    except Exception as exc:
        from sportapi_provider import SportAPIError
        if isinstance(exc, SportAPIError):
            raise HTTPException(status_code=exc.status, detail=str(exc))
        raise HTTPException(status_code=500, detail="sportapi_leagues_failed")


@app.get("/api/sportapi/sync")
def sportapi_sync(
    league: str,
    season: str | None = None,
    date: str | None = None,
    limit: int = 20,
    include_players: bool = True,
):
    try:
        from sportapi_provider import sync
        return sync(league, season, date, limit, include_players=include_players)
    except Exception as exc:
        from sportapi_provider import SportAPIError
        if isinstance(exc, SportAPIError):
            raise HTTPException(status_code=exc.status, detail=str(exc))
        raise HTTPException(status_code=500, detail="sportapi_sync_failed")


@app.get("/api/sportapi/fixtures")
def sportapi_fixtures(league: str, date: str, limit: int = 80):
    try:
        from sportapi_provider import fixtures
        return fixtures(league, date, limit)
    except Exception as exc:
        from sportapi_provider import SportAPIError
        if isinstance(exc, SportAPIError):
            raise HTTPException(status_code=exc.status, detail=str(exc))
        raise HTTPException(status_code=500, detail="sportapi_fixtures_failed")

@app.get("/api/espn/status")
def espn_status():
    from espn_provider import status
    return status()


@app.get("/api/espn/history-sync")
async def espn_history_sync(
    competitions: str,
    days: int = 400,
    max_games_per_league: int = 220,
    start_date: str | None = None,
    end_date: str | None = None,
):
    try:
        from espn_provider import sync, ESPNError
        return await run_in_threadpool(
            sync, competitions, days, max_games_per_league, start_date, end_date
        )
    except Exception as exc:
        from espn_provider import ESPNError
        if isinstance(exc, ESPNError):
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail="espn_history_sync_failed") from exc




def _normalise_label(value: Any) -> str:
    raw = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "", raw)


# Exact HoopsAPI-to-imported-history aliases observed for EuroLeague fixtures.
# Targets are checked against the imported team lookup before they are used.
_EUROLEAGUE_TEAM_ALIASES = {
    "besiktasjk": "besiktasistanbul",
    "paris": "parisbasketball",
    "crvenazvezda": "crvenazvezdameridianbetbelgrade",
    "kkcrvenazvezdabelgrade": "crvenazvezdameridianbetbelgrade",
    "crvenazvezdabelgrade": "crvenazvezdameridianbetbelgrade",
    "asvellyonvilleurbanne": "ldlcasvelvilleurbanne",
    "bayernmunich": "fcbayernmunich",
    "maccabitelaviv": "maccabirapydtelaviv",
    "valenciabc": "valenciabasket",
    "asvelvilleurbanne": "ldlcasvelvilleurbanne",
    "olimpiamilano": "ea7emporioarmanimilan",
}


def _normalise_team_label(value: Any) -> str:
    """Normalize provider team labels; gender is isolated by competition scope."""
    label = _normalise_label(value)
    for suffix in ("womens", "women", "female", "mens", "men", "male"):
        if label.endswith(suffix) and len(label) > len(suffix) + 2:
            label = label[:-len(suffix)]
            break

    # The New Orleans G League affiliate moved from Birmingham to Kenner and
    # was rebranded as Laketown Squadron for 2026-27. Treat the predecessor
    # name as the same franchise so its recent box-score history is not lost.
    if label in {"birminghamsquadron", "laketownsquadron"}:
        return "laketownsquadron"
    return label


def _team_match_tokens(value: Any) -> set[str]:
    """Return meaningful name tokens, with gender suffixes removed."""
    raw = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").casefold()
    tokens = re.findall(r"[a-z0-9]+", raw)
    while tokens and tokens[-1] in {"women", "womens", "female", "men", "mens", "male"}:
        tokens.pop()
    return set(tokens)


def _fixture_team_lookup_key(value: Any, competition: str, imported_lookup: dict[str, str]) -> str:
    key = _normalise_team_label(value)
    if key in imported_lookup:
        return key
    if competition == "euroleague":
        alias_target = _EUROLEAGUE_TEAM_ALIASES.get(key)
        if alias_target and alias_target in imported_lookup:
            return alias_target

    # Different feeds sometimes use a school's short name in one source and
    # its nickname in another (e.g. "Kansas City" vs "Kansas City Roos").
    # Accept token-containment only when the shorter name is distinctive and
    # the best candidate is clearly unique. Ambiguous city/school labels abstain.
    target_tokens = _team_match_tokens(value)
    scored: list[tuple[float, str]] = []
    for candidate_key, canonical_name in imported_lookup.items():
        candidate_tokens = _team_match_tokens(canonical_name)
        if not target_tokens or not candidate_tokens:
            continue
        shared = target_tokens & candidate_tokens
        shorter, longer = (
            (target_tokens, candidate_tokens)
            if len(target_tokens) <= len(candidate_tokens)
            else (candidate_tokens, target_tokens)
        )
        ratio = SequenceMatcher(None, key, candidate_key).ratio()
        if shorter.issubset(longer) and (
            len(shorter) >= 2 or (len(shorter) == 1 and len(next(iter(shorter))) >= 6)
        ):
            score = 0.97
        else:
            overlap = len(shared) / max(1, min(len(target_tokens), len(candidate_tokens)))
            score = 0.55 * ratio + 0.45 * overlap
        scored.append((score, candidate_key))

    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.90:
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if scored[0][0] - second_score >= 0.06:
            return scored[0][1]
    return key

def _normalise_competition(value: Any) -> str:
    """Normalize provider competition labels without merging distinct leagues or genders."""
    label = _normalise_label(value)
    if not label:
        return ""

    # WNBA and common provider display names identify the same women's league.
    if "wnba" in label or "womensnationalbasketballassociation" in label:
        return "wnba"

    # Keep the Australian NBL distinct from WNBL and other national leagues.
    if label in {"nbl", "nblmen", "nationalbasketballleague", "nationalbasketballleaguemen", "australianbl", "australiannationalbasketballleague", "nationalbasketballleagueaustralia"}:
        return "nbl"

    gender = "women" if ("women" in label or "female" in label) else ""

    if "euroleague" in label:
        return "euroleague" + gender
    if "eurocup" in label:
        return "eurocup" + gender

    # Normalize NCAA Division I women's labels used by different fixture feeds.
    # Never merge NCAA Division II/III or men's competitions into this history.
    if "ncaa" in label:
        ncaa_gender = "women" if (
            "women" in label or "female" in label
            or label.endswith(("ncaaw", "ncaadiw", "ncaad1w", "ncaadivisioniw"))
        ) else "men" if (
            "men" in label or "male" in label
            or label.endswith(("ncaam", "ncaadim", "ncaad1m", "ncaadivisionim"))
        ) else "men"
        if ncaa_gender:
            if any(token in label for token in ("divisioniii", "division3", "diviii", "div3", "ncaa3")):
                division = "3"
            elif any(token in label for token in ("divisionii", "division2", "divii", "div2", "ncaa2")):
                division = "2"
            else:
                # "NCAA Women" is the provider's short label for Division I.
                division = "1"
            return "ncaadivision" + division + ncaa_gender

    # The G League Tip-Off Tournament is part of the same NBA G League competition history.
    if label.startswith("nbagleague"):
        return "nbagleague" + ("women" if "women" in label or "female" in label else "")

    # NBA provider labels may include "men", "basketball", or country descriptors.
    if label in {"nba", "nbamen", "nationalbasketballassociation", "nationalbasketballassociationmen"}:
        return "nba"
    # Summer League and exhibition events are not NBA regular-season history.
    if label.startswith("nba") and not label.startswith(
        ("nbagleague", "nbawomen", "nbasummer", "nbacalifornia", "nbautah", "nbasaltlake", "nbapreseason")
    ) and "summerleague" not in label and "classic" not in label:
        return "nba"

    # Remove display-only suffixes, then the gender marker. Division and league
    # identifiers remain intact, so NCAA divisions are not silently merged.
    for _ in range(3):
        previous = label
        for suffix in ("boxscores", "boxscore", "basketball", "statistics", "stats"):
            if label.endswith(suffix):
                label = label[:-len(suffix)]
                break
        if label == previous:
            break
    for suffix in ("womens", "women", "female", "mens", "men", "male"):
        if label.endswith(suffix):
            label = label[:-len(suffix)]
            break
    return label + gender


def _modelable_competition_labels(
    all_games: list[dict[str, Any]],
    all_stats: list[dict[str, Any]],
    minimum_history: int,
    as_of: datetime,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Keep only leagues with at least two teams backed by complete, point-in-time box scores."""
    lookback_start = as_of - timedelta(days=400)
    stats_by_event: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_stats:
        if not isinstance(row, dict):
            continue
        event_id = str(row.get("event_id") or "").strip()
        team = str(row.get("team") or "").strip()
        if event_id and team:
            stats_by_event.setdefault(event_id, {})[team] = row

    by_competition: dict[str, dict[str, Any]] = {}
    for game in all_games:
        if not isinstance(game, dict):
            continue
        label = str(game.get("competition") or "").strip()
        comp_key = _normalise_competition(label)
        if not label or not comp_key:
            continue
        bucket = by_competition.setdefault(
            comp_key, {"labels": set(), "team_events": {}, "complete_games": 0}
        )
        bucket["labels"].add(label)
        if str(game.get("status") or "").strip().lower() not in TERMINAL_STATUSES:
            continue
        event_id = str(game.get("event_id") or "").strip()
        home = str(game.get("home_team") or "").strip()
        away = str(game.get("away_team") or "").strip()
        if not event_id or not home or not away or home == away:
            continue
        try:
            played = _historical_datetime(game.get("event_time"), "event_time")
            if played < lookback_start or played + timedelta(hours=24) >= as_of:
                continue
            per_team = stats_by_event.get(event_id, {})
            home_row, away_row = per_team.get(home), per_team.get(away)
            if not home_row or not away_row:
                continue
            _number(home_row.get("points"), "home_points")
            _number(away_row.get("points"), "away_points")
            _number(home_row.get("possessions"), "home_possessions", 1.0)
            _number(away_row.get("possessions"), "away_possessions", 1.0)
        except (TypeError, ValueError):
            continue

        bucket["complete_games"] += 1
        bucket["team_events"].setdefault(home, set()).add(event_id)
        bucket["team_events"].setdefault(away, set()).add(event_id)

    eligible_labels: list[str] = []
    audit: dict[str, dict[str, int]] = {}
    for comp_key, bucket in by_competition.items():
        teams_ready = sum(
            1 for event_ids in bucket["team_events"].values()
            if len(event_ids) >= minimum_history
        )
        team_counts = [len(event_ids) for event_ids in bucket["team_events"].values()]
        missing_team_games = sum(max(0, minimum_history - n) for n in team_counts)
        audit[comp_key] = {
            "complete_games": int(bucket["complete_games"]),
            "teams_with_minimum_history": teams_ready,
            "teams_total": len(team_counts),
            "max_games_for_one_team": max(team_counts) if team_counts else 0,
            # Each missing game fills two team slots, so halve the team-game gap.
            "estimated_games_to_import": (missing_team_games + 1) // 2,
        }
        if teams_ready >= 2:
            eligible_labels.extend(bucket["labels"])

    return sorted(set(eligible_labels)), audit


def _positive_price(value: Any) -> float | None:
    try:
        price = float(value)
        return price if math.isfinite(price) and price > 1.0001 else None
    except (TypeError, ValueError):
        return None


def _history_scope_for_fixture(fixture: dict[str, Any], all_games: list[dict], all_stats: list[dict],
                                start_time: datetime) -> tuple:
    competition = str(fixture.get("competition") or "").strip()
    if not competition:
        raise ValueError("fixture_competition_missing")
    normalized_competition = _normalise_competition(competition)
    comp_rows = [
        g for g in all_games
        if _normalise_competition(g.get("competition")) == normalized_competition
    ]
    if not comp_rows:
        imported_labels = sorted({
            str(g.get("competition") or "").strip()
            for g in all_games if isinstance(g, dict) and str(g.get("competition") or "").strip()
        })
        raise ValueError(
            "no_imported_history_for_competition_before_fixture:"
            f"fixture_competition={competition!r}:normalized={normalized_competition!r}:"
            f"imported_competitions={imported_labels[:12]!r}"
        )

    # Use a rolling 400-day window across seasons of the SAME normalized
    # competition. This supplies early-season context without mixing leagues.
    # _build_history still enforces point-in-time availability (+24h).
    lookback_start = start_time - timedelta(days=400)
    eligible_rows: list[tuple[datetime, dict]] = []
    parseable_dates = 0
    invalid_dates = 0
    latest_parseable = None
    for row in comp_rows:
        if not str(row.get("season") or "").strip():
            continue
        try:
            played = _historical_datetime(row.get("event_time"), "event_time")
            parseable_dates += 1
            if latest_parseable is None or played > latest_parseable:
                latest_parseable = played
        except ValueError:
            invalid_dates += 1
            continue
        if played >= lookback_start and played + timedelta(hours=24) < start_time:
            eligible_rows.append((played, row))

    if not eligible_rows:
        latest_text = latest_parseable.isoformat() if latest_parseable else "none"
        raise ValueError(
            "no_imported_history_for_competition_before_fixture:"
            f"fixture_competition={competition!r}:normalized={normalized_competition!r}:"
            f"matching_rows={len(comp_rows)}:parseable_dates={parseable_dates}:"
            f"invalid_dates={invalid_dates}:latest_event_time={latest_text}:"
            f"fixture_start_time={start_time.isoformat()}:lookback_days=400"
        )

    eligible_rows.sort(key=lambda item: item[0])
    scope_competition = str(eligible_rows[-1][1].get("competition") or competition)
    history_season = "rolling-400d"
    scoped_games = [
        dict(row, competition=scope_competition, season=history_season)
        for _, row in eligible_rows
    ]

    # Canonicalize team names across seasons after isolating the competition.
    team_lookup: dict[str, str] = {}
    for row in scoped_games:
        home_name = str(row.get("home_team") or "").strip()
        away_name = str(row.get("away_team") or "").strip()
        home_key = _normalise_team_label(home_name)
        away_key = _normalise_team_label(away_name)
        if home_key and home_name:
            team_lookup.setdefault(home_key, home_name)
        if away_key and away_name:
            team_lookup.setdefault(away_key, away_name)

    event_teams: dict[str, dict[str, str]] = {}
    for row in scoped_games:
        event_id = str(row.get("event_id") or "").strip()
        home_key = _normalise_team_label(row.get("home_team"))
        away_key = _normalise_team_label(row.get("away_team"))
        if not event_id or not home_key or not away_key or home_key == away_key:
            continue
        event_teams[event_id] = {
            home_key: team_lookup[home_key],
            away_key: team_lookup[away_key],
        }

    fixture_home_name = str(fixture.get("home") or "").strip()
    fixture_away_name = str(fixture.get("away") or "").strip()
    home_key = _fixture_team_lookup_key(fixture_home_name, normalized_competition, team_lookup)
    away_key = _fixture_team_lookup_key(fixture_away_name, normalized_competition, team_lookup)
    home = team_lookup.get(home_key)
    away = team_lookup.get(away_key)
    if not home or not away or home == away:
        imported_team_names = sorted(set(team_lookup.values()))
        raise ValueError(
            "fixture_team_names_do_not_match_imported_history:"
            f"fixture_home={fixture_home_name!r}:fixture_away={fixture_away_name!r}:"
            f"imported_team_names={imported_team_names[:40]!r}:history_window_days=400"
        )

    scoped_ids = set(event_teams)
    scoped_stats = []
    for row in all_stats:
        event_id = str(row.get("event_id") or "").strip()
        if event_id not in scoped_ids:
            continue
        canonical = event_teams[event_id].get(_normalise_team_label(row.get("team")))
        if canonical:
            scoped_stats.append(dict(row, team=canonical))

    as_of = start_time - timedelta(seconds=1)
    history, counters = _build_history(
        {"games": scoped_games, "team_stats": scoped_stats,
         "competition": scope_competition, "season": history_season},
        as_of,
    )
    available = [r for r in history if _aware_datetime(r.available_at, "available_at") < as_of]
    if not available:
        raise ValueError("no_eligible_team_box_scores_before_fixture")
    total_possessions = sum(r.possessions for r in available)
    if total_possessions <= 0:
        raise ValueError("invalid_historical_possessions")
    pace = sum(r.possessions for r in available) / len(available)
    ppp = sum(r.points_for for r in available) / total_possessions
    priors = LeaguePriors(pace=pace, points_per_possession=ppp)

    home_rates = fit_team_rates(history, home, as_of=as_of, priors=priors)
    away_rates = fit_team_rates(history, away, as_of=as_of, priors=priors)
    return history, counters, scope_competition, history_season, home, away, priors, home_rates, away_rates


def _fixture_market_groups(fixture: dict, now: datetime) -> dict:
    providers = fixture.get("providers") or {}
    raw_markets = ((fixture.get("periods") or {}).get("ft") or {})
    groups: dict[tuple, list] = {}
    specs = (
        ("moneyline", "moneyline", ("home", "away"), False),
        ("spread", "spread", ("home", "away"), True),
        ("total", "total", ("over", "under"), True),
        ("home_total", "team_total_home", ("over", "under"), True),
        ("away_total", "team_total_away", ("over", "under"), True),
    )
    for bookmaker, markets in raw_markets.items():
        if not isinstance(markets, dict):
            continue
        metadata = providers.get(bookmaker) or {}
        captured = metadata.get("captured_at") or fixture.get("updated_at")
        if not captured:
            continue
        try:
            age = (now - _aware_datetime(captured, "captured_at")).total_seconds()
        except ValueError:
            continue
        if age < -300 or age > MAX_QUOTE_AGE_SECONDS:
            continue
        for model_market, api_market, sides, has_line in specs:
            raw_items = markets.get(api_market)
            items = raw_items if isinstance(raw_items, list) else ([raw_items] if isinstance(raw_items, dict) else [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_captured = item.get("captured_at") or captured
                if not item_captured:
                    continue
                try:
                    item_age = (now - _aware_datetime(item_captured, "captured_at")).total_seconds()
                except ValueError:
                    continue
                if item_age < -300 or item_age > MAX_QUOTE_AGE_SECONDS:
                    continue
                line = None
                if has_line:
                    try:
                        line = _number(item.get("line"), "market_line", -10000)
                    except ValueError:
                        continue
                price_a = _positive_price(item.get(sides[0]))
                price_b = _positive_price(item.get(sides[1]))
                if not price_a or not price_b:
                    continue
                key = (model_market, None if line is None else round(line, 2))
                groups.setdefault(key, []).append({
                    "bookmaker": str(bookmaker),
                    "bookmaker_identity": str(item.get("bookmaker_identity") or bookmaker).strip(),
                    "captured_at": str(item_captured),
                    "price_a": price_a,
                    "price_b": price_b,
                    "side_a": sides[0],
                    "side_b": sides[1],
                })
    return groups


def _score_fixture(fixture: dict, all_games: list[dict], all_stats: list[dict],
                   simulations: int, minimum_history: int, now: datetime,
                   manual_groups: dict | None = None,
                   overtime_rule: str = "regulation_only",
                   diagnostics: dict | None = None,
                   relaxed: bool = False) -> tuple[list[dict], str | None]:
    start_time = _aware_datetime(fixture.get("start_time"), "fixture_start_time")
    history, counters, competition, season, home, away, priors, home_rates, away_rates = _history_scope_for_fixture(
        fixture, all_games, all_stats, start_time
    )
    if home_rates.games < minimum_history or away_rates.games < minimum_history:
        return [], f"historial_insuficiente:{home_rates.games}/{away_rates.games}"
    adjusted = None
    try:
        adjusted = fit_opponent_adjusted_ratings(
            history, as_of=start_time - timedelta(seconds=1),
            league_ppp=priors.points_per_possession,
        )
    except ValueError:
        adjusted = None
    projection = project_game(home_rates, away_rates, priors, adjusted_ratings=adjusted)
    fixture_id = str(fixture.get("game_id") or "")
    seed = 20261121 + sum(ord(ch) for ch in fixture_id)
    simulation = _simulate_game_for_settlement(
        projection, simulations=simulations, seed=seed, overtime_rule=overtime_rule
    )

    groups = manual_groups if manual_groups is not None else _fixture_market_groups(fixture, now)
    output = []
    if diagnostics is not None:
        diagnostics["market_groups_seen"] = diagnostics.get("market_groups_seen", 0) + len(groups)

    def reject(reason: str) -> None:
        if diagnostics is not None:
            diagnostics[reason] = diagnostics.get(reason, 0) + 1
    # Limit the influence of an unvalidated local model; anchor it to market consensus.
    shrink = min(0.35, min(home_rates.games, away_rates.games) / 40.0)
    for (market, line), quotes in groups.items():
        fair_samples = []
        for quote in quotes:
            inv_a, inv_b = 1.0 / quote["price_a"], 1.0 / quote["price_b"]
            total = inv_a + inv_b
            if total > 0:
                fair_samples.append({
                    quote["side_a"]: inv_a / total,
                    quote["side_b"]: inv_b / total,
                })
        if not fair_samples:
            reject("no_paired_market_prices")
            continue

        sides = ("home", "away") if market in ("moneyline", "spread") else ("over", "under")
        bookmaker_count = len({
            str(q.get("bookmaker_identity") or q.get("bookmaker") or "").strip()
            for q in quotes
            if str(q.get("bookmaker_identity") or q.get("bookmaker") or "").strip()
        })
        if diagnostics is not None and manual_groups is None:
            consensus_key = (
                "bookmaker_groups_1_house" if bookmaker_count == 1
                else "bookmaker_groups_2_houses" if bookmaker_count == 2
                else "bookmaker_groups_3plus_houses"
            )
            diagnostics[consensus_key] = diagnostics.get(consensus_key, 0) + 1
        if manual_groups is None and bookmaker_count < MIN_AUTOMATIC_BOOKMAKER_CONSENSUS:
            reject("fewer_than_2_bookmakers")
            continue

        fair = {}
        market_ranges_pp = {}
        for fair_side in sides:
            values = sorted(float(sample[fair_side]) for sample in fair_samples)
            mid = len(values) // 2
            fair[fair_side] = (
                values[mid] if len(values) % 2
                else (values[mid - 1] + values[mid]) / 2.0
            )
            market_ranges_pp[fair_side] = (values[-1] - values[0]) * 100.0
        max_dispersion_pp = max(market_ranges_pp.values(), default=0.0)
        if manual_groups is None and max_dispersion_pp > MAX_MARKET_DISPERSION * 100.0:
            reject("market_dispersion_over_4pp")
            continue

        quote_ages = []
        for quote in quotes:
            try:
                quote_ages.append(
                    max(0.0, (now - _aware_datetime(quote["captured_at"], "captured_at")).total_seconds())
                )
            except (KeyError, ValueError, TypeError):
                continue
        max_quote_age_seconds = max(quote_ages, default=0.0)
        effective_haircut, haircut_adjustments = _uncertainty_haircut(
            min(home_rates.games, away_rates.games),
            bookmaker_count,
            max_dispersion_pp,
            max_quote_age_seconds,
        )

        for side in sides:
            try:
                priced = price_game_market(
                    simulation, market, side=side, line=line,
                    overtime_rule="included" if overtime_rule == "includes_overtime" else "excluded",
                )
                raw_win = float(priced.p_win)
                raw_loss = float(priced.p_loss)
                push = float(priced.p_push)
            except (ValueError, TypeError, AttributeError):
                continue
            no_push_total = raw_win + raw_loss
            if no_push_total <= 0 or push > 0.005:
                reject("push_probability_over_0_5pp")
                continue

            raw_conditional = raw_win / no_push_total
            model_market_gap = abs(raw_conditional - fair[side])
            raw_minimum = BEST_EFFORT_MIN_RAW_MODEL_PROBABILITY if relaxed else MIN_RAW_MODEL_PROBABILITY
            market_minimum = BEST_EFFORT_MIN_MARKET_PROBABILITY if relaxed else MIN_MARKET_PROBABILITY
            gap_maximum = BEST_EFFORT_MAX_MODEL_MARKET_GAP if relaxed else MAX_MODEL_MARKET_GAP
            conservative_minimum = (
                BEST_EFFORT_MIN_ESTIMATED_LEG_PROBABILITY
                if relaxed else MIN_ESTIMATED_LEG_PROBABILITY
            )
            if raw_conditional < raw_minimum:
                reject("best_effort_model_probability_below_60pct" if relaxed else "model_probability_below_72pct")
                continue
            if fair[side] < market_minimum:
                reject("best_effort_market_probability_below_55pct" if relaxed else "no_vig_market_probability_below_60pct")
                continue
            if model_market_gap > gap_maximum:
                reject("best_effort_model_market_gap_over_15pp" if relaxed else "model_market_gap_over_12pp")
                continue

            blended = fair[side] + shrink * (raw_conditional - fair[side])
            conservative_probability = max(0.0, blended - effective_haircut)
            if conservative_probability < conservative_minimum:
                reject("best_effort_conservative_probability_below_58pct" if relaxed else "conservative_probability_below_70pct")
                continue

            for quote in quotes:
                if side == quote["side_a"]:
                    price = quote["price_a"]
                elif side == quote["side_b"]:
                    price = quote["price_b"]
                else:
                    continue
                ev = conservative_probability * price - 1.0
                edge = conservative_probability - fair[side]
                if relaxed:
                    if ev < BEST_EFFORT_MIN_EXPECTED_VALUE:
                        reject("best_effort_expected_value_below_minus_8pct")
                        continue
                elif edge <= 0.015 or ev <= 0.01:
                    reject("edge_or_expected_value_not_positive_enough")
                    continue
                passes_strict_filters = (
                    raw_conditional >= MIN_RAW_MODEL_PROBABILITY
                    and fair[side] >= MIN_MARKET_PROBABILITY
                    and model_market_gap <= MAX_MODEL_MARKET_GAP
                    and conservative_probability >= MIN_ESTIMATED_LEG_PROBABILITY
                    and edge > 0.015 and ev > 0.01
                )
                label = {
                    "moneyline": "Ganador",
                    "spread": "Hándicap",
                    "total": "Total del partido",
                    "home_total": "Total del local",
                    "away_total": "Total del visitante",
                }[market]
                if side in ("home", "away"):
                    selection = home if side == "home" else away
                    if market == "spread":
                        selection += f" {line:+g}"
                else:
                    selection = ("Más de " if side == "over" else "Menos de ") + f"{line:g}"
                    if market == "home_total":
                        selection += f" · {home}"
                    elif market == "away_total":
                        selection += f" · {away}"
                output.append({
                    "event_id": fixture_id,
                    "game": f"{home} vs {away}",
                    "competition": competition,
                    "season": season,
                    "start_time": fixture.get("start_time"),
                    "market": market,
                    "market_label": label,
                    "side": side,
                    "line": line,
                    "selection": selection,
                    "bookmaker": quote["bookmaker"],
                    "odds": price,
                    "probability": conservative_probability,
                    "passes_strict_filters": passes_strict_filters,
                    "selection_mode": "strict" if passes_strict_filters else "best_effort",
                    "raw_model_probability": raw_conditional,
                    "market_probability": fair[side],
                    "market_consensus_bookmakers": bookmaker_count,
                    "market_probability_range_pp": market_ranges_pp[side],
                    "model_market_gap_pp": model_market_gap * 100.0,
                    "probability_uncertainty_haircut_pp": effective_haircut * 100.0,
                    "probability_uncertainty_adjustments_pp": {
                        key: value * 100.0 for key, value in haircut_adjustments.items()
                    },
                    "market_dispersion_max_pp": max_dispersion_pp,
                    "quote_age_max_seconds": max_quote_age_seconds,
                    "model_weight": shrink,
                    "projected_home_points": projection.home_points_mean,
                    "projected_away_points": projection.away_points_mean,
                    "projected_total_points": projection.home_points_mean + projection.away_points_mean,
                    "projected_pace": projection.pace,
                    "edge_percentage_points": edge * 100.0,
                    "expected_value_per_unit": ev,
                    "push_probability": push,
                    "home_history_games": home_rates.games,
                    "away_history_games": away_rates.games,
                    "historical_games_used": counters.get("games_used", 0),
                    "simulation_count": simulation.simulation_count,
                    "model_version": projection.model_version,
                    "rating_method": projection.rating_method,
                    "odds_captured_at": quote["captured_at"],
                    "settlement": overtime_rule,
                    "score": edge * 0.6 + min(ev, 0.5) * 0.4,
                })
                if diagnostics is not None:
                    diagnostics["selections_passing_all_filters"] = diagnostics.get("selections_passing_all_filters", 0) + 1
    return output, None


def _score_fixture_preview(fixture: dict, all_games: list[dict], all_stats: list[dict],
                           simulations: int, minimum_history: int) -> dict:
    """Project one scheduled game using local history only; never reads bookmaker odds."""
    start_time = _aware_datetime(fixture.get("start_time"), "fixture_start_time")
    history, counters, competition, season, home, away, priors, home_rates, away_rates = _history_scope_for_fixture(
        fixture, all_games, all_stats, start_time
    )
    if home_rates.games < minimum_history or away_rates.games < minimum_history:
        raise ValueError(f"historial_insuficiente:{home_rates.games}/{away_rates.games}")
    adjusted = None
    try:
        adjusted = fit_opponent_adjusted_ratings(
            history, as_of=start_time - timedelta(seconds=1),
            league_ppp=priors.points_per_possession,
        )
    except ValueError:
        adjusted = None
    projection = project_game(home_rates, away_rates, priors, adjusted_ratings=adjusted)
    fixture_id = str(fixture.get("game_id") or "")
    seed = 20261122 + sum(ord(ch) for ch in fixture_id)
    simulation = simulate_game(projection, simulations=simulations, seed=seed)

    def probability(market: str, side: str, line: float | None = None) -> float:
        priced = price_game_market(
            simulation, market, side=side, line=line, overtime_rule="excluded"
        )
        return float(priced.p_win)

    return {
        "fixture_id": fixture_id,
        "game": f"{home} vs {away}",
        "competition": competition,
        "history_season": season,
        "start_time": fixture.get("start_time"),
        "home_team": home,
        "away_team": away,
        "projected_home_points": projection.home_points_mean,
        "projected_away_points": projection.away_points_mean,
        "projected_total_points": projection.home_points_mean + projection.away_points_mean,
        "home_win_probability": probability("moneyline", "home"),
        "away_win_probability": probability("moneyline", "away"),
        "home_100_plus_probability": probability("home_total", "over", 99.5),
        "away_100_plus_probability": probability("away_total", "over", 99.5),
        "home_history_games": home_rates.games,
        "away_history_games": away_rates.games,
        "historical_games_used": counters.get("games_used", 0),
        "model_version": projection.model_version,
        "rating_method": projection.rating_method,
        "simulations": simulation.simulation_count,
    }


@app.post("/api/parlay/preview")
async def preview_parlay(request: Request):
    """Local-only pre-analysis of supplied fixtures. This endpoint must never call HoopsAPI."""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request_body_must_be_object")
        start_time = _aware_datetime(payload.get("window_start"), "window_start")
        end_time = _aware_datetime(payload.get("window_end"), "window_end")
        if end_time <= start_time:
            raise ValueError("window_end_must_be_after_window_start")
        if end_time - start_time > timedelta(days=7):
            raise ValueError("maximum_window_is_7_days")
        simulations = int(payload.get("simulations", 25000))
        if not 1000 <= simulations <= 50000:
            raise ValueError("simulations_must_be_between_1000_and_50000")
        minimum_history = int(payload.get("minimum_history_games", MIN_HISTORICAL_GAMES))
        if not 3 <= minimum_history <= 30:
            raise ValueError("minimum_history_games_must_be_between_3_and_30")
        minimum_history = max(MIN_HISTORICAL_GAMES, minimum_history)
        all_games = payload.get("games")
        all_stats = payload.get("team_stats")
        fixtures = payload.get("fixtures")
        if not isinstance(all_games, list) or not isinstance(all_stats, list):
            raise ValueError("local_games_and_team_stats_must_be_arrays")
        if not isinstance(fixtures, list) or not fixtures:
            raise ValueError("import_upcoming_fixtures_from_sportapi_or_csv_first")
        if len(all_games) > 20000 or len(all_stats) > 40000 or len(fixtures) > 100:
            raise ValueError("too_many_local_rows_or_fixtures")

        in_window = []
        for item in fixtures:
            if not isinstance(item, dict):
                continue
            start_value = item.get("start_time") or item.get("event_time")
            try:
                kickoff = _aware_datetime(start_value, "fixture_start_time")
            except ValueError:
                continue
            if start_time <= kickoff <= end_time:
                in_window.append({
                    "game_id": str(item.get("fixture_id") or item.get("event_id") or ""),
                    "competition": str(item.get("competition") or "").strip(),
                    "start_time": start_value,
                    "home": str(item.get("home_team") or item.get("home") or "").strip(),
                    "away": str(item.get("away_team") or item.get("away") or "").strip(),
                })
        in_window.sort(key=lambda item: str(item.get("start_time") or ""))
        candidates = []
        skipped: dict[str, int] = {}
        examples: list[str] = []
        for fixture in in_window:
            if not fixture["competition"] or not fixture["home"] or not fixture["away"]:
                skipped["fixture_fields_missing"] = skipped.get("fixture_fields_missing", 0) + 1
                continue
            try:
                candidates.append(_score_fixture_preview(
                    fixture, all_games, all_stats, simulations, minimum_history
                ))
            except ValueError as exc:
                detail = str(exc)
                key = detail.split(":")[0]
                skipped[key] = skipped.get(key, 0) + 1
                if len(examples) < 8:
                    examples.append(detail)
            except Exception:
                skipped["analysis_error"] = skipped.get("analysis_error", 0) + 1

        warnings = [
            "LOCAL_ONLY_PREVIEW: esta ejecución no consulta HoopsAPI ni obtiene cuotas.",
            "Las probabilidades proceden de simulaciones exploratorias y no están calibradas ni validadas.",
            "Las proyecciones y probabilidades usan el historial local disponible y la muestra indicada por equipo.",
            "El historial usa una ventana móvil de hasta 400 días de la misma competición e incluye temporadas anteriores; los cambios de plantel no están modelados.",
            "El umbral 100+ equivale a superar 99,5 puntos en tiempo reglamentario; no incluye prórroga.",
            "Esta vista no genera una combinada ni calcula valor esperado: eso requiere cuotas actuales y comparables.",
        ]
        return {
            "status": "EXPLORATORY_NOT_VALIDATED",
            "mode": "LOCAL_ONLY_NO_ODDS",
            "window": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "fixtures_in_window": len(in_window),
            "fixtures_analysed": len(candidates),
            "local_games_count": len(all_games),
            "local_team_stats_count": len(all_stats),
            "imported_competitions": imported_competition_labels,
            "modelable_competitions": modelable_competition_labels,
            "competition_history_audit": competition_history_audit,
            "skipped": skipped,
            "skip_examples": examples,
            "candidates": candidates[:30],
            "odds_provider_called": False,
            "best_parlay": None,
            "warnings": warnings,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"local_preview_failed:{type(exc).__name__}") from exc




@app.post("/api/parlay/bookmaker-recommend")
async def recommend_bookmaker_parlay(request: Request):
    """Score manually captured, paired sportsbook quotes using local team history only."""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request_body_must_be_object")
        house = str(payload.get("house") or "").strip()
        if house not in {"GanamosVIP", "Azar247"}:
            raise ValueError("house_must_be_GanamosVIP_or_Azar247")
        start_time = _aware_datetime(payload.get("window_start"), "window_start")
        end_time = _aware_datetime(payload.get("window_end"), "window_end")
        if end_time <= start_time:
            raise ValueError("window_end_must_be_after_window_start")
        if end_time - start_time > timedelta(days=7):
            raise ValueError("maximum_window_is_7_days")
        target_odds = _number(payload.get("target_odds"), "target_odds", 1.01)
        if target_odds > 100:
            raise ValueError("target_odds_must_not_exceed_100")
        simulations = int(payload.get("simulations", 25000))
        if not 1000 <= simulations <= 50000:
            raise ValueError("simulations_must_be_between_1000_and_50000")
        minimum_history = int(payload.get("minimum_history_games", MIN_HISTORICAL_GAMES))
        if not 8 <= minimum_history <= 30:
            raise ValueError("minimum_history_games_must_be_between_8_and_30")
        minimum_history = max(MIN_HISTORICAL_GAMES, minimum_history)

        all_games = payload.get("games")
        all_stats = payload.get("team_stats")
        offers = payload.get("offers")
        if not isinstance(all_games, list) or not isinstance(all_stats, list):
            raise ValueError("local_games_and_team_stats_must_be_arrays")
        if not isinstance(offers, list) or not offers:
            raise ValueError("save_bookmaker_quotes_before_analysis")
        if len(all_games) > 20000 or len(all_stats) > 40000 or len(offers) > 500:
            raise ValueError("too_many_local_rows_or_bookmaker_quotes")

        now = datetime.now(timezone.utc)
        fixtures_by_id: dict[str, dict] = {}
        groups_by_id: dict[str, dict] = {}
        unsupported_market_quotes = 0
        unconfirmed_settlement_quotes = 0
        accepted_quote_groups = 0

        for item in offers:
            if not isinstance(item, dict) or str(item.get("house") or "").strip() != house:
                continue
            market = str(item.get("market") or "").strip().lower()
            if market not in SUPPORTED_MARKETS:
                unsupported_market_quotes += 1
                continue
            settlement = str(item.get("settlement") or "unknown")
            if settlement not in {"regulation_only", "includes_overtime"}:
                unconfirmed_settlement_quotes += 1
                continue

            event_id = str(item.get("event_id") or "").strip()
            competition = str(item.get("competition") or "").strip()
            home = str(item.get("home_team") or "").strip()
            away = str(item.get("away_team") or "").strip()
            if not event_id or not competition or not home or not away or home == away:
                continue
            start_value = item.get("start_time")
            kickoff = _aware_datetime(start_value, "fixture_start_time")
            fixture = {
                "game_id": event_id,
                "competition": competition,
                "start_time": start_value,
                "home": home,
                "away": away,
            }
            previous = fixtures_by_id.get(event_id)
            if previous and (
                previous["competition"] != competition
                or previous["home"] != home
                or previous["away"] != away
                or previous["start_time"] != start_value
            ):
                raise ValueError("same_event_id_has_conflicting_fixture_details")
            fixtures_by_id[event_id] = fixture

            captured = item.get("captured_at")
            if not captured:
                continue
            try:
                age = (now - _aware_datetime(captured, "captured_at")).total_seconds()
            except ValueError:
                continue
            if age < -300 or age > MAX_QUOTE_AGE_SECONDS:
                continue

            if market in {"moneyline", "spread"}:
                side_a, side_b = "home", "away"
            else:
                side_a, side_b = "over", "under"
            price_a = _positive_price(item.get("price_a"))
            price_b = _positive_price(item.get("price_b"))
            if not price_a or not price_b:
                continue

            if market == "moneyline":
                line = None
            else:
                line = _number(item.get("line"), "market_line", -10000)
            group_key = (market, None if line is None else round(line, 2))
            event_groups = groups_by_id.setdefault(event_id, {})
            settlement_groups = event_groups.setdefault(settlement, {})
            settlement_groups.setdefault(group_key, []).append({
                "bookmaker": house,
                "captured_at": str(captured),
                "price_a": price_a,
                "price_b": price_b,
                "side_a": side_a,
                "side_b": side_b,
            })
            accepted_quote_groups += 1

        in_window = []
        for event_id, fixture in fixtures_by_id.items():
            kickoff = _aware_datetime(fixture.get("start_time"), "fixture_start_time")
            if start_time <= kickoff <= end_time and groups_by_id.get(event_id):
                in_window.append((event_id, fixture))
        in_window.sort(key=lambda row: str(row[1].get("start_time") or ""))
        scan_items = in_window[:50]

        all_candidates = []
        skipped: dict[str, int] = {}
        history_skip_examples: list[str] = []
        team_name_skip_examples: list[str] = []
        analysed = 0
        for event_id, fixture in scan_items:
            for settlement, manual_groups in groups_by_id.get(event_id, {}).items():
                try:
                    candidates, reason = _score_fixture(
                        fixture, all_games, all_stats, simulations, minimum_history, now,
                        manual_groups=manual_groups, overtime_rule=settlement,
                    )
                    if reason:
                        key = reason.split(":")[0]
                        skipped[key] = skipped.get(key, 0) + 1
                        if key == "no_imported_history_for_competition_before_fixture" and len(history_skip_examples) < 5:
                            history_skip_examples.append(reason)
                        if key == "fixture_team_names_do_not_match_imported_history" and len(team_name_skip_examples) < 5:
                            team_name_skip_examples.append(reason)
                    else:
                        analysed += 1
                        all_candidates.extend(candidates)
                except ValueError as exc:
                    detail = str(exc)
                    key = detail.split(":")[0]
                    skipped[key] = skipped.get(key, 0) + 1
                    if key == "no_imported_history_for_competition_before_fixture" and len(history_skip_examples) < 5:
                        history_skip_examples.append(detail)
                    if key == "fixture_team_names_do_not_match_imported_history" and len(team_name_skip_examples) < 5:
                        team_name_skip_examples.append(detail)
                except Exception:
                    skipped["analysis_error"] = skipped.get("analysis_error", 0) + 1

        best_by_selection = {}
        for candidate in all_candidates:
            key = (candidate["event_id"], candidate["market"], candidate["side"], candidate["line"], candidate["settlement"])
            previous = best_by_selection.get(key)
            if previous is None or candidate["odds"] > previous["odds"]:
                best_by_selection[key] = candidate
        best_by_game = {}
        for candidate in best_by_selection.values():
            previous = best_by_game.get(candidate["event_id"])
            if previous is None or (
                candidate["probability"], candidate["edge_percentage_points"], candidate["score"]
            ) > (previous["probability"], previous["edge_percentage_points"], previous["score"]):
                best_by_game[candidate["event_id"]] = candidate
        top_picks = sorted(
            best_by_game.values(),
            key=lambda item: (item["probability"], item["edge_percentage_points"], item["score"]),
            reverse=True,
        )[:12]

        by_bookmaker: dict[str, list[dict]] = {}
        for candidate in all_candidates:
            by_bookmaker.setdefault(candidate["bookmaker"], []).append(candidate)
        best_parlay = None
        for bookmaker, items in by_bookmaker.items():
            per_game = {}
            for item in items:
                old = per_game.get(item["event_id"])
                if old is None or (
                    item["probability"], item["edge_percentage_points"], item["score"]
                ) > (old["probability"], old["edge_percentage_points"], old["score"]):
                    per_game[item["event_id"]] = item
            pool = sorted(
                per_game.values(),
                key=lambda item: (item["probability"], item["edge_percentage_points"], item["score"]),
                reverse=True,
            )[:18]
            local_best = None
            target_min, target_max = _parlay_odds_band(target_odds)
            for leg_count in range(1, min(8, len(pool)) + 1):
                for combo in combinations(pool, leg_count):
                    combined_odds = math.prod(item["odds"] for item in combo)
                    if combined_odds < target_min - 1e-9 or combined_odds > target_max + 1e-9:
                        continue
                    hit_probability = math.prod(item["probability"] for item in combo)
                    rank = _parlay_rank(hit_probability, combined_odds, target_odds, leg_count)
                    if local_best is None or rank > local_best["rank"]:
                        local_best = {
                            "rank": rank,
                            "bookmaker": bookmaker,
                            "combined_odds": combined_odds,
                            "target_odds": target_odds,
                            "target_odds_min": target_min,
                            "target_odds_max": target_max,
                            "odds_deviation": combined_odds - target_odds,
                            "odds_deviation_percent": ((combined_odds / target_odds) - 1.0) * 100.0,
                            "estimated_hit_probability": hit_probability,
                            "legs": list(combo),
                        }
            if local_best and (
                best_parlay is None or local_best["rank"] > best_parlay["rank"]
            ):
                best_parlay = local_best

        if best_parlay:
            best_parlay.pop("rank", None)
            best_parlay["status"] = "EXPLORATORY_NOT_VALIDATED"

        warnings = [
            "EXPLORATORY_NOT_VALIDATED: el modelo no tiene validación walk-forward/holdout aprobada; el 70% no está calibrado ni es una tasa de acierto demostrada.",
            "Las cuotas se cargaron manualmente; no existe una conexión autenticada ni actualización automática con la casa.",
            "Filtro reforzado: cada pierna debe superar 70% tras un descuento de incertidumbre de 2 puntos; exige simulación bruta >=72%, mercado sin margen >=60%, brecha modelo-mercado <=12 puntos y al menos 12 partidos completos por equipo.",
            "Las cotizaciones manuales usan una sola casa como referencia; no equivalen a consenso de tres operadores.",
            "Solo se evalúan ganador, hándicap, total del partido y totales por equipo; cada cotización requiere regla de liquidación confirmada. OT incluido se simula con prórrogas de cinco minutos hasta desempatar; esta extensión usa una aproximación no validada.",
            "No se usan mercados de jugadores, mitades, cuartos, margen, par/impar ni otros mercados no modelados por el motor actual.",
            "El motor no incorpora de forma fiable lesiones, titulares, minutos esperados, descanso ni viajes; se apoya principalmente en puntos y posesiones.",
            "La combinada se busca dentro de una banda de +/-5% de la cuota objetivo; se admite quedar ligeramente por debajo o por encima, pero no fuera de ese rango.",
            "La probabilidad combinada multiplica las probabilidades de las piernas y supone independencia aproximada entre partidos.",
            "Las cuotas manuales deben haberse capturado en los últimos 15 minutos y pueden cambiar antes de apostar.",
            "El historial usa una ventana móvil de hasta 400 días de la misma competición; los cambios de plantel no están modelados.",
        ]
        if unconfirmed_settlement_quotes:
            warnings.append(
                str(unconfirmed_settlement_quotes)
                + " cotizaciones se excluyeron porque la regla de prórroga no estaba confirmada."
            )
        if unsupported_market_quotes:
            warnings.append(
                str(unsupported_market_quotes)
                + " cotizaciones se excluyeron porque el mercado no está soportado por el modelo actual."
            )
        if len(in_window) > 50:
            warnings.append("Había más de 50 partidos guardados en la ventana; se analizaron los primeros 50 por hora de inicio.")

        imported_competitions = sorted({
            str(game.get("competition") or "").strip()
            for game in all_games if isinstance(game, dict) and str(game.get("competition") or "").strip()
        })
        matching_count = sum(
            1 for _, fixture in in_window
            if _normalise_competition(fixture.get("competition")) in {
                _normalise_competition(label) for label in imported_competitions
            }
        )
        return {
            "status": "EXPLORATORY_NOT_VALIDATED",
            "mode": "MANUAL_BOOKMAKER_QUOTES",
            "house": house,
            "window": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "target_odds": target_odds,
            "target_odds_band": {
                "minimum": _parlay_odds_band(target_odds)[0],
                "target": target_odds,
                "maximum": _parlay_odds_band(target_odds)[1],
                "tolerance_percent": PARLAY_ODDS_TOLERANCE_FRACTION * 100.0,
            },
            "matches_in_window": len(in_window),
            "matches_scanned": len(scan_items),
            "matches_analysed": analysed,
            "matches_matching_imported_competitions": matching_count,
            "local_games_count": len(all_games),
            "local_team_stats_count": len(all_stats),
            "imported_competitions": imported_competitions,
            "fixture_competitions_scanned": sorted({
                str(fixture.get("competition") or "") for _, fixture in scan_items
            }),
            "skipped": skipped,
            "history_skip_examples": history_skip_examples,
            "team_name_skip_examples": team_name_skip_examples,
            "quotes_supplied": len(offers),
            "quote_market_groups_used": accepted_quote_groups,
            "unconfirmed_settlement_quotes": unconfirmed_settlement_quotes,
            "unsupported_market_quotes": unsupported_market_quotes,
            "eligible_market_selections": len(all_candidates),
            "minimum_estimated_leg_probability": MIN_ESTIMATED_LEG_PROBABILITY,
            "minimum_history_games": minimum_history,
            "top_picks": top_picks,
            "best_parlay": best_parlay,
            "parlay_search": parlay_search_diagnostics,
            "warnings": warnings,
            "provider": house + " (cuotas manuales)",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"bookmaker_recommendation_failed:{type(exc).__name__}",
        ) from exc



@app.get("/api/hoopsapi/status")
def hoopsapi_status():
    from hoopsapi_provider import status
    return status()


@app.get("/api/oddspapi/status")
def oddspapi_status():
    from oddspapi_provider import account_status, OddsPapiError
    try:
        return account_status()
    except OddsPapiError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

@app.get("/api/oddspapi/market-audit")
def oddspapi_market_audit():
    from oddspapi_provider import market_catalogue_audit, OddsPapiError
    try:
        return market_catalogue_audit()
    except OddsPapiError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@app.post("/api/parlay/recommend")
async def recommend_parlay(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request_body_must_be_object")
        start_time = _aware_datetime(payload.get("window_start"), "window_start")
        end_time = _aware_datetime(payload.get("window_end"), "window_end")
        if end_time <= start_time:
            raise ValueError("window_end_must_be_after_window_start")
        if end_time - start_time > timedelta(days=7):
            raise ValueError("maximum_window_is_7_days")
        target_odds = _number(payload.get("target_odds"), "target_odds", 1.01)
        if target_odds > 100:
            raise ValueError("target_odds_must_not_exceed_100")
        simulations = int(payload.get("simulations", 25000))
        if not 1000 <= simulations <= 50000:
            raise ValueError("simulations_must_be_between_1000_and_50000")
        minimum_history = int(payload.get("minimum_history_games", MIN_HISTORICAL_GAMES))
        if not 8 <= minimum_history <= 30:
            raise ValueError("minimum_history_games_must_be_between_8_and_30")
        minimum_history = max(MIN_HISTORICAL_GAMES, minimum_history)
        # Automatic parlay requests always enforce the strict 70% policy.
        # Ignore legacy clients that still send best_effort=true.
        best_effort_requested = False
        all_games = payload.get("games")
        all_stats = payload.get("team_stats")
        if not isinstance(all_games, list) or not isinstance(all_stats, list):
            raise ValueError("local_games_and_team_stats_must_be_arrays")
        if len(all_games) > 20000 or len(all_stats) > 40000:
            raise ValueError("too_many_local_rows")

        imported_competition_labels = sorted({
            str(g.get("competition") or "").strip()
            for g in all_games
            if isinstance(g, dict) and str(g.get("competition") or "").strip()
        })
        modelable_competition_labels, competition_history_audit = _modelable_competition_labels(
            all_games, all_stats, minimum_history, start_time
        )
        imported_competitions = {
            _normalise_competition(label) for label in modelable_competition_labels
        }

        provider_name = "HoopsAPI"
        overtime_rule = "regulation_only"
        if os.environ.get("ODDSPAPI_API_KEY", "").strip():
            provider_name = "OddsPapi"
            overtime_rule = "includes_overtime"
        if not modelable_competition_labels:
            # Abstain before external calls when no league has enough complete
            # history; do not spend bookmaker quota on games the model rejects.
            feed = {
                "games": [], "count": 0, "matching_fixture_count": 0,
                "matching_competitions": [], "available_competitions": [],
                "odds_fixtures_requested": 0, "odds_fixtures_with_markets": 0,
                "quota_remaining_before_scan": None, "bookmakers_requested": [],
                "scan_cap": 0, "markets_catalog_loaded": False,
            }
        elif provider_name == "OddsPapi":
            from oddspapi_provider import games as fetch_odds_games
            # Filter by leagues with complete, sufficiently deep history before
            # OddsPapi retrieves any individual bookmaker odds.
            feed = await run_in_threadpool(
                fetch_odds_games, start_time, end_time, modelable_competition_labels
            )
        else:
            from hoopsapi_provider import games as fetch_odds_games
            feed = await run_in_threadpool(fetch_odds_games)
        now = datetime.now(timezone.utc)
        in_window = []
        for fixture in feed.get("games") or []:
            try:
                kickoff = _aware_datetime(fixture.get("start_time"), "start_time")
            except ValueError:
                continue
            if start_time <= kickoff <= end_time:
                in_window.append(fixture)
        in_window.sort(key=lambda x: str(x.get("start_time") or ""))
        total_in_window = int(feed.get("count") or len(in_window)) if provider_name == "OddsPapi" else len(in_window)

        # Never fall back to unrelated competitions: every scored fixture
        # must have a matching local competition history.
        matching_fixtures = [
            x for x in in_window
            if _normalise_competition(x.get("competition")) in imported_competitions
        ]
        scan_games = matching_fixtures[:200]

        all_candidates = []
        skipped: dict[str, int] = {}
        history_skip_examples: list[str] = []
        history_deficit_examples: list[dict] = []
        team_name_skip_examples: list[str] = []
        filter_diagnostics: dict[str, int] = {}
        analysed = 0
        for fixture in scan_games:
            if not fixture.get("home") or not fixture.get("away"):
                skipped["fixture_fields_missing"] = skipped.get("fixture_fields_missing", 0) + 1
                continue
            try:
                # Simulations are CPU-heavy; run them outside the event loop.
                candidates, reason = await run_in_threadpool(
                    _score_fixture,
                    fixture, all_games, all_stats, simulations, minimum_history, now,
                    overtime_rule=overtime_rule, diagnostics=filter_diagnostics,
                    relaxed=best_effort_requested,
                )
                if reason:
                    reason_key = reason.split(":")[0]
                    skipped[reason_key] = skipped.get(reason_key, 0) + 1
                    if reason_key == "historial_insuficiente" and len(history_deficit_examples) < 20:
                        try:
                            count_values = reason.split(":", 1)[1].split("/", 1)
                            home_games, away_games = int(count_values[0]), int(count_values[1])
                        except (IndexError, TypeError, ValueError):
                            home_games, away_games = None, None
                        history_deficit_examples.append({
                            "event_id": str(fixture.get("game_id") or ""),
                            "competition": str(fixture.get("competition") or ""),
                            "home_team": str(fixture.get("home") or ""),
                            "away_team": str(fixture.get("away") or ""),
                            "home_history_games": home_games,
                            "away_history_games": away_games,
                            "minimum_history_games": minimum_history,
                        })
                else:
                    analysed += 1
                    all_candidates.extend(candidates)
            except ValueError as exc:
                detail = str(exc)
                key = detail.split(":")[0]
                skipped[key] = skipped.get(key, 0) + 1
                if key == "no_imported_history_for_competition_before_fixture" and len(history_skip_examples) < 5:
                    history_skip_examples.append(detail)
                if key == "fixture_team_names_do_not_match_imported_history" and len(team_name_skip_examples) < 5:
                    team_name_skip_examples.append(detail)
            except Exception:
                skipped["analysis_error"] = skipped.get("analysis_error", 0) + 1

        # Keep strict candidates separate from relaxed alternatives. Best-effort never
        # relaxes history, matching, paired prices, bookmaker consensus, dispersion,
        # freshness, or push-probability safeguards inside _score_fixture.
        probability_floor = MIN_ESTIMATED_LEG_PROBABILITY
        candidates_before_probability_filter = len(all_candidates)
        strict_candidates = [x for x in all_candidates if x.get("passes_strict_filters")]

        def _top_picks(source: list[dict]) -> list[dict]:
            best_by_selection = {}
            for candidate in source:
                key = (candidate["event_id"], candidate["market"], candidate["side"], candidate["line"])
                previous = best_by_selection.get(key)
                if previous is None or candidate["odds"] > previous["odds"]:
                    best_by_selection[key] = candidate
            best_by_game = {}
            for candidate in best_by_selection.values():
                previous = best_by_game.get(candidate["event_id"])
                if previous is None or (
                    candidate["probability"], candidate["edge_percentage_points"], candidate["score"]
                ) > (previous["probability"], previous["edge_percentage_points"], previous["score"]):
                    best_by_game[candidate["event_id"]] = candidate
            return sorted(
                best_by_game.values(),
                key=lambda x: (x["probability"], x["edge_percentage_points"], x["score"]),
                reverse=True,
            )[:12]

        strict_best_parlay, parlay_search_diagnostics = best_automatic_parlay(
            strict_candidates, target_odds
        )
        best_effort_fallback_attempted = bool(best_effort_requested and strict_best_parlay is None)
        active_candidates = all_candidates if best_effort_fallback_attempted else strict_candidates
        top_picks = _top_picks(active_candidates)
        best_parlay = strict_best_parlay

        if best_parlay:
            best_parlay.pop("rank", None)
            best_parlay["status"] = (
                "BEST_EFFORT_NOT_VALIDATED"
                if best_effort_fallback_attempted else "EXPLORATORY_NOT_VALIDATED"
            )
            best_parlay["passes_strict_filters"] = all(
                leg.get("passes_strict_filters", False) for leg in best_parlay["legs"]
            )
        warnings = [
            "EXPLORATORY_NOT_VALIDATED: el modelo no tiene validación walk-forward/holdout aprobada; el 70% no está calibrado ni es una tasa de acierto demostrada.",
            ("El modelo simula prórrogas de forma aproximada para mercados que las incluyen; no equivale a una liquidación oficial exacta." if provider_name == "OddsPapi" else "El modelo base simula tiempo reglamentario; las cuotas FT de HoopsAPI incluyen prórroga. La comparación de probabilidades no es perfectamente equivalente."),
            "El 70% mínimo se exige a CADA pick individual de la combinada; no se exige 70% al conjunto. La probabilidad conjunta estimada de toda la combinada tiene un mínimo separado del 30%.",
            "Filtro reforzado: cada pierna debe superar 70% conservador; parte de un descuento base de 2 pp y suma ajustes transparentes de hasta 0,5 pp por muestra corta, consenso de solo 2 casas, dispersión y antigüedad de cuota. Exige simulación bruta >=72%, mercado sin margen >=60% y brecha modelo-mercado <=12 pp.",
            "El análisis automático exige precios emparejados de al menos 2 casas y dispersión máxima de 4 puntos porcentuales entre ellas.",
            "Se exigen al menos 12 partidos históricos completos por equipo y cuotas capturadas en los últimos 15 minutos.",
            "La combinada se busca dentro de una banda de +/-5% de la cuota objetivo; puede quedar ligeramente por debajo o por encima, pero no fuera de ese rango.",
            "La combinada exige una probabilidad conjunta estimada mínima del 30%. El cálculo multiplica probabilidades por pierna y sigue suponiendo independencia aproximada: no es una probabilidad conjunta calibrada ni un límite garantizado.",
            "El motor actual no incorpora de forma fiable lesiones, titulares confirmados, minutos esperados, descanso ni viajes; los cambios de plantel no están modelados.",
            "El motor se apoya principalmente en puntos y posesiones; todavía no aprovecha de forma completa tiro, rebotes, pérdidas ni métricas de jugadores.",
            "Solo se analizan competiciones y equipos con estadísticas históricas locales suficientes; no se inventan datos faltantes.",
            "El historial usa una ventana móvil de hasta 400 días de la misma competición e incluye temporadas anteriores.",
            "Las cuotas de terceros pueden cambiar antes de apostar.",
        ]
        if best_effort_fallback_attempted:
            if best_parlay:
                warnings.insert(
                    0,
                    "BEST_EFFORT_NOT_VALIDATED: no se encontró una combinada que cumpliera todos los filtros estrictos. "
                    "Esta alternativa relaja los umbrales de probabilidad, edge y EV; mantiene historial mínimo, "
                    "competición y equipos coincidentes, 3 casas, dispersión máxima, cuotas recientes y control de push. "
                    "No es una recomendación validada ni una tasa de acierto demostrada."
                )
            elif active_candidates:
                warnings.insert(
                    0,
                    "BEST_EFFORT_NOT_VALIDATED: no hubo combinada estricta ni alternativa dentro de la banda de cuota. "
                    "Se muestran picks exploratorios que no cumplen todos los filtros estrictos."
                )
            else:
                warnings.insert(
                    0,
                    "No se encontró una alternativa de mejor esfuerzo. Los filtros de datos, historial, "
                    "consenso, dispersión, frescura y coincidencia siguieron vigentes."
                )
        excluded_competitions = sorted(
            label for label in imported_competition_labels
            if _normalise_competition(label) not in imported_competitions
        )
        if excluded_competitions:
            warnings.append(
                "Se excluyeron antes de consultar cuotas las competiciones sin al menos dos equipos "
                f"con {minimum_history} partidos completos en los últimos 400 días: "
                + ", ".join(excluded_competitions)
                + "."
            )
            # Say exactly what is missing so the user knows what to import.
            for label in excluded_competitions:
                info = competition_history_audit.get(_normalise_competition(label)) or {}
                if info.get("teams_total"):
                    warnings.append(
                        f"Falta historial en {label}: {info['teams_total']} equipos, el más completo tiene "
                        f"{info['max_games_for_one_team']}/{minimum_history} partidos con box score. "
                        f"Necesita aproximadamente {info['estimated_games_to_import']} partidos completos más desde una fuente compatible de box scores. "
                        "El importador alternativo intentará cubrir las ligas públicas disponibles; las demás pueden requerir CSV."
                    )
        if not modelable_competition_labels:
            warnings.append(
                f"Ninguna competición importada tiene dos equipos con al menos {minimum_history} partidos "
                "completos y disponibles antes del inicio de la ventana. No se consultaron APIs ni cuotas."
            )
        elif provider_name == "OddsPapi" and int(feed.get("matching_fixture_count") or 0) == 0:
            warnings.append(
                "OddsPapi no encontró partidos con cuotas en las competiciones con historial suficiente: "
                + ", ".join(modelable_competition_labels)
                + ". No se consultaron cuotas de ligas ajenas al historial."
            )
        elif provider_name == "OddsPapi" and int(feed.get("odds_fixtures_requested") or 0) == 0:
            warnings.append(
                "Se encontraron competiciones compatibles, pero no hubo solicitudes de cuotas disponibles en la cuota restante de OddsPapi."
            )
        elif provider_name == "OddsPapi" and int(feed.get("matching_fixture_count") or 0) > int(feed.get("odds_fixtures_requested") or 0):
            warnings.append(
                f"Hay {int(feed.get('matching_fixture_count') or 0)} partidos compatibles con historial y cuotas; "
                f"se consultaron {int(feed.get('odds_fixtures_requested') or 0)} por el límite de cuota de esta búsqueda."
            )
        if total_in_window > 200:
            warnings.append("La ventana contiene más de 200 partidos; se procesan primero los partidos con historial compatible.")
        return {
            "status": "BEST_EFFORT_NOT_VALIDATED" if best_effort_fallback_attempted else "EXPLORATORY_NOT_VALIDATED",
            "window": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "target_odds": target_odds,
            "target_odds_band": {
                "minimum": _parlay_odds_band(target_odds)[0],
                "target": target_odds,
                "maximum": _parlay_odds_band(target_odds)[1],
                "tolerance_percent": PARLAY_ODDS_TOLERANCE_FRACTION * 100.0,
            },
            "matches_in_window": total_in_window,
            "matches_scanned": len(scan_games),
            "matches_with_odds_loaded": len(in_window),
            "matches_analysed": analysed,
            "matches_matching_imported_competitions": len(matching_fixtures),
            "provider_fixtures_in_window": total_in_window,
            "local_games_count": len(all_games),
            "local_team_stats_count": len(all_stats),
            "modelable_competitions": modelable_competition_labels,
            "competition_history_audit": competition_history_audit,
            "filter_diagnostics": filter_diagnostics,
            "imported_competitions": sorted({
                str(g.get("competition") or "").strip()
                for g in all_games if isinstance(g, dict) and str(g.get("competition") or "").strip()
            }),
            "fixture_competitions_scanned": sorted({
                str(x.get("competition") or "").strip() for x in scan_games
            }),
            # Use only fixtures actually returned with odds in this window.
            # feed.available_competitions is the global upcoming catalogue and
            # must not trigger history imports for unrelated competitions.
            "fixture_competitions_in_window": sorted({
                str(x.get("competition") or "").strip() for x in in_window
                if str(x.get("competition") or "").strip()
            })[:40],
            "skipped": skipped,
            "history_skip_examples": history_skip_examples,
            "history_deficit_examples": history_deficit_examples,
            "team_name_skip_examples": team_name_skip_examples,
            "eligible_market_selections": len(strict_candidates),
            "active_market_selections": len(active_candidates),
            "relaxed_candidate_count": max(0, len(all_candidates) - len(strict_candidates)),
            "candidates_before_probability_filter": candidates_before_probability_filter,
            "minimum_estimated_leg_probability": probability_floor,
            "active_minimum_estimated_leg_probability": (
                BEST_EFFORT_MIN_ESTIMATED_LEG_PROBABILITY
                if best_effort_fallback_attempted else MIN_ESTIMATED_LEG_PROBABILITY
            ),
            "best_effort": {
                "requested": best_effort_requested,
                "fallback_attempted": best_effort_fallback_attempted,
                "fallback_found": bool(best_effort_fallback_attempted and best_parlay),
                "minimum_raw_model_probability": BEST_EFFORT_MIN_RAW_MODEL_PROBABILITY,
                "minimum_market_probability": BEST_EFFORT_MIN_MARKET_PROBABILITY,
                "maximum_model_market_gap_pp": BEST_EFFORT_MAX_MODEL_MARKET_GAP * 100.0,
                "minimum_estimated_leg_probability": BEST_EFFORT_MIN_ESTIMATED_LEG_PROBABILITY,
                "minimum_expected_value": BEST_EFFORT_MIN_EXPECTED_VALUE,
            },
            "minimum_history_games": minimum_history,
            "risk_controls": {
                "minimum_estimated_leg_probability": MIN_ESTIMATED_LEG_PROBABILITY,
                "minimum_probability_scope": "each_parlay_leg_individually",
                "minimum_joint_probability_scope": "whole_parlay_product_estimate",
                "minimum_raw_model_probability": MIN_RAW_MODEL_PROBABILITY,
                "minimum_market_probability": MIN_MARKET_PROBABILITY,
                "maximum_model_market_gap_pp": MAX_MODEL_MARKET_GAP * 100.0,
                "minimum_automatic_bookmaker_consensus": MIN_AUTOMATIC_BOOKMAKER_CONSENSUS,
                "minimum_joint_parlay_probability": MIN_PARLAY_JOINT_PROBABILITY,
                "base_uncertainty_haircut_pp": PROBABILITY_UNCERTAINTY_HAIRCUT * 100.0,
                "maximum_dynamic_uncertainty_adjustment_pp": MAX_DYNAMIC_UNCERTAINTY_ADJUSTMENT * 100.0,
                "maximum_market_dispersion_pp": MAX_MARKET_DISPERSION * 100.0,
                "maximum_quote_age_minutes": MAX_QUOTE_AGE_SECONDS / 60.0,
                "uncertainty_haircut_pp": PROBABILITY_UNCERTAINTY_HAIRCUT * 100.0,
                "best_effort_minimum_raw_model_probability": BEST_EFFORT_MIN_RAW_MODEL_PROBABILITY,
                "best_effort_minimum_market_probability": BEST_EFFORT_MIN_MARKET_PROBABILITY,
                "best_effort_maximum_model_market_gap_pp": BEST_EFFORT_MAX_MODEL_MARKET_GAP * 100.0,
                "best_effort_minimum_estimated_leg_probability": BEST_EFFORT_MIN_ESTIMATED_LEG_PROBABILITY,
                "best_effort_minimum_expected_value": BEST_EFFORT_MIN_EXPECTED_VALUE,
            },
            "top_picks": top_picks,
            "best_parlay": best_parlay,
            "warnings": warnings,
            "provider": provider_name,
            "odds_fixtures_requested": feed.get("odds_fixtures_requested") if provider_name == "OddsPapi" else None,
            "odds_fixtures_with_markets": feed.get("odds_fixtures_with_markets") if provider_name == "OddsPapi" else None,
            "quota_remaining_before_scan": feed.get("quota_remaining_before_scan") if provider_name == "OddsPapi" else None,
            "quota_remaining_after_scan": feed.get("quota_remaining_after_scan") if provider_name == "OddsPapi" else None,
            "odds_requests_made": feed.get("odds_requests_made") if provider_name == "OddsPapi" else None,
            "odds_scan_method": feed.get("odds_scan_method") if provider_name == "OddsPapi" else None,
            "provider_warning": feed.get("provider_warning") if provider_name == "OddsPapi" else None,
            "quota_blocked": feed.get("quota_blocked") if provider_name == "OddsPapi" else None,
            "bookmakers_requested": feed.get("bookmakers_requested") if provider_name == "OddsPapi" else None,
            "odds_scan_cap": feed.get("scan_cap") if provider_name == "OddsPapi" else None,
            "odds_competitions_matched": feed.get("matching_competitions") if provider_name == "OddsPapi" else None,
            "odds_markets_catalog_loaded": feed.get("markets_catalog_loaded") if provider_name == "OddsPapi" else None,
            "market_catalogue_audit": feed.get("market_catalogue_audit") if provider_name == "OddsPapi" else None,
            "odds_data_audit": feed.get("odds_data_audit") if provider_name == "OddsPapi" else None,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        from hoopsapi_provider import HoopsAPIError
        if isinstance(exc, HoopsAPIError):
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        try:
            from oddspapi_provider import OddsPapiError
            if isinstance(exc, OddsPapiError):
                raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        except ImportError:
            pass
        raise HTTPException(status_code=500, detail=f"parlay_analysis_failed:{type(exc).__name__}") from exc


# ---------------------------------------------------------------------------
# Walk-forward backtest: measures how often the model's win probabilities were
# right on games that already happened, using only information available
# before each game. It changes no pick logic; it only reports calibration.
# ---------------------------------------------------------------------------

BACKTEST_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
BACKTEST_BUCKET_EDGES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01)


def _wilson_lower_bound(hits: int, n: int, z: float = 1.96) -> float | None:
    if n <= 0:
        return None
    p = hits / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def _summarise_backtest(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"games": 0}
    brier = sum((r["p_home"] - (1.0 if r["home_won"] else 0.0)) ** 2 for r in rows) / n
    base_rate = sum(1 for r in rows if r["home_won"]) / n
    baseline_brier = sum((base_rate - (1.0 if r["home_won"] else 0.0)) ** 2 for r in rows) / n
    log_loss = 0.0
    for r in rows:
        p = min(max(r["p_home"], 1e-4), 1.0 - 1e-4)
        log_loss -= math.log(p if r["home_won"] else 1.0 - p)
    log_loss /= n

    favourites = []
    for r in rows:
        p_fav = max(r["p_home"], 1.0 - r["p_home"])
        fav_won = r["home_won"] if r["p_home"] >= 0.5 else (not r["home_won"])
        favourites.append((p_fav, fav_won))

    thresholds = []
    for threshold in BACKTEST_THRESHOLDS:
        picked = [(p, w) for p, w in favourites if p >= threshold]
        hits = sum(1 for _, w in picked if w)
        count = len(picked)
        thresholds.append({
            "model_probability_at_least": threshold,
            "games": count,
            "hit_rate": hits / count if count else None,
            "hit_rate_95_lower_bound": _wilson_lower_bound(hits, count),
            "mean_predicted_probability": (sum(p for p, _ in picked) / count) if count else None,
        })

    buckets = []
    for low, high in zip(BACKTEST_BUCKET_EDGES[:-1], BACKTEST_BUCKET_EDGES[1:]):
        picked = [(p, w) for p, w in favourites if low <= p < high]
        hits = sum(1 for _, w in picked if w)
        count = len(picked)
        buckets.append({
            "range": f"{low:.2f}-{min(high, 1.0):.2f}",
            "games": count,
            "mean_predicted_probability": (sum(p for p, _ in picked) / count) if count else None,
            "hit_rate": hits / count if count else None,
        })

    total_error = sum(abs(r["projected_total"] - r["actual_total"]) for r in rows) / n
    margin_error = sum(abs(r["projected_margin"] - r["actual_margin"]) for r in rows) / n
    return {
        "games": n,
        "favourite_accuracy": sum(1 for _, w in favourites if w) / n,
        "home_win_rate_in_sample": base_rate,
        "brier_score": brier,
        "brier_score_always_home_baseline": baseline_brier,
        "brier_beats_baseline": brier < baseline_brier,
        "log_loss": log_loss,
        "mean_abs_error_total_points": total_error,
        "mean_abs_error_margin_points": margin_error,
        "by_threshold": thresholds,
        "calibration_buckets": buckets,
    }


def _run_backtest(all_games: list[dict], all_stats: list[dict], simulations: int,
                  minimum_history: int, max_games: int, competition_filter: str,
                  time_budget_seconds: float, only_eligible_leagues: bool = True) -> dict:
    import time as _time
    started = _time.monotonic()
    points_by_event: dict[str, dict[str, float]] = {}
    for row in all_stats:
        if not isinstance(row, dict):
            continue
        event_id = str(row.get("event_id") or "").strip()
        team = str(row.get("team") or "").strip()
        try:
            points = float(row.get("points"))
        except (TypeError, ValueError):
            continue
        if event_id and team and math.isfinite(points):
            points_by_event.setdefault(event_id, {})[team] = points

    candidates: list[tuple[datetime, dict]] = []
    wanted = _normalise_competition(competition_filter) if competition_filter else ""
    for game in all_games:
        if not isinstance(game, dict):
            continue
        if str(game.get("status") or "").strip().lower() not in TERMINAL_STATUSES:
            continue
        if wanted and _normalise_competition(game.get("competition")) != wanted:
            continue
        event_id = str(game.get("event_id") or "").strip()
        home = str(game.get("home_team") or "").strip()
        away = str(game.get("away_team") or "").strip()
        per_team = points_by_event.get(event_id, {})
        if not event_id or home not in per_team or away not in per_team:
            continue
        try:
            played = _historical_datetime(game.get("event_time"), "event_time")
        except ValueError:
            continue
        candidates.append((played, game))
    candidates.sort(key=lambda item: item[0])
    excluded_competitions: list[str] = []
    if only_eligible_leagues and candidates:
        # Do not spend the game budget on leagues that never reach the minimum
        # history (for example NCAA with hundreds of teams).
        latest = candidates[-1][0] + timedelta(hours=25)
        eligible_labels, _ = _modelable_competition_labels(all_games, all_stats, minimum_history, latest)
        eligible_keys = {_normalise_competition(label) for label in eligible_labels}
        kept = [
            item for item in candidates
            if _normalise_competition(item[1].get("competition")) in eligible_keys
        ]
        excluded_competitions = sorted({
            str(item[1].get("competition") or "").strip() for item in candidates
            if _normalise_competition(item[1].get("competition")) not in eligible_keys
        })
        candidates = kept
    candidates = candidates[-max_games:]

    rows: list[dict] = []
    per_competition: dict[str, list[dict]] = {}
    skipped: dict[str, int] = {}
    timed_out = False
    # Newest first: if the time budget runs out, the oldest games are the ones dropped.
    for played, game in reversed(candidates):
        if _time.monotonic() - started > time_budget_seconds:
            timed_out = True
            break
        event_id = str(game.get("event_id")).strip()
        home = str(game.get("home_team")).strip()
        away = str(game.get("away_team")).strip()
        actual_home = points_by_event[event_id][home]
        actual_away = points_by_event[event_id][away]
        if actual_home == actual_away:
            skipped["tied_final_score"] = skipped.get("tied_final_score", 0) + 1
            continue
        fixture = {
            "competition": game.get("competition"),
            "home": home, "away": away,
            "start_time": played.isoformat(),
            "game_id": event_id,
        }
        try:
            history, _, comp, _, h_name, a_name, priors, h_rates, a_rates = _history_scope_for_fixture(
                fixture, all_games, all_stats, played
            )
            if h_rates.games < minimum_history or a_rates.games < minimum_history:
                skipped["historial_insuficiente"] = skipped.get("historial_insuficiente", 0) + 1
                continue
            try:
                adjusted = fit_opponent_adjusted_ratings(
                    history, as_of=played - timedelta(seconds=1),
                    league_ppp=priors.points_per_possession,
                )
            except ValueError:
                adjusted = None
            projection = project_game(h_rates, a_rates, priors, adjusted_ratings=adjusted)
            seed = 20261123 + sum(ord(ch) for ch in event_id)
            simulation = _simulate_game_for_settlement(
                projection, simulations=simulations, seed=seed, overtime_rule="includes_overtime"
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            key = str(exc).split(":")[0][:60]
            skipped[key] = skipped.get(key, 0) + 1
            continue
        wins = sum(1 for h, a in zip(simulation.home_scores, simulation.away_scores) if h > a)
        row = {
            "p_home": wins / simulation.simulation_count,
            "home_won": actual_home > actual_away,
            "projected_total": projection.home_points_mean + projection.away_points_mean,
            "actual_total": actual_home + actual_away,
            "projected_margin": projection.home_points_mean - projection.away_points_mean,
            "actual_margin": actual_home - actual_away,
        }
        rows.append(row)
        per_competition.setdefault(str(comp), []).append(row)

    summary = _summarise_backtest(rows)
    return {
        "status": "WALK_FORWARD_BACKTEST",
        "summary": summary,
        "by_competition": {name: _summarise_backtest(items) for name, items in sorted(per_competition.items())},
        "games_considered": len(candidates),
        "games_evaluated": len(rows),
        "skipped": skipped,
        "timed_out": timed_out,
        "excluded_competitions": excluded_competitions,
        "settings": {
            "simulations": simulations, "minimum_history_games": minimum_history,
            "only_eligible_leagues": only_eligible_leagues,
            "max_games": max_games, "competition": competition_filter or None,
            "settlement": "includes_overtime",
        },
        "notes": [
            "Cada partido se predijo usando solo partidos terminados al menos 24 h antes (sin mirar el futuro).",
            "Mide la probabilidad de victoria del modelo puro, sin mezclarla con cuotas; las cuotas históricas no están guardadas.",
            "Si el acierto real de un tramo es menor que la probabilidad predicha, el modelo está sobreconfiado en ese tramo.",
            "Una muestra chica (menos de unos 100 partidos por tramo) deja mucha incertidumbre; mirá el límite inferior del 95%.",
        ],
    }


@app.post("/api/backtest")
async def backtest(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request_body_must_be_object")
        all_games = payload.get("games")
        all_stats = payload.get("team_stats")
        if not isinstance(all_games, list) or not isinstance(all_stats, list):
            raise ValueError("local_games_and_team_stats_must_be_arrays")
        if len(all_games) > 20000 or len(all_stats) > 40000:
            raise ValueError("too_many_local_rows")
        simulations = int(payload.get("simulations", 2000))
        if not 1000 <= simulations <= 10000:
            raise ValueError("simulations_must_be_between_1000_and_10000")
        minimum_history = int(payload.get("minimum_history_games", MIN_HISTORICAL_GAMES))
        if not 8 <= minimum_history <= 30:
            raise ValueError("minimum_history_games_must_be_between_8_and_30")
        max_games = int(payload.get("max_games", 250))
        if not 20 <= max_games <= 600:
            raise ValueError("max_games_must_be_between_20_and_600")
        competition = str(payload.get("competition") or "").strip()
        only_eligible = bool(payload.get("only_eligible_leagues", True))
        result = await run_in_threadpool(
            _run_backtest, all_games, all_stats, simulations, minimum_history,
            max_games, competition, 85.0, only_eligible,
        )
        result["local_games_count"] = len(all_games)
        result["local_team_stats_count"] = len(all_stats)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"backtest_failed:{type(exc).__name__}") from exc
