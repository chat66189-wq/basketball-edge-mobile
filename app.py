from __future__ import annotations

import math
import os
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

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
    TeamGame, LeaguePriors, fit_team_rates, fit_opponent_adjusted_ratings,
    project_game, simulate_game,
)
from scripts.edge_v15.market_pricing import price_game_market, ev_per_unit  # noqa: E402

app = FastAPI(
    title="Basketball Edge Analyst v20 API",
    description="Adaptador web para el motor de proyección y simulación BEA. Las salidas son exploratorias, no picks validados.",
    version="20.1.0-mobile-bridge",
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
            played = _aware_datetime(game.get("event_time"), "event_time")
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
        "note": "No sports provider, live odds feed, trained exact-scope model card, or approved pick gate is configured.",
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
