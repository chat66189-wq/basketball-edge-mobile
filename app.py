from __future__ import annotations

import math
import os
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
def sportapi_sync(league: str, season: str | None = None, date: str | None = None, limit: int = 20):
    try:
        from sportapi_provider import sync
        return sync(league, season, date, limit)
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



def _normalise_label(value: Any) -> str:
    raw = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "", raw)


# Exact HoopsAPI-to-imported-history aliases observed for EuroLeague fixtures.
# Targets are checked against the imported team lookup before they are used.
_EUROLEAGUE_TEAM_ALIASES = {
    "besiktasjk": "besiktasistanbul",
    "paris": "parisbasketball",
    "crvenazvezda": "crvenazvezdameridianbetbelgrade",
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
            return label[:-len(suffix)]
    return label


def _fixture_team_lookup_key(value: Any, competition: str, imported_lookup: dict[str, str]) -> str:
    key = _normalise_team_label(value)
    if key in imported_lookup:
        return key
    if competition == "euroleague":
        alias_target = _EUROLEAGUE_TEAM_ALIASES.get(key)
        if alias_target and alias_target in imported_lookup:
            return alias_target
    # Provider labels occasionally differ by a short suffix or abbreviation.
    # Resolve only a very close, clearly unique match; otherwise abstain.
    scored = sorted(
        ((SequenceMatcher(None, key, candidate).ratio(), candidate) for candidate in imported_lookup),
        reverse=True,
    )
    if scored and scored[0][0] >= 0.92:
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if scored[0][0] - second_score >= 0.05:
            return scored[0][1]
    return key


def _normalise_competition(value: Any) -> str:
    """Normalize provider competition labels without merging distinct leagues or genders."""
    label = _normalise_label(value)
    if not label:
        return ""

    # WNBA is intrinsically a women's competition, with or without that suffix.
    if "wnba" in label:
        return "wnba"

    gender = "women" if ("women" in label or "female" in label) else ""

    if "euroleague" in label:
        return "euroleague" + gender
    if "eurocup" in label:
        return "eurocup" + gender

    # NBA provider labels may include "men", "basketball", or country descriptors.
    if label in {"nba", "nbamen", "nationalbasketballassociation", "nationalbasketballassociationmen"}:
        return "nba"
    if label.startswith("nba") and not label.startswith(("nbagleague", "nbawomen")):
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
        if age < -300 or age > 3600:
            continue
        for model_market, api_market, sides, has_line in specs:
            item = markets.get(api_market)
            if not isinstance(item, dict):
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
                "captured_at": str(captured),
                "price_a": price_a,
                "price_b": price_b,
                "side_a": sides[0],
                "side_b": sides[1],
            })
    return groups


def _score_fixture(fixture: dict, all_games: list[dict], all_stats: list[dict],
                   simulations: int, minimum_history: int, now: datetime) -> tuple[list[dict], str | None]:
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
    simulation = simulate_game(projection, simulations=simulations, seed=seed)

    groups = _fixture_market_groups(fixture, now)
    output = []
    shrink = min(0.65, min(home_rates.games, away_rates.games) / 20.0)
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
            continue
        sides = ("home", "away") if market in ("moneyline", "spread") else ("over", "under")
        fair = {side: sum(x[side] for x in fair_samples) / len(fair_samples) for side in sides}
        for side in sides:
            try:
                priced = price_game_market(
                    simulation, market, side=side, line=line, overtime_rule="excluded"
                )
                raw_win = float(priced.p_win)
                raw_loss = float(priced.p_loss)
                push = float(priced.p_push)
            except (ValueError, TypeError, AttributeError):
                continue
            no_push_total = raw_win + raw_loss
            if no_push_total <= 0:
                continue
            raw_conditional = raw_win / no_push_total
            # Damp the unvalidated model signal toward the no-vig market consensus.
            blended = fair[side] + shrink * (raw_conditional - fair[side])
            if push > 0.005:
                continue
            for quote in quotes:
                if side == quote["side_a"]:
                    price = quote["price_a"]
                elif side == quote["side_b"]:
                    price = quote["price_b"]
                else:
                    continue
                ev = blended * price - 1.0
                edge = blended - fair[side]
                if edge <= 0.005 or ev <= 0.005:
                    continue
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
                    "probability": blended,
                    "raw_model_probability": raw_conditional,
                    "market_probability": fair[side],
                    "edge_percentage_points": edge * 100.0,
                    "expected_value_per_unit": ev,
                    "push_probability": push,
                    "home_history_games": home_rates.games,
                    "away_history_games": away_rates.games,
                    "historical_games_used": counters.get("games_used", 0),
                    "model_version": projection.model_version,
                    "rating_method": projection.rating_method,
                    "odds_captured_at": quote["captured_at"],
                    "score": edge * 0.6 + min(ev, 0.5) * 0.4,
                })
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
        simulations = int(payload.get("simulations", 10000))
        if not 1000 <= simulations <= 50000:
            raise ValueError("simulations_must_be_between_1000_and_50000")
        minimum_history = int(payload.get("minimum_history_games", 8))
        if not 3 <= minimum_history <= 30:
            raise ValueError("minimum_history_games_must_be_between_3_and_30")
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
            "imported_competitions": sorted({
                str(g.get("competition") or "").strip()
                for g in all_games if isinstance(g, dict) and str(g.get("competition") or "").strip()
            }),
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


@app.get("/api/hoopsapi/status")
def hoopsapi_status():
    from hoopsapi_provider import status
    return status()


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
        simulations = int(payload.get("simulations", 10000))
        if not 1000 <= simulations <= 50000:
            raise ValueError("simulations_must_be_between_1000_and_50000")
        minimum_history = int(payload.get("minimum_history_games", 8))
        if not 3 <= minimum_history <= 30:
            raise ValueError("minimum_history_games_must_be_between_3_and_30")
        all_games = payload.get("games")
        all_stats = payload.get("team_stats")
        if not isinstance(all_games, list) or not isinstance(all_stats, list):
            raise ValueError("local_games_and_team_stats_must_be_arrays")
        if len(all_games) > 20000 or len(all_stats) > 40000:
            raise ValueError("too_many_local_rows")

        from hoopsapi_provider import games as fetch_odds_games
        feed = fetch_odds_games()
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
        total_in_window = len(in_window)

        # Prioritize fixtures whose competition is actually represented in the
        # user's imported local history before applying the 50-game processing cap.
        imported_competitions = {
            _normalise_competition(g.get("competition"))
            for g in all_games
            if isinstance(g, dict) and str(g.get("competition") or "").strip()
        }
        matching_fixtures = [
            x for x in in_window
            if _normalise_competition(x.get("competition")) in imported_competitions
        ]
        scan_pool = matching_fixtures if matching_fixtures else in_window
        scan_games = scan_pool[:50]

        all_candidates = []
        skipped: dict[str, int] = {}
        history_skip_examples: list[str] = []
        team_name_skip_examples: list[str] = []
        analysed = 0
        for fixture in scan_games:
            if not fixture.get("home") or not fixture.get("away"):
                skipped["fixture_fields_missing"] = skipped.get("fixture_fields_missing", 0) + 1
                continue
            try:
                candidates, reason = _score_fixture(
                    fixture, all_games, all_stats, simulations, minimum_history, now
                )
                if reason:
                    skipped[reason.split(":")[0]] = skipped.get(reason.split(":")[0], 0) + 1
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

        # Best displayed price for each distinct selection; keep the bookmaker alongside it.
        best_by_selection = {}
        for candidate in all_candidates:
            key = (candidate["event_id"], candidate["market"], candidate["side"], candidate["line"])
            previous = best_by_selection.get(key)
            if previous is None or candidate["odds"] > previous["odds"]:
                best_by_selection[key] = candidate
        best_by_game = {}
        for candidate in best_by_selection.values():
            previous = best_by_game.get(candidate["event_id"])
            if previous is None or (candidate["score"], candidate["probability"]) > (previous["score"], previous["probability"]):
                best_by_game[candidate["event_id"]] = candidate
        top_picks = sorted(best_by_game.values(),
                           key=lambda x: (x["score"], x["probability"]), reverse=True)[:12]

        # A real single-bookmaker accumulator needs every leg at the same operator.
        by_bookmaker: dict[str, list[dict]] = {}
        for candidate in all_candidates:
            by_bookmaker.setdefault(candidate["bookmaker"], []).append(candidate)
        best_parlay = None
        for bookmaker, items in by_bookmaker.items():
            per_game = {}
            for item in items:
                old = per_game.get(item["event_id"])
                if old is None or item["score"] > old["score"]:
                    per_game[item["event_id"]] = item
            pool = sorted(per_game.values(), key=lambda x: x["score"], reverse=True)[:18]
            local_best = None
            for leg_count in range(1, min(8, len(pool)) + 1):
                for combo in combinations(pool, leg_count):
                    combined_odds = math.prod(x["odds"] for x in combo)
                    if combined_odds + 1e-9 < target_odds:
                        continue
                    hit_probability = math.prod(x["probability"] for x in combo)
                    overshoot = combined_odds - target_odds
                    rank = (hit_probability, -overshoot, -leg_count)
                    if local_best is None or rank > local_best["rank"]:
                        local_best = {
                            "rank": rank,
                            "bookmaker": bookmaker,
                            "combined_odds": combined_odds,
                            "estimated_hit_probability": hit_probability,
                            "legs": list(combo),
                        }
            if local_best and (best_parlay is None or local_best["rank"] > best_parlay["rank"]):
                best_parlay = local_best

        if best_parlay:
            best_parlay.pop("rank", None)
            best_parlay["status"] = "EXPLORATORY_NOT_VALIDATED"
        warnings = [
            "EXPLORATORY_NOT_VALIDATED: el modelo no tiene validación walk-forward/holdout aprobada.",
            "El modelo base simula tiempo reglamentario; las cuotas FT de HoopsAPI incluyen prórroga. La comparación de probabilidades no es perfectamente equivalente.",
            "La probabilidad combinada multiplica las probabilidades de las piernas y supone independencia aproximada entre partidos.",
            "Solo se analizan competiciones y equipos con estadísticas históricas locales suficientes; no se inventan datos faltantes.",
            "El historial usa una ventana móvil de hasta 400 días de la misma competición e incluye temporadas anteriores; los cambios de plantel no están modelados.",
            "Las cuotas se filtran por captura de hasta 60 minutos y pueden cambiar antes de apostar.",
        ]
        if total_in_window > 50:
            warnings.append("La ventana contiene más de 50 partidos; se analizaron los primeros 50 por hora de inicio.")
        return {
            "status": "EXPLORATORY_NOT_VALIDATED",
            "window": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "target_odds": target_odds,
            "matches_in_window": total_in_window,
            "matches_scanned": len(scan_games),
            "matches_analysed": analysed,
            "matches_matching_imported_competitions": len(matching_fixtures),
            "local_games_count": len(all_games),
            "local_team_stats_count": len(all_stats),
            "imported_competitions": sorted({
                str(g.get("competition") or "").strip()
                for g in all_games if isinstance(g, dict) and str(g.get("competition") or "").strip()
            }),
            "fixture_competitions_scanned": sorted({
                str(x.get("competition") or "").strip() for x in scan_games
            }),
            "fixture_competitions_in_window": sorted({
                str(x.get("competition") or "").strip() for x in in_window
                if str(x.get("competition") or "").strip()
            })[:40],
            "skipped": skipped,
            "history_skip_examples": history_skip_examples,
            "team_name_skip_examples": team_name_skip_examples,
            "eligible_market_selections": len(all_candidates),
            "top_picks": top_picks,
            "best_parlay": best_parlay,
            "warnings": warnings,
            "provider": "HoopsAPI",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        from hoopsapi_provider import HoopsAPIError
        if isinstance(exc, HoopsAPIError):
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=f"parlay_analysis_failed:{type(exc).__name__}") from exc
