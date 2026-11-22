"""OddsPapi v4 adapter for basketball fixtures and pre-match bookmaker odds.

The API key is read only from Render environment variables and is never returned
in API responses or logs. Account status is unmetered; fixtures/markets/odds use
the provider's monthly request quota.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://api.oddspapi.io/v4"
SPORT_ID_BASKETBALL = 11
MAX_ODDS_FIXTURES_PER_RUN = 20
_CACHE: dict[str, tuple[float, object]] = {}
_LOCK = threading.Lock()


class OddsPapiError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _get(path: str, params: dict | None = None, ttl: int = 0, unmetered: bool = False):
    api_key = os.environ.get("ODDSPAPI_API_KEY", "").strip()
    if not api_key:
        raise OddsPapiError("Falta ODDSPAPI_API_KEY en las variables de entorno de Render.", 503)
    query_params = dict(params or {})
    query_params["apiKey"] = api_key
    url = BASE + "/" + path.lstrip("/") + "?" + urlencode(query_params)
    with _LOCK:
        cached = _CACHE.get(url)
        if ttl and cached and cached[0] > time.time():
            return cached[1]
    try:
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        with urlopen(request, timeout=45) as response:
            raw = response.read(20 * 1024 * 1024).decode("utf-8")
            data = json.loads(raw)
    except HTTPError as exc:
        try:
            body = json.loads(exc.read(1024 * 1024).decode("utf-8"))
        except Exception:
            body = {}
        error = body.get("error") if isinstance(body, dict) else {}
        code = str((error or {}).get("code") or body.get("code") or "")
        if exc.code in (401, 403):
            raise OddsPapiError("OddsPapi rechazó la API Key. Revisá ODDSPAPI_API_KEY en Render.", 502) from exc
        if exc.code == 429 and code == "REQUEST_LIMIT_EXCEEDED":
            raise OddsPapiError("Se agotó la cuota mensual de OddsPapi. Consultá el consumo en /api/oddspapi/status.", 429) from exc
        if exc.code == 429:
            raise OddsPapiError("OddsPapi aplicó un límite temporal de solicitudes. Esperá y volvé a intentar.", 429) from exc
        if exc.code == 404:
            raise OddsPapiError("OddsPapi no encontró datos para la consulta solicitada.", 404) from exc
        raise OddsPapiError("OddsPapi respondió HTTP " + str(exc.code) + ".", 502) from exc
    except (URLError, TimeoutError):
        raise OddsPapiError("No se pudo conectar con OddsPapi.") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise OddsPapiError("OddsPapi devolvió una respuesta JSON inválida.") from None
    if ttl:
        with _LOCK:
            _CACHE[url] = (time.time() + ttl, data)
    return data


def account_status() -> dict:
    if not os.environ.get("ODDSPAPI_API_KEY", "").strip():
        return {"configured": False, "provider": "OddsPapi", "connected": False,
                "message": "Falta ODDSPAPI_API_KEY en Render."}
    data = _get("account", ttl=45, unmetered=True)
    subscriptions = data.get("subscriptions") if isinstance(data, dict) else None
    active = next((x for x in (subscriptions or []) if isinstance(x, dict) and x.get("is_active")), None)
    if not active and subscriptions:
        active = subscriptions[0]
    if not isinstance(active, dict):
        return {"configured": True, "provider": "OddsPapi", "connected": True,
                "subscription_active": False, "message": "La clave responde, pero no se encontró una suscripción activa."}
    bookmakers = active.get("bookmakers") or {}
    return {
        "configured": True,
        "provider": "OddsPapi",
        "connected": True,
        "subscription_active": bool(active.get("is_active")),
        "request_limit": active.get("request_limit"),
        "request_count": active.get("request_count"),
        "requests_remaining": max(0, int(active.get("request_limit") or 0) - int(active.get("request_count") or 0)),
        "bookmakers_available_in_plan": sorted(bookmakers.keys()) if isinstance(bookmakers, dict) else [],
        "basketball_included": SPORT_ID_BASKETBALL in (active.get("sport_ids") or []),
        "note": "La API Key no se muestra. /account no consume cuota; fixtures, markets y odds sí consumen solicitudes.",
    }


def _parse_label(raw: str) -> tuple[str, float | None] | None:
    value = str(raw or "").strip().lower().replace(",", ".")
    if value in {"home", "1", "participant1", "team1"}:
        return "home", None
    if value in {"away", "2", "participant2", "team2"}:
        return "away", None
    if value in {"over", "o"}:
        return "over", None
    if value in {"under", "u"}:
        return "under", None
    match = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*/\s*(home|away|1|2|over|under)$", value)
    if match:
        number = float(match.group(1))
        side = match.group(2)
        return ({"1": "home", "2": "away"}.get(side, side), number)
    match = re.match(r"^(home|away|over|under)\s*/\s*([+-]?\d+(?:\.\d+)?)$", value)
    if match:
        return match.group(1), float(match.group(2))
    return None


def _market_family(name: str) -> str | None:
    label = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    if "winner" in label and ("overtime" in label or "incl ot" in label):
        return "moneyline"
    if "over under" in label and ("overtime" in label or "incl ot" in label):
        return "total"
    if "handicap" in label and ("overtime" in label or "incl ot" in label):
        return "spread"
    return None


def _market_metadata() -> dict[str, str]:
    data = _get("markets", {"language": "en"}, ttl=86400)
    if not isinstance(data, list):
        raise OddsPapiError("El catálogo de mercados de OddsPapi tiene un formato inesperado.")
    return {
        str(row.get("marketId")): family
        for row in data if isinstance(row, dict)
        if int(row.get("sportId") or 0) == SPORT_ID_BASKETBALL
        and (family := _market_family(row.get("marketName"))) is not None
        and str(row.get("period") or "").lower() in {"fulltime", "full_time", "match", ""}
    }


def _fresh_timestamp(value: str | None, now: datetime) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = (now - parsed.astimezone(timezone.utc)).total_seconds()
        return -300 <= age <= 3600
    except (TypeError, ValueError):
        return False


def _normalise_odds_fixture(fixture: dict, odds: dict, market_families: dict[str, str], now: datetime) -> dict:
    home = str(fixture.get("participant1Name") or "").strip()
    away = str(fixture.get("participant2Name") or "").strip()
    groups: dict[tuple, dict[str, dict]] = {}
    boards = (odds or {}).get("bookmakerOdds") or {}
    for bookmaker, board in boards.items():
        if not isinstance(board, dict) or not board.get("bookmakerIsActive", True) or board.get("suspended"):
            continue
        markets = board.get("markets") or {}
        for market_id, market in markets.items():
            family = market_families.get(str(market_id))
            if not family or not isinstance(market, dict) or not market.get("marketActive", True):
                continue
            for outcome_id, outcome in (market.get("outcomes") or {}).items():
                players = (outcome or {}).get("players") or {}
                player = players.get("0")
                if not isinstance(player, dict) or not player.get("active", True):
                    continue
                price = player.get("price")
                try:
                    price = float(price)
                except (TypeError, ValueError):
                    continue
                if not (1.0001 < price < 1000):
                    continue
                changed = player.get("changedAt") or player.get("bookmakerChangedAt") or (odds or {}).get("updatedAt")
                if not _fresh_timestamp(changed, now):
                    continue
                parsed = _parse_label(player.get("bookmakerOutcomeId"))
                if parsed is None:
                    continue
                side, raw_line = parsed
                if family == "moneyline":
                    if side not in {"home", "away"}:
                        continue
                    line = None
                elif family == "total":
                    if side not in {"over", "under"} or raw_line is None:
                        continue
                    line = round(raw_line, 2)
                else:
                    if side not in {"home", "away"} or raw_line is None:
                        continue
                    # Normalize every handicap to the home-team line.
                    line = round(raw_line if side == "home" else -raw_line, 2)
                key = (family, line)
                groups.setdefault(key, {}).setdefault(str(bookmaker), {})[side] = {
                    "price": price, "captured_at": changed
                }

    ft: dict[str, dict] = {}
    for (family, line), by_book in groups.items():
        api_name = {"moneyline": "moneyline", "spread": "spread", "total": "total"}[family]
        for bookmaker, sides in by_book.items():
            required = ("home", "away") if family in {"moneyline", "spread"} else ("over", "under")
            if not all(side in sides for side in required):
                continue
            captured = min(sides[x]["captured_at"] for x in required)
            entry = {"line": line, required[0]: sides[required[0]]["price"],
                     required[1]: sides[required[1]]["price"], "captured_at": captured}
            ft.setdefault(bookmaker, {}).setdefault(api_name, []).append(entry)
    periods: dict[str, dict] = {"ft": ft}
    providers: dict[str, dict] = {
        bookmaker: {"captured_at": max(
            (entry.get("captured_at") for rows in markets.values() for entry in rows),
            default=None,
        )}
        for bookmaker, markets in ft.items()
    }

    return {
        "game_id": str(fixture.get("fixtureId") or ""),
        "home": home,
        "away": away,
        "competition": str(fixture.get("tournamentName") or fixture.get("categoryName") or "").strip(),
        "start_time": fixture.get("startTime"),
        "updated_at": fixture.get("updatedAt"),
        "periods": periods,
        "providers": providers,
        "oddspapi_fixture_id": fixture.get("fixtureId"),
        "settlement_scope": "includes_overtime",
    }


def games(window_start: datetime, window_end: datetime) -> dict:
    if (window_end - window_start) > timedelta(days=7):
        raise OddsPapiError("La ventana máxima de OddsPapi es de 7 días.", 422)
    # The fixtures endpoint accepts a basketball-only date range shorter than 10 days.
    fixtures = _get("fixtures", {
        "sportId": SPORT_ID_BASKETBALL,
        "from": window_start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": window_end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "statusId": 0,
    }, ttl=90)
    if not isinstance(fixtures, list):
        raise OddsPapiError("OddsPapi devolvió un calendario con formato inesperado.")
    now = datetime.now(timezone.utc)
    upcoming = [
        x for x in fixtures if isinstance(x, dict)
        and str(x.get("startTime") or "")
        and x.get("statusId") == 0
    ]
    upcoming.sort(key=lambda x: str(x.get("startTime") or ""))
    market_families = _market_metadata()
    account = account_status()
    remaining = int(account.get("requests_remaining") or 0)
    # Reserve one request for /fixtures and one for /markets. Avoid consuming the
    # entire free quota on a single scan.
    odds_limit = min(MAX_ODDS_FIXTURES_PER_RUN, max(0, remaining - 2))
    selected = [x for x in upcoming if x.get("hasOdds")][:odds_limit]
    normalized = []
    for fixture in selected:
        odds = _get("odds", {"fixtureId": fixture.get("fixtureId"), "language": "en", "verbosity": 3}, ttl=75)
        if isinstance(odds, dict):
            normalized.append(_normalise_odds_fixture(fixture, odds, market_families, now))
        time.sleep(0.55)
    return {
        "count": len(upcoming),
        "games": normalized,
        "odds_fixtures_requested": len(selected),
        "odds_fixtures_with_markets": sum(bool(x.get("periods", {}).get("ft")) for x in normalized),
        "quota_remaining_before_scan": remaining,
        "scan_cap": MAX_ODDS_FIXTURES_PER_RUN,
    }
