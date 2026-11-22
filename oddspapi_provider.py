"""OddsPapi v4 adapter for basketball fixtures and pre-match bookmaker odds.

The API key is read only from Render environment variables and is never returned
in API responses or logs. Account status is unmetered; fixtures/markets/odds use
the provider's monthly request quota.
"""
from __future__ import annotations

import json
import codecs
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


def _read_basketball_market_array(response, max_bytes: int = 128 * 1024 * 1024) -> list[dict]:
    """Stream the global /markets JSON array and retain only basketball rows.

    The provider documents /markets as a global catalogue (no sportId filter).
    Keeping the whole catalogue in memory is unnecessary and a fixed 20 MB read
    truncates larger valid responses.
    """
    text_decoder = codecs.getincrementaldecoder("utf-8-sig")()
    json_decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    total_bytes = 0
    started = False
    expect_value = True
    allow_empty_end = True
    closed = False
    selected: list[dict] = []
    basketball_rows = 0
    main_family_rows = 0
    period_counts: dict[str, int] = {}

    def consume(final: bool = False) -> None:
        nonlocal buffer, pos, started, expect_value, allow_empty_end, closed
        nonlocal basketball_rows, main_family_rows
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos >= len(buffer):
                break
            if not started:
                if buffer[pos] != "[":
                    raise OddsPapiError("OddsPapi devolvió un catálogo /markets que no es un array JSON.")
                started = True
                pos += 1
                continue
            if closed:
                if buffer[pos:].strip():
                    raise OddsPapiError("OddsPapi agregó contenido después del array JSON de /markets.")
                pos = len(buffer)
                break
            if expect_value:
                if buffer[pos] == "]":
                    if not allow_empty_end:
                        raise OddsPapiError("OddsPapi devolvió un array /markets con una coma final.")
                    closed = True
                    pos += 1
                    continue
                try:
                    item, end = json_decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if final:
                        raise OddsPapiError("OddsPapi devolvió un array /markets incompleto o JSON inválido.") from None
                    break
                if not isinstance(item, dict):
                    raise OddsPapiError("OddsPapi devolvió un elemento inesperado en /markets.")
                if str(item.get("sportId") or "") == str(SPORT_ID_BASKETBALL):
                    basketball_rows += 1
                    market_name = str(item.get("marketName") or "")
                    period = str(item.get("period") or "")
                    period_key = period.strip().lower() or "(vacío)"
                    period_counts[period_key] = period_counts.get(period_key, 0) + 1
                    family = _market_family(market_name)
                    if family:
                        main_family_rows += 1
                    # OddsPapi's period labels are not consistent across all
                    # basketball market rows. Trust the full-game market name,
                    # but reject any period explicitly marked as a half/quarter.
                    if family and _is_full_game_period(period):
                        selected.append({
                            "sportId": SPORT_ID_BASKETBALL,
                            "marketId": item.get("marketId"),
                            "marketName": market_name,
                            "period": period,
                        })
                pos = end
                expect_value = False
                allow_empty_end = False
            else:
                if buffer[pos] == ",":
                    pos += 1
                    expect_value = True
                    allow_empty_end = False
                elif buffer[pos] == "]":
                    closed = True
                    pos += 1
                else:
                    raise OddsPapiError("OddsPapi devolvió separadores inválidos en el array /markets.")
        if pos > 65536:
            buffer = buffer[pos:]
            pos = 0

    while True:
        chunk = response.read(65536)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise OddsPapiError(
                "El catálogo global /markets de OddsPapi supera el límite seguro de lectura (128 MB)."
            )
        try:
            buffer += text_decoder.decode(chunk)
        except UnicodeDecodeError:
            raise OddsPapiError("OddsPapi devolvió texto UTF-8 inválido en /markets.") from None
        consume()

    try:
        buffer += text_decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        raise OddsPapiError("OddsPapi devolvió texto UTF-8 incompleto en /markets.") from None
    consume(final=True)
    if not started or not closed:
        raise OddsPapiError("OddsPapi cerró la respuesta /markets antes de completar el array JSON.")
    if not selected:
        sample_periods = ", ".join(
            f"{name}:{count}" for name, count in sorted(period_counts.items(), key=lambda x: -x[1])[:8]
        ) or "ninguno"
        raise OddsPapiError(
            "No se pudieron identificar mercados principales de básquet en /markets "
            f"(filas sportId=11: {basketball_rows}; nombres principales reconocidos: "
            f"{main_family_rows}; períodos observados: {sample_periods})."
        )
    return selected


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
    endpoint = "/" + path.lstrip("/")
    try:
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        with urlopen(request, timeout=60) as response:
            response_status = getattr(response, "status", 200)
            content_type = response.headers.get("Content-Type", "no informado")
            if path.strip("/") == "markets":
                data = _read_basketball_market_array(response)
            else:
                raw_bytes = response.read(32 * 1024 * 1024 + 1)
                if len(raw_bytes) > 32 * 1024 * 1024:
                    raise OddsPapiError(
                        f"OddsPapi devolvió una respuesta demasiado grande en {endpoint} "
                        f"(HTTP {response_status}; Content-Type {content_type}; límite 32 MB)."
                    )
                try:
                    raw = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    raise OddsPapiError(
                        f"OddsPapi devolvió una codificación no válida en {endpoint} "
                        f"(HTTP {response_status}; Content-Type {content_type})."
                    ) from None
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    # Do not include the URL or response body here: the API key is a query parameter.
                    raise OddsPapiError(
                        f"OddsPapi devolvió contenido que no es JSON en {endpoint} "
                        f"(HTTP {response_status}; Content-Type {content_type}; "
                        f"{len(raw_bytes)} bytes)."
                    ) from None
    except HTTPError as exc:
        try:
            body = json.loads(exc.read(1024 * 1024).decode("utf-8-sig"))
        except Exception:
            body = {}
        body_obj = body if isinstance(body, dict) else {}
        error_obj = body_obj.get("error")
        error_obj = error_obj if isinstance(error_obj, dict) else {}
        code = str(error_obj.get("code") or body_obj.get("code") or "")
        provider_message = str(error_obj.get("message") or "")
        if exc.code in (401, 403):
            raise OddsPapiError("OddsPapi rechazó la API Key. Revisá ODDSPAPI_API_KEY en Render.", 502) from exc
        if exc.code == 429 and code == "REQUEST_LIMIT_EXCEEDED":
            raise OddsPapiError("Se agotó la cuota mensual de OddsPapi. Consultá el consumo en /api/oddspapi/status.", 429) from exc
        if exc.code == 429:
            raise OddsPapiError("OddsPapi aplicó un límite temporal de solicitudes. Esperá y volvé a intentar.", 429) from exc
        if exc.code == 404:
            raise OddsPapiError("OddsPapi no encontró datos para la consulta solicitada.", 404) from exc
        safe_message = f" {provider_message}" if provider_message else ""
        raise OddsPapiError(
            f"OddsPapi respondió HTTP {exc.code} en {endpoint}.{safe_message}", 502
        ) from exc
    except (URLError, TimeoutError):
        raise OddsPapiError(f"No se pudo conectar con OddsPapi en {endpoint}.") from None
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


def _is_full_game_period(value: str) -> bool:
    period = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    if not period:
        return True
    # Explicitly reject quarter/half markets, including common short labels.
    if any(token in period for token in ("half", "quarter", "1h", "2h", "1q", "2q", "3q", "4q")):
        return False
    if period in {"h1", "h2", "q1", "q2", "q3", "q4", "1st", "2nd", "3rd", "4th"}:
        return False
    # Accept documented labels plus provider-specific full-game aliases. The
    # market family name is independently checked before this helper is used.
    return True


def _market_family(name: str) -> str | None:
    label = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    # Names for half/quarter markets sometimes also say "incl. overtime";
    # never mistake those for full-game match markets.
    period_markers = (
        "first half", "second half", "first quarter", "second quarter",
        "third quarter", "fourth quarter", "1st half", "2nd half",
        "1st quarter", "2nd quarter", "3rd quarter", "4th quarter",
        "first period", "second period", "quarter", "half",
    )
    if any(marker in label for marker in period_markers):
        return None
    includes_overtime = any(token in label for token in (
        "overtime", "incl ot", "extra time", "extra period",
    ))
    if "winner" in label and includes_overtime:
        return "moneyline"
    if "over under" in label and includes_overtime:
        return "total"
    if "handicap" in label and includes_overtime:
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
        and _is_full_game_period(str(row.get("period") or ""))
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
