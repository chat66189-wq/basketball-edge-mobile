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
import unicodedata
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://api.oddspapi.io/v4"
SPORT_ID_BASKETBALL = 11
MAX_ODDS_FIXTURES_PER_RUN = 8
MAX_BOOKMAKERS_PER_FIXTURE = 8
# Live OddsPapi tournament IDs for competitions already present in the mobile
# app's local history. Names remain a fallback for other user-imported leagues.
TOURNAMENT_IDS_BY_COMPETITION = {
    "nba": {"132"},
    "euroleague": {"138"},
    "ncaadivision1women": {"21384"},
}
ODDS_RESPONSE_MAX_BYTES = 8 * 1024 * 1024
OTHER_RESPONSE_MAX_BYTES = 16 * 1024 * 1024
BOOKMAKER_PRIORITY = (
    "bet365.bet.ar", "bet365", "betano", "pinnacle",
    "betsson", "1xbet", "betplay", "apuestatotal",
    "bwin", "888sport", "22bet", "18bet", "marathonbet",
    "unibet", "williamhill", "betfair-ex",
)
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
                response_limit = (
                    ODDS_RESPONSE_MAX_BYTES
                    if path.strip("/") == "odds"
                    else OTHER_RESPONSE_MAX_BYTES
                )
                raw_bytes = response.read(response_limit + 1)
                if len(raw_bytes) > response_limit:
                    limit_mb = response_limit // (1024 * 1024)
                    raise OddsPapiError(
                        f"OddsPapi devolvió una respuesta demasiado grande en {endpoint} "
                        f"(HTTP {response_status}; Content-Type {content_type}; "
                        f"límite seguro {limit_mb} MB). Se redujeron las casas consultadas; "
                        "no se reinició el servidor ni se procesó una respuesta incompleta."
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
    value = re.sub(r"\s+", " ", str(raw or "").strip().lower().replace(",", "."))
    simple = {
        "home": ("home", None), "1": ("home", None),
        "participant1": ("home", None), "participant 1": ("home", None),
        "team1": ("home", None), "team 1": ("home", None),
        "home team": ("home", None),
        "away": ("away", None), "2": ("away", None),
        "participant2": ("away", None), "participant 2": ("away", None),
        "team2": ("away", None), "team 2": ("away", None),
        "away team": ("away", None),
        "over": ("over", None), "o": ("over", None),
        "under": ("under", None), "u": ("under", None),
    }
    if value in simple:
        return simple[value]
    match = re.match(r"^(over|under)\s*([+-]?\d+(?:\.\d+)?)$", value)
    if match:
        return match.group(1), float(match.group(2))
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


def _market_metadata() -> dict[str, dict]:
    """Read market family, line and canonical outcome names from OddsPapi's catalogue."""
    data = _get("markets", {"language": "en"}, ttl=86400)
    if not isinstance(data, list):
        raise OddsPapiError("El catálogo de mercados de OddsPapi tiene un formato inesperado.")
    result: dict[str, dict] = {}
    for row in data:
        if not isinstance(row, dict) or int(row.get("sportId") or 0) != SPORT_ID_BASKETBALL:
            continue
        family = _market_family(row.get("marketName"))
        if not family or not _is_full_game_period(str(row.get("period") or "")):
            continue
        raw_handicap = row.get("handicap")
        try:
            handicap = float(raw_handicap) if raw_handicap not in (None, "") else None
        except (TypeError, ValueError):
            handicap = None
        outcomes: dict[str, str] = {}
        raw_outcomes = row.get("outcomes") or []
        if isinstance(raw_outcomes, list):
            for outcome in raw_outcomes:
                if isinstance(outcome, dict) and outcome.get("outcomeId") is not None:
                    outcomes[str(outcome["outcomeId"])] = str(outcome.get("outcomeName") or "")
        elif isinstance(raw_outcomes, dict):
            for outcome_id, outcome in raw_outcomes.items():
                if isinstance(outcome, dict):
                    outcomes[str(outcome_id)] = str(outcome.get("outcomeName") or "")
                elif outcome is not None:
                    outcomes[str(outcome_id)] = str(outcome)
        result[str(row.get("marketId"))] = {
            "family": family,
            "handicap": handicap,
            "outcomes": outcomes,
        }
    return result


def market_catalogue_audit() -> dict:
    """Return a safe, quota-light diagnostic of full-game basketball market definitions."""
    data = _get("markets", {"language": "en"}, ttl=86400)
    if not isinstance(data, list):
        raise OddsPapiError("El catálogo de mercados de OddsPapi tiene un formato inesperado.")
    audit = {
        "sport_id": SPORT_ID_BASKETBALL,
        "catalogue_rows_received": len(data),
        "families": {},
        "catalogue_fields": [],
    }
    for row in data:
        if not isinstance(row, dict) or int(row.get("sportId") or 0) != SPORT_ID_BASKETBALL:
            continue
        family = _market_family(row.get("marketName"))
        if not family or not _is_full_game_period(str(row.get("period") or "")):
            continue
        bucket = audit["families"].setdefault(family, {
            "definitions": 0, "with_handicap": 0, "without_handicap": 0,
            "periods": {}, "samples": [],
        })
        bucket["definitions"] += 1
        raw_handicap = row.get("handicap")
        try:
            handicap = float(raw_handicap) if raw_handicap not in (None, "") else None
        except (TypeError, ValueError):
            handicap = None
        if handicap is None:
            bucket["without_handicap"] += 1
        else:
            bucket["with_handicap"] += 1
        period = str(row.get("period") or "")
        bucket["periods"][period] = bucket["periods"].get(period, 0) + 1
        if len(bucket["samples"]) < 8:
            outcomes = row.get("outcomes") or []
            if isinstance(outcomes, list):
                outcome_names = [
                    str(x.get("outcomeName") or "")
                    for x in outcomes if isinstance(x, dict)
                ]
            elif isinstance(outcomes, dict):
                outcome_names = [
                    str(x.get("outcomeName") or x or "")
                    for x in outcomes.values()
                ]
            else:
                outcome_names = []
            bucket["samples"].append({
                "market_id": str(row.get("marketId") or ""),
                "market_name": str(row.get("marketName") or ""),
                "period": period,
                "handicap_raw": None if raw_handicap is None else str(raw_handicap),
                "handicap_parsed": handicap,
                "market_type": str(row.get("marketType") or ""),
                "outcomes": outcome_names[:4],
            })
        if not audit["catalogue_fields"]:
            audit["catalogue_fields"] = sorted(str(k) for k in row.keys())
    return audit

def _select_bookmakers(available_bookmakers: list[str]) -> list[str]:
    """Choose a small, deterministic set of subscribed sportsbook slugs."""
    exact = {
        str(name).strip().lower(): str(name).strip()
        for name in available_bookmakers
        if str(name).strip()
    }
    chosen: list[str] = []
    for preferred in BOOKMAKER_PRIORITY:
        slug = exact.get(preferred.lower())
        if slug and slug not in chosen:
            chosen.append(slug)
        if len(chosen) >= MAX_BOOKMAKERS_PER_FIXTURE:
            break
    if len(chosen) < MAX_BOOKMAKERS_PER_FIXTURE:
        # Fill from the user's actual subscription, preferring sportsbook
        # slugs and excluding obvious casino-only entries.
        remainder = sorted(
            (
                slug for slug in exact.values()
                if slug not in chosen
                and "casino" not in slug.lower()
            ),
            key=str.lower,
        )
        chosen.extend(remainder[:MAX_BOOKMAKERS_PER_FIXTURE - len(chosen)])
    return chosen[:MAX_BOOKMAKERS_PER_FIXTURE]


def _source_timestamp_age(value: str | None, now: datetime) -> float | None:
    """Age of the provider's last price change; this is not the time we captured the quote."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (now - parsed.astimezone(timezone.utc)).total_seconds()
    except (TypeError, ValueError):
        return None


def _normalise_odds_fixture(fixture: dict, odds: dict, market_families: dict[str, dict], now: datetime) -> dict:
    """Normalize active paired bookmaker prices using OddsPapi's documented outcome ordering."""
    home = str(fixture.get("participant1Name") or "").strip()
    away = str(fixture.get("participant2Name") or "").strip()
    groups: dict[tuple, dict[str, dict]] = {}
    boards = (odds or {}).get("bookmakerOdds") or {}
    audit = {
        "bookmaker_boards": len(boards) if isinstance(boards, dict) else 0,
        "active_bookmaker_boards": 0,
        "recognized_market_instances": 0,
        "active_prices": 0,
        "fresh_prices": 0,
        "source_changes_within_1h": 0,
        "source_changes_over_1h": 0,
        "source_timestamp_missing_or_invalid": 0,
        "source_timestamp_future": 0,
        "parsed_outcomes": 0,
        "outcomes_mapped_by_catalogue": 0,
        "outcomes_mapped_by_position": 0,
        "lines_from_bookmaker_label": 0,
        "lines_from_catalogue": 0,
        "missing_market_line": 0,
        "unparsed_outcomes": 0,
        "stale_prices": 0,
        "incomplete_bookmaker_market_pairs": 0,
        "paired_quotes": 0,
        "normalized_market_groups": 0,
    }
    if not isinstance(boards, dict):
        boards = {}

    for bookmaker, board in boards.items():
        if not isinstance(board, dict) or not board.get("bookmakerIsActive", True) or board.get("suspended"):
            continue
        audit["active_bookmaker_boards"] += 1
        markets = board.get("markets") or {}
        if not isinstance(markets, dict):
            continue

        for market_id, market in markets.items():
            metadata = market_families.get(str(market_id))
            if not metadata or not isinstance(market, dict):
                continue
            family = metadata.get("family")
            if family not in {"moneyline", "spread", "total"} or not market.get("marketActive", True):
                continue
            audit["recognized_market_instances"] += 1

            raw_outcomes = market.get("outcomes") or {}
            if not isinstance(raw_outcomes, dict):
                continue
            outcome_ids = list(raw_outcomes.keys())
            ordered_sides = ("home", "away") if family in {"moneyline", "spread"} else ("over", "under")

            for position, (outcome_id, outcome) in enumerate(raw_outcomes.items()):
                if not isinstance(outcome, dict):
                    continue
                players = outcome.get("players") or {}
                player = players.get("0") if isinstance(players, dict) else None
                if not isinstance(player, dict) or not player.get("active", True):
                    continue
                try:
                    price = float(player.get("price"))
                except (TypeError, ValueError):
                    continue
                if not (1.0001 < price < 1000):
                    continue
                audit["active_prices"] += 1

                changed = player.get("changedAt") or player.get("bookmakerChangedAt") or (odds or {}).get("updatedAt")
                # changedAt is the last PRICE CHANGE, not the time this live
                # /odds response was captured. An unchanged active price is
                # still the current quote returned by this request.
                observed_at = now.astimezone(timezone.utc).isoformat()
                audit["fresh_prices"] += 1
                source_age = _source_timestamp_age(changed, now)
                if source_age is None:
                    audit["source_timestamp_missing_or_invalid"] += 1
                elif source_age < -300:
                    audit["source_timestamp_future"] += 1
                elif source_age <= 3600:
                    audit["source_changes_within_1h"] += 1
                else:
                    audit["source_changes_over_1h"] += 1

                allowed_sides = set(ordered_sides)
                parsed = _parse_label(player.get("bookmakerOutcomeId"))
                source = "bookmaker"
                if parsed is None or parsed[0] not in allowed_sides:
                    catalogue_label = (metadata.get("outcomes") or {}).get(str(outcome_id))
                    parsed = _parse_label(catalogue_label)
                    source = "catalogue"

                # OddsPapi's basketball examples explicitly use outcome order
                # for two-way moneylines when a bookmaker sends opaque numeric
                # bookmakerOutcomeId values. Apply the same documented order to
                # full-game two-way spread/total markets only as a last fallback.
                if parsed is None or parsed[0] not in allowed_sides:
                    if position < len(ordered_sides) and len(outcome_ids) == 2:
                        parsed = (ordered_sides[position], None)
                        source = "position"
                    else:
                        audit["unparsed_outcomes"] += 1
                        continue

                side, raw_line = parsed
                audit["parsed_outcomes"] += 1
                if source == "catalogue":
                    audit["outcomes_mapped_by_catalogue"] += 1
                elif source == "position":
                    audit["outcomes_mapped_by_position"] += 1

                if raw_line is not None:
                    audit["lines_from_bookmaker_label"] += 1

                if family == "moneyline":
                    line = None
                elif family == "total":
                    line_value = raw_line if raw_line is not None else metadata.get("handicap")
                    if line_value is None:
                        audit["missing_market_line"] += 1
                        continue
                    if raw_line is None:
                        audit["lines_from_catalogue"] += 1
                    line = round(float(line_value), 2)
                else:
                    if raw_line is not None:
                        line_value = raw_line if side == "home" else -raw_line
                    else:
                        line_value = metadata.get("handicap")
                    if line_value is None:
                        audit["missing_market_line"] += 1
                        continue
                    if raw_line is None:
                        audit["lines_from_catalogue"] += 1
                    line = round(float(line_value), 2)

                key = (family, line)
                groups.setdefault(key, {}).setdefault(str(bookmaker), {})[side] = {
                    "price": price,
                    "captured_at": observed_at,
                    "source_changed_at": str(changed or ""),
                }

    ft: dict[str, dict] = {}
    for (family, line), by_book in groups.items():
        api_name = {"moneyline": "moneyline", "spread": "spread", "total": "total"}[family]
        for bookmaker, sides in by_book.items():
            required = ("home", "away") if family in {"moneyline", "spread"} else ("over", "under")
            if not all(side in sides for side in required):
                audit["incomplete_bookmaker_market_pairs"] += 1
                continue
            captured = min(sides[x]["captured_at"] for x in required)
            entry = {"line": line, required[0]: sides[required[0]]["price"],
                     required[1]: sides[required[1]]["price"], "captured_at": captured}
            ft.setdefault(bookmaker, {}).setdefault(api_name, []).append(entry)
            audit["paired_quotes"] += 1

    audit["normalized_market_groups"] = sum(
        len(rows) for markets in ft.values() for rows in markets.values()
    )
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
        "competition": _fixture_competition_label(fixture),
        "start_time": fixture.get("startTime"),
        "updated_at": fixture.get("updatedAt"),
        "periods": periods,
        "providers": providers,
        "odds_audit": audit,
        "oddspapi_fixture_id": fixture.get("fixtureId"),
        "settlement_scope": "includes_overtime",
    }

def _competition_key(value: str) -> str:
    """Canonicalize exact league identity without merging genders or NCAA divisions."""
    label = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").casefold()
    compact = re.sub(r"[^a-z0-9]+", "", label)
    if not compact:
        return ""
    if "wnba" in compact:
        return "wnba"
    if "euroleague" in compact:
        return "euroleague" + ("women" if "women" in compact or "female" in compact else "")
    if "eurocup" in compact:
        return "eurocup" + ("women" if "women" in compact or "female" in compact else "")
    if "nba" in compact and not any(token in compact for token in ("nbagleague", "nbawomen")):
        return "nba"
    if "ncaa" in compact:
        gender = "women" if (
            "women" in compact or "female" in compact
            or compact.endswith(("ncaaw", "ncaadiw", "ncaad1w", "ncaadivisioniw"))
        ) else "men" if (
            "men" in compact or "male" in compact
            or compact.endswith(("ncaam", "ncaadim", "ncaad1m", "ncaadivisionim"))
        ) else ""
        if gender:
            if any(token in compact for token in ("divisioniii", "division3", "diviii", "div3", "ncaa3")):
                division = "3"
            elif any(token in compact for token in ("divisionii", "division2", "divii", "div2", "ncaa2")):
                division = "2"
            else:
                # Providers commonly label NCAA Division I simply "NCAA Women".
                division = "1"
            return "ncaadivision" + division + gender
    # Remove display-only suffixes without erasing league identifiers.
    for suffix in ("boxscores", "boxscore", "basketball", "statistics", "stats"):
        if compact.endswith(suffix):
            compact = compact[:-len(suffix)]
            break
    if compact.endswith("womens"):
        return compact[:-6] + "women"
    if compact.endswith("female"):
        return compact[:-6] + "women"
    if compact.endswith("women"):
        return compact
    if compact.endswith("mens"):
        return compact[:-4] + "men"
    if compact.endswith("male"):
        return compact[:-4] + "men"
    return compact


def _fixture_competition_keys(fixture: dict) -> set[str]:
    labels = (
        fixture.get("tournamentName"),
        fixture.get("tournamentSlug"),
        fixture.get("categoryName"),
    )
    return {_competition_key(label) for label in labels if _competition_key(label)}


def _fixture_competition_label(fixture: dict, preferred_keys: set[str] | None = None) -> str:
    labels = (
        fixture.get("tournamentName"),
        fixture.get("tournamentSlug"),
        fixture.get("categoryName"),
    )
    for label in labels:
        if label and (preferred_keys is None or _competition_key(label) in preferred_keys):
            return str(label).strip()
    return next((str(label).strip() for label in labels if label), "")


def games(
    window_start: datetime,
    window_end: datetime,
    preferred_competitions: list[str] | None = None,
) -> dict:
    if (window_end - window_start) > timedelta(days=7):
        raise OddsPapiError("La ventana máxima de OddsPapi es de 7 días.", 422)

    preferred_keys = {
        _competition_key(label)
        for label in (preferred_competitions or [])
        if _competition_key(label)
    }
    preferred_tournament_ids = set().union(*(
        TOURNAMENT_IDS_BY_COMPETITION.get(key, set())
        for key in preferred_keys
    )) if preferred_keys else set()
    if not preferred_keys:
        raise OddsPapiError(
            "No hay competiciones con historial local para comparar. Importá primero datos de NBA, EuroLeague o NCAA femenina.",
            422,
        )

    # Select bookmaker slugs before requesting fixtures. OddsPapi can filter
    # the fixture list by the exact bookmaker set, so we never fetch odds for
    # unrelated leagues or matches without prices at those operators.
    account = account_status()
    available_bookmakers = account.get("bookmakers_available_in_plan") or []
    selected_bookmakers = _select_bookmakers(available_bookmakers)
    if not selected_bookmakers:
        raise OddsPapiError(
            "OddsPapi no devolvió casas de apuestas habilitadas para esta suscripción."
        )

    fixtures = _get("fixtures", {
        "sportId": SPORT_ID_BASKETBALL,
        "from": window_start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": window_end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "statusId": 0,
        "hasOdds": "true",
        "bookmakers": ",".join(selected_bookmakers),
        "language": "en",
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

    matching = [
        x for x in upcoming
        if (
            str(x.get("tournamentId") or "") in preferred_tournament_ids
            or bool(_fixture_competition_keys(x) & preferred_keys)
        )
    ]
    matching.sort(key=lambda x: str(x.get("startTime") or ""))
    available_competitions = sorted({
        _fixture_competition_label(x)
        for x in upcoming
        if _fixture_competition_label(x)
    })
    matching_competitions = sorted({
        _fixture_competition_label(x, preferred_keys)
        for x in matching
        if _fixture_competition_label(x, preferred_keys)
    })

    remaining = int(account.get("requests_remaining") or 0)
    # /fixtures and /markets each consume one request. Leave both reserved.
    odds_limit = min(MAX_ODDS_FIXTURES_PER_RUN, max(0, remaining - 2))
    selected = matching[:odds_limit]

    # If no fixture belongs to the user's imported competitions, stop here.
    # In particular, do not request /markets or odds for other tournaments.
    if not selected:
        return {
            "count": len(upcoming),
            "games": [],
            "matching_fixture_count": len(matching),
            "matching_fixtures_with_odds": len(matching),
            "matching_competitions": matching_competitions,
            "matching_tournament_ids": sorted({
                str(x.get("tournamentId")) for x in matching if x.get("tournamentId") is not None
            }),
            "preferred_tournament_ids": sorted(preferred_tournament_ids),
            "available_competitions": available_competitions,
            "odds_fixtures_requested": 0,
            "odds_fixtures_with_markets": 0,
            "quota_remaining_before_scan": remaining,
            "bookmakers_requested": selected_bookmakers,
            "bookmakers_limit": MAX_BOOKMAKERS_PER_FIXTURE,
            "scan_cap": MAX_ODDS_FIXTURES_PER_RUN,
            "markets_catalog_loaded": False,
            "odds_data_audit": {
                "bookmaker_boards": 0, "active_bookmaker_boards": 0,
                "recognized_market_instances": 0, "active_prices": 0,
                "fresh_prices": 0, "source_changes_within_1h": 0,
                "source_changes_over_1h": 0, "source_timestamp_missing_or_invalid": 0,
                "source_timestamp_future": 0,
                "parsed_outcomes": 0, "outcomes_mapped_by_catalogue": 0,
                "outcomes_mapped_by_position": 0, "lines_from_bookmaker_label": 0,
                "lines_from_catalogue": 0, "missing_market_line": 0,
                "unparsed_outcomes": 0,
                "stale_prices": 0, "incomplete_bookmaker_market_pairs": 0,
                "paired_quotes": 0, "normalized_market_groups": 0,
            },
        }

    market_families = _market_metadata()
    # Reuse the cached /markets catalogue fetched above; this does not make
    # an additional provider request when the cache is warm.
    catalogue_audit = market_catalogue_audit()
    normalized = []
    audit_fields = (
        "bookmaker_boards", "active_bookmaker_boards", "recognized_market_instances",
        "active_prices", "fresh_prices", "source_changes_within_1h",
        "source_changes_over_1h", "source_timestamp_missing_or_invalid",
        "source_timestamp_future", "parsed_outcomes", "outcomes_mapped_by_catalogue",
        "outcomes_mapped_by_position", "lines_from_bookmaker_label",
        "lines_from_catalogue", "missing_market_line", "unparsed_outcomes", "stale_prices", "incomplete_bookmaker_market_pairs", "paired_quotes",
        "normalized_market_groups",
    )
    odds_data_audit = {key: 0 for key in audit_fields}
    for fixture in selected:
        # Never cache the raw /odds payload: normalize one fixture at a time.
        odds = _get(
            "odds",
            {
                "fixtureId": fixture.get("fixtureId"),
                "bookmakers": ",".join(selected_bookmakers),
                "language": "en",
                "verbosity": 3,
            },
            ttl=0,
        )
        if isinstance(odds, dict):
            observed_at = datetime.now(timezone.utc)
            normalized_fixture = _normalise_odds_fixture(fixture, odds, market_families, observed_at)
            normalized.append(normalized_fixture)
            for key in audit_fields:
                odds_data_audit[key] += int((normalized_fixture.get("odds_audit") or {}).get(key) or 0)
        del odds
        time.sleep(0.55)

    return {
        "count": len(upcoming),
        "games": normalized,
        "matching_fixture_count": len(matching),
        "matching_fixtures_with_odds": len(matching),
        "matching_competitions": matching_competitions,
        "matching_tournament_ids": sorted({
            str(x.get("tournamentId")) for x in matching if x.get("tournamentId") is not None
        }),
        "preferred_tournament_ids": sorted(preferred_tournament_ids),
        "available_competitions": available_competitions,
        "odds_fixtures_requested": len(selected),
        "odds_fixtures_with_markets": sum(bool(x.get("periods", {}).get("ft")) for x in normalized),
        "quota_remaining_before_scan": remaining,
        "bookmakers_requested": selected_bookmakers,
        "bookmakers_limit": MAX_BOOKMAKERS_PER_FIXTURE,
        "scan_cap": MAX_ODDS_FIXTURES_PER_RUN,
        "markets_catalog_loaded": True,
        "market_catalogue_audit": catalogue_audit,
        "odds_data_audit": odds_data_audit,
    }
