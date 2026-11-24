"""OddsPapi v4 adapter for basketball fixtures and pre-match bookmaker odds.

The API key is read only from Render environment variables and is never returned
in API responses or logs. Account status is unmetered; fixtures/markets/odds use
the provider's monthly request quota.
"""
from __future__ import annotations

import json
import math
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
MAX_TOURNAMENT_BOOKMAKERS = 3
MAX_TOURNAMENT_IDS_PER_CALL = 3
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
                    # A basketball match market is a two-outcome market on the
                    # final-result period. Preserve the catalogue fields required
                    # to resolve outcome IDs and alternate lines later; dropping
                    # handicap/outcomes here makes correct normalization impossible.
                    try:
                        market_length = int(item.get("marketLength") or 0)
                    except (TypeError, ValueError):
                        market_length = 0
                    if (
                        family
                        and _is_full_game_period(period)
                        and item.get("playerProp") is False
                        and market_length == 2
                    ):
                        selected.append({
                            key: item.get(key)
                            for key in (
                                "sportId", "marketId", "marketName", "marketLength",
                                "playerProp", "handicap", "period", "marketType", "outcomes",
                            )
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


def account_status(force_refresh: bool = False) -> dict:
    if not os.environ.get("ODDSPAPI_API_KEY", "").strip():
        return {"configured": False, "provider": "OddsPapi", "connected": False,
                "message": "Falta ODDSPAPI_API_KEY en Render."}
    data = _get("account", ttl=0 if force_refresh else 10, unmetered=True)
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
    """Only the basketball final-result period is eligible for these match markets."""
    period = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    # OddsPapi's basketball catalogue uses 'result' for the final score with
    # overtime. Regulation-only result markets use another market family and
    # must not be merged into this settlement scope.
    return period == "result"


def _market_family(name: str) -> str | None:
    """Recognize only the three exact two-way full-game basketball families."""
    label = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    aliases = {
        "winner incl overtime": "moneyline",
        "winner including overtime": "moneyline",
        "over under incl overtime": "total",
        "over under including overtime": "total",
        "handicap incl overtime": "spread",
        "handicap including overtime": "spread",
    }
    return aliases.get(label)


def _participant_outcome_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").lower())
    return "".join(char for char in normalized if char.isalnum())


def _parse_market_outcome_label(
    value: str, family: str, home: str = "", away: str = ""
) -> tuple[str, float | None] | None:
    parsed = _parse_label(value)
    allowed = {"home", "away"} if family in {"moneyline", "spread"} else {"over", "under"}
    if parsed is not None and parsed[0] in allowed:
        return parsed
    if family in {"moneyline", "spread"}:
        label_key = _participant_outcome_key(value)
        if label_key and label_key == _participant_outcome_key(home):
            return "home", None
        if label_key and label_key == _participant_outcome_key(away):
            return "away", None
    return None


def _market_metadata() -> dict[str, dict]:
    """Load exact two-way full-game basketball market metadata from OddsPapi."""
    data = _get("markets", {"language": "en"}, ttl=86400)
    if not isinstance(data, list):
        raise OddsPapiError("El catálogo de mercados de OddsPapi tiene un formato inesperado.")
    result: dict[str, dict] = {}
    for row in data:
        if not isinstance(row, dict) or str(row.get("sportId") or "") != str(SPORT_ID_BASKETBALL):
            continue
        family = _market_family(row.get("marketName"))
        if not family or not _is_full_game_period(str(row.get("period") or "")):
            continue
        if row.get("playerProp") is not False:
            continue
        try:
            if int(row.get("marketLength") or 0) != 2:
                continue
        except (TypeError, ValueError):
            continue

        try:
            raw_handicap = row.get("handicap")
            handicap = float(raw_handicap) if raw_handicap not in (None, "") else None
        except (TypeError, ValueError):
            handicap = None
        if handicap is not None and not math.isfinite(handicap):
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

        expected = {"home", "away"} if family in {"moneyline", "spread"} else {"over", "under"}
        canonical_sides = {
            parsed[0] for label in outcomes.values()
            if (parsed := _parse_market_outcome_label(label, family)) is not None
            and parsed[0] in expected
        }
        # Totals must explicitly identify Over and Under. For moneylines and
        # handicaps, some catalogue versions use participant names as outcome
        # labels; those are resolved against this fixture later.
        invalid_canonical_side = not canonical_sides.issubset(expected)
        if len(outcomes) != 2 or invalid_canonical_side:
            continue
        if family == "total" and canonical_sides != expected:
            continue
        if family in {"total", "spread"} and handicap is None:
            continue

        market_id = row.get("marketId")
        if market_id is None:
            continue
        result[str(market_id)] = {
            "family": family,
            "handicap": handicap,
            "outcomes": outcomes,
            "market_name": str(row.get("marketName") or ""),
            "period": str(row.get("period") or ""),
            "market_type": str(row.get("marketType") or ""),
        }
    return result


def market_catalogue_audit() -> dict:
    """Audit only eligible, two-way full-game basketball catalogue definitions."""
    data = _get("markets", {"language": "en"}, ttl=86400)
    if not isinstance(data, list):
        raise OddsPapiError("El catálogo de mercados de OddsPapi tiene un formato inesperado.")
    audit = {
        "sport_id": SPORT_ID_BASKETBALL,
        "catalogue_rows_received": len(data),
        "basketball_rows": 0,
        "excluded_wrong_period_or_family": 0,
        "excluded_props_or_non_two_way": 0,
        "excluded_invalid_outcomes": 0,
        "excluded_missing_handicap": 0,
        "families": {},
        "catalogue_fields": [],
    }
    for row in data:
        if not isinstance(row, dict) or str(row.get("sportId") or "") != str(SPORT_ID_BASKETBALL):
            continue
        audit["basketball_rows"] += 1
        family = _market_family(row.get("marketName"))
        if not family or not _is_full_game_period(str(row.get("period") or "")):
            audit["excluded_wrong_period_or_family"] += 1
            continue
        try:
            market_length = int(row.get("marketLength") or 0)
        except (TypeError, ValueError):
            market_length = 0
        if row.get("playerProp") is not False or market_length != 2:
            audit["excluded_props_or_non_two_way"] += 1
            continue

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
        expected = {"home", "away"} if family in {"moneyline", "spread"} else {"over", "under"}
        canonical_sides = {
            parsed[0] for label in outcomes.values()
            if (parsed := _parse_market_outcome_label(label, family)) is not None
            and parsed[0] in expected
        }
        if (
            len(outcomes) != 2
            or not canonical_sides.issubset(expected)
            or (family == "total" and canonical_sides != expected)
        ):
            audit["excluded_invalid_outcomes"] += 1
            continue

        raw_handicap = row.get("handicap")
        try:
            handicap = float(raw_handicap) if raw_handicap not in (None, "") else None
        except (TypeError, ValueError):
            handicap = None
        if handicap is not None and not math.isfinite(handicap):
            handicap = None
        if family in {"total", "spread"} and handicap is None:
            audit["excluded_missing_handicap"] += 1
            continue

        bucket = audit["families"].setdefault(family, {
            "definitions": 0, "with_handicap": 0, "without_handicap": 0,
            "periods": {}, "samples": [],
        })
        bucket["definitions"] += 1
        if handicap is None:
            bucket["without_handicap"] += 1
        else:
            bucket["with_handicap"] += 1
        period = str(row.get("period") or "")
        bucket["periods"][period] = bucket["periods"].get(period, 0) + 1
        if len(bucket["samples"]) < 8:
            bucket["samples"].append({
                "market_id": str(row.get("marketId") or ""),
                "market_name": str(row.get("marketName") or ""),
                "period": period,
                "handicap_raw": None if raw_handicap is None else str(raw_handicap),
                "handicap_parsed": handicap,
                "market_type": str(row.get("marketType") or ""),
                "market_length": market_length,
                "outcomes": list(outcomes.values())[:4],
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
        if len(chosen) >= MAX_TOURNAMENT_BOOKMAKERS:
            break
    if len(chosen) < MAX_TOURNAMENT_BOOKMAKERS:
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
        chosen.extend(remainder[:MAX_TOURNAMENT_BOOKMAKERS - len(chosen)])
    return chosen[:MAX_TOURNAMENT_BOOKMAKERS]


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
    """Normalize only catalogue-matched, two-sided quotes on identical lines."""
    home = str(fixture.get("participant1Name") or "").strip()
    away = str(fixture.get("participant2Name") or "").strip()
    groups: dict[tuple, dict[str, dict[str, dict]]] = {}
    conflicted_pairs: set[tuple] = set()
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
        "missing_catalogue_outcome_ids": 0,
        "outcome_label_conflicts": 0,
        "line_label_conflicts": 0,
        "duplicate_outcome_conflicts": 0,
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
            if not isinstance(raw_outcomes, dict) or len(raw_outcomes) != 2:
                audit["incomplete_bookmaker_market_pairs"] += 1
                continue
            allowed_sides = {"home", "away"} if family in {"moneyline", "spread"} else {"over", "under"}
            expected = ("home", "away") if family in {"moneyline", "spread"} else ("over", "under")

            market_line = None
            if family != "moneyline":
                try:
                    market_line = float(metadata.get("handicap"))
                except (TypeError, ValueError):
                    market_line = None
                if market_line is None or not math.isfinite(market_line):
                    audit["missing_market_line"] += len(raw_outcomes)
                    continue
                market_line = round(market_line, 2)

            for outcome_id, outcome in raw_outcomes.items():
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
                if not math.isfinite(price) or not (1.0001 < price < 1000):
                    continue
                audit["active_prices"] += 1

                changed = player.get("changedAt") or player.get("bookmakerChangedAt") or (odds or {}).get("updatedAt")
                # changedAt is the last price movement, not the observation time.
                # A quote returned by this fresh /odds request is captured now.
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

                catalogue_label = (metadata.get("outcomes") or {}).get(str(outcome_id))
                catalogue_parsed = (
                    _parse_market_outcome_label(catalogue_label, family, home, away)
                    if catalogue_label else None
                )
                bookmaker_parsed = _parse_label(player.get("bookmakerOutcomeId"))

                # outcomeId -> outcomeName in /markets is the authoritative mapping.
                # bookmakerOutcomeId may corroborate it, but array/dict position
                # is never a safe substitute: order changes across alternate lines.
                parsed = None
                source = None
                if catalogue_parsed is not None and catalogue_parsed[0] in allowed_sides:
                    parsed = catalogue_parsed
                    source = "catalogue"
                    if bookmaker_parsed is not None and bookmaker_parsed[0] in allowed_sides:
                        if bookmaker_parsed[0] != catalogue_parsed[0]:
                            audit["outcome_label_conflicts"] += 1
                            continue
                elif bookmaker_parsed is not None and bookmaker_parsed[0] in allowed_sides:
                    parsed = bookmaker_parsed
                    source = "bookmaker"
                else:
                    if catalogue_label is None:
                        audit["missing_catalogue_outcome_ids"] += 1
                    audit["unparsed_outcomes"] += 1
                    continue

                side, catalogue_line = parsed
                bookmaker_line = (
                    bookmaker_parsed[1]
                    if bookmaker_parsed is not None and bookmaker_parsed[0] == side
                    else None
                )
                # The catalogue line is the market definition. Any line embedded
                # in a bookmaker label is an independent consistency check, not
                # a replacement for the catalogue's line.
                if family != "moneyline":
                    expected_label_line = market_line
                    if family == "spread" and side == "away":
                        expected_label_line = -market_line
                    reported_lines = [
                        value for value in (catalogue_line, bookmaker_line)
                        if value is not None
                    ]
                    if any(
                        abs(float(value) - expected_label_line) > 0.001
                        for value in reported_lines
                    ):
                        audit["line_label_conflicts"] += 1
                        continue

                audit["parsed_outcomes"] += 1
                if source == "catalogue":
                    audit["outcomes_mapped_by_catalogue"] += 1
                if bookmaker_line is not None:
                    audit["lines_from_bookmaker_label"] += 1
                if family != "moneyline":
                    audit["lines_from_catalogue"] += 1
                line = None if family == "moneyline" else market_line

                group_key = (family, line)
                pair_key = (group_key, str(bookmaker))
                if pair_key in conflicted_pairs:
                    continue
                by_book = groups.setdefault(group_key, {}).setdefault(str(bookmaker), {})
                existing = by_book.get(side)
                quote = {
                    "price": price,
                    "captured_at": observed_at,
                    "source_changed_at": str(changed or ""),
                    "source_market_id": str(market_id),
                    "source_outcome_id": str(outcome_id),
                }
                if existing is not None:
                    # Never let a second market definition silently overwrite a
                    # different price for the same book/side/line.
                    if abs(existing["price"] - price) > 1e-9:
                        audit["duplicate_outcome_conflicts"] += 1
                        by_book.clear()
                        conflicted_pairs.add(pair_key)
                    continue
                by_book[side] = quote

    ft: dict[str, dict] = {}
    for (family, line), by_book in groups.items():
        api_name = {"moneyline": "moneyline", "spread": "spread", "total": "total"}[family]
        for bookmaker, sides in by_book.items():
            required = ("home", "away") if family in {"moneyline", "spread"} else ("over", "under")
            if not all(side in sides for side in required):
                audit["incomplete_bookmaker_market_pairs"] += 1
                continue
            captured = min(sides[x]["captured_at"] for x in required)
            entry = {
                "line": line,
                required[0]: sides[required[0]]["price"],
                required[1]: sides[required[1]]["price"],
                "captured_at": captured,
            }
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
    if "wnba" in compact or "womensnationalbasketballassociation" in compact:
        return "wnba"
    if compact in {"nbl", "nblmen", "nationalbasketballleague", "nationalbasketballleaguemen", "australianbl", "australiannationalbasketballleague", "nationalbasketballleagueaustralia"}:
        return "nbl"
    if "euroleague" in compact:
        return "euroleague" + ("women" if "women" in compact or "female" in compact else "")
    if "eurocup" in compact:
        return "eurocup" + ("women" if "women" in compact or "female" in compact else "")
    if compact.startswith("nbagleague"):
        return "nbagleague" + ("women" if "women" in compact or "female" in compact else "")
    if "nba" in compact and not any(token in compact for token in (
        "nbagleague", "nbawomen", "summerleague", "californiaclassic", "preseason",
    )):
        return "nba"
    if "ncaa" in compact:
        gender = "women" if (
            "women" in compact or "female" in compact
            or compact.endswith(("ncaaw", "ncaadiw", "ncaad1w", "ncaadivisioniw"))
        ) else "men" if (
            "men" in compact or "male" in compact
            or compact.endswith(("ncaam", "ncaadim", "ncaad1m", "ncaadivisionim"))
        ) else "men"
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


def _cache_is_fresh(path: str, params: dict | None = None) -> bool:
    """Check whether a billable response can be reused without another API call."""
    api_key = os.environ.get("ODDSPAPI_API_KEY", "").strip()
    if not api_key:
        return False
    query_params = dict(params or {})
    query_params["apiKey"] = api_key
    url = BASE + "/" + path.lstrip("/") + "?" + urlencode(query_params)
    with _LOCK:
        cached = _CACHE.get(url)
        return bool(cached and cached[0] > time.time())


def _odds_by_tournament_rows(payload) -> list[dict]:
    """Accept the documented fixture array and common wrapped/single-row variants."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    if payload.get("fixtureId") is not None:
        return [payload]
    for key in ("fixtures", "data", "results", "events"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict) and value.get("fixtureId") is not None:
            return [value]
    return []


def _empty_odds_result(
    upcoming_count: int,
    matching: list[dict],
    available_competitions: list[str],
    matching_competitions: list[str],
    preferred_tournament_ids: set[str],
    remaining: int,
    selected_bookmakers: list[str],
    warning: str,
) -> dict:
    return {
        "count": upcoming_count,
        "games": [],
        "matching_fixture_count": len(matching),
        "matching_fixtures_with_odds": 0,
        "matching_competitions": matching_competitions,
        "matching_tournament_ids": sorted({
            str(x.get("tournamentId")) for x in matching if x.get("tournamentId") is not None
        }),
        "preferred_tournament_ids": sorted(preferred_tournament_ids),
        "available_competitions": available_competitions,
        "odds_fixtures_requested": 0,
        "odds_fixtures_with_markets": 0,
        "odds_requests_made": 0,
        "odds_scan_method": "odds-by-tournaments",
        "quota_remaining_before_scan": remaining,
        "quota_remaining_after_scan": remaining,
        "quota_blocked": True,
        "provider_warning": warning,
        "bookmakers_requested": selected_bookmakers,
        "bookmakers_limit": len(selected_bookmakers),
        "scan_cap": 0,
        "markets_catalog_loaded": False,
        "odds_data_audit": {
            "bookmaker_boards": 0, "active_bookmaker_boards": 0,
            "recognized_market_instances": 0, "active_prices": 0,
            "fresh_prices": 0, "source_changes_within_1h": 0,
            "source_changes_over_1h": 0, "source_timestamp_missing_or_invalid": 0,
            "source_timestamp_future": 0, "parsed_outcomes": 0,
            "outcomes_mapped_by_catalogue": 0, "outcomes_mapped_by_position": 0,
            "lines_from_bookmaker_label": 0, "lines_from_catalogue": 0,
            "missing_market_line": 0, "unparsed_outcomes": 0,
            "missing_catalogue_outcome_ids": 0, "outcome_label_conflicts": 0,
            "line_label_conflicts": 0, "duplicate_outcome_conflicts": 0,
            "stale_prices": 0, "incomplete_bookmaker_market_pairs": 0,
            "paired_quotes": 0, "normalized_market_groups": 0,
        },
    }


def games(
    window_start: datetime,
    window_end: datetime,
    preferred_competitions: list[str] | None = None,
) -> dict:
    """Fetch upcoming fixtures and bulk odds by tournament, not one call per fixture."""
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

    account = account_status(force_refresh=True)
    available_bookmakers = account.get("bookmakers_available_in_plan") or []
    selected_bookmakers = _select_bookmakers(available_bookmakers)[:MAX_TOURNAMENT_BOOKMAKERS]
    if not selected_bookmakers:
        raise OddsPapiError(
            "OddsPapi no devolvió casas de apuestas habilitadas para esta suscripción."
        )

    fixture_params = {
        "sportId": SPORT_ID_BASKETBALL,
        "from": window_start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": window_end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "statusId": 0,
        "hasOdds": "true",
        "bookmakers": ",".join(selected_bookmakers),
        "language": "en",
    }
    fixtures_cached = _cache_is_fresh("fixtures", fixture_params)
    markets_cached = _cache_is_fresh("markets", {"language": "en"})
    requests_remaining = int(account.get("requests_remaining") or 0)
    minimum_initial_requests = (0 if fixtures_cached else 1) + (0 if markets_cached else 1) + 2
    if requests_remaining < minimum_initial_requests:
        reason = (
            "Cuota mensual insuficiente de OddsPapi: quedan "
            f"{requests_remaining} solicitudes y se necesitan al menos "
            f"{minimum_initial_requests} para consultar dos casas y conservar el consenso. "
            "La cuota mensual se renueva el día 1; también podés ampliar el límite del plan."
        )
        return _empty_odds_result(
            0, [], [], [], preferred_tournament_ids, requests_remaining,
            selected_bookmakers, reason,
        )

    fixtures = _get("fixtures", fixture_params, ttl=90)
    if not isinstance(fixtures, list):
        raise OddsPapiError("OddsPapi devolvió un calendario con formato inesperado.")

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
        _fixture_competition_label(x) for x in upcoming if _fixture_competition_label(x)
    })
    matching_competitions = sorted({
        _fixture_competition_label(x, preferred_keys)
        for x in matching if _fixture_competition_label(x, preferred_keys)
    })

    if not matching:
        return {
            "count": len(upcoming), "games": [], "matching_fixture_count": 0,
            "matching_fixtures_with_odds": 0, "matching_competitions": [],
            "matching_tournament_ids": [],
            "preferred_tournament_ids": sorted(preferred_tournament_ids),
            "available_competitions": available_competitions,
            "odds_fixtures_requested": 0, "odds_fixtures_with_markets": 0,
            "odds_requests_made": 0, "odds_scan_method": "odds-by-tournaments",
            "quota_remaining_before_scan": requests_remaining,
            "quota_remaining_after_scan": requests_remaining, "quota_blocked": False,
            "provider_warning": "No hay partidos de las competiciones con historial local dentro de esta ventana.",
            "bookmakers_requested": selected_bookmakers, "bookmakers_limit": len(selected_bookmakers),
            "scan_cap": 0, "markets_catalog_loaded": False,
            "odds_data_audit": {
                "bookmaker_boards": 0, "active_bookmaker_boards": 0,
                "recognized_market_instances": 0, "active_prices": 0,
                "fresh_prices": 0, "source_changes_within_1h": 0,
                "source_changes_over_1h": 0, "source_timestamp_missing_or_invalid": 0,
                "source_timestamp_future": 0, "parsed_outcomes": 0,
                "outcomes_mapped_by_catalogue": 0, "outcomes_mapped_by_position": 0,
                "lines_from_bookmaker_label": 0, "lines_from_catalogue": 0,
                "missing_market_line": 0, "unparsed_outcomes": 0,
                "missing_catalogue_outcome_ids": 0, "outcome_label_conflicts": 0,
                "line_label_conflicts": 0, "duplicate_outcome_conflicts": 0,
                "stale_prices": 0, "incomplete_bookmaker_market_pairs": 0,
                "paired_quotes": 0, "normalized_market_groups": 0,
            },
        }

    tournament_ids = sorted({
        str(x.get("tournamentId")) for x in matching
        if x.get("tournamentId") is not None and str(x.get("tournamentId")).strip()
    })
    tournament_groups = [
        tournament_ids[offset:offset + MAX_TOURNAMENT_IDS_PER_CALL]
        for offset in range(0, len(tournament_ids), MAX_TOURNAMENT_IDS_PER_CALL)
    ]
    if not tournament_groups:
        return _empty_odds_result(
            len(upcoming), matching, available_competitions, matching_competitions,
            preferred_tournament_ids, requests_remaining, selected_bookmakers,
            "Los partidos compatibles no tienen tournamentId; OddsPapi no permite consultar sus cuotas en bloque.",
        )

    refreshed_account = account_status(force_refresh=True)
    remaining_after_fixtures = int(refreshed_account.get("requests_remaining") or 0)
    market_request_cost = 0 if _cache_is_fresh("markets", {"language": "en"}) else 1
    available_odds_calls = max(0, remaining_after_fixtures - market_request_cost)
    bookmaker_count = min(
        MAX_TOURNAMENT_BOOKMAKERS,
        available_odds_calls // len(tournament_groups),
    )
    if bookmaker_count < 2:
        reason = (
            "Cuota mensual insuficiente para verificar al menos dos casas en todos los torneos detectados. "
            f"Quedan {remaining_after_fixtures} solicitudes después de consultar el calendario; "
            "no se rebajó el requisito de consenso."
        )
        return _empty_odds_result(
            len(upcoming), matching, available_competitions, matching_competitions,
            preferred_tournament_ids, requests_remaining, selected_bookmakers, reason,
        )

    selected_bookmakers = selected_bookmakers[:bookmaker_count]
    market_families = _market_metadata()
    catalogue_audit = market_catalogue_audit()

    matching_by_id = {
        str(fixture.get("fixtureId") or ""): fixture
        for fixture in matching if str(fixture.get("fixtureId") or "").strip()
    }
    merged_odds: dict[str, dict] = {}
    odds_requests_made = 0
    for bookmaker in selected_bookmakers:
        for tournament_group in tournament_groups:
            response = _get(
                "odds-by-tournaments",
                {
                    "tournamentIds": ",".join(tournament_group),
                    "bookmaker": bookmaker,
                    "language": "en",
                    "verbosity": 3,
                },
                ttl=30,
            )
            odds_requests_made += 1
            # OddsPapi documents a 1-second cooldown for this bulk endpoint.
            if odds_requests_made > 1:
                time.sleep(1.05)
            for row in _odds_by_tournament_rows(response):
                fixture_id = str(row.get("fixtureId") or "").strip()
                if fixture_id not in matching_by_id:
                    continue
                source_boards = row.get("bookmakerOdds") or {}
                if not isinstance(source_boards, dict):
                    continue
                target = merged_odds.setdefault(fixture_id, {
                    "bookmakerOdds": {},
                    "updatedAt": row.get("updatedAt"),
                })
                if row.get("updatedAt") and (
                    not target.get("updatedAt")
                    or str(row.get("updatedAt")) > str(target.get("updatedAt"))
                ):
                    target["updatedAt"] = row.get("updatedAt")
                for slug, board in source_boards.items():
                    if not isinstance(board, dict):
                        continue
                    if str(slug).lower() != bookmaker.lower():
                        continue
                    target["bookmakerOdds"][slug] = board

    now = datetime.now(timezone.utc)
    normalized = []
    odds_data_audit = {
        "bookmaker_boards": 0, "active_bookmaker_boards": 0,
        "recognized_market_instances": 0, "active_prices": 0,
        "fresh_prices": 0, "source_changes_within_1h": 0,
        "source_changes_over_1h": 0, "source_timestamp_missing_or_invalid": 0,
        "source_timestamp_future": 0, "parsed_outcomes": 0,
        "outcomes_mapped_by_catalogue": 0, "outcomes_mapped_by_position": 0,
        "lines_from_bookmaker_label": 0, "lines_from_catalogue": 0,
        "missing_market_line": 0, "unparsed_outcomes": 0,
        "missing_catalogue_outcome_ids": 0, "outcome_label_conflicts": 0,
        "line_label_conflicts": 0, "duplicate_outcome_conflicts": 0,
        "stale_prices": 0, "incomplete_bookmaker_market_pairs": 0,
        "paired_quotes": 0, "normalized_market_groups": 0,
    }
    for fixture_id, odds_payload in merged_odds.items():
        fixture = matching_by_id[fixture_id]
        result = _normalise_odds_fixture(fixture, odds_payload, market_families, now)
        normalized.append(result)
        for key in odds_data_audit:
            odds_data_audit[key] += int((result.get("odds_audit") or {}).get(key) or 0)

    normalized.sort(key=lambda item: str(item.get("start_time") or ""))
    remaining_after_scan = int(account_status(force_refresh=True).get("requests_remaining") or 0)
    fixtures_with_markets = sum(
        bool((row.get("periods") or {}).get("ft")) for row in normalized
    )
    return {
        "count": len(upcoming),
        "games": normalized,
        "matching_fixture_count": len(matching),
        "matching_fixtures_with_odds": len(normalized),
        "matching_competitions": matching_competitions,
        "matching_tournament_ids": sorted(tournament_ids),
        "preferred_tournament_ids": sorted(preferred_tournament_ids),
        "available_competitions": available_competitions,
        "odds_fixtures_requested": len(merged_odds),
        "odds_fixtures_with_markets": fixtures_with_markets,
        "odds_requests_made": odds_requests_made,
        "odds_scan_method": "odds-by-tournaments",
        "quota_remaining_before_scan": requests_remaining,
        "quota_remaining_after_scan": remaining_after_scan,
        "quota_blocked": False,
        "provider_warning": "",
        "bookmakers_requested": selected_bookmakers,
        "bookmakers_limit": len(selected_bookmakers),
        "scan_cap": len(normalized),
        "markets_catalog_loaded": True,
        "market_catalogue_audit": catalogue_audit,
        "odds_data_audit": odds_data_audit,
    }

