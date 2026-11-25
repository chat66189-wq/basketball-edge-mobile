"""Public ESPN scoreboard/summary adapter for supplemental historical box scores.

ESPN's public JSON endpoints are undocumented and may change. This adapter is
best-effort, keyless, rate-limited, cached, and only stores complete team rows.
It never derives missing box-score fields from averages or guessed values.
"""
from __future__ import annotations

import calendar
import gc
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball"
_CACHE: dict[str, tuple[float, dict]] = {}
_LOCK = threading.Lock()
MAX_MONTHS = 14
MAX_EVENTS_PER_LEAGUE = 240
MAX_SCOREBOARD_WORKERS = 2
MAX_WORKERS = 2
MAX_RESPONSE_BYTES = 32 * 1024 * 1024

# Only mappings that can be kept separate from the imported competition scope.
LEAGUES = {
    "nba": {"slug": "nba", "competition": "NBA"},
    "nbagleague": {"slug": "nba-development", "competition": "NBA G League"},
    "nbl": {"slug": "nbl", "competition": "NBL"},
    "wnba": {"slug": "wnba", "competition": "WNBA"},
    "ncaadivision1men": {
        "slug": "mens-college-basketball",
        "competition": "NCAA Division I Men's Basketball",
        "groups": "50",
    },
    "ncaadivision1women": {
        "slug": "womens-college-basketball",
        "competition": "NCAA Division I Women's Basketball",
        "groups": "50",
    },
}


class ESPNError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _competition_key(value: str) -> str:
    label = _normalise(value)
    if not label:
        return ""
    if "wnba" in label or "womensnationalbasketballassociation" in label:
        return "wnba"
    if label in {"nbl", "nationalbasketballleague", "australiannationalbasketballleague"}:
        return "nbl"
    if "nbagleague" in label or label == "nbadevelopment":
        return "nbagleague"
    if "ncaa" in label:
        return "ncaadivision1women" if ("women" in label or "female" in label or label.endswith(("ncaaw", "ncaadiw", "ncaad1w"))) else "ncaadivision1men"
    if label in {"nba", "nbamen", "nationalbasketballassociation", "nationalbasketballassociationmen"}:
        return "nba"
    if label.startswith("nba") and not label.startswith(
        ("nbasummer", "nbacalifornia", "nbautah", "nbasaltlake", "nbapreseason")
    ) and "summerleague" not in label and "classic" not in label and "development" not in label:
        return "nba"
    return label


def _get_json(path: str, params: dict | None = None, ttl: int = 3600) -> dict:
    query = urlencode({k: v for k, v in (params or {}).items() if v not in (None, "")})
    url = BASE + "/" + path.lstrip("/") + ("?" + query if query else "")
    now = time.time()
    with _LOCK:
        cached = _CACHE.get(url)
        if ttl and cached and cached[0] > now:
            return cached[1]
    data = None
    for attempt in range(2):
        try:
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BasketballEdgeAnalyst/20 (public scoreboard enrichment)",
                },
                method="GET",
            )
            with urlopen(request, timeout=20) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ESPNError(
                    "La respuesta de ESPN superó el límite seguro de 32 MB; se omitió este calendario.",
                    502,
                )
            probe = raw.lstrip()
            if probe.startswith(bytes((239, 187, 191))):
                probe = probe[3:].lstrip()
            if not probe or not probe.startswith((b"{", b"[")):
                raise json.JSONDecodeError("empty or non-JSON response", raw.decode("utf-8", "replace"), 0)
            data = json.loads(raw.decode("utf-8-sig"))
            break
        except HTTPError as exc:
            if exc.code in (400, 404):
                raise ESPNError("ESPN no ofrece ese calendario o resumen para la liga solicitada.", 404) from exc
            if exc.code in (403, 429):
                raise ESPNError("ESPN limitó temporalmente la consulta; se conservaron los datos existentes.", 429) from exc
            raise ESPNError("ESPN respondió HTTP " + str(exc.code) + ".", 502) from exc
        except (URLError, TimeoutError):
            raise ESPNError("No se pudo conectar con el servicio público de ESPN.") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            if attempt == 0:
                time.sleep(0.35)
                continue
            raise ESPNError(
                "ESPN devolvió una respuesta JSON inválida o una página no JSON después de reintentar.",
                502,
            ) from None
    if not isinstance(data, dict):
        raise ESPNError("ESPN devolvió un formato JSON inesperado.")
    if ttl:
        with _LOCK:
            _CACHE[url] = (now + ttl, data)
    return data


def status() -> dict:
    return {
        "configured": True,
        "provider": "ESPN public JSON",
        "authentication_required": False,
        "automatic_requests_to_sportapi": False,
        "supported_competitions": sorted(item["competition"] for item in LEAGUES.values()),
        "notice": "Fuente pública no oficial y no documentada por ESPN; su disponibilidad puede cambiar. Solo se importan box scores completos.",
    }


def _month_keys(start: date, end: date) -> list[str]:
    cursor = date(start.year, start.month, 1)
    keys = []
    while cursor <= end:
        keys.append(cursor.strftime("%Y%m"))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return keys


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
        number = float(value)
        if number != number or abs(number) == float("inf"):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _pair(value) -> tuple[float, float] | None:
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)\s*[-/]\s*(\d+(?:\.\d+)?)", value)
        if match:
            made, attempted = float(match.group(1)), float(match.group(2))
            if attempted >= made >= 0:
                return made, attempted
    return None


def _stats_map(team_entry: dict) -> dict[str, object]:
    raw = team_entry.get("statistics") or team_entry.get("stats") or []
    result: dict[str, object] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            result[_normalise(key)] = value
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = item.get("name") or item.get("abbreviation") or item.get("label") or item.get("shortDisplayName")
            if key:
                result[_normalise(key)] = item.get("displayValue", item.get("value"))
    return result


def _first_number(values: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _number(values.get(_normalise(key)))
        if value is not None:
            return value
    return None


def _team_stats(team_entry: dict, points: float) -> dict | None:
    values = _stats_map(team_entry)
    fg_pair = None
    ft_pair = None
    for key, value in values.items():
        if ("fieldgoalsmade" in key and "fieldgoalsattempted" in key) or key in {"fg", "fieldgoals"}:
            fg_pair = _pair(value)
        if ("freethrowsmade" in key and "freethrowsattempted" in key) or key in {"ft", "freethrows"}:
            ft_pair = _pair(value)
    fga = _first_number(values, "fieldGoalsAttempted", "fga", "field goal attempts")
    fgm = _first_number(values, "fieldGoalsMade", "fgm")
    if fg_pair:
        fgm = fg_pair[0] if fgm is None else fgm
        fga = fg_pair[1] if fga is None else fga
    fta = _first_number(values, "freeThrowsAttempted", "fta", "free throw attempts")
    ftm = _first_number(values, "freeThrowsMade", "ftm")
    if ft_pair:
        ftm = ft_pair[0] if ftm is None else ftm
        fta = ft_pair[1] if fta is None else fta
    oreb = _first_number(values, "offensiveRebounds", "offensiveRebound", "oreb")
    turnovers = _first_number(values, "totalTurnovers", "turnovers", "tov")
    if None in (fga, fta, oreb, turnovers, points):
        return None
    possessions = fga + 0.44 * fta - oreb + turnovers
    if possessions <= 0 or not (0 <= oreb <= 50 and 0 <= turnovers <= 50):
        return None
    return {
        "points": points,
        "possessions": round(possessions, 3),
        "fga": fga,
        "fgm": fgm,
        "fta": fta,
        "ftm": ftm,
        "oreb": oreb,
        "tov": turnovers,
    }


def _event_from_scoreboard(event: dict, spec: dict) -> dict | None:
    if not isinstance(event, dict):
        return None
    status = event.get("status") or {}
    status_type = status.get("type") or {}
    if not (status_type.get("completed") or str(status_type.get("state") or "").casefold() == "post"):
        return None
    kickoff = _parse_datetime(event.get("date"))
    if kickoff is None:
        return None
    competitions = event.get("competitions") or []
    if not competitions or not isinstance(competitions[0], dict):
        return None
    competitors = competitions[0].get("competitors") or []
    by_side = {}
    for competitor in competitors:
        if not isinstance(competitor, dict):
            continue
        side = str(competitor.get("homeAway") or "").casefold()
        team = competitor.get("team") or {}
        name = team.get("displayName") or team.get("shortDisplayName") or team.get("name")
        score = _number(competitor.get("score"))
        if side in {"home", "away"} and name and score is not None:
            by_side[side] = {"id": str(team.get("id") or ""), "name": str(name), "score": score}
    if set(by_side) != {"home", "away"} or by_side["home"]["name"] == by_side["away"]["name"]:
        return None
    event_id = str(event.get("id") or "").strip()
    if not event_id:
        return None
    return {
        "event_id": "espn-" + event_id,
        "espn_id": event_id,
        "competition": spec["competition"],
        "season": str((event.get("season") or {}).get("year") or kickoff.year),
        "event_time": kickoff.isoformat(),
        "home_team": by_side["home"]["name"],
        "away_team": by_side["away"]["name"],
        "status": "finished",
        "home_score": by_side["home"]["score"],
        "away_score": by_side["away"]["score"],
        "_teams": by_side,
        "_league_slug": spec["slug"],
    }


def _summary_rows(event: dict) -> list[dict]:
    summary = _get_json(
        event["_league_slug"] + "/summary",
        {"event": event["espn_id"]},
        ttl=0,
    )
    boxscore = summary.get("boxscore") or {}
    entries = boxscore.get("teams") or []
    if not isinstance(entries, list) or not entries:
        return []
    id_to_side = {event["_teams"][side]["id"]: side for side in ("home", "away") if event["_teams"][side]["id"]}
    rows = []
    seen_sides = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        team_ref = entry.get("team") or {}
        team_id = str(team_ref.get("id") or "")
        side = id_to_side.get(team_id) or str(entry.get("homeAway") or "").casefold()
        if side not in {"home", "away"} or side in seen_sides:
            continue
        seen_sides.add(side)
        points = event[side + "_score"]
        stats = _team_stats(entry, points)
        if stats is None:
            return []
        rows.append({
            "event_id": event["event_id"],
            "team": event["_teams"][side]["name"],
            **stats,
        })
    return rows if seen_sides == {"home", "away"} else []


def sync(
    competitions: str,
    days: int = 400,
    max_games_per_league: int = 180,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    try:
        days = int(days)
        max_games_per_league = int(max_games_per_league)
    except (TypeError, ValueError):
        raise ESPNError("days y max_games_per_league deben ser números enteros.", 422) from None
    if not 1 <= days <= 400:
        raise ESPNError("El historial alternativo puede consultar entre 1 y 400 días.", 422)
    if not 1 <= max_games_per_league <= MAX_EVENTS_PER_LEAGUE:
        raise ESPNError("El máximo por competición debe estar entre 1 y 240 partidos.", 422)

    labels = [x.strip() for x in re.split(r"[,;|]", str(competitions or "")) if x.strip()]
    requested_keys = {_competition_key(x) for x in labels}
    selected = [(key, spec) for key, spec in LEAGUES.items() if key in requested_keys]
    recognized = {key for key, _ in selected}
    unsupported = [label for label in labels if _competition_key(label) not in recognized]
    if not selected:
        return {
            "provider": "ESPN public JSON",
            "competitions_requested": labels,
            "competitions_supported": [],
            "games": [],
            "team_stats": [],
            "games_received": 0,
            "team_stats_rows": 0,
            "warnings": ["ESPN no ofrece un mapeo verificado para las competiciones de esta ventana."],
        }

    if (start_date is None) != (end_date is None):
        raise ESPNError("start_date y end_date deben enviarse juntos.", 422)
    if start_date is not None and end_date is not None:
        try:
            start_day = date.fromisoformat(str(start_date))
            end_day = date.fromisoformat(str(end_date))
        except ValueError:
            raise ESPNError("start_date y end_date deben usar YYYY-MM-DD.", 422) from None
        if end_day < start_day:
            raise ESPNError("end_date no puede ser anterior a start_date.", 422)
        if (end_day - start_day).days + 1 > 31:
            raise ESPNError("Cada lote ESPN puede cubrir como máximo 31 días.", 422)
        if end_day > datetime.now(timezone.utc).date():
            raise ESPNError("end_date no puede estar en el futuro.", 422)
    else:
        end_day = datetime.now(timezone.utc).date()
        start_day = end_day - timedelta(days=days - 1)
    months = _month_keys(start_day, end_day)
    if len(months) > MAX_MONTHS:
        months = months[-MAX_MONTHS:]

    all_games = []
    all_stats = []
    total_events_seen = 0
    failed_summaries = 0
    failed_scoreboards = 0
    incomplete_boxscores = 0
    for key, spec in selected:
        event_index = {}
        def fetch_month(month):
            params = {"dates": month, "limit": 1000}
            if spec.get("groups"):
                params["groups"] = spec["groups"]
            return _get_json(spec["slug"] + "/scoreboard", params, ttl=6 * 3600)
        # Process small batches. Keeping all completed Future objects in one
        # large dictionary also keeps every decoded ESPN JSON response alive.
        # Batching bounds peak memory on Render's 512 MiB free instance.
        for month_offset in range(0, len(months), MAX_SCOREBOARD_WORKERS):
            month_batch = months[month_offset:month_offset + MAX_SCOREBOARD_WORKERS]
            with ThreadPoolExecutor(max_workers=MAX_SCOREBOARD_WORKERS) as executor:
                month_futures = {
                    executor.submit(fetch_month, month): month for month in month_batch
                }
                for future in as_completed(month_futures):
                    try:
                        scoreboard = future.result()
                    except ESPNError as exc:
                        if exc.status == 404:
                            continue
                        if exc.status == 429:
                            raise
                        failed_scoreboards += 1
                        continue
                    except Exception:
                        failed_scoreboards += 1
                        continue
                    events = scoreboard.get("events") or []
                    if not isinstance(events, list):
                        continue
                    for raw_event in events:
                        event = _event_from_scoreboard(raw_event, spec)
                        if event is None:
                            continue
                        played = _parse_datetime(event["event_time"])
                        if played and start_day <= played.date() <= end_day:
                            event_index[event["espn_id"]] = event
            month_futures.clear()
            scoreboard = None
            events = None
            future = None
            gc.collect()

        ordered = sorted(event_index.values(), key=lambda item: item["event_time"], reverse=True)
        total_events_seen += len(ordered)
        chosen = ordered[:max_games_per_league]
        if not chosen:
            continue
        # Summary responses can be considerably larger than scoreboard rows.
        # Keep at most MAX_WORKERS decoded summaries resident at once.
        for event_offset in range(0, len(chosen), MAX_WORKERS):
            event_batch = chosen[event_offset:event_offset + MAX_WORKERS]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(_summary_rows, event): event for event in event_batch
                }
                for future in as_completed(futures):
                    event = futures[future]
                    try:
                        rows = future.result()
                    except ESPNError as exc:
                        failed_summaries += 1
                        if exc.status == 429:
                            raise
                        continue
                    except Exception:
                        failed_summaries += 1
                        continue
                    if len(rows) != 2:
                        incomplete_boxscores += 1
                        continue
                    all_games.append({k: v for k, v in event.items() if not k.startswith("_")})
                    all_stats.extend(rows)
            futures.clear()
            rows = None
            event = None
            future = None
            gc.collect()

    all_games.sort(key=lambda item: item["event_time"])
    all_stats.sort(key=lambda item: (item["event_id"], item["team"]))
    complete_ids = {row["event_id"] for row in all_stats}
    warnings = [
        "Fuente ESPN pública no oficial y no documentada; puede cambiar o limitar el acceso sin aviso.",
        "Solo se guardaron partidos con box scores de ambos equipos y campos suficientes para estimar posesiones.",
        "No se importaron lesiones, alineaciones confirmadas ni cuotas desde esta fuente.",
    ]
    if unsupported:
        warnings.append("Competiciones sin cobertura ESPN mapeada: " + ", ".join(sorted(set(unsupported))) + ".")
    if failed_scoreboards:
        warnings.append(
            str(failed_scoreboards)
            + " calendarios mensuales no se pudieron leer; se omitieron y se conservaron los demás meses y ligas."
        )
    if incomplete_boxscores:
        warnings.append(str(incomplete_boxscores) + " partidos no tenían box score completo de ambos equipos y se excluyeron.")
    if failed_summaries:
        warnings.append(str(failed_summaries) + " resúmenes no se pudieron recuperar; se omitieron sin inventar estadísticas.")
    return {
        "provider": "ESPN public JSON",
        "competitions_requested": labels,
        "competitions_supported": [spec["competition"] for _, spec in selected],
        "days_requested": days,
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "max_games_per_league": max_games_per_league,
        "scoreboard_events_seen": total_events_seen,
        "games_received": len(all_games),
        "complete_boxscore_games": len(complete_ids),
        "team_stats_rows": len(all_stats),
        "failed_scoreboards": failed_scoreboards,
        "incomplete_boxscores_excluded": incomplete_boxscores,
        "failed_summaries": failed_summaries,
        "games": all_games,
        "team_stats": all_stats,
        "warnings": warnings,
    }
