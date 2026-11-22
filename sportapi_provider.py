"""Adapter for SportAPI.ai basketball endpoints."""
from __future__ import annotations
import json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

BASE="https://sportapi.ai/api"
_CACHE={}
_LOCK=threading.Lock()

class SportAPIError(Exception):
    def __init__(self, message, status=502):
        super().__init__(message); self.status=status

def _get(path, params=None, ttl=0):
    key=os.environ.get("SPORTAPI_KEY","").strip()
    if not key: raise SportAPIError("Falta SPORTAPI_KEY en las variables de entorno de Render.",503)
    query=urlencode({k:v for k,v in (params or {}).items() if v not in (None,"")})
    url=BASE+"/"+path.lstrip("/") + ("?"+query if query else "")
    with _LOCK:
        hit=_CACHE.get(url)
        if ttl and hit and hit[0]>time.time(): return hit[1]
    try:
        req=Request(url,headers={"X-Api-Key":key,"Accept":"application/json"},method="GET")
        with urlopen(req,timeout=25) as response:
            data=json.loads(response.read(8*1024*1024).decode("utf-8"))
            quota={name:response.headers.get(header) for name,header in {
                "daily_limit":"X-RateLimit-Limit","daily_remaining":"X-RateLimit-Remaining",
                "monthly_limit":"X-RateLimit-Month-Limit","monthly_remaining":"X-RateLimit-Month-Remaining"}.items()}
    except HTTPError as exc:
        if exc.code in (401,403): raise SportAPIError("SportAPI.ai rechazó la clave. Revisá SPORTAPI_KEY en Render.",502) from exc
        if exc.code==429: raise SportAPIError("Se alcanzó el límite diario o mensual de SportAPI.ai.",429) from exc
        if exc.code==404: raise SportAPIError("SportAPI.ai no encontró el recurso solicitado.",404) from exc
        raise SportAPIError("SportAPI.ai respondió HTTP "+str(exc.code)+".",502) from exc
    except (URLError,TimeoutError): raise SportAPIError("No se pudo conectar con SportAPI.ai.") from None
    except (UnicodeDecodeError,json.JSONDecodeError): raise SportAPIError("SportAPI.ai devolvió una respuesta JSON inválida.") from None
    if not isinstance(data,dict): raise SportAPIError("Formato inesperado de SportAPI.ai.")
    if data.get("success") is False or data.get("error"):
        raise SportAPIError("SportAPI.ai: "+str(data.get("error") or data.get("message") or "error del proveedor")[:200],502)
    data["_quota"]=quota
    if ttl:
        with _LOCK: _CACHE[url]=(time.time()+ttl,data)
    return data

def status():
    data=_get("basketball/leagues",ttl=600)
    leagues=data.get("data") or []
    return {"configured":True,"provider":"SportAPI.ai","league_count":len(leagues),"quota":data.get("_quota")}

def leagues(search=None):
    data=_get("basketball/leagues",ttl=600)
    items=data.get("data") or []
    if not isinstance(items,list): raise SportAPIError("Lista de competiciones inválida.")
    term=(search or "").strip().casefold()
    if term:
        items=[x for x in items if term in " ".join(str(x.get(k) or "") for k in ("name","short_name","slug","country","gender")).casefold()]
    return {"count":len(items),"leagues":sorted(items,key=lambda x:str(x.get("name") or "").casefold()),"quota":data.get("_quota")}

def _num(value):
    if value in (None,"","-"): return None
    try:
        n=float(str(value).replace(",",""))
        return n if n==n and abs(n)!=float("inf") else None
    except (TypeError,ValueError): return None

def _team_row(eid,name,score,s):
    fga,fta,oreb,tov=(_num(s.get(k)) for k in ("fga","fta","oreb","tov"))
    pts=_num(score)
    if None in (fga,fta,oreb,tov,pts): return None
    poss=fga+0.44*fta-oreb+tov
    if poss<=0: return None
    return {"event_id":eid,"team":name,"possessions":round(poss,3),"points":pts,
        "fga":fga,"fgm":_num(s.get("fgm")),"fg3a":_num(s.get("tpa")),"fg3m":_num(s.get("tpm")),
        "fta":fta,"ftm":_num(s.get("ftm")),"oreb":oreb,"dreb":_num(s.get("dreb")),"tov":tov}

def _player_rows(eid,items,team):
    rows=[]
    for item in items if isinstance(items,list) else []:
        if not isinstance(item,dict) or item.get("dnp"): continue
        player=item.get("player") or {}; pid=player.get("id") or player.get("slug")
        if pid is None: continue
        rows.append({"event_id":eid,"player_id":str(pid),"team":team,"minutes":item.get("minutes"),
            "points":_num(item.get("pts")),"rebounds":_num(item.get("reb")),"assists":_num(item.get("ast")),
            "steals":_num(item.get("stl")),"blocks":_num(item.get("blk")),"turnovers":_num(item.get("tov")),
            "fga":_num(item.get("fga")),"fgm":_num(item.get("fgm")),"fg3a":_num(item.get("tpa")),
            "fg3m":_num(item.get("tpm")),"fta":_num(item.get("fta")),"ftm":_num(item.get("ftm"))})
    return rows

def _fetch_game_boxscore(g):
    """Fetch one game's team and player box scores; keep team stats if player data fails."""
    gid = g.get("id")
    home = g.get("home") or {}
    away = g.get("away") or {}
    eid = "sportapi-" + str(gid)
    team_rows, player_rows, warnings = [], [], []
    quota = None
    rate_limited = False

    try:
        details = _get("basketball/games/" + str(int(gid)), ttl=3600)
        detail = details.get("data") or details.get("game") or {}
        stats = detail.get("stats") or details.get("stats") or {}
        quota = details.get("_quota")
        if isinstance(stats, dict):
            for side, team in (("home", home), ("away", away)):
                row_stats = stats.get(side)
                if isinstance(row_stats, dict) and team.get("name"):
                    row = _team_row(eid, str(team["name"]), team.get("score"), row_stats)
                    if row:
                        team_rows.append(row)
    except SportAPIError as exc:
        warnings.append(str(exc))
        rate_limited = exc.status == 429
        return team_rows, player_rows, quota, warnings, rate_limited
    except (TypeError, ValueError):
        warnings.append("No se pudieron interpretar las estadísticas de equipo de un partido.")
        return team_rows, player_rows, quota, warnings, rate_limited

    try:
        players = _get("basketball/games/" + str(int(gid)) + "/players", ttl=3600)
        quota = players.get("_quota") or quota
        player_data = players.get("data") or {}
        if isinstance(player_data, dict):
            player_rows.extend(_player_rows(eid, player_data.get("home"), str(home["name"])))
            player_rows.extend(_player_rows(eid, player_data.get("away"), str(away["name"])))
    except SportAPIError as exc:
        warnings.append(str(exc))
        rate_limited = exc.status == 429

    return team_rows, player_rows, quota, warnings, rate_limited


def sync(league,season=None,date=None,limit=20):
    league=str(league or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9-]{1,80}",league): raise SportAPIError("Seleccioná una competición válida.",422)
    try: limit=int(limit)
    except (TypeError,ValueError): raise SportAPIError("Límite inválido.",422) from None
    if not 1<=limit<=200: raise SportAPIError("El límite por importación es de 1 a 200 partidos.",422)
    if date:
        try: datetime.strptime(date,"%Y-%m-%d")
        except ValueError: raise SportAPIError("La fecha debe tener formato AAAA-MM-DD.",422) from None
        listing=_get("basketball/games/date/"+date,{"tz":"America/Argentina/Cordoba","league":league},120)
    else:
        listing=_get("basketball/leagues/"+quote(league,safe="-")+"/games",
            {"status":"finished","season":season,"limit":limit},120)
    raw=listing.get("data") or []
    if not isinstance(raw,list): raise SportAPIError("Lista de partidos inválida.")
    finals=[]
    for g in raw:
        if not isinstance(g,dict): continue
        lg=g.get("league") or {}
        if date and str(lg.get("slug") or lg.get("id") or "")!=league: continue
        if str(g.get("status") or "").upper() not in ("FT","AOT"): continue
        if season and not date and str(g.get("season") or "")!=season: continue
        finals.append(g)
        if len(finals)>=limit: break

    games, team_stats, player_stats = [], [], []
    warnings = []
    completed = 0
    quota = listing.get("_quota")
    boxscore_games = []

    for g in finals:
        gid = g.get("id")
        home = g.get("home") or {}
        away = g.get("away") or {}
        lg = g.get("league") or {}
        if gid is None or not home.get("name") or not away.get("name") or not g.get("start"):
            warnings.append("Se omitió un partido con campos básicos incompletos.")
            continue
        eid = "sportapi-" + str(gid)
        games.append({
            "event_id": eid,
            "competition": str(lg.get("name") or league),
            "season": str(g.get("season") or season or ""),
            "event_time": str(g.get("start")),
            "home_team": str(home["name"]),
            "away_team": str(away["name"]),
            "status": "finished",
            "home_score": home.get("score"),
            "away_score": away.get("score"),
        })
        completed += 1
        if g.get("has_boxscore") is False:
            warnings.append("Algunos partidos no tienen box score; se importó solo el resultado.")
        else:
            boxscore_games.append(g)

    # Bounded concurrency keeps a 200-game import within the web-service request window.
    # Process in small batches so a provider 429 stops subsequent requests promptly.
    rate_limited = False
    batch_size = 8
    for offset in range(0, len(boxscore_games), batch_size):
        batch = boxscore_games[offset:offset + batch_size]
        with ThreadPoolExecutor(max_workers=min(batch_size, len(batch))) as executor:
            futures = [executor.submit(_fetch_game_boxscore, game) for game in batch]
            for future in as_completed(futures):
                try:
                    team_rows, player_rows, used_quota, game_warnings, hit_limit = future.result()
                    team_stats.extend(team_rows)
                    player_stats.extend(player_rows)
                    quota = used_quota or quota
                    warnings.extend(game_warnings)
                    rate_limited = rate_limited or hit_limit
                except Exception:
                    warnings.append("No se pudieron descargar las estadísticas de un partido.")
        if rate_limited:
            warnings.append("La importación se detuvo al alcanzar el límite de solicitudes de SportAPI.ai.")
            break

    if not team_stats:
        warnings.append("No se recibieron estadísticas de equipo completas para estimar posesiones.")
    if not player_stats:
        warnings.append("No se recibieron estadísticas individuales disponibles en esta importación.")
    warnings.extend([
        "Las posesiones se estiman con FGA + 0,44 × FTA − rebotes ofensivos + pérdidas.",
        "La cobertura de box scores varía por competición y partido; no se inventan datos ausentes.",
        "SportAPI.ai no ofrece cuotas de apuestas.",
    ])
    return {
        "games": games,
        "team_stats": team_stats,
        "player_stats": player_stats,
        "games_received": len(games),
        "completed_games": completed,
        "team_stats_rows": len(team_stats),
        "player_stats_rows": len(player_stats),
        "quota": quota,
        "warnings": list(dict.fromkeys(warnings)),
    }


def fixtures(league,date,limit=80):
    """Fetch only the scheduled games for one league/date; one SportAPI request, no box-score fan-out."""
    league=str(league or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9-]{1,80}",league):
        raise SportAPIError("Seleccioná una competición válida.",422)
    try:
        datetime.strptime(str(date),"%Y-%m-%d")
        limit=int(limit)
    except (TypeError,ValueError):
        raise SportAPIError("Indicá una fecha AAAA-MM-DD y un límite válido.",422) from None
    if not 1<=limit<=100:
        raise SportAPIError("El límite del calendario es de 1 a 100 partidos.",422)

    listing=_get("basketball/games/date/"+str(date),
        {"tz":"America/Argentina/Cordoba","league":league},120)
    raw=listing.get("data") or []
    if not isinstance(raw,list):
        raise SportAPIError("Calendario de partidos inválido.")
    from datetime import timezone
    from zoneinfo import ZoneInfo
    local_zone=ZoneInfo("America/Argentina/Cordoba")
    now=datetime.now(timezone.utc)
    final_statuses={"FT","AOT","AWD","CANC","CANCELLED","POSTPONED","PST"}
    results=[]
    for game in raw:
        if not isinstance(game,dict):
            continue
        league_data=game.get("league") or {}
        if str(league_data.get("slug") or league_data.get("id") or "")!=league:
            continue
        status=str(game.get("status") or "").upper()
        if status in final_statuses:
            continue
        start=game.get("start")
        home=game.get("home") or {}
        away=game.get("away") or {}
        gid=game.get("id")
        if gid is None or not start or not home.get("name") or not away.get("name"):
            continue
        try:
            parsed=datetime.fromisoformat(str(start).replace("Z","+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                parsed=parsed.replace(tzinfo=local_zone)
            parsed=parsed.astimezone(timezone.utc)
        except (TypeError,ValueError):
            continue
        # Do not forecast games already in progress or already started.
        if parsed<=now:
            continue
        results.append({
            "fixture_id":"sportapi-"+str(gid),
            "competition":str(league_data.get("name") or league),
            "season":str(game.get("season") or ""),
            "start_time":parsed.isoformat(),
            "home_team":str(home["name"]),
            "away_team":str(away["name"]),
            "status":status or "SCHEDULED",
        })
        if len(results)>=limit:
            break
    results.sort(key=lambda item:item["start_time"])
    return {
        "provider":"SportAPI.ai",
        "date":str(date),
        "count":len(results),
        "fixtures":results,
        "quota":listing.get("_quota"),
        "note":"Se consultó solo el calendario; no se descargaron estadísticas ni cuotas.",
    }
