"""Adapter for SportAPI.ai basketball endpoints."""
from __future__ import annotations
import json, os, re, threading, time
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

def sync(league,season=None,date=None,limit=20):
    league=str(league or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9-]{1,80}",league): raise SportAPIError("Seleccioná una competición válida.",422)
    try: limit=int(limit)
    except (TypeError,ValueError): raise SportAPIError("Límite inválido.",422) from None
    if not 1<=limit<=40: raise SportAPIError("El límite por importación es de 1 a 40 partidos.",422)
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
    games,team_stats,player_stats=[],[],[]
    warnings=[]; completed=0; quota=listing.get("_quota")
    for g in finals:
        gid=g.get("id"); home=g.get("home") or {}; away=g.get("away") or {}; lg=g.get("league") or {}
        if gid is None or not home.get("name") or not away.get("name") or not g.get("start"):
            warnings.append("Se omitió un partido con campos básicos incompletos."); continue
        eid="sportapi-"+str(gid)
        games.append({"event_id":eid,"competition":str(lg.get("name") or league),
            "season":str(g.get("season") or season or ""),"event_time":str(g.get("start")),
            "home_team":str(home["name"]),"away_team":str(away["name"]),"status":"finished",
            "home_score":home.get("score"),"away_score":away.get("score")})
        completed+=1
        if g.get("has_boxscore") is False:
            warnings.append("Algunos partidos no tienen box score; se importó solo el resultado."); continue
        try:
            details=_get("basketball/games/"+str(int(gid)),ttl=3600)
            detail=details.get("data") or details.get("game") or {}
            stats=detail.get("stats") or details.get("stats") or {}
            quota=details.get("_quota") or quota
            for side,team in (("home",home),("away",away)):
                s=stats.get(side) if isinstance(stats,dict) else None
                if isinstance(s,dict):
                    row=_team_row(eid,str(team["name"]),team.get("score"),s)
                    if row: team_stats.append(row)
            players=_get("basketball/games/"+str(int(gid))+"/players",ttl=3600)
            quota=players.get("_quota") or quota
            player_data=players.get("data") or {}
            player_stats.extend(_player_rows(eid,player_data.get("home"),str(home["name"])))
            player_stats.extend(_player_rows(eid,player_data.get("away"),str(away["name"])))
        except SportAPIError as exc:
            warnings.append(str(exc))
            if exc.status==429: break
    if not team_stats: warnings.append("No se recibieron estadísticas de equipo completas para estimar posesiones.")
    if not player_stats: warnings.append("No se recibieron estadísticas individuales disponibles en esta importación.")
    warnings.extend(["Las posesiones se estiman con FGA + 0,44 × FTA − rebotes ofensivos + pérdidas.",
        "La cobertura de box scores varía por competición y partido; no se inventan datos ausentes.",
        "SportAPI.ai no ofrece cuotas de apuestas."])
    return {"games":games,"team_stats":team_stats,"player_stats":player_stats,
        "games_received":len(games),"completed_games":completed,"team_stats_rows":len(team_stats),
        "player_stats_rows":len(player_stats),"quota":quota,"warnings":list(dict.fromkeys(warnings))}
