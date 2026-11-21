import os,json,time,threading
from urllib.request import Request,urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError,URLError
BASE="https://v1.basketball.api-sports.io"; CACHE={}; LOCK=threading.Lock()
class ProviderError(Exception):
 def __init__(self,msg,status=502): super().__init__(msg); self.status=status
def get(path,params=None,ttl=0):
 key=os.getenv("APISPORTS_KEY","").strip()
 if not key: raise ProviderError("Falta APISPORTS_KEY en Render.",503)
 q=urlencode({k:v for k,v in (params or {}).items() if v not in (None,"")}); url=BASE+"/"+path+("?"+q if q else "")
 with LOCK:
  hit=CACHE.get(url)
  if ttl and hit and hit[0]>time.time(): return hit[1]
 try:
  req=Request(url,headers={"x-apisports-key":key},method="GET")
  with urlopen(req,timeout=25) as r: data=json.loads(r.read(8388608).decode()); quota={"limit":r.headers.get("x-ratelimit-requests-limit"),"remaining":r.headers.get("x-ratelimit-requests-remaining")}
 except HTTPError as e:
  if e.code in (401,403): raise ProviderError("API-Sports rechazó la clave; revisá APISPORTS_KEY.",502)
  if e.code==429: raise ProviderError("Límite de solicitudes de API-Sports alcanzado.",429)
  raise ProviderError("API-Sports HTTP "+str(e.code))
 except (URLError,TimeoutError): raise ProviderError("No se pudo conectar con API-Sports.")
 if data.get("errors"): raise ProviderError("API-Sports: "+str(data["errors"])[:250],422)
 data["_quota"]=quota
 if ttl:
  with LOCK: CACHE[url]=(time.time()+ttl,data)
 return data
def status():
 d=get("status",ttl=60); r=d.get("response") or {}
 if isinstance(r,list): r=r[0] if r else {}
 return {"configured":True,"subscription":r.get("subscription"),"requests":r.get("requests")}
def leagues(season,search=None):
 if not 4<=len(str(season))<=9: raise ProviderError("Temporada inválida (ej. 2026-2027).",422)
 d=get("leagues",{"season":season,"search":search},21600)
 return {"count":d.get("results",0),"leagues":d.get("response") or []}
def num(x):
 try: return float(str(x).replace("%","")) if x not in (None,"-","") else None
 except: return None
def sync(league,season,date=None,limit=40):
 try: league=int(league); limit=int(limit)
 except: raise ProviderError("Liga o límite inválido.",422)
 if league<1 or not 1<=limit<=100: raise ProviderError("Límite permitido: 1 a 100.",422)
 params={"league":league,"season":season}
 if date: params["date"]=date
 raw=get("games",params,120).get("response") or []; games=[]; done=[]
 for x in raw:
  g=x.get("game") or {}; lg=x.get("league") or {}; t=x.get("teams") or {}; h=t.get("home") or {}; a=t.get("away") or {}; s=x.get("scores") or {}; hs=s.get("home") or {}; aw=s.get("away") or {}; st=g.get("status") or {}; code=str(st.get("short") or "").upper(); gid=g.get("id")
  if gid is None or not h.get("name") or not a.get("name"): continue
  finished=code in ("FT","AOT","AWD")
  games.append({"event_id":str(gid),"competition":str(lg.get("name") or ""),"season":str(lg.get("season") or season),"event_time":str(g.get("date") or ""),"home_team":h["name"],"away_team":a["name"],"status":"finished" if finished else code.lower(),"home_score":hs.get("total"),"away_score":aw.get("total")})
  if finished and hs.get("total") is not None and aw.get("total") is not None: done.append((str(gid),h,a,hs,aw))
 stats=[]
 for i in range(0,min(len(done),limit),20):
  batch=done[i:i+20]; d=get("games/statistics/teams",{"ids":"-".join(x[0] for x in batch)},3600); mp={x[0]:x for x in batch}
  for it in d.get("response") or []:
   gid=str((it.get("game") or {}).get("id") or ""); ref=mp.get(gid)
   if not ref: continue
   team=it.get("team") or {}; home=team.get("id")==ref[1].get("id"); name=ref[1]["name"] if home else ref[2]["name"]; st=it.get("statistics") if isinstance(it.get("statistics"),dict) else it
   fg=st.get("fieldGoals") or {}; th=st.get("threePoints") or st.get("threepoint_goals") or {}; ft=st.get("freeThrows") or st.get("freethrows_goals") or {}; rb=st.get("rebounds") or {}
   fga=num(fg.get("attempts")); fgm=num(fg.get("total")); a3=num(th.get("attempts")); m3=num(th.get("total")); fta=num(ft.get("attempts")); ftm=num(ft.get("total")); ore=num(rb.get("offence")); dre=num(rb.get("defense")); tov=num(st.get("turnovers")); pts=num((ref[3] if home else ref[4]).get("total"))
   if None in (fga,fgm,a3,m3,fta,ftm,ore,dre,tov,pts): continue
   poss=fga+.44*fta-ore+tov
   if poss>0: stats.append({"event_id":gid,"team":name,"possessions":round(poss,3),"points":pts,"fga":fga,"fgm":fgm,"fg3a":a3,"fg3m":m3,"fta":fta,"ftm":ftm,"oreb":ore,"dreb":dre,"tov":tov})
 return {"games":games,"team_stats":stats,"player_stats":[],"games_received":len(games),"completed_games":len(done),"team_stats_rows":len(stats),"quota":d.get("_quota") if 'd' in locals() else None,"warnings":["Posesiones estimadas con FGA + 0,44 × FTA − rebotes ofensivos + pérdidas.","La cobertura estadística varía por competición y partido; no se inventan datos.","Esta integración no importa estadísticas individuales de jugadores ni cuotas."]}
