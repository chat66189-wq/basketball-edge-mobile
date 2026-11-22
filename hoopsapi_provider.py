"""HoopsAPI adapter for basketball fixtures and bookmaker odds."""
from __future__ import annotations

import json
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://hoopsapi.com/api"
_CACHE = {}
_LOCK = threading.Lock()


class HoopsAPIError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _get(path: str, params: dict | None = None, ttl: int = 120) -> dict:
    api_key = os.environ.get("HOOPSAPI_KEY", "").strip()
    if not api_key:
        raise HoopsAPIError("Falta HOOPSAPI_KEY en las variables de entorno de Render.", 503)

    query = urlencode({k: v for k, v in (params or {}).items() if v not in (None, "")})
    url = BASE + "/" + path.lstrip("/") + ("?" + query if query else "")
    with _LOCK:
        hit = _CACHE.get(url)
        if ttl and hit and hit[0] > time.time():
            return hit[1]

    try:
        request = Request(
            url,
            headers={"Authorization": "Bearer " + api_key, "Accept": "application/json"},
            method="GET",
        )
        with urlopen(request, timeout=25) as response:
            data = json.loads(response.read(8 * 1024 * 1024).decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise HoopsAPIError("HoopsAPI rechazó la clave. Revisá HOOPSAPI_KEY en Render.", 502) from exc
        if exc.code == 429:
            raise HoopsAPIError("Se alcanzó el límite de solicitudes de HoopsAPI. El plan gratuito permite 10 solicitudes por día.", 429) from exc
        raise HoopsAPIError("HoopsAPI respondió HTTP " + str(exc.code) + ".", 502) from exc
    except (URLError, TimeoutError):
        raise HoopsAPIError("No se pudo conectar con HoopsAPI.") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HoopsAPIError("HoopsAPI devolvió una respuesta JSON inválida.") from None

    if not isinstance(data, dict) or not isinstance(data.get("games", []), list):
        raise HoopsAPIError("HoopsAPI devolvió un formato de partidos inesperado.")
    if ttl:
        with _LOCK:
            _CACHE[url] = (time.time() + ttl, data)
    return data


def status() -> dict:
    return {
        "configured": bool(os.environ.get("HOOPSAPI_KEY", "").strip()),
        "provider": "HoopsAPI",
        "free_plan_requests_per_day": 10,
        "note": "La clave se almacena en Render; la llamada de partidos obtiene cuotas y calendarios de múltiples ligas.",
    }


def games() -> dict:
    data = _get("/v1/games", ttl=600)
    items = data.get("games") or []
    if not isinstance(items, list):
        raise HoopsAPIError("La lista de partidos de HoopsAPI es inválida.")
    return {"count": len(items), "games": items}
