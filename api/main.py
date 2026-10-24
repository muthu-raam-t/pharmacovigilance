"""Pharmacovigilance Advisory API.

Endpoints
  GET  /health                               service + database status
  POST /analyze                              full drug-disease analysis (cached in Redis)
  GET  /drug/{drug}/side-effects             ranked side-effect profile
  GET  /disease/{disease}/alternatives       alternative drugs for a disease
"""
import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # pydantic v2
    from pydantic import field_validator as _validator
except ImportError:  # pydantic v1
    from pydantic import validator as _validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
import pv_config as cfg  # noqa: E402
from analyze_drug_disease import analyze  # noqa: E402
from alternative_drugs import suggest_alternatives_for_disease  # noqa: E402
from db import check_connections  # noqa: E402
from ranking_engine import get_ranked_side_effects  # noqa: E402
from text_utils import normalize_name  # noqa: E402

log = logging.getLogger("pharmacovigilance.api")
CACHE_VERSION = "v2"

app = FastAPI(title="Pharmacovigilance Advisory API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

try:
    import redis
    _redis = redis.Redis.from_url(cfg.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
except Exception:  # noqa: BLE001
    _redis = None


def cache_get(key):
    if _redis is None:
        return None
    try:
        raw = _redis.get(key)
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001 - the cache is optional
        return None


def cache_set(key, value):
    if _redis is None:
        return
    try:
        _redis.setex(key, cfg.CACHE_TTL_SECONDS, json.dumps(value, default=str))
    except Exception:  # noqa: BLE001
        pass


class AnalyzeRequest(BaseModel):
    drug: str = Field(..., min_length=1, max_length=100)
    disease: str = Field(..., min_length=1, max_length=150)

    @_validator("drug", "disease")
    def not_blank(cls, v):
        v = " ".join(v.split())
        if not v:
            raise ValueError("must not be blank")
        return v


@app.get("/health")
def health_check():
    status = check_connections()
    try:
        status["redis"] = "ok" if _redis is not None and _redis.ping() else "unavailable"
    except Exception:  # noqa: BLE001
        status["redis"] = "unavailable"
    status["status"] = "ok" if status["postgres"] == "ok" and status["neo4j"] == "ok" else "degraded"
    return status


@app.post("/analyze")
def analyze_drug_disease(request: AnalyzeRequest):
    key = f"pv:{CACHE_VERSION}:analyze:{normalize_name(request.drug)}|{normalize_name(request.disease)}"
    cached = cache_get(key)
    if cached is not None:
        cached["cached"] = True
        return cached
    try:
        result = analyze(request.drug, request.disease)
    except Exception as e:  # noqa: BLE001
        log.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}: {e}")
    cache_set(key, result)
    result["cached"] = False
    return result


@app.get("/drug/{drug}/side-effects")
def drug_side_effects(drug: str, limit: int = Query(50, ge=1, le=1000)):
    try:
        effects = get_ranked_side_effects(drug)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Lookup failed: {type(e).__name__}: {e}")
    return {"drug": drug, "total_side_effects": len(effects), "side_effects": effects[:limit]}


@app.get("/disease/{disease}/alternatives")
def disease_alternatives(disease: str, exclude: str = None, top_n: int = Query(5, ge=1, le=20)):
    try:
        return suggest_alternatives_for_disease(disease, exclude_drug=exclude, top_n=top_n)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Lookup failed: {type(e).__name__}: {e}")
