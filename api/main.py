"""Pravah REST API: the closed loop (GNN propagation -> RL rerouting -> before/after) over HTTP.

    uvicorn api.main:app --reload            # interactive docs at http://127.0.0.1:8000/docs

Endpoints
    GET  /health         liveness plus which agent is loaded
    GET  /hubs           hubs that can be disrupted, busiest first (optional ?q= name filter)
    POST /propagation    GNN risk spread from one hub (no rerouting)
    POST /simulate       full loop: propagation, rerouting plan, all policies compared

Same assumptions as the dashboard and CLI: the cascade rule and every delay constant are assumed
(see README.md), shipment batches are synthetic.
"""
import os
import sys
import threading
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Query

import pipeline
from api.schemas import (ExposedHub, Health, HubList, HubSummary, PropagationResponse,
                         SimulationRequest)
from app.engine import Engine
from graph.cascade import BETA, DECAY, HOPS

VERSION = "1.0.0"
_LOCK = threading.Lock()  # Engine owns a stateful RL environment, so calls are serialised


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Load GNN, RL agent and network once per process."""
    return Engine()


app = FastAPI(
    title="Pravah API",
    version=VERSION,
    description="Predict how a hub disruption spreads through the Delhivery network and get a rerouting plan.",
)


def _resolve(eng: Engine, text: str) -> int:
    try:
        hub = eng.find(text)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no hub matching '{text}'")
    if hub not in {int(h) for h in eng.pool}:
        raise HTTPException(
            status_code=422,
            detail=f"{eng.label(hub)} has too few downstream hubs to simulate, pick a larger hub (see GET /hubs)")
    return hub


@app.get("/health", response_model=Health)
def health():
    eng = get_engine()
    return Health(status="ok", agent="MaskablePPO" if eng.agent_is_rl else "risk-aware rule (fallback)",
                  hubs=int(eng.net.n), disruptable_hubs=int(len(eng.pool)), version=VERSION)


@app.get("/hubs", response_model=HubList)
def hubs(q: str = Query("", description="Case-insensitive name filter"),
         limit: int = Query(50, ge=1, le=500)):
    eng = get_engine()
    rows = [HubSummary(index=i, name=label, state=str(eng.net.state[i]), traffic=round(float(eng.traffic[i]), 1))
            for i, label in eng.hub_options() if q.lower() in label.lower()]
    return HubList(total=len(rows), hubs=rows[:limit])


@app.post("/propagation", response_model=PropagationResponse)
def propagation(req: SimulationRequest):
    eng = get_engine()
    with _LOCK:
        hub = _resolve(eng, req.hub)
        prop = eng.propagation(hub, req.severity)
    top = prop["hubs"].iloc[1:].sort_values("risk", ascending=False).head(req.top_n)
    exposed = [ExposedHub(hub=r.hub, state=str(r.state), risk=round(float(r.risk), 3), hop=int(r.hop),
                          lat=float(r.lat), lon=float(r.lon)) for r in top.itertuples() if r.risk > 0.01]
    return PropagationResponse(
        disrupted_hub=eng.label(hub), severity=req.severity, hubs_at_risk=prop["n_at_risk"],
        traffic_exposed_pct=round(prop["traffic_exposed_pct"], 2), propagation=exposed,
        latency_ms=round(prop["latency_ms"], 1),
        assumptions={"cascade": f"BETA={BETA}, DECAY={DECAY}, {HOPS} hops (assumed, no cascade in the data)"})


@app.post("/simulate")
def simulate(req: SimulationRequest):
    """Full closed loop. Response is the same JSON as `python pipeline.py --out`."""
    eng = get_engine()
    with _LOCK:
        hub = _resolve(eng, req.hub)
        res = pipeline.run(eng, hub, req.severity, req.seed)
    res["propagation"] = res["propagation"][:req.top_n]
    return res
