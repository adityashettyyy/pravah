"""Request / response models for the Pravah REST API."""
from typing import List, Optional

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    hub: str = Field(..., min_length=2, description="Part of a hub name, e.g. 'Bhiwandi'. The busiest match is used.")
    severity: float = Field(0.8, ge=0.3, le=1.0, description="Disruption severity at the source hub.")
    seed: int = Field(0, ge=0, le=9999, description="Seed for the synthetic shipment batches.")
    top_n: int = Field(15, ge=1, le=100, description="How many most-exposed hubs to return.")


class HubSummary(BaseModel):
    index: int
    name: str
    state: str
    traffic: float


class HubList(BaseModel):
    total: int
    hubs: List[HubSummary]


class Health(BaseModel):
    status: str
    agent: str
    hubs: int
    disruptable_hubs: int
    version: str


class ExposedHub(BaseModel):
    hub: str
    state: str
    risk: float
    hop: int
    lat: float
    lon: float


class PropagationResponse(BaseModel):
    disrupted_hub: str
    severity: float
    hubs_at_risk: int
    traffic_exposed_pct: float
    propagation: List[ExposedHub]
    latency_ms: float
    assumptions: Optional[dict] = None
