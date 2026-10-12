import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["hubs"] > 0 and body["disruptable_hubs"] > 0


def test_hubs_listing_and_filter(client):
    allh = client.get("/hubs", params={"limit": 5}).json()
    assert allh["total"] >= len(allh["hubs"]) == 5
    traffic = [h["traffic"] for h in allh["hubs"]]
    assert traffic == sorted(traffic, reverse=True)  # busiest first
    hits = client.get("/hubs", params={"q": "bhiwandi"}).json()
    assert hits["total"] > 0 and all("bhiwandi" in h["name"].lower() for h in hits["hubs"])


def test_propagation_endpoint(client):
    r = client.post("/propagation", json={"hub": "Bhiwandi", "severity": 0.8, "top_n": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["propagation"]) <= 5
    assert all(0 <= h["risk"] <= 1 and 1 <= h["hop"] <= 3 for h in body["propagation"])
    assert body["traffic_exposed_pct"] > 0


def test_simulate_matches_cli_pipeline_shape(client):
    r = client.post("/simulate", json={"hub": "Bhiwandi", "severity": 0.8, "seed": 0})
    assert r.status_code == 200
    body = r.json()
    assert {"propagation", "plan", "policies", "latency_ms"} <= set(body)
    assert len(body["plan"]) == 10
    assert body["policies"]["Keep planned route"]["saving_pct"] == 0


def test_simulate_is_deterministic_for_a_seed(client):
    req = {"hub": "Bhiwandi", "severity": 0.7, "seed": 3}
    a, b = client.post("/simulate", json=req).json(), client.post("/simulate", json=req).json()
    assert a["policies"] == b["policies"] and a["plan"] == b["plan"]


def test_unknown_hub_is_404(client):
    assert client.post("/simulate", json={"hub": "Atlantis", "severity": 0.8}).status_code == 404


@pytest.mark.parametrize("payload", [{"hub": "Bhiwandi", "severity": 1.5}, {"hub": "Bhiwandi", "severity": 0.1},
                                     {"hub": "x"}, {"severity": 0.8}])
def test_invalid_payload_is_422(client, payload):
    assert client.post("/simulate", json=payload).status_code == 422
