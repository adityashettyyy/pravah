import json
import pytest

from app.engine import Engine, POLICY_ORDER, AGENT
import pipeline


@pytest.fixture(scope="module")
def eng():
    return Engine()


def test_propagation_structure(eng):
    hub = eng.find("Bhiwandi")
    out = eng.propagation(hub, 0.8)
    H = out["hubs"]
    assert H["hop"].iloc[0] == 0 and H["risk"].iloc[0] >= 0.8
    assert H["risk"].between(0, 1).all()
    assert H["hop"].max() <= 3
    assert out["traffic_exposed_pct"] > 0


def test_higher_severity_means_more_exposure(eng):
    hub = eng.find("Bhiwandi")
    assert eng.propagation(hub, 1.0)["traffic_exposed_pct"] >= eng.propagation(hub, 0.3)["traffic_exposed_pct"]


def test_compare_runs_all_policies_on_same_batches(eng):
    hub = eng.find("Bhiwandi")
    res = eng.compare(hub, 0.8, 0)
    assert set(res) == set(POLICY_ORDER)
    tos = {p: [s["to"] for s in r["steps"]] for p, r in res.items()}
    assert all(v == tos["Keep planned route"] for v in tos.values())  # identical manifest for every policy
    assert res["Keep planned route"]["saving_pct"] == 0


def test_pipeline_result_is_json_serialisable(eng):
    hub = eng.find("Bhiwandi")
    res = pipeline.run(eng, hub, 0.8, 0)
    json.dumps(res)
    assert {"propagation", "plan", "policies", "latency_ms"} <= set(res)
    assert len(res["plan"]) == 10
    assert res["latency_ms"]["gnn_propagation"] < 1000  # real-time demo requirement
