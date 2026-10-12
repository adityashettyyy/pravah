import numpy as np
import pytest

from graph.cascade import Network, shares, simulate
from novelty.scenarios import multi_source_cascade, run_scenario


@pytest.fixture(scope="module")
def net():
    return Network()


def busy_hub(net):
    return int(np.argmax(net.inbound + np.bincount(net.src, weights=net.w, minlength=net.n)))


def test_shares_are_valid(net):
    ego = net.ego(busy_hub(net))
    p = shares(ego, ego["src"], ego["dst"], ego["w"])
    assert (p >= 0).all() and (p <= 1).all()


def test_source_always_disrupted_and_severity_is_monotone(net):
    ego = net.ego(busy_hub(net))
    p = shares(ego, ego["src"], ego["dst"], ego["w"])
    lo = simulate(net.vuln[ego["nodes"]], ego["src"], ego["dst"], p, 0.2, runs=300, rng=np.random.default_rng(0))
    hi = simulate(net.vuln[ego["nodes"]], ego["src"], ego["dst"], p, 1.0, runs=300, rng=np.random.default_rng(0))
    assert lo[0] == 1.0 and hi[0] == 1.0
    assert hi[1:].sum() >= lo[1:].sum()


def test_multi_source_edge_cases(net):
    zero = np.zeros(net.n)
    p_all, p_casc = multi_source_cascade(net, zero, runs=50)
    assert p_all.sum() == 0 and p_casc.sum() == 0
    one = np.zeros(net.n)
    h = busy_hub(net)
    one[h] = 1.0
    p_all, p_casc = multi_source_cascade(net, one, runs=100, rng=np.random.default_rng(1))
    assert p_all[h] == 1.0 and p_casc[h] == 0.0
    assert p_all.sum() > 1.0


def test_regional_scenario_outputs(net):
    r = run_scenario(net, "state", runs=60, state="Maharashtra", severity=0.7)
    assert 0 < r["traffic_exposed_pct"] <= 100
    assert r["expected_hubs_disrupted"] >= r["directly_hit_hubs"] * 0.5
    assert "Maharashtra" in r["regions"]
