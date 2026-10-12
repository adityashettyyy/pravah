import numpy as np
import pytest

from graph.cascade import Network
from gnn.propagation import MODEL_PATH
from rl.env import RerouteEnv, load_gnn, scenario_pool, OBS_DIM, M, N_BATCHES
from rl.policies import keep, nearest_hub, risk_aware, lookahead
from rl.evaluate import run_episode


@pytest.fixture(scope="module")
def env():
    net = Network()
    return RerouteEnv(net, load_gnn(MODEL_PATH), scenario_pool(net))


def test_reset_and_obs_shape(env):
    obs, _ = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()


def test_masks_always_allow_keep_and_hold(env):
    env.reset(seed=4)
    done = False
    while not done:
        m = env.action_masks()
        assert m.shape == (M + 2,) and m[0] and m[M + 1]
        _, _, done, _, _ = env.step(int(np.flatnonzero(m)[-1]))


def test_masked_actions_never_overload(env):
    for seed in range(15):
        env.reset(seed=seed)
        done = False
        while not done:
            m = env.action_masks()
            a = int(np.random.default_rng(seed).choice(np.flatnonzero(m)))
            _, _, done, _, info = env.step(a)
            if 1 <= a <= M:
                assert info["overload"] == 0


def test_episode_is_deterministic_given_seed(env):
    a = run_episode(env, risk_aware, 11)
    b = run_episode(env, risk_aware, 11)
    assert a["reward"] == pytest.approx(b["reward"]) and a["excess"] == pytest.approx(b["excess"])


def test_sensible_policy_ordering_on_average(env):
    seeds = range(500, 520)
    mean = lambda p: np.mean([run_episode(env, p, s)["excess"] for s in seeds])
    k, ra, la = mean(keep), mean(risk_aware), mean(lookahead)
    assert ra < k and la < k
