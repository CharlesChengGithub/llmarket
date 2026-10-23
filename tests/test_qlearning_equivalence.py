"""The step-level agent must replay the fast-path kernel exactly."""
import numpy as np
import pytest

pytest.importorskip("numba")

from llmarket.scenarios.bertrand.qlearning.agent import QLearningAgent, selfplay, to_index
from llmarket.scenarios.bertrand.qlearning.fast import prepare, run_session

CONFIGS = {
    "baseline": {},
    "memoryless": {"k": 0, "delta": 0, "alpha": 0.25, "beta": 0.0001},
    "constant_floor": {"grid": "wide", "floor": {"type": "constant", "value": 1.1177411708264418}, "beta_rule": "nu"},
    "b1_floor": {"grid": "wide", "beta": 9.622777683940684e-06, "floor": {"type": "b1", "F": 0.26627198468998864}},
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_step_agent_matches_kernel(name):
    cfg = CONFIGS[name]
    prep = prepare(dict(cfg))
    alpha = float(cfg.get("alpha", 0.15))
    steps = 30000
    fast = run_session(prep, seed=12345, alpha=alpha, max_steps=steps, stable_steps=10**9)
    step = selfplay(prep, seed=12345, alpha=alpha, max_steps=steps, stable_steps=10**9)
    assert step["steps"] == fast["steps"] == steps
    assert step["last_state"] == fast["last_state"]
    assert step["last_policy_change"] == fast["last_policy_change"]
    n = prep["prices"].shape[0]
    for i in range(n):
        c = prep["counts"][i]
        q_step = step["Q"][i]
        assert np.array_equal(q_step[:, :c], fast["Q"][i][:, :c])
        assert np.array_equal(step["policy"][i], fast["policy"][i].astype(np.int64))


def test_memoryless_converges_like_kernel():
    cfg = CONFIGS["memoryless"]
    prep = prepare(dict(cfg))
    fast = run_session(prep, seed=7, alpha=0.25)
    step = selfplay(prep, seed=7, alpha=0.25)
    assert fast["converged"] and step["converged"]
    assert step["steps"] == fast["steps"]


def test_to_index_nearest_and_ties():
    levels = np.array([1.0, 2.0, 3.0])
    assert to_index(levels, 0.2) == 0
    assert to_index(levels, 1.6) == 1
    assert to_index(levels, 1.5) == 0  # tie goes low
    assert to_index(levels, 9.0) == 2
    assert to_index(levels, 2.9, "floor") == 1
    assert to_index(levels, 2.1, "ceil") == 2


def test_frozen_agent_does_not_learn():
    prep = prepare({})
    ag = QLearningAgent.from_prepared(prep, 0, learn=False, explore=False)
    ag.reset(rng=np.random.RandomState(0), state=0)
    q_before = ag.Q.copy()
    p = ag.act()
    ag.update(np.array([p, p]), reward=1.0)
    assert np.array_equal(ag.Q, q_before)
    assert p == prep["prices"][0, ag.policy[0]]


@pytest.mark.slow
def test_step_agent_full_session_matches_golden():
    """Whole baseline session 0 of Cheng (2026) through the step-level agent (~2 minutes)."""
    prep = prepare({})
    step = selfplay(prep, seed=1095125285)
    assert (step["steps"], step["last_state"], step["last_policy_change"]) == (2185462, 144, 2085462)
