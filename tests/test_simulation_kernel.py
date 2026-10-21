"""Simulation + BertrandEnv + QLearningPricer must reproduce the fast-path kernel."""
import numpy as np
import pytest

pytest.importorskip("numba")

from tatonnement.core.record import Recorder
from tatonnement.core.simulation import Simulation
from tatonnement.scenarios.bertrand.env import BertrandEnv
from tatonnement.scenarios.bertrand.qlearning.agent import (
    QLearningAgent,
    QLearningPricer,
    kernel_initial,
)
from tatonnement.scenarios.bertrand.qlearning.fast import prepare, run_session, session_seed
from tatonnement.scenarios.bertrand.regulations import PriceFloor


@pytest.mark.parametrize("cfg,regs", [
    ({}, []),
    ({"grid": "wide", "beta": 9.622777683940684e-06, "floor": {"type": "b1", "F": 0.26627198468998864}},
     [PriceFloor("b1", enforcement="monitor")]),
])
def test_simulation_matches_kernel(cfg, regs):
    prep = prepare(dict(cfg))
    root, session, steps = 20261011, 3, 20000
    seed = session_seed(root, session)
    fast = run_session(prep, seed, max_steps=steps, stable_steps=10**9)
    env = BertrandEnv(prep["e"], regulations=regs, F=prep["F"], history_limit=1)
    agents = [QLearningPricer(QLearningAgent.from_prepared(prep, i)) for i in range(2)]
    sim = Simulation(env, agents, periods=steps, seed=root, session=session, rng_mode="kernel",
                     initial=kernel_initial(prep), recorder=Recorder(keep=False))
    res = sim.run()
    assert res.periods == steps
    for i, ag in enumerate(agents):
        c = prep["counts"][i]
        assert np.array_equal(ag.core.Q, fast["Q"][i][:, :c])
    assert agents[0].core.s == fast["last_state"]
