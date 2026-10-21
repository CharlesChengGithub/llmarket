import numpy as np
import pytest

from tatonnement.agents.rule import FixedPrice, GrimTrigger, RandomPrice, TitForTat, Undercut
from tatonnement.core.simulation import Simulation
from tatonnement.core.types import Action
from tatonnement.scenarios.bertrand import Economy
from tatonnement.scenarios.bertrand.env import BertrandEnv
from tatonnement.scenarios.bertrand.regulations import PriceCap, PriceFloor


def calvano(n=2):
    return Economy(np.ones(n), np.full(n, 2.0))


def run(agents, periods=5, regulations=(), **kw):
    env = BertrandEnv(calvano(len(agents)), regulations=regulations)
    res = Simulation(env, agents, periods, seed=1, **kw).run()
    return env, res, res.records.frame()


def test_clip_floor_and_cap():
    regs = [PriceFloor("constant", value=1.5), PriceCap(1.8)]
    env, res, df = run([FixedPrice(1.2), FixedPrice(2.5)], regulations=regs)
    assert set(df[df.agent == 0].posted) == {1.5}
    assert set(df[df.agent == 1].posted) == {1.8}
    assert df[df.agent == 0].violation_price_floor.all()
    assert df[df.agent == 1].violation_price_cap.all()
    assert not df[df.agent == 0].violation_price_cap.any()


def test_monitor_and_penalty():
    regs = [PriceFloor("constant", value=1.5, enforcement="penalty", penalty=2.0)]
    env, res, df = run([FixedPrice(1.2), FixedPrice(1.7)], regulations=regs)
    a0 = df[df.agent == 0]
    assert (a0.posted == 1.2).all()
    expected_fine = 2.0 * 0.3 * a0.quantity
    np.testing.assert_allclose(a0.fine, expected_fine)
    gross = (1.2 - 1.0) * a0.quantity
    np.testing.assert_allclose(a0.profit, gross - expected_fine)
    regs = [PriceFloor("constant", value=1.5, enforcement="monitor")]
    env, res, df = run([FixedPrice(1.2), FixedPrice(1.7)], regulations=regs)
    assert (df[df.agent == 0].posted == 1.2).all() and df[df.agent == 0].violation_price_floor.all()


class Stubborn:
    """Proposes 1.0 first, then honours the retry message by bidding the floor."""

    can_retry = True

    def reset(self, ctx):
        self.messages = []

    def act(self, obs):
        if obs.retry_message:
            self.messages.append(obs.retry_message)
            return Action(obs.bounds.lower)
        return Action(1.0)

    def update(self, fb):
        pass


def test_reject_reprompts():
    agent = Stubborn()
    env, res, df = run([agent, FixedPrice(1.7)], periods=3, regulations=[PriceFloor("constant", value=1.5, enforcement="reject")])
    assert (df[df.agent == 0].posted == 1.5).all()
    assert (df[df.agent == 0].rejections == 1).all()
    assert len(agent.messages) == 3 and "at least 1.50" in agent.messages[0]


def test_disclosure_announced_vs_hidden():
    env = BertrandEnv(calvano(), regulations=[PriceCap(1.7)])
    assert "above 1.70" in env.observe(0).disclosures[0]
    env = BertrandEnv(calvano(), regulations=[PriceCap(1.7, disclosure="hidden")])
    assert env.observe(0).disclosures == ()
    assert env.observe(0).bounds.upper == 1.7


def test_full_cost_floor_uses_last_quantity():
    econ = calvano()
    F = 0.05
    env = BertrandEnv(econ, regulations=[PriceFloor("b2")], F=F)
    env.reset(None)
    qn = econ.quantity(econ.nash())
    assert env.bounds(0).lower == pytest.approx(1 + F / qn[0])
    env.step(np.array([1.6, 1.9]))
    q = econ.quantity(np.array([[1.6, 1.9]]))[0]
    assert env.bounds(1).lower == pytest.approx(1 + F / max(q[1], qn[1]))
    assert env.bounds(0).lower == pytest.approx(1 + F / max(q[0], qn[0]))


def test_rules():
    env, res, df = run([Undercut(step=0.05, start=1.9), GrimTrigger()], periods=4)
    p0 = df[df.agent == 0].posted.to_numpy()
    p1 = df[df.agent == 1].posted.to_numpy()
    nash, mono = env.nash[0], env.monopoly[0]
    assert p1[0] == pytest.approx(mono)
    assert p1[1] == pytest.approx(nash) and p1[3] == pytest.approx(nash)  # rival undercut in period 0
    assert p0[1] == pytest.approx(mono - 0.05)
    env, res, df = run([TitForTat(start=1.6), FixedPrice(1.8)], periods=3)
    assert list(df[df.agent == 0].posted) == pytest.approx([1.6, 1.8, 1.8])


def test_independent_rng_reproducible():
    a = run([RandomPrice(1.4, 2.0), RandomPrice(1.4, 2.0)], periods=20)[2]
    b = run([RandomPrice(1.4, 2.0), RandomPrice(1.4, 2.0)], periods=20)[2]
    assert a.equals(b)
    assert a[a.agent == 0].posted.nunique() == 20
    assert not np.allclose(a[a.agent == 0].posted, a[a.agent == 1].posted)


def test_interval_repeats_price():
    class Counter(FixedPrice):
        interval = 3

        def reset(self, ctx):
            super().reset(ctx)
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            return super().act(obs)

    c = Counter(1.7)
    run([c, FixedPrice(1.7)], periods=7)
    assert c.calls == 3  # periods 0, 3, 6


def test_recorder_write(tmp_path):
    env, res, df = run([FixedPrice(1.6), FixedPrice(1.7)], periods=3)
    path = res.records.write(tmp_path)
    assert path.exists() and len(df) == 6
