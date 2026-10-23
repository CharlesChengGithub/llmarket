"""Step-level tabular Q-learning pricing agent.

The agent reproduces the fast-path kernel (:mod:`.fast`) exactly when it is given the same
tables, the same rewards and a random stream that yields the kernel's draws in the kernel's
order (see :func:`selfplay`). Unlike the kernel it observes prices, not action indices, so it
can play against agents that post arbitrary prices (LLMs, rules): every firm's last price is
mapped to that firm's price grid to form the state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fast import greedy_row


def to_index(levels: np.ndarray, price: float, mode: str = "nearest") -> int:
    """Index of the grid level that represents price. Ties go to the lower level."""
    levels = np.asarray(levels)
    if mode == "nearest":
        j = int(np.searchsorted(levels, price, side="left"))
        if j <= 0:
            return 0
        if j >= len(levels):
            return len(levels) - 1
        return j if levels[j] - price < price - levels[j - 1] else j - 1
    if mode == "floor":
        return int(np.clip(np.searchsorted(levels, price, side="right") - 1, 0, len(levels) - 1))
    if mode == "ceil":
        return int(np.clip(np.searchsorted(levels, price, side="left"), 0, len(levels) - 1))
    raise ValueError(mode)


@dataclass
class QLearningAgent:
    """Epsilon-greedy tabular Q-learning with exponentially decaying exploration.

    Parameters
    ----------
    index: this firm's position in the price profile.
    levels: price grid of every firm (model units), used to encode states; levels[index] is
        this agent's action set.
    q0: initial Q table, shape (S, M) with S = prod(len(levels)) if k == 1 else 1.
    starts: optional table of the first feasible action in each state (price floors). If
        omitted, the lower bound passed to :meth:`act` determines feasibility.
    learn / explore: False freezes the table / switches to purely greedy play.
    """

    index: int
    levels: list
    q0: np.ndarray
    alpha: float = 0.15
    beta: float = 4e-6
    delta: float = 0.95
    k: int = 1
    starts: np.ndarray | None = None
    epsilon_start: int = 0
    learn: bool = True
    explore: bool = True
    price_map: str = "nearest"
    Q: np.ndarray = field(init=False, repr=False)
    policy: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.levels = [np.asarray(x, dtype=float) for x in self.levels]
        counts = np.array([len(x) for x in self.levels], dtype=np.int64)
        self.count = int(counts[self.index])
        self.strides = np.array([int(np.prod(counts[j + 1:])) for j in range(len(counts))], np.int64)
        self.n_states = int(np.prod(counts)) if self.k else 1
        if self.q0.shape[0] != self.n_states:
            raise ValueError(f"q0 has {self.q0.shape[0]} states, expected {self.n_states}")
        self.reset()

    # -- state handling -------------------------------------------------------------------
    def encode(self, prices) -> int:
        """State index of a price profile (0 when the agent is memoryless)."""
        if not self.k:
            return 0
        return int(sum(to_index(self.levels[j], float(prices[j]), self.price_map) * self.strides[j]
                       for j in range(len(self.levels))))

    def start_for(self, s: int, lower: float | None) -> int:
        if self.starts is not None:
            return int(self.starts[s])
        if lower is None or not np.isfinite(lower):
            return 0
        return int(min(np.searchsorted(self.levels[self.index], lower, side="left"), self.count - 1))

    # -- agent protocol -------------------------------------------------------------------
    def reset(self, rng=None, state: int = 0, t: int = 0) -> None:
        self.Q = np.array(self.q0, dtype=float, copy=True)
        self.policy = np.empty(self.n_states, np.int64)
        for s in range(self.n_states):
            self.policy[s] = greedy_row(self.Q[s], self.start_for(s, None), self.count)
        self.rng = rng
        self.s = int(state)
        self.t = int(t)
        self.changed = False

    def epsilon(self) -> float:
        return float(np.exp(-self.beta * (self.t + self.epsilon_start))) if self.explore else 0.0

    def act(self, lower: float | None = None) -> float:
        """Draw two uniforms (always, to keep common random numbers) and return a price."""
        u = self.rng.random()
        v = self.rng.random()
        start = self.start_for(self.s, lower)
        a = int(self.policy[self.s])
        if a < start:  # floor rose since the policy was last updated in this state
            a = int(greedy_row(self.Q[self.s], start, self.count))
        if u < self.epsilon():
            a = start + int(v * (self.count - start))
        self.a = a
        self.start = start
        return float(self.levels[self.index][a])

    def update(self, posted, reward: float, next_lower: float | None = None) -> None:
        """Observe the posted price profile and own profit; update Q at the previous state."""
        sn = self.encode(posted)
        self.changed = False
        if self.learn:
            start_n = self.start_for(sn, next_lower)
            greedy_n = int(self.policy[sn])
            if greedy_n < start_n:
                greedy_n = int(greedy_row(self.Q[sn], start_n, self.count))
            target = reward + self.delta * self.Q[sn, greedy_n]
            s, a = self.s, self.a
            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * target
            old = self.policy[s]
            self.policy[s] = greedy_row(self.Q[s], self.start, self.count)
            self.changed = bool(old != self.policy[s])
        self.s = sn
        self.t += 1

    @classmethod
    def from_prepared(cls, prep: dict, index: int, Q=None, use_starts: bool = True, **kwargs) -> "QLearningAgent":
        """Build an agent from :func:`.fast.prepare`, optionally loading a trained Q table.

        use_starts=True takes feasibility from the prepared floor table (grid markets, exact
        kernel equivalence); False takes it from the regulation bounds each period (mixed
        markets where rivals post off-grid prices).
        """
        n = prep["prices"].shape[0]
        levels = [prep["prices"][j, : prep["counts"][j]] for j in range(n)]
        q0 = prep["q0"][index] if Q is None else Q[index]
        q0 = q0[:, : prep["counts"][index]]
        kwargs.setdefault("beta", prep["beta"])
        kwargs.setdefault("delta", prep["delta"])
        starts = prep["starts"][index] if use_starts else None
        return cls(index=index, levels=levels, q0=q0, k=prep["k"], starts=starts, **kwargs)


def kernel_stream(seed: int):
    """Random stream identical to the fast-path kernel's draws after np.random.seed(seed)."""
    return np.random.RandomState(int(seed))


def selfplay(prep: dict, seed: int, alpha: float = 0.15, max_steps: int = 10**9,
             stable_steps: int = 100000, epsilon_start: int = 0, record_actions: bool = False) -> dict:
    """Reference step-level self-play loop; mirrors :func:`.fast.train` period by period."""
    n = prep["prices"].shape[0]
    rng = kernel_stream(seed)
    s0 = int(prep["initial_states"][int(rng.random() * len(prep["initial_states"]))])
    agents = [QLearningAgent.from_prepared(prep, i, alpha=alpha, epsilon_start=epsilon_start) for i in range(n)]
    for ag in agents:
        ag.reset(rng=rng, state=s0)
    e = prep["e"]
    F = prep["F"]
    stable = 0
    last_change = 0
    actions = [] if record_actions else None
    t = 0
    for t in range(max_steps):
        posted = np.array([ag.act() for ag in agents])
        rewards = e.profit(posted[None, :], F)[0]
        for i, ag in enumerate(agents):
            ag.update(posted, float(rewards[i]))
        if record_actions:
            actions.append([ag.a for ag in agents])
        if any(ag.changed for ag in agents):
            stable = 0
            last_change = t + 1
        else:
            stable += 1
        if stable >= stable_steps:
            break
    Q = np.stack([ag.Q for ag in agents]) if len(set(ag.count for ag in agents)) == 1 else [ag.Q for ag in agents]
    return dict(Q=Q, policy=np.stack([ag.policy for ag in agents]), last_state=agents[0].s,
                steps=t + 1, converged=stable >= stable_steps, last_policy_change=last_change,
                actions=np.array(actions) if record_actions else None)


class QLearningPricer:
    """Adapter that plugs a :class:`QLearningAgent` into :class:`llmarket.core.Simulation`."""

    def __init__(self, core: QLearningAgent, name: str = "qlearning"):
        self.core = core
        self.name = name

    @property
    def learning(self) -> bool:
        return self.core.learn

    @property
    def changed(self) -> bool:
        return self.core.changed

    def reset(self, ctx) -> None:
        initial = ctx.market.get("initial_prices") or ctx.market["nash_prices"]
        self.core.reset(rng=ctx.rng, state=self.core.encode(initial), t=0)

    def act(self, obs):
        from ....core.types import Action

        price = self.core.act(lower=obs.bounds.lower)
        return Action(price, {"agent": self.name, "explored": bool(self.core.a != self.core.policy[self.core.s])})

    def update(self, fb) -> None:
        self.core.update(fb.posted, fb.own_profit, next_lower=fb.next_bounds.lower)


def kernel_initial(prep: dict):
    """Initial-profile sampler that consumes the shared stream exactly as the kernel does."""
    def draw(rng):
        states = prep["initial_states"]
        return prep["profiles"][int(states[int(rng.random() * len(states))])]
    return draw
