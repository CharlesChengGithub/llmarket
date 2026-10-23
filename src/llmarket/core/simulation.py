"""The period loop: observe, act, enforce regulations, step the market, give feedback, record."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .record import Recorder
from .rng import agent_generators, kernel_seed
from .types import Action, AgentContext, Bounds, Feedback, StopSession


@dataclass
class SessionResult:
    session: int
    periods: int
    stop_reason: str
    records: Recorder
    agents: list = field(repr=False, default_factory=list)


class Simulation:
    """Runs one session of a repeated game.

    rng_mode:
      'independent'  each agent gets its own Generator spawned from (seed, session);
      'kernel'       all agents share one RandomState(kernel_seed(seed, session)) and draw in
                     index order, as the Q-learning fast path does (bit-for-bit compatibility).
    initial: previous-period action profile before t = 0, or a callable rng -> profile.
    stop_when_stable: stop once no learning agent changed its policy for this many periods.
    before_period: called with t before each period; a returned string stops the session with
      that reason (used by the LLM budget guard).
    max_workers: agents with ``concurrent = True`` (LLM agents) act in parallel threads; the
      others act first, sequentially in index order, so shared random streams keep their order.
    An agent may raise :class:`StopSession` to end the session; the unfinished period is dropped.
    """

    def __init__(self, env, agents: Sequence[Any], periods: int, seed: int = 0, session: int = 0,
                 rng_mode: str = "independent", initial: np.ndarray | Callable | None = None,
                 stop_when_stable: int | None = None, recorder: Recorder | None = None,
                 on_period: Callable[[int, dict], bool] | None = None,
                 before_period: Callable[[int], str | None] | None = None, max_workers: int | None = None):
        if len(agents) != env.n_agents:
            raise ValueError("number of agents does not match the environment")
        self.env = env
        self.agents = list(agents)
        self.periods = int(periods)
        self.seed = int(seed)
        self.session = int(session)
        self.rng_mode = rng_mode
        self.initial = initial
        self.stop_when_stable = stop_when_stable
        self.recorder = recorder if recorder is not None else Recorder()
        self.on_period = on_period
        self.before_period = before_period
        self.max_workers = max_workers

    def _rngs(self):
        n = len(self.agents)
        if self.rng_mode == "kernel":
            shared = np.random.RandomState(kernel_seed(self.seed, self.session))
            return [shared] * n, shared
        if self.rng_mode == "independent":
            gens = agent_generators(self.seed, self.session, n)
            return gens[:n], gens[n]
        raise ValueError(self.rng_mode)

    def _propose(self, i: int, t: int, last: list) -> Action:
        agent = self.agents[i]
        interval = int(getattr(agent, "interval", 1))
        if t % interval and last[i] is not None:
            return Action(last[i].value, {"repeated": True})
        return agent.act(self.env.observe(i))

    def _enforce(self, i: int, action: Action) -> tuple[float, dict]:
        """Apply clip/reject regulations; return the posted value and violation flags."""
        agent = self.agents[i]
        regs = self.env.regulations
        state = self.env.reg_state()
        flags = {}
        hard = Bounds()
        for r in regs:
            b = r.bounds(i, state)
            if r.enforcement in ("clip", "reject"):
                hard = hard.intersect(b)
        proposed = action.value
        rejects = [r for r in regs if r.enforcement == "reject"]
        if rejects and getattr(agent, "can_retry", False):
            limit = max(r.max_rejections for r in rejects)
            tries = 0
            while not hard.contains(proposed) and tries < limit:
                tries += 1
                obs = self.env.observe(i)
                obs.retry_message = self.env.rejection_message(i, proposed, hard)
                action = agent.act(obs)
                proposed = action.value
            flags["rejections"] = tries
        flags["proposed"] = proposed
        flags["bounds_conflict"] = bool(hard.lower > hard.upper)
        flags["meta"] = action.meta
        for r in regs:
            b = r.bounds(i, state)
            flags[f"violation_{r.name}"] = not b.contains(proposed)
        posted = hard.clip(proposed) if np.isfinite(hard.lower) or np.isfinite(hard.upper) else proposed
        return float(posted), flags

    def _decide(self, i: int, t: int, last: list) -> tuple[Action, float, dict]:
        action = self._propose(i, t, last)
        posted, flags = self._enforce(i, action)
        return action, posted, flags

    def _decide_all(self, t: int, last: list, pool) -> list:
        if pool is None:
            return [self._decide(i, t, last) for i in range(len(self.agents))]
        out: list = [None] * len(self.agents)
        for i in self._sequential:
            out[i] = self._decide(i, t, last)
        futures = {i: pool.submit(self._decide, i, t, last) for i in self._parallel}
        for i, f in futures.items():
            out[i] = f.result()
        return out

    def run(self) -> SessionResult:
        self._parallel = [i for i, a in enumerate(self.agents) if getattr(a, "concurrent", False)]
        self._sequential = [i for i in range(len(self.agents)) if i not in self._parallel]
        workers = self.max_workers if self.max_workers is not None else len(self._parallel)
        if len(self._parallel) > 1 and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return self._run(pool)
        return self._run(None)

    def _run(self, pool) -> SessionResult:
        env, agents = self.env, self.agents
        n = len(agents)
        rngs, env_rng = self._rngs()
        initial = self.initial(env_rng) if callable(self.initial) else self.initial
        env.reset(env_rng, initial=None if initial is None else np.asarray(initial, dtype=float))
        market = env.market_info()
        for i, agent in enumerate(agents):
            agent.reset(AgentContext(index=i, n_agents=n, session=self.session, rng=rngs[i], market=market))
        last: list[Action | None] = [None] * n
        stable = 0
        stop_reason = "max_periods"
        t = 0
        completed = 0
        for t in range(self.periods):
            if self.before_period is not None:
                reason = self.before_period(t)
                if reason:
                    stop_reason = reason
                    break
            bounds = [env.bounds(i) for i in range(n)]
            pre_state = env.reg_state()
            try:
                decided = self._decide_all(t, last, pool)
            except StopSession as stop:
                stop_reason = stop.reason
                self.recorder.event(session=self.session, t=t, kind="stop", reason=stop.reason, detail=stop.detail)
                break
            actions = [d[0] for d in decided]
            flags = [d[2] for d in decided]
            posted = np.array([d[1] for d in decided])
            proposed = np.array([f["proposed"] for f in flags])
            result = env.step(posted, proposed)
            fines = np.zeros(n)
            for r in env.regulations:
                if r.enforcement == "penalty":
                    for i in range(n):
                        if flags[i][f"violation_{r.name}"]:
                            fines[i] += r.fine(proposed[i], r.bounds(i, pre_state), float(result.quantity[i]))
            profit = result.profit - fines
            env.charge(fines)
            for i, agent in enumerate(agents):
                agent.update(Feedback(t=t, index=i, proposed=float(proposed[i]), posted=posted,
                                      quantity=result.quantity, profit=profit,
                                      next_bounds=env.bounds(i), fine=float(fines[i])))
            last = [Action(float(proposed[i]), actions[i].meta) for i in range(n)]
            self.recorder.log_period(self.session, t, posted, proposed, result.quantity, profit,
                                     bounds, flags, fines, result.extra)
            completed = t + 1
            learners = [a for a in agents if getattr(a, "learning", False)]
            if self.stop_when_stable is not None and learners:
                stable = 0 if any(getattr(a, "changed", False) for a in learners) else stable + 1
                if stable >= self.stop_when_stable:
                    stop_reason = "policy_stable"
                    break
            if self.on_period is not None and self.on_period(t, {"posted": posted, "profit": profit}):
                stop_reason = "callback"
                break
        return SessionResult(self.session, completed, stop_reason, self.recorder, agents)
