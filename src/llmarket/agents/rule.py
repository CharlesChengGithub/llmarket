"""Rule-based pricing agents: fixed prices, static benchmarks and simple repeated-game strategies.

All prices are in model units. Rules read the market benchmarks from the agent context and
the price history from observations; they never see more than other agents do.
"""
from __future__ import annotations

import numpy as np

from ..core.types import Action


class _Rule:
    name = "rule"
    learning = False
    changed = False

    def reset(self, ctx) -> None:
        self.i = ctx.index
        self.market = ctx.market
        self.rng = ctx.rng

    def update(self, fb) -> None:
        pass

    def _rivals_last(self, obs) -> float | None:
        posted = obs.history["posted"]
        if not len(posted):
            return None
        return float(np.mean(np.delete(posted[-1], self.i)))

    def _act(self, price: float, obs, **meta) -> Action:
        return Action(float(price), {"agent": self.name, **meta})


class FixedPrice(_Rule):
    name = "fixed"

    def __init__(self, price: float):
        self.price = float(price)

    def act(self, obs):
        return self._act(self.price, obs)


class NashPrice(_Rule):
    name = "nash"

    def act(self, obs):
        return self._act(self.market["nash_prices"][self.i], obs)


class MonopolyPrice(_Rule):
    name = "monopoly"

    def act(self, obs):
        return self._act(self.market["monopoly_prices"][self.i], obs)


class RandomPrice(_Rule):
    name = "random"

    def __init__(self, low: float, high: float):
        self.low, self.high = float(low), float(high)

    def act(self, obs):
        return self._act(self.rng.uniform(self.low, self.high), obs)


class TitForTat(_Rule):
    """Start at `start` (default: monopoly price); then match the rivals' last average price."""

    name = "tit_for_tat"

    def __init__(self, start: float | None = None):
        self.start = start

    def act(self, obs):
        last = self._rivals_last(obs)
        if last is None:
            last = self.start if self.start is not None else self.market["monopoly_prices"][self.i]
        return self._act(last, obs)


class GrimTrigger(_Rule):
    """Price at `collusive` (default: monopoly) until any rival prices below `trigger`; then Nash forever."""

    name = "grim_trigger"

    def __init__(self, collusive: float | None = None, trigger: float | None = None):
        self.collusive = collusive
        self.trigger = trigger

    def reset(self, ctx) -> None:
        super().reset(ctx)
        self.punishing = False

    def act(self, obs):
        high = self.collusive if self.collusive is not None else self.market["monopoly_prices"][self.i]
        trigger = self.trigger if self.trigger is not None else high - 1e-9
        posted = obs.history["posted"]
        if len(posted) and np.any(np.delete(posted[-1], self.i) < trigger):
            self.punishing = True
        price = self.market["nash_prices"][self.i] if self.punishing else high
        return self._act(price, obs, punishing=self.punishing)


class Undercut(_Rule):
    """Price `step` below the lowest rival price, but never below `minimum` (default: own cost)."""

    name = "undercut"

    def __init__(self, step: float = 0.01, minimum: float | None = None, start: float | None = None):
        self.step = float(step)
        self.minimum = minimum
        self.start = start

    def act(self, obs):
        posted = obs.history["posted"]
        floor = self.minimum if self.minimum is not None else self.market["cost"][self.i]
        if not len(posted):
            p = self.start if self.start is not None else self.market["monopoly_prices"][self.i]
        else:
            p = float(np.min(np.delete(posted[-1], self.i))) - self.step
        return self._act(max(p, floor, obs.bounds.lower), obs)


RULES = {cls.name: cls for cls in (FixedPrice, NashPrice, MonopolyPrice, RandomPrice, TitForTat, GrimTrigger, Undercut)}
