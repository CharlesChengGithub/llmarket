"""Data passed between environments, regulations, agents and the simulation loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Bounds:
    """Legal interval for one agent's action in one period."""

    lower: float = -np.inf
    upper: float = np.inf

    def intersect(self, other: "Bounds") -> "Bounds":
        return Bounds(max(self.lower, other.lower), min(self.upper, other.upper))

    def contains(self, x: float, tol: float = 1e-12) -> bool:
        return self.lower - tol <= x <= self.upper + tol

    def clip(self, x: float) -> float:
        if self.lower > self.upper:  # conflicting rules: the floor prevails
            return float(self.lower)
        return float(min(max(x, self.lower), self.upper))


@dataclass
class AgentContext:
    """What an agent learns once, at the start of a session."""

    index: int
    n_agents: int
    session: int
    rng: Any
    market: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """What an agent sees before acting in period t (information up to t - 1).

    history holds arrays with one row per past period: 'posted' and 'proposed' (n columns),
    'quantity' and 'profit' (n columns). Agents may only use what their deployment allows;
    the environment fills the full record and LLM prompt builders filter it.
    """

    t: int
    index: int
    history: Mapping[str, np.ndarray]
    bounds: Bounds
    disclosures: tuple[str, ...] = ()
    market: Mapping[str, Any] = field(default_factory=dict)
    retry_message: str | None = None


@dataclass
class Action:
    """A proposed action (a price in Bertrand markets) plus free-form metadata for the record."""

    value: float
    meta: dict = field(default_factory=dict)


@dataclass
class Feedback:
    """Realized outcome of period t, sent to every agent after enforcement."""

    t: int
    index: int
    proposed: float
    posted: np.ndarray
    quantity: np.ndarray
    profit: np.ndarray
    next_bounds: Bounds
    fine: float = 0.0

    @property
    def own_profit(self) -> float:
        return float(self.profit[self.index])


class StopSession(Exception):
    """Raised by an agent to end the session before the current period is recorded
    (e.g. repeated unparsable LLM answers, or the hard budget limit)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class StepResult:
    quantity: np.ndarray
    profit: np.ndarray
    extra: dict = field(default_factory=dict)
