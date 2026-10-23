"""Repeated Bertrand market with logit demand and pluggable price regulations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.types import Bounds, Observation, StepResult
from .demand import Economy, Units


@dataclass
class MarketState:
    """What regulations see: the economy and last period's outcome."""

    t: int
    economy: Economy
    F: np.ndarray
    units: Units
    reference_prices: np.ndarray  # last period's prices (initial profile or Nash in period 0)
    last_quantity: np.ndarray | None


class BertrandEnv:
    """Firms post prices simultaneously; demand is logit; profit is (p - c) q - F.

    Prices, quantities and profits are kept in model units. ``units`` only affects what LLM
    agents are shown. history_limit bounds the stored history (None keeps everything).
    """

    def __init__(self, economy: Economy, regulations=(), F=0.0, units: Units | None = None,
                 history_limit: int | None = None):
        self.economy = economy
        self.n_agents = economy.n
        self.regulations = list(regulations)
        self.F = np.broadcast_to(np.asarray(F, dtype=float), (economy.n,)).copy()
        self.units = units or Units()
        self.history_limit = history_limit
        self.nash = economy.nash()
        self.monopoly = economy.monopoly()
        self._market = dict(
            n=economy.n, cost=economy.c.tolist(), a=economy.a.tolist(), mu=economy.mu, a0=economy.a0,
            fixed_cost=self.F.tolist(), units=dict(alpha=self.units.alpha, beta=self.units.beta),
            nash_prices=self.nash.tolist(), monopoly_prices=self.monopoly.tolist(),
            nash_profit=economy.profit(self.nash, self.F).tolist(),
            monopoly_profit=economy.profit(self.monopoly, self.F).tolist(),
            regulations=[r.config() for r in self.regulations],
        )
        self.reset(None)

    # -- lifecycle ------------------------------------------------------------------------
    def reset(self, rng, initial: np.ndarray | None = None) -> None:
        self.t = 0
        self.initial = None if initial is None else np.asarray(initial, dtype=float)
        self._hist = {k: [] for k in ("posted", "proposed", "quantity", "profit")}

    def market_info(self) -> dict:
        info = dict(self._market)
        info["initial_prices"] = None if self.initial is None else self.initial.tolist()
        return info

    # -- regulation state -----------------------------------------------------------------
    def reg_state(self) -> MarketState:
        if self._hist["posted"]:
            ref = self._hist["posted"][-1]
            q = self._hist["quantity"][-1]
        else:
            ref = self.initial if self.initial is not None else self.nash
            q = None
        return MarketState(self.t, self.economy, self.F, self.units, ref, q)

    def bounds(self, i: int) -> Bounds:
        """Intersection of the clip and reject regulations (what is actually enforced)."""
        state = self.reg_state()
        b = Bounds()
        for r in self.regulations:
            if r.enforcement in ("clip", "reject"):
                b = b.intersect(r.bounds(i, state))
        return b

    def disclosures(self, i: int) -> tuple[str, ...]:
        state = self.reg_state()
        return tuple(r.describe(i, state) for r in self.regulations if r.disclosure == "announced")

    def rejection_message(self, i: int, proposed: float, bounds: Bounds) -> str:
        p = float(self.units.price_to_display(proposed))
        lo, hi = (float(self.units.price_to_display(x)) for x in (bounds.lower, bounds.upper))
        rng = f"at least {lo:.2f}" if np.isfinite(lo) else ""
        if np.isfinite(hi):
            rng = f"{rng} and " if rng else ""
            rng += f"at most {hi:.2f}"
        return f"Your price {p:.2f} is not allowed. The price must be {rng}. Please choose again."

    # -- observation and step --------------------------------------------------------------
    def history(self) -> dict:
        n = self.n_agents
        return {k: (np.array(v) if v else np.empty((0, n))) for k, v in self._hist.items()}

    def observe(self, i: int) -> Observation:
        return Observation(t=self.t, index=i, history=self.history(), bounds=self.bounds(i),
                           disclosures=self.disclosures(i), market=self._market)

    def step(self, posted, proposed=None) -> StepResult:
        posted = np.asarray(posted, dtype=float)
        q = self.economy.quantity(posted[None, :])[0]
        profit = self.economy.profit(posted[None, :], self.F)[0]
        cs = float(self.economy.cs(posted[None, :])[0])
        for k, v in (("posted", posted), ("proposed", posted if proposed is None else np.asarray(proposed, float)),
                     ("quantity", q), ("profit", profit)):
            self._hist[k].append(v)
            if self.history_limit is not None and len(self._hist[k]) > self.history_limit:
                del self._hist[k][0]
        self.t += 1
        return StepResult(quantity=q, profit=profit, extra={"cs": cs})

    def charge(self, fines) -> None:
        """Subtract fines from the stored profits of the period just played."""
        fines = np.asarray(fines, dtype=float)
        if np.any(fines) and self._hist["profit"]:
            self._hist["profit"][-1] = self._hist["profit"][-1] - fines

    # -- benchmarks -------------------------------------------------------------------------
    def profit_gain(self, profits) -> float:
        """Delta = (mean profit - Nash profit) / (monopoly profit - Nash profit)."""
        pin = float(np.mean(self._market["nash_profit"]))
        pim = float(np.mean(self._market["monopoly_profit"]))
        return float((np.mean(profits) - pin) / (pim - pin))
