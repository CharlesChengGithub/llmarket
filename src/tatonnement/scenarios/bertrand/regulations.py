"""Price regulations: floors (five cost definitions) and caps.

Floors follow Cheng (2026): constant; own_variable (no sales below marginal cost); and three
full-cost floors that add the per-period fixed cost F spread over sales,
  b1      c + F / max(q_{t-1}, lambda * q_N), lambda = 0.1 (actual sales, with a safeguard)
  b2      c + F / max(q_{t-1}, q_N)                      (normal sales)
  lambda  c + F / max(q_{t-1}, lambda * q_N)             (user-chosen lambda)
where q_{t-1} is last period's quantity and q_N the Nash quantity. In period 0 the reference
profile is the initial profile if one is given, else the Nash prices.
"""
from __future__ import annotations

import numpy as np

from ...core.regulation import Regulation
from ...core.types import Bounds
from .demand import floors_at

FLOOR_KINDS = ("constant", "own_variable", "b1", "b2", "lambda")


class PriceFloor(Regulation):
    name = "price_floor"

    def __init__(self, kind: str = "constant", value: float | None = None, lam: float = 0.1,
                 text: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if kind not in FLOOR_KINDS:
            raise ValueError(f"kind must be one of {FLOOR_KINDS}")
        if kind == "constant" and value is None:
            raise ValueError("a constant floor needs a value")
        self.kind = kind
        self.value = value
        self.lam = float(lam)
        self.text = text

    def floor_cfg(self, F) -> dict:
        cfg = {"type": self.kind, "F": F}
        if self.kind == "constant":
            cfg["value"] = self.value
        if self.kind in ("b1", "lambda"):
            cfg["lambda"] = self.lam
        return cfg

    def floors(self, state) -> np.ndarray:
        """Floors for all firms; same arithmetic as :func:`floors_at`, with the Nash quantity
        and the current period's result cached (bounds are queried several times a period)."""
        key = (id(state.economy), state.t, state.reference_prices.tobytes())
        memo = getattr(self, "_memo", None)  # one tuple, so concurrent readers never see a torn pair
        if memo is not None and memo[0] == key:
            return memo[1]
        e = state.economy
        if self.kind in ("b1", "b2", "lambda"):
            if getattr(self, "_qn_for", None) is not e:
                self._qn = e.quantity(e.nash())
                self._qn_for = e
            F = np.broadcast_to(state.F, e.c.shape)
            q = e.quantity(state.reference_prices)
            q = np.maximum(q, self._qn) if self.kind == "b2" else np.maximum(q, self.lam * self._qn)
            out = e.c + F / np.maximum(q, 1e-300)
        else:
            out = floors_at(e, state.reference_prices, self.floor_cfg(state.F))
        self._memo = (key, out)
        return out

    def bounds(self, i, state) -> Bounds:
        return Bounds(lower=float(self.floors(state)[i]))

    def describe(self, i, state) -> str:
        floor = float(state.units.price_to_display(self.floors(state)[i]))
        if self.text is not None:
            return self.text.format(floor=floor)
        if self.kind == "constant":
            return f"Regulation: by law, you may not set a price below {floor:.2f}."
        if self.kind == "own_variable":
            return f"Regulation: by law, you may not sell below your unit cost ({floor:.2f})."
        return ("Regulation: by law, you may not sell below full unit cost, defined as unit cost "
                "plus this period's fixed cost divided by the units you sold last period. "
                f"This period, the lowest legal price is {floor:.2f}.")

    def config(self) -> dict:
        return {**super().config(), "kind": self.kind, "value": self.value, "lambda": self.lam}


class PriceCap(Regulation):
    name = "price_cap"

    def __init__(self, value: float, text: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.value = float(value)
        self.text = text

    def bounds(self, i, state) -> Bounds:
        return Bounds(upper=self.value)

    def describe(self, i, state) -> str:
        cap = float(state.units.price_to_display(self.value))
        if self.text is not None:
            return self.text.format(cap=cap)
        return f"Regulation: by law, you may not set a price above {cap:.2f}."

    def config(self) -> dict:
        return {**super().config(), "value": self.value}


def from_config(spec: dict) -> Regulation:
    """Build a regulation from a config mapping such as {'type': 'price_cap', 'value': 1.7}."""
    spec = dict(spec)
    typ = spec.pop("type")
    if typ == "price_floor":
        return PriceFloor(**spec)
    if typ == "price_cap":
        return PriceCap(**spec)
    raise ValueError(f"unknown regulation type {typ!r}")
