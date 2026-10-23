"""Regulation base class: a constraint with an enforcement mode and a disclosure mode.

Dhanda (2026) finds that an announced price cap coordinates LLM pricing agents while the
same cap imposed silently does not. Regulations therefore separate what is enforced from
what agents are told.
"""
from __future__ import annotations

from typing import Any, Literal

from .types import Bounds

ENFORCEMENT = ("clip", "reject", "penalty", "monitor")
DISCLOSURE = ("announced", "hidden")


class Regulation:
    """Subclasses implement :meth:`bounds` and :meth:`describe`.

    enforcement:
      clip     posted action = proposed action clipped to the legal interval;
      reject   agents that support retries re-propose (up to max_rejections), then clip;
      penalty  the proposed action stands; a fine is charged (see :meth:`fine`);
      monitor  the proposed action stands; violations are only recorded.
    disclosure: announced regulations add :meth:`describe` text to LLM observations.
    """

    name = "regulation"

    def __init__(self, enforcement: Literal["clip", "reject", "penalty", "monitor"] = "clip",
                 disclosure: Literal["announced", "hidden"] = "announced",
                 max_rejections: int = 3, penalty: float = 1.0):
        if enforcement not in ENFORCEMENT:
            raise ValueError(f"enforcement must be one of {ENFORCEMENT}")
        if disclosure not in DISCLOSURE:
            raise ValueError(f"disclosure must be one of {DISCLOSURE}")
        self.enforcement = enforcement
        self.disclosure = disclosure
        self.max_rejections = int(max_rejections)
        self.penalty = float(penalty)

    def bounds(self, i: int, state: Any) -> Bounds:
        raise NotImplementedError

    def describe(self, i: int, state: Any) -> str | None:
        return None

    def fine(self, proposed: float, bounds: Bounds, quantity: float) -> float:
        """Penalty mode: penalty x distance outside the interval x units sold."""
        gap = max(bounds.lower - proposed, proposed - bounds.upper, 0.0)
        return self.penalty * gap * quantity

    def config(self) -> dict:
        return dict(type=self.name, enforcement=self.enforcement, disclosure=self.disclosure)
