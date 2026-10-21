"""Environment protocol."""
from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

from .types import Bounds, Observation, StepResult


class Environment(Protocol):
    """A market mechanism: maps the agents' posted actions to outcomes.

    The environment owns the history and the regulation state, so that bounds and
    observations are computed from one source of truth.
    """

    n_agents: int

    def reset(self, rng: Any, initial: np.ndarray | None = None) -> None: ...

    def bounds(self, i: int) -> Bounds: ...

    def disclosures(self, i: int) -> tuple[str, ...]: ...

    def observe(self, i: int) -> Observation: ...

    def step(self, posted: Sequence[float], proposed: Sequence[float]) -> StepResult: ...

    def market_info(self) -> dict: ...
