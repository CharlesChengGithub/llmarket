"""Agent protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Action, AgentContext, Feedback, Observation


@runtime_checkable
class Agent(Protocol):
    """Anything that proposes an action each period and may learn from the outcome.

    Optional attributes the simulation honours:
    - ``interval`` (int, default 1): act only every ``interval`` periods, otherwise repeat;
    - ``can_retry`` (bool): the agent re-proposes when a regulation rejects its action;
    - ``learning`` (bool) and ``changed`` (bool): used by convergence-based stopping rules.
    """

    def reset(self, ctx: AgentContext) -> None: ...

    def act(self, obs: Observation) -> Action: ...

    def update(self, fb: Feedback) -> None: ...
